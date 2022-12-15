import torch
from torch import nn
import torch.nn.functional as F


def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def weight_init_kaiming_uniform(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1.0)
        module.bias.data.fill_(0.0)



class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6



class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = Hsigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)



class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

        

class Conv(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride=1, padding=0, dilation=1, act="leaky_relu", depthwise=False):
        super().__init__()
        
        if act == "identity":
            act_func = nn.Identity()
        elif act == "relu":
            act_func = nn.ReLU(inplace=True)
        elif act == "relu6":
            act_func = nn.ReLU6(inplace=True)
        elif act == "leaky_relu":
            act_func = nn.LeakyReLU(0.1, inplace=True)
        elif act == "mish":
            act_func = Mish()
        elif act == "hsigmoid":
            act_func = Hsigmoid()
        elif act == "hswish":
            act_func = Hswish()

        if depthwise:
            self.conv = nn.Sequential(
                ### Depth-wise ###
                nn.Conv2d(c1, c1, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                act_func,
                ### Point-wise ###
                nn.Conv2d(c1, c2, kernel_size=1, bias=False),
                nn.BatchNorm2d(c2),
                act_func,
            )
        else:
            self.conv = nn.Sequential(
                ### General Conv ###
                nn.Conv2d(c1, c2, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=False),
                nn.BatchNorm2d(c2),
                act_func,
            )

    def forward(self, x):
        return self.conv(x)



class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(make_divisible(channel // reduction, 8), channel),
                Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



class InvertedResidualV2(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super().__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                Hswish() if use_hs else nn.ReLU(inplace=True),
                SELayer(inp) if use_se else nn.Identity(),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Hswish() if use_hs else nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                Hswish() if use_hs else nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)



class InvertedResidualV1(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = Conv(inp, oup, kernel_size=3, stride=stride, padding=1, act="relu6", depthwise=True)
        else:
            self.conv = nn.Sequential(
                Conv(inp, round(inp * expand_ratio), kernel_size=1, act="relu6"),
                Conv(round(inp * expand_ratio), oup, kernel_size=3, stride=stride, padding=1, act="relu6", depthwise=True)
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)



class CSPStage(nn.Module):
    def __init__(self, c1, num_blocks=1):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.conv1 = Conv(c1, c_, kernel_size=1, act="mish")
        self.conv2 = Conv(c1, c_, kernel_size=1, act="mish")
        self.res_blocks = nn.Sequential(*[ResBlock(in_channels=c_, act="mish") for _ in range(num_blocks)])
        self.conv3 = Conv(c_*2, c1, kernel_size=1, act="mish")

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.res_blocks(self.conv2(x))
        return self.conv3(torch.cat([y1, y2], dim=1))



class ResBlock(nn.Module):
    def __init__(self, in_channels, act="leaky_relu"):
        super().__init__()
        assert in_channels % 2 == 0
        self.conv1 = Conv(in_channels, in_channels//2, kernel_size=1, padding=0, act=act)
        self.conv2 = Conv(in_channels//2, in_channels, kernel_size=3, padding=1, act=act)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out



class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out