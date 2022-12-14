import torch
from torch import nn
import torch.nn.functional as F


def weight_init_kaiming_uniform(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1.0)
        module.bias.data.fill_(0.0)



class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))



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



class Conv(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride=1, padding=0, dilation=1, act="leaky_relu", depthwise=False):
        super().__init__()
        if act == "relu":
            act_func = nn.ReLU(inplace=True)
        elif act == "leaky_relu":
            act_func = nn.LeakyReLU(0.1, inplace=True)
        elif act == "mish":
            act_func = Mish()
        elif act == "identity":
            act_func = nn.Identity()

        if depthwise:
            self.conv = nn.Sequential(
                ### Depth-wise ###
                nn.Conv2d(c1, c1, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                act_func,
                ### Point-wise ###
                nn.Conv2d(c1, c2, kernel_size=1, stride=stride, padding=0, dilation=dilation, groups=1, bias=False),
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