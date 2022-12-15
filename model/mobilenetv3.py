import torch
from torch import nn
from element import Hswish, Conv, InvertedResidualV2, make_divisible, weight_init_kaiming_uniform


class MobilenetV3(nn.Module):
    def __init__(self, mode, num_classes, width_multiple=1.0):
        super().__init__()
        assert mode in ["large", "small"]

        if mode == "large":
            width_cfg = []
            for c in (16, 24, 40, 80, 112, 160):
                width_cfg.append(make_divisible(c*width_multiple, 8))

            self.conv1 = Conv(3, width_cfg[0], kernel_size=3, padding=1, stride=2, act="hswish")
            self.convs = nn.Sequential(
                InvertedResidualV2(width_cfg[0], make_divisible(width_cfg[0]*1, 8), width_cfg[0], 3, 1, 0, 0),

                InvertedResidualV2(width_cfg[0], make_divisible(width_cfg[0]*4, 8), width_cfg[1], 3, 2, 0, 0),
                InvertedResidualV2(width_cfg[1], make_divisible(width_cfg[1]*3, 8), width_cfg[1], 3, 1, 0, 0),

                InvertedResidualV2(width_cfg[1], make_divisible(width_cfg[1]*3, 8), width_cfg[2], 5, 2, 1, 0),
                InvertedResidualV2(width_cfg[2], make_divisible(width_cfg[2]*3, 8), width_cfg[2], 5, 1, 1, 0),
                InvertedResidualV2(width_cfg[2], make_divisible(width_cfg[2]*3, 8), width_cfg[2], 5, 1, 1, 0),

                InvertedResidualV2(width_cfg[2], make_divisible(width_cfg[2]*6, 8), width_cfg[3], 3, 2, 0, 1),
                InvertedResidualV2(width_cfg[3], make_divisible(width_cfg[3]*2.5, 8), width_cfg[3], 3, 1, 0, 1),
                InvertedResidualV2(width_cfg[3], make_divisible(width_cfg[3]*2.3, 8), width_cfg[3], 3, 1, 0, 1),
                InvertedResidualV2(width_cfg[3], make_divisible(width_cfg[3]*2.3, 8), width_cfg[3], 3, 1, 0, 1),

                InvertedResidualV2(width_cfg[3], make_divisible(width_cfg[3]*6, 8), width_cfg[4], 3, 1, 1, 1),
                InvertedResidualV2(width_cfg[4], make_divisible(width_cfg[4]*6, 8), width_cfg[4], 3, 1, 1, 1),

                InvertedResidualV2(width_cfg[4], make_divisible(width_cfg[4]*6, 8), width_cfg[5], 5, 2, 1, 1),
                InvertedResidualV2(width_cfg[5], make_divisible(width_cfg[5]*6, 8), width_cfg[5], 5, 1, 1, 1),
                InvertedResidualV2(width_cfg[5], make_divisible(width_cfg[5]*6, 8), width_cfg[5], 5, 1, 1, 1),
            )
            hidden_dim1 = make_divisible(width_cfg[5]*6, 8)
            hidden_dim2 = make_divisible(1280*width_multiple, 8) if width_multiple > 1.0 else 1280
        else:
            width_cfg = []
            for c in (16, 24, 40, 48, 96):
                width_cfg.append(make_divisible(c*width_multiple, 8))

            self.conv1 = Conv(3, width_cfg[0], kernel_size=3, padding=1, stride=2, act="hswish")
            self.convs = nn.Sequential(
                InvertedResidualV2(width_cfg[0], make_divisible(width_cfg[0]*1, 8), width_cfg[0], 3, 2, 1, 0),

                InvertedResidualV2(width_cfg[0], make_divisible(width_cfg[0]*4.5, 8), width_cfg[1], 3, 2, 0, 0),
                InvertedResidualV2(width_cfg[1], make_divisible(width_cfg[1]*3.67, 8), width_cfg[1], 3, 1, 0, 0),

                InvertedResidualV2(width_cfg[1], make_divisible(width_cfg[1]*4, 8), width_cfg[2], 5, 2, 1, 1),
                InvertedResidualV2(width_cfg[2], make_divisible(width_cfg[2]*6, 8), width_cfg[2], 5, 1, 1, 1),
                InvertedResidualV2(width_cfg[2], make_divisible(width_cfg[2]*6, 8), width_cfg[2], 5, 1, 1, 1),

                InvertedResidualV2(width_cfg[2], make_divisible(width_cfg[2]*3, 8), width_cfg[3], 5, 1, 1, 1),
                InvertedResidualV2(width_cfg[3], make_divisible(width_cfg[3]*3, 8), width_cfg[3], 5, 1, 1, 1),

                InvertedResidualV2(width_cfg[3], make_divisible(width_cfg[3]*6, 8), width_cfg[4], 5, 2, 1, 1),
                InvertedResidualV2(width_cfg[4], make_divisible(width_cfg[4]*6, 8), width_cfg[4], 5, 1, 1, 1),
                InvertedResidualV2(width_cfg[4], make_divisible(width_cfg[4]*6, 8), width_cfg[4], 5, 1, 1, 1),
            )
            hidden_dim1 = make_divisible(width_cfg[4]*6, 8)
            hidden_dim2 = make_divisible(1024*width_multiple, 8) if width_multiple > 1.0 else 1024
        
        self.conv2 = Conv(width_cfg[-1], hidden_dim1, kernel_size=1, act="hswish")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            Hswish(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2, num_classes),
        )
        self.apply(weight_init_kaiming_uniform)


    def forward(self, x):
        out = self.conv1(x)
        out = self.convs(out)
        out = self.conv2(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out



def build_mobilenetv3(num_classes=1000, mode="large", width_multiple=1.0):
    model = MobilenetV3(mode=mode, num_classes=num_classes, width_multiple=width_multiple)
    return model



if __name__ == "__main__":
    input_size = 224
    num_classes = 1000
    device = torch.device('cpu')
    model = build_mobilenetv3(num_classes=1000, mode="large", width_multiple=1.0)
    
    x = torch.randn(2, 3, input_size, input_size).to(device)
    y = model(x)
    print(y.shape)

    