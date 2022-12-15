import torch
from torch import nn
from element import Conv, InvertedResidualV1, make_divisible, weight_init_kaiming_uniform


class MobilenetV2(nn.Module):
    def __init__(self, num_classes, width_multiple=1.0):
        super().__init__()

        width_cfg = []
        for c in (32, 16, 24, 32, 64, 96, 160, 320, 1280):
            width_cfg.append(make_divisible(c*width_multiple, 4 if width_multiple == 0.1 else 8))

        self.conv1 = Conv(3, width_cfg[0], kernel_size=3, padding=1, stride=2, act="relu6")
        self.convs = nn.Sequential(
            InvertedResidualV1(width_cfg[0], width_cfg[1], 1, 1),

            InvertedResidualV1(width_cfg[1], width_cfg[2], 2, 6),
            InvertedResidualV1(width_cfg[2], width_cfg[2], 1, 6),
            
            InvertedResidualV1(width_cfg[2], width_cfg[3], 2, 6),
            InvertedResidualV1(width_cfg[3], width_cfg[3], 1, 6),
            InvertedResidualV1(width_cfg[3], width_cfg[3], 1, 6),

            InvertedResidualV1(width_cfg[3], width_cfg[4], 2, 6),
            InvertedResidualV1(width_cfg[4], width_cfg[4], 1, 6),
            InvertedResidualV1(width_cfg[4], width_cfg[4], 1, 6),
            InvertedResidualV1(width_cfg[4], width_cfg[4], 1, 6),

            InvertedResidualV1(width_cfg[4], width_cfg[5], 1, 6),
            InvertedResidualV1(width_cfg[5], width_cfg[5], 1, 6),
            InvertedResidualV1(width_cfg[5], width_cfg[5], 1, 6),

            InvertedResidualV1(width_cfg[5], width_cfg[6], 2, 6),
            InvertedResidualV1(width_cfg[6], width_cfg[6], 1, 6),
            InvertedResidualV1(width_cfg[6], width_cfg[6], 1, 6),

            InvertedResidualV1(width_cfg[6], width_cfg[7], 1, 6),
        )
        self.conv2 = Conv(width_cfg[7], width_cfg[8], kernel_size=1, act="relu6")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width_cfg[8], num_classes)
        self.apply(weight_init_kaiming_uniform)


    def forward(self, x):
        out = self.conv1(x)
        out = self.convs(out)
        out = self.conv2(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out



def build_mobilenetv2(num_classes=1000, width_multiple=1.0):
    model = MobilenetV2(num_classes=num_classes, width_multiple=width_multiple)
    return model



if __name__ == "__main__":
    input_size = 224
    num_classes = 1000
    device = torch.device('cpu')
    model = build_mobilenetv2(num_classes=num_classes, width_multiple=1.0)

    x = torch.randn(2, 3, input_size, input_size).to(device)
    y = model(x)
    print(y.shape)
