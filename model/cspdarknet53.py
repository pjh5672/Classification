import torch
from torch import nn
from element import Conv, CSPStage, weight_init_kaiming_uniform



class CSPDarknet53(nn.Module):
    def __init__(self, num_classes, width_multiple=1.0, depth_multiple=1.0):
        super().__init__()
        width_cfg = [int(w*width_multiple) for w in (64, 128, 256, 512, 1024)]
        depth_cfg = [int(d*depth_multiple) for d in (1, 3, 9, 9, 6)]

        self.layer1 = nn.Sequential(
            Conv(3, width_cfg[0], kernel_size=6, padding=2, stride=2, act="mish"),
            CSPStage(c1=width_cfg[0], num_blocks=1)
        )
        self.layer2 = nn.Sequential(
            Conv(width_cfg[0], width_cfg[1], kernel_size=3, padding=1, stride=2, act="mish"),
            CSPStage(c1=width_cfg[1], num_blocks=depth_cfg[1])
        )
        self.layer3 = nn.Sequential(
            Conv(width_cfg[1], width_cfg[2], kernel_size=3, padding=1, stride=2, act="mish"),
            CSPStage(c1=width_cfg[2], num_blocks=depth_cfg[2])
        )
        self.layer4 = nn.Sequential(
            Conv(width_cfg[2], width_cfg[3], kernel_size=3, padding=1, stride=2, act="mish"),
            CSPStage(c1=width_cfg[3], num_blocks=depth_cfg[3])
        )
        self.layer5 = nn.Sequential(
            Conv(width_cfg[3], width_cfg[4], kernel_size=3, padding=1, stride=2, act="mish"),
            CSPStage(c1=width_cfg[4], num_blocks=depth_cfg[4])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width_cfg[4], num_classes)
        self.apply(weight_init_kaiming_uniform)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def build_CSPdarknet53(num_classes=1000, width_multiple=1.0, depth_multiple=1.0):
    model = CSPDarknet53(num_classes, width_multiple=width_multiple, depth_multiple=depth_multiple)
    return model


if __name__ == "__main__":
    input_size = 224
    num_classes = 1000
    device = torch.device('cpu')
    model = build_CSPdarknet53(num_classes=num_classes, width_multiple=1.0, depth_multiple=1.0)

    x = torch.randn(2, 3, input_size, input_size).to(device)
    y = model(x)
    print(y.shape)