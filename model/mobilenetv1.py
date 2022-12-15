import torch
from torch import nn
from element import Conv, weight_init_kaiming_uniform


class MobilenetV1(nn.Module):
    def __init__(self, num_classes, width_multiple=1.0):
        super().__init__()
        width_cfg = [int(w*width_multiple) for w in (32, 64, 128, 256, 512, 1024)]
        
        self.conv1 = Conv(3, width_cfg[0], kernel_size=3, padding=1, stride=2, act="leaky_relu")
        self.conv2 = nn.Sequential(
            Conv(width_cfg[0], width_cfg[1], kernel_size=3, padding=1, stride=1, act="leaky_relu", depthwise=True),
            Conv(width_cfg[1], width_cfg[2], kernel_size=3, padding=1, stride=2, act="leaky_relu", depthwise=True),
            Conv(width_cfg[2], width_cfg[2], kernel_size=3, padding=1, stride=1, act="leaky_relu", depthwise=True),
            Conv(width_cfg[2], width_cfg[3], kernel_size=3, padding=1, stride=2, act="leaky_relu", depthwise=True),
            Conv(width_cfg[3], width_cfg[3], kernel_size=3, padding=1, stride=1, act="leaky_relu", depthwise=True),
            Conv(width_cfg[3], width_cfg[4], kernel_size=3, padding=1, stride=2, act="leaky_relu", depthwise=True),

            Conv(width_cfg[4], width_cfg[4], kernel_size=3, padding=1, stride=1, act="leaky_relu", depthwise=True),
            Conv(width_cfg[4], width_cfg[4], kernel_size=3, padding=1, stride=1, act="leaky_relu", depthwise=True),
            Conv(width_cfg[4], width_cfg[4], kernel_size=3, padding=1, stride=1, act="leaky_relu", depthwise=True),
            Conv(width_cfg[4], width_cfg[4], kernel_size=3, padding=1, stride=1, act="leaky_relu", depthwise=True),
            Conv(width_cfg[4], width_cfg[4], kernel_size=3, padding=1, stride=1, act="leaky_relu", depthwise=True),

            Conv(width_cfg[4], width_cfg[5], kernel_size=3, padding=1, stride=2, act="leaky_relu", depthwise=True),
            Conv(width_cfg[5], width_cfg[5], kernel_size=3, padding=1, stride=1, act="leaky_relu", depthwise=True),
        )
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(width_cfg[5], num_classes)
        self.apply(weight_init_kaiming_uniform)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out



if __name__ == "__main__":
    from thop import profile
    
    input_size = 224
    num_classes = 1000
    device = torch.device('cpu')
    model = MobilenetV1(num_classes=num_classes, width_multiple=1.0)

    x = torch.randn(2, 3, input_size, input_size).to(device)
    y = model(x)
    print(y.shape)