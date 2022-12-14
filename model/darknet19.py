import torch
from torch import nn
from element import Conv, weight_init_kaiming_uniform



class Darknet19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = Conv(3, 32, kernel_size=3, padding=1, act="leaky_relu")
        self.conv2 = Conv(32, 64, kernel_size=3, padding=1, act="leaky_relu")
        self.conv3 = nn.Sequential(
            Conv(64, 128, kernel_size=3, padding=1, act="leaky_relu"),
            Conv(128, 64, kernel_size=1, act="leaky_relu"),
            Conv(64, 128, kernel_size=3, padding=1, act="leaky_relu")
        )
        self.conv4 = nn.Sequential(
            Conv(128, 256, kernel_size=3, padding=1, act="leaky_relu"),
            Conv(256, 128, kernel_size=1, act="leaky_relu"),
            Conv(128, 256, kernel_size=3, padding=1, act="leaky_relu")
        )
        self.conv5 = nn.Sequential(
            Conv(256, 512, kernel_size=3, padding=1, act="leaky_relu"),
            Conv(512, 256, kernel_size=1, act="leaky_relu"),
            Conv(256, 512, kernel_size=3, padding=1, act="leaky_relu"),
            Conv(512, 256, kernel_size=1, act="leaky_relu"),
            Conv(256, 512, kernel_size=3, padding=1, act="leaky_relu")
        )
        self.conv6 = nn.Sequential(
            Conv(512, 1024, kernel_size=3, padding=1, act="leaky_relu"),
            Conv(1024, 512, kernel_size=1, act="leaky_relu"),
            Conv(512, 1024, kernel_size=3, padding=1, act="leaky_relu"),
            Conv(1024, 512, kernel_size=1, act="leaky_relu"),
            Conv(512, 1024, kernel_size=3, padding=1, act="leaky_relu"),
        )
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.apply(weight_init_kaiming_uniform)


    def forward(self, x):
        out = self.pool(self.conv1(x))
        out = self.pool(self.conv2(out))
        out = self.pool(self.conv3(out))
        out = self.pool(self.conv4(out))
        out = self.pool(self.conv5(out))
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out



def build_darknet19(num_classes=1000):
    model = Darknet19(num_classes=num_classes)
    return model



if __name__ == "__main__":
    input_size = 224
    num_classes = 1000
    device = torch.device('cpu')
    model = build_darknet19(num_classes=num_classes)

    x = torch.randn(2, 3, input_size, input_size).to(device)
    y = model(x)
    print(y.shape)