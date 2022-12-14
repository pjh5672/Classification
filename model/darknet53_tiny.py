import torch
from torch import nn
from element import Conv, weight_init_kaiming_uniform



class DarkNet53_tiny(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = Conv(3, 16, kernel_size=3, padding=1, stride=1, act="leaky_relu")
        self.conv2 = Conv(16, 32, kernel_size=3, padding=1, stride=1, act="leaky_relu")
        self.conv3 = Conv(32, 64, kernel_size=3, padding=1, stride=1, act="leaky_relu")
        self.conv4 = Conv(64, 128, kernel_size=3, padding=1, stride=1, act="leaky_relu")
        self.conv5 = Conv(128, 256, kernel_size=3, padding=1, stride=1, act="leaky_relu")
        self.conv6 = Conv(256, 512, kernel_size=3, padding=1, stride=1, act="leaky_relu")
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.zeropad = nn.ZeroPad2d((0, 1, 0, 1))
        self.fc = nn.Linear(512, num_classes)
        self.apply(weight_init_kaiming_uniform)


    def forward(self, x):
        out = self.maxpool(self.conv1(x))
        out = self.maxpool(self.conv2(out))
        out = self.maxpool(self.conv3(out))
        out = self.maxpool(self.conv4(out))
        out = self.maxpool(self.conv5(out))
        out = self.zeropad(self.conv6(out))
        out = self.avgpool(self.maxpool(out))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out



def build_darknet53_tiny(num_classes=1000):
    model = DarkNet53_tiny(num_classes=num_classes)
    return model



if __name__ == "__main__":
    input_size = 416
    num_classes = 1000
    device = torch.device('cpu')
    model = build_darknet53_tiny(num_classes=num_classes)

    x = torch.randn(2, 3, input_size, input_size).to(device)
    y = model(x)
    print(y.shape)