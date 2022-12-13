import torch
from torch import nn
from element import Conv, ResBlock, weight_init_kaiming_uniform



class Darknet53(nn.Module):
    def __init__(self, num_classes, depthwise=False):
        super().__init__()
        self.conv1 = Conv(3, 32, kernel_size=3, padding=1, depthwise=depthwise)
        self.res_block1 = self.build_conv_and_resblock(in_channels=32, out_channels=64, num_blocks=1, depthwise=depthwise)
        self.res_block2 = self.build_conv_and_resblock(in_channels=64, out_channels=128, num_blocks=2, depthwise=depthwise)
        self.res_block3 = self.build_conv_and_resblock(in_channels=128, out_channels=256, num_blocks=8, depthwise=depthwise)
        self.res_block4 = self.build_conv_and_resblock(in_channels=256, out_channels=512, num_blocks=8, depthwise=depthwise)
        self.res_block5 = self.build_conv_and_resblock(in_channels=512, out_channels=1024, num_blocks=4, depthwise=depthwise)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)
        self.apply(weight_init_kaiming_uniform)


    def forward(self, x):
        out = self.conv1(x)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.res_block4(out)
        out = self.res_block5(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


    def build_conv_and_resblock(self, in_channels, out_channels, num_blocks, depthwise=False):
        model = nn.Sequential()
        model.add_module("conv", Conv(in_channels, out_channels, kernel_size=3, stride=2, padding=1, depthwise=depthwise))
        for idx in range(num_blocks):
            model.add_module(f"res{idx}", ResBlock(out_channels, depthwise=depthwise))
        return model



def build_darknet53(num_classes=1000, depthwise=False):
    model = Darknet53(num_classes=num_classes, depthwise=depthwise)
    return model



if __name__ == "__main__":
    input_size = 224
    num_classes = 1000
    device = torch.device("cpu")
    model = build_darknet53(num_classes=num_classes, depthwise=False)

    x = torch.randn(2, 3, input_size, input_size).to(device)
    y = model(x)
    print(y.shape)