import torch
from torch import nn
from element import BasicBlock, BottleNeck, weight_init_xavier_uniform



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, zero_init_residual=True):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.apply(weight_init_xavier_uniform)
        
        if zero_init_residual:
            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out



def build_resnet(arch_name="resnet18", num_classes=1000):
    if arch_name == "resnet18":
        model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
    elif arch_name == "resnet34":
        model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes)
    elif arch_name == "resnet50":
        model = ResNet(block=BottleNeck, layers=[3, 4, 6, 3], num_classes=num_classes)
    elif arch_name == "resnet101":
        model = ResNet(block=BottleNeck, layers=[3, 4, 23, 3], num_classes=num_classes)
    else:
        raise RuntimeError("Only support model in [vgg16, vgg16_bn, resnet18, resnet34, resnet50, resnet101]")
    return model



if __name__ == "__main__":
    input_size = 224
    num_classes = 1000
    device = torch.device('cpu')
    model = build_resnet(arch_name="resnet18", num_classes=num_classes)

    x = torch.randn(2, 3, input_size, input_size).to(device)
    y = model(x)
    print(y.shape)