import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from resnet import build_resnet
from darknet19 import build_darknet19
from darknet53 import build_darknet53
from darknet53_tiny import build_darknet53_tiny
from cspdarknet53 import build_CSPdarknet53



def build_model(arch_name, num_classes=1000, width_multiple=1.0, depth_multiple=1.0):
    if arch_name.startswith("resnet"):
        model = build_resnet(arch_name=arch_name, num_classes=num_classes)
    elif arch_name == "darknet19":
        model = build_darknet19(num_classes=num_classes)
    elif arch_name == "darknet53":
        model = build_darknet53(num_classes=num_classes)
    elif arch_name == "darknet53-tiny":
        model = build_darknet53_tiny(num_classes=num_classes)
    elif arch_name == "cspdarknet53":
        model = build_CSPdarknet53(num_classes=num_classes, width_multiple=width_multiple, depth_multiple=depth_multiple)
    else:
        raise RuntimeError("Only support model in [resnet18, resnet34, resnet50, resnet101, darknet19, darknet53, csp-darknet53]")
    return model


if __name__ == "__main__":
    import torch

    input_size = 224
    num_classes = 1000
    device = torch.device('cpu')
    model = build_model(arch_name="resnet18", num_classes=1000, width_multiple=1.0, depth_multiple=1.0)

    x = torch.randn(2, 3, input_size, input_size).to(device)
    y = model(x)
    print(y.shape)