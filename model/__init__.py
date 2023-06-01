import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from resnet import build_resnet
from darknet19 import build_darknet19
from darknet53 import build_darknet53
from darknet53_tiny import build_darknet53_tiny
from cspdarknet53 import build_CSPdarknet53
from mobilenetv1 import build_mobilenetv1
from mobilenetv2 import build_mobilenetv2
from mobilenetv3 import build_mobilenetv3


def build_model(arch_name, num_classes=1000, width_multiple=1.0, depth_multiple=1.0, mode="large", pretrained=False):
    if arch_name.startswith("resnet"):
        model = build_resnet(arch_name=arch_name, num_classes=num_classes)
    elif arch_name == "darknet19":
        model = build_darknet19(num_classes=num_classes)
    elif arch_name == "darknet53":
        model = build_darknet53(num_classes=num_classes)
    elif arch_name == "darknet53-tiny":
        model = build_darknet53_tiny(num_classes=num_classes)
    elif arch_name == "csp-darknet53":
        model = build_CSPdarknet53(num_classes=num_classes, width_multiple=width_multiple, depth_multiple=depth_multiple)
    elif arch_name == "mobilenet-v1":
        model = build_mobilenetv1(num_classes=num_classes, width_multiple=width_multiple)
    elif arch_name == "mobilenet-v2":
        model = build_mobilenetv2(num_classes=num_classes, width_multiple=width_multiple)
    elif arch_name == "mobilenet-v3":
        model = build_mobilenetv3(num_classes=num_classes, mode=mode, width_multiple=width_multiple)
    else:
        raise RuntimeError("Only support model in [resnet(18-101), darknet(19, 53, 53-tiny), csp-darknet53, mobilenet(v1-v3)]")
    
    if pretrained:
        if arch_name == "mobilenet-v3":
            arch_name += f"-{mode}"
        ckpt = torch.load(ROOT.parents[0] / "weights" / f"{arch_name}.pt", map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
    return model


if __name__ == "__main__":
    from thop import profile

    input_size = 224
    num_classes = 1000
    device = torch.device('cpu')
    model = build_model(arch_name="resnet18", num_classes=1000, width_multiple=1.0, depth_multiple=1.0, mode="large", pretrained=False)
    
    x = torch.randn(2, 3, input_size, input_size).to(device)
    y = model(x)
    print(y.shape)

    mb, gb = 1 << 20, 1 << 30
    macs, params = profile(model, inputs=(torch.randn(1, 3, input_size, input_size),), verbose=False)
    print(f"Params(M): {params/mb:.2f}, FLOPS(B): {2*macs/gb:.2f}")