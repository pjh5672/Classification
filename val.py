import os
import json
import pprint
import platform
import argparse
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[0]
TIMESTAMP = datetime.today().strftime('%Y-%m-%d_%H-%M')
OS_SYSTEM = platform.system()
SEED = 2023
torch.manual_seed(SEED)

from dataloader import build_dataset
from model import build_model
from utils import build_basic_logger



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@torch.no_grad()
def validate(args, dataloader, model, epoch=0):
    model.eval()
    sum_top1_acc = 0.0

    for _, minibatch in enumerate(dataloader):
        images, labels = minibatch[0], minibatch[1]
        predictions = model(images.cuda(args.rank, non_blocking=True))
        acc = accuracy(predictions, labels.cuda(args.rank, non_blocking=True), topk=(1, ))
        sum_top1_acc += acc[0].item()
        
    sum_top1_acc /= len(dataloader)
    acc_str = f"[Val-Epoch:{epoch:03d}] Top-1 Acc: {sum_top1_acc:.2f}"
    return sum_top1_acc, acc_str



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Name to log training")
    parser.add_argument("--data", type=str, default="toy.yaml", help="Path to data.yaml")
    parser.add_argument("--img_size", type=int, default=224, help="Model input size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--ckpt_name", type=str, default="best.pt", help="Path to trained model")
    parser.add_argument("--rank", type=int, default=0, help="Process id for computation")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers used in dataloader")

    args = parser.parse_args()
    args.data = ROOT / "data" / args.data
    args.exp_path = ROOT / "experiment" / args.exp
    args.ckpt_path = args.exp_path / "weight" / args.ckpt_name
    return args


def main():
    args = parse_args()
    logger = build_basic_logger(args.exp_path / "val.log", set_level=1)
    logger.info(f"[Arguments]\n{pprint.pformat(vars(args))}\n")

    dataset, _ = build_dataset(yaml_path=args.data, input_size=args.img_size)
    val_loader = DataLoader(dataset=dataset["val"], batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    ckpt = torch.load(args.ckpt_path, map_location = {"cpu":"cuda:%d" %args.rank})
    model = build_model(arch_name=ckpt["model"], num_classes=len(ckpt["class_list"]), width_multiple=ckpt["width_multiple"], depth_multiple=ckpt["depth_multiple"], depthwise=ckpt["depthwise"])
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.cuda(args.rank)

    val_loader = tqdm(val_loader, desc=f"[VAL:{0:03d}/{args.num_epochs:03d}]", ncols=115, leave=False)
    _, eval_text = validate(args=args, dataloader=val_loader, model=model)
    logger.info(f"[Validation Result]\n{eval_text}")
    

if __name__ == "__main__":
    main()