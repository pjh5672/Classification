import pprint
import platform
import argparse
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[0]
TIMESTAMP = datetime.today().strftime("%Y-%m-%d_%H-%M")
OS_SYSTEM = platform.system()
SEED = 2023
torch.manual_seed(SEED)

from dataloader import build_dataset
from model import build_model
from utils import build_logger


def parse_args():
    """parse argument parameters for evaluation

    Returns:
        argparse: parameters related to evaluation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True, help='Name to log training')
    parser.add_argument('--data', type=str, default='cub200', help='dataset name(must be match to <dataset>.yaml')
    parser.add_argument('--img-size', type=int, default=224, help='Model input size')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--ckpt-name', type=str, default='best.pt', help='Path to trained model')
    parser.add_argument('--rank', type=int, default=0, help='Process id for computation')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for dataloader')

    args = parser.parse_args()
    args.data = ROOT / 'data' / args.data
    args.exp_path = ROOT / 'experiment' / args.exp
    args.ckpt_path = args.exp_path / 'weight' / args.ckpt_name
    return args


def main_work(**kwargs):
    """load dataloader and trained classifier model and evaluate it.
    """
    logger.info(f'[Arguments]\n{pprint.pformat(vars(args))}\n')

    _, val_dataset, _ = build_dataset(str(args.data)+'.yaml', input_size=args.img_size)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    val_loader = tqdm(val_loader, desc='[VAL]', ncols=110, leave=False)
    ckpt = torch.load(args.ckpt_path, map_location = {'cpu':'cuda:%d' %args.rank})
    model = build_model(arch_name=ckpt['model'], num_classes=len(ckpt['class_list']))
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.cuda(args.rank)
    
    _, _, eval_msg = validate(args=args, dataloader=val_loader, model=model)
    logger.info(f'[Validation Result]\n{eval_msg}')
    

@torch.no_grad()
def validate(args, dataloader, model, epoch=0):
    """evaluate trained classifier model.

    Args:
        args (argparse): parameters related to training
        dataloader (torch.utils.data.DataLoader): dataloader for forwarding
        model (torch.nn.Module): trained classifier architecture
        epoch (int, optional): used only in calling train phase. Defaults to 0.

    Returns:
        float: top-1, top-5 accuracy after evaluation
        str: accuracy message for the given epoch
    """
    model.eval()
    avg_top1_acc = 0.0
    avg_top5_acc = 0.0

    for _, minibatch in enumerate(dataloader):
        images, labels = minibatch[0], minibatch[1]
        predictions = model(images.cuda(args.rank, non_blocking=True))
        acc = accuracy(predictions, labels.cuda(args.rank, non_blocking=True), topk=(1, 5))
        avg_top1_acc += acc[0].item()
        avg_top5_acc += acc[1].item()
        
    del images, predictions
    torch.cuda.empty_cache()
    
    avg_top1_acc /= len(dataloader)
    avg_top5_acc /= len(dataloader)
    acc_msg = f'[Val-Epoch:{epoch:03d}] Top1 Acc: {avg_top1_acc:.2f} Top5 Acc: {avg_top5_acc:.2f}'
    return avg_top1_acc, avg_top5_acc, acc_msg


def accuracy(output, target, topk=(1,)):
    """computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output (_type_): _description_
        target (_type_): _description_
        topk (tuple, optional): _description_. Defaults to (1,).

    Returns:
        _type_: _description_
    """
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


if __name__ == "__main__":
    args = parse_args()
    logger = build_logger(log_file_path=args.exp_path / 'val.log', set_level=1)
    main_work(args=args, logger=logger)
