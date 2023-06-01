import os
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloader import build_dataset
from model import build_model
from utils.general import colorstr, TQDM_BAR_FORMAT, LOGGER
from utils.parse import Parser
from utils.meter import AverageMeter
from utils.eval import compute_accuracy
from utils.torch_utils import select_device, build_criterion
                               
ROOT = Path(__file__).resolve().parents[0]


@torch.no_grad()
def validate(loader, model, criterion, device):
    s = ('%27s' + '%14s' * 3) % (colorstr('bright_green', 'bold', 'Validation'),
                                         'Loss', 'Acc@1', 'Acc@5')
    pbar = tqdm(enumerate(loader), desc=s, total=len(loader), bar_format=TQDM_BAR_FORMAT)

    model.eval()
    val_loss = AverageMeter('Loss', ':5.4f')
    val_top1 = AverageMeter('Acc@1', ':5.2f')
    val_top5 = AverageMeter('Acc@5', ':5.2f')
    
    for _, batch in pbar:
        images = batch[0].to(device, non_blocking=True)
        labels = batch[1].to(device, non_blocking=True)

        predictions = model(images)
        loss = criterion(predictions, labels)  
        acc1, acc5 = compute_accuracy(predictions, labels, topk=(1, 5))
        
        val_loss.update(loss.item(), images.size(0))
        val_top1.update(acc1[0].item(), images.size(0))
        val_top5.update(acc5[0].item(), images.size(0))

    LOGGER.info(('%27s' + '%14.4g' * 3) % (colorstr('bright_green', 'bold', 'Result'), 
                                            val_loss.avg, val_top1.avg, val_top5.avg))
    return val_loss.avg, val_top1.avg, val_top5.avg


def save_result(keys, vals, save_dir):
    # Log to results.csv
    result_csv = save_dir / 'result.csv'
    keys = tuple(x.strip() for x in keys)
    n = len(keys)
    s = '' if result_csv.exists() else (('%16s,' * n % keys).rstrip(',') + '\n')
    with open(result_csv, 'a') as f:
        f.write(s + ('%16.5g,' * n % vals).rstrip(',') + '\n')
        

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, required=True, help='Name to train project')
    parser.add_argument('--dataset', type=str, default='mit67', help='Dataset name')
    parser.add_argument('--val-size', type=int, default=224, help='val input size')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    
    args = parser.parse_args()
    args.project_dir = ROOT / 'experiment' / args.project
    args.weight_dir = args.project_dir / 'weight'
    args.ckpt_path = args.weight_dir / 'best.pt'
        
    parser = Parser(data_dir=ROOT / 'data', dataset=args.dataset)
    parser.args = args
    opt = parser.build_opt()
    
    for k, v in vars(args).items():
        setattr(opt, k, v)
    return opt, parser


def main(opt, parser):
    device = select_device(device=opt.device, batch_size=opt.batch_size)
    ckpt = torch.load(opt.ckpt_path)
    opt = parser.change_dataset(to=ckpt['dataset'])
    opt.train_size = opt.val_size
    _, val_dataset = build_dataset(opt=opt)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, 
                            shuffle=False, pin_memory=True, num_workers=opt.workers)
    model = build_model(arch_name=ckpt['arch_name'], num_classes=len(ckpt['class_list']), 
                        width_multiple=ckpt['width_multiple'], depth_multiple=ckpt['depth_multiple'], 
                        mode=ckpt['mobile_v3'])
    criterion = build_criterion(name=ckpt['loss_type'], 
                                label_smoothing=ckpt['label_smoothing'])
    model.load_state_dict(ckpt['model_state'], strict=True)
    model = model.to(device)
    _ = validate(loader=val_loader, model=model, criterion=criterion, device=device)
    
    
if __name__ == "__main__":
    opt, parser = build_parser()
    main(opt=opt, parser=parser)