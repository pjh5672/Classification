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
from utils import (build_logger, build_criterion, compute_accuracy, 
                   AverageMeter, ProgressMeter, yaml_load)


def parse_opt():
    """parse argument parameters for evaluation

    Returns:
        argparse: parameters related to evaluation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True, help='Name to log training')
    parser.add_argument('--dataset', type=str, default='mit67', help='Dataset name to <data>.yaml')
    parser.add_argument('--img-size', type=int, default=224, help='Model input size')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-epoch', type=int, default=0, help='Number of dummy epoch')
    parser.add_argument('--ckpt-name', type=str, default='best.pt', help='Path to trained model')
    parser.add_argument('--rank', type=int, default=0, help='Process id for computation')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for dataloader')

    args = parser.parse_args()
    args.data = ROOT / 'data' / f'{args.dataset}.data.yaml'
    args.hyp = ROOT / 'data' / f'{args.dataset}.hyp.yaml'
    args.exp_path = ROOT / 'experiment' / args.exp
    args.ckpt_path = args.exp_path / 'weight' / args.ckpt_name

    opt = argparse.Namespace()
    opt.data = argparse.Namespace(**yaml_load(args.data))
    opt.hyp = argparse.Namespace(**yaml_load(args.hyp))
    opt.args = args
    opt.input_size = args.img_size
    return opt


def main_work(**kwargs):
    """load dataloader and trained classifier model and evaluate it.
    """
    args = opt.args
    batch_size = args.batch_size
    num_workers = args.workers
    
    _, val_dataset = build_dataset(opt=opt)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                            shuffle=False, pin_memory=True, num_workers=num_workers)
    val_loader = tqdm(val_loader, desc='[VAL]', ncols=110, leave=False)
    ckpt = torch.load(args.ckpt_path, map_location = {'cpu':'cuda:%d' %args.rank})
    model = build_model(arch_name=ckpt['model'], num_classes=len(ckpt['idx2cls']))
    criterion = build_criterion(name=ckpt['loss_type'], 
                                label_smoothing=ckpt['label_smoothing'])
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.cuda(args.rank)
    
    _, val_progress_msg = validate(args=args, dataloader=val_loader, 
                                   model=model, criterion=criterion)
    logger.info(f'[Validation Result]\n{val_progress_msg}')
    

@torch.no_grad()
def validate(args, dataloader, model, criterion, epoch=0):
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
    losses = AverageMeter('Loss', ':5.4f')
    top1 = AverageMeter('Acc@1', ':5.2f')
    top5 = AverageMeter('Acc@5', ':5.2f')
    progress = ProgressMeter(num_epochs=args.num_epoch, 
                             meters=[losses, top1, top5], prefix='  Val:')
    model.eval()

    for _, batch in enumerate(dataloader):
        images = batch[0].cuda(args.rank, non_blocking=True)
        labels = batch[1].cuda(args.rank, non_blocking=True)

        predictions = model(images)
        loss = criterion(predictions, labels)
        acc1, acc5 = compute_accuracy(predictions, labels, topk=(1, 5))
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
    del images, labels, predictions
    torch.cuda.empty_cache()
    progress_msg = progress.get_summary(epoch)
    return top1.avg, progress_msg


if __name__ == "__main__":
    opt = parse_opt()
    args = opt.args
    logger = build_logger(log_file_path=args.exp_path / 'val.log', set_level=1)
    main_work(opt=opt, logger=logger)
