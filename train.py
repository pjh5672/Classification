import os
import sys
import random
import argparse
import platform
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import torch
from torch.cuda import amp
from thop import profile
from tqdm import tqdm, trange
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, distributed
from torch.nn.parallel import DistributedDataParallel as DDP

SEED = 2023
random.seed(SEED)
torch.manual_seed(SEED)
OS_SYSTEM = platform.system()
ROOT = Path(__file__).resolve().parents[0]
TIMESTAMP = datetime.today().strftime("%Y-%m-%d_%H-%M")

from dataloader import build_dataset
from model import build_model
from utils import (build_optimizer, build_criterion, build_scheduler, compute_accuracy,
                    set_lr, setup_worker_logging, setup_primary_logging, build_logger,
                    resume_state, de_parallel, AverageMeter, ProgressMeter, 
                    synchronize, yaml_save, yaml_load)
from val import validate


def setup(rank, world_size, port):
    """set DDP environment to trigger gradient synchronization across processes

    Args:
        rank (int): distributed process ID
        world_size (int): umber of distributed processes
    """
    if OS_SYSTEM == "Linux":
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        synchronize()


def cleanup():
    """close DDP backend communication.
    """
    if OS_SYSTEM == "Linux":
        dist.destroy_process_group()


def train(args, dataloader, model, criterion, optimizer, scaler):
    """train classifier model on the defined training environment.

    Args:
        args (argparse): parameters related to training
        dataloader (torch.utils.data.DataLoader): dataloader for forwarding
        model (torch.nn.Module): classifier architecture
        criterion (torch.nn): loss function
        optimizer (torch.optim): gradient optimization method
        scaler (torch.cuda.GradScaler): loss scaler in case of using mixed precision training

    Returns:
        str: train loss message for each epoch
    """
    losses = AverageMeter('Loss', ':5.4f')
    top1 = AverageMeter('Acc@1', ':5.2f')
    top5 = AverageMeter('Acc@5', ':5.2f')
    progress = ProgressMeter(num_epochs=args.num_epoch, 
                             meters=[losses, top1, top5], prefix='Train:')
    model.train()

    for i, batch in enumerate(dataloader):
        ni = i + len(dataloader) * (epoch - 1)
        if ni <= args.nw:
            set_lr(optimizer, args.base_lr * pow(ni / (args.nw), 2))
        
        images = batch[0].cuda(args.rank, non_blocking=True)
        labels = batch[1].cuda(args.rank, non_blocking=True)

        with amp.autocast(enabled=not args.no_amp):
            predictions = model(images)
            loss = criterion(predictions, labels)
        acc1, acc5 = compute_accuracy(predictions, labels, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        optimizer.zero_grad()
        scaler.scale(loss * args.world_size).backward()
        scaler.step(optimizer)
        scaler.update()

        if not torch.isfinite(loss):
            print(" @@@@@@@ Loss is Nan/Inf @@@@@@@ ")
            sys.exit(0)
    
    del images, predictions, labels
    torch.cuda.empty_cache()
    progress_msg = progress.get_meter(epoch)
    return progress_msg


def parse_opt(make_dirs=True):
    """parse argument parameters for training

    Args:
        make_dirs (bool, optional): make experiment logging folder. Defaults to True.

    Returns:
        argparse: parameters related to training
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True, help='Name to log training')
    parser.add_argument('--dataset', type=str, default='mit67', help='Dataset name')
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture')
    parser.add_argument('--mobile-v3', type=str, default='large', help='Mobilenetv3 architecture mode')
    parser.add_argument('--img-size', type=int, default=224, help='Model input size')
    parser.add_argument('--loss', type=str, default='ce', help='Loss function')
    parser.add_argument('--optim', type=str, default='sgd', help='Optimizor for training')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--num-epoch', type=int, default=200, help='Number of training epoch')
    parser.add_argument('--base-lr', type=float, default=0.1, help='Base learning rate')
    parser.add_argument('--lr-decay', type=float, default=1e-4, help='Epoch to learning rate decay')
    parser.add_argument('--warmup', type=int, default=5, help='Epoch for warming up training')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--world-size', type=int, default=1, help='Number of available GPU devices')
    parser.add_argument('--rank', type=int, default=0, help='Process id for computation')
    parser.add_argument('--no-amp', action='store_true', help='Use of FP32 training (default: AMP training)')
    parser.add_argument('--width-multiple', type=float, default=1.0, help='CSP-Layer channel multiple')
    parser.add_argument('--depth-multiple', type=float, default=1.0, help='CSP-Model depth multiple')
    parser.add_argument('--pretrained', action='store_true', help='Training with pretrained weights')
    parser.add_argument('--resume', action='store_true', help='Name to resume path')
    parser.add_argument('--cache', action='store_true', help='Cache image in RAM')
    parser.add_argument('--port', type=str, default='5555', help='Backend port for gradient communication')
    
    args = parser.parse_args()
    args.data = ROOT / 'data' / f'{args.dataset}.data.yaml'
    args.hyp = ROOT / 'data' / f'{args.dataset}.hyp.yaml'
    args.exp_path = ROOT / 'experiment' / args.exp
    args.weight_dir = args.exp_path / 'weight'
    args.load_path = args.weight_dir / 'last.pt' if args.resume else None
    assert args.world_size > 0, 'Executable GPU machine does not exist, support CUDA-available env'
    
    if make_dirs:
        os.makedirs(args.weight_dir, exist_ok=True)
    
    opt = argparse.Namespace()
    opt.data = argparse.Namespace(**yaml_load(args.data))
    opt.hyp = argparse.Namespace(**yaml_load(args.hyp))
    opt.args = args
    opt.input_size = args.img_size
    return opt


def main_work(rank, world_size, opt, logger):
    """train and evaluate classifier model on the defined training environment.

    Args:
        rank (int): distributed process ID
        world_size (int): number of distributed processes
        args (argparse): parameters related to training
        logger (logging): logging instance
    """
    args, data = opt.args, opt.data

    ################################### Init Process ####################################
    setup(rank, world_size, args.port)
    torch.cuda.set_device(rank)

    if OS_SYSTEM == "Linux":
        import logging
        setup_worker_logging(rank, logger)
    else:
        logging = logger

    ################################### Init Instance ###################################
    global epoch
    
    args.batch_size //= world_size
    if args.dataset == 'imagenet':
        args.base_lr *= (args.batch_size / 256)
    args.workers = min([os.cpu_count() // max(world_size, 1), args.batch_size if args.batch_size > 1 else 0, args.workers])

    yaml_save(args.exp_path / f'{args.dataset}.args.yaml', vars(opt.args))
    yaml_save(args.exp_path / f'{args.dataset}.data.yaml', vars(opt.data))
    yaml_save(args.exp_path / f'{args.dataset}.hyp.yaml', vars(opt.hyp), default_flow_style=None)

    ################################### Init Instance ###################################
    args.rank = rank
    img_size = args.img_size
    batch_size = args.batch_size
    
    train_dataset, val_dataset = build_dataset(opt=opt, cache_images=args.cache)
    train_sampler = distributed.DistributedSampler(dataset=train_dataset, num_replicas=world_size, 
                                                   rank=rank, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, 
                              pin_memory=True, num_workers=args.workers, sampler=train_sampler)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, 
                            pin_memory=True, num_workers=args.workers)
    args.nw = max(round(args.warmup * len(train_loader)), 100)
    model = build_model(arch_name=args.model, num_classes=len(data.CLASS_INFO), 
                        width_multiple=args.width_multiple, depth_multiple=args.depth_multiple, 
                        mode=args.mobile_v3, pretrained=args.pretrained)
    macs, params = profile(deepcopy(model), inputs=(torch.randn(1, 3, img_size, img_size),), verbose=False)
    criterion = build_criterion(name=args.loss, label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model=model, name=args.optim, lr=args.base_lr, 
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer=optimizer, name=args.dataset, 
                                num_epoch=args.num_epoch, lr_decay=args.lr_decay)
    scaler = amp.GradScaler(enabled=not args.no_amp)
    model.cuda(args.rank)

    #################################### Load Model #####################################
    if args.resume:
        assert args.load_path.is_file(), 'Not exist trained weights in the directory path'
        start_epoch = resume_state(args.load_path, args.rank, model, optimizer, scheduler, scaler)
    else:
        start_epoch = 1
        if args.rank == 0:
            logging.warning(f'Arch info - Params(M): {params/1e+6:.2f}, FLOPs(G): {2*macs/1E+9:.2f}')

    #################################### Train Model ####################################
    if OS_SYSTEM == "Linux":
        model = DDP(model, device_ids=[args.rank])
    
    if args.rank == 0:
        pbar = trange(start_epoch, args.num_epoch+1, total=args.num_epoch, initial=start_epoch, ncols=110)
    else:
        pbar = range(start_epoch, args.num_epoch+1)

    best_epoch, best_score, best_perf_msg = 0, 0, ''
    for epoch in pbar:
        if args.rank == 0:
            train_loader = tqdm(train_loader, desc=f'[TRAIN:{epoch:03d}/{args.num_epoch:03d}]', ncols=110, leave=False)
        
        train_sampler.set_epoch(epoch)
        train_progress_msg = train(args=args, dataloader=train_loader, model=model, criterion=criterion, 
                                   optimizer=optimizer, scaler=scaler)
        
        if args.rank == 0:
            save_obj = {"running_epoch": epoch,
                        "idx2cls": data.CLASS_INFO,
                        "model": args.model,
                        "mobile_v3": args.mobile_v3, 
                        "width_multiple": args.width_multiple,
                        "depth_multiple": args.depth_multiple,
                        "model_state": deepcopy(de_parallel(model)).state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "loss_type": args.loss,
                        "label_smoothing": args.label_smoothing}
            torch.save(save_obj, args.weight_dir / 'last.pt')

            val_loader = tqdm(val_loader, desc=f'[VAL:{epoch:03d}/{args.num_epoch:03d}]', ncols=110, leave=False)
            top1_acc, val_progress_msg = validate(args=args, dataloader=val_loader, 
                                                  model=model, criterion=criterion, epoch=epoch)
            logging.warning(train_progress_msg)
            logging.warning(val_progress_msg)

            if top1_acc > best_score:
                best_epoch, best_score, best_perf_msg = epoch, top1_acc, val_progress_msg
                torch.save(save_obj, args.weight_dir / 'best.pt')
        scheduler.step()

    if args.rank == 0:
        logging.warning(f'[Best Performance at {best_epoch}]\n{best_perf_msg}')
    cleanup()


if __name__ == "__main__":
    opt = parse_opt(make_dirs=True)
    args = opt.args
    world_size = args.world_size

    if OS_SYSTEM == "Linux":
        torch.multiprocessing.set_start_method('spawn', force=True)
        logger = setup_primary_logging(args.exp_path / 'train.log')
        mp.spawn(main_work, args=(world_size, opt, logger), nprocs=world_size, join=True)
    else:
        logger = build_logger(args.exp_path / 'train.log')
        main_work(rank=0, world_size=1, opt=opt, logger=logger)