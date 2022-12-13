import os
import sys
import random
import pprint
import platform
import argparse
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import torch
from torch.cuda import amp
from tqdm import tqdm, trange
from thop import profile
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, distributed

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[0]
OS_SYSTEM = platform.system()
TIMESTAMP = datetime.today().strftime('%Y-%m-%d_%H-%M')
cudnn.benchmark = True
SEED = 2023
random.seed(SEED)
torch.manual_seed(SEED)

from dataloader import build_dataset
from model import build_model
from utils import set_lr, build_basic_logger, setup_worker_logging, setup_primary_logging
from val import validate


def setup(rank, world_size):
    if OS_SYSTEM == "Linux":
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    if OS_SYSTEM == "Linux":
        dist.destroy_process_group()


def train(args, dataloader, model, criterion, optimizer, scaler):
    model.train()
    optimizer.zero_grad()
    sum_loss = 0.0

    for i, minibatch in enumerate(dataloader):
        ni = i + len(dataloader) * (epoch - 1)
        if ni <= args.nw:
            set_lr(optimizer, args.base_lr * pow(ni / (args.nw), 2))

        images, labels = minibatch[0], minibatch[1]

        with amp.autocast(enabled=not args.no_amp):
            predictions = model(images.cuda(args.rank, non_blocking=True))
            loss = criterion(predictions, labels.cuda(args.rank, non_blocking=True))

        scaler.scale(loss * args.world_size).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if not torch.isfinite(loss):
            print(f'############## Loss is Nan/Inf ! ##############')
            sys.exit(0)
        else:
            sum_loss += loss.item()
    
    del images, predictions, labels
    torch.cuda.empty_cache()
    
    loss_str = f"[Train-Epoch:{epoch:03d}] Loss: {sum_loss/len(dataloader):.4f}\t"
    return loss_str


def parse_args(make_dirs=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Name to log training")
    parser.add_argument("--data", type=str, default="imagenet.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--img_size", type=int, default=224, help="Model input size")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=120, help="Number of training epochs")
    parser.add_argument("--warmup", type=int, default=5, help="Epochs for warming up training")
    parser.add_argument("--base_lr", type=float, default=0.1, help="Base learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers used in dataloader")
    parser.add_argument("--world_size", type=int, default=1, help="Number of available GPU devices")
    parser.add_argument("--rank", type=int, default=0, help="Process id for computation")
    parser.add_argument("--no_amp", action="store_true", help="Use of FP32 training (default: AMP training)")
    parser.add_argument("--width_multiple", type=float, default=1.0, help="CSP-Layer channel multiple")
    parser.add_argument("--depth_multiple", type=float, default=1.0, help="CSP-Model depth multiple")
    parser.add_argument("--depthwise", action="store_true", help="Use of Depth-separable conv operation")
    parser.add_argument("--resume", action="store_true", help="Name to resume path")

    args = parser.parse_args()
    args.data = ROOT / "data" / args.data
    args.exp_path = ROOT / "experiment" / args.exp
    args.weight_dir = args.exp_path / "weight"
    args.load_path = args.weight_dir / "last.pt" if args.resume else None
    assert args.world_size > 0, "Executable GPU machine does not exist, This training supports on CUDA available environment."

    if make_dirs:
        os.makedirs(args.weight_dir, exist_ok=True)
    return args


def main_work(rank, world_size, args, logger):
    ################################### Init Process ####################################
    setup(rank, world_size)
    torch.manual_seed(SEED)
    torch.cuda.set_device(rank)

    if OS_SYSTEM == 'Linux':
        import logging
        setup_worker_logging(rank, logger)
    else:
        logging = logger
    ################################### Init Instance ###################################
    global epoch

    args.rank = rank
    args.batch_size //= world_size
    args.base_lr *= (args.batch_size / 256)
    args.workers = min([os.cpu_count() // max(world_size, 1), args.batch_size if args.batch_size > 1 else 0, args.workers])

    dataset, class_list = build_dataset(yaml_path=args.data, input_size=args.img_size)
    train_sampler = distributed.DistributedSampler(dataset=dataset["train"], num_replicas=world_size, rank=args.rank, shuffle=True)
    train_loader = DataLoader(dataset=dataset["train"], batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers, sampler=train_sampler)
    val_loader = DataLoader(dataset=dataset["val"], batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    args.nw = max(round(args.warmup * len(train_loader)), 100)

    model = build_model(arch_name=args.model, num_classes=len(class_list), width_multiple=args.width_multiple, depth_multiple=args.depth_multiple, depthwise=args.depthwise)
    macs, params = profile(deepcopy(model), inputs=(torch.randn(1, 3, args.img_size, args.img_size),), verbose=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.SGD(model.parameters(), args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scaler = amp.GradScaler(enabled=not args.no_amp)

    model = model.cuda(args.rank)
    if OS_SYSTEM == "Linux":
        model = DDP(model, device_ids=[args.rank])
        dist.barrier()
    #################################### Load Model #####################################

    if args.resume:
        assert args.load_path.is_file(), "Not exist trained weights in the directory path !"
        ckpt = torch.load(args.load_path, map_location="cpu")
        start_epoch = ckpt["running_epoch"]
        if hasattr(model, "module"):
            model.module.load_state_dict(ckpt["model_state"], strict=True)
        else:
            model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(args.rank)
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    else:
        start_epoch = 1
        if args.rank == 0:
            logging.warning(f"[Arguments]\n{pprint.pformat(vars(args))}\n")
            logging.warning(f"Architecture Info - Params(M): {params/1e+6:.2f}, FLOPS(B): {2*macs/1E+9:.2f}")
    
    #################################### Train Model ####################################
    if args.rank == 0:
        progress_bar = trange(start_epoch, args.num_epochs+1, total=args.num_epochs, initial=start_epoch, ncols=115)
    else:
        progress_bar = range(start_epoch, args.num_epochs+1)

    best_epoch, best_score, best_perf_str = 0, 0, ""

    for epoch in progress_bar:
        if args.rank == 0:
            train_loader = tqdm(train_loader, desc=f"[TRAIN:{epoch:03d}/{args.num_epochs:03d}]", ncols=115, leave=False)
        train_sampler.set_epoch(epoch)
        train_loss_str = train(args=args, dataloader=train_loader, model=model, criterion=criterion, optimizer=optimizer, scaler=scaler)

        if args.rank == 0:
            save_opt = {"running_epoch": epoch,
                        "class_list": class_list,
                        "model": args.model,
                        "width_multiple": args.width_multiple,
                        "depth_multiple": args.depth_multiple,
                        "depthwise": args.depthwise,
                        "model_state": deepcopy(model.module).state_dict() if hasattr(model, "module") else deepcopy(model).state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict()}
            torch.save(save_opt, args.weight_dir / "last.pt")

            val_loader = tqdm(val_loader, desc=f"[VAL:{epoch:03d}/{args.num_epochs:03d}]", ncols=115, leave=False)
            top1_acc, eval_text = validate(args=args, dataloader=val_loader, model=model, epoch=epoch)
            logging.warning(train_loss_str + eval_text)

            if top1_acc > best_score:
                best_epoch, best_score, best_perf_str = epoch, top1_acc, eval_text
                torch.save(save_opt, args.weight_dir / "best.pt")
        scheduler.step()

    if args.rank == 0:
        logging.warning(f"[Best Performance at {best_epoch}]\n{best_perf_str}")
    cleanup()



if __name__ == "__main__":
    args = parse_args(make_dirs=True)

    if OS_SYSTEM == "Linux":
        torch.multiprocessing.set_start_method("spawn", force=True)
        logger = setup_primary_logging(args.exp_path / "train.log")
        mp.spawn(main_work, args=(args.world_size, args, logger), nprocs=args.world_size, join=True)
    else:
        logger = build_basic_logger(args.exp_path / "train.log")
        main_work(rank=0, world_size=1, args=args, logger=logger)
