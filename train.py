import os
import argparse
from pathlib import Path
from copy import deepcopy

import torch
from tqdm import tqdm
from torch.cuda import amp
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from dataloader import build_dataset
from model import build_model
from val import validate, save_result
from utils.parse import Parser
from utils.general import LOGGER, print_args, init_seeds, seed_worker, TQDM_BAR_FORMAT, colorstr
from utils.torch_utils import (select_device, torch_distributed_zero_first, model_info, set_lr,
                               build_criterion, build_optimizer, build_scheduler, resume_state,
                               time_sync, de_parallel)
from utils.meter import AverageMeter
from utils.eval import compute_accuracy
from utils.evolve import Evolution

ROOT = Path(__file__).resolve().parents[0]
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
GLOBAL_RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(opt, device):
    (dataset, base_lr, batch_size, workers, seed, arch_name, class_list, 
     val_size, loss_type, label_smoothing, momentum, weight_decay, 
     lr_decay, num_epochs, no_amp, resume, load_path, warmup, project_dir,
     weight_dir, mobile_v3, width_multiple, depth_multiple, pretrained, evolve) = \
        (opt.dataset, opt.base_lr, opt.batch_size, opt.workers, 
         opt.seed, opt.arch, opt.class_list, opt.val_size, opt.loss_type, 
         opt.label_smoothing, opt.momentum, opt.weight_decay, opt.lr_decay, 
         opt.num_epochs, opt.no_amp, opt.resume, opt.load_path, opt.warmup, 
         opt.project_dir, opt.weight_dir, opt.mobile_v3, opt.width_multiple, 
         opt.depth_multiple, opt.pretrained, opt.evolve)

    init_seeds(opt.seed + 1 + GLOBAL_RANK, deterministic=True)
    batch_size //= WORLD_SIZE
    if dataset == 'imagenet':
        base_lr *= (batch_size / 256)
    
    with torch_distributed_zero_first(LOCAL_RANK):
        train_dataset, val_dataset = build_dataset(opt=opt)

    workers = min([os.cpu_count() // max(torch.cuda.device_count(), 1), 
                   batch_size if batch_size > 1 else 0, workers])

    sampler = None if LOCAL_RANK == -1 else distributed.DistributedSampler(train_dataset, 
                                                                           num_replicas=WORLD_SIZE,
                                                                           rank=GLOBAL_RANK,
                                                                           shuffle=True)
    generator = torch.Generator()
    generator.manual_seed(seed + GLOBAL_RANK)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=sampler is None, pin_memory=True, num_workers=workers, 
                              sampler=sampler, worker_init_fn=seed_worker, generator=generator)

    if GLOBAL_RANK in {-1, 0}:
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                                shuffle=False, pin_memory=True, num_workers=workers, 
                                worker_init_fn=seed_worker, generator=generator)
    
    model = build_model(arch_name=arch_name, num_classes=len(class_list), 
                        width_multiple=width_multiple, depth_multiple=depth_multiple, 
                        mode=mobile_v3, pretrained=pretrained)
    criterion = build_criterion(name=loss_type, label_smoothing=label_smoothing)
    optimizer = build_optimizer(model=model, lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer=optimizer, lr_decay=lr_decay, num_epochs=num_epochs)
    scaler = amp.GradScaler(enabled=not no_amp)

    if not evolve:
        model_info(model=model, input_size=val_size)

    on_epoch = 1
    nw = max(round(warmup * len(train_loader)), 100)
    model = model.to(device)
    
    if resume:
       assert load_path.exists(), 'Not exist trained weights in the defined directory path'
       on_epoch = resume_state(load_path, model, optimizer, scheduler, scaler, device)
       
    if GLOBAL_RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    best_epoch, best_top1, best_top5, best_loss = [0] * 4
    epoch_time = AverageMeter('Epoch', ':5.3f')
    for epoch in range(on_epoch, num_epochs+1):
        if GLOBAL_RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        
        LOGGER.info(('\n' + '%14s' * 7) % ('Epoch', 'GPU_mem', 'Time/Epoch', 
                                           'Time/Batch', 'Loss', 'Acc@1', 'Acc@5'))
        if GLOBAL_RANK in {-1, 0}:
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format=TQDM_BAR_FORMAT)

        model.train()
        optimizer.zero_grad()
        batch_time = AverageMeter('Batch', ':5.3f')
        train_loss = AverageMeter('Loss', ':5.4f')
        train_top1 = AverageMeter('Acc@1', ':5.2f')
        train_top5 = AverageMeter('Acc@5', ':5.2f')
        
        t1 = time_sync()
        for i, batch in pbar:
            t2 = time_sync()
            ni = i + len(train_loader) * (epoch - 1)
            if ni <= nw:
                set_lr(optimizer, base_lr * pow(ni / nw, 2))
            
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)

            with amp.autocast(enabled=not no_amp):
                predictions = model(images)
                loss = criterion(predictions, labels)
            acc1, acc5 = compute_accuracy(predictions, labels, topk=(1, 5))
            
            train_loss.update(loss.item(), images.size(0))
            train_top1.update(acc1[0].item(), images.size(0))
            train_top5.update(acc5[0].item(), images.size(0))

            optimizer.zero_grad()
            scaler.scale(loss * WORLD_SIZE).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_time.update(time_sync() - t2)
            if GLOBAL_RANK in {-1, 0}:
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(('%14s' * 2 + '%14.4g' * 5) % 
                                     (f'{epoch}/{num_epochs}', mem, epoch_time.val, batch_time.val,
                                      train_loss.avg, train_top1.avg, train_top5.avg))
        
        scheduler.step()
        epoch_time.update(time_sync() - t1)

        if GLOBAL_RANK in {-1, 0}:
            val_loss, val_top1, val_top5 = validate(loader=val_loader, model=model, 
                                                    criterion=criterion, device=device)
            keys = ('Epoch', 'metric/Acc@1', 'metric/Acc@5', f'Loss/{loss_type.upper()}')
            vals = (epoch, val_top1, val_top5, val_loss)
            save_result(keys=keys, vals=vals, save_dir=project_dir)
            save_obj = {
                'dataset': dataset,
                'on_epoch': epoch, 
                'class_list': class_list,
                'arch_name': arch_name, 
                'mobile_v3': mobile_v3, 
                'width_multiple': width_multiple,
                'depth_multiple': depth_multiple,
                'model_state': deepcopy(de_parallel(model)).state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss_type': loss_type,
                'label_smoothing': label_smoothing
            }
            torch.save(save_obj, weight_dir / 'last.pt')

            if val_top1 > best_top1:
                best_epoch, best_top1, best_top5, best_loss = \
                    epoch, val_top1, val_top5, val_loss
                torch.save(save_obj, weight_dir / 'best.pt')

    LOGGER.info(('\n' + '%27s' + '%14s' * 3) % (colorstr('red', 'bold', 'Final'), 
                                                'Best Epoch', 'Best Acc@1', 'Best Acc@5'))
    LOGGER.info(('%27s' + '%14i' + '%14.4g' * 2) % (colorstr('red', 'bold', 'Result'), 
                                                    best_epoch, best_top1, best_top5))
    return best_top1, best_top5, best_loss
                

def build_parser(makedirs=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, required=True, help='Name to train project')
    parser.add_argument('--dataset', type=str, default='mit67', help='Dataset name')
    parser.add_argument('--arch', type=str, default='resnet18', help='Model architecture')
    parser.add_argument('--mobile-v3', type=str, default='large', help='Mobilenetv3 architecture mode')
    parser.add_argument('--width-multiple', type=float, default=1.0, help='CSP-Layer channel multiple')
    parser.add_argument('--depth-multiple', type=float, default=1.0, help='CSP-Model depth multiple')
    parser.add_argument('--pretrained', action='store_true', help='Training with pretrained weights')
    parser.add_argument('--loss-type', type=str, default='ce', help='Loss function')
    parser.add_argument('--train-size', type=int, default=224, help='train input size')
    parser.add_argument('--val-size', type=int, default=224, help='val input size')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=300, help='Number of training epoch')
    parser.add_argument('--base-lr', type=float, default=0.1, help='Base learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.01, help='Epoch to learning rate decay')
    parser.add_argument('--warmup', type=int, default=5, help='Epoch for warming up training')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='Evolve hyperparameters for x generations')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--no-amp', action='store_true', help='Use of FP32 training (default: AMP training)')
    parser.add_argument('--resume', action='store_true', help='Name to resume path')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    
    args = parser.parse_args()
    args.project_dir = ROOT / 'experiment' / args.project
    args.weight_dir = args.project_dir / 'weight'
    args.evolve_dir = args.project_dir / 'evolve'
    args.load_path = args.weight_dir / 'last.pt' if args.resume else None
    
    if makedirs:
        os.makedirs(args.weight_dir, exist_ok=True)
        
    parser = Parser(data_dir=ROOT / 'data', dataset=args.dataset)
    parser.args = args
    opt = parser.build_opt()
    
    for k, v in vars(args).items():
        setattr(opt, k, v)
    return opt, parser


def main(opt, parser):
    if GLOBAL_RANK in {-1, 0}:
        print_args(args=vars(opt), 
                   exclude_keys=('class_list', 'weight_dir', 'evolve_dir', 'load_path'))
    if opt.evolve:
        os.makedirs(opt.evolve_dir, exist_ok=True)

    device = select_device(device=opt.device, batch_size=opt.batch_size)
    assert device.type != 'cpu', f'can not support CPU training'

    if LOCAL_RANK != -1:
        msg = 'is not compatible with Multi-GPU DDP training'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')

    if not opt.evolve:
        parser.save_yaml(save_dir=opt.project_dir)
        _ = train(opt, device)
    else:
        for _ in range(opt.evolve):  # generations to evolve
            evolve = Evolution(save_dir=opt.evolve_dir)
            hyp = {k: vars(opt)[k] for k in list(evolve.params.keys())}
            evolve.run(hyp=hyp)
            results = train(argparse.Namespace(**dict(vars(opt), **hyp)), device)
            keys = ('metric/Acc@1', 'metric/Acc@5', f'Loss/{opt.loss_type.upper()}')
            evolve.write_results(hyp=hyp, keys=keys, results=results)

        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bright_yellow', 'bold', opt.evolve_dir)}")
        
if __name__ == "__main__":
    opt, parser = build_parser(makedirs=True)
    main(opt=opt, parser=parser)
