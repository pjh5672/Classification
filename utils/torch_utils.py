import os
import time
import math
import platform
from copy import deepcopy
from contextlib import contextmanager

import torch
from torch import nn
from torch import optim
from thop import profile
import torch.distributed as dist

from utils.general import LOGGING_NAME, LOGGER, file_date, colorstr


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    # Decorator to make all processes in distributed training wait for each local_master to do something
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def model_info(model, input_size):
    macs, params = profile(deepcopy(model), 
                           inputs=(torch.randn(1, 3, input_size, input_size),), 
                           verbose=False)
    mb, gb = 1 << 20, 1 << 30
    s = colorstr('bright_magenta', 'bold', f'{model.__class__.__name__} infomation') + \
        f' Params(M): {params / mb:.2f}, FLOPs(G): {2*macs / gb:.2f}'
    LOGGER.info(s)
    
    
def select_device(device='', batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f'{colorstr("bright_magenta", "bold", LOGGING_NAME)} ' + \
        f'🚀 {file_date()} Python-{platform.python_version()} torch-{torch.__version__} '
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '') # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if not newline:
        s = s.rstrip()
    LOGGER.info(s)
    return torch.device(arg)


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def build_criterion(name='ce', label_smoothing=0.0):
    assert name in ['ce'], f"not support criterion(loss function), got {name}."
    
    if name == 'ce':
        obj_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return obj_func


def build_optimizer(model, lr=0.001, momentum=0.9, weight_decay=1e-5):
    # No bias decay heuristic recommendation
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == 'bias': # bias (no decay)
                g[2].append(p)
            elif p_name == 'weight' and isinstance(v, bn):  # Norm's weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # Conv's weight (with decay)

    optimizer = optim.SGD(g[2], lr=lr, momentum=momentum)
    optimizer.add_param_group({'params': g[0], 'weight_decay': weight_decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    return optimizer
    

def build_scheduler(optimizer, lr_decay=1e-2, num_epochs=300):
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, 
                                            lr_lambda=one_cycle(1, lr_decay, num_epochs))
    return scheduler


def resume_state(ckpt_path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    on_epoch = ckpt['on_epoch'] + 1
    
    model.load_state_dict(ckpt['model_state'], strict=True)
    optimizer.load_state_dict(ckpt['optimizer_state'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    scheduler.load_state_dict(ckpt['scheduler_state'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    return on_epoch