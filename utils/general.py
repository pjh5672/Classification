import math
from pathlib import Path

import yaml
from torch import nn
from torch import optim


def yaml_load(file):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def yaml_save(file, data={}, default_flow_style=False):
    # Single-line safe yaml saving
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, 
                       f, sort_keys=False, default_flow_style=default_flow_style)


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


def build_optimizer(model, name='sgd', lr=0.001, momentum=0.9, weight_decay=1e-5):
    assert name in ['adam', 'sgd'], f"not support optimizer, got {name}."
    
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

    if name == 'adam':
        optimizer = optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    
    optimizer.add_param_group({'params': g[0], 'weight_decay': weight_decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    return optimizer
    

def build_scheduler(optimizer, name='imagenet', num_epoch=100, lr_decay=1e-4):
    assert name in ['mit67', 'cub200', 'dogs', 'imagenet'], f"not support scheduler, got {name}."
    
    if name in ['mit67', 'cub200', 'dogs']:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    elif name in ['imagenet']:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, 
                                                lr_lambda=one_cycle(1, lr_decay, num_epoch))
    return scheduler