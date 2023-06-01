import argparse
from pathlib import Path
from collections import OrderedDict

import torch

from utils.parse import Parser

ROOT = Path(__file__).resolve().parents[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, required=True, help='Name to log training')
    
    args = parser.parse_args()
    args.project_dir = ROOT / 'experiment' / args.project
    args.weight_dir = args.project_dir / 'weight'
    args.ckpt_path = args.weight_dir / 'best.pt'
    return args

    
if __name__ == '__main__':
    args = parse_args()

    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    trained_model_state = ckpt['model_state']
    parsed_model_state = OrderedDict()

    for key, val in trained_model_state.items():
        if not key.startswith('fc'):
            parsed_model_state[key] = val
    
    torch.save({'model_state': parsed_model_state}, f'./{args.project}.pt')