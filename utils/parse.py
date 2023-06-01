import argparse
from pathlib import Path
from dataclasses import asdict

import yaml

if __package__:
    from .config import DataParam, HyperParam
else:
    from config import DataParam, HyperParam


def yaml_load(file):
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def yaml_save(file, data={}, default_flow_style=False):
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, 
                        f, sort_keys=False, default_flow_style=default_flow_style)


class Parser:
    
    def __init__(self, data_dir: Path, dataset: str):
        self.dataset = dataset
        self.data_dir = data_dir
        data_yaml = self.data_dir / f'{self.dataset}.data.yaml'
        hyp_yaml =  self.data_dir / f'{self.dataset}.hyp.yaml'
        self.args = None
        self.data = DataParam(**yaml_load(data_yaml))
        self.hyp = HyperParam(**yaml_load(hyp_yaml))
        self.opt = argparse.Namespace()
    
    def build_opt(self):
        return self._merge(self.data, self.hyp)
    
    def _merge(self, *args):
        for ob in args:
            self.opt.__dict__.update(ob.__dict__)
        return self.opt

    def save_yaml(self, save_dir):
        if self.args:
            yaml_save(save_dir / f'{self.dataset}.args.yaml', vars(self.args))
        yaml_save(save_dir / f'{self.dataset}.data.yaml', asdict(self.data))
        yaml_save(save_dir / f'{self.dataset}.hyp.yaml', asdict(self.hyp), 
                  default_flow_style=None)
    
    def change_dataset(self, to):
        data_yaml = self.data_dir / f'{to}.data.yaml'
        hyp_yaml =  self.data_dir / f'{to}.hyp.yaml'
        self.data = DataParam(**yaml_load(data_yaml))
        self.hyp = HyperParam(**yaml_load(hyp_yaml))
        return self.build_opt()


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    
    dataset = 'mit67'
    data_dir = ROOT / 'data'
    parser = Parser(data_dir=data_dir, dataset=dataset)
    opt = parser.build_opt()
    print(opt.dataroot_dir, opt.std)
    
    opt = parser.reset_dataset(to='cub200')
    opt = parser.build_opt()
    print(opt.dataroot_dir, opt.std)
