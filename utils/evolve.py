import random

import yaml
import numpy as np
import pandas as pd

from utils.general import LOGGER, colorstr


class Evolution:
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.evolve_csv = save_dir / 'evolve.csv'
        self.evolve_yaml = save_dir / 'hyp_evolve.yaml'
        
        self.mp, self.s = 0.8, 0.2  # mutation probability, sigma
        self.params = {
            'base_lr': (1, 1e-4, 1.0),
            'lr_decay': (1, 1e-3, 1.0),
            'momentum': (0.3, 0.5, 0.98),
            'weight_decay': (1, 0, 1e-3),
            'hsv_h': (1, 0.0, 0.9),
            'hsv_s': (1, 0.0, 0.9),
            'hsv_v': (1, 0.0, 0.9),
        }
        
    def run(self, hyp):
        if self.evolve_csv.exists():
            x = np.loadtxt(self.evolve_csv, ndmin=2, delimiter=',', skiprows=1)
            n = min(5, len(x))
            x = x[np.argsort(-self.compute_fitness(x))][:n]
            w = self.compute_fitness(x) - self.compute_fitness(x).min() + 1E-6
            x = x[random.choices(range(n), weights=w)[0]]
            g = np.array([self.params[k][0] for k in hyp.keys()])  # gains 0-1
            ng = len(self.params)
            v = np.ones(ng)

            while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                v = (g * (np.random.random(ng) < self.mp) * \
                    np.random.randn(ng) * np.random.random() * self.s + 1).clip(0.3, 3.0)
            
            for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                hyp[k] = float(x[i] * v[i])  # mutate

        # Constrain to limits
        for k, v in self.params.items():
            hyp[k] = max(hyp[k], v[1])  # lower limit
            hyp[k] = min(hyp[k], v[2])  # upper limit
            hyp[k] = round(hyp[k], 5)  # significant digits

    def compute_fitness(self, x):
        # Model fitness as a weighted combination of metrics
        w = [1.0, 0.0, 0.0]  # weights for [Top1-Acc, Top5-Acc, ValLoss]
        return (x[:, -len(w):] * w).sum(1)

    def write_results(self, hyp, keys, results):
        keys = tuple(hyp.keys()) + tuple(keys)
        keys = tuple(x.strip() for x in keys)
        vals = tuple(hyp.values()) + results
        n = len(keys)

        # save evolve.csv
        s = '' if self.evolve_csv.exists() else (('%16s,' * n % keys).rstrip(',') + '\n')
        with open(self.evolve_csv, 'a') as f:
            f.write(s + ('%16.5g,' * n % vals).rstrip(',') + '\n')

        # save evolve.yaml
        with open(self.evolve_yaml, 'w') as f:
            data = pd.read_csv(self.evolve_csv, skipinitialspace=True)
            data = data.rename(columns=lambda x: x.strip())
            i = np.argmax(self.compute_fitness(data.values[:, -len(keys):]))
            generations = len(data)

            f.write('# Hyperparameter Evolution Results\n' + f'# Best generation: {i}\n' +
                    f'# Last generation: {generations - 1}\n' + '# ' + 
                    ', '.join(f'{x.strip():>16s}' for x in keys[-len(keys):]) + '\n' + '# ' + 
                    ', '.join(f'{x:>16.5g}' for x in data.values[i, -len(keys):]) + '\n\n')
            yaml.safe_dump(data.loc[i][-len(keys):].to_dict(), f, sort_keys=False)

        LOGGER.info(f'\n{colorstr("bright_yellow", "bold", "Evolution")} 🌟 ' + \
                    f'{generations} generations finished, current result:\n' + 
                    ', '.join(f'{x.strip():>16s}' for x in keys) + '\n' + 
                    ', '.join(f'{x:16.5g}' for x in vals) + '\n')
