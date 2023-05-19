import os
import random
from pathlib import Path
from multiprocessing.pool import ThreadPool

import cv2
import glob
import torch
import pprint
import psutil
import numpy as np
from tqdm import tqdm


NUM_THREADS = min(8, max(1, os.cpu_count() - 1))


class Dataset:
    """classification dataset supporting MIT67, CUB200, Stanford Dogs, ImageNet dataset
    """
    def __init__(self, opt, phase, cache_images=False):
        self.data = opt.data
        self.hyp = opt.hyp
        self.transformer = None
        self.input_size = opt.input_size
        self.image_dirs, self.image_paths = [], []

        if phase.lower() == 'train':
            data_dir = Path(self.data.PATH) / self.data.TRAIN
        else:
            data_dir = Path(self.data.PATH) / self.data.VAL
        
        for fpath in sorted(glob.glob(str(data_dir / '*/*'), recursive=True)):
            if fpath.lower().endswith(('png', 'jpg', 'jpeg')):
                image_dir = fpath.split(os.sep)[-2]
                if image_dir not in self.image_dirs:
                    self.image_dirs.append(image_dir)
                self.image_paths.append(fpath)

        self.num_classes = len(self.image_dirs)
        self.idx2cls = self.data.CLASS_INFO
        if self.idx2cls is None:
            self.idx2cls = self.get_class_list(image_dirs=self.image_dirs)
            raise RuntimeError("Put the shown class list to 'CLASS_INFO' in *.yaml")

        self.images = [None] * len(self)
        if cache_images and self.check_cache_ram():
            self.cache_images()
        
        if (None in self.hyp.MEAN_BGR) or (None in self.hyp.STD_BGR):
            self.set_mean_std()
            
    def get_class_list(self, image_dirs):
        idx2cls = dict(enumerate(image_dirs))
        pprint.pprint(idx2cls)
        return idx2cls
        
    def __len__(self): return len(self.image_paths)

    def __getitem__(self, index):
        image = self.get_image(index)
        image = self.transformer(image)
        label = self.get_label(index)
        return image, label
    
    def get_image(self, index):
        image = self.images[index]
        if image is None:
            image = cv2.imread(self.image_paths[index])
        return image
        
    def get_label(self, index):
        filepath = self.image_paths[index]
        class_name = filepath.split(os.sep)[-2]
        return self.image_dirs.index(class_name)
    
    def load_transformer(self, transformer):
        self.transformer = transformer
    
    def check_cache_ram(self, safety_margin=0.1):
        """ check image caching requirements vs available memory.
        """
        b, gb = 0, 1 << 30
        n = min(len(self), 30)
        for _ in range(n):
            im = cv2.imread(random.choice(self.image_paths))
            ratio = self.input_size / max(im.shape[0], im.shape[1])
            b += im.nbytes * ratio ** 2
        mem_required = b * len(self) / n  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available
        if not cache:
            raise RuntimeError(f"{mem_required / gb:.1f}GB RAM required, "
                                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, "
                                f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
        return cache
    
    def cache_images(self):
        b, gb = 0, 1 << 30
        results = ThreadPool(NUM_THREADS).imap(self.get_image, range(len(self)))
        pbar = tqdm(enumerate(results), total=len(self), ncols=110)
        for i, x in pbar:
            self.images[i] = x
            b += x.nbytes
            pbar.desc = f"Caching images in RAM ({b / gb:.1f}GB)"
        pbar.close()

    def set_mean_std(self):
        means, stds = [], []
        results = ThreadPool(NUM_THREADS).imap(self.get_image, range(len(self)))
        pbar = tqdm(enumerate(results), desc="Computing mean, std", 
                    total=len(self), ncols=110)
        for _, x in pbar:
            means.append(np.mean(x / 255, axis=(0,1)))
            stds.append(np.std(x / 255, axis=(0,1)))
        pbar.close()

        for ch in range(3):
            self.hyp.MEAN_BGR[ch] = round(float(np.mean([m[ch] for m in means])), 3)
            self.hyp.STD_BGR[ch] = round(float(np.mean([s[ch] for s in stds])), 3)

    @staticmethod
    def collate_fn(batch):
        images, labels = [], []
        for item in batch:
            images.append(item[0])
            labels.append(item[1])
        return torch.stack(images, dim=0), torch.tensor(labels)


if __name__ == "__main__":
    import sys
    import argparse
    import cv2
    from pathlib import Path
    from torch.utils.data import DataLoader
    
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
        
    from transform import TrainTransform, ValidTransform
    from utils import visualize_dataset, yaml_load, yaml_save

    dataset = 'dogs'
    data_path = ROOT / 'data' / f'{dataset}.data.yaml'
    hyp_path =  ROOT / 'data' / f'{dataset}.hyp.yaml'
    cache_images = False
    input_size = 224

    opt = argparse.Namespace()
    opt.data = argparse.Namespace(**yaml_load(data_path))
    opt.hyp = argparse.Namespace(**yaml_load(hyp_path))
    opt.input_size = input_size

    train_dataset = Dataset(opt=opt, phase='train', cache_images=cache_images)
    val_dataset = Dataset(opt=opt, phase='val', cache_images=cache_images)
    train_transformer = TrainTransform(input_size=input_size, 
                                       mean=opt.hyp.MEAN_BGR, std=opt.hyp.STD_BGR)
    val_transformer = ValidTransform(input_size=input_size, 
                                     mean=opt.hyp.MEAN_BGR, std=opt.hyp.STD_BGR)
    train_dataset.load_transformer(transformer=train_transformer)
    val_dataset.load_transformer(transformer=val_transformer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, 
                              collate_fn=train_dataset.collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, 
                            collate_fn=train_dataset.collate_fn)
    
    for idx, batch in enumerate(train_loader):
        images, labels = batch
        print(idx, '-', images.shape)
        print(labels)
        print(torch.min(images), torch.max(images))
        if idx == 0:
            break

    vis_image = visualize_dataset(img_loader=val_loader, class_list=opt.data.CLASS_INFO, 
                                  mean=opt.hyp.MEAN_BGR, std=opt.hyp.STD_BGR, 
                                  font_scale=0.8, thickness=2, show_nums=6)
    sample_fname = data_path.with_suffix('.jpg').name.replace('.data', '')
    cv2.imwrite(str(ROOT / 'asset' / sample_fname), vis_image)

    # yaml_save(data_path, vars(opt.data))
    yaml_save(hyp_path, vars(opt.hyp), default_flow_style=None)