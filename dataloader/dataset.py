import os
import sys
from pathlib import Path
from multiprocessing.pool import ThreadPool

import glob
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.general import NUM_THREADS


class Dataset:
    """classification dataset
    """
    def __init__(self, opt, phase):
        self.opt = opt
        self.transformer = None
        self.image_dirs, self.image_paths = [], []

        if phase.lower() == 'train':
            data_dir = Path(self.opt.dataroot_dir) / self.opt.train_dir
        else:
            data_dir = Path(self.opt.dataroot_dir) / self.opt.val_dir
        
        for fpath in sorted(glob.glob(str(data_dir / '*/*'), recursive=True)):
            if fpath.lower().endswith(('png', 'jpg', 'jpeg')):
                image_dir = fpath.split(os.sep)[-2]
                if image_dir not in self.image_dirs:
                    self.image_dirs.append(image_dir)
                self.image_paths.append(fpath)

        self.num_classes = len(self.image_dirs)
        if self.opt.class_list is None:
            self.opt.class_list = self.get_class_list(image_dirs=self.image_dirs)

        if self.opt.mean is None or self.opt.std is None:
            self.set_mean_std()
            
    def get_class_list(self, image_dirs): return dict(enumerate(image_dirs))
        
    def __len__(self): return len(self.image_paths)

    def __getitem__(self, index):
        image = self.get_image(index)
        image = self.transformer(image)
        label = self.get_label(index)
        return image, label
    
    def get_image(self, index):
        return np.array(Image.open(self.image_paths[index]).convert('RGB'))
        
    def get_label(self, index):
        filepath = self.image_paths[index]
        class_name = filepath.split(os.sep)[-2]
        return self.image_dirs.index(class_name)

    def load_transformer(self, transformer):
        self.transformer = transformer
    
    def set_mean_std(self):
        self.opt.mean, self.opt.std = [None] * 3, [None] * 3
        means, stds = [], []
        results = ThreadPool(NUM_THREADS).imap(self.get_image, range(len(self)))
        pbar = tqdm(enumerate(results), desc="Computing mean, std", total=len(self), ncols=110)
        for _, x in pbar:
            means.append(np.mean(x / 255, axis=(0, 1)))
            stds.append(np.std(x / 255, axis=(0, 1)))
        pbar.close()

        means = np.array(means)
        stds = np.array(stds)
        for ch in range(3):
            self.opt.mean[ch] = round(float(means[:, ch].mean()), 3)
            self.opt.std[ch] = round(float(stds[:, ch].mean()), 3)

    @staticmethod
    def collate_fn(batch):
        images, labels = [], []
        for item in batch:
            images.append(item[0])
            labels.append(item[1])
        return torch.stack(images, dim=0), torch.tensor(labels)


if __name__ == "__main__":
    import cv2
    from torch.utils.data import DataLoader
    from transform import TrainTransform, ValidTransform
    from utils.parse import Parser
    from utils.visualize import visualize_dataset

    dataset = 'mit67'
    data_dir = ROOT / 'data'
    
    opt = Parser(data_dir=data_dir, dataset=dataset).build_opt()
    opt.img_size = 224
    
    train_dataset = Dataset(opt=opt, phase='train')
    val_dataset = Dataset(opt=opt, phase='val')
    train_transformer = TrainTransform(train_size=opt.img_size, mean=opt.mean, std=opt.std)
    val_transformer = ValidTransform(val_size=opt.img_size, mean=opt.mean, std=opt.std)
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

    vis_image = visualize_dataset(img_loader=val_loader, 
                                  class_list=opt.class_list, 
                                  mean=opt.mean, std=opt.std)
    cv2.imwrite(str(ROOT / 'asset' / f'{dataset}.jpg' ), vis_image)
