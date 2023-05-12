import os
from pathlib import Path

import cv2
import yaml
import glob
import torch
import pprint
import numpy as np
from tqdm import tqdm

from transform import TrainTransform, ValidTransform, to_tensor

class Dataset:
    def __init__(self, yaml_path, phase):
        with open(yaml_path, mode='r') as f:
            self.hyp = yaml.load(f, Loader=yaml.FullLoader)
        self.phase = phase
        self.transformer = None
        self.input_size = self.hyp['INPUT_SIZE']
        
        self.image_dirs, self.image_paths = [], []
        data_dir = Path(self.hyp['PATH']) / self.hyp[self.phase.upper()]
        for fpath in sorted(glob.glob(str(data_dir / '*/*'), recursive=True)):
            if fpath.lower().endswith(('png', 'jpg', 'jpeg')):
                image_dir = fpath.split(os.sep)[-2]
                if image_dir not in self.image_dirs:
                    self.image_dirs.append(image_dir)
                self.image_paths.append(fpath)
                
        self.num_classes = len(self.image_dirs)
        self.idx2cls = self.hyp['CLASS_INFO']
        if self.idx2cls is None:
            self.idx2cls = self.get_class_list(image_dirs=self.image_dirs)
            raise RuntimeError("Put the shown class list to 'CLASS_INFO' in *.yaml")

        self.mean, self.std = self.hyp['MEAN'], self.hyp['STD']
        if (None in self.mean) or (None in self.std):
            self.mean, self.std = self.set_mean_std()
        
    def get_class_list(self, image_dirs):
        idx2cls = dict(enumerate(image_dirs))
        pprint.pprint(idx2cls)
        return idx2cls
        
    def __len__(self): return len(self.image_paths)

    def __getitem__(self, index):
        fpath = self.image_paths[index]
        image = self.get_image(fpath)
        label = self.get_label(fpath)
        image = to_tensor(self.transformer(image))
        return image, label
    
    def get_image(self, filepath):
        return cv2.imread(filepath)
        
    def get_label(self, filepath):
        class_name = filepath.split(os.sep)[-2]
        return self.image_dirs.index(class_name)
    
    def load_transformer(self, transformer):
        self.transformer = transformer
    
    def set_mean_std(self):
        mean_bgr, std_bgr = [], []
        for index in tqdm(range(len(self))):
            x = self.get_image(self.image_paths[index])
            mean_bgr.append(np.mean(x / 255, axis=(1,2)))
            std_bgr.append(np.std(x / 255, axis=(1,2)))

        mean_b = np.mean([m[0] for m in mean_bgr])
        mean_g = np.mean([m[1] for m in mean_bgr])
        mean_r = np.mean([m[2] for m in mean_bgr])
        std_b = np.mean([s[0] for s in std_bgr])
        std_g = np.mean([s[1] for s in std_bgr])
        std_r = np.mean([s[2] for s in std_bgr])
        
        return [mean_b, mean_g, mean_r], [std_b, std_g, std_r]

    @staticmethod
    def collate_fn(batch):
        images, labels = [], []
        for item in batch:
            images.append(item[0])
            labels.append(item[1])
        return torch.stack(images, dim=0), torch.tensor(labels)


if __name__ == "__main__":
    import sys
    import cv2
    from pathlib import Path
    from torch.utils.data import DataLoader
    
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from utils import visualize_dataset
    
    yaml_path = ROOT / 'data' / 'image.yaml'
    train_dataset = Dataset(yaml_path=yaml_path, phase='train')
    val_dataset = Dataset(yaml_path=yaml_path, phase='val')
    val_dataset.mean, val_dataset.std = train_dataset.mean, train_dataset.std
    train_transformer = TrainTransform(input_size=train_dataset.input_size, 
                                        mean=train_dataset.mean, std=train_dataset.std)
    val_transformer = ValidTransform(input_size=train_dataset.input_size, 
                                    mean=train_dataset.mean, std=train_dataset.std)
    train_dataset.load_transformer(transformer=train_transformer)
    val_dataset.load_transformer(transformer=val_transformer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, 
                                collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            collate_fn=train_dataset.collate_fn)

    for idx, batch in enumerate(train_loader):
        images, labels = batch
        print(idx, '-', images.shape)
        print(labels)
        print(torch.min(images), torch.max(images))
        if idx == 0:
            break

    vis_image = visualize_dataset(img_loader=train_loader, 
                                    class_list=train_dataset.idx2cls, 
                                    mean=train_dataset.mean, std=train_dataset.std, 
                                    font_scale=0.8, thickness=1, show_nums=6)
    cv2.imwrite( str(ROOT / 'asset' / yaml_path.with_suffix('.jpg').name ), vis_image)