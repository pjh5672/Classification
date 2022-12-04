import os
import yaml
from pathlib import Path

import cv2
import torch
from torchvision import transforms


MEAN = 0.406, 0.456, 0.485 # BGR
STD = 0.225, 0.224, 0.229 # BGR



class Dataset:
    def __init__(self, yaml_path, phase):
        with open(yaml_path, mode="r") as f:
            data_item = yaml.load(f, Loader=yaml.FullLoader)
        self.phase = phase
        self.class_list = data_item["CLASS_INFO"]

        self.image_paths = []
        self.sub_dirs = []
        root_dir = Path(data_item["PATH"]) / data_item[self.phase.upper()]
        for sub_dir in os.listdir(root_dir):
            self.sub_dirs.append(sub_dir)
            image_dir = root_dir / sub_dir
            self.image_paths += [image_dir / fn for fn in os.listdir(root_dir / sub_dir) if fn.lower().endswith(("png", "jpg", "jpeg"))]


    def __len__(self): return len(self.image_paths)


    def __getitem__(self, index):
        image = cv2.imread(str(self.image_paths[index]))
        image = self.transformer(image)
        label = self.sub_dirs.index(self.image_paths[index].parent.name)
        label = torch.Tensor([label]).long()
        return image, label


    def load_transformer(self, transformer):
        self.transformer = transformer



def build_transformer(input_size, mean=MEAN, std=STD):
    transformer = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((int(input_size * 1.1), int(input_size * 1.1))),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((int(input_size * 1.1), int(input_size * 1.1))),
            transforms.CenterCrop(input_size),
            transforms.Normalize(mean=mean, std=std)
        ])
    }
    return transformer


if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from utils import visualize_dataset

    yaml_path = ROOT / 'data' / 'imagenet.yaml'
    input_size = 224
    
    transformer = build_transformer(input_size=input_size)
    train_dataset = Dataset(yaml_path=yaml_path, phase="train")
    train_dataset.load_transformer(transformer=transformer["train"])
    val_dataset = Dataset(yaml_path=yaml_path, phase="val")
    val_dataset.load_transformer(transformer=transformer["val"])
    val_dataloader = DataLoader(train_dataset, batch_size=8, num_workers=0, shuffle=False)

    print(len(train_dataset), len(val_dataset))
    for index, minibatch in enumerate(val_dataloader):
        images, labels = minibatch[0], minibatch[1].squeeze(dim=1)
        print(images.shape, labels)
        if index == 0:
            break

    vis_image = visualize_dataset(img_loader=val_dataloader, class_list=val_dataset.class_list, show_nums=6)
    cv2.imwrite(f'./asset/data.jpg', vis_image)
