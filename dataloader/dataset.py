import os
import yaml

from torchvision.datasets import ImageFolder
from torchvision import transforms


MEAN = 0.485, 0.456, 0.406 # RGB
STD = 0.229, 0.224, 0.225 # RGB


def build_dataset(yaml_path, input_size, mean=MEAN, std=STD):
    with open(yaml_path, mode="r") as f:
        data_item = yaml.load(f, Loader=yaml.FullLoader)

    class_list = data_item["CLASS_INFO"]
    train_root = os.path.join(data_item["PATH"], data_item["TRAIN"])
    val_root = os.path.join(data_item["PATH"], data_item["VAL"])

    dataset = {
        "train": ImageFolder(
            root=train_root, 
            transform=transforms.Compose([
                transforms.RandomResizedCrop(size=input_size, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=(0.6, 1.4), saturation=(0.6, 1.4), hue=(-0.4, 0.4)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        ),
        "val":  ImageFolder(
            root=val_root, 
            transform=transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        ),
    }
    return dataset, class_list


if __name__ == "__main__":
    import sys
    import cv2
    from pathlib import Path
    from torch.utils.data import DataLoader

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from utils import visualize_dataset

    yaml_path = ROOT / 'data' / 'imagenet.yaml'
    input_size = 224
    
    dataset, class_list = build_dataset(yaml_path=yaml_path, input_size=input_size)
    train_dataloader = DataLoader(dataset["train"], batch_size=8, num_workers=0, shuffle=True)
    val_dataloader = DataLoader(dataset["val"], batch_size=8, num_workers=0, shuffle=False)

    print(len(dataset["train"]), len(dataset["val"]))
    for index, minibatch in enumerate(train_dataloader):
        images, labels = minibatch[0], minibatch[1]
        print(images.shape, labels)
        if index == 0:
            break

    vis_image = visualize_dataset(img_loader=train_dataloader, class_list=class_list, show_nums=6)
    cv2.imwrite(f'./asset/data.jpg', vis_image)
