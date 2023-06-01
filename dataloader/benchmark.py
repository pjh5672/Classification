import os
import yaml

from torchvision.datasets import ImageFolder
from torchvision import transforms


MEAN = 0.485, 0.456, 0.406 # RGB
STD = 0.229, 0.224, 0.225 # RGB


def build_dataset(yaml_path, input_size, mean=MEAN, std=STD):
    with open(yaml_path, mode='r') as f:
        data_item = yaml.load(f, Loader=yaml.FullLoader)

    class_list = data_item["class_list"]
    train_root = os.path.join(data_item['dataroot_dir'], data_item['train_dir'])
    val_root = os.path.join(data_item['dataroot_dir'], data_item['val_dir'])

    dataset = {
        'train': ImageFolder(
            root=train_root, 
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(size=input_size, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
                transforms.ColorJitter(brightness=(0.6, 1.4), saturation=(0.6, 1.4), hue=(-0.4, 0.4)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        ),
        'val':  ImageFolder(
            root=val_root, 
            transform=transforms.Compose([
                transforms.Resize(size=int(input_size*1.05)),
                transforms.CenterCrop(size=input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        ),
    }
    return dataset, class_list


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from torch.utils.data import DataLoader

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    yaml_path = ROOT / 'data' / 'dogs.data.yaml'
    input_size = 224
    
    dataset, class_list = build_dataset(yaml_path=yaml_path, input_size=input_size)
    train_dataloader = DataLoader(dataset['train'], batch_size=512, num_workers=8, shuffle=False)

    print(len(dataset['train']), len(dataset['val']))
    for index, minibatch in enumerate(train_dataloader):
        images, labels = minibatch[0], minibatch[1]
        print(images.min(), images.max())
        print(images.shape)
        if index == 0:
            break

    import time
    avg_time = 0.0
    max_count = 23
    test_loader = iter(train_dataloader)
    for idx in range(max_count):
        tic = time.time()
        _ = next(test_loader)
        toc = time.time()
        elapsed_time = (toc - tic) * 1000
        avg_time += elapsed_time
        if idx == max_count:
            break
    print(f'avg time : {avg_time/max_count:.3f} ms')