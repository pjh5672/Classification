import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[0])
if ROOT not in sys.path:
    sys.path.append(ROOT)

from torch.utils.data import DataLoader

if __package__:
    from .dataset import Dataset
    from .transform import TrainTransform, ValidTransform
else:
    from dataset import Dataset
    from transform import TrainTransform, ValidTransform


def build_dataset(yaml_path, input_size=224):
    """build method for classification dataset via calling predefined <data>.yaml

    Args:
        yaml_path (pathlib.Path): path to <data>.yaml file

    Returns:
        dataset: defined dataset for loading each data
        params: parameters related to building dataset (eg. input size, class info, etc.)
    """
    train_dataset = Dataset(yaml_path=yaml_path, phase='train')
    val_dataset = Dataset(yaml_path=yaml_path, phase='val')
    val_dataset.mean, val_dataset.std = train_dataset.mean, train_dataset.std
    
    train_transformer = TrainTransform(input_size=input_size, mean=train_dataset.mean, std=train_dataset.std)
    val_transformer = ValidTransform(input_size=input_size, mean=train_dataset.mean, std=train_dataset.std)
    train_dataset.load_transformer(transformer=train_transformer)
    val_dataset.load_transformer(transformer=val_transformer)
    return train_dataset, val_dataset, train_dataset.idx2cls


if __name__ == "__main__":
    import sys
    import time
    
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
        
    yaml_path = ROOT / 'data' / 'imagenet.yaml'
    input_size = 224
    train_dataset, val_dataset, idx2cls = build_dataset(yaml_path=yaml_path, input_size=input_size)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=8)
    
    avg_time = 0.0
    max_count = 30
    test_loader = iter(train_loader)
    for idx in range(max_count):
        tic = time.time()
        _ = next(test_loader)
        toc = time.time()
        elapsed_time = (toc - tic) * 1000
        avg_time += elapsed_time
        if idx == max_count:
            break
    print(f"avg time : {avg_time/max_count:.3f} ms")
