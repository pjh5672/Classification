import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import yaml_load

if __package__:
    from .dataset import Dataset
    from .transform import TrainTransform, ValidTransform
else:
    from dataset import Dataset
    from transform import TrainTransform, ValidTransform
    

def build_dataset(opt, cache_images=False):
    """build method for classification dataset via calling predefined <data>.yaml
    """
    train_dataset = Dataset(opt=opt, phase='train', cache_images=cache_images)
    val_dataset = Dataset(opt=opt, phase='val', cache_images=cache_images)
    train_transformer = TrainTransform(input_size=opt.input_size, 
                                       mean=opt.hyp.MEAN_BGR, 
                                       std=opt.hyp.STD_BGR,
                                       h_gain=opt.hyp.H_GAIN, 
                                       s_gain=opt.hyp.S_GAIN, 
                                       v_gain=opt.hyp.V_GAIN)
    val_transformer = ValidTransform(input_size=opt.input_size, 
                                     mean=opt.hyp.MEAN_BGR, 
                                     std=opt.hyp.STD_BGR,)
    train_dataset.load_transformer(transformer=train_transformer)
    val_dataset.load_transformer(transformer=val_transformer)
    return train_dataset, val_dataset


if __name__ == "__main__":
    import sys
    import time
    from torch.utils.data import DataLoader
    
    dataset = 'dogs'
    data_path = ROOT / 'data' / f'{dataset}.data.yaml'
    hyp_path =  ROOT / 'data' / f'{dataset}.hyp.yaml'
    cache_images = False
    input_size = 224

    opt = argparse.Namespace()
    opt.data = argparse.Namespace(**yaml_load(data_path))
    opt.hyp = argparse.Namespace(**yaml_load(hyp_path))
    opt.input_size = input_size

    train_dataset, val_dataset = build_dataset(opt=opt, cache_images=cache_images)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=8)
    
    avg_time = 0.0
    max_count = 23
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
