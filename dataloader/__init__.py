if __package__:
    from .dataset import Dataset
    from .transform import TrainTransform, ValidTransform
else:
    from dataset import Dataset
    from transform import TrainTransform, ValidTransform


def build_dataset(opt):
    """build method for classification dataset via calling predefined <data>.yaml
    """
    train_dataset = Dataset(opt=opt, phase='train')
    val_dataset = Dataset(opt=opt, phase='val')
    train_transformer = TrainTransform(train_size=opt.train_size, mean=opt.mean, std=opt.std,
                                       hsv_h=opt.hsv_h, hsv_s=opt.hsv_s, hsv_v=opt.hsv_v)
    val_transformer = ValidTransform(val_size=opt.val_size, mean=opt.mean, std=opt.std)
    train_dataset.load_transformer(transformer=train_transformer)
    val_dataset.load_transformer(transformer=val_transformer)
    return train_dataset, val_dataset


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    
    from torch.utils.data import DataLoader
    
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    
    from utils.parse import Parser
    
    dataset = 'dogs'
    data_dir = ROOT / 'data'

    opt = Parser(data_dir=data_dir, dataset=dataset).build_opt()
    opt.train_size = opt.val_size = 224
    train_dataset, val_dataset = build_dataset(opt=opt)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=8)
    
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
    print(f'avg time : {avg_time/max_count:.3f} ms')
