from typing import List, Dict
from dataclasses import dataclass


@dataclass
class DataParam:
    dataroot_dir : str
    train_dir: str
    val_dir: str
    class_list: Dict[int, str]


@dataclass
class HyperParam:
    mean : List[float]
    std: List[float]
    hsv_h: float
    hsv_s: float
    hsv_v: float   