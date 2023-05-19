import math

import cv2
import torch
import numpy as np


class TrainTransform:
    """augmentation class for model training
    """
    def __init__(self, input_size, mean, std,
                 scale=(0.08, 1.0), ratio=(3/4, 4/3),
                 h_gain=0.4, s_gain=0.4, v_gain=0.4):
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        self.tfs = Compose([
            ##### Geometric Augment #####
            HorizontalFlip(),
            RandomResizedCrop(size=input_size, scale=scale, ratio=ratio),
            #### Photometric Augment ####
            AugmentHSV(h_gain=h_gain, s_gain=s_gain, v_gain=v_gain),
            ##### End-of-Augment #####
            Normalize(mean=mean, std=std),
            ToTensor()
        ])
    
    def __call__(self, image):
        return self.tfs(image)


class ValidTransform:
    """augmentation class for model evaluation
    """
    def __init__(self, input_size, mean, std):
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        self.tfs = Compose([
            Resize(size=256),
            CenterCrop(size=input_size),
            Normalize(mean=mean, std=std),
            ToTensor()
        ])
    
    def __call__(self, image):
        return self.tfs(image)


class Compose:
    """compositor for augmentation combination
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for tf in self.transforms:
            image = tf(image)
        return image


class AugmentHSV:
    
    def __init__(self, h_gain=0.5, s_gain=0.5, v_gain=0.5):
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain
    
    def __call__(self, image):
        r = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return image


class Resize:
    """resize the input image to the given size.
    if size is an int, smaller edge of t he image will be matched to this number,
    else size is an tuple or list, resize the image to size of (width, height)
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        h0, w0 = image.shape[:2]
        if isinstance(self.size, int):
            scale_h, scale_w = h0 / min(h0, w0), w0 / min(h0, w0)
            h1, w1 = int(scale_h * self.size), int(scale_w * self.size)
        else:
            h1, w1 = self.size
        return cv2.resize(image, (w1, h1))


class HorizontalFlip:
    """horizontally flip the given image randomly with a 0.5 probability.
    """
    def __call__(self, image):
        if np.random.randint(2):
            image = image[:, ::-1, :]
        return image


class CenterCrop:
    """crop the given image at the center.
    """
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image):
        h, w = image.shape[:2]
        crop_h, crop_w = self.size, self.size
        return image[h//2-crop_h//2:h//2+crop_h//2, w//2-crop_w//2:w//2+crop_w//2, :]


class RandomResizedCrop:
    """crop a random portion of image and resize it to a given size.
    """
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3/4, 4/3)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, image):
        h0, w0 = image.shape[:2]
        area = h0 * w0
        log_ratio = np.log1p(self.ratio)

        for _ in range(10):
            target_area = area * np.random.uniform(self.scale[0], self.scale[1])
            aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))
            h1 = int(round(math.sqrt(target_area * aspect_ratio)))
            w1 = int(round(math.sqrt(target_area * aspect_ratio)))
            
            if 0 < w1 <= w0 and 0 < h1 <= h0:
                i = np.random.randint(0, h0 - h1 + 1)
                j = np.random.randint(0, w0 - w1 + 1)
                return self.resized_crop(image, i, j, h1, w1)
        
        in_ratio = float(w0) / float(h0)
        if in_ratio < min(self.ratio):
            w1 = w0
            h1 = int(round(w1 / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h1 = h0
            w1 = int(round(h1 * max(self.ratio)))
        else:
            w1, h1 = w0, h0
        i, j = (h0 - h1) // 2, (w0 - w1) // 2
        return self.resized_crop(image, i, j, h1, w1)

    def resized_crop(self, image, top, left, height, width):
        return cv2.resize(image[top:top+height, left:left+width, :], (self.size, self.size))


class Normalize:
    """normalize a tensor image with mean and standard deviation.
    """
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image):
        if not isinstance(image.dtype, np.float32):
            image = image.astype(np.float32)
        image /= 255
        image -= self.mean
        image /= self.std
        return image


class ToTensor:
    
    def __call__(self, image):
        image = np.ascontiguousarray(image.transpose(2,0,1))
        return torch.from_numpy(image).float()
