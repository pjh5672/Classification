import math

import cv2
import torch
import numpy as np


def to_tensor(image):
    image = np.ascontiguousarray(image.transpose(2,0,1))
    return torch.from_numpy(image).float()


def to_image(tensor, mean, std):
    denorm_tensor = tensor.clone()
    for t, m, s in zip(denorm_tensor, mean, std):
        t.mul_(s).add_(m)
    denorm_tensor.clamp_(min=0, max=1.)
    denorm_tensor *= 255
    return denorm_tensor.permute(1,2,0).numpy().astype(np.uint8)


class TrainTransform:
    """augmentation class for model training
    """
    def __init__(self, input_size, mean, std):
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        self.tfs = Compose([
            ##### Geometric Augment #####
            HorizontalFlip(),
            RandomResizedCrop(size=input_size),
            #### Photometric Augment ####
            AddPCANoise(std=0.1),
            RandomBrightness(delta=102),
            AdjustHSV(h_delta=72, s_value=(0.6, 1.4)),
            ##### End-of-Augment #####
            Normalize(mean=mean, std=std)
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
            Normalize(mean=mean, std=std)
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
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)):
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
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image


class RandomBrightness:
    """adjust brightness of an image using uniform distribution.
    """
    def __init__(self, delta=102):
        assert delta >= 0 and delta <= 255
        self.delta = delta

    def __call__(self, image):
        if not isinstance(image.dtype, np.float32):
            image = image.astype(np.float32)
            
        if np.random.randint(2):
            image += np.random.uniform(-self.delta, self.delta)
        return np.clip(image, 0, 255)


class RandomContrast:
    """adjust contrast of an image using uniform distribution.
    """
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        if not isinstance(image.dtype, np.float32):
            image = image.astype(np.float32)
            
        if np.random.randint(2):
            image *= np.random.uniform(self.lower, self.upper)
        return np.clip(image, 0, 255)


class RandomHue:
    """adjust hue of an image using uniform distribution.
    """
    def __init__(self, delta=72):
        assert delta >= 0 and delta < 180
        self.delta = delta

    def __call__(self, image):
        if not isinstance(image.dtype, np.float32):
            image = image.astype(np.float32)
            
        image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
        image[:, :, 0] %= 180
        return image


class RandomSaturation:
    """adjust saturation of an image using uniform distribution.
    """
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        if not isinstance(image.dtype, np.float32):
            image = image.astype(np.float32)
        image[:, :, 1] *= np.random.uniform(self.lower, self.upper)
        return np.clip(image, 0, 255)


class AdjustHSV:
    """convert between BGR and HSV color and adjust hue and saturation of the given image. 
    """
    def __init__(self, h_delta=72, s_value=(0.5, 1.5)):
        self.random_hue = RandomHue(delta=h_delta)
        self.random_sat = RandomSaturation(lower=s_value[0], upper=s_value[1])
        
    def __call__(self, image):
        if np.random.randint(2):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image = self.random_sat(self.random_hue(image=image))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image


class AddPCANoise:
    """add noise to the given image using PCA components.
    """
    def __init__(self, std=0.1):
        self.std = std
    
    def __call__(self, image):
        if not isinstance(image.dtype, np.float32):
            image = image.astype(np.float32)

        image = image / image.max()  # rescale to 0 to 1 range
        img_rs = image.reshape(-1, 3) # flatten image to columns of RGB
        img_centered = img_rs - np.mean(img_rs, axis=0) # center mean
        img_cov = np.cov(img_centered, rowvar=False) # 3x3 covariance matrix
        try:
            eig_vals, eig_vecs = np.linalg.eigh(img_cov) # eigen values and eigen vectors
        except Exception as e:
            print(e)
            print((img_cov == 0).sum())
            print(np.isnan(img_cov))
            print(img_cov.max())
        # sort values and vector
        sort_perm = eig_vals[::-1].argsort() 
        eig_vals[::-1].sort()
        eig_vecs = eig_vecs[:, sort_perm]
        
        m1 = eig_vecs.T
        m2 = np.random.normal(0, self.std, size=3) * eig_vals
        image[..., :] += (m1 @ m2)
        return np.clip(image, 0, 1) * 255
