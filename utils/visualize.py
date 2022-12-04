import cv2
import numpy as np


MEAN = 0.406, 0.456, 0.485 # BGR
STD = 0.225, 0.224, 0.229 # BGR
TEXT_COLOR = (10, 250, 10)


def visualize_dataset(img_loader, class_list, show_nums=5):
    batch = next(iter(img_loader))
    images, labels = batch[0], batch[1]

    check_images = []
    for i in range(len(images)):
        image = to_image(images[i]).copy()
        class_name = class_list[labels[i].item()]
        cv2.putText(image, text=class_name, org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=2, color=TEXT_COLOR)
        check_images.append(image)

        if len(check_images) >= show_nums:
            concat_result = np.concatenate(check_images, axis=1)
            return concat_result


def to_image(tensor, mean=MEAN, std=STD):
    denorm_tensor = tensor.clone()
    for t, m, s in zip(denorm_tensor, mean, std):
        t.mul_(s).add_(m)
    denorm_tensor.clamp_(min=0, max=1.)
    denorm_tensor *= 255
    image = denorm_tensor.permute(1,2,0).numpy().astype(np.uint8)
    return image
