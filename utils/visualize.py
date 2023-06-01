import cv2
import numpy as np


def visualize_dataset(img_loader, 
                      class_list, 
                      mean, std, 
                      show_nums=6, 
                      font_scale=0.8, 
                      thickness=2, 
                      text_color=(10, 250, 10)):
    
    batch = next(iter(img_loader))
    images, labels = batch[0], batch[1]
    check_images = []
    for i in range(len(images)):
        image = to_image(images[i], mean, std).copy()
        class_name = class_list[labels[i].item()]
        cv2.putText(image, text=class_name, org=(10, 20), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, 
                    thickness=thickness, color=text_color)
        check_images.append(image)

        if len(check_images) >= show_nums:
            concat_result = np.concatenate(check_images, axis=1)
            return concat_result[..., ::-1]


def to_image(tensor, mean, std):
    denorm_tensor = tensor.clone()
    for t, m, s in zip(denorm_tensor, mean, std):
        t.mul_(s).add_(m)
    denorm_tensor.clamp_(min=0, max=1.)
    denorm_tensor *= 255
    return denorm_tensor.permute(1,2,0).numpy().astype(np.uint8)