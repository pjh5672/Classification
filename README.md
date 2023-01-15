# <div align="center">Image Classification</div>

---

## [Content]
1. [Description](#description)   
2. [Usage](#usage)  
2-1. [Model Training](#model-training)  
2-2. [Detection Evaluation](#detection-evaluation)  
3. [Contact](#contact)   

---

## [Description]

This is a repository for PyTorch training implementation of general purposed classifier. With the training code, various feature extractor backbones such as Resnet, Darknet, CSP-Darknet53, Mobilenet series can be created.  


 - **Performance Table**

| Model | Dataset | Train | Valid | Size<br><sup>(pixel) | Accuracy<br><sup>(Top-1) | Params<br><sup>(M) | FLOPs<br><sup>(B) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Darknet19 | ImageNet | train2012 | val2012 | 224 | 73.50 | 20.84 | 5.62 |
| Darknet19 | ImageNet | train2012 | val2012 | 224 | 76.22 | 20.84 | 22.47 |
| Darknet53 | ImageNet | train2012 | val2012 | 224 | 77.67 | 41.61 | 14.29 |
| Darknet53 | ImageNet | train2012 | val2012 | 448 | 79.77 | 41.61 | 57.17 |
| Darknet53-tiny | ImageNet | train2012 | val2012 | 224 | 52.75 | 2.09 | 0.64 |
| CSP-Darknet53 | ImageNet | train2012 | val2012 | 224 | - | - | - |
| Mobilenet-v1 | ImageNet | train2012 | val2012 | 224 | - | - | - |
| Mobilenet-v2 | ImageNet | train2012 | val2012 | 224 | - | - | - |
| Mobilenet-v3-small | ImageNet | train2012 | val2012 | 224 | - | - | - |
| Mobilenet-v3-large | ImageNet | train2012 | val2012 | 224 | - | - | - |


 - **Pretrained Model Weights Download**

    - [DarkNet19-224](https://drive.google.com/file/d/1qVZjStNyHkTjQQSsyHkg9XgfnyxtFQau/view?usp=share_link)
    - [DarkNet19-448](https://drive.google.com/file/d/1qjOfb-KESJFZoJRlwZ_2FbT4-9PkgdjZ/view?usp=share_link)
    - [Darknet53-224](https://drive.google.com/file/d/1rLGltJki7OujkehJZNgnhj1Bp3JficIE/view?usp=share_link)
    - [Darknet53-448](https://drive.google.com/file/d/1pYc_UlnG2G4EnVwidR7T5XAu16KZj6YP/view?usp=share_link)
    - [Darknet53-tiny-224](https://drive.google.com/file/d/1fcSmzXvK7Zpr8OJnfN7RryysIibyhIsG/view?usp=share_link)


![result](./asset/data.jpg)


## [Usage]


#### Model Training 

 - You can train various classifier architectures (ResNet18, ResNet34, ResNet50, ResNet101, DarkNet19, DarkNet53, and CSP-DarkNet53, Mobilenetv1-v3). In addition, you can finetune classifier adding "--pretrained" in training command after putting the weight files into "./weights" directory with {model name}.  
 - In case of fine-tuning with 448px image, we finetune pretrained 224px model for 15~20 epochs without warming up learning rate. That is the comment like shown below.
    `python3 train.py --exp darknet19-448 --model darknet19 --batch-size 128 --base-lr 0.001 --warmup 0 --img-size 448 --num-epochs 15 --pretrained`


```python
python train.py --exp my_test --data imagenet.yaml --model resnet18
                                                   --model resnet34
                                                   --model resnet50
                                                   --model resnet101
                                                   --model darknet19
                                                   --model darknet53
                                                   --model darknet53-tiny
                                                   --model csp-darknet53 --width_multiple 1.0 --depth_multiple 1.0
                                                   --model mobilenet-v1 --width_multiple 1.0
                                                   --model mobilenet-v2 --width_multiple 1.0
                                                   --model mobilenet-v3 --mode {large, small} --width_multiple 1.0
```


#### Classification Evaluation

 - It computes Top-1 accuracy. Top-1 accuracy is the conventional accuracy, the model answer (the one with highest probability) must be exactly the expected answer. 

```python
python val.py --exp my_test --data voc.yaml --ckpt-name best.pt
```


---
## [Contact]
- Author: Jiho Park  
- Email: pjh5672.dev@gmail.com  