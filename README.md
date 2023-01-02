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

This is a repository for PyTorch training implementation of general purposed classifier. With the training code, various feature extractor backbones such as resnets, darknet19, darknet53, mobilenets can be created.  


 - **Performance Table**

| Model | Dataset | Train | Valid | Size<br><sup>(pixel) | Accuracy<br><sup>(Top-1) | Params<br><sup>(M) | FLOPs<br><sup>(B) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Darknet19 | ImageNet | train2012 | val2012 | 256 | 74.08 | 20.84 | 7.34 |
| Darknet53 | ImageNet | train2012 | val2012 | 256 | 78.21 | 41.61 | 18.67 |
| Darknet53-tiny | ImageNet | train2012 | val2012 | 256 | 52.01 | 2.09 | 0.83 |
| CSP-Darknet53 | ImageNet | train2012 | val2012 | 256 | 76.82 | 21.74 | 8.77 |
| Mobilenet-v1 | ImageNet | train2012 | val2012 | 256 | - | 4.23 | 1.18 |
| Mobilenet-v2 | ImageNet | train2012 | val2012 | 256 | - | 3.50 | 0.66 |
| Mobilenet-v3-small | ImageNet | train2012 | val2012 | 256 | - | 2.54 | 0.12 |
| Mobilenet-v3-large | ImageNet | train2012 | val2012 | 256 | - | 5.48 | 0.47 |


 - **Pretrained Model Weights Download**

    - [Darknet19_256](https://drive.google.com/file/d/18mBYZ6-X0HqzPNHSyzrqiynaZer1TRK2/view?usp=share_link)
    - [Darknet53_256](https://drive.google.com/file/d/1Mz0ARtsGSOYeHoPZYeH22shI3-ZbJm1q/view?usp=share_link)
    - [Darknet53-tiny_256](https://drive.google.com/file/d/1JeZ7eUuOo9yTxyrLDD6A6IgFVYa0I34o/view?usp=share_link)
    - [CSP-Darknet53_256](https://drive.google.com/file/d/1fOHGCnX-kCpBIPH5yWs0Qq17UbMkY68A/view?usp=share_link)
    - [DarkNet19_224](https://drive.google.com/file/d/1UlCDGDjGKl_Cx8HehaIgsnB09j-YccuR/view?usp=share_link)
    - [DarkNet19_448](https://drive.google.com/file/d/1VA4Lc5MUFzL_WQ2-HVQMkH6sLF44fWsj/view?usp=share_link)



![result](./asset/data.jpg)



## [Usage]


#### Model Training 

 - You can train various classifier architectures (ResNet18, ResNet34, ResNet50, ResNet101, DarkNet19, DarkNet53, and CSP-DarkNet53, Mobilenetv1-v3). In addition, you can finetune classifier adding "--pretrained" in training command after putting the weight files into "./weights" directory with {model name}.  


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