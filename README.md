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
| Darknet19<br><sup>(<u>Our:star:</u>)</br> | ImageNet | train2012 | val2012 | 224 | 72.52 | 20.84 | 5.62 |
| Darknet53<br><sup>(<u>Our:star:</u>)</br> | ImageNet | train2012 | val2012 | 224 | - | 41.61 | 14.29 |
| Darknet53-tiny<br><sup>(<u>Our:star:</u>)</br> | ImageNet | train2012 | val2012 | 224 | - | 2.09 | 0.64 |
| Mobilenetv1<br><sup>(<u>Our:star:</u>)</br> | ImageNet | train2012 | val2012 | 224 | - | 4.23 | 1.18 |
| Mobilenetv2<br><sup>(<u>Our:star:</u>)</br> | ImageNet | train2012 | val2012 | 224 | - | 3.50 | 0.66 |
| Mobilenetv3-small<br><sup>(<u>Our:star:</u>)</br> | ImageNet | train2012 | val2012 | 224 | - | 2.54 | 0.12 |
| Mobilenetv3-large<br><sup>(<u>Our:star:</u>)</br> | ImageNet | train2012 | val2012 | 224 | - | 5.48 | 0.47 |


 - **Pretrained Model Weights Download**

	- [DarkNet19-224](https://drive.google.com/file/d/1UlCDGDjGKl_Cx8HehaIgsnB09j-YccuR/view?usp=share_link)
	- [DarkNet19-448](https://drive.google.com/file/d/1VA4Lc5MUFzL_WQ2-HVQMkH6sLF44fWsj/view?usp=share_link)


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
                                                   --model darknet53_tiny
                                                   --model cspdarknet53 --width_multiple 1.0 --depth_multiple 1.0
                                                   --model mobilenetv1 --width_multiple 1.0
                                                   --model mobilenetv2 --width_multiple 1.0
                                                   --model mobilenetv3 --mode {large, small} --width_multiple 1.0
```


#### Classification Evaluation

 - It computes Top-1 accuracy. Top-1 accuracy is the conventional accuracy, the model answer (the one with highest probability) must be exactly the expected answer. 

```python
python val.py --exp my_test --data voc.yaml --ckpt_name best.pt
```


---
## [Contact]
- Author: Jiho Park  
- Email: pjh5672.dev@gmail.com  