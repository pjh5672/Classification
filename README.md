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

This is a repository for PyTorch training implementation of general purposed classifier. With the training code, various feature extractor, also known as backbone architectures can be created.   


 - **Performance Table**

| Model | Dataset | Train | Valid | Size<br><sup>(pixel) | Accuracy<br><sup>(Top-1) | Params<br><sup>(M) | FLOPs<br><sup>(B) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet18<br><sup>(<u>Our:star:</u>)</br> | ImageNet | train2012 | val2012 | 224 | 68.29 | 11.69 | 3.65 |
| DarkNet19<br><sup>(<u>Our:star:</u>)</br> | ImageNet | train2012 | val2012 | 224 | 72.79 | 20.84 | 5.62 |


![result](./asset/data.jpg)



## [Usage]


#### Model Training 

 - You can train various classifier architectures (ResNet18, ResNet34, ResNet50, ResNet101, DarkNet19, DarkNet53, and CSP-DarkNet53). We support finetune training for parameter changes. For finetuning, put the file with the model name into the "./weights" directory and add "--finetune" command in training command. In addition, you can change model with depth-wise separable convolution layers with adding "--depthwise".

```python
python train.py --exp my_test --data imagenet.yaml --model resnet18 --finetune (optional) --depthwise (optional)
                                                   --model resnet34
                                                   --model resnet50
                                                   --model resnet101
                                                   --model darknet19
                                                   --model darknet53
                                                   --model cspdarknet53 --width_multiple 1.0 --depth_multiple 1.0
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