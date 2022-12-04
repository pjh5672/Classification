# <div align="center">Classifier</div>

---

## [Content]
1. [Description](#description)   
2. [Usage](#usage)  
2-1. [Model Training](#model-training)  
2-2. [Detection Evaluation](#detection-evaluation)  
3. [Contact](#contact)   

---

## [Description]

This is a repository for PyTorch training implementation of general classifier.    



## [Usage]


#### Model Training 

```python
python train.py --exp my_test --data imagenet.yaml
```


#### Classification Evaluation

```python
python val.py --exp my_test --data voc.yaml --ckpt_name best.pt
```



---
## [Contact]
- Author: Jiho Park  
- Email: pjh5672.dev@gmail.com  