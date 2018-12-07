# SOS: Sequence-of-Sequences Model for 3D Object Recognition

This repository holds the codes and models for the papers

**SOS: Sequence-of-Sequences Model for 3D Object Recognition** Tan Yu, Zhou Ren, Yuncheng Li, Enxu Yan, Ning Xu, Jianxiong Yin, Simon See, Junsong Yuan 

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for training and testing purposes. 

### Prerequisites
Pytorch

```
conda install pytorch torchvision -c pytorch
```

### Data Prepare

You need download ModelNet40 dataset

with orientation assumption, 6-view settings
```
wget http://www.cim.mcgill.ca/dscnn-data/ModelNet40_rendered_rgb.tar; tar -xvf ModelNet40_rendered_rgb.tar 
```

without orientation assumption, 20-view settings
```
 wget https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar; tar -xvf modelnet40v2png_ori4.tar
```

### Download trained models
* [cache_model.tar](https://drive.google.com/open?id=1CB01BrLuPBUCNhq9pDiwIyteUecbN8BO)

```
 tar -xvf cache_model.tar  
```

## Training

### 6-view ModelNet40
```
python mainalex.py -d modelnet40 -v 6 
```

### 6-view ModelNet10
```
python mainalex.py -d modelnet10 -v 6
```

### 20-view ModelNet40
```
python mainalex.py -d modelnet40 -v 20
```

### 20-view ModelNet10
```
python mainalex.py -d modelnet10 -v 20
```

## Testing

### 6-view ModelNet40
```
python testalex.py -d modelnet40 -v 6 --resume cache_models/
```

### 6-view ModelNet10
```
python testalex.py -d modelnet10 -v 6 --resume cache_models/
```

### 20-view ModelNet40
```
python testalex.py -d modelnet40 -v 20 --resume cache_models/
```

### 20-view ModelNet10
```
python testalex.py -d modelnet10 -v 20 --resume cache_models/
```







## Authors

* **Tan YU** - [Homepage](https://sites.google.com/site/tanyuspersonalwebsite/home)
