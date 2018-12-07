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

6-view ModelNet40
```
python mainalex.py -d modelnet40 -v 6 
```

6-view ModelNet10
```
python mainalex.py -d modelnet10 -v 6
```

20-view ModelNet40
```
python mainalex.py -d modelnet40 -v 20
```

20-view ModelNet10
```
python mainalex.py -d modelnet10 -v 20
```

## Testing

6-view ModelNet40
```
python testalex.py -d modelnet40 -v 6 --resume cache_models/alexnet_modelnet40_6view.tar
```

6-view ModelNet10
```
python testalex.py -d modelnet10 -v 6 --resume cache_models/alexnet_modelnet40_6view.tar
```

20-view ModelNet40
```
python testalex.py -d modelnet40 -v 20 --resume cache_models/alexnet_modelnet40_6view.tar
```

20-view ModelNet10
```
python testalex.py -d modelnet10 -v 20 --resume cache_models/alexnet_modelnet40_6view.tar

```


## Authors
* [Tan Yu](https://sites.google.com/site/tanyuspersonalwebsite/home)
* [Zhou Ren](http://web.cs.ucla.edu/~zhou.ren/)
* [Yuncheng Li](http://www.cs.rochester.edu/~yli/)
* [Enxu Yan](http://ianyen.site/)
* [Ning Xu](https://www.linkedin.com/in/ningxu01/)
* [Jianxiong Yin](https://www.linkedin.com/in/jianxiong-yin-3a25541b/)
* [Simon See](https://www.linkedin.com/in/simonsee/)
* [Junsong Yuan](https://cse.buffalo.edu/~jsyuan/)

