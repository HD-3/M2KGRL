## Datasets:
#### FB15K:  
FB15K is a widely used dataset in KG embedding extracted from Freebase. You can download [here](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/tree/master/data/FB15k).  
#### WN18：  
WN18 is a well-known KG which is originally extracted from WordNet. You can download [here](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/tree/master/data/wn18).  
#### FB15K-IMG:  
In [mmkb](https://github.com/mniepert/mmkb), they provide a list of URLs that can be downloaded with a script which also scales the images,and they also provide the links of the Freebase IDs to their image URLs.  
#### WN18-IMG：  
Entity images in WN18 can be obtained from ImageNet, the specific steps can refer to RSME. the RSME [repository](https://github.com/wangmengsd/RSME).  
## Model Architecture
![image](https://github.com/HiddenDragon33/M2KGRL/blob/main/model.png)
## Requirements  
Pytorch 1.11.0, python 3.7  
##  Reproducing the Results in the Paper  
#### WN18:  
RUN python main.py -ne 1000 -lr 0.03 -reg 0.1 -dataset WN18 -emb_dim 600 -neg_ratio 5 -batch_size 1415 -save_each 50  
#### FB15K:  
RUN python main.py -ne 1000 -lr 0.03 -reg 0.3 -dataset FB15K -emb_dim 500 -neg_ratio 20 -batch_size 4832 -save_each 50  
## Experiments Results
![image](https://github.com/HiddenDragon33/M2KGRL/blob/main/result.png)


