# 无监督的各种GNN（GAE等）
## 框架和库
本项目基于[Pytorch geometric](https://github.com/rusty1s/pytorch_geometric)实现，请根据官方指引安装最新版（可能要直接从github安装，直接pip安装可能版本不够新）

其他库版本：
- pytorch   v1.1.0
- xgboost   v0.90
- sklearn   v0.21.2
- networkx  v2.3
- pandas    v0.24.2
- matplotlib     v2.2.3
- numpy  v1.16.4
##参数说明
model
> 使用什么无监督框架，可供选择的有：GAE, VGAE, ARGA, ARGVA

dataset
> 使用什么标准数据集，可供选择的有：Cora, CiteSeer, PubMed, Reddit

encoder
> 使用什么模型当做encoder，可供选择的有：GCN, GAT, AGNN, ARMA

dropout
> dropout rate

lr
> 学习率

l2
> L2 loss 系数

dis_loss_para
> ARGA和ARVGA中使用到的参数，请参考[官方文档](https://pytorch-geometric.readthedocs.io/en/latest/)

reg_loss_para
> VGAE和ARVGA中使用到的参数，请参考[官方文档](https://pytorch-geometric.readthedocs.io/en/latest/)

epoch
> 最大训练epoch

hidden_channels
> 产生的embedding的维度，为2或3时进行unsupervised测试后将自动画出聚类图

patience
> early stop的patience

subgraph_num
> 将整个大图随机划分为subgraph_num个子图并轮流进行训练（子图node数量为原图node数量//subgraph_num，不同子图之间的node可以出现重复）

batch
> 进行matrix completion（对每对节点进行link prediction）的时候的batch size（在其他地方无效）
##说明文档
请参考Graph Neural Network.pdf和推荐系统最新进展.pdf
## 运行方法
全部使用默认参数：
```
python main.py
```
指定部分参数：
```
python main.py --model GAE --encoder GAT --dataset Cora --hidden_channels 64
```
