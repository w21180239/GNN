from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import GCNConv, GATConv, GAE, VGAE, ARMAConv,AGNNConv,ARGA,ARGVA
from torch_geometric.data import DataLoader,Dataset,Data
from sklearn import cluster as cl
from sklearn import metrics as me
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import classification_report
from time import time
from xgboost import XGBClassifier
from sklearn import neighbors
import argparse
import warnings
import matplotlib.pyplot as plt
from pytorchtools import EarlyStopping
from mpl_toolkits.mplot3d import Axes3D
import os
import networkx as nx
import numpy as np
from sklearn.preprocessing import Imputer,RobustScaler
from torch_geometric.utils import remove_self_loops
import pandas as pd

warnings.filterwarnings('ignore')


discriminator_loss_para = 1
reg_loss_para = 1
batch_size = 256
epoch = 500
learning_rate = 1e-4
weight_decay = 5e-4
channels = 2
patience = 50
early = True

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GAE')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--encoder', type=str, default='GCN')
parser.add_argument('--dropout', type=int, default=0.5)
args = parser.parse_args()
kwargs = {'GAE': GAE, 'VGAE': VGAE,'ARGA':ARGA,'ARGVA':ARGVA}
encoder_args = {'GCN': GCNConv, 'GAT': GATConv, 'ARMA': ARMAConv,'AGNN':AGNNConv}
assert args.model in kwargs.keys()
assert args.encoder in encoder_args.keys()
assert args.dataset in ['Cora', 'CiteSeer', 'PubMed']

def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        if args.encoder in ['GCN']:
            self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        elif args.encoder in ['GAT']:
            self.conv1 = GATConv(in_channels, 8, heads=8, dropout=args.dropout)
        elif args.encoder in ['ARMA']:
            self.conv1 = ARMAConv(
                in_channels,
                2 * out_channels,
                num_stacks=3,
                num_layers=2,
                shared_weights=True,
                dropout=args.dropout)
        elif args.encoder in ['AGNN']:
            self.lin1 = torch.nn.Linear(in_channels, 2*out_channels)
            self.lin2 = torch.nn.Linear(2*out_channels, out_channels)

            self.conv1 = AGNNConv(requires_grad=False)

        if args.model in ['GAE','ARGA']:
            if args.encoder in ['GCN']:
                self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
            elif args.encoder in ['GAT']:
                self.conv2 = GATConv(8 * 8, out_channels, dropout=args.dropout)
            elif args.encoder in ['ARMA']:
                self.conv2 = ARMAConv(
                    2 * out_channels,
                    out_channels,
                    num_stacks=3,
                    num_layers=2,
                    shared_weights=True,
                    dropout=0.25,
                    act=None)
            elif args.encoder in ['AGNN']:
                self.conv2 = AGNNConv(requires_grad=True)
        elif args.model in ['VGAE','ARGVA']:
            if args.encoder in ['GCN']:
                self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
                self.conv_logvar = GCNConv(
                    2 * out_channels, out_channels, cached=True)
            elif args.encoder in ['GAT']:
                self.conv_mu = GATConv(8 * 8, out_channels, dropout=args.dropout)
                self.conv_logvar = GATConv(8 * 8, out_channels, dropout=args.dropout)
            elif args.encoder in ['ARMA']:
                self.conv_mu = ARMAConv(
                    2 * out_channels,
                    out_channels,
                    num_stacks=3,
                    num_layers=2,
                    shared_weights=True,
                    dropout=0.25,
                    act=None)
                self.conv_logvar = ARMAConv(
                    2 * out_channels,
                    out_channels,
                    num_stacks=3,
                    num_layers=2,
                    shared_weights=True,
                    dropout=0.25,
                    act=None)
            elif args.encoder in ['AGNN']:
                self.lin3 = torch.nn.Linear(2*out_channels, out_channels)
                self.conv_mu = AGNNConv(requires_grad=True)
                self.conv_logvar = AGNNConv(requires_grad=True)


    def forward(self, x, edge_index):
        x = F.dropout(data.x, p=args.dropout, training=self.training)
        if args.encoder in ['AGNN']:
            x = F.relu(self.lin1(x))
        x = self.conv1(x, edge_index)
        if args.encoder not in ['AGNN']:
            x = F.relu(x)
        if args.model in ['GAE','ARGA']:
            x = self.conv2(x, edge_index)
            if args.encoder in ['AGNN']:
                x = F.dropout(x, training=self.training,p=args.dropout)
                x = self.lin2(x)
            return x
        elif args.model in ['VGAE','ARGVA']:
            mu = self.conv_mu(x, edge_index)
            logvar = self.conv_logvar(x, edge_index)
            if args.encoder in ['AGNN']:
                mu = F.dropout(mu, training=self.training,p=args.dropout)
                mu = self.lin2(mu)
                logvar = F.dropout(logvar, training=self.training, p=args.dropout)
                logvar = self.lin2(logvar)
            return mu,logvar

class Discriminator(torch.nn.Module):
    def __init__(self,in_channels, out_channels=1):
        super(Discriminator,self).__init__()
        self.mlp = Seq(
            MLP([in_channels, 64]), Dropout(args.dropout),
            Lin(64, out_channels))
    def forward(self,x):
        return self.mlp(x)

# dataset = Planetoid(root='/Cora', name='Cora')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_df = pd.DataFrame(pd.read_csv('new_playerCharge-4.csv'))
id_list = list(data_df['openid'])
y = list(data_df['charge'])
G = nx.read_gpickle('my_graph.gpickle')
G = G.subgraph(id_list)
x = np.load('feature.npy')[id_list,1:]
x = Imputer().fit_transform(x)
x = RobustScaler().fit_transform(x)
num_feature = x.shape[1]
x = torch.from_numpy(x).to(torch.float)
edge_index = torch.tensor(G.edges())
edge_index = edge_index.t().contiguous()
edge_index = edge_index- edge_index.min()
edge_index, _ = remove_self_loops(edge_index)

data = Data(x=x,edge_index=edge_index,y=y)
del G,x,edge_index
np.save('sur_x.npy',x)
np.save('sur_x.npy',y)

#无监督
parameter2model = [Encoder(num_feature, channels)]
if args.model in ['ARGVA','ARGA']:
    dis = Discriminator(channels)
    parameter2model.append(dis)

model = kwargs[args.model](*parameter2model).to(dev)
#半监督
# model = Encoder(num_feature, 1)
# data = data.to(dev)

data.train_mask = data.val_mask = data.test_mask = data.y = None
x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train():
    print('Train:')
    print('--------------------------------------------------\n\n')
    global epoch, learning_rate, weight_decay,model
    if early:
        early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epoch):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)
        loss = model.recon_loss(z, train_pos_edge_index)
        if early:
            val_loss = model.recon_loss(z, data.val_pos_edge_index)
        if args.model in ['VGAE','ARGVA']:
            loss = loss + (1 / data.num_nodes) * model.kl_loss()
            if early:
                val_loss = val_loss + (1 / data.num_nodes) * model.kl_loss()
        if args.model in ['ARGA','ARGVA']:
            loss = loss + discriminator_loss_para*model.discriminator_loss(z) + reg_loss_para*model.reg_loss(z)
            if early:
                val_loss = val_loss + discriminator_loss_para*model.discriminator_loss(z) + reg_loss_para*model.reg_loss(z)
        loss.backward()
        optimizer.step()
        if not epoch % 5:
            model.eval()
            with torch.no_grad():
                z = model.encode(data.x, train_pos_edge_index)
                roc, mean = model.test(z, data.val_pos_edge_index, data.val_neg_edge_index)
                print(f'epoch:{epoch}\tloss:{loss}\troc:{roc}\tmean:{mean}')
        if early:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    if early:
        if os.path.exists('checkpoint.pt'):
            model.load_state_dict(torch.load('checkpoint.pt'))


    return model, data


def test_unsupervised(model, data):
    cluster_model = cl.KMeans(4)

    model.eval()
    z = model.encode(data.x, data.train_pos_edge_index)
    roc, mean = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
    z = z.cpu().detach().numpy()
    if data.y is None:
        cluster_model.fit(z)
    else:
        true_label = data.y.cpu().detach().numpy()

        t = time()
        predict_label = cluster_model.fit_predict(data.x.cpu().detach().numpy())
        t_before = time() - t
        X = pairwise_distances(data.x.cpu().detach().numpy(), metric='cosine')
        before_embedding_1 = (true_label, predict_label)
        before_embedding_2 = (X, true_label)

        t = time()
        predict_label = cluster_model.fit_predict(z)
        t_after = time() - t
        X = pairwise_distances(z, metric='cosine')

        after_embedding_1 = (true_label, predict_label)
        after_embedding_2 = (X, true_label)
        metric_before = {'time': t_before,
                         'adjusted_rand_score': me.adjusted_rand_score(*before_embedding_1),
                         'adjuested_mutual_info_score': me.adjusted_mutual_info_score(*before_embedding_1),
                         'homogeneity_score': me.homogeneity_score(*before_embedding_1),
                         'completeness_score': me.completeness_score(*before_embedding_1),
                         'v_measure_score': me.v_measure_score(*before_embedding_1),
                         'fowlkes_mallows_score': me.fowlkes_mallows_score(*before_embedding_1),
                         'silhouette_score': me.silhouette_score(*before_embedding_2),
                         'calinski_harabaz_score': me.calinski_harabaz_score(*before_embedding_2)}

        metric_after = {'time': t_after,
                        'roc_score': roc,
                        'mean_acc': mean,
                        'adjusted_rand_score': me.adjusted_rand_score(*after_embedding_1),
                        'adjuested_mutual_info_score': me.adjusted_mutual_info_score(*after_embedding_1),
                        'homogeneity_score': me.homogeneity_score(*after_embedding_1),
                        'completeness_score': me.completeness_score(*after_embedding_1),
                        'v_measure_score': me.v_measure_score(*after_embedding_1),
                        'fowlkes_mallows_score': me.fowlkes_mallows_score(*after_embedding_1),
                        'silhouette_score': me.silhouette_score(*after_embedding_2),
                        'calinski_harabaz_score': me.calinski_harabaz_score(*after_embedding_2)}
        print('\n\n\nTest:')
        print('--------------------------------------------------\n\n')
        print('Before embedding:')
        for key, value in metric_before.items():
            print(f'{key}:{value}')
        print('--------------------------------------------------\n\n')
        print('After embedding:')
        for key, value in metric_after.items():
            print(f'{key}:{value}')

    return cluster_model,z





def test_supervised(model, data):
    model.eval()
    z = model.encode(data.x, data.train_pos_edge_index)
    train_label = data.y[data.train_mask].cpu().detach().numpy()
    test_label = data.y[data.test_mask].cpu().detach().numpy()
    target_names = [f'class{i}' for i in range(7)]

    print('\n\n\nNormal classifier')
    classify_model = neighbors.KNeighborsClassifier()
    classify_model.fit(data.x[data.train_mask].cpu().detach().numpy(), train_label)

    ori_pre = classify_model.predict(data.x[data.test_mask].cpu().detach().numpy())
    print('--------------------------------------------------\n\n')
    print('Original:')
    print(classification_report(test_label, ori_pre, target_names=target_names))

    classify_model = neighbors.KNeighborsClassifier()
    classify_model.fit(z[data.train_mask].cpu().detach().numpy(), train_label)
    ori_pre = classify_model.predict(z[data.test_mask].cpu().detach().numpy())
    print('--------------------------------------------------\n\n')
    print('Embedded:')
    print(classification_report(test_label, ori_pre, target_names=target_names))

    print('\n\n\nStrong classifier')
    classify_model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
    # classify_model = neighbors.KNeighborsClassifier()
    classify_model.fit(data.x[data.train_mask].cpu().detach().numpy(), train_label)
    ori_pre = classify_model.predict(data.x[data.test_mask].cpu().detach().numpy())
    print('--------------------------------------------------\n\n')
    print('Original:')
    print(classification_report(test_label, ori_pre, target_names=target_names))

    classify_model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
    # classify_model = neighbors.KNeighborsClassifier()
    classify_model.fit(z[data.train_mask].cpu().detach().numpy(), train_label)
    ori_pre = classify_model.predict(z[data.test_mask].cpu().detach().numpy())
    print('--------------------------------------------------\n\n')
    print('Embedded:')
    print(classification_report(test_label, ori_pre, target_names=target_names))

def plot_graph(cluster_model,z):
    print('--------------------------------------------------\n\n')
    print('Ploting...')
    label_pred = cluster_model.labels_  # 获取聚类标签
    mark = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    c_list = [mark[i] for i in label_pred]
    if z.shape[1]==2:
        ax = plt.subplot(111)
        ax.scatter(z[:, 0], z[:, 1], c=c_list, s=10)
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    elif z.shape[1]==3:
        ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
        ax.scatter(z[:, 0], z[:, 1],z[:, 2], c=c_list, s=20)
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    else:
        print('Wrong dimension!')
    plt.show()




model, data = train()
# test_supervised(model, data)
cluster_model,z = test_unsupervised(model, data)
plot_graph(cluster_model,z)

