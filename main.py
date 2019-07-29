import argparse
import os.path as osp
import random
import warnings
from time import time, sleep

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn import cluster as cl
from sklearn import metrics as me
from sklearn import neighbors
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import pairwise_distances
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, GAE, VGAE, ARMAConv, AGNNConv, ARGA, ARGVA
from torch_geometric.utils import subgraph
from xgboost import XGBClassifier

from pytorchtools import EarlyStopping

warnings.filterwarnings('ignore')

discriminator_loss_para = 1
reg_loss_para = 1
epoch = 500
learning_rate = 1e-3
weight_decay = 0
channels = 3
patience = 50
early = True
times = 2

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GAE')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--encoder', type=str, default='GCN')
parser.add_argument('--dropout', type=int, default=0.5)
args = parser.parse_args()
kwargs = {'GAE': GAE, 'VGAE': VGAE, 'ARGA': ARGA, 'ARGVA': ARGVA}
encoder_args = {'GCN': GCNConv, 'GAT': GATConv, 'ARMA': ARMAConv, 'AGNN': AGNNConv}
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
            self.conv1 = GCNConv(in_channels, 2 * out_channels)
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
            self.lin1 = torch.nn.Linear(in_channels, 2 * out_channels)
            self.lin2 = torch.nn.Linear(2 * out_channels, out_channels)

            self.conv1 = AGNNConv(requires_grad=False)

        if args.model in ['GAE', 'ARGA']:
            if args.encoder in ['GCN']:
                self.conv2 = GCNConv(2 * out_channels, out_channels)
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
        elif args.model in ['VGAE', 'ARGVA']:
            if args.encoder in ['GCN']:
                self.conv_mu = GCNConv(2 * out_channels, out_channels)
                self.conv_logvar = GCNConv(
                    2 * out_channels, out_channels)
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
                self.lin3 = torch.nn.Linear(2 * out_channels, out_channels)
                self.conv_mu = AGNNConv(requires_grad=True)
                self.conv_logvar = AGNNConv(requires_grad=True)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=args.dropout, training=self.training)
        if args.encoder in ['AGNN']:
            x = F.relu(self.lin1(x))
        x = self.conv1(x, edge_index)
        if args.encoder not in ['AGNN']:
            x = F.relu(x)
        if args.model in ['GAE', 'ARGA']:
            x = self.conv2(x, edge_index)
            if args.encoder in ['AGNN']:
                x = F.dropout(x, training=self.training, p=args.dropout)
                x = self.lin2(x)
            return x
        elif args.model in ['VGAE', 'ARGVA']:
            mu = self.conv_mu(x, edge_index)
            logvar = self.conv_logvar(x, edge_index)
            if args.encoder in ['AGNN']:
                mu = F.dropout(mu, training=self.training, p=args.dropout)
                mu = self.lin2(mu)
                logvar = F.dropout(logvar, training=self.training, p=args.dropout)
                logvar = self.lin2(logvar)
            return mu, logvar


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(Discriminator, self).__init__()
        self.mlp = Seq(
            MLP([in_channels, 128, 64]), Dropout(args.dropout),
            Lin(64, out_channels))

    def forward(self, x):
        return self.mlp(x)


def train(data):
    print('Train:')
    print('--------------------------------------------------\n\n')
    global epoch, learning_rate, weight_decay, model
    if early:
        early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epoch):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.train_pos_edge_index)
        loss = model.recon_loss(z, data.train_pos_edge_index)
        if early:
            test_loss = model.recon_loss(z, data.test_pos_edge_index)
        if args.model in ['VGAE', 'ARGVA']:
            loss = loss + (1 / data.num_nodes) * model.kl_loss()
            if early:
                test_loss = test_loss + (1 / data.num_nodes) * model.kl_loss()
        if args.model in ['ARGA', 'ARGVA']:
            loss = loss + discriminator_loss_para * model.discriminator_loss(z) + reg_loss_para * model.reg_loss(z)
            if early:
                test_loss = test_loss + discriminator_loss_para * model.discriminator_loss(
                    z) + reg_loss_para * model.reg_loss(z)
        loss.backward()
        optimizer.step()
        if not epoch % 5:
            model.eval()
            with torch.no_grad():
                z = model.encode(data.x, data.train_pos_edge_index)
                roc, mean = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
                print(f'graph_num:{graph_num}\tepoch:{epoch}\tloss:{loss}\troc:{roc}\tmean:{mean}')
        if early:
            early_stopping(test_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    if early:
        model.load_state_dict(torch.load('checkpoint.pt'))


def test_unsupervised(model, data):
    cluster_model = cl.KMeans(data.y.cpu().numpy().max() + 1)

    model.eval()
    z = model.encode(data.x.to(dev), data.train_pos_edge_index.to(dev))
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

    return cluster_model, z


def test_supervised(model, data):
    model.eval()
    z = model.encode(data.x, data.train_pos_edge_index)
    train_label = data.y[data.train_mask].cpu().detach().numpy()
    test_label = data.y[data.test_mask].cpu().detach().numpy()
    target_names = [f'class{i}' for i in range(data.y.cpu().numpy().max() + 1)]

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


def plot_graph(cluster_model, z):
    print('--------------------------------------------------\n\n')
    print('Ploting...')
    label_pred = cluster_model.labels_  # 获取聚类标签
    mark = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    c_list = [mark[i] for i in label_pred]
    if z.shape[1] == 2:
        ax = plt.subplot(111)
        ax.scatter(z[:, 0], z[:, 1], c=c_list, s=10)
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    elif z.shape[1] == 3:
        ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=c_list, s=20)
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    else:
        print('Wrong dimension!')
    plt.show()


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(
    osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
dataset = Planetoid(path, args.dataset)
data = dataset[0]

# 无监督
parameter2model = [Encoder(data.num_features, channels)]
if args.model in ['ARGVA', 'ARGA']:
    dis = Discriminator(channels)
    parameter2model.append(dis)

model = kwargs[args.model](*parameter2model).to(dev)

# 半监督
# model = Encoder(num_feature, 1)
# data = data.to(dev)

ori_data = data.clone().to(dev)
for graph_num in range(times):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x, y, edge_index = None, None, None

    node_list = [i for i in range(ori_data.num_nodes)]
    random.shuffle(node_list)
    node_list = node_list[:ori_data.num_nodes // times]
    if ori_data.x is not None:
        x = ori_data.x[node_list]
    if ori_data.y is not None:
        y = ori_data.y[node_list]
    if ori_data.edge_index is not None:
        edge_index = subgraph(node_list, ori_data.edge_index, None, True, ori_data.num_nodes)[0]
    data = Data(x=x, y=y, edge_index=edge_index).to(dev)
    data = model.split_edges(data)
    train(data)
    print(f'complete {graph_num}th subgraph, sleep 1s...')
    sleep(1)
test_supervised(model, model.split_edges(ori_data))
cluster_model, z = test_unsupervised(model, data)
plot_graph(cluster_model, z)
