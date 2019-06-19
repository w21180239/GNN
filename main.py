import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import os
import pandas as pd
from torch_geometric.data import NeighborSampler,Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import remove_self_loops
from sklearn.preprocessing import Imputer, RobustScaler,MinMaxScaler, StandardScaler
from torch_geometric.data import DataLoader
import random
from torch_geometric.nn import GATConv

batchsize = 1


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = GATConv(in_channels,8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, 1, heads=1, concat=True, dropout=0.6)



    def forward(self, x,edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

y_scaler= StandardScaler()

data_list = torch.load('subway_data_train.npz')
train_loader = DataLoader(
    data_list, batch_size=batchsize, shuffle=True)

val_loader = DataLoader(
    torch.load('subway_data_val.npz'), batch_size=batchsize, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(data_list[0].num_features,data_list[0].y.size(1)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


def train(loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = F.mse_loss(out, data.y.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader.dataset)


def test(loader):
    model.eval()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = F.mse_loss(out, data.y.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader.dataset)


for epoch in range(1, 1001):
    loss = train(train_loader)
    test_acc = test(val_loader)
    print('Epoch: {:02d}, Loss: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, test_acc))