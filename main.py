import os

import torch
import torch.nn.functional as F
# import networkx as nx
# import numpy as np
# import os
# import pandas as pd
# from torch_geometric.data import NeighborSampler,Data
# from torch_geometric.nn import SAGEConv
# from torch_geometric.utils import remove_self_loops
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batchsize = 10


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, 3, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


train_data_list = torch.load('subway_data_train.npz')
val_data_list = torch.load('subway_data_val.npz')
y_scaler = StandardScaler()
y = torch.cat([data.y for data in train_data_list], 0)
y_scaler.fit(y)
for data in train_data_list:
    y = y_scaler.transform(data.y)
    data.y = torch.from_numpy(y).to(torch.float)
for data in val_data_list:
    y = y_scaler.transform(data.y)
    data.y = torch.from_numpy(y).to(torch.float)
train_loader = DataLoader(
    train_data_list, batch_size=batchsize, shuffle=True)

val_loader = DataLoader(
    val_data_list, batch_size=batchsize, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_data_list[0].num_features, train_data_list[0].y.size(1)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


def train(loader):
    model.train()

    total_loss = []
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = F.mse_loss(out, data.y.to(device))
        loss.backward()
        optimizer.step()
        loss_list = [F.mse_loss(out[:, i], data.y.to(device)[:, i]).item() for i in range(3)]
        total_loss.append(loss_list)
    total_loss = y_scaler.inverse_transform(total_loss)
    total_loss = [total_loss[:, i].sum() / len(total_loss) for i in range(3)]
    return total_loss


def test(loader):
    model.eval()

    total_loss = []
    for data in loader:
        out = model(data.x.to(device), data.edge_index.to(device))
        loss_list = [F.mse_loss(out[:, i], data.y.to(device)[:, i]).item() for i in range(3)]
        total_loss.append(loss_list)
    total_loss = y_scaler.inverse_transform(total_loss)
    total_loss = [total_loss[:, i].sum() / len(total_loss) for i in range(3)]
    return total_loss


for epoch in range(1, 1001):
    loss = train(train_loader)
    test_acc = test(val_loader)
    print(f'Epoch:{epoch}\nTrain:{loss[0]}\t{loss[1]}\t{loss[2]}\nTest:{test_acc[0]}\t{test_acc[1]}\t{test_acc[2]}')
