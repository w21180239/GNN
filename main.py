import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import os
import pandas as pd
from torch_geometric.data import NeighborSampler,Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import remove_self_loops
from sklearn.preprocessing import Imputer, RobustScaler,StandardScaler

if os.path.exists('data.npz'):
    data = torch.load('data.npz')
    y_scaler = StandardScaler()
    y = np.load('sur_y.npy')
    y = y.reshape(-1, 1)
    y_scaler.fit(y)
else:
    G = nx.read_gpickle('my_graph.gpickle')
    if os.path.exists('sur_x.npy'):
        x = np.load('sur_x.npy')
        y = np.load('sur_y.npy')
        data_df = pd.DataFrame(pd.read_csv('new_playerCharge-4.csv'))
        id_list = np.array(data_df['openid'])
    else:
        data_df = pd.DataFrame(pd.read_csv('new_playerCharge-4.csv'))
        id_list = list(data_df['openid'])
        y = data_df['charge']
        G = G.subgraph(id_list)
        x = np.load('feature.npy')[id_list, 1:]
        x = Imputer().fit_transform(x)
        x = RobustScaler().fit_transform(x)
        np.save('sur_x.npy', x)
        np.save('sur_y.npy', y)
    train_mask = np.array([0 for i in range(x.shape[0])])
    val_mask = np.array([0 for i in range(x.shape[0])])
    test_mask = np.array([0 for i in range(x.shape[0])])
    for i in range(0, int(x.shape[0] * 0.8)):
        train_mask[i] = 1
    for i in range(int(x.shape[0] * 0.8), int(x.shape[0] * 0.9)):
        val_mask[i] = 1
    for i in range(int(x.shape[0] * 0.9), x.shape[0]):
        test_mask[i] = 1

    G = G.subgraph(list(id_list))
    ori_key = list(G.node.keys())
    ori_key.sort()
    projection = {}
    for i in range(len(id_list)):
        projection[ori_key[i]] = i
    edge_index = G.edges()
    edge_index = [[projection[i[0]],projection[i[1]]] for i in edge_index]
    # edge_index = edge_index[:10]
    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_index = edge_index - edge_index.min()
    edge_index, _ = remove_self_loops(edge_index)

    x = torch.from_numpy(x).to(torch.float)
    y = y.reshape(-1, 1)
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y)       # transform
    y = torch.from_numpy(y).to(torch.float)
    train_mask =  torch.from_numpy(train_mask).to(torch.uint8)
    val_mask =  torch.from_numpy(val_mask).to(torch.uint8)
    test_mask =  torch.from_numpy(test_mask).to(torch.uint8)

    data = Data(edge_index=edge_index, x=x, y=y, train_mask=train_mask, val_mask=val_mask,
                test_mask=test_mask)
    torch.save(data, 'data.npz')
loader = NeighborSampler(
    data,
    size=[100, 100],
    num_hops=2,
    batch_size=256,
    shuffle=True,
    add_self_loops=True)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(in_channels, 32)
        self.conv2 = SAGEConv(32, out_channels)



    def forward(self, x, data_flow):
        data = data_flow[0]
        x = x[data.n_id]
        x = F.relu(self.conv1(x, data.edge_index,size=data.size))
        x = F.dropout(x, training=self.training)
        data = data_flow[1]
        x = self.conv2(x, data.edge_index,size=data.size)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(67, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()

    total_loss = []
    for data_flow in loader(data.train_mask):
        optimizer.zero_grad()
        out = model(data.x.to(device), data_flow.to(device))
        loss = F.mse_loss(out, data.y[data_flow.n_id].to(device))
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item() * data_flow.batch_size)
    total_loss = y_scaler.inverse_transform(total_loss)
    total_loss = total_loss.sum()
    return total_loss / data.train_mask.sum().item()


def test(mask):
    model.eval()

    total_loss = []
    for data_flow in loader(mask):
        pred = model(data.x.to(device), data_flow.to(device))
        total_loss.append(F.mse_loss(pred, data.y[data_flow.n_id].to(device))* data_flow.batch_size)
    total_loss = y_scaler.inverse_transform(total_loss)
    total_loss = total_loss.sum()
    return total_loss / data.test_mask.sum().item()


for epoch in range(1, 1001):
    loss = train()
    test_acc = test(data.test_mask)
    print('Epoch: {:02d}, Loss: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, test_acc))