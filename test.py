import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataListLoader,DataLoader
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler,RobustScaler

from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel

train_data_list = torch.load('subway_data_train.npz')
val_data_list = torch.load('subway_data_val.npz')
pre_data_list = []
pre_data_list.append(torch.load('subway_data_pre_15.npz'))
pre_data_list.append(torch.load('subway_data_pre_30.npz'))
pre_data_list.append(torch.load('subway_data_pre_45.npz'))
x_scaler = RobustScaler()
y_scaler = RobustScaler()
y = torch.cat([data.y[:,1].unsqueeze(1) for data in train_data_list], 0)
x = torch.cat([data.x for data in train_data_list], 0)
x_scaler.fit(x)
y_scaler.fit(y)
del x, y
for data in train_data_list:
    x = x_scaler.transform(data.x)
    data.x = torch.from_numpy(x).to(torch.float)
    y = y_scaler.transform(data.y)
    data.y = torch.from_numpy(y).to(torch.float)

for data in val_data_list:
    x = x_scaler.transform(data.x)
    data.x = torch.from_numpy(x).to(torch.float)

for pre in pre_data_list:
    for data in pre:
        x = x_scaler.transform(data.x)
        data.x = torch.from_numpy(x).to(torch.float)
        data.y = data.y.squeeze()

train_loader = DataLoader(
    train_data_list, batch_size=100, shuffle=True)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(300, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.lin1 = torch.nn.Linear(64, 128)
        self.lin2 = torch.nn.Linear(128, 3)

    def forward(self, data):
        print('Inside Model:  num graphs: {}, device: {}'.format(
            data.num_graphs, data.batch.device))

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = global_mean_pool(x, data.batch)
        x = F.elu(self.lin1(x))
        return F.log_softmax(self.lin2(x), dim=1)


model = Net()
print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
model = DataParallel(model)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for data_list in train_loader:
    optimizer.zero_grad()
    output = model(data_list)
    print('Outside Model: num graphs: {}'.format(output.size(0)))
    y = torch.cat([data.y for data in data_list]).to(output.device)
    loss = F.nll_loss(output, y)
    loss.backward()
    optimizer.step()