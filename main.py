import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler,RobustScaler
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, AGNNConv, ARMAConv, SplineConv

from pytorchtools import EarlyStopping
import warnings

warnings.filterwarnings('ignore')
torch.cuda.set_device(2)

batchsize = 400
drop_rate = 0
mmodel = 'GAT'


def MLP(channels):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])


class Breadth(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Breadth, self).__init__()
        self.gatconv = GATConv(in_dim, out_dim, heads=1)

    def forward(self, x, edge_index):
        x = torch.tanh(self.gatconv(x, edge_index))
        return x


class Depth(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super(Depth, self).__init__()
        self.lstm = torch.nn.LSTM(in_dim, hidden, 1, bias=False)

    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))
        return x, (h, c)


class GeniePathLayer(torch.nn.Module):
    def __init__(self, in_dim):
        super(GeniePathLayer, self).__init__()
        self.breadth_func = Breadth(in_dim, 64)
        self.depth_func = Depth(64, 64)

    def forward(self, x, edge_index, h, c):
        x = self.breadth_func(x, edge_index)
        x = x[None, :]
        x, (h, c) = self.depth_func(x, h, c)
        x = x[0]
        return x, (h, c)


class Net(torch.nn.Module):
    def __init__(self, FEA, OUT):
        super(Net, self).__init__()
        self.mlp = MLP([64, 128, 256, 128, 64])
        self.heads = nn.ModuleList([MLP([64, 32,32, 1]) for i in range(3)])

        if mmodel == 'GAT':
            self.conv1 = GATConv(FEA, 12, heads=12)
            self.conv2 = GATConv(
                12 * 12, 12, heads=12)
            self.conv3 = GATConv(
                12 * 12, 12, heads=12)
            self.conv4 = GATConv(
                12 * 12, 8, heads=8)
        elif mmodel == 'AGNN':
            self.lin1 = torch.nn.Linear(FEA, 64)
            self.prop1 = AGNNConv(requires_grad=False)
            self.prop2 = AGNNConv(requires_grad=True)
            self.prop3 = AGNNConv(requires_grad=True)
        elif mmodel == 'ARMA':
            self.conv1 = ARMAConv(
                FEA,
                64,
                num_stacks=3,
                num_layers=2,
                shared_weights=True,
                dropout=drop_rate)

            self.conv2 = ARMAConv(
                64,
                64,
                num_stacks=3,
                num_layers=2,
                shared_weights=True,
                dropout=drop_rate,
                act=None)
        elif mmodel == 'Spline':
            self.conv1 = SplineConv(FEA, 16, dim=1, kernel_size=2)
            self.conv2 = SplineConv(16, OUT, dim=1, kernel_size=2)
        elif mmodel == 'Genie':
            self.lin1 = torch.nn.Linear(FEA, 64)
            self.gplayers = torch.nn.ModuleList(
                [GeniePathLayer(64) for i in range(2)])

    def forward(self, x, edge_index):
        if mmodel == 'GAT':
            x = F.dropout(x, p=drop_rate, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=drop_rate, training=self.training)
            # x = F.elu(self.conv2(x, edge_index))
            # x = F.dropout(x, p=drop_rate, training=self.training)
            # x = F.elu(self.conv3(x, edge_index))
            # x = F.dropout(x, p=drop_rate, training=self.training)
            x = F.elu(self.conv4(x, edge_index))
            x = self.mlp(x)
            out_list = [head(x) for head in self.heads]
            x = torch.cat(out_list, 1)
            # x = self.heads[0](x)
        elif mmodel == 'AGNN':
            x = F.dropout(x, p=drop_rate, training=self.training)
            x = F.relu(self.lin1(x))
            x = self.prop1(x, edge_index)
            x = self.prop2(x, edge_index)
            x = self.prop3(x, edge_index)
            x = F.dropout(x, p=drop_rate, training=self.training)
            x = self.mlp(x)
        elif mmodel == 'ARMA':
            x = F.dropout(x, p=drop_rate, training=self.training)
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=drop_rate, training=self.training)
            x = F.relu(self.conv2(x, edge_index))
            x = self.mlp(x)
        elif mmodel == 'Spline':
            x = F.dropout(x, p=drop_rate, training=self.training)
            x = F.elu(self.conv1(x, edge_index, edge_attr))
            x = F.dropout(x, p=drop_rate, training=self.training)
            x = F.elu(self.conv2(x, edge_index, edge_attr))
            x = self.mlp(x)
        elif mmodel == 'Genie':
            x = self.lin1(x)
            h = torch.zeros(1, x.shape[0], 64, device=x.device)
            c = torch.zeros(1, x.shape[0], 64, device=x.device)
            for i, l in enumerate(self.gplayers):
                x, (h, c) = self.gplayers[i](x, edge_index, h, c)
            x = self.mlp(x)
        return x


ea = EarlyStopping(verbose=True, patience=10)
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
    train_data_list, batch_size=batchsize, shuffle=True)

val_loader = DataLoader(
    val_data_list, batch_size=batchsize, shuffle=True)

pre_loader_list = [DataLoader(pre, batch_size=1, shuffle=False) for pre in pre_data_list]
del train_data_list, val_data_list, pre_data_list
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(300, 3).to(device)


def train(loader,first):
    model.train()
    total_loss = []
    for data in loader:
        if first:
            for i in range(20000):
                optimizer.zero_grad()
                out = model(data.x.to(device), data.edge_index.to(device))
                loss = F.mse_loss(out, data.y.to(device), size_average=False)
                loss.backward()
                optimizer.step()
                tmp_out = torch.from_numpy(y_scaler.inverse_transform(out.cpu().detach().numpy())).to(torch.float)
                tmp_y = torch.from_numpy(y_scaler.inverse_transform(data.y.cpu().detach().numpy())).to(torch.float)
                loss_list = [F.mse_loss(tmp_out[:, i], tmp_y[:, i]).item() for i in range(3)]
                print(f'{i}\t{loss_list[0] ** 0.5}\t{loss_list[1] ** 0.5}\t{loss_list[2] ** 0.5}')
                if loss_list[0] ** 0.5 < 20 and loss_list[1] ** 0.5 < 20 and loss_list[2] ** 0.5 < 20:
                    return
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = F.mse_loss(out, data.y.to(device), size_average=False)
        loss.backward()
        optimizer.step()
        tmp_out = torch.from_numpy(y_scaler.inverse_transform(out.cpu().detach().numpy())).to(torch.float)
        tmp_y = torch.from_numpy(y_scaler.inverse_transform(data.y.cpu().detach().numpy())).to(torch.float)
        loss_list = [F.mse_loss(tmp_out[:, i], tmp_y[:, i]).item() for i in range(3)]
        total_loss.append(loss_list)
    # total_loss = y_scaler.inverse_transform(total_loss)
    total_loss = np.array(total_loss)
    total_loss = [(total_loss[:, i].sum() / len(total_loss)) ** 0.5 for i in range(3)]
    return total_loss


def test(loader):
    model.eval()

    total_loss = []
    for data in loader:
        out = model(data.x.to(device), data.edge_index.to(device))
        tmp_out = torch.from_numpy(y_scaler.inverse_transform(out.cpu().detach().numpy())).to(torch.float)
        loss_list = [F.mse_loss(tmp_out[:, i], data.y[:, i]).item() for i in range(3)]
        total_loss.append(loss_list)
    total_loss = np.array(total_loss)
    total_loss = [(total_loss[:, i].sum() / len(total_loss)) ** 0.5 for i in range(3)]
    return total_loss


def predict(loader_list):
    model.eval()
    result = []

    lo = []
    for i in range(len(loader_list)):
        tmp = []
        total_loss = []
        for data in loader_list[i]:
            out = model(data.x.to(device), data.edge_index.to(device))
            out = y_scaler.inverse_transform(out.cpu().detach().numpy())
            loss = F.mse_loss(torch.from_numpy(out[:, i]).to(torch.float), data.y).item()
            total_loss.append(loss)
            tmp.append(np.reshape(out[:, i], (-1, 1)))
        cat_tmp = np.concatenate(tmp, 1)
        result.append(cat_tmp)
        total_loss = np.array(total_loss)
        total_loss = (total_loss.sum() / len(total_loss)) ** 0.5
        lo.append(total_loss)
    return result, lo


def write_out(result):
    df = pd.DataFrame(result[0])
    df.to_csv('predict_out_15.csv', index=False, header=None)
    df = pd.DataFrame(result[1])
    df.to_csv('predict_out_30.csv', index=False, header=None)
    df = pd.DataFrame(result[2])
    df.to_csv('predict_out_45.csv', index=False, header=None)


optimizer = torch.optim.Adam(model.parameters())
loss = train(train_loader,True)
optimizer = torch.optim.Adam(model.parameters())
sgd_lr = 1e-5
for epoch in range(1, 10001):
    loss = train(train_loader,False)
    val_loss = test(val_loader)
    print(f'Epoch:{epoch}\nTrain:{loss[0]}\t{loss[1]}\t{loss[2]}\nTest:{val_loss[0]}\t{val_loss[1]}\t{val_loss[2]}')
    # if not epoch % 10:
    ea(sum(val_loss) / len(val_loss), model)
    if ea.early_stop:
        print('early stop!')
        ea = EarlyStopping(10,True)
        model.load_state_dict(torch.load('checkpoint.pt'))
        optimizer = torch.optim.SGD(model.parameters(),lr=sgd_lr)
        sgd_lr /= 2
        continue
print('predicting...')
re, lo = predict(pre_loader_list)
print(f'predict RMSE:{lo[0]}\t{lo[1]}\t{lo[2]}')
write_out(re)
