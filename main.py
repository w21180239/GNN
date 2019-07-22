import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv

from pytorchtools import EarlyStopping

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
batchsize = 100
torch.cuda.set_device(0)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, 64, heads=1, concat=True, dropout=0.6)
        self.cla_15 = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
        self.cla_30 = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
        self.cla_45 = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x_list = [self.cla_15(x), self.cla_30(x), self.cla_45(x)]
        x = torch.cat(x_list, 1)
        return x


ea = EarlyStopping(verbose=True)
train_data_list = torch.load('subway_data_train.npz')
val_data_list = torch.load('subway_data_val.npz')
pre_data_list = []
pre_data_list.append(torch.load('subway_data_pre_15.npz'))
pre_data_list.append(torch.load('subway_data_pre_30.npz'))
pre_data_list.append(torch.load('subway_data_pre_45.npz'))
x_scaler = RobustScaler()
y_scaler = StandardScaler()
y = torch.cat([data.y for data in train_data_list], 0)
x = torch.cat([data.x for data in train_data_list], 0)
x_scaler.fit(x)
y_scaler.fit(y)
for data in train_data_list:
    x = x_scaler.transform(data.x)
    data.x = torch.from_numpy(x).to(torch.float)
    y = y_scaler.transform(data.y)
    data.y = torch.from_numpy(y).to(torch.float)
for data in val_data_list:
    x = x_scaler.transform(data.x)
    data.x = torch.from_numpy(x).to(torch.float)
    y = y_scaler.transform(data.y)
    data.y = torch.from_numpy(y).to(torch.float)
for pre in pre_data_list:
    for data in pre:
        x = x_scaler.transform(data.x)
        data.x = torch.from_numpy(x).to(torch.float)
        data.y = data.y.squeeze()
        # y = y_scaler.transform(data.y.expand(81, 3))
        # data.y = torch.from_numpy(y[:, 0]).to(torch.float)
train_loader = DataLoader(
    train_data_list, batch_size=batchsize, shuffle=True)

val_loader = DataLoader(
    val_data_list, batch_size=batchsize, shuffle=True)

pre_loader_list = [DataLoader(pre, batch_size=batchsize, shuffle=False) for pre in pre_data_list]
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
        tmp_y = torch.from_numpy(y_scaler.inverse_transform(data.y.cpu().detach().numpy())).to(torch.float)
        loss_list = [F.mse_loss(tmp_out[:, i], tmp_y[:, i]).item() for i in range(3)]
        total_loss.append(loss_list)
    # total_loss = y_scaler.inverse_transform(total_loss)
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
            tmp.append(out[:, i])
        cat_tmp = np.concatenate(tmp, 0)
        result.append(cat_tmp)
        total_loss = np.array(total_loss)
        total_loss = (total_loss.sum() / len(total_loss)) ** 0.5
        lo.append(total_loss)
    return result, lo


def write_out(result):
    df = pd.DataFrame({'15min': result[0], '30min': result[1], '45min': result[2]})
    df.to_csv('predict_out.csv', index=False, )


for epoch in range(1, 10001):
    loss = train(train_loader)
    val_loss = test(val_loader)
    print(f'Epoch:{epoch}\nTrain:{loss[0]}\t{loss[1]}\t{loss[2]}\nTest:{val_loss[0]}\t{val_loss[1]}\t{val_loss[2]}')
    print('predicting...')
    re, lo = predict(pre_loader_list)
    print(f'predict RMSE:{lo[0]}\t{lo[1]}\t{lo[2]}')
    if not epoch % 10:
        ea(sum(val_loss) / len(val_loss), model)
        if ea.early_stop:
            print('early stop!')
            break
print('predicting...')
re, lo = predict(pre_loader_list)
print(f'predict RMSE:{lo[0]}\t{lo[1]}\t{lo[2]}')
write_out(re)
