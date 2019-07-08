import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import os
import pickle
import pandas as pd
from torch_geometric.data import NeighborSampler,Data
from torch_geometric.nn import SAGEConv, GATConv,AGNNConv,ARMAConv,SplineConv
from torch_geometric.utils import remove_self_loops
from sklearn.preprocessing import Imputer, RobustScaler,StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error,accuracy_score
import random
from keras.utils import to_categorical

FEA = 67
OUT = 2
baseline = 180


num_node = 10000
rerun = False
sk = False


mmodel = 'ARMA'

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_data():
    if os.path.exists('data.npz') and not rerun:
        data = torch.load('data.npz')
    else:
        G = nx.read_gpickle('my_graph.gpickle')
        seed = random.randint(0, 100)
        if os.path.exists('sur_x.npy') and not rerun:
            x = np.load('sur_x.npy')
            y = np.load('sur_y.npy')
            data_df = pd.DataFrame(pd.read_csv('new_playerCharge-4.csv'))
            id_list = list(data_df['openid'])
            random.shuffle(id_list)
            id_list = np.array(id_list[:num_node])
        else:
            data_df = pd.DataFrame(pd.read_csv('new_playerCharge-4.csv'))
            id_list = list(data_df['openid'])
            random.seed(seed)
            random.shuffle(id_list)
            id_list = np.array(id_list[:num_node])
            y = list(data_df['charge'])
            random.seed(seed)
            random.shuffle(y)
            y = np.array(y[:num_node])
            y[y <= baseline] = 0
            y[y > baseline] = 1
            np.save('ori_y.npy', y)
            # y = to_categorical(y)
            G = G.subgraph(id_list)
            feature = load_obj('feature')
            x = np.array([feature[i] for i in id_list])
            x = Imputer().fit_transform(x)
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
        ori_key = id_list
        # ori_key.sort()
        projection = {}
        for i in range(len(id_list)):
            projection[ori_key[i]] = i
        edge_index = G.edges()
        edge_index = [[projection[i[0]],projection[i[1]]] for i in edge_index]
        # edge_index = []
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_index = edge_index - edge_index.min()
        edge_index, _ = remove_self_loops(edge_index)

        # x = RobustScaler().fit_transform(x)
        # x = torch.from_numpy(x).to(torch.float)
        # y = y.reshape(-1, 1)
        # st = StandardScaler()
        # y = st.fit_transform(y)           # transform
        y = torch.LongTensor(y)
        train_mask =  torch.from_numpy(train_mask).to(torch.uint8)
        val_mask =  torch.from_numpy(val_mask).to(torch.uint8)
        test_mask =  torch.from_numpy(test_mask).to(torch.uint8)

        data = Data(edge_index=edge_index, x=x, y=y, train_mask=train_mask, val_mask=val_mask,
                    test_mask=test_mask)
        torch.save(data, 'data.npz')
    return data

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
        self.breadth_func = Breadth(in_dim, 256)
        self.depth_func = Depth(256, 256)

    def forward(self, x, edge_index, h, c):
        x = self.breadth_func(x, edge_index)
        x = x[None, :]
        x, (h, c) = self.depth_func(x, h, c)
        x = x[0]
        return x, (h, c)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if mmodel == 'GAT':
            self.conv1 = GATConv(FEA, 8, heads=10, dropout=0.5)
            self.conv2 = GATConv(
                8 * 10, OUT, heads=1, concat=True, dropout=0.5)
        elif mmodel == 'AGNN':
            self.lin1 = torch.nn.Linear(FEA, 64)
            self.prop1 = AGNNConv(requires_grad=False)
            self.prop2 = AGNNConv(requires_grad=True)
            self.lin2 = torch.nn.Linear(64, OUT)
        elif mmodel == 'ARMA':
            self.conv1 = ARMAConv(
                FEA,
                16,
                num_stacks=3,
                num_layers=2,
                shared_weights=True,
                dropout=0.25)

            self.conv2 = ARMAConv(
                16,
                OUT,
                num_stacks=3,
                num_layers=2,
                shared_weights=True,
                dropout=0.25,
                act=None)
        elif mmodel == 'Spline':
            self.conv1 = SplineConv(FEA, 16, dim=1, kernel_size=2)
            self.conv2 = SplineConv(16, OUT, dim=1, kernel_size=2)
        elif mmodel == 'Genie':
            self.lin1 = torch.nn.Linear(FEA, 256)
            self.gplayers = torch.nn.ModuleList(
                [GeniePathLayer(256) for i in range(4)])
            self.lin2 = torch.nn.Linear(256, OUT)

    def forward(self):
        if mmodel == 'GAT':
            x = F.dropout(data.x, p=0.5, training=self.training)
            x = F.elu(self.conv1(x, data.edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, data.edge_index)
        elif mmodel == 'AGNN':
            x = F.dropout(data.x, training=self.training)
            x = F.relu(self.lin1(x))
            x = self.prop1(x, data.edge_index)
            x = self.prop2(x, data.edge_index)
            x = F.dropout(x, training=self.training)
            x = self.lin2(x)
        elif mmodel == 'ARMA':
            x, edge_index = data.x, data.edge_index
            x = F.dropout(x, training=self.training)
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
        elif mmodel == 'Spline':
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            x = F.dropout(x, training=self.training)
            x = F.elu(self.conv1(x, edge_index, edge_attr))
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index, edge_attr)
        elif mmodel == 'Genie':
            x, edge_index = data.x, data.edge_index
            x = self.lin1(x)
            h = torch.zeros(1, x.shape[0], 256, device=x.device)
            c = torch.zeros(1, x.shape[0], 256, device=x.device)
            for i, l in enumerate(self.gplayers):
                x, (h, c) = self.gplayers[i](x, edge_index, h, c)
            x = self.lin2(x)
        return F.log_softmax(x,dim=1)


data =get_data()

x_tran = RobustScaler()
data.x = torch.from_numpy(x_tran.fit_transform(data.x)).to(torch.float)
# data.y = torch.from_numpy(y_tran.fit_transform(data.y)).to(torch.float)


if sk:
    train_x = data.x[data.train_mask].numpy()
    train_y = data.y[data.train_mask].numpy()

    val_x = data.x[data.val_mask].numpy()
    val_y = data.y[data.val_mask].numpy()

    test_x = data.x[data.test_mask].numpy()
    test_y = data.y[data.test_mask].numpy()

    sk_model = XGBClassifier(tree_method='gpu_hist')
    sk_model.fit(train_x, train_y)
    print(accuracy_score(sk_model.predict(val_x), val_y))
    print(accuracy_score(sk_model.predict(test_x), test_y))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    # pre_y = torch.from_numpy(y_tran.inverse_transform(logits.cpu().detach())).to(torch.float)
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 100001):
    train()
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, *test()))