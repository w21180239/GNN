import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import os
import pickle
import pandas as pd
from torch_geometric.data import NeighborSampler,Data
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.utils import remove_self_loops
from sklearn.preprocessing import Imputer, RobustScaler,StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import random

num_node = 10000
rerun = True
def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
if os.path.exists('data.npz') and not rerun:
    data = torch.load('data.npz')
else:
    G = nx.read_gpickle('my_graph.gpickle')
    seed = random.randint(0, 100)
    random.seed(seed)
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
        random.shuffle(id_list)
        id_list = np.array(id_list[:num_node])
        y = np.array(data_df['charge'][:num_node])
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
    y = y.reshape(-1, 1)
    # st = StandardScaler()
    # y = st.fit_transform(y)           # transform
    y = torch.from_numpy(y).to(torch.float)
    train_mask =  torch.from_numpy(train_mask).to(torch.uint8)
    val_mask =  torch.from_numpy(val_mask).to(torch.uint8)
    test_mask =  torch.from_numpy(test_mask).to(torch.uint8)

    data = Data(edge_index=edge_index, x=x, y=y, train_mask=train_mask, val_mask=val_mask,
                test_mask=test_mask)
    torch.save(data, 'data.npz')




class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(67, 8, heads=10, dropout=0.5)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(
            8 * 10, 1, heads=1, concat=True, dropout=0.5)

    def forward(self):
        x = F.dropout(data.x, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)
        return x


y_tran = StandardScaler()
x_tran = RobustScaler()
ori_y = data.y
data.x = torch.from_numpy(x_tran.fit_transform(data.x)).to(torch.float)
data.y = torch.from_numpy(y_tran.fit_transform(data.y)).to(torch.float)


train_x = data.x[data.train_mask].numpy()
train_y = data.y[data.train_mask].numpy()

val_x = data.x[data.val_mask].numpy()
val_y = data.y[data.val_mask].numpy()

test_x = data.x[data.test_mask].numpy()
test_y = data.y[data.test_mask].numpy()

sk_model = XGBRegressor(tree_method='gpu_hist')
sk_model.fit(train_x,train_y)
print(mean_squared_error(y_tran.inverse_transform(sk_model.predict(val_x)),y_tran.inverse_transform(val_y)))
print(mean_squared_error(y_tran.inverse_transform(sk_model.predict(test_x)),y_tran.inverse_transform(test_y)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    F.mse_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    pre_y = torch.from_numpy(y_tran.inverse_transform(logits.cpu().detach())).to(torch.float)
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        acc = F.mse_loss(pre_y[mask], ori_y[mask])
        accs.append(acc)
    return accs


for epoch in range(1, 100001):
    train()
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, *test()))