import networkx as nx
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, RobustScaler,MinMaxScaler, StandardScaler
from torch_geometric.data import NeighborSampler,Data
from torch_geometric.utils import remove_self_loops
import torch
import os
import random


windows = 21     # n-1个输入 1个输出

df = pd.DataFrame(pd.read_csv('backup/game_friend.csv'))
G = nx.Graph()
G.add_edges_from(df.values)
df = pd.DataFrame(pd.read_csv('backup/plat_friend.csv'))
G.add_edges_from(df.values)
nx.write_gpickle(G,'my_graph.gpickle')
exit(0)

# G = nx.Graph()
# labels = range(81)
# data_df = pd.DataFrame(pd.read_csv('Subway_net.csv',header=None))
# DF_adj = pd.DataFrame(data_df.values,index=labels,columns=labels)
# #Network graph
# G = nx.Graph()
# G.add_nodes_from(labels)
#
# #Connect nodes
# for i in range(DF_adj.shape[0]):
#     col_label = DF_adj.columns[i]
#     for j in range(DF_adj.shape[1]):
#         row_label = DF_adj.index[j]
#         node = DF_adj.iloc[i,j]
#         if node == 1:
#             G.add_edge(col_label,row_label)
#
#
# nx.write_gpickle(G,'subway_g.gpickle')
G = nx.read_gpickle('subway_g.gpickle')
df = pd.DataFrame(pd.read_csv('Subway_instation_data_01.csv',header=None))

G = nx.read_gpickle('subway_g.gpickle')

data_df = pd.DataFrame(pd.read_csv('Subway_instation_data_01.csv', header=None))
data = data_df.values
x_list = [data[i:i+windows,:] for i in range(data.shape[0]-windows+1)]
y_list = [x[windows-1] for x in x_list]
y_list = list(StandardScaler().fit_transform(y_list))
x_list = [x[:windows-1] for x in x_list]
x_list = [np.swapaxes(x,0,1)  for x in x_list]

randnum = random.randint(0,100)
random.seed(randnum)
random.shuffle(x_list)
random.seed(randnum)
random.shuffle(y_list)
# train_mask = np.array([0 for i in range(81)])
# val_mask = np.array([0 for i in range(81)])
# test_mask = np.array([0 for i in range(81)])
# index = list(range(81))
# random.shuffle(index)
# for i in range(0, int(len(index) * 0.8)):
#     train_mask[index[i]] = 1
# for i in range(int(len(index) * 0.8), int(len(index) * 0.9)):
#     val_mask[index[i]] = 1
# for i in range(int(len(index) * 0.9), len(index)):
#     test_mask[index[i]] = 1

edge_index = G.edges()
edge_index = torch.tensor(edge_index).t().contiguous()
edge_index = edge_index - edge_index.min()
edge_index, _ = remove_self_loops(edge_index)

for i,x in enumerate(x_list):
    x_list[i] = torch.from_numpy(x).to(torch.float)
for i,y in enumerate(y_list):
    y_list[i] = torch.from_numpy(y).to(torch.float).unsqueeze(1)
# train_mask =  torch.from_numpy(train_mask).to(torch.uint8)
# val_mask =  torch.from_numpy(val_mask).to(torch.uint8)
# test_mask =  torch.from_numpy(test_mask).to(torch.uint8)


train_list = []
val_list = []
i=0
total = len(x_list)
for x,y in zip(x_list,y_list):
    data = Data(edge_index=edge_index, x=x, y=y)
    if i>int(total*0.8):
        val_list.append(data)
    else:
        train_list.append(data)
    i+=1
torch.save(train_list, 'subway_data_train.npz')
torch.save(val_list, 'subway_data_val.npz')



