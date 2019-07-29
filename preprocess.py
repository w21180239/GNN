import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import keras



station_num = 207
base_windows = 12
windows = base_windows + 9  # n-1个输入 1个输出

DF_adj = pd.DataFrame(pd.read_csv('METR_LA_stand_w8.csv', header=None))
DF_adj[DF_adj > 0] = 1
G = nx.Graph()
labels = range(station_num)
#Network graph
G = nx.Graph()
G.add_nodes_from(labels)

#Connect nodes
for i in range(DF_adj.shape[0]):
    col_label = DF_adj.columns[i]
    for j in range(DF_adj.shape[1]):
        row_label = DF_adj.index[j]
        node = DF_adj.iloc[i,j]
        if node == 1:
            G.add_edge(col_label,row_label)

nx.draw(G, with_labels=True, pos=nx.spring_layout(G), node_size=150)
plt.show()
nx.write_gpickle(G,'subway_g.gpickle')

G = nx.read_gpickle('subway_g.gpickle')

data_df = pd.DataFrame(pd.read_csv('METR_LA.csv', header=None))
data = data_df.values

x_list = [data[i:i + windows, :] for i in range(data.shape[0] - windows + 1 - 3 * 288)]
y_list = [[x[windows - 7] for x in x_list], [x[windows - 4] for x in x_list], [x[windows - 1] for x in x_list]]
y_list = [np.array([y_list[0][i], y_list[1][i], y_list[2][i]]) for i in range(len(y_list[0]))]
pre_win = [windows-6, windows-3, windows]
pre_x_list = []
pre_y_list = []
start_time = [(len(data) - 3 * 288 - win + 1) % 288 for win in pre_win]
for win in pre_win:
    hh = [data[i:i + win, :] for i in range(len(data) - 3 * 288 - win + 1, data.shape[0] - win + 1)]
    yy = [x[len(x) - 1, :] for x in hh]
    hh = [x[:base_windows] for x in hh]
    pre_x_list.append(hh)
    pre_y_list.append(yy)



x_list = [x[:windows - 9] for x in x_list]
x_list = [np.swapaxes(x, 0, 1) for x in x_list]
y_list = [np.swapaxes(y, 0, 1) for y in y_list]

ready = []
for i in range(len(x_list)):
    ready.extend([i % 288 for j in range(station_num)])
time_stamp = keras.utils.to_categorical(ready, num_classes=None, dtype='float')
for i in range(len(x_list)):
    hh = x_list[i]
    jj = time_stamp[i*station_num:(i+1)*station_num,:]
    x_list[i] = np.concatenate([hh,jj],1)


for i in range(3):
    pre_x_list[i] = [np.swapaxes(x, 0, 1) for x in pre_x_list[i]]
    pre_y_list[i] = [np.reshape(y, (station_num, -1)) for y in pre_y_list[i]]


for j in range(3):
    ready = []
    for i in range(len(pre_x_list[0])):
        ready.extend([(i+start_time[j]) % 288 for z in range(station_num)])
    time_stamp = keras.utils.to_categorical(ready, num_classes=None, dtype='float')
    for i in range(len(pre_x_list[0])):
        hh = pre_x_list[j][i]
        jj = time_stamp[i*station_num:(i+1)*station_num,:]
        pre_x_list[j][i] = np.concatenate([hh,jj],1)

tmp = np.array(pre_x_list[0])



edge_index = list(G.edges())
edge_index = torch.tensor(edge_index).t().contiguous()
edge_index = edge_index - edge_index.min()
# edge_index, _ = remove_self_loops(edge_index)

for i, x in enumerate(x_list):
    x_list[i] = torch.from_numpy(x).to(torch.float)
for i, y in enumerate(y_list):
    y_list[i] = torch.from_numpy(y).to(torch.float)

for j in range(3):
    for i, x in enumerate(pre_x_list[j]):
        pre_x_list[j][i] = torch.from_numpy(x).to(torch.float)
    for i, y in enumerate(pre_y_list[j]):
        pre_y_list[j][i] = torch.from_numpy(y).to(torch.float)



train_list = []
val_list = []
pre_list = [[Data(edge_index=edge_index, x=x, y=y) for x, y in zip(pre_x_list[j], pre_y_list[j])] for j in range(3)]
i = 0
total = len(x_list)
for x, y in zip(x_list, y_list):
    data = Data(edge_index=edge_index, x=x, y=y)
    if i > int(total * 0.8):
        val_list.append(data)
    else:
        train_list.append(data)
    i += 1

torch.save(train_list, 'subway_data_train.npz')
torch.save(val_list, 'subway_data_val.npz')
torch.save(pre_list[0], 'subway_data_pre_15.npz')
torch.save(pre_list[1], 'subway_data_pre_30.npz')
torch.save(pre_list[2], 'subway_data_pre_45.npz')
