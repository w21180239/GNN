import networkx as nx
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt

G = nx.Graph()
# G.add_nodes_from([i for i in range(1151372)])
# nx.write_gml(G,'hh.gml')
# G = nx.read_gml('hh.gml')
# G.add_nodes_from([i for i in range(115)])

data_df = pd.DataFrame(pd.read_csv('new_playerCharge-4.csv'))
jj = len(data_df)
hh = max(data_df['openid'])
exit(0)
# im = Imputer()
# val = im.fit_transform(data_df.values)
# np.save('feature.npy',val)
G = nx.read_gpickle('my_graph.gpickle')
nx.draw(G)
plt.show()
G.add_nodes_from(data_df.values[:,0].astype(int))
df = pd.DataFrame(pd.read_csv('game_friend.csv'))
G.add_edges_from(df.values.astype(int))
df = pd.DataFrame(pd.read_csv('plat_friend.csv'))
G.add_edges_from(df.values.astype(int))
nx.write_gpickle(G, "my_graph.gpickle")
exit(0)



