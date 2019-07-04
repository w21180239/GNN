import networkx as nx
import pandas as pd
import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

df =pd.DataFrame(pd.read_csv('backup/whole_feature.csv'))
hh = df.values
dd={}
total = len(hh)
for i in range(len(hh)):
    dd[int(hh[i][0])]=hh[i][2:]
    if i%1000 == 0:
        print(f'{i}/{total}')
save_obj(dd,'feature')
