from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.distance import great_circle

def dis2we_for_bay(path_data):
    id_list = [773869,767541,767542,717447,717446,717445,773062,767620,737529,717816,765604,767471,716339,773906,765273,716331,771667,716337,769953,769402,769403,769819,769405,716941,717578,716960,717804,767572,767573,773012,773013,764424,769388,716328,717819,769941,760987,718204,718045,769418,768066,772140,773927,760024,774012,774011,767609,769359,760650,716956,769831,761604,717495,716554,773953,767470,716955,764949,773954,767366,769444,773939,774067,769443,767750,767751,767610,773880,764766,717497,717490,717491,717492,717493,765176,717498,717499,765171,718064,718066,765164,769431,769430,717610,767053,767621,772596,772597,767350,767351,716571,773023,767585,773024,717483,718379,717481,717480,717486,764120,772151,718371,717489,717488,717818,718076,718072,767455,767454,761599,717099,773916,716968,769467,717576,717573,717572,717571,717570,764760,718089,769847,717608,767523,716942,718090,769867,717472,717473,759591,764781,765099,762329,716953,716951,767509,765182,769358,772513,716958,718496,769346,773904,718499,764853,761003,717502,759602,717504,763995,717508,765265,773996,773995,717469,717468,764106,717465,764794,717466,717461,717460,717463,717462,769345,716943,772669,717582,717583,717580,716949,717587,772178,717585,716939,768469,764101,767554,773975,773974,717510,717513,717825,767495,767494,717821,717823,717458,717459,769926,764858,717450,717452,717453,759772,717456,771673,772167,769372,774204,769806,717590,717592,717595,772168,718141,769373]
    data = pd.read_csv(path_data)
    # print(data['from'])
    data_from = data[data['from'].isin(id_list)]
    data_from_to = data_from[data_from['to'].isin(id_list)]
    print(data_from_to)
    # print(id_list[0])
    return data_from_to

def local2dis(data):
    ID = np.array(data['ID'])
    dis = []
    fro = []
    to = []
    for _ in range(ID.shape[0]):
        for __ in range(ID.shape[0]):
            f = (data['l'][_],data['a'][_])
            t = (data['l'][__],data['a'][__])
            g_dis =great_circle(f, t).kilometers
            print(g_dis*1000)
            dis.append(g_dis)
            fro.append(_)
            to.append(__)
    fro = np.asarray(fro).reshape(len(fro),1)
    to = np.asarray(to).reshape(len(to),1)
    dis = np.asarray(dis).reshape(len(dis),1)
    data = np.hstack((fro,to))
    data = np.hstack((data,dis))
    return data


def darw_hit_map(data):
    sns.set()
    # Load the example flights dataset and conver to long-form
    # flights_long = sns.load_dataset("flights",)
    # flights_long = pd.read_csv(path,header=None,names=['From SensorID','To SensorID','Distance'])

    # print(flights_long)
    flights = data.pivot("from", "to", "cost")
    #
    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(flights, annot=True, fmt="f", linewidths=.5, ax=ax)
    plt.show()

def darw_hit_map_(path):
    sns.set(style="dark")
    # Generate a large random dataset
    rs = np.random.RandomState(33)
    d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                     columns=list(ascii_letters[26:]))
    print(d)

    # Compute the correlation matrix
    corr = d.corr()
    print(corr)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(0, 5, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

def use_fit(data):
    dis  = data['cost']
    dis = np.array(dis)
    std = dis.std()
    dis = np.exp(-np.square(dis / std))
    normalized_k = 0.1
    dis[dis < normalized_k] = 0
    data['cost'] = dis
    return data

def darw_my_heat(data):
    max = data['cost'].max()
    min = data['cost'].min()
    data = data[~data['cost'].isin([0.0])]
    data = data.pivot("from", "to", "cost")
    ax = sns.heatmap(data,vmin=min,vmax=max,xticklabels='auto',yticklabels='auto')
    plt.show()


if __name__ == '__main__':
    # path = 'PeMSD7_W_228.csv'
    # data_path = 'distances_la_2012.csv'
    data = pd.read_csv('stationID.csv')
    # data = dis2we_for_bay(data_path)

    data = local2dis(data)
    data = pd.DataFrame(data,columns=['from', 'to', 'cost'])
    darw_my_heat(data)
    # print(data)
    data = use_fit(data)
    # darw_hit_map(data)
    darw_my_heat(data)