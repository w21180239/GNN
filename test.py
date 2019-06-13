import os

f_list = os.listdir('./')
hh=[]
for i in f_list:
    if os.path.splitext(i)[1] == '.csv':
        hh.append(open(i))
xx=1