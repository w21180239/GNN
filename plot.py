import matplotlib.pyplot as plt
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

# # 嵌入尺寸图
x = [8, 16, 32, 64, 128, 256]
x = [f'{i}维' for i in x]
yy = [[0.9037983069827061, 0.9057966578931261, 0.9113164271646101, 0.9226152112310922, 0.9251680595112501,
       0.9181792322731873],
      [0.8772370486656201, 0.9149764521193091, 0.9189711387513585, 0.9291969568892645, 0.9419876826470233,
       0.9338171718391498],
      [0.8817282762221583, 0.9107759228094983, 0.9188146654947933, 0.9190827050398155, 0.9301419453938539,
       0.9078514662806763],
      [0.8970014077761279, 0.909650497954473, 0.9253041036274401, 0.9239802707843209, 0.9335488967817379,
       0.9108458721343908]]
color = ['red', 'blue', 'lime', 'yellow']
dataset = ['Cora', 'CiteSeer', 'PubMed', 'Reddit-s']
for y, c, d in zip(yy, color, dataset):
    plt.plot(x, y, label=d, color=c, linewidth=2)
plt.xlabel("嵌入尺寸")
plt.ylabel("ROC")
plt.title("嵌入尺寸与ROC")
plt.legend()
plt.savefig(fname='test.svg', format='svg')
plt.show()

# # AUC图
# color = ['red','blue','silver','yellow','lime']
# dataset = ['Cora','CiteSeer','Pubmed','Reddit-s']
# model_list = ['SC','DW','node2vec','LINE','GATAE']
# re_list = [[84.6,80.5,84.2,83.2],
#            [83.1,80.5,84.4,83.4],
#            [83.3,80.4,85.6,84.1],
#            [83.1,81.1,85.4,85.3],
#            [92.5,94.2,93.1,93.4]]
# x = [0,10,20,30]
# total_width = 5
# width = total_width / len(dataset)
# plt.ylim(80,100)
# for co,model,re in zip(color,model_list,re_list):
#     # re = [i/100 for i in re]
#     plt.bar(x, re, width=width, label=model, fc=co)
#     x = [i + width for i in x]
# plt.xticks([3,13,23,33],dataset)
# plt.legend()
# plt.xlabel('数据集')
# plt.ylabel('AUC')
# plt.title('模型AUC')
# plt.savefig(fname='test.svg',format='svg')
# plt.show()


# # AP图
# color = ['red','blue','silver','yellow','lime']
# dataset = ['Cora','CiteSeer','Pubmed','Reddit-s']
# model_list = ['SC','DW','node2vec','LINE','GATAE']
# re_list = [[88.5,85.0,87.8,84.3],
#            [85.0,83.6,84.1,83.4],
#            [85.2,82.8,88.1,84.1],
#            [84.5,82.5,85.7,85.4],
#            [93.1,94.8,93.3,94.1]]
# x = [0,10,20,30]
# total_width = 5
# width = total_width / len(dataset)
# plt.ylim(80,100)
# for co,model,re in zip(color,model_list,re_list):
#     # re = [i/100 for i in re]
#     plt.bar(x, re, width=width, label=model, fc=co)
#     x = [i + width for i in x]
# plt.xticks([3,13,23,33],dataset)
# plt.legend()
# plt.xlabel('数据集')
# plt.ylabel('AUC')
# plt.title('模型AP')
# plt.savefig(fname='test.svg',format='svg')
# plt.show()
