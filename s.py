# -*- coding: UTF-8 -*-
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

actor_edge = pd.read_csv('network_actor_rankavgbox.csv') #导入演员网络边的csv文件,导入后为Dataframe格式
print(actor_edge.head(3)) #显示表的前3行

weight_edge = []
for _,row in actor_edge.iterrows(): #把边及边的权重加入列表，数据格式为（节点，节点，权重）
    weight_edge.append((row['ActorID_1'],row['ActorID_2'],row['rankaveragebox']))

AW = nx.Graph() #初始化无向图
AW.add_weighted_edges_from(weight_edge) #把带权重边的信息加入无向图中

degree_hist = nx.degree_histogram(AW) #返回图中所有节点的度分布序列

x = range(len(degree_hist)) #生成x轴序列
y = [z / float(sum(degree_hist)) for z in degree_hist] #生产y轴序列，将频次转换为频率
plt.loglog(x,y,color="blue",linewidth=2) #在双对数坐标轴上绘制度分布曲线
plt.title('Degree Distribution Actor ') #图表标题
plt.xlabel('Degree') #x轴标题
plt.ylabel('Probability') #y轴标题
plt.savefig('Degree Distribution Actor.png') #保存图片
plt.show() #显示图表