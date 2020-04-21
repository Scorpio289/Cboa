import community_louvain
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
actor_edge = pd.read_csv('2.csv') #导入演员网络边的csv文件,导入后为Dataframe格式
print(actor_edge.head(3)) #显示表的前3行

weight_edge = []
for _,row in actor_edge.iterrows(): #把边及边的权重加入列表，数据格式为（节点，节点，权重）
    weight_edge.append((row['A'],row['B'],row['rankaveragebox']))

G = nx.Graph() #初始化无向图
G.add_weighted_edges_from(weight_edge) #把带权重边的信息加入无向图中

#first compute the best partition
partition = community_louvain.best_partition(G)
print("0")
#drawing
size = float(len(set(partition.values())))
print("0")
pos = nx.spring_layout(G)

count = 0.
for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))


nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()
