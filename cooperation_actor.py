import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import completeness_score, homogeneity_score
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from pylab import figure, subplot, hist, xlim, show,plot
from numpy import genfromtxt, zeros
from sklearn.metrics import confusion_matrix
from numpy import mean
from sklearn.cross_validation import cross_val_score
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pymysql as pm
import re
from itertools import  combinations
import community_louvain
import networkx as nx



#连接数据库
cnx = pm.connect(user='root',password='260877289',host='127.0.0.1',database='moviedata')

cur1 = cnx.cursor()
cur1.execute("select MovieID from movie")

cur2 = cnx.cursor()
cur2.execute("select ActorID from actor")

cur3=cnx.cursor()

cur4=cnx.cursor()
cur4.execute("select MovieSumBoxOffice from movie")

movieid=cur1.fetchone()
actorid = cur2.fetchone()
movieboxoffice=cur4.fetchone()

actor_one = [0 for i in range(15040)]
for i in range(len(actor_one)):
    actorid = cur2.fetchone()
    if actorid is not None:
        actor_one[i]=int(re.sub("\D", "",list(actorid)[0]))#正则
print (actor_one)

actor_tow = [[0 for i in range(15040)] for i in range(15040)]
actor_tow_pf=[[0 for i in range(15040)] for i in range(15040)]

movieid=cur1.fetchone()
movieboxoffice=cur4.fetchone()
while movieboxoffice is not None and movieid is not None:
    if re.sub("\D", "",list(movieboxoffice)[0]) !="":
        movie_int_boxoffice=int(re.sub("\D", "",list(movieboxoffice)[0]))
    else:
        movie_int_boxoffice=0
    movie_int_ID=int(re.sub("\D", "",list(movieid)[0]))
    sqlstring1="select ActorID from movie_actor where MovieID ='"+str(movie_int_ID)+"'"
    cur3.execute(sqlstring1)
    actorid2=cur3.fetchone()
    actor_cooperation_id=[]
    while actorid2 is not None:
        actor_cooperation_id.append(int(re.sub("\D", "",list(actorid2)[0])))#正则
        actorid2=cur3.fetchone()
    #print(actor_cooperation_id)
    actor_coo_combinations=list(combinations(actor_cooperation_id,2))
    #print(actor_coo_combinations)
    for i in range(len(actor_coo_combinations)):
        x=actor_one.index(actor_coo_combinations[i][0])
        y=actor_one.index(actor_coo_combinations[i][1])
        actor_tow[min(x,y)][max(x,y)]+=1
        actor_tow_pf[min(x,y)][max(x,y)]+=movie_int_boxoffice
    movieid=cur1.fetchone()
    movieboxoffice=cur4.fetchone()
print(actor_tow_pf[1])
print(actor_tow[1])


data1 = pd.DataFrame(actor_tow)
data1.to_csv('coo_times.csv',na_rep='NA',header=0,index=0)

data2=pd.DataFrame(actor_tow_pf)
data2.to_csv('coo_boxoffice.csv',na_rep='NA',header=0,index=0)


print("k")
row_num=0
actor_coo_times_arr=[[0 for i in range(3)] for i in range(201494)]
actor_coo_office_arr=[[0 for i in range(3)] for i in range(201494)]
for i in range(len(actor_tow)):
    for j in range(len(actor_tow)-i):
        if actor_tow[i][i+j] !=0 :
            actor_coo_times=actor_tow[i][i+j]
            actor_coo_office=actor_tow_pf[i][i+j]
            actor_coo_times_arr[row_num][0]=actor_one[i]
            actor_coo_office_arr[row_num][0] = actor_one[i]
            actor_coo_times_arr[row_num][1]=actor_one[i+j]
            actor_coo_office_arr[row_num][1] = actor_one[i + j]
            actor_coo_times_arr[row_num][2]=actor_coo_times
            actor_coo_office_arr[row_num][2]=actor_coo_office
            row_num+=1
            print(row_num)
print(actor_coo_office_arr[0])
print(actor_coo_times_arr[0])

G = nx.Graph() #初始化无向图
G.add_weighted_edges_from(actor_coo_office_arr) #把带权重边的信息加入无向图中

#first compute the best partition
partition = community_louvain.best_partition(G)
"""
#drawing
size = float(len(set(partition.values())))

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


data3 = pd.DataFrame(actor_coo_times_arr)
data3.to_csv('coo_times_arr.csv',na_rep='NA',header=0,index=0)
data4=pd.DataFrame(actor_coo_office_arr)
data4.to_csv('coo_office_arr.csv',na_rep='NA',header=0,index=0)
print("ok")
"""
print(partition)
"""
import json
import datetime
import numpy as np

class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)

def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename,'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)
save_dict("partition",partition)

import json

def load_dict(filename):
    '''load dict from json file'''
    with open(filename,"r") as json_file:
	    dic = json.load(json_file)
    return dic
"""

list1 = list(partition.keys())
list2 = list(partition.values())
z = list(zip(list1,list2))
data5= pd.DataFrame(z)
data5.to_csv('partition_arr_office.csv',na_rep='NA',header=0,index=0)