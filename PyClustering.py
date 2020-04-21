# 这里选择聚类的方法.
from pyclustering.cluster.birch import birch;
# 这里选择k-means聚类方法，具体的介绍查看 https://codedocs.xyz/annoviko/pyclustering/
from pyclustering.cluster.kmeans import kmeans
# 这里选择聚类的案例数据
from pyclustering.utils import read_sample;
from pyclustering.samples.definitions import FCPS_SAMPLES;
# 可视化
from pyclustering.cluster import cluster_visualizer
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


actoridata=genfromtxt('C:\\Users\\26087\\PycharmProjects\\untitled\\venv\\coo_times_arr.csv',encoding='utf-8',delimiter=',',usecols=(0,1,2),dtype=str)
print(actoridata)
sample = read_sample(actoridata);
# 使用birch算法，聚成三类,这里将类实例化，变成对象
birch_instance = birch(actoridata, 128);
# 使用对象里的方法，开始聚类
birch_instance.process();
# 获取聚类结果
clusters = birch_instance.get_clusters();
# 查看形状，可以看到长度为3，被分为三类
visualizer = cluster_visualizer();
visualizer.append_clusters(clusters,sample);
visualizer.show();