# -*- coding:utf-8 -*-
from sklearn.cluster import KMeans
# from sklearn.cluster import k_means#这个是先写的，他们两的参数就相差一个数据集，不过还是建议用KMeans
import numpy as np
from sklearn.datasets import load_iris


def loadData(filePath):
    dataSet = []
    file = open(filePath, 'r')

    for lines in file.readlines():
        row = []
        # curLine = lines.strip().split()#２维数据
        curLine = lines.strip().split(',')
        for line in curLine:
            x = float(line)
            row.append(x)

        dataSet.append(row)
    file.close()

    return np.mat(dataSet)


# filePath = '../data/training_4k2_far.txt'
filePath = 'C:/Users/26087/PycharmProjects/untitled/data_notype.csv'
dataSet = loadData(filePath)

print
dataSet
'''直接调用sklearn中的数据'''
# dataSet = load_iris().data
estimator = KMeans(n_clusters=4, max_iter=300, n_init=10).fit(dataSet)  # 构造聚类器
'''这个是必须写的，相当于上面构造出来，配置好，下面这句调用，当然也可以写到上面去
fit方法对数据做training 并得到模型'''
# estimator.fit(dataSet)#聚类

# 下面是三个属性
'''把聚类的样本打标签'''
labelPred = estimator.labels_
'''显示聚类的质心'''
centroids = estimator.cluster_centers_
'''这个也可以看成损失，就是样本距其最近样本的平方总和'''
inertia = estimator.inertia_

print
labelPred
print
centroids
print
inertia
# 这下面是库里包装的方法
'''返回预测的样本属于的类的聚类中心'''
print
estimator.fit_predict(dataSet)
print
estimator.predict(dataSet)
'''这个是返回每个样本与聚类质心的距离'''
print
estimator.fit_transform(dataSet)
print
estimator.transform(dataSet)
'''这个我觉得和损失一样，评价聚类好坏'''
print
estimator.score(dataSet)
