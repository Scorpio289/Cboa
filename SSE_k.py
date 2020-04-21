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

fff=open('data_notype.csv',encoding='utf-8')
actoridata=genfromtxt('data_notype.csv',delimiter=',',usecols=(0),dtype=str)
data = genfromtxt(fff,delimiter=',',usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69))
SSE=[]
index=[]

for k in range(1,69):
    for i in range(10):
        kk = KMeans(n_clusters=k, max_iter=300, n_init=10).fit(data)
        index.append(kk.inertia_)
    SSE.append(sum(index)/len(index))
    index=[]

X = range(1, 69)

plt.xlabel('k')

plt.ylabel('SSE')

plt.plot(X, SSE, 'o-')

plt.show()

print("ok")
#sse 和 k图片