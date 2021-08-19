# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 16:30:52 2021

@author: Administrator
"""


from __future__ import print_function

import os
import struct
import numpy as np 
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    #os.path.join()函数用于路径拼接文件路径
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels




 




def max2(vec):
    i0=0
    i1=1
    max0=vec[0]
    max1=vec[1]
    if max1> max0:
        i0=1
        i1=0
        max0=vec[1]
        max1=vec[0]
    for i in range(2,len(vec)):
        if vec[i]> max0:
            i1=i0
            i0=i
            max1=max0
            mxa0=vec[i]
        if vec[i]>max1:
            i1=i
            max1=vec[i]
    result=np.array([i0,i1])
    
    return result

def down(data):
    #去中心化 uncen_data 减均值后shape 60000*784 
    ave=np.sum(data,axis=0)/(data.shape[0])
    uncen_data=data-ave
    
    #计算协方差 cov_data 784*784
    cov_data=np.cov(uncen_data.transpose())
    
    #计算特征值向量 
    evals, evecs = linalg.eig(cov_data)
    sumvals=evals.sum()
    maxd=max2(evals)
    max2vals=evals[maxd].sum()
    print("贡献率：",max2vals/sumvals)
    #挑选最大2个特征值对应特征向量
    down_vector=evecs[maxd]
    #计算降维后矩阵
    down_matrix = np.dot(uncen_data, down_vector.transpose())
    return down_matrix
    
   
X_train, y_train = load_mnist('pca', kind='train')

X_test, y_test = load_mnist('pca', kind='t10k')


res=down(X_test)

print(res)


deal=['r','cyan','springgreen','yellow','m','c','b','red','gold','y']
col=[]
for i in range(10000):
    col.append(deal[y_test[i]])
        
        
plt.scatter(res[:,0],res[:,1],c=col)
plt.xlabel('feature-1')
plt.ylabel('feature-2')
plt.title("R2")

'''pca = PCA(n_components=2)
newX = pca.fit_transform(X_train)
print(newX)
plt.scatter(newX[:,0],newX[:,1],c=y_train)
'''