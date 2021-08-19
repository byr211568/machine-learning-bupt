# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 20:06:39 2021

@author: Administrator
"""

import  pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.style as mplstyle
mplstyle.use('fast')

clusternum=4
samplenum=1000





#更新隐变量Z Z:i*k i:samplenum k:cluster number
def update_Z(Pi,U,O,X):
    y = np.zeros(((samplenum, clusternum)))
    sum1=np.zeros((samplenum,1))#分母
    i=0
    while i<len(Pi):
        y[:,i]=Pi[i]*multivariate_normal.pdf(X,mean=U[i],cov=np.diag(O[i]))#分子
        sum1[:,0]+=y[:,i]
        i=i+1
    i=0
    while i<samplenum:
        y[i,:]=y[i,:]/sum1[i,0]
        i+=1
    return y

def update_U(Z,X,U):
    i=0
    while i<clusternum:
        U[i]=np.average(X, axis=0, weights=Z[:, i])
        i+=1
   
    return U

def update_O(Z,X,U,O):
    i=0
    while i< clusternum:
    
        O[i] = np.average((X - U[i]) ** 2, axis=0, weights=Z[:, i])
        i+=1
    return O 
    
def update_Pi(Z,Pi):
    Pi = Z.sum(axis=0) /Z.sum()
    return Pi

def clusterindex(Z,cluster):
    for i in range(samplenum):
        cluster[i]=0
        maxpro=Z[i,0]
        for j in range(clusternum):
            if Z[i,j]>maxpro:
                maxpro=Z[i,j]
                cluster[i]=j
    return cluster     

def Is_over(U1,U2):
   u3=U1.reshape(clusternum*2) 
   u4=U2.reshape(clusternum*2)
   #print(u3,u4)
   for i in range(clusternum*2):
       if abs(u3[i]-u4[i])<0.001:
           continue
       else:
           return 0
       
   return 1      

if __name__ == '__main__':
    df = pd.read_csv("D:\QQdownload\cluster.dat", sep = " ", header=None, names=["X","Y"])
    x1=np.array(df['X'])
    x2=np.array(df['Y'])

    X=np.vstack((x1,x2)).T
    train, test = train_test_split(X, test_size = 0.2)
    
    model = KMeans(n_clusters=clusternum, random_state=170)
    y=model.fit_predict(X)
    
    
    #初始化U,O,Pi,Z
    U=model.cluster_centers_
    Ulast=np.ones((clusternum,2))
    O=np.ones((clusternum,2))
    Pi=np.array([1/clusternum]*clusternum)
    Z=np.zeros(((samplenum,clusternum)))
    cluster=np.array([0]*samplenum)
    i=0
   
    while Is_over(Ulast,U)==0:
        #print(i)
        Z=update_Z(Pi,U,O,X)
        Ulast=U.copy()
        #print('Ulast: ',Ulast)
        U=update_U(Z,X,U)
        O=update_O(Z,X,U,O)
        Pi=update_Pi(Z,Pi)
        C=np.zeros((samplenum,1))
        
       
        
        cluster=clusterindex(Z, cluster)
        #print('U;',U)
        
        
        plt.scatter(X[:,0],X[:,1],c=cluster)
        colors = ['r','y','b','g']
       
        ax = plt.gca()
        for j in range(clusternum):
            plot_args = {'fc': 'None', 'lw': 4, 'edgecolor': colors[j], 'ls': '--'}
            ellipse = Ellipse(U[j], 1.5* O[j][0], 1.5* O[j][1], **plot_args)
            ax.add_patch(ellipse)
        plt.title(i)
        plt.show()
        i+=1
    silavg = silhouette_score(X,cluster)
    print('clusternum: ',clusternum)
    print('silavg: ',silavg)