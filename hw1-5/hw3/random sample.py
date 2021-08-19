# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:09:50 2021

@author: Administrator
"""


import random
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math


num=200

def pdf(x):#分布的概率密度函数
    if x>0:
        y=1/(np.sqrt(2))*np.exp(-1*x*np.sqrt(2))
    else:
        y=1/(np.sqrt(2))*np.exp(x*np.sqrt(2))
        
    return y

def inserve(y):#分布的逆分布函数
    if y>0.5:
        x=-1/(np.sqrt(2))*np.log(2-2*y)
    else:
        x=1/(np.sqrt(2))*np.log(2*y)

    return x


X=np.zeros(num)


for i in range(num):
    y=random.random()
    x=inserve(y)
    X[i]=x
    

bins = np.linspace(X.min(), X.max(), 20)  
b = np.linspace(X.min(), X.max(), 40)  
out=np.zeros(40)
for i in range(40):
    out[i]=pdf(b[i])
frequency_each, _1, _2 = plt.hist(X,
                                  bins,
                                  alpha=1,
                                  density=True)  


plt.xlim(X.min(), X.max()) 
gap=(X.max()-X.min())/20
for i in range(20):
    bins[i]+=gap/2
plt.plot(b,out,color='g')
plt.xlabel('sample')
plt.ylabel('frequency')
plt.legend('p')
plt.title('N:200')
plt.show()



print(X)