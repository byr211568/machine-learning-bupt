# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:03:21 2021

@author: Administrator
"""


import scipy.io as io
import numpy as np
import random
import matplotlib.pyplot as plt
N=3
T=100

def forword(a,tran,emis,O,t):
    
    for i in range(N):
        sum1=0
        for j in range(N):
            sum1+=a[t-1][j]*tran[j][i]*emis[i][O[t]-1]#利用放射概率和转移概率计算下一时间步概率
        a[t][i]=sum1
    return a


def veterbi(dp,tran,emis,O,prior):
    path=[]#记录状态序列
    dp=np.zeros((T+2,N))
    p=np.zeros((102,N))
    pre=np.zeros((T+2,N),dtype=int)
    dp[0]=prior
    for i in range(1,102):
        for n in range(N):
            p = dp[i-1,:] * tran[:,n] * emis[n,O[i-1]-1]#利用放射概率和转移概率计算下一时间步概率
            pre[i,n] = np.argmax(p)#选择最大概率对应隐状态
            dp[i][n] = np.max(p)
   
    maxid=np.argmax(dp[100,:])
    path.append(maxid)
    i=100
    while i>1:
        path.append(pre[i][maxid])#回溯记录隐状态序列
        maxid=pre[i][maxid]
        i=i-1
    
    path.reverse()
    return path


def transing(tran,emis,O):
    hide=2
    O_my=np.zeros(28,dtype=int)
    for i in range(100,128):
        tra=tran[hide]
        proc=np.zeros(N)
        sum2=0
        for j in range(N):
            proc[j]=sum2
            sum2+=tra[j]
        for j in range(N):
            proc[j]=proc[j]/sum2
    
        ran=random.random()
   
        j=N-1
        while j>0:
            if ran>proc[j]:
                break
            j=j-1
        hide=j
      
        proc=np.zeros(5)
        sum2=0
        for k in range(5):
        
            proc[k]=sum2
            sum2+=emis[j][k]
    
        k=4
        ran=random.random()
    
        while k>0:
            if ran>proc[k]:
                break
            k=k-1
   
        O_my[i-100]=k+1    
    
    return O_my

    
def bars(pt):
    t=100
    plt.figure(1,(40,12))
    plt1=plt.subplot(111)
    plt1.bar(x = range(1,t+1),
        height = pt[:,0],
        width=0.5,
        bottom = 0, 
        color = 'red', 
        label = 'up',  #图形的标签
        )

    plt1.bar(x = range(1,t+1),  #指定条形图x轴的刻度值
        height = pt[:,1],  
        width=0.5,
        bottom = pt[:,0],
        color = 'lawngreen',  
        label = 'down', 
        )

    plt1.bar(x = range(1,t+1), 
        height = pt[:,2],  #指定条形图y轴的数值
        width=0.5,
        bottom =  pt[:,0]+ pt[:,1],
        
        color = 'lightskyblue',  
        label = 'same',  #图形的标签
        )

    plt.ylabel('p')
    plt.xlabel('day')
    plt.title('State Sequence')
    plt.legend()
    plt.show()


def scatter(path):
    
    
    path_deal=[]
    for i in range(T):
        if path[i]==0:
            path_deal.append('red')
        elif path[i]==1:
            path_deal.append('lawngreen')
        else:
            path_deal.append('lightskyblue')
    day=range(0,T)
    plt.figure(1,figsize=(16,8))
    plt.scatter(day,path[0:100],c=path_deal)
    plt.ylabel('state')
    plt.xlabel('day')
    plt.title('State Sequence')
    plt.show()


mat_path='D:\QQdownload\hmm_params.mat'

param=io.loadmat(mat_path)
a=np.zeros((T+32,N))
tran=param['transition']
emis=param['emission']
O=param['price_change'][0]
O_my=O.copy()
pt=np.zeros((T+32,N))
prior=param['prior'].reshape(1,3)


#初始化a矩阵
for j in range(N):
    a[0][j]=prior[0,j]*emis[j][O[0]-1]

#前向算法
for i in range(1,100):
    a=forword(a,tran,emis,O,i)
    
    
for j in range(N):
    pt[:,j]=a[:,j]/a.sum(axis=1)
#打印及可视化隐状态概率
print(pt[0:100])
bars(pt[0:100])

#veterbi求解隐状态序列
dp=np.zeros((T+2,N))
path=veterbi(dp,tran,emis,O,prior)
#打印及可视化隐状态序列
print("path:")
print(path) 
scatter(path) 

#预测
prec=np.zeros(100)
my=np.zeros((100,28))
for i in range(100):
    my[i]=transing(tran,emis,O[100:128])
    prec1=0
    for l in range(28):
        if(my[i][l]!=O[l+100]):
            prec1+=1
    prec[i]=1-prec1/28
finprec=prec.mean(axis=0)

print('precise: ',finprec)
mean=my.mean(axis=0)
std=my.std(axis=0)
print('mean:',mean)
print('std: ',std)




