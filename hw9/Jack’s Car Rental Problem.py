# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 11:57:09 2021

@author: Daisy

question: Jack rents cars
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
import seaborn as sns



MAX_CAR_NUM=20
MAX_TRANS_CAR=5
MINLOSS=1e-4


RENT=[3,4]
REC=[3,2]


GAMMA=0.09

RENT_MONEY=10
TRANS_MONEY=2
LIM=20


poisson_cache=dict()

def action_get():
    action=[]
    for i in range(-MAX_TRANS_CAR,MAX_TRANS_CAR+1):
        action.append(i)
        
    return action


def poisson_probability(n, lam):
    global poisson_cache
    key = n * 10 + lam # 定义唯一key值，除了索引没有实际价值
    if key not in poisson_cache:
        # 计算泊松概率，这里输入为n与lambda，输出泊松分布的概率质量函数，并保存到poisson_cache中
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]  



def state_value_calcu(state,action,value_matrix):
    '''
    在状态state  执行action 后 state_value值
    

    Parameters
    ----------
    state : TYPE
        DESCRIPTION.
    action : TYPE
        DESCRIPTION.
    value_matrix : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    #初始化 
    value=0.0
    car_now=np.zeros(2,dtype=int)
    
    #车辆调配从1号到2号车库
    #保证车辆不超过20 不是负数
    car_now[0]=max(0,state[0]+action)
    car_now[1]=max(0,state[1]-action)
    car_now[0]=min(MAX_CAR_NUM,state[0]+action)
    car_now[1]=min(MAX_CAR_NUM,state[1]-action)
    
    #print('diao:',car_now)
    #车辆调配花钱
    value-=abs(car_now[0]-state[0]) * TRANS_MONEY
    
    
    #模拟租车，遍历所有租车组合
    r1=LIM
    r2=LIM
    if car_now[0]<LIM:
        r1=car_now[0]
    if car_now[1]<LIM:
        r2=car_now[1]
    for rent_num1 in range(car_now[0]+1):     #car_now[0]+1
        for rent_num2 in range(car_now[1]+1):
            
            car_num=car_now.copy()
            rent_prob=poisson_probability(rent_num1,RENT[0])*poisson_probability(rent_num2,RENT[1])
            
            
            rent_earn=RENT_MONEY * (rent_num1+rent_num2)
            
            car_now[0]-=rent_num1
            car_now[1]-=rent_num2
           
            
            
            car_now=car_num.copy()
            
            
            #模拟还车
            for rec_num1 in range(MAX_CAR_NUM - car_now[0]+1):
                for rec_num2 in range(MAX_CAR_NUM - car_now[1]+1):
                    
                    car_num1=car_now.copy()
                    
                    rec_prob=poisson_probability(rec_num1,REC[0])*poisson_probability(rec_num2,REC[1])
                    
                    car_now[0]+=rec_num1
                    car_now[1]+=rec_num2
                    
                    value+=rent_prob * rec_prob * (rent_earn + GAMMA * value_matrix[car_now[0],car_now[1]])
                    
                    car_now=car_num1.copy()
                
    return value

def policy_iteration():
    #初始化 值函数矩阵为0 策略矩阵为0
    value_matrix=np.zeros((MAX_CAR_NUM+1,MAX_CAR_NUM+1))
    policy=np.zeros((MAX_CAR_NUM+1,MAX_CAR_NUM+1),dtype=np.int)
    action=action_get()
    
     # 准备画布大小，并准备多个子图
    _, axes = plt.subplots(2, 2, figsize=(40, 40))
    # 调整子图的间距，wspace=0.1为水平间距，hspace=0.2为垂直间距
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    axes = axes.flatten()
    print(policy)
    
    idnum=0
    while True:
        
        print('Epoch {}'.format(idnum))
        
        fig = sns.heatmap(np.flipud(policy), cmap="rainbow", ax=axes[idnum])
        
        # 定义标签与标题
        fig.set_ylabel('No.1 car park', fontsize=30)
        fig.set_yticks(list(reversed(range(21))))
        fig.set_xlabel('No.2 car park', fontsize=30)
        fig.set_title('policy {}'.format(idnum), fontsize=30)
        
        
        
        #draw()
        
        #策略评估
        value_matrix=evaluation(value_matrix,policy)
            
    
        #策略改进
         #改进信号 True 不需要改进 
        
        policy,flag=improving(action,policy,value_matrix)
        
        #print(policy)
        
        
        print('the {} epoch , policy stable {}'.format(idnum,flag))
        idnum=idnum+1
        
        if flag:
            print('policy')
            print(policy)
            print('Value')
            print(value_matrix)
            fig = sns.heatmap(np.flipud(value_matrix), cmap="rainbow", ax=axes[-1])
            fig.set_ylabel('No.1 car park ', fontsize=30)
            fig.set_yticks(list(reversed(range(MAX_CAR_NUM + 1))))
            fig.set_xlabel('No.1 car park', fontsize=30)
            fig.set_title('optimal value', fontsize=30)
            
            
            break
         
        
    plt.show() 
    return value_matrix
            
  

def evaluation(value_matrix,policy):
    
    changing=10
    index=0
    while (changing>MINLOSS):
        
        
        value_last=value_matrix.copy()
            
        for i in range(MAX_CAR_NUM+1):
            
            for j in range(MAX_CAR_NUM+1):
            
                value_matrix[i,j]=state_value_calcu([i,j],policy[i,j],value_matrix)
        
        
        changing=abs(value_last - value_matrix).max()
        print("经过{}轮策略评估，评估震荡为{}".format(index,changing))
        index+=1
        
    return value_matrix      
        
    
def improving(action,policy,value_matrix):
    policy_flag=True
    for i in range(MAX_CAR_NUM+1):
        for j in range(MAX_CAR_NUM+1):
            
            action_last=policy[i,j]
            action_reward=-np.inf
            arg=-5
            
            for act in action:
                 
                if (act+i)<0 or (j-act)<0:
                    arg=arg
                else:
                    reward=state_value_calcu([i,j],act,value_matrix)
                    if reward > action_reward:
                        action_reward=reward
                        arg=act
                
                
            
            policy[i,j]=arg
            if action_last!=policy[i,j]:
                policy_flag=False
    
    return policy,policy_flag
                    
            


            

if __name__ == '__main__':
    
    value=policy_iteration()