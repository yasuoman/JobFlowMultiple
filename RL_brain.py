# project : Job_Flow
# file   : RL_brain.py
# author:yasuoman
# datetime:2020/7/22 10:50
# software: PyCharm

"""
description：Reinforcement learning algorithm part, parameter update
说明：强化学习算法部分，参数的更新
"""
import numpy as np
class Linear_TD_lambda:
    # def __init__(self,alpha,gamma,my_lambda,m):  #m用于计算向量的大小
    #     self.alpha = alpha  #α
    #     self.gamma = gamma  #γ
    #     self.my_lambda = my_lambda #λ
    #     self.vector_C = np.ones(10*m)  #每个机器有11个特征
    #
    #     self.vector_E = np.zeros(10*m)

    def __init__(self, alpha, gamma, my_lambda, m):  # m用于计算向量的大小
        self.alpha = alpha  # α
        self.gamma = gamma  # γ
        self.my_lambda = my_lambda  # λ
        self.vector_C = np.ones(10 * m)  # 每个机器有11个特征
        # 按照强化书上说的来试试，适合迹向量持续时间通常少于一次训练
        # self.vector_E = np.zeros(10*m)

    # def update_parameter(self,s,reward,s_):#传过来当前状态的特征向量，下一个状态特征向量和奖励
    #     # δ Delta
    #     np.set_printoptions(precision=6)
    #     delta = float(format(reward + self.gamma*np.dot(s_,self.vector_C)-np.dot(s,self.vector_C),'.6f'))
    #     self.vector_E = self.gamma*self.my_lambda*self.vector_E + s
    #     self.vector_C = self.vector_C + self.alpha*delta*self.vector_E

    def update_parameter(self,s,reward,s_,vector_E):#传过来当前状态的特征向量，下一个状态特征向量和奖励
        # δ Delta
        np.set_printoptions(precision=6)
        delta = float(format(reward + self.gamma*np.dot(s_,self.vector_C)-np.dot(s,self.vector_C),'.6f'))
        vector_E = self.gamma * self.my_lambda * vector_E + s
        self.vector_C = self.vector_C + self.alpha*delta*vector_E
        return vector_E

    def calc_state_value(self,s):
        return self.gamma*np.dot(s,self.vector_C)

