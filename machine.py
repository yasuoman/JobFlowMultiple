# project : Job_Flow
# file   : machine.py
# author:yasuoman
# datetime:2020/7/22 10:49
# software: PyCharm

"""
description：According to the characteristics of environment computing, the executable behavior set is obtained
说明：根据环境计算特征，得到可执行的行为集合
"""

import numpy as np
class Machine:
    # id表示实例化的机器的编号(从1到m)，time_tables表示所有的加工时间表，
    #由环境传过来的numpy array二维数组，大小为m*n，
    def __init__(self,id,m,n,time_tables):
        # -------------- 初始化特征值 --------------
        self.f1_job_num = 0.0
        self.f2_all_pro_mean = 0.0
        self.f3_Q1_all_pro_mean = 0.0
        self.f4_max_pro = 0.0
        self.f5_min_pro = 0.0
        self.f6_now_rem_pro = 0.0
        self.f7_2_SPT = 0.0
        self.f8_2_LPT = 0.0
        self.f9_3_SPT = 0.0
        self.f10_3_LPT = 0.0
        # self.f11_keep_lazy = 0.0



        # -------------- 初始化一些数据结构 --------------
        self.time_tables = time_tables
        # time_table表示关于这个机器加工工件时间,
        self.time_table = time_tables[id - 1]
        self.id = id
        self.m = m #m为机器的台数
        self.n = n  #n为工件个数
        #用于计算特征6,next_time_point是下一个状态的时间点，
        # now_time_point是当前状态的时间点
        #rem_pro_time指的是当前的加工的工件的剩余加工时间
        # self.next_time_point = 0
        # self.now_time_point = 0
        self.rem_pro_time =0


        self.Q = {}    #队列Q，存放该机器的缓冲区作业及其相应的加工时间,可被调度改变
        # Queues是从Q1到Qi的所有作业，是一个字典，由调度get_Q()每个机器对象然后组合起来得到，然后截取一部分传给这个机器对象
        self.Queues = {}
        # self.optional_aciton = []   #根据当前环境，可选的行为集合，传给schedule
        self.feature_vector = []    #该机器的特征向量，传给RL_brain
        self.mean_pro_time = np.mean(self.time_table)   #:所有作业该机器上的加工时间平均值
        #初始化第一台机器的缓冲区作业
        self.init_first_machine()

    def calc_all_features(self):
        self.calc_f1_job_num() #先计算特征1的值
        self.calc_f3_Q1_all_pro_mean()
        self.calc_f11_keep_lazy()
        set1=[]
        set2=[]
        set3=[]
        set4=[]
        if self.f1_job_num == 0:    #如果特征1的值为0,很多特征直接赋值为0
            self.f2_all_pro_mean = 0
            self.f4_max_pro = 0
            self.f5_min_pro = 0
            self.f6_now_rem_pro = 0
            self.f7_2_SPT = 0
            self.f8_2_LPT = 0
            self.f9_3_SPT = 0
            self.f10_3_LPT = 0
        else:
            self.calc_f2_all_pro_mean()
            self.calc_f4_max_pro()
            self.calc_f5_min_pro()
            self.calc_f6_now_rem_pro()
            set1 = self.get_set1()
            set2 = self.get_set2()
            set3 = self.get_set3()
            set4 = self.get_set4()
        self.feature_vector = [self.f1_job_num,self.f2_all_pro_mean,self.f3_Q1_all_pro_mean,
                               self.f4_max_pro,self.f5_min_pro,self.f6_now_rem_pro,self.f7_2_SPT,
                               self.f8_2_LPT,self.f9_3_SPT,self.f10_3_LPT]
        return set1,set2,set3,set4


    def init_first_machine(self):  #初始化的第一台机器的缓冲区的作业为所有作业
        if self.id ==1:
            machine_list = [i for i in range(1,self.n+1)]
            self.Q = dict(zip(machine_list,self.time_tables[0]))

    def set_rem_pro_time(self,time_span):
        if self.rem_pro_time ==0:
            self.rem_pro_time=0
        else:
            self.rem_pro_time = self.rem_pro_time - time_span

    def get_feature_vector(self):
        return self.feature_vector

    def get_Q(self):
        return self.Q

    def get_id(self):
        return self.id

    def set_Q(self,Q):
        self.Q = Q

    def set_Queues(self,Queues):
        self.Queues = Queues



    # -------------- 计算特征函数 --------------
    def calc_f1_job_num(self):
       self.f1_job_num = len(self.Q)/self.n

    def calc_f2_all_pro_mean(self):
        # values=0.0
        # for value in self.Q.values():
        #     values += value
        #不晓得哪个效率高
        values = sum(self.Q.values())
        mean_values = values/len(self.Q)
        self.f2_all_pro_mean = mean_values/self.mean_pro_time

    def calc_f3_Q1_all_pro_mean(self):

        values = sum(self.Queues.values())
        #values为0说明字典为空了，不能取len()
        if values==0:
            mean_values=0
        else:
            mean_values = values/len(self.Queues)
        self.f3_Q1_all_pro_mean = mean_values/self.mean_pro_time

    def calc_f4_max_pro(self):
        max_pro = max(self.Q.values())
        self.f4_max_pro = max_pro/self.mean_pro_time

    def calc_f5_min_pro(self):
        min_pro = min(self.Q.values())
        self.f5_min_pro = min_pro/self.mean_pro_time

    def calc_f6_now_rem_pro(self):
        # if self.rem_pro_time==0:
        #     self.f6_now_rem_pro=0
        # else:
        # #类似如,这个零件的加工时间是6秒，当前的时间是10，下一个时间点是15，那么下一个状态下的剩余加工时间是1秒
        #     self.rem_pro_time = self.rem_pro_time-(self.next_time_point-self.now_time_point)
        #     #计算特征6的值
        #     self.f6_now_rem_pro = self.rem_pro_time/self.mean_pro_time
        # # 记得将当前时间点变成新的时间点
        # self.now_time_point = self.next_time_point
        #大改
        self.f6_now_rem_pro = self.rem_pro_time/self.mean_pro_time

    # -------------- 计算特征11的函数，比较特殊,未实现--------------
    def calc_f11_keep_lazy(self):
        return 0


    # -------------- 关于行为的函数 --------------
    def a1_FCFS(self):  #先到先服务
        fcfs_tuple = list(self.Q.items())[0] #返回一个元组，形如（3，6），表示选择了工件3，处理时间为6
        self.rem_pro_time = fcfs_tuple[1]  #讲特征6设为选中的工件的加工时间
        return fcfs_tuple

    # -------------- 关于行为2和3的函数 --------------
    def get_set1(self):  #得到集合1，且设置特征7的值，schedule还要保留这个列表,schedule根据特征的值来决定要选择的行为集合
        sublist=[]
        if self.id >= 1 and self.id <= self.m-1:
            job_list = list(self.Q.keys())
            sublist = [i for i in job_list if self.time_tables[self.id-1, i-1] <=self.time_tables[self.id, i-1]]
            if sublist:   #列表不为空,且id满足约束
                self.f7_2_SPT = 1
            else:#需要加else,因为不加的话，默认保存上次的特征的值
                self.f7_2_SPT = 0
        return  sublist

    def get_set2(self):  #得到集合2，
        sublist=[]
        if self.id >= 1 and self.id <= self.m - 1:
            job_list = list(self.Q.keys())
            sublist = [i for i in job_list if self.time_tables[self.id-1, i-1] > self.time_tables[self.id, i-1]]
            if sublist:  # 列表不为空,且id满足约束
                self.f8_2_LPT = 1
            else:
                self.f8_2_LPT = 0
        return sublist


    def a2_2_SPT(self,set1):  #行为2，执行这个函数就已经默认传过来的集合不为空
        subdict = dict([(key, self.Q[key]) for key in set1]) #得到符合条件的集合的字典
        min_pro_tuple = min(zip(subdict.values(), subdict.keys()))
        min_pro_tuple = (min_pro_tuple[1], min_pro_tuple[0])
        self.rem_pro_time = min_pro_tuple[1]
        return min_pro_tuple

    def a3_2_LPT(self,set2): #行为3，执行这个函数就已经默认传过来的集合不为空
        subdict = dict([(key, self.Q[key]) for key in set2])  # 得到符合条件的集合的字典
        max_pro_tuple = max(zip(subdict.values(), subdict.keys()))
        max_pro_tuple = (max_pro_tuple[1], max_pro_tuple[0])
        self.rem_pro_time = max_pro_tuple[1]
        return max_pro_tuple

    # -------------- 关于行为4和5的函数 --------------

    def get_set3(self):
        sublist=[]
        if self.id >= 1 and self.id <= self.m - 2:
            job_list = list(self.Q.keys())
            next_max_pro = max(self.time_tables[self.id,np.array(job_list)-1]) #找到在下一个机器上加工工件时间的最大值,工件也是从1开始的
            sublist = [i for i in job_list if self.Q[i]>=next_max_pro ]  #不知道查询字典和查询numpy二维数组哪个快
            if sublist:  # 列表不为空,且id满足约束
                self.f9_3_SPT = 1
            else:
                self.f9_3_SPT = 0

        return sublist

    def get_set4(self):
        sublist=[]
        if self.id >= 1 and self.id <= self.m - 2:
            job_list = list(self.Q.keys())
            next_max_pro = max(self.time_tables[self.id,np.array(job_list)-1]) #找到在下一个机器上加工工件时间的最大值,工件也是从1开始的
            sublist = [i for i in job_list if self.time_tables[self.id+1, i - 1] >=next_max_pro ]
            if sublist:
                self.f10_3_LPT = 1
            else:
                self.f10_3_LPT = 0
        return sublist


    def a4_3_SPT(self,set3):

        # 有问题，argmin里面组成了一个新的array，索引序号变了
        # min_job = [ i for i in sete3 max(self.time_tables[id-1,i-1] +  self.time_tables[id,i-1])]
        # arg_min_job=np.argmin(self.time_tables[id-1,np.array(set3)-1] +  self.time_tables[id,np.array(set3)-1])

        #得到一个元组，第一项为pi + pi+1,第二项为所需要的工件序号
        new_tuple= min(zip(self.time_tables[self.id-1,np.array(set3)-1] + self.time_tables[self.id,np.array(set3)-1],set3))
        min_pro_tuple = (new_tuple[1],self.Q[new_tuple[1]])
        self.rem_pro_time = min_pro_tuple[1]
        return min_pro_tuple

    def a5_3_LPT(self,set4):

        new_tuple = max(zip(self.time_tables[self.id+1, np.array(set4) - 1] + self.time_tables[self.id, np.array(set4) - 1],set4))
        max_pro_tuple = (new_tuple[1], self.Q[new_tuple[1]])
        self.rem_pro_time = max_pro_tuple[1]
        return max_pro_tuple

    # -------------- 行为6和7的函数 --------------
    def a6_SPT(self):
        min_pro_tuple = min(zip(self.Q.values(),self.Q.keys()))#得到的元组，第一个元素为加工时间，需要调换位置
        min_pro_tuple =(min_pro_tuple[1],min_pro_tuple[0])
        self.rem_pro_time = min_pro_tuple[1]
        return min_pro_tuple

    def a7_LPT(self):
        max_pro_tuple = max(zip(self.Q.values(), self.Q.keys()))#得到的元组，第一个元素为加工时间，需要调换位置
        max_pro_tuple = (max_pro_tuple[1], max_pro_tuple[0])
        self.rem_pro_time = max_pro_tuple[1]
        return max_pro_tuple
    # -------------- 行为8的函数，比较特殊 --------------
    def a8_SRPT(self):
        #得到队列中每个工件的剩余加工时间，这里要加list...
        rem_pro = np.sum(self.time_tables[self.id-1:,np.array(list(self.Q.keys()))-1],axis=0)
        min_rem_pro_tuple = min(zip(rem_pro,self.Q.keys()))   #得到相应的作业序号
        min_rem_pro_tuple = (min_rem_pro_tuple[1],self.Q[min_rem_pro_tuple[1]])
        self.rem_pro_time = min_rem_pro_tuple[1]
        return min_rem_pro_tuple

    def a9_do_nothing(self):
        return ()  #返回空元组,调度的时候先判断元组是否为空，对于行为9和行为10来说，为空其实操作都一样，就是啥也不做

    # -------------- 关于行为10的函数--------------
    def a10_keep_lazy(self):
        return () #返回空元组


