# project : Job_Flow
# file   : schedule.py
# author:yasuoman
# datetime:2020/7/22 11:01
# software: PyCharm

"""
desc
说明：调度部分，相关函数的实现，主要在run_this中进行调用
"""

import copy
from machine import Machine
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_optional_action(feature_vector):  #根据特征来得到行为集合，feature_vector是一个大小为11的列表

    if feature_vector[0]==0 or feature_vector[5]:  #如果特征1的值为0，或者特征6的值不为0，则选择行为9
        optional_action_list = [9]
    else:
        optional_action_list=[1,6,7,8]
        if feature_vector[6]:
            optional_action_list.append(2)
        if feature_vector[7]:
            optional_action_list.append(3)
        if feature_vector[8]:
            optional_action_list.append(4)
        if feature_vector[9]:
            optional_action_list.append(5)
        # if feature_vector[10]:
        #     optional_action_list.append(10)
    return optional_action_list

def get_all_machine_optional_action(all_machine_features):
    #前m-1个机器都是类似的处理，最后一个机器是特殊情况
    all_machine_optional_action = [get_optional_action(i) for i in all_machine_features[:-1]]
    if all_machine_features[-1][0]==0 or all_machine_features[-1][5]:  #如果特征1的值为0，或者特征6的值不为0，则选择行为9
        optional_action_list = [9]
    else:#最后一个机器空闲且缓冲区有对列的时候，只选择行为1，先来先服务。
        optional_action_list = [1]
    all_machine_optional_action.append(optional_action_list)
    return  all_machine_optional_action


#实例化对象
def instantiate_machines(m,n,time_tables):   #实例化所有的机器在一个字典里面
    machines_object_dict = {}        #所有实例化机器对象的字典
    for i in range(m):
        machines_object_dict[i+1] = Machine(id=i+1,m=m,n=n,time_tables=time_tables)
    return machines_object_dict

def get_all_Q(machines_object_dict):  #得到所有的机器缓冲区的作业，是一个字典列表
    #all_Q_list = [machines_object_dict[key].get_Q() for key in machines_object_dict.keys() ]
    all_Q_list = [value.get_Q() for value in machines_object_dict.values()]
    return all_Q_list      #得到一个字典列表形如[{1:2,2；3}....]

#将新的作业加入到机器的缓冲区
def add_job_to_Q(machine_object,job, pro_time):
    old_Q= copy.deepcopy(machine_object.get_Q())
    add_Q = {job: pro_time}
    old_Q.update(add_Q)
    machine_object.set_Q(old_Q)

def add_jobs_to_all_Q(machines_object_dict,time_tables,rem_pro_time_tables):
    for i in range(rem_pro_time_tables.shape[0]-1): #最后一台机器后面没有机器可以再加了
        for j in range(rem_pro_time_tables.shape[1]):
            #如果某个工件在某个机器上加工完成了
            if rem_pro_time_tables[i][j]==0:
                #将这个某件的下一个工序加到下一台机器的缓冲区内
                #i+2，j+1是因为机器和工件都是从1开始的，
                add_job_to_Q(machines_object_dict[i + 2], j+1, time_tables[i+1][j])

#从缓冲区中移除工件
def remove_job_from_Q(machine_object,pro_job_time):


        new_Q = copy.deepcopy(machine_object.get_Q())
        #删除相应的工件
        new_Q.pop(pro_job_time[0])
        # 赋给机器中的队列
        machine_object.set_Q(new_Q)





def remove_jobs_from_all_Q(machines_object_dict,all_pro_job_time):
    for i,j in zip(machines_object_dict.values(),all_pro_job_time):
        # 如果执行的工件不为空，即有工件被执行了
        if j:
            remove_job_from_Q(i,j)

# 将一个类似[{'aa': 'a3'},
#                  {'bb': 'b5'},
#                  {'cc': 'c6'},
#                  {'dd': 'd7'}]
# 的列表转化为{'aa': 'a3', 'bb': 'b5', 'cc': 'c6', 'dd': 'd7'}
def list_dict(list_data):
   dict_data = {}
   for i in list_data:
       dict_data.update(i)
       # key, = i
       # value, = i.values()
       # dict_data[key] = value

   return dict_data
#重新赋值所有Q1到Qi的缓冲区队列，用于计算特征3
def set_all_Queues(machines_object_dict,all_Q_list): #给每个机器赋予Queues(从Q1到Qi的缓冲区作业)，用于计算特征3
    for key in machines_object_dict.keys():
        dict_data = list_dict(all_Q_list[:key])  #得到一个字典
        machines_object_dict[key].set_Queues(dict_data)




#计算所有机器的特征值
def calc_all_machine_features(machines_object_dict):
    all_machine_sets = [machines_object_dict[key].calc_all_features() for key in machines_object_dict.keys()]

    return all_machine_sets

#得到所有机器的特征直
def get_all_machine_features(machines_object_dict):
    #all_machine_features = [machines_object_dict[key].get_feature_vector() for key in machines_object_dict.keys()]
    all_machine_features = [value.get_feature_vector() for value in machines_object_dict.values()]
    return all_machine_features


def calc_reward(t,t_,machines_object_dict):
    machine_lazy_num=0
    for value in machines_object_dict.values():
        #不管是还没进行工作，还是上次刚好工作完，但是还在等待中的机器
        #目前我的写法是这样
        if value.rem_pro_time==0:
            machine_lazy_num=machine_lazy_num+1

    return (t - t_)*machine_lazy_num/len(machines_object_dict)


def all_machine_lazy_do_nonthing(machine_work_state,machine_choose_action):#用于步骤7
        for i in range(len(machine_work_state)):
            if machine_work_state[i]!=0 or machine_choose_action[i]!=9:
                return False
        return True

def get_pro_job_time(action_id,machine_object,machine_sets):
    pro_job_time=()
    if action_id==1:
        pro_job_time=machine_object.a1_FCFS()
    elif action_id==2:
        pro_job_time=machine_object.a2_2_SPT(machine_sets[0])
    elif action_id==3:
        pro_job_time=machine_object.a3_2_LPT(machine_sets[1])
    elif action_id==4:
        pro_job_time=machine_object.a4_3_SPT(machine_sets[2])
    elif action_id==5:
        pro_job_time=machine_object.a5_3_LPT(machine_sets[3])
    elif action_id==6:
        pro_job_time=machine_object.a6_SPT()
    elif action_id==7:
        pro_job_time=machine_object.a7_LPT()
    elif action_id==8:
        pro_job_time=machine_object.a8_SRPT()
    elif action_id==9:
        pro_job_time=machine_object.a9_do_nothing()
    # elif action_id==10:
    #     pro_job_time=machine_object.a10_keep_lazy()

    return pro_job_time


def get_all_pro_job_time(all_machine_action,machines_object_dict,all_machine_set):
    all_pro_job_time=[get_pro_job_time(i,j,k) for i,j,k in zip(all_machine_action,machines_object_dict.values(),all_machine_set)]
    return all_pro_job_time

#执行所有机器的行为，这是不输出的版本，用于训练过程。
def exe_all_machine_action(all_pro_job_time,time_point,all_Q_list,all_machine_action,rem_pro_time_tables,work_done_tables):
     exe_time_list=[]

     for i in range(len(all_pro_job_time)):
         if all_pro_job_time[i]:
             work_done_tables[i][all_pro_job_time[i][0]-1] =1
             exe_time_list.append(all_pro_job_time[i][1])

     #选择的应该是所有加工时间和正在加工的时间最短的那个时间间隔
     rem_pro_time_list = list(rem_pro_time_tables[rem_pro_time_tables>0])
     return min(exe_time_list+rem_pro_time_list)+time_point

#执行所有机器的行为，这是输出的版本，用于展示机器执行工件的细节展示
def exe_all_machine_action_print(all_pro_job_time, time_point, all_Q_list, all_machine_action, rem_pro_time_tables,
                           work_done_tables):  # 实现输出语句，表示正在执行,返回下一个时间点

    print("系统时间"+str(time_point)+":")
    exe_time_list = []

    done_list = np.argwhere(rem_pro_time_tables==0).tolist()
    if done_list:
        for i in done_list:
            print("机器"+str(i[0]+1)+"已加工完成工件"+str(i[1]+1))

    for i in range(len(all_pro_job_time)):
        if all_pro_job_time[i]:
            print("机器"+str(i+1)+"的缓冲区队列上的工件："+str(all_Q_list[i]),",将采取的行为："
                  +str(all_machine_action[i])+",马上执行工件:"
                  +str(all_pro_job_time[i][0])+",预计处理时间："+str(all_pro_job_time[i][1]))
            #如果某个工件在某个机器上被执行了，将work_done_tables中的记录记为1
            work_done_tables[i][all_pro_job_time[i][0] - 1] = 1
            exe_time_list.append(all_pro_job_time[i][1])
        else:
            rem_pro_time_arg = np.argwhere(rem_pro_time_tables[i]>0)

            rem_pro_time_arg = rem_pro_time_arg.tolist()
            # 如果有，也只会存在一个,故直接取[0][0]
            if rem_pro_time_arg:
                print("机器" + str(i + 1) + "缓冲区队列上的工件：" + str(all_Q_list[i]), ",正在加工工件"
                      + str(rem_pro_time_arg[0][0] + 1) + "......"+"   剩余加工时间：" + str(rem_pro_time_tables[i][rem_pro_time_arg[0][0]]))

            else:
                print("机器" + str(i + 1) + "缓冲区队列上的工件：" + str(all_Q_list[i]), ",采取的行为："
                      + str(all_machine_action[i]) + ",处于空闲状态")

    print("\n" )

    # 选择的应该是所有加工时间和正在加工的时间最短的那个时间间隔

    rem_pro_time_list = list(rem_pro_time_tables[rem_pro_time_tables > 0])
    return min(exe_time_list + rem_pro_time_list) + time_point

#重置所有机器的剩余加工时间
def set_all_rem_pro_time(machines_object_dict,time_span):
    for value in machines_object_dict.values():
        value.set_rem_pro_time(time_span)

#这个目前没用于判断结束，因为存在逻辑漏洞
# def all_Q_empty(all_Q_list):
#     #如果某个集合不为空，那么返回True
#     for i in all_Q_list:
#         if i:
#             return True
#     return False


 #初始默认work_done_tables中的值为0， 如果某个工件在某个机器上被执行了，将work_done_tables中的记录记为1,
#如果还有工件的工序未被执行则返回true
def work_done_tables_one(work_done_tables):
     for i in range(work_done_tables.shape[0]):
         for j in range(work_done_tables.shape[1]):
             if work_done_tables[i][j]==0:
                 return True
     return False

#当剩余加工时间表中的剩余加工时间大于0的话，就返回true
def rem_pro_time_positive(rem_pro_time_tables):
    for i in range(rem_pro_time_tables.shape[0]):
        for j in range(rem_pro_time_tables.shape[1]):
            if rem_pro_time_tables[i][j]>0:
                return True
    return False

#基准给出的解
benchmark_solution = [4437,4144,3779,4302,4373,
                      4821,4779,4944,4886,4717,
                      5226,5304,5079,5605,5147,
                      6183,6037,6241,6095,5822,
                      6986,6351,6506,6845,6783,
                      7154,7528,7469,7608,7219,
                      7673,7679,7416,7548,7750,
                      8838,8539,8417,8590,8493]

#书中给出的解，我对其中的解持怀疑态度，狗头保命🐕
book_solution=[4211,3962,3776,4264,4257,
               4794,4755,4828,4858,4642,
               4907,5095,4972,5242,5113,
               5854,5979,6236,5889,5805,
               6559,6154,6402,6510,6431,
               7116,7439,7426,7466,7203,
               7466,7221,7212,7383,7673,
               8360,7896,8370,8219,8461
               ]


#画图，保存图片
def plot(run_num,working_times,i,n,m):
    # 设置x轴的文本，用于描述x轴代表的是什么
    plt.xlabel("训练次数")
    # 设置y轴的文本，用于描述y轴代表的是什么
    plt.ylabel("完工时间")

    plt.plot(run_num, working_times, 'b-')

    plt.title('flcmax_'+str(n)+'_'+str(m)+'_'+str(i))

    benchmark_line=np.ones(len(run_num))*benchmark_solution[i]
    book_line2=np.ones(len(run_num))*book_solution[i]
    p1, = plt.plot(benchmark_line,'y:')
    p2, = plt.plot(book_line2,'r:')

    plt.legend([p1, p2], ["bechmark:"+str(benchmark_solution[i]), "book:"+str(book_solution[i])], loc='upper left')
    plt.savefig('./photos/'+'flcmax_'+str(n)+'_'+str(m)+'_'+str(i)+'.png')
    #连续作图的需要
    plt.close('all')
    plt.show()


def choose_machine_action(all_aciton_reward_value):
    return [i[0] for i in all_aciton_reward_value]

