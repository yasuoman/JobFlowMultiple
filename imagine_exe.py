# project : Job_Flow
# file   : imagine_exe.py
# author:yasuoman
# datetime:2020/8/1 11:15
# software: PyCharm
from memento import Memento
import copy
import schedule
"""
description：
说明：想象假设机器执行某个行为
"""
import numpy as np

def instantiate_Memento(machines_object_dict,rem_pro_time_tables,m):
    mementos_dict = {}
    for i in range(m):
        mementos_dict[i+1] = Memento(i+1,copy.deepcopy(machines_object_dict),copy.deepcopy(rem_pro_time_tables))
    return mementos_dict


def get_single_machine_optional_action(id,machines_object_dict):#id代表这个机器的序号
    # 得到所有的机器缓冲区的作业
    all_Q_list = schedule.get_all_Q(machines_object_dict)

    # 赋值单个机器的Queue
    dict_data = schedule.list_dict(all_Q_list[:id])  # 得到一个字典
    machines_object_dict[id].set_Queues(dict_data)

    #计算单个机器的特征值
    single_machine_sets = machines_object_dict[id].calc_all_features()
    #得到单个机器的特征值
    single_machine_features = machines_object_dict[id].get_feature_vector()
    #得到单个机器的可选行为
    single_machine_optional_action = schedule.get_optional_action(single_machine_features)

    return single_machine_sets,single_machine_optional_action

#执行相应行为  获得其奖励 + 折扣后的状态价值
def get_action_reward_value(single_machine_sets,single_machine_optional_action,id,
                            machines_object_dict,now_time,rem_pro_time_tables,time_tables
                            ,RL):
    reward_value_list=[]
    for i in single_machine_optional_action:
        #恶心的地方，这里还需要为每个行为都重新深拷贝machines_object_dict和rem_pro_time_tables
        machines_object_dict_copy = copy.deepcopy(machines_object_dict)
        rem_pro_time_tables_copy = copy.deepcopy(rem_pro_time_tables)
        #得到相应行为选中的工件及其加工时间
        single_pro_job_time = schedule.get_pro_job_time(i,machines_object_dict_copy[id],single_machine_sets)
        #print(single_pro_job_time)
        rem_pro_time_list = list(rem_pro_time_tables_copy[rem_pro_time_tables_copy > 0])
        #如果这个机器选择的行为不是行为9，即选择了某个工件进行加工，那么决定下一个时刻
        #由这个工件加工的时间和rem_pro_time_table中的最小值来确定
        if single_pro_job_time:
            #得到下一个时刻
            rem_pro_time_list.append(single_pro_job_time[1])
            next_time = min(rem_pro_time_list) + now_time
            #重新赋值rem_pro_time_tables为相应的加工时间,single_pro_job_time[0]表示加工的作业
            rem_pro_time_tables_copy[id-1][single_pro_job_time[0]-1] = single_pro_job_time[1]
            # 计算奖励值
            reward = schedule.calc_reward(now_time, next_time, machines_object_dict_copy)
            # 将所有机器的剩余加工时间减少一个时间间隔
            schedule.set_all_rem_pro_time(machines_object_dict_copy, next_time - now_time)
            # rem_pro_time_tables减少一个时间间隔，0和负数减少会一直变成负数
            rem_pro_time_tables_copy = rem_pro_time_tables_copy - (next_time - now_time)
            # 移除机器的缓冲区中已经加工完成的工件,这里只移除当前加工的这个工件
            schedule.remove_job_from_Q(machines_object_dict_copy[id], single_pro_job_time)

            # 将time_table中的作业加入待加工的机器的缓冲区中
            schedule.add_jobs_to_all_Q(machines_object_dict_copy, time_tables, rem_pro_time_tables_copy)
            # 重新获取所有机器中的缓冲区的作业
            all_Q_list_ = schedule.get_all_Q(machines_object_dict_copy)
            # 重新赋值Queue
            schedule.set_all_Queues(machines_object_dict_copy, all_Q_list_)

            # -------------- 重新计算特征值 --------------
            all_machine_sets_ = schedule.calc_all_machine_features(machines_object_dict_copy)
            all_machine_features_ = schedule.get_all_machine_features(machines_object_dict_copy)
            # 二维列表转化为一维列表
            all_machine_features_RL = [j for i in all_machine_features_ for j in i]
            #print(all_machine_features_)
            # print(RL.vector_C[0])
            discount_state_value = RL.calc_state_value(all_machine_features_RL)
            reward_value_list.append(reward + discount_state_value)
        # 如果这个机器选择的行为是行为9,即什么作业都不采取，那么直接不计算下一个状态，直接返回空列表




    return reward_value_list

def imagine_exe_single(id,machines_object_dict,now_time,rem_pro_time_tables,time_tables,RL
              ,epsilon,train):
    #得到机器的可选行为
    single_machine_sets,single_machine_optional_action = get_single_machine_optional_action(id,machines_object_dict)
    #遍历所有的行为，得到相应的奖励+折扣状态值
    reward_value_list = get_action_reward_value(single_machine_sets,single_machine_optional_action,id,
                            machines_object_dict,now_time,rem_pro_time_tables,time_tables
                            ,RL)



    #如果列表非空
    if reward_value_list:
        #找到其中奖励+状态值的最大值
        max_action_tuple = max(zip(reward_value_list,single_machine_optional_action))
        #生成一个对应的字典
        choose_action_dict = dict(zip(single_machine_optional_action,reward_value_list))
        # print(choose_action_dict)
        # print('\n')
        # print(max_action_tuple)
        #如果是在训练过程，那么需要进行e_greedy选择行为
        if train:
            choose_action = e_greedy(epsilon,max_action_tuple[1],single_machine_optional_action)
        #如果是在训练完成，为了测试的时候，直接选择贪婪行为
        else:
            choose_action = max_action_tuple[1]
        #选择
        return (choose_action,choose_action_dict[choose_action])
    else:#说明选择了行为9，直接返回一个None

        return (9,None)











def e_greedy(epsilon,greedy_action,optional_action):  #ε
    if np.random.uniform()>epsilon:
        choose_action = greedy_action
    else:
        choose_action = np.random.choice(optional_action)
    return choose_action



