# project : Job_Flow
# file   : run_this.py
# author:yasuoman
# datetime:2020/7/22 10:58
# software: PyCharm

"""
description：The whole TD (lambda) algorithm steps for flow job problem
说明：用于flow job问题的整个TD(lambda)算法步骤
"""

from RL_brain import Linear_TD_lambda
import schedule
import environment as env
import copy
import imagine_exe
import gc
import numpy as np

# -------------- 定义一个全局变量列表，用来存储每个状态的时间点 ---------------------------
#以下是这个全局列表的一些常用方法
def init_schedule_time_list():
    schedule_time_list.clear()
    schedule_time_list.append(0)
def get_schedule_time_list():
    return schedule_time_list[-1]
def add_schedule_time_list(time_point):
    schedule_time_list.append(time_point)

def del_last_schedule_time_list():
    schedule_time_list.pop()

#赋值给rem_pro_time_tables
def set_rem_pro_time_tables(all_pro_job_time):
    for i in range(len(all_pro_job_time)):
        # 如果非空，即采取了某个行为
        if all_pro_job_time[i]:
            rem_pro_time_tables[i][all_pro_job_time[i][0] - 1] = all_pro_job_time[i][1]




if  __name__ =="__main__":
    #画图的数量40
    for j in range(40):
        # 得到时间表
        m,n,time_tables = env.create_time_tables(j)
        # print(time_tables)
        # # 实例化强化学习算法对象，两台机器的参数
        # RL = Linear_TD_lambda(alpha=0.002,gamma=0.05,my_lambda=0.1,m=m)
        # #实例化强化学习算法对象，多台机器的参数
        RL = Linear_TD_lambda(alpha=0.002, gamma=0.005, my_lambda=0.005, m=m)
        #训练次数，我这里设的600
        train_times =600
        #用于画图的横坐标
        run_num=[i+1 for i in range(train_times)]
        #用于画图的纵坐标
        working_time=[]
        for i in range(train_times):
            #适合迹向量初始化为0向量
            vector_E = np.zeros(10 * m)
            #初始剩余加工时间都是-1
            rem_pro_time_tables = np.ones((m, n))*(-1)
            #初始都没完成工件的工序
            work_done_tables=np.zeros((m,n))
            schedule_time_list = []
            init_schedule_time_list()
            #下一个时间点初始化为0
            next_time_point=0

            # 实例化机器对象
            machines_object_dict = schedule.instantiate_machines(m=m,n=n,time_tables=time_tables)
            # 得到所有的机器缓冲区的作业
            all_Q_list = schedule.get_all_Q(machines_object_dict=machines_object_dict)

            # 赋值Queue
            schedule.set_all_Queues(machines_object_dict=machines_object_dict,all_Q_list=all_Q_list)

            #得到所有机器的满足条件约束的集合
            all_machine_sets = schedule.calc_all_machine_features(machines_object_dict=machines_object_dict)

            #得到所有机器的特征值
            all_machine_features = schedule.get_all_machine_features(machines_object_dict=machines_object_dict)
            #得到所有机器的可选的行为集合
            all_machine_optional_action = schedule.get_all_machine_optional_action(all_machine_features=all_machine_features)
            #这里有可能出现比如，某一个时刻某个机器还未加工完成
            #而它的下一台机器并不能得到工件进行加工，但是此时所有缓冲区的作业都是空的。也许还有其他的情况，我暂时没想到
            #这里用或的原因是因为，初始的时候rem_pro_time_tables都是-1，与结束状态类似都并不是大于0的，不好区分
            #因此多加了一个work_done_tables_one的条件，当两者都返回false的时候，
            #才说明此时机器已经加工完了工件
            while( schedule.work_done_tables_one(work_done_tables) or schedule.rem_pro_time_positive(rem_pro_time_tables)):

                # 深复制m*m个machine对象
                mementos_dict = imagine_exe.instantiate_Memento(machines_object_dict, rem_pro_time_tables, m)
                # 得到所有行为和其相对应的奖励+折扣状态值
                all_aciton_reward_value = [imagine_exe.imagine_exe_single(i + 1,
                                                                        mementos_dict[i + 1].machines_object_dict, get_schedule_time_list(),
                                                                        mementos_dict[i + 1].rem_pro_time_tables, time_tables, RL, epsilon=0.05,
                                                                        train=True) for i in range(m)]

                #得到所有机器的行为
                all_machine_action = schedule.choose_machine_action(all_aciton_reward_value)

                #得到所有机器将要执行的工件和相应的时间，
                all_pro_job_time=schedule.get_all_pro_job_time(all_machine_action,machines_object_dict,all_machine_sets)

                #机器执行工件（用输出语句的形式化表达），并得到下一个时刻
                #若需要形式化的表达，用函数exe_all_machine_action_print()
                #若不需要，则用函数exe_all_machine_action（）
                next_time_point=schedule.exe_all_machine_action(all_pro_job_time,get_schedule_time_list(),all_Q_list,all_machine_action,rem_pro_time_tables,work_done_tables)


                #赋值给rem_pro_time_tables
                set_rem_pro_time_tables(all_pro_job_time)
                #计算奖励值
                reward = schedule.calc_reward(get_schedule_time_list(),next_time_point,machines_object_dict)

                #将所有机器的剩余加工时间减少一个时间间隔
                schedule.set_all_rem_pro_time(machines_object_dict,next_time_point - get_schedule_time_list())
                #rem_pro_time_tables减少一个时间间隔，0和负数减少会一直变成负数
                rem_pro_time_tables =rem_pro_time_tables -(next_time_point - get_schedule_time_list())
                #将下一个时刻存入调度时间列表
                add_schedule_time_list(next_time_point)



                # -------------- 关于重新计算特征值的事先准备 --------------

                #移除机器的缓冲区中已经加工完成的工件
                schedule.remove_jobs_from_all_Q(machines_object_dict,all_pro_job_time)
                #将time_table中的作业加入待加工的机器的缓冲区中
                schedule.add_jobs_to_all_Q(machines_object_dict,time_tables,rem_pro_time_tables)

                #重新获取所有机器中的缓冲区的作业
                all_Q_list_ = schedule.get_all_Q(machines_object_dict=machines_object_dict)
                # 重新赋值Queue
                schedule.set_all_Queues(machines_object_dict=machines_object_dict, all_Q_list=all_Q_list_)



                # -------------- 重新计算并获取特征值 --------------
                all_machine_sets_ = schedule.calc_all_machine_features(machines_object_dict=machines_object_dict)
                all_machine_features_ = schedule.get_all_machine_features(machines_object_dict)
                # -------------- 更新参数 --------------
                #二维列表换成一维列表
                all_machine_features_RL = [j for i in all_machine_features for j in i]
                all_machine_features_next_RL = [j for i in all_machine_features_ for j in i]

                vector_E = RL.update_parameter(all_machine_features_RL,reward,all_machine_features_next_RL,vector_E)


                #覆盖原来的一些变量
                all_Q_list = all_Q_list_
                all_machine_sets = all_machine_sets_
                all_machine_features = all_machine_features_

            # -------------- 为了计算最后状态的加工时间，该状态应为所有时间的最大值，而不是最小值 --------------
            working_time.append(get_schedule_time_list())
            #打印出来时间表
            print('第'+str(j+1)+'个数据集,'+'第'+str(i+1)+'训练完成时间：'+str(get_schedule_time_list()))
            #删除所有机器对象，重新开始新的训练
            del machines_object_dict
            # 删除所有的剩余加工时间，重新开始新的训练
            del rem_pro_time_tables
            gc.collect()

       #作图
        schedule.plot(run_num,working_time,j,n,m)

