# project : Job_Flow
# file   : memento.py
# author:yasuoman
# datetime:2020/8/1 11:04
# software: PyCharm

"""
description：
说明：用来当作临时变量存储对象字典和rem_pro_time_tables
"""
class Memento:
    def __init__(self,id,machines_object_dict,rem_pro_time_tables):
        self.id = id
        self.machines_object_dict =machines_object_dict
        self.rem_pro_time_tables =rem_pro_time_tables


