# project : Job_Flow
# file   : environment.py
# author:yasuoman
# datetime:2020/7/22 11:03
# software: PyCharm

"""
description：Information about the environment
说明：关于产生加工工件的初始时间,机器和工件个数
"""
import numpy as np
seed_list = [1732558531,1740217881,543992763,72917236,825413466,
             3505090,133522409,504290499,378666277,1828548364,
             1936165087,1061369127,1840119330,573685653,424074829,
             1703164209,1220969682,1248138445,2138036458,778787377,
             1180357694,987825092,2065640710,2026781897,1658903808,
             540891060,43674311,894233988,203439220,2146296924,
             137867715,233213300,890278950,191456126,1072794331,
             245351478,671975015,94477980,1125494561,210095985]

#n的取值有四种情况
n_list=[20,30,40,50]

# m_list=[15,20]
#Random Number Generator
def rng(i):
    seed_list[i] = int((seed_list[i]*16807.0)%2147483647.0)
    pro_time = seed_list[i]/2147483648.0
    return pro_time
def create_time_tables(k):
    low = 1
    high = 200
    #因为n具有规律性，20，30，40，50，每个都是10组
    n = n_list[int(k/10)]
    if (int(k/5)%2==0):
        # m的取值有两种情况
        m=15
    else:
        m=20

    time_tables = np.ones((n, m))
    for i in range(n):
        for j in range(m):
            time_tables[i][j] = int(low + rng(k) * high)
    return m, n, time_tables.T

