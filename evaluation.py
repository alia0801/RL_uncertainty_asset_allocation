# %%
import os
import pandas as pd
import numpy as np
import statistics
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
from util import *
from config import *
# %%

def get_df_ABC(df_org,init_money=ORG_INIT_AMOUNT):
    df_org = df_org.reset_index()

    df_org['Close'] = init_money
    for i in range(len(df_org)-1):
        df_org.loc[i+1,'Close'] = (df_org['daily_return'][i]+1)*df_org['Close'][i]
    
    df_cp = df_org.copy(deep=True)
    sub_df = df_cp[df_cp['index']%5==0].reset_index(drop=True)
    sub_df['w_return'] = 0 
    for i in range(len(sub_df)-1):
        sub_df.loc[i+1,'w_return'] = (sub_df['Close'][i+1] - sub_df['Close'][i])/sub_df['Close'][i]
    try:
        stdev_week = statistics.stdev(sub_df['w_return'])* math.pow( 52, 0.5 )
    except:
        stdev_week=0.00001

    df_org = df_org.fillna(0)
    df_org['max']=0
    s1 = df_org['Close']
    for i in range(len(df_org)):
        df_org.loc[i,'max'] = s1[0:i+1].max() 
    
    df_org['dd'] = 0
    df_org['dd'] = 1-(df_org['Close']/df_org['max'])
    
    mdd = df_org['dd'].max()

    df_org['total_value'] = init_money
    for i in range(1,len(df_org)):
        df_org.loc[i,'total_value'] = df_org['total_value'][i-1]*(df_org['daily_return'][i]+1)
    try:
        ann_stdev = statistics.stdev(df_org['daily_return'])* math.pow( 252, 0.5 )
    except:
        ann_stdev = 0.00001

    ann_reward = (df_org['total_value'][len(df_org)-1]/df_org['total_value'][0])**(252/len(df_org))-1
    
    df_org['ann_reward'] = 0
    for i in range(1,len(df_org)):
        df_org.loc[i,'ann_reward'] = (df_org['total_value'][i]/df_org['total_value'][0])**(252/i)-1

    sharpe = (252**0.5)*df_org['daily_return'].mean()/ \
                       df_org['daily_return'].std()

    return ann_reward,ann_stdev,mdd,stdev_week,sharpe
