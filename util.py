# %%
from contextlib import closing
from io import StringIO
from os import path
from typing import Optional
import statistics
import math
import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces.box import Box
from gymnasium.utils import EzPickle
import yfinance as yf
from finrl.finrl_meta.finrl_meta_config import DOW_30_TICKER
from finrl.finrl_meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
from finrl.apps.config import *
import pandas as pd
from config import *
import datetime
from stable_baselines3.common.callbacks import BaseCallback
import json
# %%
with open('portfolio_list.json') as f:
    portfolio_list = json.load(f)
with open('./ok_tickers.txt','r') as f:
    lines = f.read().split('\n')[:-1]
ok_stickers = [(x) for x in lines]

# %%
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True
# %%

def get_df(asset_list):
    yfp = YahooFinanceProcessor()
    download_df = yfp.download_data(start_date = TRAIN_START_DATE,
                         end_date = TEST_END_DATE,
                         ticker_list = asset_list,
                         time_interval='1D')
    # df = yf.download(DOW_30_TICKER,start=TRAIN_START_DATE,end=TEST_END_DATE)
    df = yfp.clean_data(download_df)

    # df.rename(columns = {'date':'time'}, inplace = True)
    df = yfp.add_technical_indicator(df, TECHNICAL_INDICATORS_LIST)

    # df.rename(columns = {'time':'date'}, inplace = True)
    df['date'] = df['time']
    df = yfp.add_turbulence(df)

    df = yfp.add_vix(df)


    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = []
    return_list = []

    # look back is one year
    lookback=252
    for i in range(lookback,len(df.index.unique())):
      data_lookback = df.loc[i-lookback:i,:]
      price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
      return_lookback = price_lookback.pct_change().dropna()
      return_list.append(return_lookback)

      covs = return_lookback.cov().values 
      cov_list.append(covs)
    
    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)
    return df

def get_ABC(input_df):
    input_df = input_df.reset_index()

    # cal week ann std
    input_df_cp = input_df.copy(deep=True)
    sub_df = input_df_cp[input_df_cp['index']%5==0].reset_index(drop=True)
    sub_df['w_return'] = 0 
    for i in range(len(sub_df)-1):
        sub_df.loc[i+1,'w_return'] = (sub_df['Close'][i+1] - sub_df['Close'][i])/sub_df['Close'][i]
    week_ann_stdev = statistics.stdev(sub_df['w_return'])* math.pow( 52, 0.5 )
    
    input_df['daily_return'] = 0 
    for i in range(len(input_df)-1):
        input_df.loc[i+1,'daily_return'] = (input_df['Close'][i+1] - input_df['Close'][i])/input_df['Close'][i]
    
    # cal mdd
    input_df = input_df.fillna(0)
    input_df['max']=0
    s1 = input_df['Close']
    for i in range(len(input_df)):
        input_df.loc[i,'max'] = s1[0:i+1].max() 
    input_df['dd'] = 0
    input_df['dd'] = 1-(input_df['Close']/input_df['max'])
    mdd = input_df['dd'].max()

    # cal ann stdev
    input_df['total_value'] = ORG_INIT_AMOUNT
    for i in range(1,len(input_df)):
        input_df.loc[i,'total_value'] = input_df['total_value'][i-1]*(input_df['daily_return'][i]+1)
    ann_stdev = statistics.stdev(input_df['daily_return'])* math.pow( 252, 0.5 )

    # cal ann reward
    ann_reward = (input_df['total_value'][len(input_df)-1]/input_df['total_value'][0])**(252/len(input_df))-1
    sharpe = (252**0.5)*input_df['daily_return'].mean()/ \
                       input_df['daily_return'].std()
    return ann_reward,ann_stdev,mdd,week_ann_stdev,sharpe

def get_comb_ABC(comb,start,end,w=None):
    if w is None:
        w = [1/len(comb)]*len(comb)
    close_data = pd.read_csv('./price_indicator_data/big_stock_close_2001_2022.csv')
    close_data.drop('Unnamed: 0', inplace=True, axis=1)
    col = comb.copy()
    col.insert(0,'Date')
    sub_df = close_data[col]
    
    start_list = start.split('-')
    start_list2 = [str(int(i)) for i in start_list]
    start = '/'.join(start_list2)
    
    end_list = end.split('-')
    end_list2 = [str(int(i)) for i in end_list]
    end = '/'.join(end_list2)
     
    
    sub_df = sub_df[(sub_df['Date']>=start)& (sub_df['Date']<=end) ]
    
    sub_df = sub_df.reset_index(drop=True)
    for i in range(len(comb)):
        name = comb[i]
        sub_df[name+'_d_return'] = (sub_df[name] - sub_df[name].shift(1))/sub_df[name].shift(1)
        sub_df[name+'_money'] = ORG_INIT_AMOUNT*w[i]
        for i in range(1,len(sub_df)):
            sub_df.loc[i,name+'_money'] = sub_df[name+'_money'][i-1]*(1+sub_df[name+'_d_return'][i])
    sub_df['Close'] = 0
    for name in comb:
        sub_df['Close']+=sub_df[name+'_money']

    sub_df = sub_df.reset_index(drop=True)
    
    ann_reward,ann_stdev,mdd,week_ann_stdev,sharpe = get_ABC(sub_df)
    return ann_reward,ann_stdev,mdd,week_ann_stdev,sharpe