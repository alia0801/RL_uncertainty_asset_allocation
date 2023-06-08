# %%
from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

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
