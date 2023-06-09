import pandas as pd
import yfinance as yf
# from util import *
from tickers import *
all_tickers = list(set(DOW_30_TICKER)|set(NAS_100_TICKER)|set(SP_100_TICKER))
# all_etf1 = all_tickers[:2000]
df_all = yf.download(list(all_tickers),start='2001-01-01',end='2023-03-01')
df_all_close = df_all['Close']
df_all_close = df_all_close.reset_index()
df_all_close.to_csv('./price_indicator_data/big_stock_close_2001_2022.csv')