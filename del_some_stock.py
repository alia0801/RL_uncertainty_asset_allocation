import pandas as pd
import yfinance as yf
from util import *
from tickers import *
all_tickers = list(set(DOW_30_TICKER)|set(NAS_100_TICKER)|set(SP_100_TICKER))
new_all_tickers = portfolio_list[str(0)]
for portfolio_code in range(30):
    if portfolio_code not in [1,4,6,8,9,10,12,16,17,18,19,20,21]:
        new_all_tickers = (set(new_all_tickers)|set(portfolio_list[str(portfolio_code)]))
print(len(new_all_tickers))
file = open('ok_tickers.txt','w')
for item in new_all_tickers:
	file.write(str(item)+"\n")
file.close()
