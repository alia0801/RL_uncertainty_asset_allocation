from tickers import *
import random
import json
all_tickers = list(set(DOW_30_TICKER)|set(NAS_100_TICKER)|set(SP_100_TICKER))

print(len(all_tickers))

portfolios={}
for i in range(30):
    portf = random.choices(all_tickers, k=30)
    portfolios[i] = portf

with open('portfolio_list.json', 'w') as fp:
    json.dump(portfolios, fp)
# file = open('portfolio_list.txt','w')
# for portf in portfolios:
#      for tic in portfolios[portf]:
#           file.write(tic+'\t')
#      file.write('\n')
# file.close()
