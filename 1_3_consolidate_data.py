import pandas as pd
from functools import reduce
'''
Part 1: consolidate all the crypto bars into one dataframe
    - Use daily aggbar
Part 2: merge crypto prices with macro data

'''
# Part 1
most_traded_crypto_tickers = ['X:BTCUSD', 'X:ETHUSD', 'X:USDTUSD', 'X:XRPUSD',
                              'X:ADAUSD', 'X:DOGEUSD', 'X:LTCUSD']
path = './data/time_bar/'
crypto_data_list = []
for ticker in most_traded_crypto_tickers:
    bar_data = pd.read_pickle(path+ticker[2:]+'_daily.pkl')
    bar_data = bar_data[['Timestamp', 'Vwap']]
    bar_data = bar_data.rename(columns={'Vwap':f'{ticker[2:]}_vwap'})
    crypto_data_list.append(bar_data)

crypto_data = reduce(lambda x, y: pd.merge(x, y, on = 'Timestamp'), crypto_data_list)

# Part 2
macro_data = pd.read_pickle('./data/macro/macro_01_01_2010to03_31_2023.pkl')
dataset = pd.merge(crypto_data, macro_data, how='left', left_on='Timestamp', right_index=True)
# output to pickle
print(dataset)
dataset.to_pickle('./data/dataset.pkl')

