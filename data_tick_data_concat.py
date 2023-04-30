import pandas as pd
import glob
from tqdm import tqdm
'''
Takes multiple tick trades data (in .pkl format) and combines them into one single .pkl file
'''
# most_traded_crypto_tickers = ['X:BTCUSD', 'X:ETHUSD', 'X:USDTUSD', 'X:XRPUSD',
#                               'X:ADAUSD', 'X:DOGEUSD', 'X:LTCUSD']
output_path = 'data/tick_consolidated/'
# tick_data_path = '/Users/simon/Desktop/notebooks/algo_trading/project/data/BTCUSD_consolidated/'
tick_data_path = 'data/LTCUSD/'
most_traded_crypto_tickers = ['X:LTCUSD']
for ticker in most_traded_crypto_tickers:
    print(ticker)
    ticker = ticker[2:]
    list_timestamp = []
    list_price = []
    list_size = []
    for f in tqdm(glob.glob(f'{tick_data_path}{ticker}*')):
        df = pd.read_pickle(f)
        list_timestamp += df['timestamp'].values.tolist()
        list_price += df['price'].values.tolist()
        list_size += df['size'].values.tolist()
        print(len(list_size))

    dict_data = {'timestamp': list_timestamp, 'price': list_price, 'size': list_size}
    tick_data = pd.DataFrame(dict_data)
    tick_data.to_pickle(output_path+f'{ticker}.pkl')

