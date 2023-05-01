import fn_data_download as data_download
import pandas as pd
import datetime
from tqdm import tqdm
# Most traded crypto
most_traded_crypto_tickers = ['X:BTCUSD', 'X:ETHUSD', 'X:USDTUSD', 'X:XRPUSD', 'X:ADAUSD', 'X:DOGEUSD', 'X:LTCUSD']
''' Earlist dates:
    ETHUSD: 20170101
    BTCUSD: 20170101
    LTCUSD: 20170101
    XRPUSD: 20170103
    USDTUSD: 20170329
    DOGEUSD: 20170601
    ADAUSD: 20180206
'''
# config
start_date = datetime.date(2017, 1, 1)
end_date = datetime.date(2023, 3, 31)
list_start_dt = pd.date_range(start=start_date, end=end_date, freq='MS').strftime("%Y-%m-%d")
list_end_dt = pd.date_range(start=start_date, end=end_date, freq='M').strftime("%Y-%m-%d")
path = './data/'

# Download Tick data
# for ticker in tqdm(most_traded_crypto_tickers):
#     for ind, start in enumerate(tqdm(list_start_dt)):
#         print(start, list_end_dt[ind])
#         df = data_download.get_ticker_trades(start, list_end_dt[ind], ticker, path)


# Donwload time bars
start_date = '2017-01-01'
end_date = '2023-03-31'
multiplier = 1
timespan = 'minute'
for ticker in tqdm(most_traded_crypto_tickers):
    # iterate over range of dates
    aggbars_df = data_download.get_time_aggbars(start_date, end_date, ticker, multiplier, timespan, path)