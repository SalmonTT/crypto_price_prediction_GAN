import data_download_fn as data_download
import pandas as pd
import datetime
from tqdm import tqdm
# Most traded crypto
# most_traded_crypto_tickers = ['X:BTCUSD', 'X:ETHUSD', 'X:USDTUSD', 'X:XRPUSD', 'X:ADAUSD', 'X:DOGEUSD]
most_traded_crypto_tickers = ['X:DOGEUSD']
# config
start_date = datetime.date(2017, 1, 1)
end_date = datetime.date(2023, 3, 31)
list_start_dt = pd.date_range(start=start_date, end=end_date, freq='MS').strftime("%Y-%m-%d")
list_end_dt = pd.date_range(start=start_date, end=end_date, freq='M').strftime("%Y-%m-%d")


path = './data/'
for ticker in tqdm(most_traded_crypto_tickers):
    for ind, start in enumerate(tqdm(list_start_dt)):
        print(start, list_end_dt[ind])
        df = data_download.get_ticker_trades(start, list_end_dt[ind], ticker, path)

