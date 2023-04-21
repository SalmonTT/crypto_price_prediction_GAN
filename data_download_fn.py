import pandas as pd
from fredapi import Fred
import nasdaqdatalink
from polygon import RESTClient
import os
API_KEY = 'g6cqlamIZ9mGf_Xn4Az33iespu2f10bN'
client = RESTClient(api_key=API_KEY)
NASDAQ_DATA_LINK_API_KEY = 'eP1dykvbQZ4DhM-v6fBL'
fred = Fred(api_key='c0840fcbad7715d56f29e034547b90b8')

def get_ticker_trades(start_dt, end_dt, ticker, path):
    # crypto only
    file_name = path + ticker[2:] + '_' + start_dt + '_' + end_dt + '.pkl'
    if os.path.isfile(file_name):
        print(file_name + ' found locally!')
        return pd.read_pickle(file_name)
    else:
        print('downloading data and saving locally at' + path)
        timestamp = []
        exchange = []
        # https://polygon.io/glossary/us/stocks/conditions-indicators
        conditions = []
        price = []
        size = []
        for t in client.list_trades(ticker=ticker, timestamp_gte=start_dt, timestamp_lte=end_dt, limit=50000):
            timestamp.append(t.participant_timestamp)
            exchange.append(t.exchange)
            conditions.append(t.conditions[0])
            price.append(t.price)
            size.append(t.size)
        df_dict = {'timestamp': timestamp, 'exchange': exchange, 'conditions': conditions, 'price': price, 'size': size}
        trades = pd.DataFrame(df_dict)

        # post processing:
        trades['timestamp'] = pd.to_datetime(trades['timestamp'], unit='ns')
        trades = trades.set_index('timestamp').sort_index(ascending=True).reset_index()
        # save local
        trades.to_pickle(file_name)
    return trades

def load_FRED(data_id, start_date, end_date):
    output_file = 'data/macro/'+data_id+start_date.strftime("%m_%d_%Y")+'to'+end_date.strftime("%m_%d_%Y")+'.pkl'
    try:
        data = pd.read_pickle(output_file)
        print('reading from local')
    except:
        print('local file not found...downloading')
        data = fred.get_series(data_id, start_date, end_date)
        data.to_pickle(output_file)
    return data

def load_NASDAQ(data_id, start_date, end_date):
    data_id_mod = data_id.split('/')[1]
    output_file = 'data/macro/'+data_id_mod+start_date.strftime("%m_%d_%Y")+'to'+end_date.strftime("%m_%d_%Y")+'.pkl'
    try:
        data = pd.read_pickle(output_file)
        print('reading from local')
    except:
        print('local file not found...downloading')
        data = nasdaqdatalink.get(data_id, start_date=start_date, end_date=end_date, api_key=NASDAQ_DATA_LINK_API_KEY)
        data.to_pickle(output_file)
    return data
