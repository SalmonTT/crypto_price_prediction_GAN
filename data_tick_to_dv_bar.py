import data_bar_generator as bg
import pandas as pd
import os
'''
Generate dollar value bars 
'''
most_traded_crypto_tickers = ['X:BTCUSD', 'X:ETHUSD', 'X:USDTUSD', 'X:XRPUSD',
                              'X:ADAUSD', 'X:DOGEUSD', 'X:LTCUSD']
small_cap_crypto = ['X:USDTUSD', 'X:XRPUSD', 'X:ADAUSD', 'X:DOGEUSD', 'X:LTCUSD']
config_list = ['5KKK', '1M', '3M', '10M']

for ticker in small_cap_crypto:
    for config in config_list:
        print(f'data/bar_data/{ticker[2:]}_{config}.pkl')
        if not os.path.isfile(f'data/bar_data/{ticker[2:]}_{config}.pkl'):
            if config == '1M':
                dv_per_bar = 1_000_000
            elif config == '3M':
                dv_per_bar = 3_000_000
            elif config == '10M':
                dv_per_bar = 10_000_000
            elif config == '5KKK':
                dv_per_bar = 500_000
            tick_data = pd.read_pickle(f'data/tick_consolidated/{ticker[2:]}.pkl')
            tick_data = tick_data.sort_values(by='timestamp', ascending=True)
            tick_data['dv'] = tick_data['price'] * tick_data['size']
            # generate bar_id
            tick_data['bar_id'] = bg.dollar_bar_ids(tick_data['dv'], dv_per_bar)
            # calculate OHLCV and other bar features
            dv_bar_data = bg.generate_dollar_bar(tick_data)
            # check duplicates
            print(dv_bar_data[dv_bar_data['timestamp_end'].duplicated(keep=False)].sort_values('timestamp_start'))
            '''
            The cause for duplicates: 
                    - An instant where there is huge amount of transactions
                Solution: 
                    - simply keep the last for now
            '''
            # drop duplicates
            # reference_bar = dv_bar_data.drop_duplicates(subset=['timestamp_end'], keep='last')
            # save to local
            dv_bar_data.to_pickle(f'data/bar_data/{ticker[2:]}_{config}.pkl')
        else:
            print(f'data/bar_data/{ticker[2:]}_{config}.pkl' + ' found locally!')
