import numpy as np
import pandas as pd
from tqdm import tqdm

def vwap_trades(df):
    vwap = np.sum(df['price'] * df['size']) / np.sum(df['size'])
    return vwap
def dollar_bar_ids(dvs, dv_per_bar):
    """
    Sample by the dollar value of 'size'
    """
    # assign unique ID to rows where the cumulative sum of value X exceeds threshold Y
    cumulative_sum = 0
    unique_id = 1
    id_list = []
    for dv in tqdm(dvs):
        cumulative_sum += dv
        if cumulative_sum > dv_per_bar:
            id_list.append(unique_id)
            unique_id += 1
            cumulative_sum = 0
        else:
            id_list.append(unique_id)
    return id_list

def generate_dollar_bar(tick_df):
    bars_df = tick_df.groupby('bar_id').agg(timestamp_start=('timestamp', min), timestamp_end=('timestamp', max))
    print('Timestamp')
    HL = tick_df.groupby('bar_id').agg(High=('price', max), Low=('price', min))
    print('HL')
    bars_df = pd.concat([bars_df, HL], axis=1, ignore_index=False)
    bars_df['vwap'] = tick_df.groupby('bar_id').apply(vwap_trades)
    print('VWAP')
    # bars_df['twap'] = tick_df[['bar_id', 'price']].groupby('bar_id').mean()
    bars_df['Open'] = tick_df[['bar_id', 'price']].groupby('bar_id').first()
    print('Open')
    bars_df['Close'] = tick_df[['bar_id', 'price']].groupby('bar_id').last()
    print('Close')
    bars_df['Volume'] = tick_df[['bar_id', 'size']].groupby('bar_id').sum()
    print('Volume')
    bars_df['dollar_volume'] = tick_df[['bar_id', 'dv']].groupby('bar_id').sum()
    print('dollar_volume')
    bars_df['n_ticks'] = tick_df[['bar_id', 'price']].groupby('bar_id').count()
    return bars_df

def generate_bar_entry(df, start, end):
    bar_entry = {}
    bar_entry['timestamp_start'] = start
    bar_entry['timestamp_end'] = end
    bar_entry['High'] = df['price'].max()
    bar_entry['Low'] = df['price'].min()
    bar_entry['vwap'] = np.sum(df['price'] * df['size']) / np.sum(df['size'])
    bar_entry['twap'] = df['price'].mean()
    bar_entry['Open'] = df['price'].iloc[0]
    bar_entry['Close'] = df['price'].iloc[-1]
    bar_entry['Volume'] = df['size'].sum()
    bar_entry['dollar_volume'] = df['dv'].sum()
    bar_entry['n_ticks'] = df['price'].count()
    return bar_entry
