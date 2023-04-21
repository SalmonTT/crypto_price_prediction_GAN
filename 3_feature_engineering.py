# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# # !jupytext --to notebook data_analysis_process.py
# # !jupytext --to py data_analysis_process.ipynb
# -

# # 1. Config

# +
import pandas as pd
import numpy as np
import datetime
import pytz

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

# graphs and plots
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('seaborn-v0_8-notebook')
plt.rcParams['figure.figsize'] = (26, 9)

# misc imports
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings("ignore")

# +
# symbol of the underlying stock we are predicting
symbol = 'GS'
# start and end dates of the entire data period
start_date = datetime.datetime(2010, 1, 1, tzinfo=pytz.UTC)
end_date = datetime.datetime(2018, 12, 31, tzinfo=pytz.UTC)

# start and end dates of the training sample
start_date_train = datetime.datetime(2010, 1, 1, tzinfo=pytz.UTC)
end_date_train = datetime.datetime(2017, 1, 1, tzinfo=pytz.UTC)
# start and end dates of the training sample
start_date_test = datetime.datetime(2010, 1, 2, tzinfo=pytz.UTC)
end_date_test = datetime.datetime(2018, 12, 31, tzinfo=pytz.UTC)
# -

# # 2. Load dataset

path = 'data/'+'dataset_2'+start_date.strftime("%m_%d_%Y")+'to'+end_date.strftime("%m_%d_%Y")+'.pkl'
dataset = pd.read_pickle(path)

dataset.index = pd.to_datetime(dataset.index)


# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # 3. Generate TA indicators
# -

def get_technical_indicators(dataset, symbol):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset[symbol+'_close'].rolling(window=7).mean()
    dataset['ma21'] = dataset[symbol+'_close'].rolling(window=21).mean()
    
    # Create MACD
    dataset['26ema'] = dataset[symbol+'_close'].ewm(span=26).mean()
    dataset['12ema'] = dataset[symbol+'_close'].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

    # Create Bollinger Bands
    dataset['20sd'] = dataset[symbol+'_close'].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    # Create Exponential moving average
    dataset['ema'] = dataset[symbol+'_close'].ewm(com=0.5).mean()
    
    # Create Momentum
    dataset['momentum'] = dataset[symbol+'_close']-1
    dataset['log_momentum'] = np.log(dataset['momentum'])
    
    return dataset


dataset = get_technical_indicators(dataset, 'GS')

# #### Missing Technical Indicator (from rolling)
# - Deal by  backwards fill

dataset[['20sd','upper_band', 'lower_band', 'ma7', 'ma21']] = dataset[['20sd','upper_band', 'lower_band', 'ma7', 'ma21']].interpolate(method='time', limit_direction='backward')

dataset.columns[dataset.isnull().any()]


def plot_technical_indicators(dataset, symbol, last_days):
    plt.figure(figsize=(26, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0-last_days
    
    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ =list(dataset.index)
    
    # Plot first subplot
    plt.plot(dataset['ma7'],label='MA 7', color='g',linestyle='--')
    plt.plot(dataset[symbol+'_close'],label='Closing Price', color='b')
    plt.plot(dataset['ma21'],label='MA 21', color='r',linestyle='--')
    plt.plot(dataset['upper_band'],label='Upper Band', color='c')
    plt.plot(dataset['lower_band'],label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators for Goldman Sachs - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()

    plt.legend()
    plt.show()


plot_technical_indicators(dataset, 'GS', 400)

# # 4. Fourier Transform

close_fft = np.fft.fft(np.asarray(dataset[symbol+'_close'].tolist()))
fft_df = pd.DataFrame({'fft':close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

plt.figure(figsize=(26, 10), dpi=100)
fft_list = np.asarray(fft_df['fft'].tolist())
for num_ in [3, 6, 9, 100]:
    fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
    dataset[symbol+'_close_fft_'+str(num_)] = np.fft.ifft(fft_list_m10)
    plt.plot(dataset[symbol+'_close_fft_'+str(num_)], label='Fourier transform with {} components'.format(num_))
plt.plot(dataset['GS_close'],  label='Real')
plt.xlabel('Days')
plt.ylabel('USD')
plt.title(f'Figure 3: {symbol} (close) stock prices & Fourier transforms')
plt.legend()
plt.show()

# # 7. ARIMA features
