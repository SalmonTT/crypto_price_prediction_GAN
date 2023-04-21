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

path = 'data/'+'dataset_1'+start_date.strftime("%m_%d_%Y")+'to'+end_date.strftime("%m_%d_%Y")+'.pkl'
dataset = pd.read_pickle(path)

dataset.index = pd.to_datetime(dataset.index)

dataset.columns

dataset.shape

dataset.columns[dataset.isnull().any()]

# # 3. Data Imputation

# +
missing_cols = dataset.columns[dataset.isnull().any()]
dataset[missing_cols] = dataset[missing_cols].ffill()

# drop '30_YR' since 2010 data is missing from source
dataset = dataset.drop(columns=['30_YR'])
# -

dataset.columns[dataset.isnull().any()]

# + [markdown] tags=[]
# # 4. Data Exploration

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## 4.1 Plot data to in timeseries 

# + tags=[]
for feature in dataset.columns:
    fig = plt.figure()
    plt.figure(figsize=(20,6))
    ax1 = fig.add_subplot(111, ylabel=feature)
    dataset[feature].plot(ax=ax1, color='b', lw=.5, )
# -

# #### Observation: 
# - Remove uk_bank_rate because datapoints are too infrequent

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## 4.2 Time Series Analysis
# -

col_name = 'GS_close'

data = dataset[[col_name]].copy()
# log
data[col_name+'_log'] = np.log(dataset[col_name])
# log shift1
data[col_name+'_log_shift1'] = data[col_name+'_log'].shift()
# log diff
data[col_name+'_log_diff'] = data[col_name+'_log'] - data[col_name+'_log_shift1']
# log diff
data[col_name+'_log_diff_shift1'] = data[col_name+'_log_diff'].shift()
# mean groupby year
df_yearly_mean = data.groupby(data.index.year)[col_name + "_log"].mean()
# 12 days rolling average log
data[col_name+'_logMA12'] = data[col_name+'_log'].rolling(12).mean()
# exponential moving average
data[col_name+'_exp12'] = data[col_name+'_log'].ewm(halflife=12).mean()


def adf(ts):
    rolmean = ts.rolling(12).mean() 
    rolstd = ts.rolling(12).std()
    
    plt.figure(figsize=(20,6))
    orig = plt.plot(ts.values, color='blue',label='Original')
    mean = plt.plot(rolmean.values, color='red', label='Rolling Mean')
    std = plt.plot(rolstd.values, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    
    # adftest = adfuller(ts, autolag='AIC')
    adftest = adfuller(ts, autolag='t-stat')
    adfoutput = pd.Series(adftest[0:4], index=['Test Statistic','p-value','# of Lags Used',
                                              'Number of Observations Used'])
    for key,value in adftest[4].items():
        adfoutput['Critical Value (%s)'%key] = value
    return adfoutput

title = 'histogram of log close'
data[col_name+'_log'].plot(kind = "hist", bins = 30, title=title)

title = 'Mean close price for each year'
df_yearly_mean.plot(title=title)

title='difference between log close and log close shifted by 1 period'
data[col_name+'_log_diff'].plot(title=title)

title= 'scatter plot of log close and log close shifted by 1 period'
data.plot(kind= "scatter", y = col_name+'_log', 
          x = col_name+'_log_shift1', s = 50, title=title)

title= 'scatter plot of log close and log close shifted by 1 period'
data.plot(kind= "scatter", y = col_name+'_log_diff', 
          x = col_name+'_log_diff_shift1', s = 50, title=title)

data.plot(kind ="line", y=[col_name+'_log', col_name+'_logMA12'], )

ts = data[col_name+'_log_diff']
ts.dropna(inplace = True)
adf(ts)

ts = data[col_name+'_log'] - data[col_name+'_logMA12']
ts.dropna(inplace=True)
adf(ts)

title='exponential moving average of log close and log close'
data.plot(kind ="line", y=[col_name+'_exp12', col_name+"_log"], title=title, )

ts = data[col_name+'_log'] - data[col_name+'_exp12']
ts.dropna(inplace = True)
adf(ts)

decomposition = seasonal_decompose(data[col_name+'_log'], model='multiplicative', period=31)
# decomposition = seasonal_decompose(data[col_name+'_log'], model='additive', period=14)
fig = decomposition.plot()
fig.set_size_inches((20, 10))
# Tight layout to realign things
fig.tight_layout()
plt.show()

lag_acf = acf(data[col_name+'_log_diff'].dropna(), nlags=31)
ACF = pd.Series(lag_acf)
plt.axhline(y = 0.1, color = 'r', linestyle = '-')
plt.axhline(y = -0.1, color = 'r', linestyle = '-')
ACF.plot(kind = "bar")

lag_pacf = pacf(data[col_name+'_log_diff'].dropna(), nlags=31, method='ols')
PACF = pd.Series(lag_pacf)
plt.axhline(y = 0.1, color = 'r', linestyle = '-')
plt.axhline(y = -0.1, color = 'r', linestyle = '-')
PACF.plot(kind = "bar")


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


ta_dataset = get_technical_indicators(dataset, 'GS')

# #### Missing Technical Indicator (from rolling)
# - Deal by  backwards fill

ta_dataset[['20sd','upper_band', 'lower_band', 'ma7', 'ma21']] = ta_dataset[['20sd','upper_band', 'lower_band', 'ma7', 'ma21']].interpolate(method='time', limit_direction='backward')

ta_dataset.columns[dataset.isnull().any()]


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


plot_technical_indicators(ta_dataset, 'GS', 400)

close_fft = np.fft.fft(np.asarray(ta_dataset[symbol+'_close'].tolist()))
fft_df = pd.DataFrame({'fft':close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

plt.figure(figsize=(26, 10), dpi=100)
fft_list = np.asarray(fft_df['fft'].tolist())
for num_ in [3, 6, 9, 100]:
    fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
    ta_dataset[symbol+'_close_fft_'+str(num_)] = np.fft.ifft(fft_list_m10)
    plt.plot(ta_dataset['fft_'+str(num_)], label='Fourier transform with {} components'.format(num_))
plt.plot(ta_dataset['GS_close'],  label='Real')
plt.xlabel('Days')
plt.ylabel('USD')
plt.title(f'Figure 3: {symbol} (close) stock prices & Fourier transforms')
plt.legend()
plt.show()

# ### Output to pickle

output_file = 'data/'+'dataset_2'+start_date.strftime("%m_%d_%Y")+'to'+end_date.strftime("%m_%d_%Y")+'.pkl'
dataset.to_pickle(output_file)


