import pandas as pd
import datetime
import data_download_fn as data_download
'''
Download macroeconomic data from various sources. 
If data frequency is less than daily, forward fill the last value
'''
# start and end dates of the entire data period
start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2023, 3, 31)

# early start date for monthly freq data
early_start_date = datetime.datetime(2009, 11, 1)
# daily freq empty dataframe
dates = pd.date_range(early_start_date, end_date, freq='D')
dates.name = 'Date'

### Macro indicators
# macro economic indices (e.g. LIBOR and fixed income securities were used in the project)
# FRED Data: US
# Fed funds effective rate (daily, but in reality only changes when FOMC announces a change)
# DFF = data_download.load_FRED('DFF', start_date, end_date)

# TIPS (actual daily, contains missing values)
DFII10 = data_download.load_FRED('DFII10', start_date, end_date)

# Non-farm payrolls (monthly --> daily)
# PAYEMS = data_download.load_FRED('PAYEMS', start_date, end_date)
# PAYEMS_daily = PAYEMS.reindex(dates, method='ffill')

# inflation (monthly --> daily)
# inflation_USA = data_download.load_NASDAQ('RATEINF/INFLATION_USA', early_start_date, end_date).reindex(dates, method='ffill')
# inflation_GBR = data_download.load_NASDAQ("RATEINF/INFLATION_GBR", start_date=early_start_date, end_date=end_date).reindex(dates, method='ffill')
# inflation_EUR = data_download.load_NASDAQ("RATEINF/INFLATION_EUR", start_date=early_start_date, end_date=end_date).reindex(dates, method='ffill')
# inflation_JPN = data_download.load_NASDAQ("RATEINF/INFLATION_JPN", start_date=early_start_date, end_date=end_date).reindex(dates, method='ffill')

# CPI (monthly --> daily)
# cpi_USA = data_download.load_NASDAQ("RATEINF/CPI_USA", start_date=early_start_date, end_date=end_date).reindex(dates, method='ffill')
# cpi_GBR = data_download.load_NASDAQ("RATEINF/CPI_GBR", start_date=early_start_date, end_date=end_date).reindex(dates, method='ffill')
# cpi_EUR = data_download.load_NASDAQ("RATEINF/CPI_EUR", start_date=early_start_date, end_date=end_date).reindex(dates, method='ffill')
# cpi_JPN = data_download.load_NASDAQ("RATEINF/CPI_JPN", start_date=early_start_date, end_date=end_date).reindex(dates, method='ffill')


# U.S. Treasury： https://data.nasdaq.com/data/USTREASURY-us-treasury
# Daily frequency
US_real_yields = data_download.load_NASDAQ("USTREASURY/REALYIELD", start_date=start_date, end_date=end_date)
US_billrates = data_download.load_NASDAQ("USTREASURY/BILLRATES", start_date=start_date, end_date=end_date)

# UK fixed income: https://data.nasdaq.com/data/BOE-bank-of-england-official-statistics/documentation
# Daily, infrequent
# daily_gilt_repo = data_download.load_NASDAQ("BOE/IUDGRON", start_date=start_date, end_date=end_date)
# Yield from British Government Securities, 20 Year Nominal Implied forward
# Daily, frequent
uk_20y_nif = data_download.load_NASDAQ("BOE/IUDLNIF", start_date=start_date, end_date=end_date)
# Yield from British Government Securities, 10 Year Nominal Implied forward
# Daily, frequent
uk_10y_nif = data_download.load_NASDAQ("BOE/IUDMNIF", start_date=start_date, end_date=end_date)
# Yield from British Government Securities, 5 Year Nominal Implied forward
# Daily, frequent
uk_5y_nif = data_download.load_NASDAQ("BOE/IUDSNIF", start_date=start_date, end_date=end_date)
# UK official bank rate
# Daily, infrequent
uk_bank_rate = data_download.load_NASDAQ("BOE/IUDBEDR", start_date=start_date, end_date=end_date)

# Corporate Bonds: https://data.nasdaq.com/data/ML-corporate-bond-yield-rates
# Daily, frequent
EM_HY = data_download.load_NASDAQ("ML/EMHYY", start_date=start_date, end_date=end_date)
US_Corp = data_download.load_NASDAQ("ML/USEY", start_date=start_date, end_date=end_date)

# ### VIX
# Daily, frequent
vix=data_download.load_FRED('VIXCLS', start_date, end_date)

# ### Currencies

# nominal USD broad nominal
# Daily, frequent
usd_nominal = data_download.load_FRED('DTWEXBGS', early_start_date, end_date)
# real USD broad nominal
# Daily, frequent
usd_real = data_download.load_FRED('RTWEXBGS', early_start_date, end_date).reindex(dates, method='ffill')
# USD EURO spot
# Daily, frequent
usd_euro = data_download.load_FRED('DEXUSEU', start_date, end_date)

### Bullions
# Daily, frequent
# London Bullion Market Association： https://data.nasdaq.com/data/LBMA-london-bullion-market-association
gold_london = data_download.load_NASDAQ("LBMA/GOLD", start_date=start_date, end_date=end_date)
# this one was discontinued 8 years ago
# gold_forward = data_download.load_NASDAQ("LBMA/GOFO", start_date=start_date, end_date=end_date)
silver_london = data_download.load_NASDAQ("LBMA/SILVER", start_date=start_date, end_date=end_date)

# ----- 3. Data Consolidation -----

cols = US_real_yields.columns.values.tolist()
cols_new = [x.replace(' ', '_') for x in cols]
dict_rename = dict(zip(cols, cols_new))
US_real_yields = US_real_yields.rename(columns=dict_rename)
# merge to dataset
dataset = US_real_yields.copy()

US_billrates = US_billrates[['4 Wk Bank Discount Rate', '13 Wk Bank Discount Rate', '52 Wk Bank Discount Rate']]
cols = US_billrates.columns.values.tolist()
cols_new = [x.replace(' ', '_') for x in cols]
dict_rename = dict(zip(cols, cols_new))
US_billrates = US_billrates.rename(columns=dict_rename)
# merge to dataset
dataset = pd.merge(dataset, US_billrates, how='left', left_index=True, right_index=True)

# +
# dataset['fed_funds_rate'] = DFF
dataset['tips'] = DFII10
# dataset['US_payroll'] = PAYEMS_daily
# dataset['inflation_USA'] = inflation_USA
# dataset['inflation_GBR'] = inflation_GBR
# dataset['inflation_EUR'] = inflation_EUR
# dataset['inflation_JPN'] = inflation_JPN
# dataset['cpi_USA'] = cpi_USA
# dataset['cpi_GBR'] = cpi_GBR
# dataset['cpi_EUR'] = cpi_EUR
# dataset['cpi_JPN'] = cpi_JPN

# dataset['daily_gilt_repo'] = daily_gilt_repo
dataset['uk_20y_nif'] = uk_20y_nif
dataset['uk_10y_nif'] = uk_10y_nif
dataset['uk_5y_nif'] = uk_5y_nif
dataset['uk_bank_rate'] = uk_bank_rate

dataset['EM_HY'] = EM_HY
dataset['US_Corp'] = US_Corp
dataset['vix'] = vix

dataset['usd_nominal'] = usd_nominal
dataset['usd_real'] = usd_real
dataset['usd_euro'] = usd_euro
# -

dataset['gold_london'] = gold_london['USD (PM)']
dataset['silver_london'] = silver_london['USD']
# dataset['LIBOR_1M'] = gold_forward['LIBOR - 1 Month']
# dataset['LIBOR_6M'] = gold_forward['LIBOR - 6 Months']
# dataset['LIBOR_12M'] = gold_forward['LIBOR - 12 Months']
# dataset['GOFO_1M'] = gold_forward['GOFO - 1 Month']
# dataset['GOFO_6M'] = gold_forward['GOFO - 6 Months']
# dataset['GOFO_12M'] = gold_forward['GOFO - 12 Months']

missing_cols = dataset.columns[dataset.isnull().any()]
dataset[missing_cols] = dataset[missing_cols].ffill()

# drop '30_YR' since 2010 data is missing from source
dataset = dataset.drop(columns=['30_YR'])

print(dataset.columns[dataset.isnull().any()])

output_file = 'data/'+'macro_'+start_date.strftime("%m_%d_%Y")+'to'+end_date.strftime("%m_%d_%Y")+'.pkl'
dataset.to_pickle(output_file)



