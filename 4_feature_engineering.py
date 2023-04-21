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
# # !jupytext --to notebook 4_feature_engineering.py
# # !jupytext --to py 4_feature_engineering.ipynb
# -

# # 1. Config

# +
import pandas as pd
import numpy as np
import datetime
import pytz

import shap
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
import seaborn as sns # for correlation heatmap
from xgboost import XGBRegressor

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
start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2018, 12, 31)

# start and end dates of the training sample
start_date_train = datetime.datetime(2010, 1, 1)
end_date_train = datetime.datetime(2017, 1, 1)
# start and end dates of the training sample
start_date_test = datetime.datetime(2010, 1, 2)
end_date_test = datetime.datetime(2018, 12, 31)
# -

# # 2. Load dataset

path = 'data/'+'dataset_3'+start_date.strftime("%m_%d_%Y")+'to'+end_date.strftime("%m_%d_%Y")+'.pkl'
dataset = pd.read_pickle(path)

dataset.index = pd.to_datetime(dataset.index)

print('Total dataset has {} samples, and {} features.'.format(dataset.shape[0], \
                                                              dataset.shape[1]))

# # 3. Feature importance w/ XGBoost

dataset.columns

price_only = ['GS_close','ma7', 'ma21','26ema', '12ema', 'MACD', '20sd', 
              'upper_band', 'lower_band', 'ema','momentum', 'log_momentum', 'GS_close_fft_3_real',]

dataset = dataset[price_only]

# Split training and testing sample
X_train = dataset[dataset.index < end_date_train].copy().drop(columns=[symbol+'_close'])
y_train = dataset[dataset.index < end_date_train][symbol+'_close'].copy()
X_test = dataset[dataset.index >= end_date_train].copy().drop(columns=[symbol+'_close'])
y_test = dataset[dataset.index >= end_date_train][symbol+'_close'].copy()

xgb = XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
xgb.fit(X_train, y_train)

features = dataset.columns.values.tolist()
features.remove(symbol+'_close')

plt.barh(features, xgb.feature_importances_)

perm_importance = permutation_importance(xgb, X_test, y_test)

sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(dataset.columns.values[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")

# ### SHAP

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")



