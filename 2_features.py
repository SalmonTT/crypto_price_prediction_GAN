import pandas as pd
import numpy as np
import misc_fn as utils
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load dataset
dataset = pd.read_pickle('data/dataset_1.pkl')
# add technical indicator to the price of BTC
dataset = utils.get_technical_indicators(dataset)
# fourier transform
fourier_df = utils.get_fourier_transfer(dataset)
final_data = pd.concat([dataset, fourier_df], axis=1)
utils.plot_Fourier(dataset)
utils.plot_technical_indicators(final_data, 300)

# Deal with missing values


# Set the date to datetime data
final_data['Timestamp'] = pd.to_datetime(final_data['Timestamp'])
final_data = final_data.set_index(['Timestamp'])
final_data = final_data.sort_index()

# Get features and target (ETHUSD_vwap)
target = 'ETHUSD_vwap'
feature_list = final_data.columns.to_list()
feature_list.remove(target)
print(feature_list)
X = final_data[feature_list]
y = final_data[[target]]

# Auto-correlation Check
sm.graphics.tsa.plot_acf(y.squeeze(), lags=100)
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    shuffle=False)

# Normalization
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X_train)
X_scaled = scaler.transform(X)
scaler.fit(y_train)
y_scaled = scaler.transform(y)

X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(X_scaled, y_scaled,
                                                                test_size=0.3, shuffle=False)
# reshape data for LSTM:
n_steps_in = 3
n_steps_out = 1

X_train = utils.reformat_features(X_train, n_steps_in, n_steps_out)
X_test = utils.reformat_features(X_test, n_steps_in, n_steps_out)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

X_train_sc = utils.reformat_features(X_train_sc, n_steps_in, n_steps_out)
X_test_sc = utils.reformat_features(X_test_sc, n_steps_in, n_steps_out)

# index of training data (Timestamp)
train_len = X_train.shape[0]
test_len = X_test.shape[0]
index_train = X.head(train_len).index.to_numpy()
index_test = X.tail(test_len).index.to_numpy()

# save to local
training_data_path = 'train_test_data/'
config = str(n_steps_in)+'_'+str(n_steps_out)
np.save(training_data_path+f"X_train_{config}.npy", X_train)
np.save(training_data_path+f"y_train_{config}.npy", y_train)
np.save(training_data_path+f"X_test_{config}.npy", X_test)
np.save(training_data_path+f"y_test_{config}.npy", y_test)
np.save(training_data_path+f"yc_train_{config}.npy", y_train_sc)
np.save(training_data_path+f"yc_test_{config}.npy", y_test_sc)
np.save(training_data_path+f'index_train_{config}.npy', index_train)
np.save(training_data_path+f'index_test_{config}.npy', index_test)