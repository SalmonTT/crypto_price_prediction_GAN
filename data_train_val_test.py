import pandas as pd
import numpy as np
import fn_data_processing as utils
from sklearn.model_selection import train_test_split
from pickle import dump
from sklearn.preprocessing import MinMaxScaler

def generate_train_val_test(configs):
    source, target, train_size_ratio, trainval_size_ratio, n_steps_in, n_steps_out = configs
    # Load dataset
    dataset = pd.read_pickle(source)
    # add technical indicator to the price of ETH
    dataset = utils.get_technical_indicators(dataset, target)
    # fourier transform
    fourier_df = utils.get_fourier_transfer(dataset, target)
    final_data = pd.concat([dataset, fourier_df], axis=1)
    # utils.plot_Fourier(dataset, target)
    # utils.plot_technical_indicators(final_data, 300, target)

    # Set the date to datetime data
    final_data['Timestamp'] = pd.to_datetime(final_data['Timestamp'])
    final_data = final_data.set_index(['Timestamp'])
    final_data = final_data.sort_index()

    # Drop NaN
    final_data = final_data.dropna()
    # Get features and target (ETHUSD_vwap)
    feature_list = final_data.columns.to_list()
    # feature_list.remove(target)
    print(feature_list)
    X = final_data[feature_list]
    y = final_data[[target]]

    # Auto-correlation Check
    # sm.graphics.tsa.plot_acf(y.squeeze(), lags=100)
    # plt.show()

    # Split into test (out of sample) and in-sample dataset (training and validation)
    train_val_size = int(X.shape[0]*trainval_size_ratio)
    X_train_val, X_test, y_train_val, y_test, = utils.split_train_test(X, y, train_val_size)
    # further split train_val into training and validation sets
    train_size = int(X_train_val.shape[0]*train_size_ratio)
    X_train, X_val, y_train, y_val = utils.split_train_test(X_train_val, y_train_val, train_size)

    # Normalization X (using training data to fit)
    X_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaler.fit(X_train)
    X_train_sc = X_scaler.transform(X_train)
    X_val_sc = X_scaler.transform(X_val)
    X_test_sc = X_scaler.transform(X_test)
    # Normalization y (using training data to fit)
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler.fit(y_train)
    y_train_sc = y_scaler.transform(y_train)
    y_val_sc = y_scaler.transform(y_val)

    # reshape data for LSTM:
    n_features = X.shape[1]
    print(f"number of features: {n_features}")
    # Reshape X and y for ML model
    X_train_reshaped, y_train_reshaped, y_train_c = utils.get_X_y(X_train_sc, y_train_sc, n_steps_in, n_steps_out)
    X_val_reshaped, y_val_reshaped, y_val_c = utils.get_X_y(X_val_sc, y_val_sc, n_steps_in, n_steps_out)
    X_test_reshaped, y_test_reshaped, y_test_c = utils.get_X_y(X_test_sc, y_test.values, n_steps_in, n_steps_out)

    # Get index
    index_train = X_train.index.to_numpy()[n_steps_in:]
    index_val = X_val.index.to_numpy()[n_steps_in:]
    index_test = X_test.index.to_numpy()[n_steps_in:]

    print(f'X_Train{X_train_reshaped.shape}, y_train {y_train_reshaped.shape}')
    print(f'X_val{X_val_reshaped.shape}, y_val {y_val_reshaped.shape}')
    print(f'X_test{X_test_reshaped.shape}, y_test {y_test_reshaped.shape}')

    # save to local
    training_data_path = 'train_val_test_data/'
    config = str(n_steps_in)+'_'+str(n_steps_out)
    np.save(training_data_path+f"X_train_{config}.npy", X_train_reshaped)
    np.save(training_data_path+f"y_train_{config}.npy", y_train_reshaped)
    np.save(training_data_path+f"yc_train_{config}.npy", y_train_c)
    np.save(training_data_path+f"X_val_{config}.npy", X_val_reshaped)
    np.save(training_data_path+f"y_val_{config}.npy", y_val_reshaped)
    np.save(training_data_path+f"yc_val_{config}.npy", y_val_c)
    np.save(training_data_path+f"X_test_{config}.npy", X_test_reshaped)
    np.save(training_data_path+f"y_test_{config}.npy", y_test_reshaped)
    np.save(training_data_path+f"yc_test_{config}.npy", y_test_c)
    np.save(training_data_path+f'index_train_{config}.npy', index_train)
    np.save(training_data_path+f'index_val_{config}.npy', index_val)
    np.save(training_data_path+f'index_test_{config}.npy', index_test)
    dump(X_scaler, open(f'{training_data_path}X_scaler_{config}.pkl', 'wb'))
    dump(y_scaler, open(f'{training_data_path}y_scaler_{config}.pkl', 'wb'))
    pass