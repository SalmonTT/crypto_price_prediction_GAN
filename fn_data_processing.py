import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_technical_indicators(dataset, price):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset[price].rolling(window=7).mean()
    dataset['ma21'] = dataset[price].rolling(window=21).mean()

    # Create MACD
    dataset['26ema'] = dataset[price].ewm(span=26).mean()
    dataset['12ema'] = dataset[price].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])

    # Create Bollinger Bands
    dataset['20sd'] = dataset[price].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)
    # Create Exponential moving average
    dataset['ema'] = dataset[price].ewm(com=0.5).mean()

    # Create Momentum
    # Calculate the price change
    delta = dataset[price].diff()
    # Calculate the gain and loss from the price change
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # Calculate the average gain and average loss for the RSI period
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss
    # Calculate the RSI using the RS
    rsi = 100 - (100 / (1 + rs))
    # Add the RSI to the DataFrame
    dataset['RSI'] = rsi
    return dataset

def get_fourier_transfer(dataset, price):
    # Get the columns for doing fourier
    data_FT = dataset[['Timestamp', price]]

    close_fft = np.fft.fft(np.asarray(data_FT[price].tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    fft_list = np.asarray(fft_df['fft'].tolist())
    fft_com_df = pd.DataFrame()
    for num_ in [3, 6, 9]:
        fft_list_m10 = np.copy(fft_list);
        fft_list_m10[num_:-num_] = 0
        fft_ = np.fft.ifft(fft_list_m10)
        fft_com = pd.DataFrame({'fft': fft_})
        fft_com['absolute_' + str(num_) + '_comp'] = fft_com['fft'].apply(lambda x: np.abs(x))
        fft_com['angle_' + str(num_) + '_comp'] = fft_com['fft'].apply(lambda x: np.angle(x))
        fft_com = fft_com.drop(columns='fft')
        fft_com_df = pd.concat([fft_com_df, fft_com], axis=1)

    return fft_com_df


def plot_technical_indicators(dataset, last_days, price):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0 - last_days

    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ = list(dataset.index)

    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'], label='MA 7', color='g', linestyle='--')
    plt.plot(dataset[price], label='Closing Price', color='b')
    plt.plot(dataset['ma21'], label='MA 21', color='r', linestyle='--')
    plt.plot(dataset['upper_band'], label='Upper Band', color='c')
    plt.plot(dataset['lower_band'], label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators for ETHUSD - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['MACD'], label='MACD', linestyle='-.')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.plot(dataset['RSI'], label='RSI', color='b', linestyle='-')

    plt.legend()
    plt.show()


def plot_Fourier(dataset, price):
    data_FT = dataset[['Timestamp', price]]
    close_fft = np.fft.fft(np.asarray(data_FT[price].tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 9]:
        fft_list_m10 = np.copy(fft_list);
        fft_list_m10[num_:-num_] = 0
        plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
    plt.plot(data_FT[price], label='Real')
    plt.xlabel('Days')
    plt.ylabel('USD')
    plt.title('ETHUSD (vwap) prices & Fourier transforms')
    plt.legend()
    plt.show()

def reformat_features(data, in_steps, out_steps):
    '''data: dataframme of features (X)'''
    if not type(data) == np.ndarray:
        data = data.to_numpy()
    print(data.shape)
    samples = []
    length = int(in_steps)
    n = int(data.shape[0]/length)
    for i in range(0, n):
        sample = data[i:i + length]
        samples.append(sample)
    print(len(samples))
    data = np.array(samples)
    print(data.shape)
    return data

def get_X_y(X_data, y_data, n_steps_in, n_steps_out):
    X = list()
    y = list()
    yc = list()
    length = len(X_data)
    for i in range(0, length):
        X_value = X_data[i: i + n_steps_in][:, :]
        y_value = y_data[i + n_steps_in: i + (n_steps_in + n_steps_out)][:, 0]
        yc_value = y_data[i: i + n_steps_in][:, :]
        if len(X_value) == n_steps_in and len(y_value) == n_steps_out:
            X.append(X_value)
            y.append(y_value)
            yc.append(yc_value)
    return np.array(X), np.array(y), np.array(yc)

def split_train_test(X, y, train_size):
    X_train = X[0:train_size]
    X_test = X[train_size:]
    y_train = y[0:train_size]
    y_test = y[train_size:]
    return X_train, X_test, y_train, y_test