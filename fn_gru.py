import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error


def basic_lstm(X_train, y_train, X_val, y_val, LR, BATCH_SIZE, N_EPOCH, custom_lr) -> tf.keras.models.Model:
    input_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_dim = y_train.shape[1]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128), input_shape=(input_dim, feature_size)))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(units=output_dim))
    model.compile(optimizer=tf.optimizers.Adam(lr=LR), loss='mse')
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=LR), loss='mse')
    # add custom scheduler and early stop loss
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    if custom_lr:
        history = model.fit(X_train, y_train, epochs=N_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
                            verbose=2, shuffle=False, callbacks=[lr_callback])
    else:
        history = model.fit(X_train, y_train, epochs=N_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
                            verbose=2, shuffle=False)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.show()

    return model
def basic_GRU(X_train, y_train, X_val, y_val, LR, BATCH_SIZE, N_EPOCH, custom_lr) -> tf.keras.models.Model:
    input_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_dim = y_train.shape[1]
    def Model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(units=128, return_sequences=True, input_shape=(input_dim, feature_size)),
            # tf.keras.layers.GRU(units=256, recurrent_dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(units=64, input_shape=(input_dim, feature_size)),
            # tf.keras.layers.Dense(units=128),
            tf.keras.layers.Dense(units=32),
            # tf.keras.layers.Dense(units=32),
            tf.keras.layers.Dense(units=output_dim)
        ])
        return model

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    # Use CPU to train
    with tf.device('/CPU:0'):
        model = Model()
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=LR), loss='mse')
        if custom_lr:
            # with callbacks
            history = model.fit(X_train, y_train, epochs=N_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
                            shuffle=False, callbacks=[lr_callback])
        else:
            # no callbacks
            history = model.fit(X_train, y_train, epochs=N_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
                            verbose=2, shuffle=False)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.savefig(f"images/GRU_loss_{input_dim}_{output_dim}_{N_EPOCH}_{BATCH_SIZE}_{LR}_{custom_lr}.png")
    plt.show()

    return model

def plot_prediction(model, X, y, y_scaler, predict_index, dataset_name, model_name, configs):
    timesteps, path, epoch, batch_size, lr, custom_lr = configs
    prediction = model.predict(X, verbose=0)
    if dataset_name != 'Test':
        y = y_scaler.inverse_transform(y)
    prediction = y_scaler.inverse_transform(prediction)
    df = pd.DataFrame(index=predict_index)
    df['prediction'] = prediction.T[0]
    df['actual'] = y.T[0]

    # Plot the predicted result
    plt.figure(figsize=(16, 8))
    plt.plot(df["actual"])
    plt.plot(df["prediction"], color='r')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(("Actual price", "Predicted price"), loc="upper left", fontsize=16)
    plt.title(f"{dataset_name}", fontsize=20)
    if dataset_name == 'Test':
        plt.savefig(f"images/{model_name}_outsample_{timesteps}_{epoch}_{batch_size}_{lr}_{custom_lr}.png")
    plt.show()

    # Calculate RMSE
    RMSE = np.sqrt(mean_squared_error(df["prediction"], df["actual"]))
    pred_acc = price_trend_acc(df["prediction"], df["actual"])
    print(f"Price Prediction Accuracy is {pred_acc}")
    return RMSE

def scheduler(epoch):
    if epoch <= 150:
        lrate = (10 ** -5) * (epoch / 150)
    elif epoch <= 400:
        initial_lrate = (10 ** -5)
        k = 0.01
        lrate = initial_lrate * math.exp(-k * (epoch - 150))
    else:
        lrate = (10 ** -6)

    return lrate

def price_trend_acc(pred, actual):
    '''
    pred: list/series of predicted prices
    actual: list/series of actual prices
    Check whether the predicted price has the same price trend as actual
    '''
    d = {'pred':pred, 'actual':actual}
    df = pd.DataFrame(d)
    df['pred_diff'] = df['pred'].diff().fillna(0)
    df['pred_trend'] = np.where(df['pred_diff'] > 0, 1, 0)
    df['actual_diff'] = df['actual'].diff().fillna(0)
    df['actual_trend'] = np.where(df['actual_diff'] > 0, 1, 0)
    df['trend_result'] = df['pred_trend'] == df['actual_trend']
    prediction_accuracy = df['trend_result'].value_counts(normalize=True).values[0]
    return prediction_accuracy
