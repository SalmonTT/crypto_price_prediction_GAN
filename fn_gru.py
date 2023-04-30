import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import numpy as np
import math

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
    plt.show()

    return model

def plot_prediction(model, X, y, y_scaler, predict_index, dataset_name):
    prediction = model.predict(X, verbose=0)
    if dataset_name != 'Test':
        y = y_scaler.inverse_transform(y)
    prediction = y_scaler.inverse_transform(prediction)
    # (1200, 1)
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
    plt.show()

    # Calculate RMSE
    RMSE = np.sqrt(mean_squared_error(df["prediction"], df["actual"]))

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
