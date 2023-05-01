import pandas as pd
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fn_gru as fn_gru

def run_gan_predictions(configs):
    gan_type, timesteps, path, epoch, batch_size, lr = configs

    X_test = np.load(f"{path}X_test_{timesteps}.npy", allow_pickle=True)
    y_test = np.load(f"{path}y_test_{timesteps}.npy", allow_pickle=True)
    y_scaler = pd.read_pickle(f'{path}y_scaler_{timesteps}.pkl')
    test_predict_index = np.load(f"{path}index_test_{timesteps}.npy", allow_pickle=True)
    G_model = tf.keras.models.load_model(f'Models/{gan_type}_model_{timesteps}_{lr}_'
                                                   f'{batch_size}_{epoch}.h5')
    # Get predicted data
    y_predicted = G_model(X_test)
    y_predicted = y_scaler.inverse_transform(y_predicted)
    test_predict_index = test_predict_index[:len(y_predicted.T[0])]
    price_df = pd.DataFrame(index=test_predict_index)

    price_df['prediction'] = y_predicted.T[0]
    price_df['actual'] = y_test.T[0]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(y=price_df['prediction'], x=price_df.index,
                             name='Prediction'), secondary_y=False)
    fig.add_trace(go.Scatter(y=price_df['actual'], x=price_df.index,
                             name='Actual'), secondary_y=False)
    # Set title
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text='price', secondary_y=False)
    fig.update_layout(
        legend=dict(orientation="h", y=1.1),
        title=go.layout.Title(
            text=f"{gan_type} Out-sample prediction<br><sup>{timesteps}_{lr}_{batch_size}_{epoch}</sup>",
            xref="paper",
            x=0
        ))
    fig.write_image(f"images/{gan_type}_test_{timesteps}_{lr}_{batch_size}_{epoch}.png")
    fig.show()

    # price trend prediction accuracy
    pred_acc = fn_gru.price_trend_acc(price_df["prediction"], price_df["actual"])
    print(f"Price Prediction Accuracy is {pred_acc}")
    pass