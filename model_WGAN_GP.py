import pandas as pd
import numpy as np
import fn_wgan as gan_fn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run_wgan(configs):
    timesteps, path, epoch, batch_size, lr = configs
    opt = {"lr": lr, "epoch": epoch, 'bs': batch_size, 'timesteps': timesteps, 'input_shape':4}
    # Load data
    X_train = np.load(f"{path}X_train_val_{timesteps}.npy", allow_pickle=True)
    y_train = np.load(f"{path}y_train_val_{timesteps}.npy", allow_pickle=True)
    yc_train = np.load(f"{path}yc_train_val_{timesteps}.npy", allow_pickle=True)
    y_scaler = pd.read_pickle(f'{path}y_scaler_{timesteps}.pkl')
    train_predict_index = np.load(f"{path}index_train_val_{timesteps}.npy", allow_pickle=True)

    output_dim = y_train.shape[1]

    generator = gan_fn.Generator(X_train.shape[1], output_dim, X_train.shape[2])
    discriminator = gan_fn.Discriminator(X_train.shape[1], output_dim)
    opt['input_shape'] = X_train.shape[1] + output_dim
    gan = gan_fn.GAN(generator, discriminator, opt)
    # list of prediction prices, list of rmspe from each epoch and the real prices
    pred_price, real_price, rmse = gan.train(X_train, y_train, yc_train, epoch)
    # Convert into dataframe
    # force shape alignment
    train_predict_index = train_predict_index[:pred_price[0].reshape(-1, 1).shape[0]]
    price_df = pd.DataFrame(index=train_predict_index)
    for i in range(len(pred_price)):
        # rescale prices
        price_df[f'pred_price_{i}'] = y_scaler.inverse_transform(pred_price[i].reshape(-1, 1))
    # rescale and add real prices
    price_df['real_price'] = y_scaler.inverse_transform(real_price.reshape(-1, 1))

    # ------ Plot the result ------ #
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(y=price_df[f"pred_price_{epoch-1}"],
                             x=price_df.index, name='Predicted prices - last epoch'), secondary_y=False)
    fig.add_trace(go.Scatter(y=price_df[f"pred_price_0"],
                             x=price_df.index, name='Predicted prices - first epoch'), secondary_y=False)
    fig.add_trace(go.Scatter(y=price_df['real_price'],
                             x=price_df.index, name='Real prices'), secondary_y=False)
    # Set title
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text='Price', secondary_y=False)
    fig.update_layout(
        legend=dict(orientation="h", y=1),
        title=go.layout.Title(
            text=f"WGAN_GP Model Predicted Prices vs. Real Prices<br><sup>Timestep: {timesteps}, LR: {lr}, Batch Size: "
                 f"{batch_size}, Epochs: {epoch}</sup>",
            xref="paper",
            x=0
        ))
    fig.write_image(f"images/wgan_prices_{timesteps}_{lr}_{batch_size}_{epoch}.png")
    fig.show()

    # ----- plot error ----- #
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(y=rmse, x=list(range(1, epoch)), name='RMSE'), secondary_y=False)
    # Set title
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text='Error', secondary_y=False)
    fig.update_yaxes(title_text='Error %', secondary_y=True)
    fig.update_layout(
        legend=dict(orientation="h", y=1.2),
        title=go.layout.Title(
            text=f"WGAN_GP Model Error vs. Epochs<br><sup>Timestep: {timesteps}, LR: {lr}, Batch Size: "
                 f"{batch_size}, Epochs: {epoch}</sup>",
            xref="paper",
            x=0
        ))
    fig.write_image(f"images/wgan_rmse_{timesteps}_{lr}_{batch_size}_{epoch}.png")
    fig.show()

    pass