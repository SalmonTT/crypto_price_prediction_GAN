import pandas as pd
import numpy as np
import GAN_fn as gan_fn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# ----- Configurations ----- #
opt = {"lr": 0.00016, "epoch": 2, 'bs': 128}


# Load data
config = '3_1'
train_test_path = 'train_test_data/'
X_train = np.load(f"{train_test_path}X_train_val_{config}.npy", allow_pickle=True)
y_train = np.load(f"{train_test_path}y_train_val_{config}.npy", allow_pickle=True)
yc_train = np.load(f"{train_test_path}yc_train_val_{config}.npy", allow_pickle=True)
X_test = np.load(f"{train_test_path}X_test_{config}.npy", allow_pickle=True)
y_test = np.load(f"{train_test_path}y_test_{config}.npy", allow_pickle=True)
yc_test = np.load(f"{train_test_path}yc_test_{config}.npy", allow_pickle=True)
x_scaler = pd.read_pickle(f'{train_test_path}x_scaler_{config}.pkl')
y_scaler = pd.read_pickle(f'{train_test_path}y_scaler_{config}.pkl')
train_predict_index = np.load(f"{train_test_path}index_train_val_{config}.npy", allow_pickle=True)
test_predict_index = np.load(f"{train_test_path}index_test_{config}.npy", allow_pickle=True)

input_dim = X_train.shape[1]
feature_size = X_train.shape[2]
output_dim = y_train.shape[1]

## For Bayesian
generator = gan_fn.Generator(X_train.shape[1], output_dim, X_train.shape[2])
discriminator = gan_fn.Discriminator()
gan = gan_fn.GAN(generator, discriminator, opt)
pred_price, real_price, rmse = gan.train(X_train, y_train, yc_train, opt)

# Convert into dataframe
price_df = pd.DataFrame(index=train_predict_index)
for i in range(len(pred_price)):
    # rescale prices
    price_df[f'pred_price_{i}'] = y_scaler.inverse_transform(pred_price[i].reshape(-1, 1))
# rescale and add real prices
price_df['real_price'] = y_scaler.inverse_transform(real_price.reshape(-1, 1))

# ------ Plot the result ------ #
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(y=price_df[f"pred_price_{opt['epoch']-1}"],
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
        text=f"Basic GAN Model Predicted Prices vs. Real Prices<br><sup></sup>",
        xref="paper",
        x=0
    ))
fig.write_image("images/basic_gan_prices.png")
fig.show()

# ----- plot error ----- #
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(y=rmse, x=list(range(1, opt['epoch'])), name='RMSE'), secondary_y=False)
# Set title
fig.update_xaxes(title_text="Epoch")
fig.update_yaxes(title_text='Error', secondary_y=False)
fig.update_yaxes(title_text='Error %', secondary_y=True)
fig.update_layout(
    legend=dict(orientation="h", y=1.2),
    title=go.layout.Title(
        text=f"Basic GAN Model Error vs. Epochs<br><sup></sup>",
        xref="paper",
        x=0
    ))
fig.write_image("images/basic_gan_rmse.png")
fig.show()