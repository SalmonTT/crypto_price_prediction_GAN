import pandas as pd
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

config = '3_1'
train_test_path = 'train_test_data/'
X_test = np.load(f"{train_test_path}X_test_{config}.npy", allow_pickle=True)
y_test = np.load(f"{train_test_path}y_test_{config}.npy", allow_pickle=True)
y_scaler = pd.read_pickle(f'{train_test_path}y_scaler_{config}.pkl')
test_predict_index = np.load(f"{train_test_path}index_test_{config}.npy", allow_pickle=True)


G_model = tf.keras.models.load_model('Models/WGAN_GP_model.h5')
# Get predicted data
y_predicted = G_model(X_test)
y_predicted = y_scaler.inverse_transform(y_predicted)

price_df = pd.DataFrame(index=test_predict_index)
price_df['prediction'] = y_predicted.T[0]
price_df['actual'] = y_test.T[0]

print(price_df)
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(y=price_df['prediction'], x=price_df.index,
                         name='Prediction'), secondary_y=False)
fig.add_trace(go.Scatter(y=price_df['actual'], x=price_df.index,
                         name='Actual'), secondary_y=False)
# Set title
fig.update_xaxes(title_text="Time")
fig.update_yaxes(title_text='price', secondary_y=False)
fig.update_layout(
    legend=dict(orientation="h", y=1.2),
    title=go.layout.Title(
        text=f"WGAN_GP test<br><sup></sup>",
        xref="paper",
        x=0
    ))
fig.write_image("images/WGAN_GP_test.png")
fig.show()