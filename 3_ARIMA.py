from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

config = '3_1'
train_test_path = 'train_test_data/'

train = np.load(f"{train_test_path}y_train_val_{config}.npy", allow_pickle=True)
test = np.load(f"{train_test_path}y_test_{config}.npy", allow_pickle=True)
y_scaler = pd.read_pickle(f'{train_test_path}y_scaler_{config}.pkl')
train = y_scaler.inverse_transform(train)
# reshape data
train = train.T[0].tolist()
test = test.T[0].tolist()
predictions = list()
for t in range(len(test)):
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    train.append(obs)

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(y=test, name='Actual'), secondary_y=False)
fig.add_trace(go.Scatter(y=predictions, name='predicted'), secondary_y=False)
# Set title
fig.update_xaxes(title_text="days")
fig.update_yaxes(title_text='price', secondary_y=False)
fig.update_layout(
    legend=dict(orientation="h", y=1.2),
    title=go.layout.Title(
        text=f"ARIMA<br><sup></sup>",
        xref="paper",
        x=0
    ))
fig.write_image("images/ARIMA.png")
fig.show()
