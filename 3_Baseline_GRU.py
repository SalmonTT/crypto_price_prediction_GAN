import pandas as pd
import numpy as np
import GRU_fn as gru

# Load data
config = '3_1'
train_val_path = 'train_val_test_data/'
X_train = np.load(f"{train_val_path}X_train_{config}.npy", allow_pickle=True)
y_train = np.load(f"{train_val_path}y_train_{config}.npy", allow_pickle=True)
X_val = np.load(f"{train_val_path}X_val_{config}.npy", allow_pickle=True)
y_val = np.load(f"{train_val_path}y_val_{config}.npy", allow_pickle=True)
X_test = np.load(f"{train_val_path}X_test_{config}.npy", allow_pickle=True)
y_test = np.load(f"{train_val_path}y_test_{config}.npy", allow_pickle=True)

y_scaler = pd.read_pickle(f'{train_val_path}y_scaler_{config}.pkl')
train_predict_index = np.load(f"{train_val_path}index_train_{config}.npy", allow_pickle=True)
val_predict_index = np.load(f"{train_val_path}index_val_{config}.npy", allow_pickle=True)
test_predict_index = np.load(f"{train_val_path}index_test_{config}.npy", allow_pickle=True)

# Parameters
LR = 0.0001
BATCH_SIZE = 10
N_EPOCH = 200

print(f"input_dim: {X_train.shape[1]}, feature_size: {X_train.shape[2]}, "
      f"output_dim: {y_train.shape[1]}")

## ------ Run Model Training ------ ##
model = gru.basic_GRU(X_train, y_train, X_val, y_val,
                      LR, BATCH_SIZE, N_EPOCH)
print(model.summary())
# model.save('GRU_30to3.h5')

## ------ Plot results ------ ##

train_RMSE = gru.plot_prediction(model, X_train, y_train, y_scaler, train_predict_index, 'Training')
print("----- Train_RMSE_LSTM -----", train_RMSE)

val_RMSE = gru.plot_prediction(model, X_val, y_val, y_scaler, val_predict_index, 'Validation')
print("----- Train_RMSE_LSTM -----", val_RMSE)

test_RMSE = gru.plot_prediction(model, X_test, y_test, y_scaler, test_predict_index, 'Test')
print("----- Train_RMSE_LSTM -----", test_RMSE)