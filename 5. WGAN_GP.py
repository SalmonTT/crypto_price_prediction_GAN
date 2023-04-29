import pandas as pd
import numpy as np
import WGAN_GP_fn as gan_fn
import matplotlib.pyplot as plt
# Load data
config = '3_1'
train_val_path = './train_test_data/'
X_train = np.load(f"{train_val_path}X_train_{config}.npy", allow_pickle=True)
y_train = np.load(f"{train_val_path}y_train_{config}.npy", allow_pickle=True)
yc_train = np.load(f"{train_val_path}yc_train_{config}.npy", allow_pickle=True)
X_val = np.load(f"{train_val_path}X_val_{config}.npy", allow_pickle=True)
y_val = np.load(f"{train_val_path}y_val_{config}.npy", allow_pickle=True)
yc_val = np.load(f"{train_val_path}yc_val_{config}.npy", allow_pickle=True)
X_test = np.load(f"{train_val_path}X_test_{config}.npy", allow_pickle=True)
y_test = np.load(f"{train_val_path}y_test_{config}.npy", allow_pickle=True)
yc_test = np.load(f"{train_val_path}yc_test_{config}.npy", allow_pickle=True)

input_dim = X_train.shape[1]
feature_size = X_train.shape[2]
output_dim = y_train.shape[1]
epoch = 100

generator = gan_fn.Generator(X_train.shape[1], output_dim, X_train.shape[2])
discriminator = gan_fn.Discriminator()
gan = gan_fn.GAN(generator, discriminator)
Predicted_price, Real_price, RMSPE = gan.train(X_train, y_train, yc_train, epoch)

# %% --------------------------------------- Plot the result -----------------------------------------------------

# Rescale back the real dataset
x_scaler = pd.read_pickle(f'x_scaler_{config}.pkl')
y_scaler = pd.read_pickle(f'y_scaler_{config}.pkl')
train_predict_index = np.load(f"{train_val_path}index_train_{config}.npy", allow_pickle=True)
val_predict_index = np.load(f"{train_val_path}index_val_{config}.npy", allow_pickle=True)
test_predict_index = np.load(f"{train_val_path}index_test_{config}.npy", allow_pickle=True)

print("----- predicted price -----", Predicted_price)

rescaled_Real_price = y_scaler.inverse_transform(Real_price)
rescaled_Predicted_price = y_scaler.inverse_transform(Predicted_price)

print("----- rescaled predicted price -----", rescaled_Predicted_price)
print("----- SHAPE rescaled predicted price -----", rescaled_Predicted_price.shape)

predict_result = pd.DataFrame()
for i in range(rescaled_Predicted_price.shape[0]):
    y_predict = pd.DataFrame(rescaled_Predicted_price[i], columns=["predicted_price"], index=train_predict_index[i:i+output_dim])
    predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)

real_price = pd.DataFrame()
for i in range(rescaled_Real_price.shape[0]):
    y_train = pd.DataFrame(rescaled_Real_price[i], columns=["real_price"], index=train_predict_index[i:i+output_dim])
    real_price = pd.concat([real_price, y_train], axis=1, sort=False)

predict_result['predicted_mean'] = predict_result.mean(axis=1)
real_price['real_mean'] = real_price.mean(axis=1)

# Plot the predicted result
plt.figure(figsize=(16, 8))
plt.plot(real_price["real_mean"])
plt.plot(predict_result["predicted_mean"], color = 'r')
plt.xlabel("Date")
plt.ylabel("Stock price")
plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
plt.title("The result of Training", fontsize=20)
plt.show()
plt.savefig('train_plot.png')

# Calculate RMSE
predicted = predict_result["predicted_mean"]
real = real_price["real_mean"]
For_MSE = pd.concat([predicted, real], axis = 1)
RMSE = np.sqrt(mean_squared_error(predicted, real))
print('-- RMSE -- ', RMSE)
