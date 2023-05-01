from data_train_test import generate_train_test
from data_train_val_test import generate_train_val_test
from model_Baseline_GRU import run_gru
from model_Baseline_LSTM import run_lstm
from model_Basic_GAN import run_basic_gan
from model_WGAN_GP import run_wgan
from test_GAN_prediction import run_gan_predictions

generate_data = False
# ---- Generate Dataset ---- #
if generate_data:
    # common configs
    source = 'data/dataset_1.pkl'
    target = 'ETHUSD_vwap'
    n_steps_in = 30 # 3, 15, 30
    n_steps_out = 3 # 1, 2, 3

    # -- Train test data (for GAN and WGAN Models) -- #
    # configs:
    split_ratio = 0.7
    data_configs = source, target, split_ratio, n_steps_in, n_steps_out
    generate_train_test(data_configs)

    # -- Train val test data (for GRU and LSTM Models) -- #
    # configs:
    trainval_split_ratio = 0.95
    data_configs = source, target, split_ratio, trainval_split_ratio, n_steps_in, n_steps_out
    generate_train_val_test(data_configs)


# ---- Run Models ----- #
# Common Configs
timesteps = ['3_1', '15_2', '30_3']
lstm_gru_path = 'train_val_test_data/'
gan_path = 'train_test_data/'
i = 0 # an index to set the params from lists

# GRU
epoch = 50
batch_size = 128
lr = 0.0001
custom_lr = False # If true, will use a custom learning rate scheduler
gru_configs = timesteps[i], lstm_gru_path, epoch, batch_size, lr, custom_lr
# run_gru(gru_configs)


# LSTM
epoch = 50 # 50,
batch_size = 64 # 64,
lr = 0.001 # 0.001
custom_lr = False # If true, will use a custom learning rate scheduler
lstm_configs = timesteps[i], lstm_gru_path, epoch, batch_size, lr, custom_lr
# run_lstm(lstm_configs)

# -- GAN Models -- #
# GAN
epoch = 200 # 165,
batch_size = 128 # 128,
lr = 0.00016 # 0.00016,
gan_type = 'gan'
gan_configs = timesteps[i], gan_path, epoch, batch_size, lr
run_basic_gan(gan_configs)
# Outsample
gan_pred_configs = gan_type, timesteps[i], gan_path, epoch, batch_size, lr
run_gan_predictions(gan_pred_configs)

# WGAN
epoch = 100
batch_size = 100
lr = 0.0001
gan_type = 'wgan'
wgan_configs = timesteps[i], gan_path, epoch, batch_size, lr
# run_wgan(wgan_configs)
# Outsample
wgan_pred_configs = gan_type, timesteps[i], gan_path, epoch, batch_size, lr
# run_gan_predictions(wgan_pred_configs)