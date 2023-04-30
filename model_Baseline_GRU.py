import pandas as pd
import numpy as np
import fn_gru as gru

def run_gru(configs):
    timesteps, path, epoch, batch_size, lr, custom_lr = configs

    # Load data
    X_train = np.load(f"{path}X_train_{timesteps}.npy", allow_pickle=True)
    y_train = np.load(f"{path}y_train_{timesteps}.npy", allow_pickle=True)
    X_val = np.load(f"{path}X_val_{timesteps}.npy", allow_pickle=True)
    y_val = np.load(f"{path}y_val_{timesteps}.npy", allow_pickle=True)
    X_test = np.load(f"{path}X_test_{timesteps}.npy", allow_pickle=True)
    y_test = np.load(f"{path}y_test_{timesteps}.npy", allow_pickle=True)
    
    y_scaler = pd.read_pickle(f'{path}y_scaler_{timesteps}.pkl')
    train_predict_index = np.load(f"{path}index_train_{timesteps}.npy", allow_pickle=True)
    val_predict_index = np.load(f"{path}index_val_{timesteps}.npy", allow_pickle=True)
    test_predict_index = np.load(f"{path}index_test_{timesteps}.npy", allow_pickle=True)

    print(f"input_dim: {X_train.shape[1]}, feature_size: {X_train.shape[2]}, "
          f"output_dim: {y_train.shape[1]}")

    ## ------ Run Model Training ------ ##
    model = gru.basic_GRU(X_train, y_train, X_val, y_val,
                          lr, batch_size, epoch, custom_lr)
    print(model.summary())
    # model.save('GRU_30to3.h5')

    ## ------ Plot results ------ ##
    train_RMSE = gru.plot_prediction(model, X_train, y_train, y_scaler, train_predict_index, 'Training')
    print("----- Train_RMSE_GRU -----", train_RMSE)

    val_RMSE = gru.plot_prediction(model, X_val, y_val, y_scaler, val_predict_index, 'Validation')
    print("----- Val_RMSE_GRU -----", val_RMSE)

    test_RMSE = gru.plot_prediction(model, X_test, y_test, y_scaler, test_predict_index, 'Test')
    print("----- Test_RMSE_GRU -----", test_RMSE)
    pass
