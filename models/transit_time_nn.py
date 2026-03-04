"""
transit_time_nn.py
------------------
Physics-informed neural network for CME transit-time prediction (Case Cross-w (↘)).
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle as sklearn_shuffle

from utils.augmentation import augment_data

# Physical constants
RSUN          = 6.957e5     # km   
CONVERSION_AU = 6.68459e-9  # km -> AU  

# ---------------------------------------------------------------------------
# Custom physics-informed loss 
# ---------------------------------------------------------------------------

def CustomLoss_4_time(y_true, y_pred, i,
                      max_input_4, min_input_4,
                      max_scaler_4, min_scaler_4):
    """
    Cross-w (↘) physics-informed loss function.

    y_true columns: [transit_time_s (col 0), arrival_speed (col 1), a_opt (col 2)]
    y_pred        : predicted transit time (softplus output)
    i             : scaled feature input [v_0, m, A, rho, w]

    Physics term  : lambda_physics * 1e2  * (r_a - 1)^2
    Data term     : lambda_time    * 1e-9 * (t_star - t)^2

    Inverse MinMax unscaling: physical = min + (scaled - min_sc) * (max - min) / (max_sc - min_sc)
    """
    lambda_time    = 0.5
    lambda_physics = 1.0 - lambda_time

    v_0 = tf.cast(tf.gather(i, [0], axis=1), tf.float32)
    m   = tf.cast(tf.gather(i, [1], axis=1), tf.float32)
    A   = tf.cast(tf.gather(i, [2], axis=1), tf.float32)
    rho = tf.cast(tf.gather(i, [3], axis=1), tf.float32)
    w   = tf.cast(tf.gather(i, [4], axis=1), tf.float32)

    # Inverse MinMax scaling [10, 100] -> physical units
    v_0 = min_input_4[0] + (v_0 - min_scaler_4[0]) * (max_input_4[0] - min_input_4[0]) / (max_scaler_4[0] - min_scaler_4[0])
    m   = min_input_4[1] + (m   - min_scaler_4[1]) * (max_input_4[1] - min_input_4[1]) / (max_scaler_4[1] - min_scaler_4[1])
    A   = min_input_4[2] + (A   - min_scaler_4[2]) * (max_input_4[2] - min_input_4[2]) / (max_scaler_4[2] - min_scaler_4[2])
    rho = min_input_4[3] + (rho - min_scaler_4[3]) * (max_input_4[3] - min_input_4[3]) / (max_scaler_4[3] - min_scaler_4[3])
    w   = min_input_4[4] + (w   - min_scaler_4[4]) * (max_input_4[4] - min_input_4[4]) / (max_scaler_4[4] - min_scaler_4[4])

    r_0     = 20.0 * RSUN
    C_fixed = 100.0
    gamma   = C_fixed * A * rho / m

    t   = tf.cast(tf.gather(y_true, [0], axis=1), tf.float32)
    a   = tf.cast(tf.gather(y_true, [2], axis=1), tf.float32)

    t_star = y_pred

    epsilon = -1e-11  

    r_a = CONVERSION_AU * (
        (w - tf.sqrt(-(a + epsilon) / gamma)) * t + r_0
        - (1.0 / gamma) * (
            tf.atan(tf.sqrt(-gamma / (a + epsilon)) * (w - v_0))
            + tf.math.log(
                0.5 * tf.sqrt((a + epsilon) / ((a + epsilon) - gamma * (v_0 - w) ** 2))
                * (tf.exp(-2.0 * (tf.sqrt(-(a + epsilon) * gamma) * t
                                  - tf.atan(tf.sqrt(-gamma / (a + epsilon)) * (v_0 - w)))) + 1.0)
            )
        )
    )

    loss = (lambda_physics * 1e2  * tf.math.pow(r_a - 1.0, 2.0)
            + lambda_time  * 1e-9 * tf.math.pow(t_star - t, 2.0))
    return loss


# ---------------------------------------------------------------------------
# Model architecture 
# ---------------------------------------------------------------------------

def build_model(initializer):
    """
    Input(5) -> Dense(200,relu) -> Dropout(0.4) -> Dense(150,relu) -> Dropout(0.4)
             -> Dense(100,relu) -> Dense(75,relu) -> Dense(50,relu)
             -> Dense(30,relu) -> Dense(25,relu) -> Dense(10,relu)
             -> Dense(1, softplus)
    Two-input: [features (5,), target (1,)]
    """
    inp = tf.keras.Input(shape=(5,))
    x   = Dense(200, activation='relu', kernel_initializer=initializer)(inp)
    x   = Dropout(0.4)(x)
    x   = Dense(150, activation='relu', kernel_initializer=initializer)(x)
    x   = Dropout(0.4)(x)
    x   = Dense(100, activation='relu', kernel_initializer=initializer)(x)
    x   = Dense(75,  activation='relu', kernel_initializer=initializer)(x)
    x   = Dense(50,  activation='relu', kernel_initializer=initializer)(x)
    x   = Dense(30,  activation='relu', kernel_initializer=initializer)(x)
    x   = Dense(25,  activation='relu', kernel_initializer=initializer)(x)
    x   = Dense(10,  activation='relu', kernel_initializer=initializer)(x)
    out = Dense(1,   activation='softplus', kernel_initializer=initializer)(x)

    target = tf.keras.Input((1,))          
    model  = keras.Model([inp, target], out)
    return model, inp, out, target


# ---------------------------------------------------------------------------
# Main training loop 
# ---------------------------------------------------------------------------

def run_realizations(
        X_tot_4_extended,   # pd.DataFrame, shape (N, 6): 5 features + flag col (0=real)
        y_4,                # pd.DataFrame, shape (N, 3): [t, v_arr, a_opt]
        X_tot_4_temp,       # pd.DataFrame, shape (N, 5): 5 features, no flag col
        split_4_df,         # pd.Series, shape (N,): stratification labels (0/1)
        uniform_split_val,  # function from data_loading.py
        opt_and_clean_fn,   # function from optimization.py
        initial_seed=57,
        num_realizations=25,
        N_epochs=1000,
        max_retries=3,
        min_epochs=131,
):
    """
    Returns a dict of metric lists.
    """
    list_mae_t_train_tot,    list_mae_t_val_tot,    list_mae_t_test_tot    = [], [], []
    list_mae_t_train_real,   list_mae_t_val_real,   list_mae_t_test_real   = [], [], []
    list_mae_t_train_aug,    list_mae_t_val_aug,    list_mae_t_test_aug    = [], [], []
    list_medae_t_train_tot,  list_medae_t_val_tot,  list_medae_t_test_tot  = [], [], []
    list_medae_t_train_real, list_medae_t_val_real, list_medae_t_test_real = [], [], []
    list_medae_t_train_aug,  list_medae_t_val_aug,  list_medae_t_test_aug  = [], [], []
    list_abs_error_t_train_real, list_abs_error_t_val_real                 = [], []
    list_abs_error_t_train_aug,  list_abs_error_t_val_aug                  = [], []
    list_abs_error_t_test_tot                                               = []
    list_rel_error_t_train_real, list_rel_error_t_val_real                 = [], []
    list_rel_error_t_train_aug,  list_rel_error_t_val_aug                  = [], []
    list_rel_error_t_test_tot                                               = []

    for i in range(num_realizations):
        print(f"Running realization {i+1}/{num_realizations}")

        seed    = initial_seed + i
        retries = 0
        training_success = False

        while retries < max_retries and not training_success:
            print(f"\nTry {retries + 1} with seed {seed}")

            initializer = tf.keras.initializers.GlorotUniform(seed=seed)

            # --- REAL split ---
            X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx = \
                uniform_split_val(
                    X_tot_4_extended, y_4,
                    stratify_column=split_4_df,
                    test_size=0.2, val_size=0.1, random_state=seed
                )

            # --- REAL TEMP (for augmentation — no flag col) ---
            X_train_temp = X_tot_4_temp.iloc[train_idx]
            X_val_temp   = X_tot_4_temp.iloc[val_idx]
            X_test_temp  = X_tot_4_temp.iloc[test_idx]

            y_train_temp = y_4.iloc[train_idx]
            y_val_temp   = y_4.iloc[val_idx]
            y_test_temp  = y_4.iloc[test_idx]

            # --- Data augmentation ---
            X_aug_train, y_aug_train = augment_data(
                X_train_temp, y_train_temp,
                num_samples_per_real=100, min_increase=0.05, max_increase=0.10,
                random_seed=seed
            )
            X_aug_val, y_aug_val = augment_data(
                X_val_temp, y_val_temp,
                num_samples_per_real=100, min_increase=0.05, max_increase=0.10,
                random_seed=seed
            )

            # opt_and_clean re-optimizes a; adds flag col of 2s to X_aug
            X_aug_train, y_aug_train = opt_and_clean_fn(X_aug_train, y_aug_train)
            X_aug_val,   y_aug_val   = opt_and_clean_fn(X_aug_val,   y_aug_val)

            # --- Merge real + augmented ---
            X_train = np.vstack([X_train, X_aug_train])
            y_train = np.vstack([y_train, y_aug_train])
            X_val   = np.vstack([X_val,   X_aug_val])
            y_val   = np.vstack([y_val,   y_aug_val])
            X_test  = np.array(X_test)
            y_test  = np.array(y_test)

            X_train, y_train = sklearn_shuffle(X_train, y_train, random_state=seed)
            X_val,   y_val   = sklearn_shuffle(X_val,   y_val,   random_state=seed)

            # --- Strip flag column, store separately ---
            X_train_extra = X_train[:, -1].reshape(-1, 1)
            X_train       = X_train[:, :-1]

            X_val_extra   = X_val[:, -1].reshape(-1, 1)
            X_val         = X_val[:, :-1]

            X_test_extra  = X_test[:, -1].reshape(-1, 1)
            X_test        = X_test[:, :-1]

            # --- MinMax scaling [10, 100] ---
            scaling   = MinMaxScaler(feature_range=(10, 100))
            X_total   = np.vstack([X_train, X_val, X_test])
            scaling.fit(X_total)

            X_train = scaling.transform(X_train)
            X_val   = scaling.transform(X_val)
            X_test  = scaling.transform(X_test)

            max_input_4  = np.array([np.max(X_total[:, k]) for k in range(5)])
            min_input_4  = np.array([np.min(X_total[:, k]) for k in range(5)])
            min_scaler_4 = np.ones(5) * 10.0
            max_scaler_4 = np.ones(5) * 100.0

            # --- Cast ---
            X_train = X_train.astype(np.float32)
            y_train = np.array(y_train).astype(np.float32)
            X_val   = X_val.astype(np.float32)
            y_val   = np.array(y_val).astype(np.float32)
            X_test  = X_test.astype(np.float32)
            y_test  = np.array(y_test).astype(np.float32)

            # --- Bookkeeping arrays (for print_split_info) ---
            X_train_count = np.hstack([X_train, X_train_extra])
            X_val_count   = np.hstack([X_val,   X_val_extra])
            X_test_count  = np.hstack([X_test,  X_test_extra])

            def print_split_info(name, X_split):
                real_count = (X_split[:, -1] == 0).sum()
                aug_count  = (X_split[:, -1] == 2).sum()
                total      = len(X_split)
                print(f"{name} - Total: {total}")
                print(f"  Real:    {real_count} ({real_count/total*100:.2f}%)")
                print(f"  Augmented:{aug_count}  ({aug_count/total*100:.2f}%)")

            print_split_info("Training set",   X_train_count)
            print_split_info("Validation set", X_val_count)
            print_split_info("Test set",       X_test_count)

            # --- Build model ---
            model, inp, out, target = build_model(initializer)
            opt = keras.optimizers.Adam(learning_rate=1e-3)
            model.add_loss(CustomLoss_4_time(
                target, out, inp,
                max_input_4, min_input_4, max_scaler_4, min_scaler_4
            ))
            model.compile(loss=None, optimizer=opt)

            early_stopper = EarlyStopping(
                monitor='val_loss', restore_best_weights=True, patience=100)
            checkpoint = ModelCheckpoint(
                'best_model.h5', monitor='val_loss', save_best_only=True)
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=50, min_lr=1e-7)

            # --- Fit ---
            history = model.fit(
                x=[np.double(X_train), np.double(y_train)],
                y=None,
                validation_data=([np.double(X_val), np.double(y_val)], None),
                epochs=N_epochs,
                batch_size=16,
                callbacks=[checkpoint, reduce_lr, early_stopper],
                shuffle=True,
                verbose=0
            )

            if early_stopper.stopped_epoch < min_epochs:
                print(f"Early stopping too soon. Trying with new seed.")
                print(f"LAST EPOCH: {len(history.epoch)}")
                seed    += 1000
                retries += 1
            else:
                training_success = True

        if not training_success:
            continue

        print(f"LAST EPOCH: {len(history.epoch)}")

        best_epoch    = np.argmin(history.history['val_loss'])
        best_filename = f'best_model_epoch_{best_epoch + 1}_{N_epochs}.h5'
        model.save(best_filename)
        print(f"Best model saved: {best_filename}")
        fine_tune_model = keras.models.load_model(best_filename, compile=False)

        # ------------------------------------------------------------------ TRAIN
        predictions = fine_tune_model.predict([np.double(X_train), np.double(y_train)])
        y_train_t   = np.double(y_train[:, 0]).reshape(-1, 1)

        mae_t_train   = np.mean(np.abs(predictions - y_train_t) / 3600)
        medae_t_train = np.median(np.abs(predictions - y_train_t) / 3600)
        list_mae_t_train_tot.append(mae_t_train)
        list_medae_t_train_tot.append(medae_t_train)
        print(f'MAE train t. (Total): {mae_t_train}')

        real_mask_train = (X_train_extra == 0).ravel()
        aug_mask_train  = (X_train_extra == 2).ravel()

        mae_t_train_real   = np.mean(np.abs(predictions[real_mask_train] - y_train_t[real_mask_train]) / 3600)
        medae_t_train_real = np.median(np.abs(predictions[real_mask_train] - y_train_t[real_mask_train]) / 3600)
        list_mae_t_train_real.append(mae_t_train_real)
        list_medae_t_train_real.append(medae_t_train_real)
        print(f'MAE train t (Real): {mae_t_train_real}')

        mae_t_train_aug   = np.mean(np.abs(predictions[aug_mask_train] - y_train_t[aug_mask_train]) / 3600)
        medae_t_train_aug = np.median(np.abs(predictions[aug_mask_train] - y_train_t[aug_mask_train]) / 3600)
        list_mae_t_train_aug.append(mae_t_train_aug)
        list_medae_t_train_aug.append(medae_t_train_aug)

        abs_err_real = np.abs(predictions[real_mask_train] - y_train_t[real_mask_train]) / 3600
        list_abs_error_t_train_real.extend(abs_err_real)
        list_rel_error_t_train_real.extend(abs_err_real / (y_train_t[real_mask_train] / 3600))

        abs_err_aug = np.abs(predictions[aug_mask_train] - y_train_t[aug_mask_train]) / 3600
        list_abs_error_t_train_aug.extend(abs_err_aug)
        list_rel_error_t_train_aug.extend(abs_err_aug / (y_train_t[aug_mask_train] / 3600))

        # ------------------------------------------------------------------ VAL
        predictions = fine_tune_model.predict([np.double(X_val), np.double(y_val)])
        y_val_t     = np.double(y_val[:, 0]).reshape(-1, 1)

        mae_t_val   = np.mean(np.abs(predictions - y_val_t) / 3600)
        medae_t_val = np.median(np.abs(predictions - y_val_t) / 3600)
        list_mae_t_val_tot.append(mae_t_val)
        list_medae_t_val_tot.append(medae_t_val)
        print(f'MAE val t. (Total): {mae_t_val}')

        real_mask_val = (X_val_extra == 0).ravel()
        aug_mask_val  = (X_val_extra == 2).ravel()

        mae_t_val_real   = np.mean(np.abs(predictions[real_mask_val] - y_val_t[real_mask_val]) / 3600)
        medae_t_val_real = np.median(np.abs(predictions[real_mask_val] - y_val_t[real_mask_val]) / 3600)
        list_mae_t_val_real.append(mae_t_val_real)
        list_medae_t_val_real.append(medae_t_val_real)
        print(f'MAE val t (Real): {mae_t_val_real}')

        mae_t_val_aug   = np.mean(np.abs(predictions[aug_mask_val] - y_val_t[aug_mask_val]) / 3600)
        medae_t_val_aug = np.median(np.abs(predictions[aug_mask_val] - y_val_t[aug_mask_val]) / 3600)
        list_mae_t_val_aug.append(mae_t_val_aug)
        list_medae_t_val_aug.append(medae_t_val_aug)

        abs_err_real_v = np.abs(predictions[real_mask_val] - y_val_t[real_mask_val]) / 3600
        list_abs_error_t_val_real.extend(abs_err_real_v)
        list_rel_error_t_val_real.extend(abs_err_real_v / (y_val_t[real_mask_val] / 3600))

        abs_err_aug_v = np.abs(predictions[aug_mask_val] - y_val_t[aug_mask_val]) / 3600
        list_abs_error_t_val_aug.extend(abs_err_aug_v)
        list_rel_error_t_val_aug.extend(abs_err_aug_v / (y_val_t[aug_mask_val] / 3600))

        # ------------------------------------------------------------------ TEST
        predictions = fine_tune_model.predict([np.double(X_test), np.double(y_test)])
        y_test_t    = np.double(y_test[:, 0]).reshape(-1, 1)

        mae_t_test   = np.mean(np.abs(predictions - y_test_t) / 3600)
        medae_t_test = np.median(np.abs(predictions - y_test_t) / 3600)
        list_mae_t_test_tot.append(mae_t_test)
        list_medae_t_test_tot.append(medae_t_test)
        print(f'MAE test t mean: {mae_t_test}')
        print(f'MAE test t min:  {np.min(np.abs(predictions - y_test_t)/3600)}')
        print(f'MAE test t max:  {np.max(np.abs(predictions - y_test_t)/3600)}')

        abs_err_test = np.abs(predictions - y_test_t) / 3600
        list_abs_error_t_test_tot.extend(abs_err_test)
        list_rel_error_t_test_tot.extend(abs_err_test / (y_test_t / 3600))

    return {
        'mae_train_tot':    list_mae_t_train_tot,
        'mae_val_tot':      list_mae_t_val_tot,
        'mae_test_tot':     list_mae_t_test_tot,
        'mae_train_real':   list_mae_t_train_real,
        'mae_val_real':     list_mae_t_val_real,
        'mae_train_aug':    list_mae_t_train_aug,
        'mae_val_aug':      list_mae_t_val_aug,
        'medae_train_tot':  list_medae_t_train_tot,
        'medae_val_tot':    list_medae_t_val_tot,
        'medae_test_tot':   list_medae_t_test_tot,
        'medae_train_real': list_medae_t_train_real,
        'medae_val_real':   list_medae_t_val_real,
        'medae_train_aug':  list_medae_t_train_aug,
        'medae_val_aug':    list_medae_t_val_aug,
        'abs_error_train_real': list_abs_error_t_train_real,
        'abs_error_val_real':   list_abs_error_t_val_real,
        'abs_error_train_aug':  list_abs_error_t_train_aug,
        'abs_error_val_aug':    list_abs_error_t_val_aug,
        'abs_error_test_tot':   list_abs_error_t_test_tot,
        'rel_error_train_real': list_rel_error_t_train_real,
        'rel_error_val_real':   list_rel_error_t_val_real,
        'rel_error_train_aug':  list_rel_error_t_train_aug,
        'rel_error_val_aug':    list_rel_error_t_val_aug,
        'rel_error_test_tot':   list_rel_error_t_test_tot,
    }