"""
run_transit_time.py
-------------------

Usage
-----
    python run_transit_time.py
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '2'   # adjust as needed

from utils.data_loading    import load_and_clean, compute_features, add_wind_speed_type, \
                            classify_cases, uniform_split_val
from utils.optimization    import optimize_case4, remove_failed_events, opt_and_clean
from models.transit_time_nn import run_realizations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ICME_PATH = 'Data/ICME_complete_dataset_rev.csv'
OUTPUT_DIR = 'Results'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'results.csv')

SEED = 57
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def save_results(results: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def to_flat(x):
        out = []
        for item in x:
            if hasattr(item, '__iter__'):
                out.extend(np.array(item).ravel().tolist())
            else:
                out.append(float(item))
        return out

    columns = {
        'mae_t_train_real':       results['mae_train_real'],
        'mae_t_train_aug':        results['mae_train_aug'],
        'medae_t_train_real':     results['medae_train_real'],
        'medae_t_train_aug':      results['medae_train_aug'],
        'mae_t_val_real':         results['mae_val_real'],
        'mae_t_val_aug':          results['mae_val_aug'],
        'medae_t_val_real':       results['medae_val_real'],
        'medae_t_val_aug':        results['medae_val_aug'],
        'mae_t_train_tot':        results['mae_train_tot'],
        'mae_t_val_tot':          results['mae_val_tot'],
        'mae_t_test_tot':         results['mae_test_tot'],
        'medae_t_train_tot':      results['medae_train_tot'],
        'medae_t_val_tot':        results['medae_val_tot'],
        'medae_t_test_tot':       results['medae_test_tot'],
        'abs_error_t_train_real': to_flat(results['abs_error_train_real']),
        'abs_error_t_val_real':   to_flat(results['abs_error_val_real']),
        'abs_error_t_test_tot':   to_flat(results['abs_error_test_tot']),
        'abs_error_t_train_aug':  to_flat(results['abs_error_train_aug']),
        'abs_error_t_val_aug':    to_flat(results['abs_error_val_aug']),
        'rel_error_t_train_real': to_flat(results['rel_error_train_real']),
        'rel_error_t_val_real':   to_flat(results['rel_error_val_real']),
        'rel_error_t_test_tot':   to_flat(results['rel_error_test_tot']),
        'rel_error_t_train_aug':  to_flat(results['rel_error_train_aug']),
        'rel_error_t_val_aug':    to_flat(results['rel_error_val_aug']),
    }

    max_len = max(len(v) for v in columns.values())
    padded  = {k: list(v) + [np.nan] * (max_len - len(v)) for k, v in columns.items()}

    df = pd.DataFrame(padded)
    df.to_csv(path, index=False)
    print(f"\nResults saved to: {path}  ({len(df)} rows x {len(df.columns)} columns)")


def main():
    # -----------------------------------------------------------------------
    # 1. Load and clean
    # -----------------------------------------------------------------------
    print("Loading data...")
    df = load_and_clean(ICME_PATH)

    # -----------------------------------------------------------------------
    # 2. Feature engineering
    # -----------------------------------------------------------------------
    X, y, v_0, m, A, rho, w, v, t = compute_features(df)

    # -----------------------------------------------------------------------
    # 3. Case classification
    # -----------------------------------------------------------------------
    print("\nClassifying cases...")
    cases = classify_cases(v_0, v, w)

    case4_mask = cases['Cross-w (↘)']
    X_tot_4    = X[case4_mask]          # (N4, 5)
    y_4        = y[case4_mask]          # (N4, 2): [transit_time_s, arrival_speed]

    # -----------------------------------------------------------------------
    # 4. Wind speed type (stratification label)
    # Must be done BEFORE optimization to preserve index alignment
    # -----------------------------------------------------------------------
    w_speed_series = add_wind_speed_type(df)   # pd.Series aligned with df
    split_4_series = w_speed_series[case4_mask]
    split_4_series = split_4_series.reset_index(drop=True)   # reset to 0-based

    # -----------------------------------------------------------------------
    # 5. Optimize a for Case Cross-w (↘)
    # -----------------------------------------------------------------------
    print("\nOptimizing Case Cross-w (↘) acceleration parameter...")
    ar_results, drop_index = optimize_case4(X_tot_4, y_4, C=100, ar_guess=-1e-3)

    # -----------------------------------------------------------------------
    # 6. Remove failed events, build y with 3 cols [t, v_arr, a_opt]
    # -----------------------------------------------------------------------
    X_tot_4_temp, y_4_arr, ar_clean = remove_failed_events(
        X_tot_4, y_4, ar_results, drop_index
    )
    # X_tot_4_temp: (N, 5) — no flag col (used for augmentation)
    # y_4_arr:      (N, 3) — [transit_time_s, arrival_speed, a_opt]

    # drop the same rows from the stratification series
    split_4_series = split_4_series.drop(drop_index).reset_index(drop=True)

    # -----------------------------------------------------------------------
    #  5. Build DataFrames; add 'sim?' flag column (0 = real)
    # -----------------------------------------------------------------------
    X_tot_4_extended = pd.DataFrame(X_tot_4_temp)
    X_tot_4_extended['sim?'] = 0                     # real events
    X_tot_4_extended = np.array(X_tot_4_extended)
    X_tot_4_extended = pd.DataFrame(X_tot_4_extended)   # back to DataFrame

    y_4_df = pd.DataFrame(y_4_arr)                   # (N, 3)

    X_tot_4_temp_df = pd.DataFrame(X_tot_4_temp)    # (N, 5), no flag

    split_4_df = pd.Series(split_4_series.values.ravel())

    print("\nX_tot_4_extended shape:", X_tot_4_extended.shape)
    print("y_4 shape:", y_4_df.shape)

    # -----------------------------------------------------------------------
    #  6. Multi-realization training
    # -----------------------------------------------------------------------
    print("\nStarting training loop...")
    results = run_realizations(
        X_tot_4_extended  = X_tot_4_extended,
        y_4               = y_4_df,
        X_tot_4_temp      = X_tot_4_temp_df,
        split_4_df        = split_4_df,
        uniform_split_val = uniform_split_val,
        opt_and_clean_fn  = opt_and_clean,
        initial_seed      = 57,
        num_realizations  = 25,
        N_epochs          = 1000,
        max_retries       = 3,
        min_epochs        = 131,
    )

    # -----------------------------------------------------------------------
    # 7. Print metrics
    # -----------------------------------------------------------------------
    print('\nMAE REAL')
    print(np.mean(results['mae_train_real']))
    print(np.mean(results['mae_val_real']))
    print(np.mean(results['mae_test_tot']))

    print('\nMedAE REAL')
    print(np.mean(results['medae_train_real']))
    print(np.mean(results['medae_val_real']))
    print(np.mean(results['medae_test_tot']))

    print('\nMAE AUG')
    print(np.mean(results['mae_train_aug']))
    print(np.mean(results['mae_val_aug']))

    # -----------------------------------------------------------------------
    # 8. Save results to CSV
    # -----------------------------------------------------------------------

    save_results(results, OUTPUT_CSV)

    return results






if __name__ == '__main__':
    main()