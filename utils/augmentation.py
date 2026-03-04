"""
augmentation.py
---------------
Data augmentation and *a* optimization.

"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from sklearn.model_selection import train_test_split

RSUN = 6.957e5 # km
CONVERSION_AU = 6.68459e-9 # km -> AU


def augment_data(X, y, num_samples_per_real, min_increase, max_increase, random_seed):
    """
    y has 3 cols [t, v_arr, a_opt] when called from the training loop.
    Case-4 constraint check: v=new_label[1], v_0=new_sample[0], w=new_sample[4].
    """
    np.random.seed(random_seed)
    X = np.array(X)
    y = np.array(y)
    augmented_data, augmented_labels = [], []

    for i in range(len(X)):
        base_sample = X[i]
        base_label = y[i]
        for _ in range(num_samples_per_real):
            fx = 1 + np.random.uniform(min_increase, max_increase, size=base_sample.shape) * np.random.choice([-1, 1], size=base_sample.shape)
            fy = 1 + np.random.uniform(min_increase, max_increase, size=base_label.shape) * np.random.choice([-1, 1], size=base_label.shape)
            new_sample = base_sample * fx
            new_label = base_label * fy
            v = new_label[1]
            v_0 = new_sample[0]
            w = new_sample[4]
            if v < w and v_0 > w:
                augmented_data.append(new_sample)
                augmented_labels.append(new_label)

    return np.array(augmented_data), np.array(augmented_labels)


def _eq_4(a, v_0, w, gamma, t, r_0):
    if a >= 0:
        return np.inf
    r = CONVERSION_AU * (
        (w - np.sqrt(-a / gamma)) * t + r_0
        - (1.0 / gamma) * (
            np.arctan(np.sqrt(-gamma / a) * (w - v_0))
            + np.log(
                0.5 * np.sqrt(a / (a - gamma * (v_0 - w) ** 2))
                * (np.exp(-2.0 * (np.sqrt(-a * gamma) * t - np.arctan(np.sqrt(-gamma / a) * (v_0 - w)))) + 1.0)
            )
        )
    )
    return r - 1.0


def opt_and_clean(X, y):
    """
    Calculate new *a* for augmented data, clean and reformat.

    y enters with 3 cols [t, v_arr, a_old].
    y = y[:, :-1] strips old a -> (N,2), then appends new a -> (N,3).
    X gets info column of 2s appended -> (N,6).
    """
    X = np.array(X)
    y = np.array(y)
    C_fixed = 100
    r_0 = 20.0 * RSUN
    n = X.shape[0]
    ar_results = np.zeros(n)

    for i in range(n):
        v_0_i, m_i, A_i, rho_i, w_i = X[i, 0], X[i, 1], X[i, 2], X[i, 3], X[i, 4]
        t_i = y[i, 0]
        gamma_i = C_fixed * A_i * rho_i / m_i

        def eq(a, v0=v_0_i, wi=w_i, gi=gamma_i, ti=t_i, r=r_0):
            return _eq_4(a, v0, wi, gi, ti, r)

        a_sol, _, ier, mesg = fsolve(eq, -1e-3, full_output=True, xtol=1e-6)
        a_sol = a_sol[0]
        ar_results[i] = a_sol

        t_cross = (1.0 / np.sqrt(-ar_results[i] * gamma_i)) * np.arctan(np.sqrt(-gamma_i / ar_results[i]) * (w_i - v_0_i))
        if np.abs(eq(ar_results[i])) > 1e-6 or t_i < t_cross:
            ar_results[i] = 0.0

    drop_index = np.where(ar_results == 0.0)[0]
    X = np.delete(X, drop_index, axis=0)
    y = np.delete(y, drop_index, axis=0)
    ar_results = np.delete(ar_results, drop_index)

    y = y[:, :-1]                                   # strip old a col: (N,3)->(N,2)
    y = np.hstack([y, ar_results.reshape(-1, 1)])   # append new a: (N,2)->(N,3)

    X_info = np.zeros((X.shape[0], 1)) + 2
    X = np.hstack([X, X_info])                      # (N,5)->(N,6)

    return X, y


def uniform_split_val(X, y, stratify_column, test_size=0.2, val_size=0.1, random_state=0):
    """
    Stratify_column: pd.Series aligned with X (DataFrame with original index).
    """
    X = X.copy()
    X['__temp_stratify_col__'] = stratify_column
    stratify_target = X['__temp_stratify_col__']
    indices = np.arange(X.shape[0])

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, indices,
        test_size=test_size,
        stratify=stratify_target,
        random_state=random_state
    )

    X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
        X_train, y_train, train_idx,
        test_size=val_size / (1 - test_size),
        stratify=stratify_target[X_train.index],
        random_state=random_state
    )

    for name, Xs in [('Train', X_train), ('Validation', X_val), ('Test', X_test)]:
        ones = (Xs['__temp_stratify_col__'] == 1).sum()
        zeros = len(Xs) - ones
        print(f"{name} set: {len(Xs)} rows - 1s: {ones}, 0s: {zeros}")

    X_train = X_train.drop(columns=['__temp_stratify_col__'])
    X_val   = X_val.drop(columns=['__temp_stratify_col__'])
    X_test  = X_test.drop(columns=['__temp_stratify_col__'])

    return X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx