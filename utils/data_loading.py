"""
data_loading.py
---------------
Data loading and preprocessing for CME transit-time prediction.

"""

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Physical constants and conversion factors
RSUN          = 6.957e5          # km
R0            = 20.0 * RSUN      # km
AU            = 1.495978707e8    # km  (1 AU in km)
CONVERSION_AU = 1.0 / AU         # km -> AU  (used in preprocessing / feature building)
HYDROGEN_MASS = 1.6735575e-24    # g


def load_and_clean(icme_path: str) -> pd.DataFrame:
    """
    Load CSV, cast columns, drop bad rows, drop header duplicate.
    """
    data = pd.read_csv(icme_path)
    df = pd.DataFrame(data, columns=[
        'Start_Date', 'Arrival_Date', 'Transit_time',
        'v_r', 'Mass', 'rel_wid', 'Wind dens', 'Wind speed', 'Arrival_v'
    ])

    df['Wind dens'][1:]    = df['Wind dens'][1:].astype(np.double)
    df['Wind speed'][1:]   = df['Wind speed'][1:].astype(np.double)
    df['v_r'][1:]          = df['v_r'][1:].astype(np.double)
    df['Transit_time'][1:] = df['Transit_time'][1:].astype(np.double)
    df['Mass'][1:]         = df['Mass'][1:].astype(np.double)
    df['rel_wid'][1:]      = df['rel_wid'][1:].astype(np.double)
    df['Arrival_v'][1:]    = df['Arrival_v'][1:].astype(np.double)

    print('# events with wind speed = 0 ', len(np.where(df['Wind speed'] == 0.)[0]))
    print('# events with mass = -9999 ',   len(np.where(df['Mass'] == -9999.)[0]))

    df = df.drop(index=df.index[np.where(df['Wind speed'] == 0.)[0]])
    df = df.drop(index=df.index[np.where(df['Mass'] == -9999.0)[0]])
    df = df[1:]   
    print('Size df: ', len(df))
    return df


def compute_features(df: pd.DataFrame):
    """
    X columns: [v_0 (km/s), m (g), A (km^2), rho (g/km^3), w (km/s)]
    y columns: [transit_time (s), arrival_speed (km/s)]
    """
    X = df[['v_r', 'Mass', 'rel_wid', 'Wind dens', 'Wind speed']].values.astype(float)
    y = df[['Transit_time', 'Arrival_v']].values

    # rho: cm^-3 -> g * km^-3
    X[:, 3] = X[:, 3] * 1e15 * HYDROGEN_MASS

    # transit time: hours -> seconds
    y[:, 0] = y[:, 0] * 3600.0

    # angular half-width -> spherical cap area (rel_wid already in radians)
    angle_rad  = df['rel_wid'].values.astype(float)
    cos_angles = np.array([math.cos(a) for a in angle_rad])
    A = 2 * np.pi * R0**2 * (1.0 - cos_angles)
    X[:, 2] = A

    v_0 = X[:, 0]
    m   = X[:, 1]
    rho = X[:, 3]
    w   = X[:, 4]

    v = df['Arrival_v'].values.ravel()
    t = df['Transit_time'].values.ravel() * 3600.0   # seconds

    return X, y, v_0, m, A, rho, w, v, t


def add_wind_speed_type(df: pd.DataFrame) -> pd.Series:
    """
    w_speed_type = 1 if Wind speed > 500 km/s, else 0.
    Returns a pandas Series with the same index as df.
    """
    df = df.copy()
    df['w_speed_type'] = df['Wind speed'].apply(lambda x: 1 if x > 500 else 0)
    return df['w_speed_type']

def classify_cases(v_0, v, w):
    """
    Boolean masks for each propagation case.
    """
    cases = {
        'Sub-w':  (v_0 <= w) & (v <= w),
        'Sub-w (↗)': (v_0 <= w) & (v <= w) & (v >  v_0),
        'Sub-w (↘)': (v_0 <= w) & (v <= w) & (v <  v_0),
        'Super-w':  (v_0 >= w) & (v >= w),
        'Super-w (↗)': (v_0 >= w) & (v >= w) & (v >  v_0),
        'Super-w (↘)': (v_0 >= w) & (v >= w) & (v <  v_0),
        'Cross-w (↗)':  (v_0 <  w) & (v >  w),
        'Cross-w (↘)':  (v  <  w) & (v_0 >  w),
    }
    for k, mask in cases.items():
        print(f"Case {k}: {mask.sum()} events")
    return cases


def uniform_split_val(X, y, stratify_column=None, test_size=0.2, val_size=0.1, random_state=0):
    """
    Stratified train / val / test split.

    X must be a DataFrame (indices are used for the second stratify call).
    stratify_column must be a pd.Series aligned with X.
    Returns X_train, X_val, X_test, y_train, y_val, y_test,
            train_idx, val_idx, test_idx.
    """
    if isinstance(stratify_column, pd.Series):
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

    count_ones_train  = np.sum(X_train['__temp_stratify_col__'] == 1)
    count_zeros_train = len(X_train) - count_ones_train
    count_ones_val    = np.sum(X_val['__temp_stratify_col__'] == 1)
    count_zeros_val   = len(X_val) - count_ones_val
    count_ones_test   = np.sum(X_test['__temp_stratify_col__'] == 1)
    count_zeros_test  = len(X_test) - count_ones_test

    print(f"Train set: {len(X_train)} righe - 1s: {count_ones_train}, 0s: {count_zeros_train}")
    print(f"Validation set: {len(X_val)} righe - 1s: {count_ones_val}, 0s: {count_zeros_val}")
    print(f"Test set: {len(X_test)} righe - 1s: {count_ones_test}, 0s: {count_zeros_test}")

    X_train = X_train.drop(columns=['__temp_stratify_col__'])
    X_val   = X_val.drop(columns=['__temp_stratify_col__'])
    X_test  = X_test.drop(columns=['__temp_stratify_col__'])

    return X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx