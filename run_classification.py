"""
run_classification.py
---------------------
Main script for CME propagation-case classification.

Usage
-----
    python run_classification.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from utils.data_loading import (
    load_and_clean,
    compute_features,
    classify_cases,
)
from models.classification import (
    optimize_and_evaluate_multiclass_logistic_regression
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ICME_PATH = 'Data/ICME_complete_dataset_rev.csv'
SEED      = 78
N_SPLITS  = 15


def main():
    # -----------------------------------------------------------------------
    # 1. Data loading and preprocessing
    # -----------------------------------------------------------------------
    print("Loading data...")
    df     = load_and_clean(ICME_PATH)
    X, y, v_0, m, A, rho, w, v, t = compute_features(df)

    # -----------------------------------------------------------------------
    # 2. Case classification
    # -----------------------------------------------------------------------
    print("\nClassifying CME propagation cases...")
    cases = classify_cases(v_0, v, w)

    case_1A = cases['Sub-w (↗)']
    case_1B = cases['Sub-w (↘)']
    case_2A = cases['Super-w (↗)']
    case_2B = cases['Super-w (↘)']
    case_3  = cases['Cross-w (↗)']
    case_4  = cases['Cross-w (↘)']

    # -----------------------------------------------------------------------
    # 3. Build target dataframe with integer case labels
    #
    #    y_tot columns:
    #      0 : transit_time  [s]
    #      1 : arrival_speed [km/s]
    #      2 : CASE   (multiclass: 0=1A, 1=1B, 2=2A, 3=2B, 4=3, 5=4)
    # -----------------------------------------------------------------------
    y_tot = pd.DataFrame(y, columns=['transit_time', 'arrival_speed'])

    case_col = np.full(len(y_tot), '', dtype=object)
    case_col[case_1A] = 0
    case_col[case_1B] = 1
    case_col[case_2A] = 2
    case_col[case_2B] = 3
    case_col[case_3]  = 4
    case_col[case_4]  = 5
    y_tot['CASE'] = case_col

    y_tot['CASE 4'] = y_tot['CASE'].apply(
        lambda x: 1 if x == 5 else (0 if x in [0, 3] else 2)
    )

    print(y_tot)
    print(y_tot.shape)

    X_tot = X.copy()

    # -----------------------------------------------------------------------
    # 4. Multi-class logistic regression
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("Multi-class classification (6 propagation cases)")
    print("="*60)

    results_multi = optimize_and_evaluate_multiclass_logistic_regression(
        X_tot, y_tot, n_splits=N_SPLITS, seed=SEED
    )

    return results_multi


if __name__ == '__main__':
    results_multi = main()

    all_best_models_multi    = results_multi['all_best_models']
    all_best_params_multi    = results_multi['all_best_params']
    class_metrics_test_multi = results_multi['class_metrics_test']