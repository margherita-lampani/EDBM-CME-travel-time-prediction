"""
run_optimization.py
-------------------
Optimize the drag-based model acceleration parameter *a* for Case Cross-w (↘) events.

Usage
-----
    python run_optimization.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loading import load_and_clean, compute_features, classify_cases
from utils.optimization import optimize_case4, remove_failed_events

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ICME_PATH = 'Data/ICME_complete_dataset_rev.csv'


def main():
    # -----------------------------------------------------------------------
    # 1. Load and preprocess data
    # -----------------------------------------------------------------------
    print("Loading data...")
    df = load_and_clean(ICME_PATH)
    X, y, v_0, m, A, rho, w, v, t = compute_features(df)

    # -----------------------------------------------------------------------
    # 2. Classify events into propagation cases
    # -----------------------------------------------------------------------
    print("\nClassifying CME propagation cases...")
    cases = classify_cases(v_0, v, w)

    case4_mask = cases['Cross-w (↘)']
    X_tot_4    = X[case4_mask]
    y_4        = y[case4_mask]

    # -----------------------------------------------------------------------
    # 3. Optimize acceleration parameter *a* for Case 4
    # -----------------------------------------------------------------------
    print(f"\nOptimizing Case Cross-w (↘) ({case4_mask.sum()} events)...")
    ar_results, drop_index = optimize_case4(X_tot_4, y_4, C=100, ar_guess=-1e-3)

    X_clean, y_clean, ar_clean = remove_failed_events(
        X_tot_4, y_4, ar_results, drop_index
    )

    # -----------------------------------------------------------------------
    # 4. Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 45)
    print(f"Case Cross-w (↘): {case4_mask.sum()} total  |  {len(X_clean)} valid  |  {len(drop_index)} dropped")
    print("=" * 45)
    print(f"a values — min: {ar_clean.min():.4e}  max: {ar_clean.max():.4e}  mean: {ar_clean.mean():.4e}")
    print("\nOptimization complete.")

    return X_clean, y_clean, ar_clean


if __name__ == '__main__':
    main()