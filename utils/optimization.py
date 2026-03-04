"""
optimization.py
---------------
Optimization of the EDBM acceleration parameter *a* for Case Cross-w (↘) events.
"""

import numpy as np
from scipy.optimize import fsolve

RSUN          = 6.957e5   # km    
CONVERSION_AU = 6.68459e-9 # km -> AU    


def _eq_4(a, v_0, w, gamma, t, r_0):
    """
    Case Cross-w (↘) position equation 
    """
    if a >= 0:
        return np.inf
    eqr = CONVERSION_AU * (
        (w - np.sqrt(-a / gamma)) * t + r_0
        - (1.0 / gamma) * (
            np.arctan(np.sqrt(-gamma / a) * (w - v_0))
            + np.log(
                0.5 * np.sqrt(a / (a - gamma * (v_0 - w) ** 2))
                * (np.exp(-2.0 * (np.sqrt(-a * gamma) * t
                                  - np.arctan(np.sqrt(-gamma / a) * (v_0 - w)))) + 1.0)
            )
        )
    )
    return eqr - 1.0


def optimize_case4(X_case, y_case, C=100, ar_guess=-1e-3):
    """
    Optimize *a* for all Case Cross-w (↘) events.

    Parameters
    ----------
    X_case : array (N, 5)   [v_0, m, A, rho, w]
    y_case : array (N, 2)   [transit_time_s, arrival_speed_km_s]

    Returns
    -------
    ar_results : array (N,)   optimized a values; 0.0 where failed
    drop_index : array        indices of failed events
    """
    r_0 = 20.0 * RSUN
    n   = X_case.shape[0]
    ar_results = np.zeros(n)

    for i in range(n):
        v_0_i = X_case[i, 0]
        m_i   = X_case[i, 1]
        A_i   = X_case[i, 2]
        rho_i = X_case[i, 3]
        w_i   = X_case[i, 4]
        t_i   = y_case[i, 0]

        gamma_i = C * A_i * rho_i / m_i

        def eq(a, v0=v_0_i, wi=w_i, gi=gamma_i, ti=t_i, r=r_0):
            return _eq_4(a, v0, wi, gi, ti, r)

        ar_result, info, ier, mesg = fsolve(eq, ar_guess, full_output=True, xtol=1e-6)
        ar_results[i] = ar_result[0]

        if ier != 1:
            print(f"Warning: solution not found for i={i}, {mesg}")

        ar_results[i] = ar_result[0]

        t_cross_i = (1.0 / np.sqrt(-ar_results[i] * gamma_i)) * np.arctan(
            np.sqrt(-gamma_i / ar_results[i]) * (w_i - v_0_i))
        if np.abs(eq(ar_results[i])) > 1e-6 or t_i < t_cross_i:
            ar_results[i] = 0.0

        res = eq(ar_results[i]) + 1
        print(f"eq({ar_results[i]}) = {res}")

    print(ar_results)
    drop_index = np.where(ar_results == 0.0)[0]
    print(drop_index)
    print(len(drop_index))
    return ar_results, drop_index


def remove_failed_events(X, y, ar_results, drop_index):
    """
    Remove failed events and append a_opt as 3rd y column.

    y enters: (N, 2)  [transit_time, arrival_speed]
    y exits:  (N, 3)  [transit_time, arrival_speed, a_opt]
    """
    X_c  = np.delete(X,          drop_index, axis=0)
    y_c  = np.delete(y,          drop_index, axis=0)   # (N, 2)
    ar_c = np.delete(ar_results, drop_index)
    y_c  = np.hstack([y_c, ar_c.reshape(-1, 1)])        # (N, 3)
    return X_c, y_c, ar_c


def opt_and_clean(X, y):
    """
    Re-optimize 'a' for augmented data, clean and reformat.

    y enters: (N, 3)  [transit_time, arrival_speed, a_old]  (3 cols from augmentation)
    y exits:  (N, 3)  [transit_time, arrival_speed, a_new]

    Steps:
        1. Solve for new a using _eq_4.
        2. Drop events where |eq(a)| > 1e-6 or t < t_cross.
    """
    X = np.array(X)
    y = np.array(y)

    r_0     = 20.0 * RSUN
    C_fixed = 100
    n       = X.shape[0]
    ar_results = np.zeros(n)

    for i in range(n):
        v_0_i = X[i, 0]
        m_i   = X[i, 1]
        A_i   = X[i, 2]
        rho_i = X[i, 3]
        w_i   = X[i, 4]

        t_i = y[i, 0]
        v_i = y[i, 1]

        gamma_i = C_fixed * A_i * rho_i / m_i

        def eq(a, v0=v_0_i, wi=w_i, gi=gamma_i, ti=t_i, r=r_0):
            if a >= 0:
                return np.inf
            eqr = CONVERSION_AU * (
                (wi - np.sqrt(-a / gi)) * ti + r
                - (1.0 / gi) * (
                    np.arctan(np.sqrt(-gi / a) * (wi - v0))
                    + np.log(
                        0.5 * np.sqrt(a / (a - gi * (v0 - wi) ** 2))
                        * (np.exp(-2.0 * (np.sqrt(-a * gi) * ti
                                          - np.arctan(np.sqrt(-gi / a) * (v0 - wi)))) + 1.0)
                    )
                )
            )
            return eqr - 1.0

        ar_result, info, ier, mesg = fsolve(eq, -1e-3, full_output=True, xtol=1e-6)
        ar_results[i] = ar_result[0]

        if ier != 1:
            ar_results[i] = ar_result[0]

        t_cross_i = (1.0 / np.sqrt(-ar_results[i] * gamma_i)) * np.arctan(
            np.sqrt(-gamma_i / ar_results[i]) * (w_i - v_0_i))
        if np.abs(eq(ar_results[i])) > 1e-6 or t_i < t_cross_i:
            ar_results[i] = 0.0

    drop_index = np.where(ar_results == 0.0)[0]

    X          = np.delete(X,          drop_index, axis=0)
    y          = np.delete(y,          drop_index, axis=0)
    ar_results = np.delete(ar_results, drop_index)

    y = y[:, :-1]                                   # strip old a col: (N,3) -> (N,2)
    y = np.hstack([y, ar_results.reshape(-1, 1)])   # append new a:    (N,2) -> (N,3)

    X_info = np.zeros((X.shape[0], 1)) + 2          # flag = 2 (augmented)
    X      = np.hstack([X, X_info])                 # (N,5) -> (N,6)

    return X, y