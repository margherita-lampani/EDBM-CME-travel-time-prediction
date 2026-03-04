"""
classification.py
-----------------
Multi-class classification of CME propagation regimes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    confusion_matrix, classification_report, make_scorer
)
from sklearn.utils import shuffle

# ---------------------------------------------------------------------------
# Stratified splitting
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def uniform_split_val_multi(X, y, test_size=0.2, val_size=0.2, random_state=0):
    """
    Stratified train / validation / test split.

    Stratification is performed on the multi-class case label stored in
    column 2 of ``y`` (0=1A, 1=1B, 2=2A, 3=2B, 4=3, 5=4).

    Parameters
    ----------
    X : array-like, shape (N, 5)
        Feature matrix.
    y : array-like, shape (N, ≥4)
        Target dataframe; column 2 = multiclass label, column 3 = binary label.
    test_size : float
        Fraction reserved for the test set.
    val_size : float
        Fraction reserved for the validation set.
    random_state : int
        Random seed.

    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame
    y_train, y_val, y_test : pd.DataFrame
    train_idx, val_idx, test_idx : np.ndarray
        Integer indices into the original arrays.
    """
    X = pd.DataFrame(X).copy()
    y = pd.DataFrame(y).copy()

    # Stratify on the multiclass case column (column 2)
    stratify_target = y.iloc[:, 2]
    indices = np.arange(y.shape[0])

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, indices,
        test_size=test_size,
        stratify=stratify_target,
        random_state=random_state,
    )

    # Second split: train → train + val (stratify on multiclass label)
    stratify_target_train = y_train.iloc[:, 2]
    X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
        X_train, y_train, train_idx,
        test_size=val_size / (1.0 - test_size),
        stratify=stratify_target_train,
        random_state=random_state,
    )

    # Print binary label distribution for monitoring
    print("\nTrain - Binary:")
    print(y_train.iloc[:, 3].value_counts(normalize=True))
    print(y_train.iloc[:, 3].value_counts())

    print("\nValidation - Binary:")
    print(y_val.iloc[:, 3].value_counts(normalize=True))
    print(y_val.iloc[:, 3].value_counts())

    print("\nTest - Binary:")
    print(y_test.iloc[:, 3].value_counts(normalize=True))
    print(y_test.iloc[:, 3].value_counts())

    return X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx

# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def augment_data_multi(X, y, num_samples_per_real=None, min_increase=None,
                 max_increase=None, random_seed=None):
    """
    Augment the dataset by generating perturbed copies of each real event.

    Parameters
    ----------
    X : array-like, shape (N, 5)
        Feature matrix [v_0, m, A, rho, w].
    y : array-like, shape (N, ≥3)
        Target matrix; column 2 = integer case label (0–5).
    num_samples_per_real : int
        Number of synthetic samples generated per real event.
    min_increase : float
        Minimum relative perturbation (e.g. 0.05 → ±5%).
    max_increase : float
        Maximum relative perturbation (e.g. 0.10 → ±10%).
    random_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    X_aug : np.ndarray
        Combined real + augmented features.
    y_aug : np.ndarray
        Combined real + augmented targets (with flag column appended).
    """
    np.random.seed(random_seed)
    X = np.array(X)
    y = np.array(y)

    # Append a flag column: 0 = real, 1 = augmented
    y = np.hstack((y, np.zeros((y.shape[0], 1))))

    augmented_data   = []
    augmented_labels = []

    for i in range(len(X)):
        base_sample = X[i]
        base_label  = y[i]
        case        = int(base_label[2])

        # Keep the real event
        augmented_data.append(base_sample)
        augmented_labels.append(base_label)

        for _ in range(num_samples_per_real):
            new_sample = base_sample.copy()
            new_label  = base_label.copy()

            # Perturb all features independently
            factors_X = (1 + np.random.uniform(min_increase, max_increase, size=new_sample.shape)
                         * np.random.choice([-1, 1], size=new_sample.shape))
            new_sample = new_sample * factors_X

            # Perturb only transit_time (col 0) and arrival_speed (col 1)
            for col in [0, 1]:
                factor = 1 + np.random.uniform(min_increase, max_increase) * np.random.choice([-1, 1])
                new_label[col] = new_label[col] * factor

            new_label[-1] = 1  # Mark as augmented

            v_0 = new_sample[0]
            v   = new_label[1]
            w   = new_sample[4]

            # Validate case constraint
            valid = False
            if   case == 0 and (v_0 <= w and v <= w and v > v_0):   valid = True  # 1A
            elif case == 1 and (v_0 <= w and v <= w and v < v_0):   valid = True  # 1B
            elif case == 2 and (v_0 >= w and v >= w and v > v_0):   valid = True  # 2A
            elif case == 3 and (v_0 >= w and v >= w and v < v_0):   valid = True  # 2B
            elif case == 4 and (v_0 < w  and v > w):                 valid = True  # 3
            elif case == 5 and (v_0 > w  and v < w):                 valid = True  # 4

            if valid:
                augmented_data.append(new_sample)
                augmented_labels.append(new_label)

    return np.array(augmented_data), np.array(augmented_labels)

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def standardize_data(X_train, X_val, X_test):
    """
    Fit a StandardScaler on the training set and apply it to all splits.

    Returns
    -------
    X_train_std, X_val_std, X_test_std : np.ndarray
    scaler : sklearn.preprocessing.StandardScaler
    """
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std   = scaler.transform(X_val)
    X_test_std  = scaler.transform(X_test)
    return X_train_std, X_val_std, X_test_std, scaler


def calculate_class_metrics(y_true, y_pred, class_map=None):
    """
    Compute per-class and macro-averaged classification metrics.

    For each class the one-vs-rest confusion matrix is computed, yielding:
    TP, FP, TN, FN, sensitivity (recall), specificity, precision, F1, TSS,
    accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    class_map : dict, optional
        Mapping integer → class name string. Defaults to the 6-class map.

    Returns
    -------
    class_metrics : dict
        {class_name: {metric_name: value}}.
    avg_metrics : dict
        Macro-averaged TSS, precision, recall, specificity, F1.
    """
    if class_map is None:
        class_map = {0: "1A", 1: "1B", 2: "2A", 3: "2B", 4: "3", 5: "4"}

    classes = np.unique(np.concatenate([y_true, y_pred]))

    class_metrics = {}
    tss_scores, precision_scores, recall_scores, specificity_scores, f1_scores = [], [], [], [], []

    for cls in classes:
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1          = (2 * precision * sensitivity / (precision + sensitivity)
                       if (precision + sensitivity) > 0 else 0)
        tss         = sensitivity + specificity - 1
        accuracy    = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        cls_name = class_map.get(cls, str(cls))
        class_metrics[cls_name] = {
            'true_positive':  tp,
            'false_positive': fp,
            'true_negative':  tn,
            'false_negative': fn,
            'sensitivity':    sensitivity,
            'specificity':    specificity,
            'precision':      precision,
            'f1_score':       f1,
            'tss':            tss,
            'accuracy':       accuracy,
        }

        tss_scores.append(tss)
        precision_scores.append(precision)
        recall_scores.append(sensitivity)
        specificity_scores.append(specificity)
        f1_scores.append(f1)

    avg_metrics = {
        'mean_tss':         np.mean(tss_scores),
        'mean_precision':   np.mean(precision_scores),
        'mean_recall':      np.mean(recall_scores),
        'mean_specificity': np.mean(specificity_scores),
        'mean_f1':          np.mean(f1_scores),
    }
    return class_metrics, avg_metrics


def _mean_tss_scorer(y_true, y_pred):
    """Sklearn-compatible scorer: returns macro-averaged TSS."""
    _, avg = calculate_class_metrics(y_true, y_pred)
    return avg['mean_tss']


def evaluate_multiclass_model(model, X, y_true, class_map=None, print_predictions=False):
    """
    Predict and evaluate a fitted sklearn classifier.

    Parameters
    ----------
    model : sklearn estimator
        Fitted classifier with a ``predict`` method.
    X : np.ndarray
        Standardised feature matrix.
    y_true : np.ndarray
        Ground-truth integer labels.
    class_map : dict, optional
        Label → name mapping for metrics.
    print_predictions : bool
        If True, print each (predicted, actual) pair.

    Returns
    -------
    dict
        Keys: ``'tss'``, ``'accuracy'``, ``'macro_f1'``, ``'confusion_matrix'``,
        ``'y_pred'``, ``'class_metrics'``, ``'avg_metrics'``.
    """
    y_pred = model.predict(X)

    if print_predictions:
        print("\nPredicted and Actual Values:")
        for pred, actual in zip(y_pred, y_true):
            print(f"  Predicted: {pred}, Actual: {actual}")

    class_metrics, avg_metrics = calculate_class_metrics(y_true, y_pred, class_map)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm     = confusion_matrix(y_true, y_pred)

    return {
        'tss':              avg_metrics['mean_tss'],
        'accuracy':         report['accuracy'],
        'macro_f1':         report['macro avg']['f1-score'],
        'confusion_matrix': cm,
        'y_pred':           y_pred,
        'class_metrics':    class_metrics,
        'avg_metrics':      avg_metrics,
    }


# ---------------------------------------------------------------------------
# Multi-class logistic regression (6 cases)
# ---------------------------------------------------------------------------

def optimize_and_evaluate_multiclass_logistic_regression(
        X_tot, y_tot, n_splits=15, seed=78):
    """
    Run multinomial logistic regression over multiple stratified splits.

    Hyperparameters are fixed after prior grid search:
        C=50, solver='saga', multi_class='multinomial',
        class_weight='balanced', tol=1e-5, max_iter=2500.

    Parameters
    ----------
    X_tot : np.ndarray or pd.DataFrame, shape (N, 5)
        Full feature matrix.
    y_tot : pd.DataFrame
        Target dataframe with columns [transit_time, arrival_speed, case_mc, case_bin].
    n_splits : int
        Number of independent train/val/test splits.
    seed : int
        Base random seed (split i uses seed + i).

    Returns
    -------
    dict
        Full results: all split results, aggregate metrics, confusion matrices,
        per-class metrics per split.
    """
    print("Performing multiple train-test-val splits with data augmentation...")

    # Hyperparameter grid (single configuration = no search overhead)
    param_grid = {
        'C':            [50],
        'solver':       ['saga'],
        'multi_class':  ['multinomial'],
        'class_weight': ['balanced'],
        'tol':          [1e-5],
        'max_iter':     [2500],
    }

    tss_scorer = make_scorer(_mean_tss_scorer)

    # 6-class label map
    class_map_6 = {
        0: "Sub-w (↗)", 1: "Sub-w (↘)",
        2: "Super-w (↗)", 3: "Super-w (↘)",
        4: "Cross-w (↗)", 5: "Cross-w (↘)",
    }

    all_train_results, all_val_results, all_test_results = [], [], []
    all_best_params, all_best_models, all_cms_test = [], [], []
    all_y_test_multi, all_y_test_pred = [], []

    all_class_metrics_train = {}
    all_class_metrics_val   = {}
    all_class_metrics_test  = {}

    X_tot_arr = np.array(X_tot)

    for split_idx in range(n_splits):
        print(f"\n\n{'='*50}")
        print(f"Processing Split {split_idx+1}/{n_splits}")
        print(f"{'='*50}")

        random_seed = seed + split_idx

        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         train_idx, val_idx, test_idx) = uniform_split_val_multi(
            X_tot_arr, y_tot, random_state=random_seed)

        # Append a zero-flag column to real train/val targets before augmentation
        y_train = np.hstack((y_train, np.zeros((y_train.shape[0], 1))))
        y_val   = np.hstack((y_val,   np.zeros((y_val.shape[0],   1))))

        # Augment training and validation sets
        X_aug_train, y_aug_train = augment_data_multi(
            X_tot_arr[train_idx], y_tot.loc[train_idx],
            num_samples_per_real=100, min_increase=0.05, max_increase=0.10,
            random_seed=random_seed)

        X_aug_val, y_aug_val = augment_data_multi(
            X_tot_arr[val_idx], y_tot.loc[val_idx],
            num_samples_per_real=100, min_increase=0.05, max_increase=0.10,
            random_seed=random_seed)

        X_train = np.vstack([X_train, X_aug_train])
        y_train = np.vstack([y_train, y_aug_train])
        X_val   = np.vstack([X_val,   X_aug_val])
        y_val   = np.vstack([y_val,   y_aug_val])
        y_test  = np.array(y_test)

        X_train, y_train = shuffle(X_train, y_train, random_state=random_seed)
        X_val,   y_val   = shuffle(X_val,   y_val,   random_state=random_seed)
        X_test,  y_test  = shuffle(X_test,  y_test,  random_state=random_seed)

        # Keep only columns 2 (multiclass) and 3 (binary)
        y_train = y_train[:, [2, 3]]
        y_val   = y_val[:,   [2, 3]]
        y_test  = y_test[:,  [2, 3]]

        print("Train:", X_train.shape, y_train.shape)
        print("Val:  ", X_val.shape,   y_val.shape)
        print("Test: ", X_test.shape,  y_test.shape)

        y_train_multi = y_train[:, 0].astype(int)
        y_val_multi   = y_val[:,   0].astype(int)
        y_test_multi  = y_test[:,  0].astype(int)

        X_train_std, X_val_std, X_test_std, _ = standardize_data(X_train, X_val, X_test)

        # GridSearchCV on training set
        log_reg = LogisticRegression(random_state=random_seed)
        skf     = StratifiedKFold(n_splits=5)
        grid_search = GridSearchCV(
            log_reg, param_grid, cv=skf, scoring=tss_scorer, n_jobs=-1, verbose=1)
        grid_search.fit(X_train_std, y_train_multi)

        best_params = grid_search.best_params_
        best_model  = grid_search.best_estimator_
        print(f"Best parameters: {best_params}")
        print(f"Best TSS during CV: {grid_search.best_score_:.4f}")

        # --- TRAIN ---
        print("\nEvaluation on training set...")
        train_res = evaluate_multiclass_model(
            best_model, X_train_std, y_train_multi, class_map_6)
        _print_split_metrics("Training", train_res)
        _accumulate_class_metrics(all_class_metrics_train, train_res['class_metrics'])

        unique_cls = np.unique(np.concatenate([y_train_multi, train_res['y_pred']]))
        cls_labels = [class_map_6.get(c, str(c)) for c in unique_cls]
        _plot_confusion_matrix(
            train_res['confusion_matrix'], cls_labels,
            f"Confusion Matrix - Split {split_idx+1} - Training", cmap='Blues')

        # --- VAL ---
        print("\nEvaluation on validation set...")
        val_res = evaluate_multiclass_model(
            best_model, X_val_std, y_val_multi, class_map_6)
        _print_split_metrics("Validation", val_res)
        _accumulate_class_metrics(all_class_metrics_val, val_res['class_metrics'])

        unique_cls = np.unique(np.concatenate([y_val_multi, val_res['y_pred']]))
        cls_labels = [class_map_6.get(c, str(c)) for c in unique_cls]
        _plot_confusion_matrix(
            val_res['confusion_matrix'], cls_labels,
            f"Confusion Matrix - Split {split_idx+1} - Validation", cmap='Greens')

        # --- TEST ---
        print("\nEvaluation on test set...")
        test_res = evaluate_multiclass_model(
            best_model, X_test_std, y_test_multi, class_map_6,
            print_predictions=(split_idx == 0))
        _print_split_metrics("Test", test_res)
        _accumulate_class_metrics(all_class_metrics_test, test_res['class_metrics'])

        unique_cls = np.unique(np.concatenate([y_test_multi, test_res['y_pred']]))
        cls_labels = [class_map_6.get(c, str(c)) for c in unique_cls]
        _plot_confusion_matrix(
            test_res['confusion_matrix'], cls_labels,
            f"Confusion Matrix - Split {split_idx+1} - Test", cmap='Reds')

        all_train_results.append(train_res)
        all_val_results.append(val_res)
        all_test_results.append(test_res)
        all_best_params.append(best_params)
        all_best_models.append(best_model)
        all_cms_test.append(test_res['confusion_matrix'])
        all_y_test_multi.append(y_test_multi)
        all_y_test_pred.append(test_res['y_pred'])

    # --- Aggregate summary ---
    avg_train, std_train = _aggregate_scalar_metrics(all_train_results)
    avg_val,   std_val   = _aggregate_scalar_metrics(all_val_results)
    avg_test,  std_test  = _aggregate_scalar_metrics(all_test_results)

    _print_aggregate_summary(
        avg_train, std_train, avg_val, std_val, avg_test, std_test,
        all_class_metrics_test)

    # Aggregate confusion matrices
    _plot_aggregate_confusion_matrix(
        all_y_test_multi, all_y_test_pred, n_splits,
        class_map_6, list_classes=None)

    # TSS and accuracy per split
    _plot_metric_per_split('tss',      all_train_results, all_val_results, all_test_results, n_splits)
    _plot_metric_per_split('accuracy', all_train_results, all_val_results, all_test_results, n_splits)

    return {
        'all_train_results': all_train_results,
        'all_val_results':   all_val_results,
        'all_test_results':  all_test_results,
        'all_cms_test':      all_cms_test,
        'all_best_models':   all_best_models,
        'all_best_params':   all_best_params,
        'all_y_test_true':   all_y_test_multi,
        'all_y_test_pred':   all_y_test_pred,
        'class_metrics_train': all_class_metrics_train,
        'class_metrics_val':   all_class_metrics_val,
        'class_metrics_test':  all_class_metrics_test,
        'avg_train_metrics': {**avg_train, **{f'std_{k}': v for k, v in std_train.items()}},
        'avg_val_metrics':   {**avg_val,   **{f'std_{k}': v for k, v in std_val.items()}},
        'avg_test_metrics':  {**avg_test,  **{f'std_{k}': v for k, v in std_test.items()}},
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _print_split_metrics(split_name, results):
    """Print accuracy, TSS, F1, and per-class metrics for one split."""
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print(f"TSS:      {results['tss']:.4f}")
    print(f"F1 macro: {results['macro_f1']:.4f}")
    print(f"\nPer-class metrics ({split_name} set):")
    for cls_name, m in results['class_metrics'].items():
        print(f"\n  Class {cls_name}:")
        print(f"    Precision:   {m['precision']:.4f}")
        print(f"    Sensitivity: {m['sensitivity']:.4f}")
        print(f"    Specificity: {m['specificity']:.4f}")
        print(f"    F1-Score:    {m['f1_score']:.4f}")
        print(f"    TSS:         {m['tss']:.4f}")
        print(f"    Accuracy:    {m['accuracy']:.4f}")
        print(f"    TP={m['true_positive']}, FP={m['false_positive']}, "
              f"FN={m['false_negative']}, TN={m['true_negative']}")


def _accumulate_class_metrics(store, class_metrics_dict):
    """Append per-class metrics from one split into the running store."""
    for cls_name, m in class_metrics_dict.items():
        if cls_name not in store:
            store[cls_name] = {k: [] for k in
                               ('precision', 'sensitivity', 'specificity',
                                'f1_score', 'tss', 'accuracy')}
        for key in store[cls_name]:
            store[cls_name][key].append(m[key])


def _plot_confusion_matrix(cm, class_labels, title, cmap):
    """Plot a single confusion matrix heatmap."""
    vmin, vmax = cm.min(), cm.max()
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                     xticklabels=class_labels, yticklabels=class_labels,
                     annot_kws={"fontsize": 20},
                     cbar_kws={'ticks': np.linspace(vmin, vmax, 5)})
    plt.title(title)
    plt.ylabel('TRUE', fontsize=20)
    plt.xlabel('PREDICTED', fontsize=20)
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(rotation=0,  fontsize=20)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    plt.tight_layout()
    plt.show()


def _aggregate_scalar_metrics(results_list):
    """Return mean and std dicts for accuracy, tss, macro_f1 across splits."""
    avg = {k: np.mean([r[k] for r in results_list])
           for k in ('accuracy', 'tss', 'macro_f1')}
    std = {k: np.std([r[k] for r in results_list])
           for k in ('accuracy', 'tss', 'macro_f1')}
    return avg, std


def _print_aggregate_summary(avg_train, std_train, avg_val, std_val,
                              avg_test, std_test, all_class_metrics_test):
    """Print the aggregate mean ± std summary across all splits."""
    print("\n\n" + "="*50)
    print("AVERAGE METRICS ACROSS ALL SPLITS")
    print("="*50)

    for set_name, avg, std in [("Training",   avg_train, std_train),
                                ("Validation", avg_val,   std_val),
                                ("Test",       avg_test,  std_test)]:
        print(f"\n{set_name} Set Metrics:")
        print(f"  Average Accuracy: {avg['accuracy']:.4f} ± {std['accuracy']:.4f}")
        print(f"  Average TSS:      {avg['tss']:.4f} ± {std['tss']:.4f}")
        print(f"  Average F1 Macro: {avg['macro_f1']:.4f} ± {std['macro_f1']:.4f}")

    print("\n\n" + "="*50)
    print("AVERAGE CLASS METRICS ACROSS ALL SPLITS")
    print("="*50)
    print("\nTest Set Class Metrics:")
    for cls_name in sorted(all_class_metrics_test.keys()):
        m = all_class_metrics_test[cls_name]
        print(f"\n  Class {cls_name}:")
        for key, label in [('precision', 'Precision'), ('sensitivity', 'Recall (Sensitivity)'),
                            ('specificity', 'Specificity'), ('f1_score', 'F1-Score'),
                            ('tss', 'TSS'), ('accuracy', 'Accuracy')]:
            print(f"    {label}: {np.mean(m[key]):.4f} ± {np.std(m[key]):.4f}")


def _plot_aggregate_confusion_matrix(all_y_true, all_y_pred, n_splits, class_map, list_classes):
    """Plot mean and std confusion matrices aggregated across splits."""
    all_cls = sorted(set(
        y for ys in all_y_true  for y in ys).union(
        y for ys in all_y_pred  for y in ys))
    n_cls = len(all_cls)
    all_full_cms = np.zeros((n_splits, n_cls, n_cls))

    for i in range(n_splits):
        for tl, pl in zip(all_y_true[i], all_y_pred[i]):
            ti = all_cls.index(tl)
            pi = all_cls.index(pl)
            all_full_cms[i, ti, pi] += 1

    mean_cm = np.mean(all_full_cms, axis=0)
    std_cm  = np.std(all_full_cms,  axis=0)
    cls_lbl = [class_map.get(c, str(c)) for c in all_cls]

    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_cm, annot=True, fmt='.1f', cmap='YlOrBr',
                xticklabels=cls_lbl, yticklabels=cls_lbl)
    plt.title("Average Confusion Matrix (Test Set)")
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(std_cm, annot=True, fmt='.2f', cmap='YlOrBr',
                xticklabels=cls_lbl, yticklabels=cls_lbl)
    plt.title("Standard Deviation of Confusion Matrix (Test Set)")
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.show()

    combined_cm = np.empty(mean_cm.shape, dtype=object)
    for i in range(mean_cm.shape[0]):
        for j in range(mean_cm.shape[1]):
            combined_cm[i, j] = (f"{mean_cm[i,j]:.1f}±{std_cm[i,j]:.1f}"
                                 if mean_cm[i, j] > 0 else "0")
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(mean_cm, cmap='YlOrBr',
                     xticklabels=cls_lbl, yticklabels=cls_lbl, annot=False)
    for i in range(mean_cm.shape[0]):
        for j in range(mean_cm.shape[1]):
            color = "black" if mean_cm[i, j] < np.max(mean_cm) / 2 else "white"
            ax.text(j + 0.5, i + 0.5, combined_cm[i, j], ha="center", va="center", color=color)
    plt.title("Confusion Matrix with Mean ± Std (Test Set)")
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.show()


def _plot_metric_per_split(metric_key, all_train, all_val, all_test, n_splits):
    """Line plot of a scalar metric across all splits with mean reference lines."""
    train_vals = [r[metric_key] for r in all_train]
    val_vals   = [r[metric_key] for r in all_val]
    test_vals  = [r[metric_key] for r in all_test]
    xs = range(1, n_splits + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(xs, train_vals, 'o-', color='blue',  label=f'Train {metric_key}')
    plt.plot(xs, val_vals,   's-', color='green', label=f'Validation {metric_key}')
    plt.plot(xs, test_vals,  'D-', color='red',   label=f'Test {metric_key}')

    plt.axhline(np.mean(train_vals), color='blue',  linestyle='--', alpha=0.5, label='Train Mean')
    plt.axhline(np.mean(val_vals),   color='green', linestyle='--', alpha=0.5, label='Validation Mean')
    plt.axhline(np.mean(test_vals),  color='red',   linestyle='--', alpha=0.5, label='Test Mean')

    plt.title(f'{metric_key} for each Split')
    plt.xlabel('Split')
    plt.ylabel(metric_key)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()