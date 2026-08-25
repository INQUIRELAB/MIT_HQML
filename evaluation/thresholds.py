"""Validation-only threshold tuning (F1, balanced accuracy, Youden's J)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def tune_threshold_f1(y_true: np.ndarray, scores: np.ndarray, n_grid: int = 1001) -> tuple[float, float]:
    """Grid search threshold in [0, 1] maximizing F1 on validation."""
    y_true = y_true.astype(np.int64)
    thresholds = np.linspace(0.0, 1.0, n_grid)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int64)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def tune_threshold_balanced_accuracy(
    y_true: np.ndarray, scores: np.ndarray, n_grid: int = 1001
) -> tuple[float, float]:
    y_true = y_true.astype(np.int64)
    thresholds = np.linspace(0.0, 1.0, n_grid)
    best_t, best_ba = 0.5, -1.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int64)
        ba = float(balanced_accuracy_score(y_true, pred))
        if ba > best_ba:
            best_ba = ba
            best_t = float(t)
    return best_t, best_ba


def tune_threshold_youden_j(y_true: np.ndarray, scores: np.ndarray, n_grid: int = 1001) -> tuple[float, float]:
    """
    Youden's J = TPR - FPR = sensitivity + specificity - 1, maximized on a [0,1] threshold grid.
    """
    from sklearn.metrics import confusion_matrix

    y_true = y_true.astype(np.int64)
    thresholds = np.linspace(0.0, 1.0, n_grid)
    best_t, best_j = 0.5, -1e18
    for t in thresholds:
        pred = (scores >= t).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_t = float(t)
    if best_j < -1e17:
        return 0.5, float("nan")
    return best_t, float(best_j)
