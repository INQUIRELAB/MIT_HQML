"""Calibration: Brier, Platt scaling, isotonic regression, reliability bins."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss as sk_brier_score_loss


def compute_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    probs = np.clip(probs.astype(np.float64), 1e-15, 1.0 - 1e-15)
    return float(sk_brier_score_loss(y_true, probs))


def fit_platt_scaler(val_scores: np.ndarray, val_y: np.ndarray) -> LogisticRegression:
    """Platt scaling: logistic regression on unbounded scores (use logits or probs)."""
    X = val_scores.reshape(-1, 1).astype(np.float64)
    y = val_y.astype(np.int64)
    lr = LogisticRegression(solver="lbfgs", C=1e9, max_iter=1000)
    lr.fit(X, y)
    return lr


def apply_platt(lr: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    X = scores.reshape(-1, 1).astype(np.float64)
    return lr.predict_proba(X)[:, 1].astype(np.float64)


def fit_isotonic(val_probs: np.ndarray, val_y: np.ndarray) -> IsotonicRegression:
    """Isotonic regression on validation probabilities -> [0,1]."""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(val_probs.astype(np.float64), val_y.astype(np.float64))
    return ir


def apply_isotonic(ir: IsotonicRegression, probs: np.ndarray) -> np.ndarray:
    return ir.predict(probs.astype(np.float64)).astype(np.float64)


def reliability_bins(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list]:
    """
    Histogram binning for reliability diagram.
    Returns bin edges, mean predicted prob per bin, fraction of positives per bin, counts.
    """
    y_true = y_true.astype(np.int64)
    probs = probs.astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_low = bins[:-1]
    bin_high = bins[1:]
    mean_pred: list[float] = []
    frac_pos: list[float] = []
    counts: list[int] = []
    for lo, hi in zip(bin_low, bin_high):
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        counts.append(n)
        if n == 0:
            mean_pred.append(float("nan"))
            frac_pos.append(float("nan"))
        else:
            mean_pred.append(float(np.mean(probs[mask])))
            frac_pos.append(float(np.mean(y_true[mask])))
    return {
        "bin_low": [float(x) for x in bin_low],
        "bin_high": [float(x) for x in bin_high],
        "mean_predicted": mean_pred,
        "fraction_positive": frac_pos,
        "counts": counts,
    }
