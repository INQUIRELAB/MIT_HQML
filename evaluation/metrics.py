"""Classification metrics (ranking + hard labels)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, pred_labels: np.ndarray) -> dict[str, Any]:
    y_true = y_true.astype(np.int64)
    out: dict[str, Any] = {}
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred_labels))
    out["f1"] = float(f1_score(y_true, pred_labels, zero_division=0))
    out["recall"] = float(recall_score(y_true, pred_labels, zero_division=0))
    out["precision"] = float(precision_score(y_true, pred_labels, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, pred_labels, labels=[0, 1]).ravel()
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    out["TN"] = int(tn)
    out["FP"] = int(fp)
    out["FN"] = int(fn)
    out["TP"] = int(tp)
    if len(np.unique(y_true)) < 2:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
        out["average_precision"] = float("nan")
    else:
        out["roc_auc"] = float(roc_auc_score(y_true, probs))
        ap = float(average_precision_score(y_true, probs))
        out["pr_auc"] = ap
        out["average_precision"] = ap
    return out
