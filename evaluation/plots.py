"""ROC, PR, reliability, and histogram plots for head evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def plot_roc_pr_multi(
    fold_id: int,
    y_test: np.ndarray,
    prob_by_model: dict[str, np.ndarray],
    out_dir: Path,
    title_suffix: str = "",
) -> None:
    """One figure with ROC and PR subplots for all models on the same test labels."""
    y_test = y_test.astype(np.int64)
    if len(np.unique(y_test)) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(prob_by_model), 1)))

    for ax, kind in zip(axes, ("roc", "pr")):
        for (name, probs), c in zip(prob_by_model.items(), colors):
            if kind == "roc":
                fpr, tpr, _ = roc_curve(y_test, probs)
                xx, yy = fpr, tpr
                score = auc(fpr, tpr)
                xlab, ylab = "FPR", "TPR"
                lbl = f"{name} (AUC={score:.3f})"
            else:
                prec, rec, _ = precision_recall_curve(y_test, probs)
                xx, yy = rec, prec
                score = auc(rec, prec)
                xlab, ylab = "Recall", "Precision"
                lbl = f"{name} (AP={score:.3f})"
            ax.plot(xx, yy, color=c, lw=1.8, label=lbl)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f"Fold {fold_id} — {kind.upper()} {title_suffix}".strip())
        ax.legend(fontsize=7, loc="best")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"fold_{fold_id}_roc_pr.png", dpi=150)
    plt.close(fig)


def plot_reliability(
    bin_result: dict,
    title: str,
    out_path: Path,
) -> None:
    mean_pred = np.array(bin_result["mean_predicted"], dtype=float)
    frac_pos = np.array(bin_result["fraction_positive"], dtype=float)
    centers = (np.array(bin_result["bin_low"]) + np.array(bin_result["bin_high"])) / 2.0
    valid = np.isfinite(mean_pred) & np.isfinite(frac_pos)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    ax.plot(mean_pred[valid], frac_pos[valid], "o-", lw=1.2, markersize=6)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_prob_histograms(
    y_test: np.ndarray,
    probs: np.ndarray,
    title: str,
    out_path: Path,
    bins: int = 30,
) -> None:
    y_test = y_test.astype(np.int64)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        probs[y_test == 0],
        bins=bins,
        alpha=0.55,
        label="non-MIT (y=0)",
        density=True,
        color="C0",
    )
    ax.hist(
        probs[y_test == 1],
        bins=bins,
        alpha=0.55,
        label="MIT (y=1)",
        density=True,
        color="C1",
    )
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
