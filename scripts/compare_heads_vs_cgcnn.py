#!/usr/bin/env python3
"""
Compare post-hoc head CV test metrics (from results/heads/summary.json) to the
raw CGCNN binary classifier probabilities on the **same** grouped test folds.

Uses the same embedding-valid rows and fold JSON test indices as train_heads_on_embeddings.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _emb_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("emb_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x)


def _valid_embedding_mask(df: pd.DataFrame, emb_cols: list[str]) -> np.ndarray:
    return df[emb_cols].notna().all(axis=1).to_numpy()


def _filter_indices(indices: list[int], valid: np.ndarray) -> list[int]:
    return [i for i in indices if 0 <= i < len(valid) and valid[i]]


def _binary_metrics(y_true: np.ndarray, probs: np.ndarray, pred_labels: np.ndarray) -> dict[str, Any]:
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


def _aggregate_metric_list(
    fold_metrics: list[dict[str, Any]], keys: list[str]
) -> dict[str, dict[str, float] | None]:
    agg: dict[str, dict[str, float] | None] = {}
    for k in keys:
        vals = []
        for m in fold_metrics:
            v = m.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(float(v))
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": float(len(vals))}
        else:
            agg[k] = None
    return agg


def _fmt_agg(agg: dict[str, dict[str, float] | None] | None, key: str) -> str:
    if not agg or not agg.get(key):
        return "n/a"
    b = agg[key]
    assert b is not None
    return f"{b['mean']:.4f} ± {b['std']:.4f}"


def evaluate_cgcnn_baseline(
    df: pd.DataFrame,
    emb_cols: list[str],
    valid: np.ndarray,
    splits_dir: Path,
    prob_col: str = "cgcnn_prob_mit",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from evaluation.fold_utils import list_fold_indices

    if prob_col not in df.columns:
        raise ValueError(f"Parquet missing column {prob_col!r}")
    y_all = df["Label"].to_numpy(dtype=np.int64)
    fold_rows: list[dict[str, Any]] = []
    for k in list_fold_indices(splits_dir):
        fold_path = splits_dir / f"fold_{k}.json"
        with open(fold_path) as f:
            spec = json.load(f)
        test_idx = _filter_indices(spec["test_indices"], valid)
        probs = df.iloc[test_idx][prob_col].to_numpy(dtype=np.float64)
        y = y_all[test_idx]
        ok = np.isfinite(probs) & np.isfinite(y.astype(float))
        if not ok.all():
            n_bad = int((~ok).sum())
            raise ValueError(f"Fold {k}: {n_bad} test rows with non-finite {prob_col} or label.")
        pred = (probs >= 0.5).astype(np.int64)
        m = _binary_metrics(y, probs, pred)
        fold_rows.append({"fold": k, "n_test": len(test_idx), "test_metrics": m})
    keys = [
        "balanced_accuracy",
        "f1",
        "recall",
        "precision",
        "specificity",
        "roc_auc",
        "pr_auc",
        "average_precision",
    ]
    aggregate = _aggregate_metric_list([r["test_metrics"] for r in fold_rows], keys)
    return fold_rows, aggregate


def main() -> int:
    p = argparse.ArgumentParser(description="Compare heads summary vs CGCNN-only baseline on grouped folds.")
    p.add_argument("--embedding_path", type=str, default="embeddings/cgcnn_binary_embeddings.parquet")
    p.add_argument("--splits_dir", type=str, default="splits/grouped_5fold")
    p.add_argument("--heads_summary", type=str, default="results/heads/summary.json")
    p.add_argument("--output", type=str, default="results/heads/comparison_vs_cgcnn.json")
    p.add_argument(
        "--merge_summary",
        action="append",
        default=[],
        metavar="REL_PATH",
        help=(
            "Additional summary.json (relative to repo root) to merge into by_head "
            "(e.g. results/reup/my_run/summary.json). Repeatable; later keys override earlier."
        ),
    )
    args = p.parse_args()

    emb_path = (ROOT / args.embedding_path).resolve()
    splits_dir = (ROOT / args.splits_dir).resolve()
    summary_path = (ROOT / args.heads_summary).resolve()
    out_path = (ROOT / args.output).resolve()

    df = pd.read_parquet(emb_path)
    emb_cols = _emb_columns(df)
    valid = _valid_embedding_mask(df, emb_cols)

    cgcnn_folds, cgcnn_agg = evaluate_cgcnn_baseline(df, emb_cols, valid, splits_dir)

    with open(summary_path) as f:
        summary = json.load(f)
    by_head: dict[str, Any] = dict(summary.get("by_head", {}))
    for rel in args.merge_summary or []:
        mp = (ROOT / rel).resolve()
        if not mp.is_file():
            continue
        with open(mp) as f:
            other = json.load(f)
        for k, v in (other.get("by_head") or {}).items():
            by_head[k] = v

    rows_out: list[dict[str, Any]] = [
        {
            "model": "cgcnn_only",
            "description": "Frozen CGCNN classifier probability (cgcnn_prob_mit), same test folds",
            "aggregate": cgcnn_agg,
            "folds": cgcnn_folds,
        }
    ]
    head_descriptions = {
        "linear": "Post-hoc linear on frozen embeddings",
        "mlp": "Post-hoc MLP on frozen embeddings",
        "quantum": "Post-hoc quantum (PennyLane angle + entangler) on frozen embeddings",
        "reup": "Post-hoc REUP (data re-upload) on frozen embeddings",
    }
    for name in ("linear", "mlp", "quantum", "reup"):
        if name not in by_head:
            continue
        block = by_head[name]
        rows_out.append(
            {
                "model": f"head_{name}",
                "description": head_descriptions.get(name, f"Post-hoc {name} on frozen embeddings"),
                "aggregate": block.get("aggregate"),
                "source": str(args.heads_summary),
            }
        )

    ranking: list[tuple[str, float, float]] = []
    for r in rows_out:
        agg = r["aggregate"]
        pr_m = agg.get("pr_auc") if agg else None
        roc_m = agg.get("roc_auc") if agg else None
        pr_mean = pr_m["mean"] if pr_m else float("-inf")
        roc_mean = roc_m["mean"] if roc_m else float("-inf")
        ranking.append((r["model"], pr_mean, roc_mean))
    ranking.sort(key=lambda t: (t[1], t[2]), reverse=True)
    best_model = ranking[0][0]

    payload = {
        "best_model_by_pr_auc_then_roc_auc": best_model,
        "ranking": [{"model": m, "pr_auc_mean": pr, "roc_auc_mean": roc} for m, pr, roc in ranking],
        "models": rows_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print()
    print("=" * 72)
    print("  Test-set comparison (grouped folds, embedding-resolved test rows only)")
    print("=" * 72)
    print(f"Embeddings: {args.embedding_path}")
    print(f"Heads summary: {args.heads_summary}")
    print()
    hdr = f"{'model':<14} {'bal_acc':<18} {'ROC-AUC':<18} {'PR-AUC':<18} {'F1':<18}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows_out:
        agg = r["aggregate"]
        label = r["model"]
        print(
            f"{label:<14} "
            f"{_fmt_agg(agg, 'balanced_accuracy'):<18} "
            f"{_fmt_agg(agg, 'roc_auc'):<18} "
            f"{_fmt_agg(agg, 'pr_auc'):<18} "
            f"{_fmt_agg(agg, 'f1'):<18}"
        )
    print()
    print(f"Best (rank by PR-AUC mean, then ROC-AUC mean): {best_model}")
    print(f"Wrote {out_path.relative_to(ROOT)}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
