#!/usr/bin/env python3
"""
Aggregate reviewer-eval experiment outputs and run paired statistical tests.

Recursively collects ``predictions_test.csv`` and ``metrics.json`` under:
  - results/reviewer_eval/main/
  - results/reviewer_eval/parameter_matched/
  - results/reviewer_eval/imbalance_ablation/
  - results/reviewer_eval/quantum_architecture_ablation/
  - results/reviewer_eval/embedding_compression_ablation/
  - results/reviewer_eval/shot_noise_ablation/

Writes to results/reviewer_eval/aggregated/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_DIRS: dict[str, str] = {
    "main": "results/reviewer_eval/main",
    "parameter_matched": "results/reviewer_eval/parameter_matched",
    "imbalance_ablation": "results/reviewer_eval/imbalance_ablation",
    "quantum_architecture_ablation": "results/reviewer_eval/quantum_architecture_ablation",
    "embedding_compression_ablation": "results/reviewer_eval/embedding_compression_ablation",
    "shot_noise_ablation": "results/reviewer_eval/shot_noise_ablation",
}

# Default workspace root for this package's fair-protocol work.
DEFAULT_WORK_ROOT = "qml_mit_classification/results/reviewer_eval"

METRIC_COLS = [
    "pr_auc",
    "roc_auc",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "mcc",
    "brier_score",
    "ece",
]

# Maps metrics.json keys -> canonical names (tuned threshold preferred where noted)
METRICS_JSON_MAP = {
    "pr_auc": "pr_auc",
    "roc_auc": "roc_auc",
    "balanced_accuracy": "balanced_accuracy_at_tuned_threshold",
    "f1": "f1_at_tuned_threshold",
    "precision": "precision_at_tuned_threshold",
    "recall": "recall_at_tuned_threshold",
    "mcc": "mcc_at_tuned_threshold",
    "brier_score": "brier_score",
    "ece": "ece",
}

PAIRED_COMPARISONS: list[tuple[str, str]] = [
    ("quantum", "cgcnn_only"),
    ("quantum", "mlp"),
    ("quantum", "linear"),
    ("reup", "quantum"),
    ("reup", "mlp"),
    ("reup", "cgcnn_only"),
    ("reup", "latent_mlp_reup_matched"),
    ("reup", "proj_bottleneck_mlp"),
    ("reup", "latent_bottleneck_mlp"),
    ("quantum", "latent_mlp_vqc_matched"),
    ("quantum", "proj_bottleneck_mlp"),
    ("quantum", "latent_bottleneck_mlp"),
]

TOP_K_VALUES = (5, 10, 20)


@dataclass
class RunRecord:
    experiment_type: str
    run_dir: Path
    rel_path: str
    head_type: str
    fold: int
    seed: int
    mit_fraction: float
    imbalance_mode: str = "original"
    embedding_variant: str = ""
    architecture_config: str = ""
    quantum_backend: str = ""
    shots: int | None = None
    noise_model: str = ""
    noise_prob: float | None = None
    shot_noise_sweep: str = ""
    stage: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    predictions: pd.DataFrame | None = None


def _parse_mit_fraction(tag: str) -> float | None:
    m = re.match(r"mit_fraction_(.+)$", tag)
    if not m:
        return None
    return float(m.group(1))


def _parse_seed(tag: str) -> int | None:
    m = re.match(r"seed_(\d+)$", tag)
    return int(m.group(1)) if m else None


def _parse_fold(tag: str) -> int | None:
    m = re.match(r"fold_(\d+)$", tag)
    return int(m.group(1)) if m else None


def _architecture_config_from_hp(hp: dict[str, Any]) -> str:
    if not hp:
        return ""
    keys = (
        "n_qubits",
        "n_q_layers",
        "n_reup_layers",
        "entanglement_type",
        "encoding_type",
        "measurement_type",
    )
    parts = [f"{k}={hp[k]}" for k in keys if k in hp]
    return "|".join(parts)


def _architecture_config_from_path(parts: list[str]) -> str:
    out: dict[str, str] = {}
    for p in parts:
        if p.startswith("stage_"):
            out["stage"] = p.replace("stage_", "")
        elif p.startswith("nq_"):
            out["nq"] = p.replace("nq_", "")
        elif p.startswith("layers_"):
            out["layers"] = p.replace("layers_", "")
        elif p.startswith("ent_"):
            out["ent"] = p.replace("ent_", "")
        elif p.startswith("enc_"):
            out["enc"] = p.replace("enc_", "")
        elif p.startswith("meas_"):
            out["meas"] = p.replace("meas_", "")
    if not out:
        return ""
    return "|".join(f"{k}={v}" for k, v in sorted(out.items()))


def parse_run_metadata(experiment_type: str, run_dir: Path, exp_root: Path) -> dict[str, Any]:
    """Infer run metadata from path relative to experiment root and optional config.json."""
    rel = run_dir.relative_to(exp_root)
    parts = list(rel.parts)
    meta: dict[str, Any] = {
        "head_type": "",
        "fold": None,
        "seed": None,
        "mit_fraction": None,
        "imbalance_mode": "original",
        "embedding_variant": "",
        "architecture_config": "",
        "quantum_backend": "",
        "shots": None,
        "noise_model": "",
        "noise_prob": None,
        "shot_noise_sweep": "",
        "stage": "",
    }

    cfg_path = run_dir / "config.json"
    cfg: dict[str, Any] = {}
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            pass

    if cfg:
        meta["head_type"] = str(cfg.get("head_type", meta["head_type"]))
        meta["fold"] = cfg.get("fold", meta["fold"])
        meta["seed"] = cfg.get("seed", meta["seed"])
        meta["mit_fraction"] = cfg.get("mit_fraction", meta["mit_fraction"])
        meta["imbalance_mode"] = str(cfg.get("imbalance_mode", meta["imbalance_mode"]))
        meta["embedding_variant"] = str(cfg.get("embedding_variant", meta["embedding_variant"]))
        meta["quantum_backend"] = str(cfg.get("quantum_backend", meta["quantum_backend"]))
        meta["shots"] = cfg.get("shots", meta["shots"])
        meta["noise_model"] = str(cfg.get("noise_model", meta["noise_model"]))
        meta["noise_prob"] = cfg.get("noise_prob", meta["noise_prob"])
        hp = cfg.get("model_hyperparameters") or {}
        meta["architecture_config"] = _architecture_config_from_hp(hp)

    # Path-based fallbacks
    for i, p in enumerate(parts):
        if p.startswith("mit_fraction_"):
            meta["mit_fraction"] = _parse_mit_fraction(p)
        elif p.startswith("seed_"):
            meta["seed"] = _parse_seed(p)
        elif p.startswith("fold_"):
            meta["fold"] = _parse_fold(p)

    if experiment_type == "main" or experiment_type == "parameter_matched":
        if parts and not meta["head_type"]:
            meta["head_type"] = parts[0]
    elif experiment_type == "imbalance_ablation":
        if len(parts) >= 1 and not meta["head_type"]:
            meta["head_type"] = parts[0]
        if len(parts) >= 2 and parts[1] not in ("mit_fraction",) and not parts[1].startswith("mit_fraction_"):
            meta["imbalance_mode"] = parts[1]
    elif experiment_type == "quantum_architecture_ablation":
        meta["architecture_config"] = meta["architecture_config"] or _architecture_config_from_path(parts)
        for p in parts:
            if p.startswith("stage_"):
                meta["stage"] = p
            if p in ("quantum", "reup") and not meta["head_type"]:
                meta["head_type"] = p
    elif experiment_type == "embedding_compression_ablation":
        if len(parts) >= 1 and not meta["embedding_variant"]:
            meta["embedding_variant"] = parts[0]
        if len(parts) >= 2 and not meta["head_type"]:
            meta["head_type"] = parts[1]
    elif experiment_type == "shot_noise_ablation":
        if len(parts) >= 1 and not meta["head_type"]:
            meta["head_type"] = parts[0]
        if len(parts) >= 2:
            meta["shot_noise_sweep"] = parts[1]
            if parts[1] == "shot_sweep" and len(parts) >= 3:
                tag = parts[2].replace("shots_", "")
                if tag == "exact":
                    meta["quantum_backend"] = meta["quantum_backend"] or "exact"
                else:
                    meta["quantum_backend"] = meta["quantum_backend"] or "shots"
                    try:
                        meta["shots"] = int(tag)
                    except ValueError:
                        pass
            elif parts[1] == "noise_sweep" and len(parts) >= 3:
                meta["quantum_backend"] = meta["quantum_backend"] or "noisy"
                nm = parts[2]
                for nm_name in ("depolarizing", "readout", "amplitude_damping"):
                    if nm.startswith(nm_name):
                        meta["noise_model"] = nm_name
                        prob_s = nm[len(nm_name) :].lstrip("_").replace("p", ".")
                        try:
                            meta["noise_prob"] = float(prob_s)
                        except ValueError:
                            pass
                        break

    return meta


def extract_canonical_metrics(metrics_raw: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for canon, raw_key in METRICS_JSON_MAP.items():
        v = metrics_raw.get(raw_key)
        if v is None and canon in metrics_raw:
            v = metrics_raw.get(canon)
        if v is None:
            out[canon] = None
        else:
            try:
                out[canon] = float(v)
            except (TypeError, ValueError):
                out[canon] = None
    if metrics_raw.get("ece_computed") is False:
        out["ece"] = None
    return out


def discover_fold_dirs(exp_root: Path, logger: logging.Logger) -> list[Path]:
    """Return directories containing both metrics.json and predictions_test.csv.

    Follows directory symlinks so workspace merge trees (fair + legacy heads) work.
    """
    found: list[Path] = []
    if not exp_root.is_dir():
        logger.warning("Experiment root missing: %s", exp_root)
        return found
    # pathlib.Path.rglob does not follow symlinks; walk does when requested.
    import os

    for root, _dirs, files in os.walk(exp_root, followlinks=True):
        if "metrics.json" not in files:
            continue
        fold_dir = Path(root)
        if not fold_dir.name.startswith("fold_"):
            continue
        pred_path = fold_dir / "predictions_test.csv"
        if pred_path.is_file():
            found.append(fold_dir)
        else:
            logger.warning("Missing predictions_test.csv in %s", fold_dir)
    return sorted(set(found))


def load_run(
    experiment_type: str,
    fold_dir: Path,
    exp_root: Path,
    logger: logging.Logger,
    *,
    load_predictions: bool = True,
) -> RunRecord | None:
    metrics_path = fold_dir / "metrics.json"
    pred_path = fold_dir / "predictions_test.csv"
    try:
        metrics_raw = json.loads(metrics_path.read_text())
    except Exception as exc:
        logger.warning("Failed to read %s: %s", metrics_path, exc)
        return None

    meta = parse_run_metadata(experiment_type, fold_dir, exp_root)
    if meta["fold"] is None or meta["seed"] is None or meta["mit_fraction"] is None:
        logger.warning("Incomplete metadata for %s", fold_dir)
    if not meta["head_type"]:
        logger.warning("Missing head_type for %s", fold_dir)

    preds = None
    if load_predictions:
        try:
            preds = pd.read_csv(pred_path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", pred_path, exc)

    return RunRecord(
        experiment_type=experiment_type,
        run_dir=fold_dir,
        rel_path=str(fold_dir.relative_to(ROOT)),
        head_type=str(meta["head_type"]),
        fold=int(meta["fold"]) if meta["fold"] is not None else -1,
        seed=int(meta["seed"]) if meta["seed"] is not None else -1,
        mit_fraction=float(meta["mit_fraction"]) if meta["mit_fraction"] is not None else float("nan"),
        imbalance_mode=str(meta["imbalance_mode"] or "original"),
        embedding_variant=str(meta["embedding_variant"] or ""),
        architecture_config=str(meta["architecture_config"] or ""),
        quantum_backend=str(meta["quantum_backend"] or ""),
        shots=int(meta["shots"]) if meta["shots"] is not None else None,
        noise_model=str(meta["noise_model"] or ""),
        noise_prob=float(meta["noise_prob"]) if meta["noise_prob"] is not None else None,
        shot_noise_sweep=str(meta["shot_noise_sweep"] or ""),
        stage=str(meta["stage"] or ""),
        metrics=extract_canonical_metrics(metrics_raw),
        predictions=preds,
    )


def runs_to_metrics_long(runs: list[RunRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in runs:
        row: dict[str, Any] = {
            "experiment_type": r.experiment_type,
            "run_path": r.rel_path,
            "head_type": r.head_type,
            "fold": r.fold,
            "seed": r.seed,
            "mit_fraction": r.mit_fraction,
            "imbalance_mode": r.imbalance_mode,
            "embedding_variant": r.embedding_variant or "default",
            "architecture_config": r.architecture_config or "default",
            "quantum_backend": r.quantum_backend or "",
            "shots": r.shots,
            "noise_model": r.noise_model or "",
            "noise_prob": r.noise_prob,
            "shot_noise_sweep": r.shot_noise_sweep or "",
            "stage": r.stage or "",
        }
        row.update(r.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def _agg_ci(values: np.ndarray) -> dict[str, float]:
    v = values[~np.isnan(values)]
    n = len(v)
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "se": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "n": 0}
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else np.nan
    z = 1.96
    return {
        "mean": mean,
        "std": std,
        "se": float(se),
        "ci95_low": mean - z * se,
        "ci95_high": mean + z * se,
        "n": n,
    }


def build_metrics_summary(metrics_long: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "experiment_type",
        "head_type",
        "mit_fraction",
        "imbalance_mode",
        "embedding_variant",
        "architecture_config",
    ]
    rows: list[dict[str, Any]] = []
    for keys, grp in metrics_long.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for metric in METRIC_COLS:
            if metric not in grp.columns:
                continue
            stats_d = _agg_ci(grp[metric].to_numpy(dtype=float))
            row = {**base, "metric": metric, **stats_d}
            rows.append(row)
    return pd.DataFrame(rows)


def _ece_from_probs(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float | None:
    y = y.astype(np.int64)
    p = np.clip(p.astype(np.float64), 0.0, 1.0)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y[mask] == 1))
        conf = float(np.mean(p[mask]))
        ece += abs(acc - conf) * (np.sum(mask) / len(y))
    return float(ece)


def _calibration_slope(y: np.ndarray, p: np.ndarray) -> float | None:
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    if len(y) < 3 or len(np.unique(y)) < 2:
        return None
    x = np.log(p / (1 - p))
    try:
        slope, _intercept, _r, _p, _se = stats.linregress(x, y.astype(np.float64))
        return float(slope)
    except Exception:
        return None


def _topk_metrics(y: np.ndarray, p: np.ndarray, k: int) -> dict[str, float | None]:
    n = len(y)
    if n == 0:
        return {f"precision@{k}": None, f"recall@{k}": None, f"enrichment@{k}": None}
    order = np.argsort(-p)
    top = order[: min(k, n)]
    y_top = y[top]
    tp = int(np.sum(y_top == 1))
    n_pos = int(np.sum(y == 1))
    prec = tp / len(top) if len(top) else None
    rec = tp / n_pos if n_pos > 0 else None
    base = n_pos / n if n > 0 else None
    enrich = (prec / base) if (prec is not None and base and base > 0) else None
    return {f"precision@{k}": prec, f"recall@{k}": rec, f"enrichment@{k}": enrich}


def compute_prediction_derived_tables(
    runs: list[RunRecord],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build top-k and calibration tables from loaded predictions."""
    topk_rows: list[dict[str, Any]] = []
    cal_rows: list[dict[str, Any]] = []

    for r in runs:
        if r.predictions is None or r.predictions.empty:
            continue
        df = r.predictions
        if "true_label" not in df.columns or "probability" not in df.columns:
            logger.warning("Predictions missing columns in %s", r.run_dir)
            continue
        y = df["true_label"].to_numpy(dtype=np.int64)
        p = df["probability"].to_numpy(dtype=np.float64)

        base = {
            "experiment_type": r.experiment_type,
            "run_path": r.rel_path,
            "head_type": r.head_type,
            "fold": r.fold,
            "seed": r.seed,
            "mit_fraction": r.mit_fraction,
            "imbalance_mode": r.imbalance_mode,
            "embedding_variant": r.embedding_variant or "default",
            "architecture_config": r.architecture_config or "default",
            "n_test": len(y),
        }

        brier = r.metrics.get("brier_score")
        if brier is None:
            try:
                brier = float(brier_score_loss(y, p))
            except Exception:
                brier = None
        ece = r.metrics.get("ece")
        if ece is None:
            ece = _ece_from_probs(y, p)
        cal_rows.append(
            {
                **base,
                "brier_score": brier,
                "ece": ece,
                "calibration_slope": _calibration_slope(y, p),
            }
        )

        for k in TOP_K_VALUES:
            tk = _topk_metrics(y, p, k)
            topk_rows.append({**base, "k": k, **tk})

    return pd.DataFrame(topk_rows), pd.DataFrame(cal_rows)


def predictions_to_combined(runs: list[RunRecord]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for r in runs:
        if r.predictions is None or r.predictions.empty:
            continue
        df = r.predictions.copy()
        df["experiment_type"] = r.experiment_type
        df["run_path"] = r.rel_path
        if "head_type" not in df.columns:
            df["head_type"] = r.head_type
        if "mit_fraction" not in df.columns:
            df["mit_fraction"] = r.mit_fraction
        if "imbalance_mode" not in df.columns:
            df["imbalance_mode"] = r.imbalance_mode
        df["embedding_variant"] = r.embedding_variant or "default"
        df["architecture_config"] = r.architecture_config or "default"
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def _f1_binary(y: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = y_pred.astype(np.int64)
    tp = int(np.sum((y_pred == 1) & (y == 1)))
    fp = int(np.sum((y_pred == 1) & (y == 0)))
    fn = int(np.sum((y_pred == 0) & (y == 1)))
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _mcnemar_pvalue(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, int, int, int, int]:
    """McNemar test on paired binary predictions (exact binomial if discordant small)."""
    pred_a = pred_a.astype(np.int64)
    pred_b = pred_b.astype(np.int64)
    # b: A correct B wrong, c: A wrong B correct — use sklearn-style discordant pairs
    n01 = int(np.sum((pred_a == y) & (pred_b != y)))  # A wins, B loses
    n10 = int(np.sum((pred_a != y) & (pred_b == y)))  # B wins, A loses
    n_disc = n01 + n10
    if n_disc == 0:
        return 1.0, n01, n10, n_disc, 0
    # Exact two-sided binomial test on discordant count
    k = min(n01, n10)
    p = 2 * stats.binom.cdf(k, n_disc, 0.5)
    return float(min(1.0, p)), n01, n10, n_disc, n_disc


def _paired_bootstrap_ci(
    deltas: np.ndarray,
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    d = deltas[~np.isnan(deltas)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    if len(d) == 1:
        return float(d[0]), float(d[0]), float(d[0])
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = len(d)
    for i in range(n_boot):
        samp = d[rng.integers(0, n, size=n)]
        boots[i] = float(np.mean(samp))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(np.mean(d)), lo, hi


def _permutation_test_pvalue(deltas: np.ndarray, *, n_perm: int = 5000, seed: int = 42) -> float:
    d = deltas[~np.isnan(deltas)]
    if len(d) == 0:
        return np.nan
    obs = float(np.mean(d))
    if len(d) == 1:
        return 1.0
    rng = np.random.default_rng(seed)
    n = len(d)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_mean = float(np.mean(d * signs))
        if abs(perm_mean) >= abs(obs):
            count += 1
    return float((count + 1) / (n_perm + 1))


def _align_predictions(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    key = "index"
    if key not in df_a.columns or key not in df_b.columns:
        if "compound_id" in df_a.columns and "compound_id" in df_b.columns:
            key = "compound_id"
        else:
            return None
    merged = df_a[[key, "true_label", "probability"]].merge(
        df_b[[key, "probability"]],
        on=key,
        suffixes=("_a", "_b"),
        how="inner",
    )
    if merged.empty:
        return None
    y = merged["true_label"].to_numpy(dtype=np.int64)
    pa = merged["probability_a"].to_numpy(dtype=float)
    pb = merged["probability_b"].to_numpy(dtype=float)
    return y, pa, pb, merged[key].to_numpy()


def build_pairing_index(runs: list[RunRecord]) -> dict[tuple[Any, ...], RunRecord]:
    """Index runs for statistical pairing (main + parameter_matched, default settings)."""
    idx: dict[tuple[Any, ...], RunRecord] = {}
    for r in runs:
        if r.experiment_type not in ("main", "parameter_matched"):
            continue
        if r.experiment_type == "main" and r.imbalance_mode not in ("original", ""):
            continue
        if r.fold < 0 or r.seed < 0:
            continue
        key = (float(r.mit_fraction), int(r.fold), int(r.seed), str(r.head_type))
        idx[key] = r
    return idx


def run_paired_statistical_tests(
    runs: list[RunRecord],
    logger: logging.Logger,
    *,
    n_boot: int = 5000,
    n_perm: int = 5000,
) -> pd.DataFrame:
    index = build_pairing_index(runs)
    # Pre-load predictions for indexed runs
    for key, r in index.items():
        if r.predictions is None:
            pred_path = r.run_dir / "predictions_test.csv"
            if pred_path.is_file():
                try:
                    r.predictions = pd.read_csv(pred_path)
                except Exception as exc:
                    logger.warning("Could not load %s: %s", pred_path, exc)

    mit_fractions = sorted({k[0] for k in index})
    rows: list[dict[str, Any]] = []

    for mit_frac in mit_fractions:
        for head_a, head_b in PAIRED_COMPARISONS:
            pr_deltas: list[float] = []
            f1_deltas: list[float] = []
            mcnemar_stats: list[tuple[int, int]] = []
            n_pairs = 0
            n_samples = 0

            for fold in range(10):
                for seed in (42, 43, 44, 45, 46):
                    ka = (mit_frac, fold, seed, head_a)
                    kb = (mit_frac, fold, seed, head_b)
                    if ka not in index or kb not in index:
                        continue
                    ra, rb = index[ka], index[kb]
                    if ra.predictions is None or rb.predictions is None:
                        continue
                    aligned = _align_predictions(ra.predictions, rb.predictions)
                    if aligned is None:
                        logger.warning(
                            "Could not align predictions for %s vs %s fold=%s seed=%s",
                            head_a,
                            head_b,
                            fold,
                            seed,
                        )
                        continue
                    y, pa, pb, _keys = aligned
                    pr_deltas.append(_pr_auc(y, pa) - _pr_auc(y, pb))

                    key_col = "index" if "index" in ra.predictions.columns else "compound_id"
                    ma = ra.predictions.set_index(key_col)
                    mb = rb.predictions.set_index(key_col)
                    common = ma.index.intersection(mb.index)
                    ya = ma.loc[common, "true_label"].to_numpy(dtype=np.int64)
                    lab_col = (
                        "predicted_label_at_val_f1_threshold"
                        if "predicted_label_at_val_f1_threshold" in ma.columns
                        else "predicted_label_at_0p5"
                    )
                    pa_lab = ma.loc[common, lab_col].to_numpy(dtype=np.int64)
                    pb_lab = mb.loc[common, lab_col].to_numpy(dtype=np.int64)
                    f1_deltas.append(_f1_binary(ya, pa_lab) - _f1_binary(ya, pb_lab))

                    _p_mcn, b01, b10, n_disc, _ = _mcnemar_pvalue(ya, pa_lab, pb_lab)
                    mcnemar_stats.append((b01, b10))
                    n_pairs += 1
                    n_samples += len(ya)

            if n_pairs == 0:
                logger.warning("No pairs for %s vs %s at mit_fraction=%s", head_a, head_b, mit_frac)
                continue

            pr_arr = np.array(pr_deltas, dtype=float)
            f1_arr = np.array(f1_deltas, dtype=float)
            pr_mean, pr_lo, pr_hi = _paired_bootstrap_ci(pr_arr, n_boot=n_boot)
            f1_mean, f1_lo, f1_hi = _paired_bootstrap_ci(f1_arr, n_boot=n_boot)
            perm_p = _permutation_test_pvalue(pr_arr, n_perm=n_perm)

            # Pooled McNemar across all aligned samples
            if mcnemar_stats:
                b01_tot = sum(x[0] for x in mcnemar_stats)
                b10_tot = sum(x[1] for x in mcnemar_stats)
                n_disc = b01_tot + b10_tot
                if n_disc > 0:
                    k = min(b01_tot, b10_tot)
                    mcnemar_p = float(min(1.0, 2 * stats.binom.cdf(k, n_disc, 0.5)))
                else:
                    mcnemar_p = 1.0
            else:
                mcnemar_p = np.nan
                b01_tot = b10_tot = 0

            rows.append(
                {
                    "mit_fraction": mit_frac,
                    "head_a": head_a,
                    "head_b": head_b,
                    "comparison": f"{head_a}_vs_{head_b}",
                    "delta_definition": f"{head_a}_minus_{head_b}",
                    "n_fold_seed_pairs": n_pairs,
                    "n_aligned_samples": n_samples,
                    "delta_pr_auc_mean": pr_mean,
                    "delta_pr_auc_bootstrap_ci_low": pr_lo,
                    "delta_pr_auc_bootstrap_ci_high": pr_hi,
                    "delta_f1_mean": f1_mean,
                    "delta_f1_bootstrap_ci_low": f1_lo,
                    "delta_f1_bootstrap_ci_high": f1_hi,
                    "permutation_pvalue_delta_pr_auc": perm_p,
                    "mcnemar_pvalue_tuned_threshold": mcnemar_p,
                    "mcnemar_n_discordant": b01_tot + b10_tot,
                    "mcnemar_b01": b01_tot,
                    "mcnemar_b10": b10_tot,
                }
            )

    return pd.DataFrame(rows)


def collect_all_runs(
    experiment_dirs: dict[str, str],
    logger: logging.Logger,
    *,
    load_predictions: bool = True,
) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for exp_type, rel_dir in experiment_dirs.items():
        exp_root = (ROOT / rel_dir).resolve()
        fold_dirs = discover_fold_dirs(exp_root, logger)
        logger.info("%s: found %d fold runs under %s", exp_type, len(fold_dirs), exp_root)
        for fold_dir in fold_dirs:
            rec = load_run(exp_type, fold_dir, exp_root, logger, load_predictions=load_predictions)
            if rec is not None:
                if load_predictions and rec.predictions is None:
                    pred_path = fold_dir / "predictions_test.csv"
                    if pred_path.is_file():
                        try:
                            rec.predictions = pd.read_csv(pred_path)
                        except Exception as exc:
                            logger.warning("Failed predictions %s: %s", pred_path, exc)
                runs.append(rec)
    return runs


def print_experiment_summary(runs: list[RunRecord], logger: logging.Logger) -> None:
    logger.info("=== Experiment summary ===")
    df = pd.DataFrame(
        [
            {
                "experiment_type": r.experiment_type,
                "head_type": r.head_type,
                "mit_fraction": r.mit_fraction,
            }
            for r in runs
        ]
    )
    if df.empty:
        logger.info("No runs collected.")
        return
    summary = (
        df.groupby("experiment_type")
        .agg(n_runs=("head_type", "count"), heads=("head_type", lambda s: sorted(set(s))))
        .reset_index()
    )
    for _, row in summary.iterrows():
        logger.info(
            "  %s: %d runs, heads=%s",
            row["experiment_type"],
            row["n_runs"],
            row["heads"],
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate reviewer-eval results and run paired tests.")
    ap.add_argument(
        "--root",
        type=str,
        default=DEFAULT_WORK_ROOT,
        help="Experiment root containing main/, parameter_matched/, ablations/, etc. "
        f"(default: {DEFAULT_WORK_ROOT})",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Aggregation output dir (default: <root>/aggregated).",
    )
    ap.add_argument("--n_bootstrap", type=int, default=5000)
    ap.add_argument("--n_permutation", type=int, default=5000)
    ap.add_argument("--skip_predictions", action="store_true", help="Skip loading predictions (faster).")
    args = ap.parse_args()

    root = args.root.strip().rstrip("/")
    exp_dirs = {k: f"{root}/{k}" for k in EXPERIMENT_DIRS}
    out_rel = args.output_dir or f"{root}/aggregated"
    out_dir = (ROOT / out_rel).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "aggregation_warnings.log"

    logger = logging.getLogger("aggregate_reviewer_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    load_preds = not args.skip_predictions
    runs = collect_all_runs(exp_dirs, logger, load_predictions=load_preds)
    print_experiment_summary(runs, logger)

    if not runs:
        logger.error("No runs found; exiting.")
        return 1

    metrics_long = runs_to_metrics_long(runs)
    metrics_long.to_csv(out_dir / "all_metrics_long.csv", index=False)
    logger.info("Wrote all_metrics_long.csv (%d rows)", len(metrics_long))

    summary = build_metrics_summary(metrics_long)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    logger.info("Wrote metrics_summary.csv (%d rows)", len(summary))

    if load_preds:
        combined = predictions_to_combined(runs)
        combined.to_csv(out_dir / "all_predictions_test.csv", index=False)
        logger.info("Wrote all_predictions_test.csv (%d rows)", len(combined))

        topk_df, cal_df = compute_prediction_derived_tables(runs, logger)
        topk_df.to_csv(out_dir / "topk_metrics.csv", index=False)
        cal_df.to_csv(out_dir / "calibration_metrics.csv", index=False)
        logger.info("Wrote topk_metrics.csv (%d rows)", len(topk_df))
        logger.info("Wrote calibration_metrics.csv (%d rows)", len(cal_df))

        paired = run_paired_statistical_tests(runs, logger, n_boot=args.n_bootstrap, n_perm=args.n_permutation)
        paired.to_csv(out_dir / "paired_statistical_tests.csv", index=False)
        logger.info("Wrote paired_statistical_tests.csv (%d rows)", len(paired))
    else:
        logger.info("Skipped prediction-derived outputs (--skip_predictions)")

    meta = {
        "n_runs": len(runs),
        "experiments": {k: str((ROOT / v).resolve()) for k, v in exp_dirs.items()},
        "output_dir": str(out_dir.relative_to(ROOT)),
        "root": root,
    }
    with open(out_dir / "aggregation_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Done. Outputs in %s", out_dir.relative_to(ROOT))
    logger.info("Warnings log: %s", log_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
