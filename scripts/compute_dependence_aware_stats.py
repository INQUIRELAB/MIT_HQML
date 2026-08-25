#!/usr/bin/env python3
"""
Paired and dependence-aware statistics for reviewer-eval runs.

Extends the fold×seed IID bootstrap with:
  - fold-blocked (cluster) bootstrap of mean metrics / paired deltas
  - seed-blocked bootstrap within folds
  - paired permutation tests that shuffle signs within fold clusters

Writes under ``<root>/aggregated/`` by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from aggregate_reviewer_eval import (  # noqa: E402
    DEFAULT_WORK_ROOT,
    PAIRED_COMPARISONS,
    _align_predictions,
    _pr_auc,
    build_pairing_index,
    collect_all_runs,
)


DEFAULT_COMPARISONS = PAIRED_COMPARISONS + [
    ("reup", "proj_bottleneck_mlp"),
    ("reup", "latent_bottleneck_mlp"),
    ("quantum", "proj_bottleneck_mlp"),
    ("quantum", "latent_bottleneck_mlp"),
]

# De-duplicate while preserving order (PAIRED_COMPARISONS may already include some rows).
_seen: set[tuple[str, str]] = set()
_deduped: list[tuple[str, str]] = []
for pair in DEFAULT_COMPARISONS:
    if pair in _seen:
        continue
    _seen.add(pair)
    _deduped.append(pair)
DEFAULT_COMPARISONS = _deduped


def _fold_seed_metric_table(runs, head: str, mit_frac: float, metric: str = "pr_auc") -> pd.DataFrame:
    rows = []
    for r in runs:
        if r.experiment_type not in ("main", "parameter_matched"):
            continue
        if str(r.head_type) != head:
            continue
        if abs(float(r.mit_fraction) - float(mit_frac)) > 1e-9:
            continue
        if r.fold < 0 or r.seed < 0:
            continue
        val = r.metrics.get(metric)
        if val is None:
            continue
        rows.append({"fold": int(r.fold), "seed": int(r.seed), "value": float(val), "head": head})
    return pd.DataFrame(rows)


def _cluster_bootstrap_mean(
    values_by_cluster: dict[Any, list[float]],
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> tuple[float, float, float]:
    clusters = sorted(values_by_cluster)
    if not clusters:
        return float("nan"), float("nan"), float("nan")
    cluster_means = np.array(
        [float(np.nanmean(values_by_cluster[c])) for c in clusters],
        dtype=float,
    )
    obs = float(np.nanmean(cluster_means))
    if len(cluster_means) == 1:
        return obs, obs, obs
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    k = len(cluster_means)
    for i in range(n_boot):
        samp = cluster_means[rng.integers(0, k, size=k)]
        boots[i] = float(np.mean(samp))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return obs, lo, hi


def _iid_bootstrap_mean(values: np.ndarray, *, n_boot: int, alpha: float, seed: int) -> tuple[float, float, float]:
    d = values[~np.isnan(values)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    obs = float(np.mean(d))
    if len(d) == 1:
        return obs, obs, obs
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = len(d)
    for i in range(n_boot):
        boots[i] = float(np.mean(d[rng.integers(0, n, size=n)]))
    return obs, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def _paired_deltas_by_fold(
    index,
    head_a: str,
    head_b: str,
    mit_frac: float,
) -> dict[int, list[float]]:
    by_fold: dict[int, list[float]] = defaultdict(list)
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
                continue
            y, pa, pb, _ = aligned
            by_fold[fold].append(_pr_auc(y, pa) - _pr_auc(y, pb))
    return by_fold


def _cluster_sign_permutation_pvalue(
    values_by_cluster: dict[Any, list[float]],
    *,
    n_perm: int,
    seed: int,
) -> float:
    clusters = sorted(values_by_cluster)
    if not clusters:
        return float("nan")
    cluster_means = np.array(
        [float(np.nanmean(values_by_cluster[c])) for c in clusters],
        dtype=float,
    )
    obs = float(np.mean(cluster_means))
    if len(cluster_means) == 1:
        return 1.0
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(cluster_means))
        if abs(float(np.mean(cluster_means * signs))) >= abs(obs):
            count += 1
    return float((count + 1) / (n_perm + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description="Dependence-aware paired statistics.")
    ap.add_argument("--root", type=str, default=DEFAULT_WORK_ROOT)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--n_bootstrap", type=int, default=5000)
    ap.add_argument("--n_permutation", type=int, default=5000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--metric", type=str, default="pr_auc")
    args = ap.parse_args()

    root = args.root.strip().rstrip("/")
    exp_dirs = {k: f"{root}/{k}" for k in ("main", "parameter_matched")}
    out_rel = args.output_dir or f"{root}/aggregated"
    out_dir = (ROOT / out_rel).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("dependence_aware_stats")
    fh = logging.FileHandler(out_dir / "dependence_aware_stats.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)

    runs = collect_all_runs(exp_dirs, logger, load_predictions=True)
    index = build_pairing_index(runs)
    for r in index.values():
        if r.predictions is None:
            pred_path = r.run_dir / "predictions_test.csv"
            if pred_path.is_file():
                try:
                    r.predictions = pd.read_csv(pred_path)
                except Exception as exc:
                    logger.warning("Could not load %s: %s", pred_path, exc)

    mit_fractions = sorted({float(r.mit_fraction) for r in runs if r.mit_fraction == r.mit_fraction})
    heads = sorted({str(r.head_type) for r in runs if r.experiment_type in ("main", "parameter_matched")})

    marginal_rows: list[dict[str, Any]] = []
    for mit_frac in mit_fractions:
        for head in heads:
            tab = _fold_seed_metric_table(runs, head, mit_frac, args.metric)
            if tab.empty:
                continue
            vals = tab["value"].to_numpy(dtype=float)
            iid_mu, iid_lo, iid_hi = _iid_bootstrap_mean(
                vals, n_boot=args.n_bootstrap, alpha=args.alpha, seed=42
            )
            by_fold = {
                int(f): g["value"].tolist() for f, g in tab.groupby("fold")
            }
            fold_mu, fold_lo, fold_hi = _cluster_bootstrap_mean(
                by_fold, n_boot=args.n_bootstrap, alpha=args.alpha, seed=43
            )
            by_seed = {
                int(s): g["value"].tolist() for s, g in tab.groupby("seed")
            }
            seed_mu, seed_lo, seed_hi = _cluster_bootstrap_mean(
                by_seed, n_boot=args.n_bootstrap, alpha=args.alpha, seed=44
            )
            marginal_rows.append(
                {
                    "mit_fraction": mit_frac,
                    "head_type": head,
                    "metric": args.metric,
                    "n_fold_seed": int(len(tab)),
                    "n_folds": int(tab["fold"].nunique()),
                    "mean": iid_mu,
                    "iid_bootstrap_ci_low": iid_lo,
                    "iid_bootstrap_ci_high": iid_hi,
                    "fold_blocked_mean": fold_mu,
                    "fold_blocked_ci_low": fold_lo,
                    "fold_blocked_ci_high": fold_hi,
                    "seed_blocked_ci_low": seed_lo,
                    "seed_blocked_ci_high": seed_hi,
                }
            )

    paired_rows: list[dict[str, Any]] = []
    for mit_frac in mit_fractions:
        for head_a, head_b in DEFAULT_COMPARISONS:
            by_fold = _paired_deltas_by_fold(index, head_a, head_b, mit_frac)
            if not by_fold:
                continue
            flat = np.array([d for ds in by_fold.values() for d in ds], dtype=float)
            iid_mu, iid_lo, iid_hi = _iid_bootstrap_mean(
                flat, n_boot=args.n_bootstrap, alpha=args.alpha, seed=45
            )
            fold_mu, fold_lo, fold_hi = _cluster_bootstrap_mean(
                by_fold, n_boot=args.n_bootstrap, alpha=args.alpha, seed=46
            )
            perm_p_iid = float("nan")
            if len(flat):
                rng = np.random.default_rng(47)
                obs = float(np.mean(flat))
                count = 0
                for _ in range(args.n_permutation):
                    signs = rng.choice([-1.0, 1.0], size=len(flat))
                    if abs(float(np.mean(flat * signs))) >= abs(obs):
                        count += 1
                perm_p_iid = float((count + 1) / (args.n_permutation + 1))
            perm_p_fold = _cluster_sign_permutation_pvalue(
                by_fold, n_perm=args.n_permutation, seed=48
            )
            paired_rows.append(
                {
                    "mit_fraction": mit_frac,
                    "head_a": head_a,
                    "head_b": head_b,
                    "comparison": f"{head_a}_vs_{head_b}",
                    "delta_definition": f"{head_a}_minus_{head_b}",
                    "metric": args.metric,
                    "n_fold_seed_pairs": int(len(flat)),
                    "n_folds": int(len(by_fold)),
                    "delta_mean": iid_mu,
                    "iid_bootstrap_ci_low": iid_lo,
                    "iid_bootstrap_ci_high": iid_hi,
                    "fold_blocked_delta_mean": fold_mu,
                    "fold_blocked_ci_low": fold_lo,
                    "fold_blocked_ci_high": fold_hi,
                    "permutation_pvalue_iid": perm_p_iid,
                    "permutation_pvalue_fold_blocked": perm_p_fold,
                }
            )

    marg_df = pd.DataFrame(marginal_rows)
    pair_df = pd.DataFrame(paired_rows)
    marg_path = out_dir / "dependence_aware_marginal.csv"
    pair_path = out_dir / "dependence_aware_paired.csv"
    meta_path = out_dir / "dependence_aware_meta.json"
    marg_df.to_csv(marg_path, index=False)
    pair_df.to_csv(pair_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "root": root,
                "n_bootstrap": args.n_bootstrap,
                "n_permutation": args.n_permutation,
                "alpha": args.alpha,
                "metric": args.metric,
                "method": {
                    "iid": "Resample fold×seed units with replacement.",
                    "fold_blocked": "Average seeds within fold, then resample folds.",
                    "seed_blocked": "Average folds within seed, then resample seeds.",
                    "paired_fold_blocked": "Paired deltas averaged within fold, then fold cluster bootstrap / sign-flip.",
                },
                "n_marginal_rows": int(len(marg_df)),
                "n_paired_rows": int(len(pair_df)),
            },
            indent=2,
        )
    )
    logger.info("Wrote %s (%d rows)", marg_path, len(marg_df))
    logger.info("Wrote %s (%d rows)", pair_path, len(pair_df))
    print(f"Wrote {marg_path.relative_to(ROOT)} ({len(marg_df)} rows)")
    print(f"Wrote {pair_path.relative_to(ROOT)} ({len(pair_df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
