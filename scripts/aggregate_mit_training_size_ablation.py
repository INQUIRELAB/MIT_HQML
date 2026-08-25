#!/usr/bin/env python3
"""
Aggregate MIT training-size ablation: discover low-data runs under
``{ablation_root}/{linear,mlp,quantum,reup}/mit_fraction_*/summary.json``,
compare to frozen CGCNN baseline (same test folds for all fractions).

Default training mode matches ``train_heads_on_embeddings.py`` **without**
``--match_negatives_to_positives``: subsample only **positive** MIT training
examples; **all negatives** kept → **class imbalance** in training grows as
fraction shrinks.

Writes:
  - results/mit_training_size_ablation/ablation_table.json
  - results/mit_training_size_ablation/ablation_report.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import compare_heads_vs_cgcnn as ch  # noqa: E402


def _parse_fraction(name: str) -> float | None:
    m = re.match(r"mit_fraction_(.+)$", name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _mean_from_agg_across_seeds(summary: dict[str, Any], key: str) -> float | None:
    block = summary.get("aggregate_across_seeds") or {}
    m = block.get(key)
    if not m or not isinstance(m, dict):
        return None
    v = m.get("mean")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


def _mean_from_cgcnn_agg(agg: dict[str, Any] | None, key: str) -> float | None:
    if not agg:
        return None
    m = agg.get(key)
    if not m or not isinstance(m, dict):
        return None
    v = m.get("mean")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


def _winner(
    scores: dict[str, float | None], higher_is_better: bool = True
) -> tuple[str | None, float | None]:
    items = [(k, v) for k, v in scores.items() if v is not None and not np.isnan(v)]
    if not items:
        return None, None
    items.sort(key=lambda t: t[1], reverse=higher_is_better)
    return items[0][0], items[0][1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding_path", type=str, default="embeddings/cgcnn_binary_embeddings.parquet")
    ap.add_argument("--splits_dir", type=str, default="splits/grouped_5fold")
    ap.add_argument(
        "--ablation_root",
        type=str,
        default="results/mit_training_size_ablation",
        help="Root containing head_type subdirs with mit_fraction_* summaries.",
    )
    ap.add_argument("--out_json", type=str, default="results/mit_training_size_ablation/ablation_table.json")
    ap.add_argument("--out_md", type=str, default="results/mit_training_size_ablation/ablation_report.md")
    args = ap.parse_args()

    import pandas as pd

    emb_path = (ROOT / args.embedding_path).resolve()
    splits_dir = (ROOT / args.splits_dir).resolve()
    ablation_root = (ROOT / args.ablation_root).resolve()
    out_json = (ROOT / args.out_json).resolve()
    out_md = (ROOT / args.out_md).resolve()

    df = pd.read_parquet(emb_path)
    emb_cols = ch._emb_columns(df)
    valid = ch._valid_embedding_mask(df, emb_cols)
    _, cgcnn_agg = ch.evaluate_cgcnn_baseline(df, emb_cols, valid, splits_dir)

    heads = ["linear", "mlp", "quantum", "reup"]
    metric_keys = ["pr_auc", "roc_auc", "balanced_accuracy", "f1", "recall", "precision"]

    # Collect all fractions present under any head
    fracs: set[float] = set()
    for h in heads:
        d = ablation_root / h
        if not d.is_dir():
            continue
        for sub in d.iterdir():
            if not sub.is_dir():
                continue
            pf = _parse_fraction(sub.name)
            if pf is not None and (sub / "summary.json").is_file():
                fracs.add(pf)

    rows: list[dict[str, Any]] = []
    for frac in sorted(fracs):
        row: dict[str, Any] = {"mit_fraction": frac}
        model_metrics: dict[str, dict[str, float | None]] = {"cgcnn_only": {}}
        for mk in metric_keys:
            model_metrics["cgcnn_only"][mk] = _mean_from_cgcnn_agg(cgcnn_agg, mk)

        for h in heads:
            # Resolve directory by scanning for matching float (folder tag may differ)
            head_dir = ablation_root / h
            summary_path: Path | None = None
            if head_dir.is_dir():
                for sub in sorted(head_dir.iterdir()):
                    if not sub.is_dir():
                        continue
                    pf = _parse_fraction(sub.name)
                    if pf is None:
                        continue
                    if abs(pf - frac) < 1e-5:
                        cand = sub / "summary.json"
                        if cand.is_file():
                            summary_path = cand
                            break
            block: dict[str, float | None] = {}
            if summary_path:
                with open(summary_path) as f:
                    summ = json.load(f)
                for mk in metric_keys:
                    block[mk] = _mean_from_agg_across_seeds(summ, mk)
            else:
                for mk in metric_keys:
                    block[mk] = None
            model_metrics[f"head_{h}"] = block

        row["models"] = model_metrics

        winners: dict[str, Any] = {}
        for mk in metric_keys:
            sc = {name: vals[mk] for name, vals in model_metrics.items()}
            w_model, w_val = _winner(sc, higher_is_better=True)
            winners[mk] = {"best": w_model, "value": w_val}
        row["winners"] = winners
        rows.append(row)

    payload = {
        "description": (
            "Rows: mit_fraction of training positives used per fold (negatives unchanged unless "
            "train_heads was run with --match_negatives_to_positives). "
            "Head metrics = mean of cross-fold means across seeds (aggregate_across_seeds). "
            "cgcnn_only is identical across fractions."
        ),
        "embedding_path": str(emb_path.relative_to(ROOT)),
        "splits_dir": str(splits_dir.relative_to(ROOT)),
        "ablation_root": str(ablation_root.relative_to(ROOT)),
        "cgcnn_only_aggregate": {k: cgcnn_agg.get(k) for k in metric_keys},
        "rows": rows,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    # Markdown report (imbalance-focused)
    lines = [
        "# MIT training-size ablation (real binary dataset)",
        "",
        "Training: fraction of **MIT (positive) labels** in each outer-fold **train** set; "
        "**non-MIT negatives** are kept at full train size unless you used "
        "`--match_negatives_to_positives`. Test folds unchanged.",
        "",
        f"- Embeddings: `{payload['embedding_path']}`",
        f"- Splits: `{payload['splits_dir']}`",
        f"- Runs: `{payload['ablation_root']}/{{linear,mlp,quantum,reup}}/mit_fraction_*/`",
        "",
        "**CGCNN-only** uses frozen classifier probabilities; same for every row.",
        "",
        "## Winners by `mit_fraction` (highest mean test metric)",
        "",
        "| mit_fraction | best PR-AUC | best ROC-AUC | best bal_acc | best F1 | best recall |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        w = r["winners"]
        lines.append(
            f"| {r['mit_fraction']:.6g} | "
            f"{w['pr_auc']['best'] or '—'} | "
            f"{w['roc_auc']['best'] or '—'} | "
            f"{w['balanced_accuracy']['best'] or '—'} | "
            f"{w['f1']['best'] or '—'} | "
            f"{w['recall']['best'] or '—'} |"
        )
    lines.extend(
        [
            "",
            "## PR-AUC vs training fraction (screening under imbalance)",
            "",
            "| mit_fraction | cgcnn_only | linear | mlp | quantum | reup |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        m = r["models"]
        fmt = lambda x: f"{x:.4f}" if x is not None else "—"
        lines.append(
            f"| {r['mit_fraction']:.6g} | "
            f"{fmt(m['cgcnn_only'].get('pr_auc'))} | "
            f"{fmt(m['head_linear'].get('pr_auc'))} | "
            f"{fmt(m['head_mlp'].get('pr_auc'))} | "
            f"{fmt(m['head_quantum'].get('pr_auc'))} | "
            f"{fmt(m['head_reup'].get('pr_auc'))} |"
        )
    lines.extend(
        [
            "",
            "## Balanced accuracy vs fraction",
            "",
            "| mit_fraction | cgcnn_only | linear | mlp | quantum | reup |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        m = r["models"]
        fmt = lambda x: f"{x:.4f}" if x is not None else "—"
        lines.append(
            f"| {r['mit_fraction']:.6g} | "
            f"{fmt(m['cgcnn_only'].get('balanced_accuracy'))} | "
            f"{fmt(m['head_linear'].get('balanced_accuracy'))} | "
            f"{fmt(m['head_mlp'].get('balanced_accuracy'))} | "
            f"{fmt(m['head_quantum'].get('balanced_accuracy'))} | "
            f"{fmt(m['head_reup'].get('balanced_accuracy'))} |"
        )

    lines.extend(["", f"Full JSON: `{out_json.relative_to(ROOT)}`", ""])
    out_md.write_text("\n".join(lines))

    print(f"Wrote {out_json.relative_to(ROOT)}")
    print(f"Wrote {out_md.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
