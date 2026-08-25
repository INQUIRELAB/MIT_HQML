#!/usr/bin/env python3
"""
Build publication figures from JSON results.

Defaults:
  --ablation_table: results/mit_training_size_ablation/ablation_table.json
  --output_dir: figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from figure_style import (  # noqa: E402
    MANUSCRIPT_LINEWIDTH_IN,
    apply_figure_style,
    finalize_figure,
    fs,
    manuscript_figure,
)

apply_figure_style(6.5, MANUSCRIPT_LINEWIDTH_IN)

MODEL_STYLES = {
    "cgcnn_only": {"label": "CGCNN (end-to-end)", "color": "#1f77b4", "ls": "-"},
    "head_linear": {"label": "Linear head", "color": "#ff7f0e", "ls": "--"},
    "head_mlp": {"label": "MLP head", "color": "#2ca02c", "ls": "--"},
    "head_quantum": {"label": "VQC head (angle embed.)", "color": "#9467bd", "ls": "-"},
    "head_reup": {"label": "REUP head", "color": "#d62728", "ls": "-"},
}


def load_ablation(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing ablation table: {path}")
    with path.open() as f:
        return json.load(f)


def fig_metric_vs_fraction(rows: list, metric: str, out_path: Path, ylabel: str, title: str) -> None:
    fracs = [r["mit_fraction"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for key, sty in MODEL_STYLES.items():
        ys = [r["models"][key][metric] for r in rows]
        ax.plot(fracs, ys, marker="o", label=sty["label"], color=sty["color"], ls=sty["ls"], lw=1.5, ms=5)
    ax.set_xlabel(r"MIT training fraction $\rho$ (positive subsample)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_xticks(fracs)
    ax.set_xticklabels([f"{x:g}" for x in fracs])
    ax.grid(True, which="both", alpha=0.35, ls=":")
    ax.set_ylim(0.0, 1.02)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))
    finalize_figure(fig, bottom=0.28, left=0.11, right=0.98, top=0.95)
    fig.savefig(out_path)
    plt.close(fig)


def fig_bar_full_data(row_rho1: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    m = row_rho1["models"]
    keys = ["cgcnn_only", "head_linear", "head_mlp", "head_quantum", "head_reup"]
    labels = [MODEL_STYLES[k]["label"] for k in keys]
    colors = [MODEL_STYLES[k]["color"] for k in keys]
    pr = [m[k]["pr_auc"] for k in keys]
    roc = [m[k]["roc_auc"] for k in keys]
    ba = [m[k]["balanced_accuracy"] for k in keys]

    x = range(len(keys))
    w = 0.25
    ax.bar([i - w for i in x], pr, w, label="PR-AUC", color="#3366cc")
    ax.bar(list(x), roc, w, label="ROC-AUC", color="#dc3912")
    ax.bar([i + w for i in x], ba, w, label="Bal. Acc.", color="#ff9900")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Metric score")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.35, ls=":")
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    finalize_figure(fig, bottom=0.30, left=0.10, right=0.98, top=0.95)
    fig.savefig(out_path)
    plt.close(fig)


def fig_reup_conceptual(out_path: Path) -> None:
    """Block diagram: classical front-end + re-upload layers."""
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    def box(x0, y0, w, h, text, fc):
        rect = mpatches.FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#333333",
            facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=fs(9), wrap=True)

    box(0.2, 1.0, 1.6, 1.0, "Frozen\nembedding", "#e8e8e8")
    box(2.1, 1.0, 1.5, 1.0, "Classical\nMLP", "#cfe2f3")
    box(3.9, 1.0, 2.4, 1.0, r"REUP block $\ell$: RY data\n+ RX/RY/RZ + CZ", "#f4cccc")
    box(6.6, 1.0, 1.5, 1.0, r"$\cdots$ layers", "#fce5cd")
    box(8.4, 1.0, 1.4, 1.0, r"$\langle Z_i\rangle$\n+ linear", "#d9ead3")

    for x1, x2 in [(1.8, 2.1), (3.6, 3.9), (6.3, 6.6), (8.1, 8.4)]:
        ax.annotate(
            "",
            xy=(x2, 1.5),
            xytext=(x1, 1.5),
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333333"),
        )
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures.")
    parser.add_argument(
        "--ablation_table",
        type=str,
        default="results/mit_training_size_ablation/ablation_table.json",
        help="Path to ablation_table.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Output directory for generated PDF figures.",
    )
    args = parser.parse_args()

    ablation_path = Path(args.ablation_table).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_ablation(ablation_path)
    rows = sorted(data["rows"], key=lambda r: r["mit_fraction"])
    with manuscript_figure(6.5, MANUSCRIPT_LINEWIDTH_IN):
        fig_metric_vs_fraction(
            rows,
            "pr_auc",
            out_dir / "fig_pr_auc_vs_mit_fraction.pdf",
            "Mean PR-AUC (5 seeds, 5 folds)",
            "MIT training-size ablation: precision–recall ranking quality",
        )
        fig_metric_vs_fraction(
            rows,
            "roc_auc",
            out_dir / "fig_roc_auc_vs_mit_fraction.pdf",
            "Mean ROC-AUC (5 seeds, 5 folds)",
            "MIT training-size ablation: class-separation (ROC-AUC)",
        )
        full_row = next(r for r in rows if abs(r["mit_fraction"] - 1.0) < 1e-9)
        fig_bar_full_data(full_row, out_dir / "fig_bar_rho1.pdf")
        fig_reup_conceptual(out_dir / "fig_reup_conceptual.pdf")
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
