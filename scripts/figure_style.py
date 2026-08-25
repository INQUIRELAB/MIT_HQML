"""Shared matplotlib styling for manuscript figures.

LaTeX scales the whole PDF by ``display_width / figure_width``.  Source fonts
are scaled so printed text stays near TARGET sizes.  Multi-panel figures also
need enough inches per panel so labels do not overlap before scaling.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np

# Intended width when included in the manuscript (inches).
MANUSCRIPT_LINEWIDTH_IN = 6.5
MANUSCRIPT_NARROW_IN = 0.7 * MANUSCRIPT_LINEWIDTH_IN

# Target sizes after LaTeX scaling (approximate printed points).
TARGET = {
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
}

# Minimum source size per subplot cell (inches) to avoid in-figure overlap.
PANEL_W = 4.4
PANEL_H = 3.6


def _width_factor(fig_width_in: float, display_width_in: float) -> float:
    return max(1.0, fig_width_in / display_width_in)


def scaled_rcparams(
    fig_width_in: float = 7.0,
    display_width_in: float | None = None,
) -> dict[str, float | str]:
    if display_width_in is None:
        display_width_in = MANUSCRIPT_LINEWIDTH_IN
    factor = _width_factor(fig_width_in, display_width_in)
    params: dict[str, float | str] = {
        key: round(value * factor, 1) for key, value in TARGET.items()
    }
    params.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
        }
    )
    return params


def grid_figsize(n_cols: int, n_rows: int, *, header: float = 0.8) -> tuple[float, float]:
    """Return (width, height) with enough room per panel."""
    return PANEL_W * n_cols, PANEL_H * n_rows + header


def apply_figure_style(
    fig_width_in: float = 7.0,
    display_width_in: float | None = None,
) -> None:
    plt.rcParams.update(scaled_rcparams(fig_width_in, display_width_in))


def fs(size: float) -> float:
    """Scale an explicit annotation size relative to the active body font."""
    return round(size * (plt.rcParams["font.size"] / TARGET["font.size"]), 1)


def panel_title_size() -> float:
    return plt.rcParams["axes.titlesize"]


def style_colorbar(cbar, label: str | None = None, *, labelpad: float = 4.0) -> None:
    tick = plt.rcParams["xtick.labelsize"]
    cbar.ax.tick_params(labelsize=tick)
    if label is not None:
        cbar.set_label(label, fontsize=plt.rcParams["axes.labelsize"], labelpad=labelpad)


def style_axes(ax) -> None:
    tick = plt.rcParams["xtick.labelsize"]
    ax.tick_params(axis="both", labelsize=tick)
    ax.xaxis.label.set_size(plt.rcParams["axes.labelsize"])
    ax.yaxis.label.set_size(plt.rcParams["axes.labelsize"])
    ax.title.set_size(panel_title_size())


def hide_inner_labels(axes, n_rows: int, n_cols: int, n_panels: int) -> None:
    """Keep axis labels only on the outer edges of a panel grid."""
    hide_inner_xlabels(axes, n_rows, n_cols, n_panels)
    hide_inner_ylabels(axes, n_rows, n_cols, n_panels)


def hide_inner_xlabels(axes, n_rows: int, n_cols: int, n_panels: int) -> None:
    """Remove x-axis labels from all but the bottom row."""
    axes_flat = np.atleast_1d(axes).flatten()
    for i in range(n_panels):
        row, _col = divmod(i, n_cols)
        if row < n_rows - 1:
            axes_flat[i].set_xlabel("")


def hide_inner_ylabels(axes, n_rows: int, n_cols: int, n_panels: int) -> None:
    """Remove y-axis labels from all but the left column."""
    axes_flat = np.atleast_1d(axes).flatten()
    for i in range(n_panels):
        _row, col = divmod(i, n_cols)
        if col > 0:
            axes_flat[i].set_ylabel("")


def add_shared_xlabel(fig, label: str, *, y: float = 0.02) -> None:
    fig.supxlabel(label, y=y, fontsize=plt.rcParams["axes.labelsize"])


def add_shared_ylabel(fig, label: str, *, x: float = 0.02) -> None:
    fig.supylabel(label, x=x, fontsize=plt.rcParams["axes.labelsize"])


def panel_tag(ax, text: str, *, x: float = 0.03, y: float = 0.97, ha: str = "left") -> None:
    """Small in-panel identifier that stays off the data."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        va="top",
        ha=ha,
        fontsize=plt.rcParams["legend.fontsize"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.92, edgecolor="none"),
    )


def _legend_entries(ax):
    handles, labels = ax.get_legend_handles_labels()
    return [
        (h, lab)
        for h, lab in zip(handles, labels, strict=True)
        if lab and not lab.startswith("_")
    ]


def add_figure_legend(
    fig,
    axes,
    *,
    ncol: int = 3,
    y: float = 0.02,
) -> None:
    axes_flat = np.atleast_1d(axes).flatten()
    for ax in axes_flat:
        entries = _legend_entries(ax)
        if entries:
            handles, labels = zip(*entries, strict=True)
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=ncol,
                bbox_to_anchor=(0.5, y),
                frameon=False,
            )
            return


def add_figure_legend_top(
    fig,
    axes,
    *,
    ncol: int = 3,
    y: float = 0.99,
) -> None:
    axes_flat = np.atleast_1d(axes).flatten()
    for ax in axes_flat:
        entries = _legend_entries(ax)
        if entries:
            handles, labels = zip(*entries, strict=True)
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=ncol,
                bbox_to_anchor=(0.5, y),
                frameon=False,
            )
            return


def add_row_labels(
    fig,
    axes,
    labels: list[str],
    *,
    left: float = 0.04,
) -> None:
    """Place row titles in the left margin instead of overlapping y tick labels."""
    axes_arr = np.atleast_1d(axes)
    if axes_arr.ndim == 1:
        refs = axes_arr
    else:
        refs = axes_arr[:, 0]
    for ax, label in zip(refs, labels, strict=True):
        pos = ax.get_position()
        fig.text(
            left,
            pos.y0 + pos.height / 2,
            label,
            ha="right",
            va="center",
            rotation=90,
            fontsize=plt.rcParams["axes.labelsize"],
        )
        ax.set_ylabel("")


def add_column_labels(
    fig,
    axes,
    labels: list[str],
    *,
    top: float = 0.98,
) -> None:
    axes_arr = np.atleast_2d(axes)
    for j, label in enumerate(labels):
        pos = axes_arr[0, j].get_position()
        fig.text(
            pos.x0 + pos.width / 2,
            top,
            label,
            ha="center",
            va="bottom",
            fontsize=plt.rcParams["axes.labelsize"],
        )


def finalize_figure(
    fig,
    *,
    bottom: float = 0.08,
    top: float = 0.98,
    left: float = 0.08,
    right: float = 0.98,
    hspace: float = 0.48,
    wspace: float = 0.34,
    legend_below: bool = False,
) -> None:
    if legend_below:
        bottom = max(bottom, 0.16)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=hspace, wspace=wspace)


def save_manuscript_figure(fig, base_path: str | Path, *, dpi: int = 300) -> None:
    """Save PDF/PNG without tight bbox cropping (preserves margins)."""
    path = Path(base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=dpi, bbox_inches=None, pad_inches=0.02)


@contextmanager
def manuscript_figure(
    fig_width_in: float,
    display_width_in: float | None = None,
) -> Iterator[None]:
    with plt.rc_context(scaled_rcparams(fig_width_in, display_width_in)):
        yield
