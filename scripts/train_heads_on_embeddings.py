#!/usr/bin/env python3
"""
Train a single post-hoc head (linear / mlp / quantum / reup) on frozen CGCNN embeddings
using grouped 5-fold splits (no compound leakage).

Default: full training data, one seed, outputs under results/heads/.

Low-data mode (--mit_fraction): subsample training positives (optional balanced negatives),
5 seeds, outputs under results/heads_lowdata/{head_type}/mit_fraction_{value}/.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler

# Repo root: MIT_LoRA_Pipeline/
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluation.fold_utils import list_fold_indices  # noqa: E402
from evaluation.thresholds import tune_threshold_f1  # noqa: E402
from create_embedding_variants import apply_embedding_variant  # noqa: E402
from models.posthoc_heads import (  # noqa: E402
    NoisyBackendUnavailableError,
    LatentBottleneckMLPHead,
    LatentMLPMatchedHead,
    ProjBottleneckMLPHead,
    REUPHead,
    RandomFourierLogisticHead,
    build_head,
    count_parameters,
    count_trainable_parameters,
    default_fourier_dim_for_vqc_match,
    reference_quantum_trainable_count,
    reference_reup_trainable_count,
)


def _emb_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("emb_")]
    if not cols:
        raise ValueError("No emb_* columns in embeddings parquet.")
    return sorted(cols, key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x)


def _valid_embedding_mask(df: pd.DataFrame, emb_cols: list[str]) -> np.ndarray:
    return df[emb_cols].notna().all(axis=1).to_numpy()


def _filter_indices(indices: list[int], valid: np.ndarray) -> list[int]:
    return [i for i in indices if 0 <= i < len(valid) and valid[i]]


def _mit_fraction_dir_tag(mit_fraction: float) -> str:
    x = round(float(mit_fraction), 6)
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return f"mit_fraction_{s}"


def _resolve_splits_dir(args: argparse.Namespace) -> Path:
    """Prefer --split_dir when set; otherwise --splits_dir (legacy)."""
    rel = getattr(args, "split_dir", None) or args.splits_dir
    return (ROOT / rel).resolve()


def _parse_seeds_arg(seeds_str: str | None, seed: int, n_lowdata_seeds: int) -> list[int]:
    if seeds_str:
        out = [int(x.strip()) for x in seeds_str.split(",") if x.strip()]
        if not out:
            raise ValueError("--seeds must contain at least one integer.")
        return out
    return [seed + i for i in range(n_lowdata_seeds)]


def set_global_seed(seed: int) -> None:
    """Deterministic seeding for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _id_column(df: pd.DataFrame) -> str | None:
    for c in ("Compound", "compound_id", "material_id", "materialID"):
        if c in df.columns:
            return c
    return None


def _class_balance(y: np.ndarray, idx: list[int]) -> dict[str, int]:
    yy = y[np.asarray(idx, dtype=int)]
    n_pos = int(np.sum(yy == 1))
    n_neg = int(np.sum(yy == 0))
    return {"n_total": int(len(idx)), "n_mit_1": n_pos, "n_non_mit_0": n_neg}


def _quantum_simulation_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Resolved PennyLane backend settings (optional; defaults preserve exact simulation)."""
    backend = str(getattr(args, "quantum_backend", "exact")).lower().strip()
    shots = getattr(args, "shots", None)
    if backend == "shots" and shots is not None:
        shots = int(shots)
    else:
        shots = None
    noise_model = str(getattr(args, "noise_model", "none")).lower().strip()
    noise_prob = float(getattr(args, "noise_prob", 0.0))
    if backend != "noisy":
        noise_model = "none"
        noise_prob = 0.0
    return {
        "quantum_backend": backend,
        "shots": shots,
        "noise_model": noise_model,
        "noise_prob": noise_prob,
    }


def _quantum_simulation_build_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return _quantum_simulation_settings(args)


def _write_quantum_backend_unavailable(out_root: Path, note: str, args: argparse.Namespace) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "not_available",
        "message": note,
        "quantum_backend": str(getattr(args, "quantum_backend", "noisy")),
        "noise_model": str(getattr(args, "noise_model", "none")),
        "noise_prob": float(getattr(args, "noise_prob", 0.0)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_root / "quantum_backend_unavailable.json", "w") as f:
        json.dump(payload, f, indent=2)


def _skip_if_quantum_backend_unavailable(args: argparse.Namespace, out_root: Path) -> bool:
    if args.head_type not in ("quantum", "reup"):
        return False
    sim = _quantum_simulation_settings(args)
    if sim["quantum_backend"] != "noisy":
        return False
    from models.posthoc_heads import probe_noise_model_support, probe_noisy_quantum_support

    ok, note = probe_noisy_quantum_support()
    if not ok:
        _write_quantum_backend_unavailable(out_root, note or "noisy backend not available", args)
        return True
    if sim["noise_model"] not in ("none", ""):
        ok_nm, note_nm = probe_noise_model_support(sim["noise_model"])
        if not ok_nm:
            _write_quantum_backend_unavailable(
                out_root, note_nm or f"noise_model {sim['noise_model']!r} not available", args
            )
            return True
    return False


def _head_hyperparameters(args: argparse.Namespace, head_type: str) -> dict[str, Any]:
    if head_type == "linear":
        return {}
    if head_type == "mlp":
        return {"hidden_dim": int(args.hidden_dim), "dropout": 0.1}
    if head_type == "quantum":
        hp = {
            "proj_dim": 16,
            "n_qubits": int(args.n_qubits),
            "n_q_layers": int(args.n_q_layers),
            "entanglement_type": str(args.entanglement_type),
            "encoding_type": str(args.encoding_type),
            "measurement_type": str(args.measurement_type),
            "measurement_readout_hidden": int(args.measurement_readout_hidden),
            "dropout": 0.1,
        }
        hp.update(_quantum_simulation_settings(args))
        return hp
    if head_type == "reup":
        hp = {
            "proj_dim": int(args.reup_proj_dim),
            "n_qubits": int(args.reup_n_qubits),
            "n_reup_layers": int(args.reup_n_layers),
            "reup_method": str(args.reup_method),
            "entanglement": bool(args.reup_entanglement),
            "entanglement_type": str(args.entanglement_type),
            "encoding_type": str(args.encoding_type),
            "measurement_type": str(args.measurement_type),
            "measurement_readout_hidden": int(args.measurement_readout_hidden),
            "dropout": 0.1,
        }
        hp.update(_quantum_simulation_settings(args))
        return hp
    if head_type == "latent_mlp_vqc_matched":
        return {
            "proj_dim": 16,
            "n_qubits": int(args.n_qubits),
            "n_q_layers": int(args.n_q_layers),
            "target_circuit_params": int(args.n_q_layers * args.n_qubits),
            "dropout": 0.1,
            "reference_head": "quantum",
        }
    if head_type == "latent_mlp_reup_matched":
        return {
            "proj_dim": int(args.reup_proj_dim),
            "n_qubits": int(args.reup_n_qubits),
            "n_reup_layers": int(args.reup_n_layers),
            "target_circuit_params": int(args.reup_n_layers * 3 * args.reup_n_qubits),
            "dropout": 0.1,
            "reference_head": "reup",
        }
    if head_type == "proj_bottleneck_mlp":
        return {
            "proj_dim": int(args.reup_proj_dim),
            "dropout": 0.1,
            "reference_head": "reup",
            "note": "proj_dim-matched classical bottleneck (no circuit-param sizing)",
        }
    if head_type == "latent_bottleneck_mlp":
        return {
            "proj_dim": int(args.reup_proj_dim),
            "n_qubits": int(args.reup_n_qubits),
            "latent_dim": int(args.reup_n_qubits),
            "dropout": 0.1,
            "reference_head": "reup",
            "note": "latent/qubit-width classical bottleneck (no circuit-param sizing)",
        }
    if head_type == "random_fourier_logistic":
        fd = args.fourier_dim
        if fd is None:
            fd = default_fourier_dim_for_vqc_match(64)
        return {
            "fourier_dim": int(fd),
            "rff_scale": float(args.rff_scale),
            "note": "seed stored per-run in config.json",
        }
    return {}


def _ece_score(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float | None:
    """Expected calibration error (histogram bins). Returns None if undefined."""
    y_true = y_true.astype(np.int64)
    probs = np.clip(probs.astype(np.float64), 0.0, 1.0)
    if len(y_true) < 2:
        return None
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(probs[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def _metrics_at_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probs >= threshold).astype(np.int64)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }


def _build_predictions_df(
    indices: list[int],
    y_all: np.ndarray,
    probs: np.ndarray,
    df: pd.DataFrame,
    id_col: str | None,
    fold: int,
    seed: int,
    head_type: str,
    mit_fraction: float,
    imbalance_mode: str,
    threshold_05: float = 0.5,
    threshold_tuned: float | None = None,
) -> pd.DataFrame:
    rows = []
    for i, row_i in enumerate(indices):
        p = float(probs[i])
        row: dict[str, Any] = {
            "index": int(row_i),
            "true_label": int(y_all[row_i]),
            "probability": p,
            "predicted_label_at_0p5": int(p >= threshold_05),
            "fold": int(fold),
            "seed": int(seed),
            "head_type": head_type,
            "mit_fraction": float(mit_fraction),
            "imbalance_mode": str(imbalance_mode),
        }
        if id_col:
            row["compound_id"] = str(df.iloc[row_i][id_col])
        if threshold_tuned is not None:
            row["predicted_label_at_val_f1_threshold"] = int(p >= threshold_tuned)
        rows.append(row)
    return pd.DataFrame(rows)


def _counts_from_labels(y: np.ndarray) -> tuple[int, int]:
    y = y.astype(np.int64)
    return int((y == 1).sum()), int((y == 0).sum())


def _sanity_check_train_counts(n_pos: int, n_neg: int, *, context: str) -> None:
    if n_pos < 1:
        raise ValueError(f"{context}: training split has no positives after sampling.")
    if n_neg < 1:
        raise ValueError(f"{context}: training split has no negatives after sampling.")
    if n_pos < 2:
        raise ValueError(f"{context}: too few positives after sampling (n_pos={n_pos}).")


def _subsample_train_indices_with_mode(
    *,
    train_idx: list[int],
    y_all: np.ndarray,
    mit_fraction: float,
    imbalance_mode: str,
    rng: np.random.Generator,
) -> tuple[list[int], dict[str, int]]:
    """
    Return the final *unique* training indices used for optimization under the chosen mode.
    Validation and test are never resampled here.
    """
    if not (0.0 < mit_fraction <= 1.0):
        raise ValueError(f"mit_fraction must be in (0, 1], got {mit_fraction}")
    mode = str(imbalance_mode).lower().strip()
    arr = np.array(train_idx, dtype=np.int64)
    y = y_all[arr].astype(np.int64)
    pos = arr[y == 1].copy()
    neg = arr[y == 0].copy()
    n_pos_full = int(len(pos))
    n_neg_full = int(len(neg))
    if n_pos_full < 1:
        raise ValueError("Low-data subsample: no positive (MIT) samples in training fold.")

    if mit_fraction >= 1.0 - 1e-12:
        n_keep_pos = n_pos_full
    else:
        n_keep_pos = max(1, int(round(n_pos_full * mit_fraction)))
        n_keep_pos = min(n_keep_pos, n_pos_full)
    rng.shuffle(pos)
    pos_sel = pos[:n_keep_pos]

    if mode in ("original", "oversample_positive", "focal_loss"):
        neg_sel = neg
    elif mode == "matched_negatives":
        k = int(len(pos_sel))
        if n_neg_full < k:
            raise ValueError(f"matched_negatives needs at least {k} negatives; got {n_neg_full}.")
        rng.shuffle(neg)
        neg_sel = neg[:k]
    elif mode == "preserve_ratio":
        # Keep the original neg:pos ratio in the subsampled set.
        ratio = (n_neg_full / n_pos_full) if n_pos_full > 0 else 0.0
        n_keep_neg = max(1, int(round(len(pos_sel) * ratio)))
        n_keep_neg = min(n_keep_neg, n_neg_full)
        rng.shuffle(neg)
        neg_sel = neg[:n_keep_neg]
    else:
        raise ValueError(
            f"Unknown imbalance_mode={imbalance_mode!r}; expected "
            "original, matched_negatives, preserve_ratio, oversample_positive, focal_loss"
        )

    out_idx = np.concatenate([pos_sel, neg_sel]).astype(np.int64)
    out_idx.sort()

    n_pos_used = int((y_all[out_idx] == 1).sum())
    n_neg_used = int((y_all[out_idx] == 0).sum())
    _sanity_check_train_counts(n_pos_used, n_neg_used, context=f"imbalance_mode={mode}")

    stats = {
        "n_pos_train_full": n_pos_full,
        "n_pos_train_used": n_pos_used,
        "n_neg_train_full": n_neg_full,
        "n_neg_train_used": n_neg_used,
    }
    return out_idx.tolist(), stats


class FocalLossWithLogits(nn.Module):
    """Binary focal loss on logits. Targets are floats in {0,1}."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"focal_alpha must be in [0,1], got {alpha}")
        if gamma < 0.0:
            raise ValueError(f"focal_gamma must be >= 0, got {gamma}")
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = targets * p + (1.0 - targets) * (1.0 - p)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss = alpha_t * ((1.0 - pt) ** self.gamma) * bce
        return loss.mean()


def subsample_train_for_lowdata(
    train_idx: list[int],
    y_all: np.ndarray,
    mit_fraction: float,
    match_negatives_to_positives: bool,
    rng: np.random.Generator,
) -> tuple[list[int], dict[str, int]]:
    """
    Subsample MIT (positive) training indices to ``mit_fraction`` of the training positives.
    Negatives: all kept, or downsampled to match positive count if ``match_negatives_to_positives``.
    """
    if not (0.0 < mit_fraction <= 1.0):
        raise ValueError(f"mit_fraction must be in (0, 1], got {mit_fraction}")
    arr = np.array(train_idx, dtype=np.int64)
    y = y_all[arr]
    pos = arr[y == 1].copy()
    neg = arr[y == 0].copy()  # copy before any in-place shuffle
    n_pos = int(len(pos))
    n_neg = int(len(neg))
    if n_pos < 1:
        raise ValueError("Low-data subsample: no positive (MIT) samples in training fold.")
    if mit_fraction >= 1.0 - 1e-12:
        n_keep_pos = n_pos
    else:
        n_keep_pos = max(1, int(round(n_pos * mit_fraction)))
        n_keep_pos = min(n_keep_pos, n_pos)
    rng.shuffle(pos)
    pos_sel = pos[:n_keep_pos]

    if match_negatives_to_positives:
        k = int(len(pos_sel))
        if n_neg < k:
            raise ValueError(
                f"match_negatives_to_positives needs at least {k} negatives in train; got {n_neg}."
            )
        rng.shuffle(neg)
        neg_sel = neg[:k]
    else:
        neg_sel = neg

    out_idx = np.concatenate([pos_sel, neg_sel]).astype(np.int64)
    out_idx.sort()
    stats = {
        "n_pos_train_full": n_pos,
        "n_pos_train_used": int(len(pos_sel)),
        "n_neg_train_full": n_neg,
        "n_neg_train_used": int(len(neg_sel)),
    }
    return out_idx.tolist(), stats


def _make_val_from_train(train_idx: list[int], labels: np.ndarray, rng: np.random.Generator, frac: float = 0.1) -> tuple[list[int], list[int]]:
    """Stratified-ish holdout from train when val_indices is empty."""
    if not train_idx:
        return [], []
    tr = np.array(train_idx)
    y = labels[tr]
    pos = tr[y == 1]
    neg = tr[y == 0]
    n_pos_val = max(1, int(round(len(pos) * frac)))
    n_neg_val = max(1, int(round(len(neg) * frac)))
    rng.shuffle(pos)
    rng.shuffle(neg)
    val = np.concatenate([pos[:n_pos_val], neg[:n_neg_val]]).tolist()
    new_train = [i for i in train_idx if i not in set(val)]
    return new_train, val


def _pos_weight_from_labels(y: np.ndarray) -> torch.Tensor:
    """BCE pos_weight = n_negative / n_positive (training split only)."""
    y = y.astype(np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos < 1:
        raise ValueError("Training split has no positive samples; cannot set pos_weight.")
    w = float(n_neg) / float(n_pos)
    return torch.tensor([w], dtype=torch.float32)


def _batch_tensors(dataset: TensorDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _parse_bool_str(s: str) -> bool:
    x = str(s).lower().strip()
    if x in ("true", "1", "yes", "y"):
        return True
    if x in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


def _confusion_matrix_2x2_dict(m: dict[str, Any]) -> dict[str, Any]:
    """TN/FP/FN/TP from :func:`_binary_metrics` → labelled 2×2 matrix (rows = true class)."""
    return {
        "labels": [0, 1],
        "matrix": [[int(m["TN"]), int(m["FP"])], [int(m["FN"]), int(m["TP"])]],
    }


@torch.no_grad()
def _reup_pre_readout_statistics(model: REUPHead, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """Mean/std of PauliZ expectations before the readout layer (test set)."""
    model.eval()
    chunks: list[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(device)
        z = model.get_pre_readout(xb).detach().cpu().numpy()
        chunks.append(z)
    z = np.concatenate(chunks, axis=0)
    return {
        "global_mean": float(np.mean(z)),
        "global_std": float(np.std(z)),
        "per_qubit_mean": [float(x) for x in np.mean(z, axis=0).tolist()],
        "per_qubit_std": [float(x) for x in np.std(z, axis=0).tolist()],
    }


@torch.no_grad()
def _forward_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    outs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).squeeze(-1)
        outs.append(logits.detach().cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(outs), np.concatenate(ys)


def _pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


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


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb).squeeze(-1)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def _save_compression_fold_artifacts(fold_dir: Path, compression_meta: dict[str, Any] | None) -> None:
    """Write per-fold PCA / feature-selection / autoencoder artifacts."""
    if not compression_meta:
        return
    fold_dir.mkdir(parents=True, exist_ok=True)
    if "pca_explained_variance_ratio" in compression_meta:
        with open(fold_dir / "pca_explained_variance.json", "w") as f:
            json.dump(
                {
                    "explained_variance_ratio": compression_meta["pca_explained_variance_ratio"],
                    "explained_variance_total": compression_meta.get("pca_explained_variance_total"),
                    "singular_values": compression_meta.get("pca_singular_values"),
                },
                f,
                indent=2,
            )
    if "selected_feature_indices" in compression_meta:
        with open(fold_dir / "selected_feature_indices.json", "w") as f:
            json.dump(
                {
                    "selected_feature_indices": compression_meta["selected_feature_indices"],
                    "selected_emb_columns": compression_meta.get("selected_emb_columns"),
                    "mutual_information_scores": compression_meta.get("mutual_information_scores"),
                },
                f,
                indent=2,
            )
    if "autoencoder_training_history" in compression_meta:
        pd.DataFrame(compression_meta["autoencoder_training_history"]).to_csv(
            fold_dir / "autoencoder_training_history.csv", index=False
        )
    with open(fold_dir / "embedding_compression.json", "w") as f:
        json.dump(compression_meta, f, indent=2)


def train_fold(
    fold_id: int,
    df: pd.DataFrame,
    emb_cols: list[str],
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    head_type: str,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
    *,
    strict_val: bool = False,
    seed: int | None = None,
    X_all: np.ndarray | None = None,
) -> dict[str, Any]:
    imbalance_mode = str(getattr(args, "imbalance_mode", "original"))
    if X_all is None:
        X_all = df[emb_cols].to_numpy(dtype=np.float32)
    else:
        X_all = np.asarray(X_all, dtype=np.float32)
    y_all = df["Label"].to_numpy(dtype=np.int64)
    compounds = df["Compound"].astype(str).tolist()

    # Apply class-imbalance sampling mode on the *training* indices only.
    train_idx, imbalance_stats = _subsample_train_indices_with_mode(
        train_idx=train_idx,
        y_all=y_all,
        mit_fraction=float(getattr(args, "mit_fraction", 1.0) or 1.0),
        imbalance_mode=imbalance_mode,
        rng=rng,
    )

    X_tr = torch.from_numpy(X_all[train_idx])
    y_tr = torch.from_numpy(y_all[train_idx].astype(np.float32))
    train_ds = TensorDataset(X_tr, y_tr)
    if imbalance_mode == "oversample_positive":
        y_tr_np = y_all[train_idx].astype(np.int64)
        n_pos = int((y_tr_np == 1).sum())
        n_neg = int((y_tr_np == 0).sum())
        _sanity_check_train_counts(n_pos, n_neg, context="oversample_positive")
        # WeightedRandomSampler: sample with replacement so positives appear ~as often as negatives.
        weights = np.where(y_tr_np == 1, float(n_neg) / float(n_pos), 1.0).astype(np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, drop_last=False)
    else:
        train_loader = _batch_tensors(train_ds, args.batch_size, shuffle=True)

    if val_idx:
        X_va = torch.from_numpy(X_all[val_idx])
        y_va = torch.from_numpy(y_all[val_idx].astype(np.float32))
        val_ds = TensorDataset(X_va, y_va)
        val_loader = _batch_tensors(val_ds, args.batch_size, shuffle=False)
    elif strict_val:
        raise RuntimeError(
            f"Fold {fold_id}: strict nested splits require non-empty val_indices "
            "(validation must not be carved from train)."
        )
    else:
        new_tr, val_idx = _make_val_from_train(train_idx, y_all, rng)
        if not val_idx:
            raise RuntimeError(f"Fold {fold_id}: could not build validation from train.")
        train_idx = new_tr
        X_tr = torch.from_numpy(X_all[train_idx])
        y_tr = torch.from_numpy(y_all[train_idx].astype(np.float32))
        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = _batch_tensors(train_ds, args.batch_size, shuffle=True)
        X_va = torch.from_numpy(X_all[val_idx])
        y_va = torch.from_numpy(y_all[val_idx].astype(np.float32))
        val_ds = TensorDataset(X_va, y_va)
        val_loader = _batch_tensors(val_ds, args.batch_size, shuffle=False)

    input_dim = len(emb_cols)
    run_seed = int(seed) if seed is not None else int(args.seed)
    if head_type in ("linear", "logistic_regression"):
        model = build_head("linear", input_dim)
    elif head_type == "mlp":
        model = build_head(head_type, input_dim, hidden_dim=args.hidden_dim)
    elif head_type == "quantum":
        try:
            model = build_head(
                head_type,
                input_dim,
                n_qubits=args.n_qubits,
                n_q_layers=args.n_q_layers,
                entanglement_type=args.entanglement_type,
                encoding_type=args.encoding_type,
                measurement_type=args.measurement_type,
                measurement_readout_hidden=args.measurement_readout_hidden,
                **_quantum_simulation_build_kwargs(args),
            )
        except NoisyBackendUnavailableError as e:
            raise RuntimeError(str(e)) from e
    elif head_type == "reup":
        try:
            model = build_head(
                head_type,
                input_dim,
                proj_dim=args.reup_proj_dim,
                n_qubits=args.reup_n_qubits,
                n_reup_layers=args.reup_n_layers,
                reup_method=args.reup_method,
                entanglement=args.reup_entanglement,
                entanglement_type=args.entanglement_type,
                encoding_type=args.encoding_type,
                measurement_type=args.measurement_type,
                measurement_readout_hidden=args.measurement_readout_hidden,
                **_quantum_simulation_build_kwargs(args),
            )
        except NoisyBackendUnavailableError as e:
            raise RuntimeError(str(e)) from e
    elif head_type == "latent_mlp_vqc_matched":
        model = build_head(
            head_type,
            input_dim,
            proj_dim=16,
            n_qubits=args.n_qubits,
            n_q_layers=args.n_q_layers,
            dropout=0.1,
        )
    elif head_type == "latent_mlp_reup_matched":
        model = build_head(
            head_type,
            input_dim,
            proj_dim=args.reup_proj_dim,
            n_qubits=args.reup_n_qubits,
            n_reup_layers=args.reup_n_layers,
            dropout=0.1,
        )
    elif head_type == "proj_bottleneck_mlp":
        model = build_head(
            head_type,
            input_dim,
            proj_dim=args.reup_proj_dim,
            dropout=0.1,
            reference_head="reup",
        )
    elif head_type == "latent_bottleneck_mlp":
        model = build_head(
            head_type,
            input_dim,
            proj_dim=args.reup_proj_dim,
            n_qubits=args.reup_n_qubits,
            dropout=0.1,
            reference_head="reup",
        )
    elif head_type == "random_fourier_logistic":
        fd = args.fourier_dim
        if fd is None:
            fd = default_fourier_dim_for_vqc_match(input_dim)
        model = build_head(
            head_type,
            input_dim,
            fourier_dim=int(fd),
            seed=run_seed,
            rff_scale=float(args.rff_scale),
        )
    else:
        model = build_head(head_type, input_dim)
    model = model.to(device)

    # Loss
    if imbalance_mode == "focal_loss":
        pos_w = None
        criterion: nn.Module = FocalLossWithLogits(alpha=float(args.focal_alpha), gamma=float(args.focal_gamma))
    else:
        pos_w = _pos_weight_from_labels(y_all[train_idx]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_pr: float | None = None
    best_val_epoch = 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_left = args.patience
    last_epoch = 0
    training_history: list[dict[str, Any]] = []

    for epoch in range(args.epochs):
        last_epoch = epoch
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_logits, val_y = _forward_logits(model, val_loader, device)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        pr = _pr_auc(val_y.astype(np.int64), val_probs)
        training_history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_pr_auc": float(pr) if not np.isnan(pr) else None,
            }
        )
        improved = False
        if not np.isnan(pr):
            if best_pr is None or pr > best_pr:
                best_pr = pr
                improved = True
        if improved:
            best_val_epoch = int(epoch)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)

    # Validation probabilities (best checkpoint) for threshold tuning / calibration in downstream eval
    val_logits_final, val_y_final = _forward_logits(model, val_loader, device)
    val_probs_final = 1.0 / (1.0 + np.exp(-val_logits_final))
    val_predictions: list[dict[str, Any]] = []
    for i, row_i in enumerate(val_idx):
        val_predictions.append(
            {
                "sample_index": int(row_i),
                "Compound": compounds[row_i],
                "Label": int(y_all[row_i]),
                "predicted_probability": float(val_probs_final[i]),
            }
        )

    # Test evaluation
    X_te = torch.from_numpy(X_all[test_idx])
    y_te = torch.from_numpy(y_all[test_idx].astype(np.float32))
    test_ds = TensorDataset(X_te, y_te)
    test_loader = _batch_tensors(test_ds, batch_size=max(1, len(test_idx)), shuffle=False)
    test_logits, test_y = _forward_logits(model, test_loader, device)
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    pred_labels = (test_probs >= 0.5).astype(np.int64)
    metrics = _binary_metrics(test_y.astype(np.int64), test_probs, pred_labels)

    val_y_np = val_y_final.astype(np.int64)
    t_f1_val, best_val_f1 = tune_threshold_f1(val_y_np, val_probs_final)
    pred_tuned_test = (test_probs >= t_f1_val).astype(np.int64)
    metrics_tuned = _binary_metrics(test_y.astype(np.int64), test_probs, pred_tuned_test)

    n_params = count_parameters(model)
    n_params_trainable = count_trainable_parameters(model)
    param_meta: dict[str, Any] = {
        "n_parameters": n_params,
        "n_parameters_trainable": n_params_trainable,
    }
    if head_type == "latent_mlp_vqc_matched" and isinstance(model, LatentMLPMatchedHead):
        ref = reference_quantum_trainable_count(input_dim, model.proj_dim, model.readout_dim, args.n_q_layers)
        param_meta.update(
            {
                "reference_head": "quantum",
                "reference_trainable_parameters": ref,
                "target_circuit_parameters": model.target_circuit_params,
                "classical_hidden_dim": model.classical_hidden_dim,
            }
        )
    elif head_type == "latent_mlp_reup_matched" and isinstance(model, LatentMLPMatchedHead):
        ref = reference_reup_trainable_count(
            input_dim, model.proj_dim, model.readout_dim, args.reup_n_layers, model.latent_dim
        )
        param_meta.update(
            {
                "reference_head": "reup",
                "reference_trainable_parameters": ref,
                "target_circuit_parameters": model.target_circuit_params,
                "classical_hidden_dim": model.classical_hidden_dim,
            }
        )
    elif head_type == "proj_bottleneck_mlp" and isinstance(model, ProjBottleneckMLPHead):
        param_meta.update(
            {
                "reference_head": model.reference_head,
                "proj_dim": model.proj_dim,
                "match_type": "proj_dim_bottleneck",
            }
        )
    elif head_type == "latent_bottleneck_mlp" and isinstance(model, LatentBottleneckMLPHead):
        param_meta.update(
            {
                "reference_head": model.reference_head,
                "proj_dim": model.proj_dim,
                "latent_dim": model.latent_dim,
                "match_type": "latent_width_bottleneck",
            }
        )
    elif head_type == "random_fourier_logistic" and isinstance(model, RandomFourierLogisticHead):
        ref = reference_quantum_trainable_count(input_dim)
        param_meta.update(
            {
                "fourier_dim": model.fourier_dim,
                "random_fourier_seed": model.seed,
                "rff_scale": model.rff_scale,
                "reference_head": "quantum",
                "reference_trainable_parameters": ref,
            }
        )

    # Save estimated circuit depth if provided by the head
    if head_type == "quantum" and hasattr(model, "estimated_circuit_depth"):
        param_meta["estimated_circuit_depth"] = int(getattr(model, "estimated_circuit_depth"))
    if head_type == "reup" and hasattr(model, "estimated_circuit_depth"):
        param_meta["estimated_circuit_depth"] = int(getattr(model, "estimated_circuit_depth"))

    samples: list[dict[str, Any]] = []
    for i, row_i in enumerate(test_idx):
        samples.append(
            {
                "sample_index": int(row_i),
                "Compound": compounds[row_i],
                "Label": int(y_all[row_i]),
                "predicted_probability": float(test_probs[i]),
                "predicted_label_0.5": int(pred_labels[i]),
            }
        )

    out: dict[str, Any] = {
        "fold": fold_id,
        "head_type": head_type,
        "epochs_trained": last_epoch + 1,
        "validation_best_epoch": int(best_val_epoch),
        "best_val_pr_auc": float(best_pr) if best_pr is not None else None,
        "pos_weight": float(pos_w.item()) if pos_w is not None else None,
        "imbalance_mode": str(imbalance_mode),
        "imbalance_train_stats": imbalance_stats,
        "train_indices_used": train_idx,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "test_metrics": metrics,
        "test_metrics_at_tuned_threshold": metrics_tuned,
        "val_predictions": val_predictions,
        "test_predictions": samples,
        "training_history": training_history,
        "tuned_threshold_f1_on_val": float(t_f1_val),
        "best_val_f1_at_threshold": float(best_val_f1),
        "n_parameters": n_params,
        "n_parameters_trainable": n_params_trainable,
        "parameter_metadata": param_meta,
        "compression_metadata": getattr(args, "_fold_compression_meta", None),
    }

    if head_type == "reup" and isinstance(model, REUPHead):
        t_f1_val, _ = tune_threshold_f1(val_y_final.astype(np.int64), val_probs_final)
        pred_tuned = (test_probs >= t_f1_val).astype(np.int64)
        metrics_tuned = _binary_metrics(test_y.astype(np.int64), test_probs, pred_tuned)
        z_stats = _reup_pre_readout_statistics(model, test_loader, device)
        out["reup_diagnostics"] = {
            "hyperparameters": {
                "proj_dim": int(model.proj_dim),
                "latent_dim": int(model.latent_dim),
                "n_qubits": int(model.n_qubits),
                "n_reup_layers": int(model.n_reup_layers),
                "reup_method": str(model.reup_method),
                "entanglement": bool(model.entanglement),
                "dropout": float(model.dropout_p),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "batch_size": int(args.batch_size),
                "epochs_max": int(args.epochs),
                "patience": int(args.patience),
            },
            "test_probabilities": [float(p) for p in test_probs],
            "tuned_threshold_f1_on_val": float(t_f1_val),
            "confusion_matrix_at_0.5": _confusion_matrix_2x2_dict(metrics),
            "test_metrics_at_tuned_threshold": metrics_tuned,
            "confusion_matrix_at_tuned_threshold": _confusion_matrix_2x2_dict(metrics_tuned),
            "pre_readout_pauli_z_expectations": z_stats,
        }

    return out


_METRIC_KEYS = [
    "balanced_accuracy",
    "f1",
    "recall",
    "precision",
    "specificity",
    "roc_auc",
    "pr_auc",
    "average_precision",
]


def _aggregate_folds(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {}
    for k in _METRIC_KEYS:
        vals = []
        for fr in fold_results:
            v = fr["test_metrics"].get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(v)
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        else:
            agg[k] = None
    return agg


def _aggregate_across_seeds(per_seed_aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/std of each metric's cross-fold mean, across seeds."""
    out: dict[str, Any] = {}
    for k in _METRIC_KEYS:
        vals = []
        for agg in per_seed_aggregates:
            block = agg.get(k)
            if not block:
                continue
            m = block.get("mean")
            if m is None or (isinstance(m, float) and np.isnan(m)):
                continue
            vals.append(float(m))
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_seeds": len(vals)}
        else:
            out[k] = None
    return out


def _aggregate_per_fold_across_seeds(
    fold_results_by_seed: list[list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """For each fold id, mean/std of test metrics across seeds."""
    n_folds = len(fold_results_by_seed[0]) if fold_results_by_seed else 0
    by_fold: dict[str, dict[str, Any]] = {}
    for fold_id in range(n_folds):
        fold_key = str(fold_id)
        by_fold[fold_key] = {}
        for k in _METRIC_KEYS:
            vals = []
            for seed_folds in fold_results_by_seed:
                if fold_id >= len(seed_folds):
                    continue
                v = seed_folds[fold_id]["test_metrics"].get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
            if vals:
                by_fold[fold_key][k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n_seeds": len(vals),
                }
            else:
                by_fold[fold_key][k] = None
    return by_fold


def _load_summary(path: Path) -> dict[str, Any]:
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return {"by_head": {}}


def _fold_summary_row(fr: dict[str, Any]) -> dict[str, Any]:
    """Metrics-only row for summary.json (full predictions live in per-fold files)."""
    return {
        "fold": fr["fold"],
        "head_type": fr["head_type"],
        "epochs_trained": fr["epochs_trained"],
        "best_val_pr_auc": fr.get("best_val_pr_auc"),
        "pos_weight": fr.get("pos_weight"),
        "n_train": fr.get("n_train"),
        "n_val": fr.get("n_val"),
        "n_test": fr.get("n_test"),
        "n_train_with_embedding": fr.get("n_train_with_embedding"),
        "embedding_path": fr.get("embedding_path"),
        "split_json": fr.get("split_json"),
        "test_metrics": fr["test_metrics"],
    }


def _write_summary(path: Path, head_type: str, fold_results: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    data = _load_summary(path)
    data["by_head"][head_type] = {
        "folds": [_fold_summary_row(fr) for fr in fold_results],
        "aggregate": aggregate,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


DEFAULT_LOWDATA_SEEDS = 5


def _reviewer_metrics_json(
    y_test: np.ndarray,
    test_probs: np.ndarray,
    tuned_threshold: float,
) -> dict[str, Any]:
    y_test = y_test.astype(np.int64)
    m05 = _metrics_at_threshold(y_test, test_probs, 0.5)
    mt = _metrics_at_threshold(y_test, test_probs, tuned_threshold)
    ece = _ece_score(y_test, test_probs)
    out: dict[str, Any] = {
        "pr_auc": float(_pr_auc(y_test, test_probs)),
        "roc_auc": float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else float("nan"),
        "balanced_accuracy_at_0p5": m05["balanced_accuracy"],
        "f1_at_0p5": m05["f1"],
        "precision_at_0p5": m05["precision"],
        "recall_at_0p5": m05["recall"],
        "mcc_at_0p5": m05["mcc"],
        "balanced_accuracy_at_tuned_threshold": mt["balanced_accuracy"],
        "f1_at_tuned_threshold": mt["f1"],
        "precision_at_tuned_threshold": mt["precision"],
        "recall_at_tuned_threshold": mt["recall"],
        "mcc_at_tuned_threshold": mt["mcc"],
        "brier_score": float(brier_score_loss(y_test, test_probs)),
        "ece": ece,
        "ece_computed": ece is not None,
        "tuned_threshold_val_f1": float(tuned_threshold),
    }
    if ece is None:
        out["ece_note"] = "not_computed_insufficient_bins_or_samples"
    return out


def save_reviewer_fold_artifacts(
    fold_dir: Path,
    result: dict[str, Any],
    df: pd.DataFrame,
    y_all: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    args: argparse.Namespace,
    seed: int,
    mit_fraction: float,
    split_dir: Path,
    emb_path: Path,
    fold_path: Path,
) -> None:
    """Write reviewer-proof per-fold artifacts (metrics, predictions, config, history)."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    head_type = str(result["head_type"])
    fold_id = int(result["fold"])
    id_col = _id_column(df)

    val_idx_list = val_idx
    test_idx_list = test_idx
    val_preds = result["val_predictions"]
    test_preds = result["test_predictions"]

    val_probs = np.array([p["predicted_probability"] for p in val_preds], dtype=np.float64)
    test_probs = np.array([p["predicted_probability"] for p in test_preds], dtype=np.float64)
    val_y = np.array([p["Label"] for p in val_preds], dtype=np.int64)
    test_y = np.array([p["Label"] for p in test_preds], dtype=np.int64)

    t_f1 = float(result["tuned_threshold_f1_on_val"])

    metrics = _reviewer_metrics_json(test_y, test_probs, t_f1)
    metrics["best_val_pr_auc"] = result.get("best_val_pr_auc")
    metrics["epochs_trained"] = result.get("epochs_trained")
    metrics["validation_best_epoch"] = result.get("validation_best_epoch")
    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(result["training_history"]).to_csv(fold_dir / "training_history.csv", index=False)

    val_df = _build_predictions_df(
        val_idx_list,
        y_all,
        val_probs,
        df,
        id_col,
        fold_id,
        seed,
        head_type,
        mit_fraction,
        str(getattr(args, "imbalance_mode", "original")),
        threshold_tuned=t_f1,
    )
    val_df.to_csv(fold_dir / "predictions_val.csv", index=False)

    test_df = _build_predictions_df(
        test_idx_list,
        y_all,
        test_probs,
        df,
        id_col,
        fold_id,
        seed,
        head_type,
        mit_fraction,
        str(getattr(args, "imbalance_mode", "original")),
        threshold_tuned=t_f1,
    )
    test_df.to_csv(fold_dir / "predictions_test.csv", index=False)

    with open(fold_dir / "best_threshold.json", "w") as f:
        json.dump(
            {
                "criterion": "max_f1_on_validation",
                "threshold": t_f1,
                "best_val_f1": result.get("best_val_f1_at_threshold"),
                "best_val_pr_auc": result.get("best_val_pr_auc"),
            },
            f,
            indent=2,
        )

    param_payload: dict[str, Any] = {
        "n_parameters": result.get("n_parameters"),
        "n_parameters_trainable": result.get("n_parameters_trainable"),
        "validation_best_epoch": result.get("validation_best_epoch"),
    }
    if result.get("parameter_metadata"):
        param_payload.update(result["parameter_metadata"])
    with open(fold_dir / "model_parameter_count.json", "w") as f:
        json.dump(param_payload, f, indent=2)

    config = {
        "head_type": head_type,
        "seed": int(seed),
        "fold": fold_id,
        "mit_fraction": float(mit_fraction),
        "embedding_variant": str(getattr(args, "embedding_variant", "raw_cgcnn_64")),
        "quantum_backend": _quantum_simulation_settings(args)["quantum_backend"],
        "shots": _quantum_simulation_settings(args)["shots"],
        "noise_model": _quantum_simulation_settings(args)["noise_model"],
        "noise_prob": _quantum_simulation_settings(args)["noise_prob"],
        "imbalance_mode": str(getattr(args, "imbalance_mode", "original")),
        "focal_alpha": float(getattr(args, "focal_alpha", 0.25)),
        "focal_gamma": float(getattr(args, "focal_gamma", 2.0)),
        "split_dir": str(split_dir.relative_to(ROOT)),
        "split_json": str(fold_path.relative_to(ROOT)),
        "embedding_path": str(emb_path.relative_to(ROOT)),
        "train_counts": _class_balance(y_all, train_idx),
        "val_counts": _class_balance(y_all, val_idx),
        "test_counts": _class_balance(y_all, test_idx),
        "optimizer": "Adam",
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "max_epochs": int(args.epochs),
        "early_stopping_patience": int(args.patience),
        "early_stopping_metric": "val_pr_auc",
        "pos_weight": result.get("pos_weight"),
        "model_hyperparameters": _head_hyperparameters(args, head_type),
        "match_negatives_to_positives": bool(getattr(args, "match_negatives_to_positives", False)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if result.get("parameter_metadata", {}).get("estimated_circuit_depth") is not None:
        config["estimated_circuit_depth"] = int(result["parameter_metadata"]["estimated_circuit_depth"])
    _save_compression_fold_artifacts(fold_dir, result.get("compression_metadata"))
    with open(fold_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Legacy-compatible single JSON for debugging
    with open(fold_dir / f"{head_type}_results.json", "w") as f:
        json.dump(result, f, indent=2)


def _run_reviewer_eval(args: argparse.Namespace) -> None:
    """Train with nested splits; save structured reviewer_eval outputs."""
    mit_fraction = float(args.mit_fraction) if args.mit_fraction is not None else 1.0
    if not (0.0 < mit_fraction <= 1.0):
        raise ValueError(f"mit_fraction must be in (0, 1], got {mit_fraction}")

    seeds = _parse_seeds_arg(args.seeds, args.seed, args.n_lowdata_seeds)
    emb_path = (ROOT / args.embedding_path).resolve()
    splits_dir = _resolve_splits_dir(args)
    out_root = (ROOT / args.output_dir).resolve()
    if _skip_if_quantum_backend_unavailable(args, out_root):
        print(
            json.dumps(
                {
                    "skipped": True,
                    "reason": "quantum_backend_unavailable",
                    "output_dir": str(out_root.relative_to(ROOT)),
                },
                indent=2,
            )
        )
        return
    strict_val = "nested" in splits_dir.name or getattr(args, "strict_nested", False)

    df = pd.read_parquet(emb_path)
    emb_cols = _emb_columns(df)
    valid = _valid_embedding_mask(df, emb_cols)
    y_all = df["Label"].to_numpy(dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_ids = list_fold_indices(splits_dir)
    if not fold_ids:
        raise RuntimeError(f"No fold_*.json under {splits_dir}")

    rho_tag = _mit_fraction_dir_tag(mit_fraction)
    all_fold_results_by_seed: list[list[dict[str, Any]]] = []

    for seed in seeds:
        set_global_seed(seed)
        rng = np.random.default_rng(seed)
        fold_results: list[dict[str, Any]] = []

        for k in fold_ids:
            fold_path = splits_dir / f"fold_{k}.json"
            with open(fold_path) as f:
                spec = json.load(f)
            raw_tr = spec["train_indices"]
            raw_va = spec.get("val_indices") or []
            raw_te = spec["test_indices"]

            train_idx_full = _filter_indices(raw_tr, valid)
            val_idx = _filter_indices(raw_va, valid)
            test_idx = _filter_indices(raw_te, valid)

            if len(train_idx_full) < 2:
                raise RuntimeError(f"Fold {k}: insufficient train after embedding filter.")
            if strict_val and not val_idx:
                raise RuntimeError(f"Fold {k}: empty val_indices in nested split.")
            if not test_idx:
                raise RuntimeError(f"Fold {k}: empty test after embedding filter.")

            # Note: train_fold applies imbalance_mode sampling on the train split.
            # Here we only pass the full training pool from the nested split.
            train_idx = train_idx_full
            ld_stats = {
                "note": "train sampling handled by imbalance_mode inside train_fold",
                "mit_fraction": mit_fraction,
                "imbalance_mode": str(getattr(args, "imbalance_mode", "original")),
            }

            emb_variant = str(getattr(args, "embedding_variant", "raw_cgcnn_64"))
            X_raw = df[emb_cols].to_numpy(dtype=np.float64)
            if emb_variant == "raw_cgcnn_64":
                X_fold = None
                emb_cols_fold = emb_cols
                compression_meta = {"embedding_variant": emb_variant, "output_dim": len(emb_cols)}
            else:
                X_fold, emb_cols_fold, compression_meta = apply_embedding_variant(
                    X_raw,
                    y_all,
                    train_idx_full,
                    val_idx,
                    test_idx,
                    emb_variant,
                    seed=seed,
                    ae_max_epochs=int(getattr(args, "ae_max_epochs", 200)),
                    ae_patience=int(getattr(args, "ae_patience", 15)),
                )
            args._fold_compression_meta = compression_meta  # noqa: SLF001 — per-fold artifact hook

            fold_dir = out_root / rho_tag / f"seed_{seed}" / f"fold_{k}"
            # Resume support for architecture-selection grids: reuse complete folds.
            if getattr(args, "skip_existing_folds", False) and (fold_dir / "metrics.json").is_file():
                try:
                    prev_metrics = json.loads((fold_dir / "metrics.json").read_text())
                    prev_cfg = {}
                    if (fold_dir / "config.json").is_file():
                        prev_cfg = json.loads((fold_dir / "config.json").read_text())
                    result = {
                        "fold": k,
                        "seed": seed,
                        "mit_fraction": mit_fraction,
                        "head_type": args.head_type,
                        "skipped_existing_fold": True,
                        "best_val_pr_auc": prev_metrics.get("best_val_pr_auc"),
                        "validation_best_epoch": prev_metrics.get("validation_best_epoch"),
                        "epochs_trained": prev_metrics.get("epochs_trained"),
                        "test_metrics": {
                            "pr_auc": prev_metrics.get("pr_auc"),
                            "roc_auc": prev_metrics.get("roc_auc"),
                            "f1": prev_metrics.get("f1_at_tuned_threshold"),
                            "balanced_accuracy": prev_metrics.get("balanced_accuracy_at_tuned_threshold"),
                        },
                        "test_metrics_at_tuned_threshold": {
                            "f1": prev_metrics.get("f1_at_tuned_threshold"),
                            "balanced_accuracy": prev_metrics.get("balanced_accuracy_at_tuned_threshold"),
                            "precision": prev_metrics.get("precision_at_tuned_threshold"),
                            "recall": prev_metrics.get("recall_at_tuned_threshold"),
                            "mcc": prev_metrics.get("mcc_at_tuned_threshold"),
                        },
                        "n_train": (prev_cfg.get("train_counts") or {}).get("n_total"),
                        "n_val": (prev_cfg.get("val_counts") or {}).get("n_total"),
                        "n_test": (prev_cfg.get("test_counts") or {}).get("n_total"),
                    }
                    fold_results.append(result)
                    continue
                except Exception:
                    pass

            result = train_fold(
                k,
                df,
                emb_cols_fold,
                train_idx,
                val_idx,
                test_idx,
                args.head_type,
                args,
                device,
                rng,
                strict_val=strict_val,
                seed=seed,
                X_all=X_fold,
            )
            result["embedding_path"] = str(emb_path.relative_to(ROOT))
            result["split_json"] = str(fold_path.relative_to(ROOT))
            result["seed"] = seed
            result["mit_fraction"] = mit_fraction
            result["lowdata_train_stats"] = ld_stats
            result["train_indices_used"] = result.get("train_indices_used", train_idx)
            result["val_indices_used"] = val_idx
            result["test_indices_used"] = test_idx

            save_reviewer_fold_artifacts(
                fold_dir,
                result,
                df,
                y_all,
                train_idx,
                val_idx,
                test_idx,
                args,
                seed,
                mit_fraction,
                splits_dir,
                emb_path,
                fold_path,
            )
            fold_results.append(result)

        all_fold_results_by_seed.append(fold_results)

        per_seed_agg = _aggregate_folds(fold_results)
        seed_summary_dir = out_root / rho_tag / f"seed_{seed}"
        seed_summary_dir.mkdir(parents=True, exist_ok=True)
        with open(seed_summary_dir / "summary.json", "w") as f:
            json.dump(
                {
                    "head_type": args.head_type,
                    "seed": seed,
                    "mit_fraction": mit_fraction,
                    "aggregate_over_folds": per_seed_agg,
                    "folds": [_fold_summary_row(fr) for fr in fold_results],
                },
                f,
                indent=2,
            )

    per_seed_aggs = [_aggregate_folds(fr) for fr in all_fold_results_by_seed]
    summary_payload = {
        "head_type": args.head_type,
        "mit_fraction": mit_fraction,
        "seeds": seeds,
        "split_dir": str(splits_dir.relative_to(ROOT)),
        "output_root": str(out_root.relative_to(ROOT)),
        "aggregate_across_seeds": _aggregate_across_seeds(per_seed_aggs),
        "per_fold_across_seeds": _aggregate_per_fold_across_seeds(all_fold_results_by_seed),
    }
    with open(out_root / rho_tag / "summary.json", "w") as f:
        json.dump(summary_payload, f, indent=2)

    print(json.dumps({"head_type": args.head_type, "aggregate_across_seeds": summary_payload["aggregate_across_seeds"]}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Train post-hoc head on frozen CGCNN embeddings.")
    p.add_argument(
        "--head_type",
        type=str,
        required=True,
        choices=[
            "linear",
            "logistic_regression",
            "mlp",
            "quantum",
            "reup",
            "latent_mlp_vqc_matched",
            "latent_mlp_reup_matched",
            "proj_bottleneck_mlp",
            "latent_bottleneck_mlp",
            "random_fourier_logistic",
        ],
    )
    p.add_argument("--embedding_path", type=str, default="embeddings/cgcnn_binary_embeddings.parquet")
    p.add_argument(
        "--embedding_variant",
        type=str,
        default="raw_cgcnn_64",
        help=(
            "Embedding compression variant (per-fold fit on train only). "
            "See scripts/create_embedding_variants.py for names."
        ),
    )
    p.add_argument("--ae_max_epochs", type=int, default=200, help="Autoencoder variant: max epochs.")
    p.add_argument("--ae_patience", type=int, default=15, help="Autoencoder variant: early-stopping patience.")
    p.add_argument(
        "--splits_dir",
        type=str,
        default="splits/grouped_5fold",
        help="Legacy split directory (used if --split_dir is not set).",
    )
    p.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Split directory (e.g. splits/nested_grouped_5fold). Overrides --splits_dir when set.",
    )
    p.add_argument("--output_dir", type=str, default="results/heads")
    p.add_argument(
        "--mit_fraction",
        type=float,
        default=None,
        metavar="F",
        help=(
            "Low-data mode: on each fold, use this fraction of training positives (val/test unchanged). "
            "Runs --n_lowdata_seeds seeds; writes under {--lowdata_output_root}/{head_type}/mit_fraction_<F>/. "
            "Omit for default full-data training."
        ),
    )
    p.add_argument(
        "--lowdata_output_root",
        type=str,
        default="results/heads_lowdata",
        help="Directory under project root for low-data runs (mit_fraction mode only).",
    )
    p.add_argument(
        "--n_lowdata_seeds",
        type=int,
        default=DEFAULT_LOWDATA_SEEDS,
        help=f"Number of random seeds in mit_fraction mode (default {DEFAULT_LOWDATA_SEEDS}).",
    )
    p.add_argument(
        "--match_negatives_to_positives",
        action="store_true",
        help="With --mit_fraction: subsample negatives to match the number of training positives used.",
    )
    p.add_argument(
        "--imbalance_mode",
        type=str,
        default="original",
        choices=[
            "original",
            "matched_negatives",
            "preserve_ratio",
            "oversample_positive",
            "focal_loss",
        ],
        help=(
            "Class-imbalance ablation mode (train split only). "
            "original=subsample positives only; matched_negatives=balance by downsampling negatives; "
            "preserve_ratio=subsample both classes preserving ratio; oversample_positive=weighted sampler; "
            "focal_loss=use focal loss instead of BCE."
        ),
    )
    p.add_argument("--focal_alpha", type=float, default=0.25, help="Focal loss alpha (focal_loss mode).")
    p.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma (focal_loss mode).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--n_qubits", type=int, default=4)
    p.add_argument("--n_q_layers", type=int, default=2)
    p.add_argument("--reup_proj_dim", type=int, default=16, help="REUPHead: Linear input_dim→proj_dim width.")
    p.add_argument("--reup_n_qubits", type=int, default=4)
    p.add_argument("--reup_n_layers", type=int, default=4, help="REUPHead: number of data re-upload layers.")
    p.add_argument(
        "--reup_method",
        type=str,
        default="symmetrical",
        choices=["symmetrical", "asymmetrical"],
    )
    p.add_argument(
        "--reup_entanglement",
        type=_parse_bool_str,
        default=True,
        help="REUPHead: ring CZ entanglement between re-upload layers (true/false).",
    )
    p.add_argument(
        "--entanglement_type",
        type=str,
        default="ring",
        choices=["none", "linear", "ring", "full"],
        help="Quantum/REUP entanglement pattern (circuit).",
    )
    p.add_argument(
        "--encoding_type",
        type=str,
        default="ry",
        choices=["ry", "rx_ry", "rx_ry_rz"],
        help="Quantum/REUP encoding gates per feature.",
    )
    p.add_argument(
        "--measurement_type",
        type=str,
        default="all_z",
        choices=["single_z", "all_z", "z_with_classical_readout"],
        help="Quantum/REUP measurement readout type.",
    )
    p.add_argument(
        "--measurement_readout_hidden",
        type=int,
        default=8,
        help="For measurement_type=z_with_classical_readout: hidden size of small readout MLP.",
    )
    p.add_argument(
        "--quantum_backend",
        type=str,
        default="exact",
        choices=["exact", "shots", "noisy"],
        help="PennyLane simulation mode for quantum/reup heads (default: exact statevector).",
    )
    p.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Finite shots when --quantum_backend=shots (e.g. 8192, 4096, 1024, 512, 128).",
    )
    p.add_argument(
        "--noise_model",
        type=str,
        default="none",
        choices=["none", "depolarizing", "readout", "amplitude_damping"],
        help="Noise channel when --quantum_backend=noisy.",
    )
    p.add_argument(
        "--noise_prob",
        type=float,
        default=0.0,
        help="Noise strength for --quantum_backend=noisy.",
    )
    p.add_argument(
        "--fourier_dim",
        type=int,
        default=None,
        help="RandomFourierLogisticHead: number of RFF frequencies (default: match VQC param count).",
    )
    p.add_argument(
        "--rff_scale",
        type=float,
        default=1.0,
        help="RandomFourierLogisticHead: Gaussian scale for random frequencies.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (e.g. '42' or '42,43,44'). Enables reviewer_eval structured outputs.",
    )
    p.add_argument(
        "--strict_nested",
        action="store_true",
        help="Require val_indices in split JSON (no holdout carved from train).",
    )
    p.add_argument(
        "--skip_existing_folds",
        action="store_true",
        help="If fold metrics.json already exists, reuse it (architecture-selection resume).",
    )
    args = p.parse_args()

    if args.mit_fraction is not None and not (0.0 < args.mit_fraction <= 1.0):
        p.error("--mit_fraction must be in (0, 1].")
    if args.n_lowdata_seeds < 1:
        p.error("--n_lowdata_seeds must be >= 1.")
    # Back-compat: if old flag is used, map to the new mode unless user explicitly set otherwise.
    if getattr(args, "match_negatives_to_positives", False) and args.imbalance_mode == "original":
        args.imbalance_mode = "matched_negatives"

    # Reviewer-eval mode: nested splits + structured per-fold artifacts
    if args.seeds is not None:
        _run_reviewer_eval(args)
        return

    emb_path = (ROOT / args.embedding_path).resolve()
    splits_dir = _resolve_splits_dir(args)

    df = pd.read_parquet(emb_path)
    emb_cols = _emb_columns(df)
    valid = _valid_embedding_mask(df, emb_cols)
    y_all = df["Label"].to_numpy(dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mit_fraction is None:
        set_global_seed(args.seed)
        rng = np.random.default_rng(args.seed)

        out_root = (ROOT / args.output_dir).resolve()
        fold_results: list[dict[str, Any]] = []
        fold_ids = list_fold_indices(splits_dir)
        if not fold_ids:
            raise RuntimeError(f"No fold_*.json found under {splits_dir}")

        for k in fold_ids:
            fold_path = splits_dir / f"fold_{k}.json"
            with open(fold_path) as f:
                spec = json.load(f)
            raw_tr = spec["train_indices"]
            raw_va = spec.get("val_indices") or []
            raw_te = spec["test_indices"]

            train_idx = _filter_indices(raw_tr, valid)
            val_idx = _filter_indices(raw_va, valid)
            test_idx = _filter_indices(raw_te, valid)

            if len(train_idx) < 2 or not val_idx:
                raise RuntimeError(f"Fold {k}: insufficient train/val after embedding filter.")
            if not test_idx:
                raise RuntimeError(f"Fold {k}: empty test after embedding filter.")

            result = train_fold(
                k,
                df,
                emb_cols,
                train_idx,
                val_idx,
                test_idx,
                args.head_type,
                args,
                device,
                rng,
            )
            result["embedding_path"] = str(emb_path.relative_to(ROOT))
            result["split_json"] = str(fold_path.relative_to(ROOT))
            result["n_train_raw"] = len(raw_tr)
            result["n_train_with_embedding"] = len(train_idx)

            fold_dir = out_root / f"fold_{k}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            out_json = fold_dir / f"{args.head_type}_results.json"
            with open(out_json, "w") as f:
                json.dump(result, f, indent=2)

            fold_results.append(result)

        aggregate = _aggregate_folds(fold_results)
        _write_summary(out_root / "summary.json", args.head_type, fold_results, aggregate)
        print(json.dumps({"head_type": args.head_type, "aggregate": aggregate}, indent=2))
        return

    # Low-data: {lowdata_output_root}/{head_type}/mit_fraction_*/
    low_root = (ROOT / args.lowdata_output_root).resolve() / args.head_type / _mit_fraction_dir_tag(
        args.mit_fraction
    )
    low_root.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed + i for i in range(args.n_lowdata_seeds)]
    fold_results_by_seed: list[list[dict[str, Any]]] = []

    for seed in seeds:
        set_global_seed(seed)
        rng = np.random.default_rng(seed)

        fold_results: list[dict[str, Any]] = []
        fold_ids_ld = list_fold_indices(splits_dir)
        if not fold_ids_ld:
            raise RuntimeError(f"No fold_*.json found under {splits_dir}")
        for k in fold_ids_ld:
            fold_path = splits_dir / f"fold_{k}.json"
            with open(fold_path) as f:
                spec = json.load(f)
            raw_tr = spec["train_indices"]
            raw_va = spec.get("val_indices") or []
            raw_te = spec["test_indices"]

            train_idx_full = _filter_indices(raw_tr, valid)
            val_idx = _filter_indices(raw_va, valid)
            test_idx = _filter_indices(raw_te, valid)

            if len(train_idx_full) < 2 or not val_idx:
                raise RuntimeError(f"Fold {k}: insufficient train/val after embedding filter.")
            if not test_idx:
                raise RuntimeError(f"Fold {k}: empty test after embedding filter.")

            train_idx, ld_stats = subsample_train_for_lowdata(
                train_idx_full,
                y_all,
                args.mit_fraction,
                args.match_negatives_to_positives,
                rng,
            )
            if len(train_idx) < 2:
                raise RuntimeError(f"Fold {k}: low-data train subset too small (n={len(train_idx)}).")

            result = train_fold(
                k,
                df,
                emb_cols,
                train_idx,
                val_idx,
                test_idx,
                args.head_type,
                args,
                device,
                rng,
            )
            result["embedding_path"] = str(emb_path.relative_to(ROOT))
            result["split_json"] = str(fold_path.relative_to(ROOT))
            result["n_train_raw"] = len(raw_tr)
            result["n_train_with_embedding_full"] = len(train_idx_full)
            result["n_train_with_embedding"] = len(train_idx)
            result["seed"] = seed
            result["mit_fraction"] = args.mit_fraction
            result["match_negatives_to_positives"] = bool(args.match_negatives_to_positives)
            result["lowdata_train_stats"] = ld_stats

            seed_dir = low_root / f"seed_{seed}"
            fold_dir = seed_dir / f"fold_{k}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            out_json = fold_dir / f"{args.head_type}_results.json"
            with open(out_json, "w") as f:
                json.dump(result, f, indent=2)

            fold_results.append(result)

        fold_results_by_seed.append(fold_results)

    per_seed_aggs = [_aggregate_folds(fr) for fr in fold_results_by_seed]
    across_seeds = _aggregate_across_seeds(per_seed_aggs)
    per_fold_across = _aggregate_per_fold_across_seeds(fold_results_by_seed)

    summary_payload: dict[str, Any] = {
        "head_type": args.head_type,
        "mit_fraction": args.mit_fraction,
        "match_negatives_to_positives": bool(args.match_negatives_to_positives),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "output_root": str(low_root.relative_to(ROOT)),
        "per_seed": [
            {
                "seed": seeds[i],
                "aggregate_over_folds": per_seed_aggs[i],
                "folds": [_fold_summary_row(fr) for fr in fold_results_by_seed[i]],
            }
            for i in range(len(seeds))
        ],
        "aggregate_across_seeds": across_seeds,
        "per_fold_across_seeds": per_fold_across,
    }
    with open(low_root / "summary.json", "w") as f:
        json.dump(summary_payload, f, indent=2)

    print(
        json.dumps(
            {
                "head_type": args.head_type,
                "mit_fraction": args.mit_fraction,
                "aggregate_across_seeds": across_seeds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
