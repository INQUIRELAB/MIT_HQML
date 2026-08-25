#!/usr/bin/env python3
"""
Embedding compression / variant transforms for frozen CGCNN embeddings.

Variants are applied **per fold** with fitting on training data only (no leakage):
  - PCA: fit on train, transform train/val/test
  - Feature selection: mutual information on train labels only
  - Autoencoder: train on train, early-stop on val, encode all splits

Used by ``train_heads_on_embeddings.py`` via :func:`apply_embedding_variant`.

CLI (optional):
  python scripts/create_embedding_variants.py --write-catalog
  python scripts/create_embedding_variants.py --export-raw embeddings/variants/raw_cgcnn_64.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]

VARIANT_NAMES = [
    "raw_cgcnn_64",
    "pca_4",
    "pca_8",
    "pca_16",
    "pca_32",
    "feature_selected_4",
    "feature_selected_8",
    "feature_selected_16",
    "autoencoder_4",
    "autoencoder_8",
    "autoencoder_16",
]

_VARIANT_RE = re.compile(
    r"^(raw_cgcnn_64|pca_(\d+)|feature_selected_(\d+)|autoencoder_(\d+))$"
)


@dataclass(frozen=True)
class EmbeddingVariantSpec:
    name: str
    kind: str  # raw | pca | feature_selected | autoencoder
    n_components: int | None = None


def parse_embedding_variant(name: str) -> EmbeddingVariantSpec:
    key = str(name).strip()
    if key == "raw_cgcnn_64":
        return EmbeddingVariantSpec(name=key, kind="raw", n_components=64)
    m = _VARIANT_RE.match(key)
    if not m:
        raise ValueError(
            f"Unknown embedding_variant {name!r}. Expected one of: {', '.join(VARIANT_NAMES)}"
        )
    if m.group(1).startswith("pca_"):
        k = int(m.group(2))
        return EmbeddingVariantSpec(name=key, kind="pca", n_components=k)
    if m.group(1).startswith("feature_selected_"):
        k = int(m.group(3))
        return EmbeddingVariantSpec(name=key, kind="feature_selected", n_components=k)
    k = int(m.group(4))
    return EmbeddingVariantSpec(name=key, kind="autoencoder", n_components=k)


def emb_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("emb_")]
    if not cols:
        raise ValueError("No emb_* columns in embeddings parquet.")
    return sorted(cols, key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x)


def _subset(X: np.ndarray, idx: list[int]) -> np.ndarray:
    return np.asarray(X, dtype=np.float64)[np.asarray(idx, dtype=int)]


class _EmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


def apply_embedding_variant(
    X_raw: np.ndarray,
    y_all: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    variant_name: str,
    *,
    seed: int = 42,
    ae_max_epochs: int = 200,
    ae_patience: int = 15,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """
    Fit compression on ``train_idx`` only; transform train/val/test without leakage.

    Parameters
    ----------
    X_raw
        Full embedding matrix ``(n_samples, 64)`` in original row order.
    y_all
        Binary labels for feature selection (train indices only).
    train_idx, val_idx, test_idx
        Row indices into ``X_raw``.

    Returns
    -------
    X_all
        Transformed matrix ``(n_samples, d)`` with NaN rows unchanged only if input had NaN
        (caller should pre-filter valid rows).
    emb_cols
        Column names ``emb_0`` ... ``emb_{d-1}``.
    metadata
        Artifacts to save per fold (PCA variance, selected indices, AE history, etc.).
    """
    spec = parse_embedding_variant(variant_name)
    n_samples, input_dim = X_raw.shape
    meta: dict[str, Any] = {
        "embedding_variant": spec.name,
        "kind": spec.kind,
        "input_dim_raw": int(input_dim),
        "seed": int(seed),
    }

    X_tr = _subset(X_raw, train_idx)
    X_va = _subset(X_raw, val_idx) if val_idx else np.empty((0, input_dim))
    X_te = _subset(X_raw, test_idx)
    y_tr = y_all[np.asarray(train_idx, dtype=int)].astype(np.int64)

    if spec.kind == "raw":
        d = input_dim
        meta["output_dim"] = d
        return X_raw.astype(np.float32), emb_columns_from_dim(d), meta

    k = int(spec.n_components or 0)
    if k < 1 or k > input_dim:
        raise ValueError(f"{spec.name}: n_components={k} invalid for input_dim={input_dim}")

    if spec.kind == "pca":
        pca = PCA(n_components=k, random_state=seed)
        pca.fit(X_tr)
        X_out = np.full((n_samples, k), np.nan, dtype=np.float32)
        for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            if not idx:
                continue
            X_out[np.asarray(idx, dtype=int)] = pca.transform(_subset(X_raw, idx)).astype(np.float32)
        ev = pca.explained_variance_ratio_.tolist()
        meta.update(
            {
                "output_dim": k,
                "pca_explained_variance_ratio": ev,
                "pca_explained_variance_total": float(np.sum(pca.explained_variance_ratio_)),
                "pca_singular_values": pca.singular_values_.tolist(),
            }
        )
        return X_out, emb_columns_from_dim(k), meta

    if spec.kind == "feature_selected":
        scores = mutual_info_classif(X_tr, y_tr, random_state=seed)
        order = np.argsort(scores)[::-1]
        selected = order[:k].astype(int).tolist()
        X_out = np.full((n_samples, k), np.nan, dtype=np.float32)
        for idx in train_idx + val_idx + test_idx:
            row = int(idx)
            X_out[row] = X_raw[row, selected].astype(np.float32)
        meta.update(
            {
                "output_dim": k,
                "selected_feature_indices": selected,
                "selected_emb_columns": [f"emb_{i}" for i in selected],
                "mutual_information_scores": scores.tolist(),
            }
        )
        return X_out, emb_columns_from_dim(k), meta

    if spec.kind == "autoencoder":
        if not val_idx:
            raise ValueError(f"{spec.name}: autoencoder requires non-empty val_idx for early stopping.")
        X_out, meta_ae = _apply_autoencoder_variant(
            X_raw, train_idx, val_idx, test_idx, k, seed, ae_max_epochs, ae_patience
        )
        meta.update(meta_ae)
        return X_out, emb_columns_from_dim(k), meta

    raise ValueError(f"Unhandled variant kind {spec.kind!r}")


def _apply_autoencoder_variant(
    X_raw: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    latent_dim: int,
    seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train AE on train, early-stop on val, encode all splits."""
    n_samples, input_dim = X_raw.shape
    X_tr = _subset(X_raw, train_idx)
    X_va = _subset(X_raw, val_idx)
    X_te = _subset(X_raw, test_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = _EmbeddingAutoencoder(input_dim, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr.astype(np.float32))),
        batch_size=32,
        shuffle=True,
    )
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va.astype(np.float32))),
        batch_size=32,
        shuffle=False,
    )

    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    patience_left = patience
    history: list[dict[str, Any]] = []

    for epoch in range(max_epochs):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for (xb,) in tr_loader:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            _, x_hat = model(xb)
            loss = criterion(x_hat, xb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * xb.size(0)
            n_tr += xb.size(0)

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for (xb,) in va_loader:
                xb = xb.to(device)
                _, x_hat = model(xb)
                va_loss += float(loss.item()) * xb.size(0)
                n_va += xb.size(0)
        history.append(
            {
                "epoch": int(epoch),
                "train_mse": float(tr_loss / max(n_tr, 1)),
                "val_mse": float(va_loss / max(n_va, 1)),
            }
        )
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()

    def _encode_rows(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z, _ = model(torch.from_numpy(X.astype(np.float32)).to(device))
        return z.cpu().numpy().astype(np.float32)

    X_out = np.full((n_samples, latent_dim), np.nan, dtype=np.float32)
    X_out[np.asarray(train_idx, dtype=int)] = _encode_rows(X_tr)
    X_out[np.asarray(val_idx, dtype=int)] = _encode_rows(X_va)
    X_out[np.asarray(test_idx, dtype=int)] = _encode_rows(X_te)

    return X_out, {
        "output_dim": latent_dim,
        "autoencoder_training_history": history,
        "autoencoder_best_val_mse": float(best_val),
        "autoencoder_epochs_trained": len(history),
    }


def emb_columns_from_dim(d: int) -> list[str]:
    return [f"emb_{i}" for i in range(int(d))]


def write_variant_catalog(path: Path) -> None:
    catalog = {
        "variants": VARIANT_NAMES,
        "notes": "Per-fold transforms are applied in train_heads_on_embeddings.py to avoid leakage.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(catalog, f, indent=2)


def export_raw_variant(emb_path: Path, out_path: Path) -> None:
    df = pd.read_parquet(emb_path)
    cols = emb_columns(df)
    keep = cols + [c for c in ("Label", "Compound") if c in df.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_parquet(out_path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Embedding variant utilities and optional exports.")
    ap.add_argument("--embedding_path", type=str, default="embeddings/cgcnn_binary_embeddings.parquet")
    ap.add_argument("--write-catalog", action="store_true", help="Write embeddings/variants/variant_catalog.json")
    ap.add_argument(
        "--export-raw",
        type=str,
        default=None,
        metavar="PATH",
        help="Export raw_cgcnn_64 columns to parquet (static copy).",
    )
    args = ap.parse_args()

    if args.write_catalog:
        write_variant_catalog(ROOT / "embeddings" / "variants" / "variant_catalog.json")
        print("Wrote embeddings/variants/variant_catalog.json")

    if args.export_raw:
        export_raw_variant(
            (ROOT / args.embedding_path).resolve(),
            (ROOT / args.export_raw).resolve(),
        )
        print(f"Exported raw embeddings to {args.export_raw}")

    if not args.write_catalog and not args.export_raw:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
