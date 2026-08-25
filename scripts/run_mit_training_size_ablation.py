#!/usr/bin/env python3
"""
Orchestrate training-size ablation on the real MIT binary dataset (frozen CGCNN
embeddings + post-hoc heads). Subsamples **positive** training labels per
``mit_fraction``; leaves **all negatives** in train unless you pass
``--match_negatives_to_positives``.

Example (full grid, 5 seeds each):
  python scripts/run_mit_training_size_ablation.py

Quick smoke (1 seed, two fractions, linear only):
  python scripts/run_mit_training_size_ablation.py \\
    --fractions 0.25,1.0 --heads linear --n_lowdata_seeds 1

Then aggregate:
  python scripts/aggregate_mit_training_size_ablation.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FRACTIONS = [
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.333333,
    0.5,
    0.75,
    1.0,
]


def _mit_fraction_dir_tag(mit_fraction: float) -> str:
    x = round(float(mit_fraction), 6)
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return f"mit_fraction_{s}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fractions",
        type=str,
        default=",".join(str(f) for f in DEFAULT_FRACTIONS),
        help="Comma-separated mit_fraction values (0,1].",
    )
    ap.add_argument("--heads", type=str, default="linear,mlp,quantum")
    ap.add_argument("--n_lowdata_seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--lowdata_output_root",
        type=str,
        default="results/mit_training_size_ablation",
    )
    ap.add_argument("--embedding_path", type=str, default="embeddings/cgcnn_binary_embeddings.parquet")
    ap.add_argument("--splits_dir", type=str, default="splits/grouped_5fold")
    ap.add_argument(
        "--match_negatives_to_positives",
        action="store_true",
        help="Balanced train bags (not recommended for imbalance ablation).",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip if summary.json already exists for that head/fraction.",
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    fracs = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]

    for f in fracs:
        if not (0.0 < f <= 1.0):
            print(f"ERROR: invalid fraction {f}", file=sys.stderr)
            return 1

    emb = ROOT / args.embedding_path
    if not emb.is_file():
        print(f"ERROR: missing embeddings {emb}", file=sys.stderr)
        print("Run: python scripts/export_cgcnn_embeddings.py", file=sys.stderr)
        return 1
    splits = ROOT / args.splits_dir
    if not any(splits.glob("fold_*.json")):
        print(f"ERROR: no fold_*.json under {splits}", file=sys.stderr)
        print("Run: python scripts/create_grouped_splits.py", file=sys.stderr)
        return 1

    out_root = ROOT / args.lowdata_output_root
    n_ok = 0
    n_skip = 0
    for frac in fracs:
        tag = _mit_fraction_dir_tag(frac)
        for head in heads:
            summ = out_root / head / tag / "summary.json"
            if args.skip_existing and summ.is_file():
                print(f"skip existing {summ.relative_to(ROOT)}")
                n_skip += 1
                continue
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "train_heads_on_embeddings.py"),
                "--head_type",
                head,
                "--mit_fraction",
                str(frac),
                "--lowdata_output_root",
                args.lowdata_output_root,
                "--n_lowdata_seeds",
                str(args.n_lowdata_seeds),
                "--seed",
                str(args.seed),
                "--embedding_path",
                args.embedding_path,
                "--splits_dir",
                args.splits_dir,
            ]
            if args.match_negatives_to_positives:
                cmd.append("--match_negatives_to_positives")
            print("RUN:", " ".join(cmd))
            if args.dry_run:
                n_ok += 1
                continue
            r = subprocess.run(cmd, cwd=str(ROOT))
            if r.returncode != 0:
                print(f"ERROR: failed head={head} fraction={frac}", file=sys.stderr)
                return r.returncode
            n_ok += 1

    print(f"Done. ran={n_ok} skipped={n_skip}")
    print("Aggregate: python scripts/aggregate_mit_training_size_ablation.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
