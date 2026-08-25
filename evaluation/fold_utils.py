"""Discover grouped CV fold indices from splits directory."""

from __future__ import annotations

import re
from pathlib import Path


def list_fold_indices(splits_dir: Path) -> list[int]:
    """Return sorted fold ids (integers) for each ``fold_{k}.json`` present."""
    splits_dir = Path(splits_dir)
    if not splits_dir.is_dir():
        return []
    ids: list[int] = []
    for p in splits_dir.glob("fold_*.json"):
        m = re.match(r"^fold_(\d+)\.json$", p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)
