"""A per-target multiplier on the amplitude, alpha_t = alpha * factor[t], read from a table.

    --alpha-table PATH      on sidechain.submit.build and sidechain.eval.loco

Reads a parquet or CSV with `target` and `factor` columns (a perturbation it does not name keeps 1.0);
writes nothing, and returns the record the caller writes (`<stem>.alpha_table.json`, loco's summary).
Notes: private/code-notes/src/sidechain/submit/alpha_table.py.md
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def read_alpha_table(path: Path) -> pd.DataFrame:
    path = Path(path).expanduser()
    if not path.exists():
        raise SystemExit(f"--alpha-table: {path} does not exist")
    if path.suffix == ".parquet":
        tab = pd.read_parquet(path)
    else:
        tab = pd.read_csv(path)
    missing = [c for c in ("target", "factor") if c not in tab.columns]
    if missing:
        raise SystemExit(f"--alpha-table: {path} lacks the column(s) {missing}; it needs `target` and `factor`")
    tab = tab[["target", "factor"]].copy()
    tab["target"] = tab["target"].astype(str)
    f = pd.to_numeric(tab["factor"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(f).all():
        raise SystemExit(f"--alpha-table: {path} carries a non-finite factor")
    if (f < 0).any():
        raise SystemExit(f"--alpha-table: {path} carries a negative factor; a multiplier on alpha is >= 0")
    if tab["target"].duplicated().any():
        dup = tab["target"][tab["target"].duplicated()].iloc[0]
        raise SystemExit(f"--alpha-table: {path} names {dup!r} more than once")
    tab["factor"] = f
    return tab


def load_alpha_table(path: Path, perts: list[str]) -> tuple[dict[str, float], dict]:
    """The per-target factor for every perturbation in `perts` (1.0 where the table names none), and its record."""
    path = Path(path).expanduser()
    tab = read_alpha_table(path)
    named = dict(zip(tab["target"].tolist(), tab["factor"].tolist()))
    perts = [str(p) for p in perts]
    factor = {p: float(named.get(p, 1.0)) for p in perts}
    matched = np.array([named[p] for p in perts if p in named], dtype=float)
    vals = np.array(list(factor.values()), dtype=float)
    record = {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": int(len(tab)), "matched": int(len(matched)),
        "unmatched_in_table": int(sum(1 for t in named if t not in factor)),
        "perturbations": int(len(perts)),
        "mean_on_matched": float(matched.mean()) if len(matched) else None,
        "min": float(vals.min()) if len(vals) else None, "max": float(vals.max()) if len(vals) else None,
        "below_one": int((vals < 1.0).sum()), "above_one": int((vals > 1.0).sum()), "zero": int((vals == 0.0).sum()),
    }
    if record["matched"] == 0:
        raise SystemExit(f"--alpha-table: no target of {path} is a perturbation of this build ({len(perts)} perturbations)")
    return factor, record
