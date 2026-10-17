"""A per-target multiplier on the amplitude: alpha_t = alpha * factor[t], read from a table.

`--alpha-table PATH` on `sidechain.submit.build` and `sidechain.eval.loco`: a parquet or CSV table with a
`target` column (perturbation labels as the build names them) and a `factor` column (finite, non-negative).
Every perturbation the table does not name keeps 1.0, so the knob is exactly "scale these knockdowns'
transferred log2FC by this much on top of alpha" and nothing else; a target named twice, a negative or
non-finite factor, or a missing column is refused before any work. The record written beside the build
(`<stem>.alpha_table.json`, and `alpha_table` in loco's summary) carries the file's sha256 and what matched
the perturbations, so a run can be tied to its table.

Where it acts: with alpha, on the pooled vector every other knob has already shaped (the neighbour arm's
fusion, the basal slope, the per-gene factor), in both channels of a two-channel emission; the target's own
gene is pinned after it as before. It is not alpha (`a`: one constant for every knockdown) and not the
per-gene factor (`--gene-factor`: one multiplier per response gene, the same for every knockdown): it is one
multiplier per knockdown, the same for every gene. T1's fitness-size prior was its first arm:
private research/ideas/knockdown-fitness-size-prior.md.
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
