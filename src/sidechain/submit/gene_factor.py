"""A per-gene multiplier on the pooled delta, applied before alpha and the knockdown pin.

`--gene-factor PATH` on `sidechain.submit.build` and `sidechain.eval.loco`: a parquet or CSV table with a
`gene` column (symbols on the build's axis) and a `factor` column (finite, non-negative). Every gene the
table does not name keeps 1.0, so the knob is exactly "multiply these genes' transferred log2FC by this
much" and nothing else; a gene named twice, a negative or non-finite factor, or a missing column is refused
before any work. The record written beside the build (`<stem>.gene_factor.json`, and `gene_factor` in
loco's summary) carries the file's sha256 and what matched the axis, so a run can be tied to its table.

Where it acts: the last thing before alpha, on the pooled vector every other knob has already shaped (the
neighbour arm's fusion, the basal slope), in both channels of a two-channel emission; the target's own gene
is pinned after it as before. It is not a shrinkage rule (`s`: a function of each fold change's own size
and spread) and not alpha (`a`: one constant for every gene). T102's trust shrink was its first arm:
private research/ideas/utr-layer-context-invariance.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def read_gene_factor_table(path: Path) -> pd.DataFrame:
    path = Path(path).expanduser()
    if not path.exists():
        raise SystemExit(f"--gene-factor: {path} does not exist")
    if path.suffix == ".parquet":
        tab = pd.read_parquet(path)
    else:
        tab = pd.read_csv(path)
    missing = [c for c in ("gene", "factor") if c not in tab.columns]
    if missing:
        raise SystemExit(f"--gene-factor: {path} lacks the column(s) {missing}; it needs `gene` and `factor`")
    tab = tab[["gene", "factor"]].copy()
    tab["gene"] = tab["gene"].astype(str)
    f = pd.to_numeric(tab["factor"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(f).all():
        raise SystemExit(f"--gene-factor: {path} carries a non-finite factor")
    if (f < 0).any():
        raise SystemExit(f"--gene-factor: {path} carries a negative factor; a multiplier on a log2FC is >= 0")
    if tab["gene"].duplicated().any():
        dup = tab["gene"][tab["gene"].duplicated()].iloc[0]
        raise SystemExit(f"--gene-factor: {path} names {dup!r} more than once")
    tab["factor"] = f
    return tab


def load_gene_factor(path: Path, axis: np.ndarray) -> tuple[np.ndarray, dict]:
    """The factor vector aligned to `axis` (1.0 where the table names no gene), and its record."""
    path = Path(path).expanduser()
    tab = read_gene_factor_table(path)
    pos = {g: i for i, g in enumerate(np.asarray(axis, dtype=object).astype(str))}
    factor = np.ones(len(axis), dtype=np.float64)
    hit = tab["gene"].map(pos)
    on_axis = hit.notna().to_numpy()
    factor[hit[on_axis].astype(int).to_numpy()] = tab["factor"].to_numpy()[on_axis]
    matched = factor[hit[on_axis].astype(int).to_numpy()] if on_axis.any() else np.array([])
    record = {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": int(len(tab)), "matched": int(on_axis.sum()), "unmatched_in_table": int((~on_axis).sum()),
        "axis_genes": int(len(axis)),
        "mean_on_matched": float(matched.mean()) if len(matched) else None,
        "min": float(factor.min()), "max": float(factor.max()),
        "below_one": int((factor < 1.0).sum()), "above_one": int((factor > 1.0).sum()),
        "zero": int((factor == 0.0).sum()),
    }
    if record["matched"] == 0:
        raise SystemExit(f"--gene-factor: no gene of {path} is on the build's axis ({len(axis)} genes)")
    return factor, record
