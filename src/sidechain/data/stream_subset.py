"""Extract the cells of chosen labels from a large h5ad into a small CSR h5ad.

The corpora we hold are stored dense + gzip (65 GB logical for K562 genome-wide;
5-10 GB for the others), so "give me the cells of these 100 perturbations plus
10,000 controls" cannot go through anndata's backed indexing. One streaming pass
over row blocks (the same reader as `stream_pseudobulk`) keeps the wanted rows
as sparse blocks and writes an ordinary h5ad at the end -- raw counts, obs and
var carried over, so the result is a valid `real` side for cell-eval2.

    uv run python -m sidechain.data.stream_subset FILE.h5ad --label-col perturbation \
        --keep perts.csv --control control --max-per-label 400 --max-control 10000 \
        --seed 0 --out ~/data/sidechain/cache/vcc2026/hepg2_real_subset.h5ad
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sidechain.data.stream_pseudobulk import _iter_blocks, _read_keep, read_elem


def extract_cells(
    path: str | Path,
    label_col: str,
    keep: set[str],
    *,
    control: str | None = None,
    max_per_label: int | None = None,
    max_control: int | None = None,
    seed: int = 0,
    block_rows: int | None = None,
    relabel_control: str | None = None,
) -> ad.AnnData:
    """Cells of `keep` (+ control), at most `max_per_label` each, chosen uniformly at random."""
    path = Path(path).expanduser()
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        labels = obs[label_col].astype(str).to_numpy()
        wanted = set(keep) | ({control} if control else set())
        chosen = np.zeros(len(obs), dtype=bool)
        for lab in wanted:
            idx = np.where(labels == lab)[0]
            cap = max_control if (control and lab == control) else max_per_label
            if cap is not None and idx.size > cap:
                idx = np.sort(rng.choice(idx, size=cap, replace=False))
            chosen[idx] = True
        blocks = []
        for r0, block in _iter_blocks(f, block_rows):
            r1 = r0 + block.shape[0]
            sel = np.where(chosen[r0:r1])[0]
            if sel.size == 0:
                continue
            sub = block[sel]
            blocks.append(sp.csr_matrix(sub, dtype=np.float32))
    X = sp.vstack(blocks, format="csr")
    sub_obs = obs.iloc[np.where(chosen)[0]].copy()
    # A categorical column keeps every level of the SOURCE file; downstream groupers
    # (pdex, cell-eval2) then see thousands of empty groups and divide by zero.
    for col in sub_obs.columns:
        if isinstance(sub_obs[col].dtype, pd.CategoricalDtype):
            sub_obs[col] = sub_obs[col].cat.remove_unused_categories()
    if control and relabel_control and relabel_control != control:
        # cell-eval2's competition rule hashes the control LABEL ('non-targeting');
        # a line whose controls are called 'control' would otherwise only ever
        # yield a "diagnostic" bundle. Relabel at extraction, once.
        col = sub_obs[label_col].astype(str)
        sub_obs[label_col] = col.where(col != control, relabel_control)
    out = ad.AnnData(X=X, obs=sub_obs, var=var.copy())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--keep", required=True, help="CSV of labels (column target_gene or first column)")
    ap.add_argument("--control")
    ap.add_argument("--max-per-label", type=int)
    ap.add_argument("--max-control", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-rows", type=int)
    ap.add_argument("--relabel-control", help="rename the control label on output, e.g. non-targeting")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    a = extract_cells(args.file, args.label_col, _read_keep(args.keep), control=args.control,
                      max_per_label=args.max_per_label, max_control=args.max_control,
                      seed=args.seed, block_rows=args.block_rows, relabel_control=args.relabel_control)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(out, compression="gzip")
    counts = a.obs[args.label_col].astype(str).value_counts()
    print(json.dumps({"out": str(out), "cells": int(a.n_obs), "genes": int(a.n_vars),
                      "labels": int(counts.size), "cells_per_label_min": int(counts.min()),
                      "cells_per_label_median": float(counts.median())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
