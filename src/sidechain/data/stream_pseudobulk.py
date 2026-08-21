"""Stream an h5ad in row blocks and accumulate per-label pseudobulks.

Why streaming: the corpora we pool from do not fit in memory. Replogle's K562
genome-wide file is 1.99M cells x 8,248 genes stored DENSE and gzip-chunked
(65 GB logical); the H1 files are 6-14 GB CSR. One row block is resident at a
time and only the per-label sums are kept, so memory is O(labels x genes).

Both on-disk layouts anndata writes are read directly with h5py -- a
`csr_matrix` group (data/indices/indptr) and a plain 2-D dataset -- because
anndata's backed mode cannot slice a dense gzip dataset efficiently.

What is accumulated per label, per gene:
  count_sum   sum of raw counts                      -> pseudobulk-sum profiles (PDS space)
  cpm_sum     sum over cells of counts / libsize * 1e6 -> arithmetic-mean CPM (the DE-test space)
  cpm_sq_sum  sum of squares of the same              -> per-gene variance of that mean
plus n_cells and libsize_sum per label.

    uv run python -m sidechain.data.stream_pseudobulk FILE.h5ad [FILE2 ...] \
        --label-col target_gene --keep targets.csv --control non-targeting \
        --out ~/data/sidechain/cache/<name>.npz [--control-once]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

try:  # anndata >= 0.11
    from anndata.io import read_elem
except ImportError:  # pragma: no cover - older anndata
    from anndata.experimental import read_elem


@dataclass
class PseudobulkSums:
    labels: list[str]
    genes: np.ndarray
    count_sum: np.ndarray      # (L, G) float64
    cpm_sum: np.ndarray        # (L, G) float64
    cpm_sq_sum: np.ndarray     # (L, G) float64
    n_cells: np.ndarray        # (L,) int64
    libsize_sum: np.ndarray    # (L,) float64
    sources: list[str] = field(default_factory=list)

    def mean_cpm(self) -> np.ndarray:
        return self.cpm_sum / np.maximum(self.n_cells, 1)[:, None]

    def var_cpm(self) -> np.ndarray:
        """Per-gene variance of per-cell CPM within a label (population form)."""
        n = np.maximum(self.n_cells, 1)[:, None]
        m = self.cpm_sum / n
        return np.maximum(self.cpm_sq_sum / n - m * m, 0.0)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path).expanduser(),
            labels=np.asarray(self.labels, dtype=object),
            genes=np.asarray(self.genes, dtype=object),
            count_sum=self.count_sum, cpm_sum=self.cpm_sum, cpm_sq_sum=self.cpm_sq_sum,
            n_cells=self.n_cells, libsize_sum=self.libsize_sum,
            sources=np.asarray(self.sources, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> PseudobulkSums:
        z = np.load(Path(path).expanduser(), allow_pickle=True)
        return cls(
            labels=[str(x) for x in z["labels"]], genes=z["genes"].astype(str),
            count_sum=z["count_sum"], cpm_sum=z["cpm_sum"], cpm_sq_sum=z["cpm_sq_sum"],
            n_cells=z["n_cells"], libsize_sum=z["libsize_sum"],
            sources=[str(x) for x in z["sources"]],
        )


def _obs_labels(f: h5py.File, label_col: str) -> np.ndarray:
    obs = read_elem(f["obs"])
    if label_col not in obs.columns:
        raise KeyError(f"{label_col!r} not in obs; columns: {list(obs.columns)[:20]}")
    return obs[label_col].astype(str).to_numpy()


def _var_names(f: h5py.File) -> np.ndarray:
    var = read_elem(f["var"])
    return var.index.astype(str).to_numpy()


def _iter_blocks(f: h5py.File, block_rows: int | None):
    """Yield (row0, block) with block a csr_matrix or dense ndarray of raw values."""
    X = f["X"]
    if isinstance(X, h5py.Group):  # csr_matrix group
        enc = X.attrs.get("encoding-type", b"")
        enc = enc.decode() if isinstance(enc, bytes) else str(enc)
        if enc != "csr_matrix":
            raise ValueError(f"unsupported sparse encoding {enc!r}; only csr_matrix is streamed")
        n, g = (int(v) for v in X.attrs["shape"])
        indptr = X["indptr"][:]
        rows = block_rows or 5000
        for r0 in range(0, n, rows):
            r1 = min(n, r0 + rows)
            s, e = int(indptr[r0]), int(indptr[r1])
            data = X["data"][s:e]
            idx = X["indices"][s:e]
            ip = indptr[r0 : r1 + 1] - s
            yield r0, sp.csr_matrix((data, idx, ip), shape=(r1 - r0, g))
    else:  # dense 2-D dataset; read whole chunk-rows so gzip chunks decompress once
        n, g = X.shape
        crows = X.chunks[0] if X.chunks else 4096
        rows = block_rows or max(crows, (crows * max(1, (64 << 20) // (crows * g * 4))))
        rows = (rows // crows) * crows or crows
        for r0 in range(0, n, rows):
            r1 = min(n, r0 + rows)
            yield r0, X[r0:r1, :]


def stream_pseudobulk(
    path: str | Path,
    label_col: str,
    keep: set[str] | None = None,
    *,
    block_rows: int | None = None,
    skip_labels: set[str] | None = None,
    progress: bool = False,
) -> PseudobulkSums:
    """One pass over `path`; sums for every label in `keep` (None = all labels)."""
    path = Path(path).expanduser()
    t0 = time.time()
    with h5py.File(path, "r") as f:
        labels_all = _obs_labels(f, label_col)
        genes = _var_names(f)
        wanted = sorted(set(labels_all) if keep is None else (set(labels_all) & set(keep)))
        if skip_labels:
            wanted = [w for w in wanted if w not in skip_labels]
        code_of = {lab: i for i, lab in enumerate(wanted)}
        L, G = len(wanted), len(genes)
        count_sum = np.zeros((L, G)); cpm_sum = np.zeros((L, G)); cpm_sq = np.zeros((L, G))
        n_cells = np.zeros(L, dtype=np.int64); lib_sum = np.zeros(L)
        codes_all = np.array([code_of.get(lab, -1) for lab in labels_all], dtype=np.int64)
        n_total = len(labels_all)
        for r0, block in _iter_blocks(f, block_rows):
            r1 = r0 + block.shape[0]
            codes = codes_all[r0:r1]
            sel = np.where(codes >= 0)[0]
            if sel.size == 0:
                continue
            sub = block[sel]
            c = codes[sel]
            lib = np.asarray(sub.sum(axis=1)).ravel().astype(np.float64)
            ok = lib > 0
            sub, c, lib = sub[ok], c[ok], lib[ok]
            if sub.shape[0] == 0:
                continue
            ind = sp.csr_matrix((np.ones(len(c)), (c, np.arange(len(c)))), shape=(L, len(c)))
            if sp.issparse(sub):
                sub = sp.csr_matrix(sub, dtype=np.float64)
                cpm = sp.diags(1e6 / lib) @ sub
                count_sum += (ind @ sub).toarray()
                cpm_sum += (ind @ cpm).toarray()
                cpm_sq += (ind @ cpm.multiply(cpm)).toarray()
            else:
                sub = np.asarray(sub, dtype=np.float64)
                cpm = sub * (1e6 / lib)[:, None]
                count_sum += ind @ sub
                cpm_sum += ind @ cpm
                cpm_sq += ind @ (cpm * cpm)
            np.add.at(n_cells, c, 1)
            np.add.at(lib_sum, c, lib)
            if progress:
                print(f"  {path.name}: {r1:>9}/{n_total} rows  {time.time() - t0:6.0f}s", flush=True)
    return PseudobulkSums(wanted, genes, count_sum, cpm_sum, cpm_sq, n_cells, lib_sum, [str(path)])


def merge(a: PseudobulkSums, b: PseudobulkSums) -> PseudobulkSums:
    """Add two passes over files sharing the same gene axis (label union)."""
    if not np.array_equal(a.genes, b.genes):
        raise ValueError("gene axes differ; cannot merge")
    labels = sorted(set(a.labels) | set(b.labels))
    L, G = len(labels), len(a.genes)
    out = PseudobulkSums(labels, a.genes, np.zeros((L, G)), np.zeros((L, G)), np.zeros((L, G)),
                         np.zeros(L, dtype=np.int64), np.zeros(L), a.sources + b.sources)
    for src in (a, b):
        pos = np.array([labels.index(lab) for lab in src.labels])
        out.count_sum[pos] += src.count_sum; out.cpm_sum[pos] += src.cpm_sum
        out.cpm_sq_sum[pos] += src.cpm_sq_sum; out.n_cells[pos] += src.n_cells
        out.libsize_sum[pos] += src.libsize_sum
    return out


def _read_keep(path: str) -> set[str]:
    df = pd.read_csv(path)
    col = "target_gene" if "target_gene" in df.columns else df.columns[0]
    return set(df[col].astype(str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--keep", help="CSV of labels to keep (column target_gene or first column); default all")
    ap.add_argument("--control", help="control label, always kept")
    ap.add_argument("--control-once", action="store_true",
                    help="take the control label from the FIRST file only (the 2025 val/test files repeat training's controls)")
    ap.add_argument("--block-rows", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    keep = _read_keep(args.keep) if args.keep else None
    if keep is not None and args.control:
        keep = keep | {args.control}
    total = None
    for i, fpath in enumerate(args.files):
        skip = {args.control} if (args.control_once and i > 0 and args.control) else None
        print(f"== {fpath}", flush=True)
        part = stream_pseudobulk(fpath, args.label_col, keep, block_rows=args.block_rows,
                                 skip_labels=skip, progress=True)
        total = part if total is None else merge(total, part)
    assert total is not None
    total.save(args.out)
    meta = {"labels": len(total.labels), "genes": len(total.genes),
            "cells": int(total.n_cells.sum()), "sources": total.sources}
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
