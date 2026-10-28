"""Stream h5ads in row blocks and accumulate per-label pseudobulk sums.

    uv run python -m sidechain.data.stream_pseudobulk FILE.h5ad [FILE2 ...] \\
        --label-col target_gene --keep targets.csv --control non-targeting \\
        --out ~/data/sidechain/cache/<name>.npz [--control-once]

Per label and gene: count_sum (raw counts), cpm_sum (counts / libsize * 1e6) and cpm_sq_sum; per label n_cells and
libsize_sum. Reads each h5ad's X (a csr_matrix group or a dense 2-D dataset) with h5py, one block at a time;
writes the `PseudobulkSums` .npz and prints a JSON summary.
Notes: private/code-notes/src/sidechain/data/stream_pseudobulk.py.md
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import time
import zipfile
from collections.abc import Sequence
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

    def n_eff(self, i: int) -> np.ndarray:
        """Per gene of label row `i`, `cpm_sum^2 / cpm_sq_sum` capped at the row's cell count: a floor on the cells
        that detected the gene, not a detection count."""
        s1, s2 = self.cpm_sum[i], self.cpm_sq_sum[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.where(s2 > 0, s1 * s1 / np.maximum(s2, np.finfo(np.float64).tiny), 0.0)
        # Cannot exceed the arm's own cell count; float error at the equality case
        # (every detecting cell identical) would otherwise put it a hair above.
        return np.minimum(e, float(max(int(self.n_cells[i]), 0)))

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

    @staticmethod
    def peek(path: str | Path) -> tuple[list[str], list[str]]:
        """Read only the labels and genes, not the matrices."""
        with zipfile.ZipFile(Path(path).expanduser()) as z:
            def small(name):
                with z.open(name) as fh:
                    return np.load(io.BytesIO(fh.read()), allow_pickle=True)
            return [str(x) for x in small("labels.npy")], [str(x) for x in small("genes.npy")]

    @classmethod
    def load_subset(cls, path: str | Path, labels: Sequence[str],
                    genes: Sequence[str]) -> PseudobulkSums:
        """Load only these labels and genes, reading each member once; `count_sum` is zero-filled, not read."""
        path = Path(path).expanduser()
        with zipfile.ZipFile(path) as z:
            def small(name):
                with z.open(name) as fh:
                    return np.load(io.BytesIO(fh.read()), allow_pickle=True)
            labels_all = [str(x) for x in small("labels.npy")]
            genes_all = small("genes.npy").astype(str)
            n_cells_all = small("n_cells.npy")
            libsize_all = small("libsize_sum.npy")

        if len(set(genes_all)) != len(genes_all):
            raise ValueError(f"{path.name}: duplicate gene names; a name must identify one "
                             "column, or load_subset would silently pick the last")
        gpos = {g: i for i, g in enumerate(genes_all)}
        missing_g = [g for g in genes if g not in gpos]
        if missing_g:
            raise KeyError(f"{path.name}: {len(missing_g)} gene(s) not in this artifact, "
                           f"first few {missing_g[:5]}")
        cols = np.array([gpos[g] for g in genes], dtype=np.int64)

        lpos = {lab: i for i, lab in enumerate(labels_all)}
        missing_l = [x for x in labels if x not in lpos]
        if missing_l:
            raise KeyError(f"{path.name}: {len(missing_l)} label(s) not in this artifact, "
                           f"first few {missing_l[:5]}")
        rows = np.array([lpos[x] for x in labels], dtype=np.int64)

        cpm_sum = _read_npz_rows(path, "cpm_sum.npy", rows, cols)
        cpm_sq_sum = _read_npz_rows(path, "cpm_sq_sum.npy", rows, cols)
        return cls(
            labels=list(labels), genes=np.asarray(list(genes), dtype=object),
            count_sum=np.zeros_like(cpm_sum), cpm_sum=cpm_sum, cpm_sq_sum=cpm_sq_sum,
            n_cells=np.asarray(n_cells_all)[rows],
            libsize_sum=np.asarray(libsize_all)[rows],
            sources=["subset"],
        )


def _npy_header(fh) -> tuple[np.dtype, tuple, bool]:
    """Parse a .npy header off a stream. Returns (dtype, shape, fortran_order)."""
    if fh.read(6) != b"\x93NUMPY":
        raise ValueError("not a .npy stream")
    major = fh.read(1)[0]
    fh.read(1)                                     # minor, unused
    width = 2 if major == 1 else 4
    hlen = int.from_bytes(fh.read(width), "little")
    meta = ast.literal_eval(fh.read(hlen).decode("latin1"))
    return np.dtype(meta["descr"]), meta["shape"], meta["fortran_order"]


def _read_npz_rows(npz_path: Path, member: str, rows: np.ndarray,
                   cols: np.ndarray) -> np.ndarray:
    """Return ``rows x cols`` of one (L, G) npz member, in the order `rows` gives, reading it once."""
    order = np.argsort(rows)
    want = np.asarray(rows)[order]
    out = np.empty((len(rows), len(cols)), dtype=np.float64)
    with zipfile.ZipFile(npz_path) as z, z.open(member) as fh:
        dt, shape, fortran = _npy_header(fh)
        if fortran:
            raise ValueError(f"{member}: fortran-order, this reader assumes C order")
        if len(shape) != 2:
            raise ValueError(f"{member}: expected a 2-D member, got shape {shape}")
        rowbytes = shape[1] * dt.itemsize
        reader = io.BufferedReader(fh, buffer_size=1 << 22)
        pos = 0
        for slot, target_row in enumerate(want):
            while pos < target_row:
                if len(reader.read(rowbytes)) != rowbytes:
                    raise EOFError(f"{member}: short read seeking row {target_row}")
                pos += 1
            raw = reader.read(rowbytes)
            if len(raw) != rowbytes:
                raise EOFError(f"{member}: short read at row {target_row}")
            out[order[slot]] = np.frombuffer(raw, dtype=dt)[cols]
            pos += 1
    return out


def iter_npz_row_blocks(npz_path: str | Path, members: Sequence[str], *, block_rows: int = 128):
    """Yield ``(row0, {member: block})`` over every row of several same-shape (L, G) members in lockstep.

    `block_rows` rows at a time; each member is read once, start to finish, on its own handle.
    """
    npz_path = Path(npz_path).expanduser()
    if block_rows < 1:
        raise ValueError(f"block_rows must be >= 1, got {block_rows}")
    handles, readers, meta = [], {}, {}
    try:
        for member in members:
            z = zipfile.ZipFile(npz_path)
            fh = z.open(member)
            handles += [z, fh]
            dt, shape, fortran = _npy_header(fh)
            if fortran:
                raise ValueError(f"{member}: fortran-order, this reader assumes C order")
            if len(shape) != 2:
                raise ValueError(f"{member}: expected a 2-D member, got shape {shape}")
            readers[member] = io.BufferedReader(fh, buffer_size=1 << 22)
            meta[member] = (dt, tuple(int(s) for s in shape))
        shapes = {m: s for m, (_, s) in meta.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"members differ in shape: {shapes}")
        n_rows, n_cols = next(iter(shapes.values()))
        for r0 in range(0, n_rows, block_rows):
            k = min(block_rows, n_rows - r0)
            out = {}
            for member in members:
                dt = meta[member][0]
                raw = readers[member].read(k * n_cols * dt.itemsize)
                if len(raw) != k * n_cols * dt.itemsize:
                    raise EOFError(f"{member}: short read at row {r0}")
                out[member] = np.frombuffer(raw, dtype=dt).reshape(k, n_cols)
            yield r0, out
    finally:
        for h in reversed(handles):
            h.close()


def _obs_labels(f: h5py.File, label_col: str) -> np.ndarray:
    obs = read_elem(f["obs"])
    if label_col not in obs.columns:
        raise KeyError(f"{label_col!r} not in obs; columns: {list(obs.columns)[:20]}")
    return obs[label_col].astype(str).to_numpy()


def _var_names(f: h5py.File) -> np.ndarray:
    var = read_elem(f["var"])
    return var.index.astype(str).to_numpy()


def _iter_blocks(f: h5py.File, block_rows: int | None, x_path: str = "X"):
    """Yield (row0, block) with block a csr_matrix or dense ndarray of raw values.

    `x_path` selects which matrix streams: "X" for every corpus that stores raw
    counts there, "layers/counts" for files whose X was normalised in place and
    whose raw counts live in a layer (perturbench's processed h5ads).
    """
    X = f[x_path]
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


def stream_pseudobulk_file(
    f: h5py.File,
    label_col: str,
    keep: set[str] | None = None,
    *,
    block_rows: int | None = None,
    skip_labels: set[str] | None = None,
    progress: bool = False,
    labels_all: np.ndarray | None = None,
    genes: np.ndarray | None = None,
    source: str = "",
    x_path: str = "X",
) -> PseudobulkSums:
    """The accumulator on an open h5 handle, a local file or a lamindb `Artifact.open()`.

    `labels_all` / `genes` override the file's obs column / var index; the caller proves them row-aligned.
    """
    t0 = time.time()
    name = source or str(getattr(f, "filename", "<remote>"))
    display = Path(name).name if "/" in name else name
    if labels_all is None:
        labels_all = _obs_labels(f, label_col)
    if genes is None:
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
    for r0, block in _iter_blocks(f, block_rows, x_path):
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
            print(f"  {display}: {r1:>9}/{n_total} rows  {time.time() - t0:6.0f}s", flush=True)
    return PseudobulkSums(wanted, np.asarray(genes), count_sum, cpm_sum, cpm_sq, n_cells,
                          lib_sum, [name])


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
    with h5py.File(path, "r") as f:
        return stream_pseudobulk_file(
            f, label_col, keep, block_rows=block_rows, skip_labels=skip_labels,
            progress=progress, source=str(path))


def merge(a: PseudobulkSums, b: PseudobulkSums) -> PseudobulkSums:
    """Add two passes over files sharing the same gene axis (label union)."""
    if not np.array_equal(a.genes, b.genes):
        raise ValueError("gene axes differ; cannot merge")
    labels = sorted(set(a.labels) | set(b.labels))
    L, G = len(labels), len(a.genes)
    out = PseudobulkSums(labels, a.genes, np.zeros((L, G)), np.zeros((L, G)), np.zeros((L, G)),
                         np.zeros(L, dtype=np.int64), np.zeros(L), a.sources + b.sources)
    for src in (a, b):
        pos = np.array([labels.index(lab) for lab in src.labels], dtype=np.int64)
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
                    help="take the control label from the first file only")
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
    # the on-target knockdown check, recorded in the summary rather than raised
    if args.control:
        from sidechain.ingest.checks import require_on_target_knockdown
        try:
            meta["on_target"] = require_on_target_knockdown(total, args.control)
        except ValueError as exc:
            meta["on_target"] = {"status": "FAILED", "detail": str(exc)}
            print(f"!! {exc}", flush=True)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
