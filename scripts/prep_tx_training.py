#!/usr/bin/env python
"""Prepare our LOCO cell-level h5ads for ``state tx train`` / ``cell_load``.

``cell_load`` reads cells straight out of an h5ad, and it needs four obs columns whose names and
values it is told in the config: a perturbation column, a **cell type** column (which is what its
``zeroshot`` split holds out, and is therefore how the 2026 task's held-out-cell-line shape is
expressed), a batch column, and an exact control label.

Our fold files were built for a different consumer and are missing one of those and inconsistent
on another:

* **no cell-type column at all.** ``loco_hct116_real`` and ``loco_hek293t_real`` carry ``batch`` and
  ``sample``; ``loco_k562gwps_union_real`` carries ``batch`` and ``cell_line``. Without one, every
  cell looks like the same context and the zeroshot split cannot be written.
* **the control label differs by corpus** — X-Atlas writes ``Non-Targeting``, K562-gwps writes
  ``non-targeting``. ``cell_load`` matches ``control_pert`` by exact string, so mixing them in one
  dataset silently reclassifies one corpus's control arm as a perturbation. This is the same class
  of error as the Feng 2026 undercount: the control definition is a fact about the experiment, and
  it is declared here rather than inferred from whatever strings happen to be in the column.

This script does the smallest thing that fixes both: it **copies** the file and adds/normalises obs
columns with h5py. By default it does not rewrite X, so it costs a disk copy rather than an
AnnData round-trip.

``--log1p`` additionally rewrites X as **scanpy's shifted logarithm** --
``normalize_total(target_sum=None)`` then ``log1p`` -- and writes ``uns/log1p`` so ``cell_load``
autodetects it. Three things about that recipe are deliberate:

* **The size factor is the dataset's median raw count depth**, which is what
  ``target_sum=None`` means, not CP10k and not CPM. Fixed 1e4/1e6 targets inflate the
  overdispersion relative to what single-cell data actually shows; the median-depth factor is
  scanpy's own default and the one the single-cell best-practices book argues for.
* **It is applied because the model's objective is a distance, not a likelihood.** ``state tx``
  trains against ``geomloss``'s energy distance between two cell sets. A Euclidean distance on
  raw counts is dominated by the handful of highest-expressed genes -- which is the pathology
  the shifted logarithm exists to remove. Raw counts belong to models with a count likelihood
  (scVI and friends), and this is not one. Arc's own ``state tx`` training file,
  ``arcinstitute/State-Replogle-Filtered/replogle_concat.h5ad``, holds log-transformed values.
* **It runs streamed, never through AnnData.** The fold files are 1.5-2.4 GB of CSR on a 17 GB
  Mac. Both passes read ``X/data`` in blocks of nonzeros and write it back in place; the
  sparsity structure never changes, because scaling a row and ``log1p`` both map 0 to 0.

``uns/log1p`` is also the idempotence guard: a file that already carries it is left alone rather
than logged twice, which would be silent and unrecoverable.

What it deliberately does NOT do: harmonise gene axes. The two X-Atlas files already share 38,584
genes; K562-gwps is a different 8,248 and their three-way intersection is 7,976 (40.7% of the 2026
challenge axis). Mixing all three would train the model to emit 41% of the axis four of the six
board metrics read. That trade is a modelling decision, recorded in
private/research/ideas/gene-embedding-arm.md, not something a prep script should make silently.

``sidechain.data.union_axis`` is the tool that makes it, when it is made: it projects every corpus
onto one declared axis and records which genes each one measured, so nothing is intersected away.
Run it first, then this script with ``--in-place`` on what it wrote.

Usage::

    python scripts/prep_tx_training.py \
        --src ~/data/sidechain/cache/vcc2026/loco_hct116_real.h5ad  --cell-type HCT116 \
        --src ~/data/sidechain/cache/vcc2026/loco_hek293t_real.h5ad --cell-type HEK293T \
        --pert-col gene_target --control-label Non-Targeting \
        --log1p \
        --out-dir ~/data/sidechain/derived/tx-train-xatlas-log1p
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def write_categorical(obs, name: str, values: list[str]) -> None:
    """Write one obs column in AnnData's categorical encoding."""
    import h5py

    cats = sorted(set(values))
    lookup = {c: i for i, c in enumerate(cats)}
    codes = np.asarray([lookup[v] for v in values], dtype=np.int8 if len(cats) < 128 else np.int32)
    if name in obs:
        del obs[name]
    g = obs.create_group(name)
    g.attrs["encoding-type"] = "categorical"
    g.attrs["encoding-version"] = "0.2.0"
    g.attrs["ordered"] = False
    g.create_dataset("categories", data=np.array(cats, dtype=h5py.string_dtype()))
    g.create_dataset("codes", data=codes)


def csr_group(h5):
    """The `X` group, checked to be CSR with a float payload before anything writes to it."""
    X = h5["X"]
    enc = X.attrs.get("encoding-type")
    if enc != "csr_matrix":
        raise SystemExit(
            f"X is {enc!r}, not 'csr_matrix'. The streamed rewrite walks indptr/data and has "
            "nothing to walk on a dense or CSC matrix; convert first, or drop --log1p."
        )
    if X["data"].dtype.kind != "f":
        raise SystemExit(
            f"X/data is {X['data'].dtype}, an integer type. Writing the shifted logarithm back "
            "into it would truncate every value to 0 or 1 with no error. Rewrite X as float32 "
            "first."
        )
    return X


def row_totals(h5, block_nnz: int = 8_000_000) -> np.ndarray:
    """Per-cell total counts, read in blocks of nonzeros so nothing dense is ever allocated.

    A block is a whole number of rows spanning about `block_nnz` nonzeros, and the per-row sums
    come from one cumulative sum over the block -- which handles empty rows without the
    off-by-one that `np.add.reduceat` has when a row starts at the block's end.
    """
    X = csr_group(h5)
    indptr = X["indptr"][:].astype(np.int64)
    data = X["data"]
    n_rows = len(indptr) - 1
    out = np.zeros(n_rows, dtype=np.float64)
    i = 0
    while i < n_rows:
        lo = int(indptr[i])
        j = min(max(int(np.searchsorted(indptr, lo + block_nnz, side="right")) - 1, i + 1), n_rows)
        hi = int(indptr[j])
        cum = np.concatenate(([0.0], np.cumsum(data[lo:hi], dtype=np.float64)))
        rel = indptr[i : j + 1] - lo
        out[i:j] = cum[rel[1:]] - cum[rel[:-1]]
        i = j
    return out


def write_log1p_uns(h5) -> None:
    """`uns/log1p = {'base': None}`, in the encoding anndata writes for it.

    `cell_load` looks for the key's presence only (`perturbation_dataloader.py:513`), but
    anndata has to be able to read the file back, so the shape matches what
    `sc.pp.log1p` + `adata.write_h5ad` produces: a dict group holding a null-encoded `base`.
    """
    import h5py

    uns = h5.require_group("uns")
    if "encoding-type" not in uns.attrs:
        uns.attrs["encoding-type"] = "dict"
        uns.attrs["encoding-version"] = "0.1.0"
    g = uns.create_group("log1p")
    g.attrs["encoding-type"] = "dict"
    g.attrs["encoding-version"] = "0.1.0"
    base = g.create_dataset("base", data=h5py.Empty("f4"))
    base.attrs["encoding-type"] = "null"
    base.attrs["encoding-version"] = "0.1.0"


def shifted_log(h5, target_sum: float | None = None, block_nnz: int = 8_000_000) -> dict:
    """Rewrite X in place as `normalize_total(target_sum)` then `log1p`, and stamp `uns/log1p`.

    Matches scanpy 1.11's CSR path exactly: the size factor is the **median of every cell's
    total**, each row is divided by `total / target_sum`, and a row totalling zero is divided
    by 1 rather than by 0 (`allow_divide_by_zero=False`).

    Returns the numbers worth printing and asserting on, so a caller can prove what it wrote
    without re-reading 726 million values.
    """
    if "log1p" in h5.get("uns", {}):
        raise SystemExit(
            "uns/log1p is already present, so this file has been logged once. Logging it again "
            "would be silent and unrecoverable; delete the file and rebuild it from the source."
        )
    X = csr_group(h5)
    data = X["data"]
    head = data[: min(len(data), 1_000_000)]
    n_fractional = int((head != np.rint(head)).sum())
    if n_fractional:
        raise SystemExit(
            f"{n_fractional:,} of the first {len(head):,} values in X/data are not integers, so "
            "this matrix is not raw counts. The median-depth size factor is only meaningful on "
            "counts; refusing to normalise something already normalised."
        )

    depths = row_totals(h5, block_nnz)
    if target_sum is None:
        target_sum = float(np.median(depths))
    if not target_sum > 0:
        raise SystemExit(f"target_sum must be positive, got {target_sum}")
    factors = (depths / target_sum).astype(np.float32)
    factors[factors == 0] = 1.0

    indptr = X["indptr"][:].astype(np.int64)
    n_rows = len(indptr) - 1
    peak = 0.0
    i = 0
    while i < n_rows:
        lo = int(indptr[i])
        j = min(max(int(np.searchsorted(indptr, lo + block_nnz, side="right")) - 1, i + 1), n_rows)
        hi = int(indptr[j])
        if hi > lo:
            per_row = np.repeat(factors[i:j], np.diff(indptr[i : j + 1]))
            block = np.log1p(data[lo:hi] / per_row)
            data[lo:hi] = block
            peak = max(peak, float(block.max()))
        i = j

    write_log1p_uns(h5)
    return {
        "n_cells": int(n_rows),
        "target_sum": target_sum,
        "depth_min": float(depths.min()),
        "depth_median": float(np.median(depths)),
        "depth_max": float(depths.max()),
        "max_value": peak,
    }


def read_categorical(obs, name: str) -> list[str]:
    """The column's values, whichever of the two encodings anndata used for it.

    A column is categorical in one file and a plain string array in the next -- `batch` is
    categorical in the X-Atlas fold files and a string array after `stream_subset` -- and
    reading only the categorical shape crashed on the second (2026-09-05).
    """
    import h5py

    g = obs[name]
    if isinstance(g, h5py.Group):
        cats = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return [cats[i] for i in g["codes"][:]]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in g[:]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", required=True, type=Path,
                    help="source h5ad (repeatable; pair each with a --cell-type)")
    ap.add_argument("--cell-type", action="append", required=True,
                    help="the cell-line name to stamp on that --src (repeatable, same order)")
    ap.add_argument("--pert-col", default="gene_target",
                    help="obs column holding the perturbed gene in the SOURCE files")
    ap.add_argument("--control-label", default="Non-Targeting",
                    help="the control value as it appears in --pert-col. Declared, never inferred.")
    ap.add_argument("--batch-col", default="batch")
    ap.add_argument("--out-dir", type=Path,
                    help="where the copies land; omit only with --in-place")
    ap.add_argument("--in-place", action="store_true",
                    help="normalise each --src where it lies instead of copying it first. For "
                         "files this project just wrote (a union-axis projection), where the "
                         "copy would be a second pass over tens of GB for nothing.")
    ap.add_argument("--out-pert-col", default="gene_target")
    ap.add_argument("--out-cell-type-col", default="cell_type")
    ap.add_argument("--log1p", action="store_true",
                    help="also rewrite X as the shifted logarithm -- normalize_total(target_sum) "
                         "then log1p -- and stamp uns/log1p so cell_load autodetects it. "
                         "Streamed; a file that already carries uns/log1p is refused.")
    ap.add_argument("--target-sum", type=float, default=None,
                    help="size factor for --log1p. Default (and the recommendation) is the "
                         "dataset's own median raw count depth, which is what scanpy's "
                         "target_sum=None means. Pass 1e4 only to deliberately do CP10k.")
    ap.add_argument("--block-nnz", type=int, default=8_000_000,
                    help="nonzeros per streamed block for --log1p; the RAM knob")
    args = ap.parse_args()

    if args.target_sum is not None and not args.log1p:
        raise SystemExit("--target-sum only means something with --log1p")

    import h5py

    if len(args.src) != len(args.cell_type):
        raise SystemExit(f"{len(args.src)} --src against {len(args.cell_type)} --cell-type; pair them.")

    if args.in_place == bool(args.out_dir):
        raise SystemExit("give exactly one of --out-dir or --in-place")
    out_dir = args.out_dir.expanduser() if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    axes: dict[str, set[str]] = {}
    written = []
    logged: list[bool] = []

    for src, ct in zip(args.src, args.cell_type):
        src = src.expanduser()
        dst = src if out_dir is None else out_dir / src.name
        if out_dir is None:
            print(f"\n{src}  (in place)")
        else:
            print(f"\n{src.name}  ->  {dst}")
            if dst.exists():
                print("  destination exists; leaving it alone (delete it to rebuild)")
            else:
                shutil.copy2(src, dst)
                print(f"  copied {dst.stat().st_size / 2**30:.2f} GB")

        with h5py.File(dst, "a") as h:
            obs = h["obs"]
            perts = read_categorical(obs, args.pert_col)
            n = len(perts)

            n_ctrl = sum(1 for p in perts if p == args.control_label)
            if n_ctrl == 0:
                seen = sorted({p for p in perts if "arget" in p.lower()})
                raise SystemExit(
                    f"  control label {args.control_label!r} matches ZERO cells. "
                    f"Control-looking labels present: {seen}. Refusing to write a file whose "
                    f"control arm cell_load would silently treat as a perturbation."
                )
            print(f"  {n:,} cells | {len(set(perts)):,} distinct labels | "
                  f"control {args.control_label!r} = {n_ctrl:,} cells ({100 * n_ctrl / n:.1f}%)")

            if args.out_pert_col != args.pert_col:
                write_categorical(obs, args.out_pert_col, perts)
                print(f"  wrote obs/{args.out_pert_col} (copy of {args.pert_col})")

            write_categorical(obs, args.out_cell_type_col, [ct] * n)
            print(f"  wrote obs/{args.out_cell_type_col} = {ct!r} on all {n:,} cells")

            if args.batch_col not in obs:
                raise SystemExit(f"  no obs/{args.batch_col} to use as batch_col")
            nb = len(set(read_categorical(obs, args.batch_col)))
            print(f"  obs/{args.batch_col}: {nb} batches")

            order = list(obs.attrs.get("column-order", []))
            for col in (args.out_cell_type_col, args.out_pert_col):
                if col not in order:
                    order.append(col)
            obs.attrs["column-order"] = np.array(order, dtype=object)

            vi = h["var"]["_index"]
            if hasattr(vi, "keys"):
                genes = [x.decode() if isinstance(x, bytes) else str(x)
                         for x in vi["categories"][:]] if "categories" in vi else \
                        [x.decode() if isinstance(x, bytes) else str(x) for x in vi["values"][:]]
            else:
                genes = [x.decode() if isinstance(x, bytes) else str(x) for x in vi[:]]
            axes[src.name] = set(genes)
            print(f"  gene axis: {len(genes):,}")

            if args.log1p:
                stats = shifted_log(h, args.target_sum, args.block_nnz)
                print(f"  X -> shifted log: size factor {stats['target_sum']:,.1f} "
                      f"({'given' if args.target_sum else 'median raw depth'}) | "
                      f"raw depth min {stats['depth_min']:,.0f} median "
                      f"{stats['depth_median']:,.0f} max {stats['depth_max']:,.0f} | "
                      f"max logged value {stats['max_value']:.3f}")
                print("  wrote uns/log1p — cell_load will report is_log1p ENABLED")
            logged.append("log1p" in h.get("uns", {}))

        written.append(dst)

    if len(axes) > 1:
        shared = set.intersection(*axes.values())
        print(f"\nshared gene axis across the {len(axes)} outputs: {len(shared):,}")
        for k, v in axes.items():
            print(f"  {k:36s} {len(v):,}")
        if any(len(v) != len(shared) for v in axes.values()):
            print("  WARNING: the axes differ. cell_load will happily train on the union of files "
                  "with different var; the model's output space then means different genes in "
                  "different rows. Harmonise before training, or train one axis at a time.")

    if len(set(logged)) > 1:
        mixed = ", ".join(f"{p.name}={'log1p' if lg else 'counts'}" for p, lg in zip(written, logged))
        print(f"\n  WARNING: the outputs disagree on uns/log1p ({mixed}). cell_load refuses that "
              "outright for output_space='all', and it would mean two value scales in one loss "
              "for anything else. Rebuild the odd one out.")

    where = out_dir if out_dir is not None else "each file's own directory"
    print(f"\nready for a cell_load TOML pointing at: {where}")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
