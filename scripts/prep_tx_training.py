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
columns with h5py, leaving the count matrix untouched. It does not rewrite X, so it costs a disk
copy rather than an AnnData round-trip.

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
        --out-dir ~/data/sidechain/derived/tx-train-xatlas
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
    args = ap.parse_args()

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

    where = out_dir if out_dir is not None else "each file's own directory"
    print(f"\nready for a cell_load TOML pointing at: {where}")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
