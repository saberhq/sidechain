"""Put every corpus on ONE gene axis without throwing away the genes only some of them measure.

`state tx train` emits a single gene axis for a whole run, so corpora quantified against
different gene lists cannot simply be concatenated -- column *j* would mean a different gene
in different rows. The way round it up to now was to intersect: keep only the genes every
corpus measures. Measured here 2026-09-05 over the seven cell-level perturbation corpora we
hold, that costs almost everything::

    intersection of all seven   6,301 genes ->  6,054 of the 18,533 scored (32.7%)
    UNION      of all seven    40,573 genes -> 18,528 of the 18,533 scored (99.97%)

Four of the six board members read differential expression across the gene axis, so training
on the intersection means predicting a third of the axis we are scored on.

This module takes the other road. Every corpus is **projected onto one wide axis**, and each
one carries a **mask** naming the genes it actually measured. A gene a corpus never measured
is a structural zero in that corpus's rows, and the training loss is told to ignore it
(`sidechain.models.masked_axis`) rather than to fit it as a real zero. Nothing is discarded
and nothing is invented.

The mask has to reach the loss, and this is what makes that cheap: `cell_load`'s sampler
builds each cell *set* from a single h5 file, so **every cell in a set shares one mask** --
it is a per-set row, not a per-cell matrix.

Three axis modes, declared rather than inferred:

* ``anchor``       -- the axis IS the anchor list, in its order. With the challenge's
                      ``gene_names.csv`` that means the model's output is already in
                      submission order and no re-indexing happens at inference.
* ``union``        -- anchor genes first (anchor order), then every other gene any source
                      measures, sorted. Widest; carries genes that are never scored.
* ``intersection`` -- what we do today, kept so the fallback is expressible in the same code
                      and the two can be compared without a second tool.

Usage -- plan first (reads only `var`, seconds), then project::

    uv run python -m sidechain.data.union_axis plan \
        --source k562_gwps=~/data/sidechain/external/zenodo-13350497/ReplogleWeissman2022_K562_gwps.h5ad \
        --source thp1=~/data/sidechain/external/zenodo-13350497/WesselsSatija2023.h5ad \
        --mode anchor --anchor ~/data/sidechain/vcc2026/gene_names.csv \
        --manifest ~/data/sidechain/derived/tx-train-union/axis_manifest.json

    uv run python -m sidechain.data.union_axis project \
        --manifest ~/data/sidechain/derived/tx-train-union/axis_manifest.json \
        --out-dir ~/data/sidechain/derived/tx-train-union

`project` writes one directory per source, because `cell_load` keys a TOML ``[datasets]``
entry to a **directory** and hands every h5ad in it the same name -- and that name is how the
mask is looked up at training time. Two axes under one key would silently share one mask.

It does not touch obs. Run `scripts/prep_tx_training.py --in-place` afterwards to stamp the
cell type and normalise the perturbation column and control label; that script owns the
control-label refusal and this one has no business repeating it.

Subsampling is `stream_subset` and stays there: projecting does not change how many
nonzeros a corpus has, only which columns they sit in, so project what you mean to train on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

from sidechain.data.stream_pseudobulk import _iter_blocks

MODES = ("anchor", "union", "intersection")


# --------------------------------------------------------------------------- axis


def read_gene_axis(path: str | Path) -> list[str]:
    """The gene symbols of an h5ad's var index, read with h5py (no matrix touched)."""
    with h5py.File(Path(path).expanduser(), "r") as f:
        return _read_index(f["var"])


def _read_index(group: h5py.Group) -> list[str]:
    name = group.attrs.get("_index", "_index")
    if isinstance(name, bytes):
        name = name.decode()
    d = group[name]
    if isinstance(d, h5py.Group):  # categorical
        cats = [_s(x) for x in d["categories"][:]]
        return [cats[i] for i in d["codes"][:]]
    return [_s(x) for x in d[:]]


def _s(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def read_anchor(path: str | Path) -> list[str]:
    """The anchor gene list from a one-column csv.

    2025's `gene_names.csv` has NO header and 2026's has one (`gene_name`) -- the trap in
    the public CLAUDE.md. Rather than guess, treat a first row that is not a plausible gene
    symbol (lowercase, or containing a space or underscore) as a header and drop it.
    """
    rows = [line.strip() for line in Path(path).expanduser().read_text().splitlines()]
    rows = [r.split(",")[0].strip().strip('"') for r in rows if r]
    if rows and (rows[0].islower() or " " in rows[0] or "_" in rows[0]):
        rows = rows[1:]
    return rows


@dataclass(frozen=True)
class AxisPlan:
    """One gene axis, plus which of its genes each source actually measured."""

    genes: tuple[str, ...]
    mode: str
    sources: dict[str, Path]
    measured: dict[str, np.ndarray]  # source key -> sorted target positions it measures
    dropped: dict[str, int]  # source key -> genes it measures that the axis does not carry

    def mask(self, source: str) -> np.ndarray:
        """Boolean over `genes`: True where `source` measured that gene."""
        m = np.zeros(len(self.genes), dtype=bool)
        m[self.measured[source]] = True
        return m

    @property
    def genes_sha256(self) -> str:
        return hashlib.sha256("\n".join(self.genes).encode()).hexdigest()


def build_axis(
    axes: dict[str, Sequence[str]],
    *,
    mode: str = "anchor",
    anchor: Sequence[str] | None = None,
    sources: dict[str, Path] | None = None,
) -> AxisPlan:
    """Plan the shared axis from each source's gene list.

    Raises rather than guessing on the two ways this goes silently wrong: a source whose var
    index repeats a symbol (which column wins?), and a source that shares no gene with the
    axis (a mapping bug that would otherwise write an all-zero corpus).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    if not axes:
        raise ValueError("no sources given")
    if mode == "anchor" and not anchor:
        raise ValueError("mode 'anchor' needs an --anchor gene list")

    for key, genes in axes.items():
        dupes = _duplicates(genes)
        if dupes:
            raise ValueError(
                f"source {key!r} repeats {len(dupes)} gene symbol(s) in its var index "
                f"(e.g. {dupes[:5]}); a duplicated symbol has no single column to project, "
                "so de-duplicate the file before it goes on a shared axis."
            )

    sets = {k: set(g) for k, g in axes.items()}
    if mode == "anchor":
        genes = list(dict.fromkeys(anchor))
    elif mode == "intersection":
        shared = set.intersection(*sets.values())
        genes = [g for g in anchor if g in shared] if anchor else sorted(shared)
    else:  # union
        every = set.union(*sets.values())
        head = [g for g in dict.fromkeys(anchor) if g in every] if anchor else []
        genes = head + sorted(every - set(head))

    if not genes:
        raise ValueError(f"mode {mode!r} produced an empty axis")

    pos = {g: i for i, g in enumerate(genes)}
    measured: dict[str, np.ndarray] = {}
    dropped: dict[str, int] = {}
    for key, gs in axes.items():
        hit = np.array(sorted(pos[g] for g in sets[key] if g in pos), dtype=np.int64)
        if hit.size == 0:
            raise ValueError(
                f"source {key!r} shares no gene with the {len(genes):,}-gene axis. Its first "
                f"symbols are {list(gs)[:5]} -- almost certainly a different ID space, not an "
                "empty overlap. Refusing to project a corpus that would be entirely masked."
            )
        measured[key] = hit
        dropped[key] = len(sets[key]) - hit.size

    return AxisPlan(
        genes=tuple(genes),
        mode=mode,
        sources=dict(sources or {}),
        measured=measured,
        dropped=dropped,
    )


def _duplicates(genes: Iterable[str]) -> list[str]:
    seen, dupes = set(), []
    for g in genes:
        if g in seen:
            dupes.append(g)
        else:
            seen.add(g)
    return dupes


def coverage_rows(plan: AxisPlan, scored: Sequence[str] | None = None) -> list[dict]:
    """One row per source: axis size, genes kept, genes dropped, and scored-axis coverage."""
    scored_set = set(scored or ())
    on_axis = [g for g in plan.genes if g in scored_set] if scored_set else []
    rows = []
    for key in plan.measured:
        idx = plan.measured[key]
        mine = {plan.genes[i] for i in idx}
        rows.append(
            {
                "source": key,
                "measured_on_axis": int(idx.size),
                "dropped_off_axis": int(plan.dropped[key]),
                "scored_covered": len(mine & scored_set) if scored_set else None,
            }
        )
    rows.append(
        {
            "source": "AXIS",
            "measured_on_axis": len(plan.genes),
            "dropped_off_axis": 0,
            "scored_covered": len(on_axis) if scored_set else None,
        }
    )
    return rows


# --------------------------------------------------------------------- projection


class _CsrWriter:
    """Append CSR row-blocks into an h5ad-encoded sparse group without holding the matrix."""

    def __init__(self, group: h5py.Group, n_cols: int, dtype: np.dtype):
        self.g = group
        self.n_cols = n_cols
        self.n_rows = 0
        self.nnz = 0
        self.g.attrs["encoding-type"] = "csr_matrix"
        self.g.attrs["encoding-version"] = "0.1.0"
        self.data = group.create_dataset(
            "data", shape=(0,), maxshape=(None,), dtype=dtype, chunks=(1 << 20,), compression="gzip"
        )
        self.indices = group.create_dataset(
            "indices", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(1 << 20,), compression="gzip"
        )
        self.indptr = group.create_dataset(
            "indptr", shape=(1,), maxshape=(None,), dtype=np.int64, chunks=(1 << 16,), compression="gzip"
        )
        self.indptr[0] = 0

    def append(self, block: sp.csr_matrix) -> None:
        # The scatter reorders columns, so a row's indices are no longer ascending; readers
        # (anndata, cell_load) assume they are.
        block.has_sorted_indices = False
        block.sort_indices()
        n, add = block.shape[0], block.nnz
        self.data.resize((self.nnz + add,))
        self.data[self.nnz :] = block.data
        self.indices.resize((self.nnz + add,))
        self.indices[self.nnz :] = block.indices.astype(np.int32)
        self.indptr.resize((self.n_rows + n + 1,))
        self.indptr[self.n_rows + 1 :] = block.indptr[1:] + self.nnz
        self.n_rows += n
        self.nnz += add

    def close(self) -> None:
        self.g.attrs["shape"] = np.array([self.n_rows, self.n_cols], dtype=np.int64)


def project_h5ad(
    src: str | Path,
    dst: str | Path,
    plan: AxisPlan,
    source: str,
    *,
    block_rows: int | None = None,
    dtype: str = "float32",
    progress: bool = False,
) -> dict:
    """Rewrite `src` with its columns scattered onto `plan.genes`, obs carried over.

    Nonzeros are neither created nor destroyed -- only their column index changes -- so the
    output is the same matrix on a wider axis, written CSR whatever the input encoding was.
    Genes the axis does not carry are dropped here and counted in the return value; genes the
    source never measured are absent (structural zeros), and `uns/sidechain_axis/mask` says
    which is which so nothing downstream has to guess.

    Only X, obs and var cross over: `layers`, `obsm`, `varm`, `obsp` and `varp` are gene-axis
    or derived data that would have to be re-projected or re-computed, and `state tx` with
    `embed_key=null` reads none of them. Anything non-empty is named on stdout rather than
    dropped quietly.
    """
    src, dst = Path(src).expanduser(), Path(dst).expanduser()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if source not in plan.measured:
        raise KeyError(f"{source!r} is not a source of this plan ({sorted(plan.measured)})")

    with h5py.File(src, "r") as fin:
        src_genes = _read_index(fin["var"])
        pos = {g: i for i, g in enumerate(plan.genes)}
        col_map = np.array([pos.get(g, -1) for g in src_genes], dtype=np.int64)
        keep = col_map >= 0
        if not keep.any():
            raise ValueError(f"{src.name}: no column of it lands on the axis")
        n_rows = _n_rows(fin)
        skipped = [k for k in ("layers", "obsm", "varm", "obsp", "varp")
                   if k in fin and len(fin[k].keys())]
        if skipped:
            print(f"  not carried over: {', '.join(f'{k}={list(fin[k].keys())}' for k in skipped)}")

        with h5py.File(dst, "w") as fout:
            fout.attrs["encoding-type"] = "anndata"
            fout.attrs["encoding-version"] = "0.1.0"
            writer = _CsrWriter(fout.create_group("X"), len(plan.genes), np.dtype(dtype))
            done = 0
            target_of_kept = col_map[keep]
            for _r0, block in _iter_blocks(fin, block_rows):
                block = block.tocsr() if sp.issparse(block) else sp.csr_matrix(block)
                sub = block[:, keep].astype(dtype)
                # Scatter: a kept column's nonzeros keep their values and move to the column
                # that gene occupies on the shared axis. Same nnz, wider matrix.
                scattered = sp.csr_matrix(
                    (sub.data, target_of_kept[sub.indices].astype(np.int32), sub.indptr),
                    shape=(sub.shape[0], len(plan.genes)),
                )
                writer.append(scattered)
                done += sub.shape[0]
                if progress:
                    print(f"  {done:,}/{n_rows:,} cells", end="\r", flush=True)
            writer.close()
            if progress:
                print()

            fin.copy(fin["obs"], fout, "obs")
            _write_var(fout, plan.genes)
            _write_uns(fout, plan, source, src)

    return {
        "source": source,
        "src": str(src),
        "dst": str(dst),
        "cells": writer.n_rows,
        "nnz": writer.nnz,
        "genes": len(plan.genes),
        "measured": int(plan.measured[source].size),
        "dropped_off_axis": int(plan.dropped[source]),
    }


def _n_rows(f: h5py.File) -> int:
    X = f["X"]
    return int(X.attrs["shape"][0]) if isinstance(X, h5py.Group) else int(X.shape[0])


def _write_var(f: h5py.File, genes: Sequence[str]) -> None:
    var = f.create_group("var")
    var.attrs["encoding-type"] = "dataframe"
    var.attrs["encoding-version"] = "0.2.0"
    var.attrs["_index"] = "_index"
    var.attrs["column-order"] = np.array([], dtype=h5py.string_dtype())
    d = var.create_dataset("_index", data=np.array(list(genes), dtype=h5py.string_dtype()))
    d.attrs["encoding-type"] = "string-array"
    d.attrs["encoding-version"] = "0.2.0"


def _write_uns(f: h5py.File, plan: AxisPlan, source: str, src: Path) -> None:
    uns = f.create_group("uns")
    uns.attrs["encoding-type"] = "dict"
    uns.attrs["encoding-version"] = "0.1.0"
    g = uns.create_group("sidechain_axis")
    g.attrs["encoding-type"] = "dict"
    g.attrs["encoding-version"] = "0.1.0"
    mask = g.create_dataset("mask", data=plan.mask(source).astype(np.uint8))
    mask.attrs["encoding-type"] = "array"
    mask.attrs["encoding-version"] = "0.2.0"
    for k, v in (
        ("source", source),
        ("mode", plan.mode),
        ("genes_sha256", plan.genes_sha256),
        ("origin", str(src)),
    ):
        d = g.create_dataset(k, data=np.array(v, dtype=h5py.string_dtype()))
        d.attrs["encoding-type"] = "string"
        d.attrs["encoding-version"] = "0.2.0"


# ----------------------------------------------------------------------- manifest


def write_manifest(plan: AxisPlan, path: str | Path, *, scored: Sequence[str] | None = None) -> Path:
    """The one file training reads: the axis, and each source's measured positions on it."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    scored_set = set(scored or ())
    payload = {
        "mode": plan.mode,
        "n_genes": len(plan.genes),
        "genes_sha256": plan.genes_sha256,
        "genes": list(plan.genes),
        "scored_on_axis": len(scored_set & set(plan.genes)) if scored_set else None,
        "sources": {
            key: {
                "path": str(plan.sources.get(key, "")),
                "n_measured": int(idx.size),
                "dropped_off_axis": int(plan.dropped[key]),
                "measured_index": idx.tolist(),
            }
            for key, idx in plan.measured.items()
        },
    }
    path.write_text(json.dumps(payload, indent=1))
    return path


def load_manifest(path: str | Path) -> tuple[list[str], dict[str, np.ndarray], dict]:
    """`(genes, {source: measured positions}, raw payload)` -- the reader half of the above."""
    payload = json.loads(Path(path).expanduser().read_text())
    genes = list(payload["genes"])
    got = hashlib.sha256("\n".join(genes).encode()).hexdigest()
    if got != payload["genes_sha256"]:
        raise ValueError(f"{path}: gene list does not match its own sha256; the file was edited")
    measured = {k: np.asarray(v["measured_index"], dtype=np.int64) for k, v in payload["sources"].items()}
    return genes, measured, payload


def plan_from_manifest(path: str | Path) -> AxisPlan:
    genes, measured, payload = load_manifest(path)
    return AxisPlan(
        genes=tuple(genes),
        mode=payload["mode"],
        sources={k: Path(v["path"]) for k, v in payload["sources"].items() if v.get("path")},
        measured=measured,
        dropped={k: int(v["dropped_off_axis"]) for k, v in payload["sources"].items()},
    )


# ---------------------------------------------------------------------------- CLI


def _parse_sources(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--source wants NAME=PATH, got {item!r}")
        name, _, path = item.partition("=")
        if name in out:
            raise SystemExit(f"--source {name!r} given twice")
        out[name] = Path(path).expanduser()
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="read every source's var, write the axis manifest")
    p.add_argument("--source", action="append", required=True, metavar="NAME=PATH",
                   help="NAME is the cell_load [datasets] key the mask will be looked up by")
    p.add_argument("--mode", choices=MODES, default="anchor")
    p.add_argument("--anchor", type=Path, help="one-column csv of the axis to anchor on "
                                               "(e.g. ~/data/sidechain/vcc2026/gene_names.csv)")
    p.add_argument("--scored", type=Path, help="one-column csv scored by the board, for the "
                                               "coverage table; defaults to --anchor")
    p.add_argument("--manifest", type=Path, required=True)

    q = sub.add_parser("project", help="rewrite each source onto the manifest's axis")
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--out-dir", type=Path, required=True)
    q.add_argument("--only", action="append", help="project just this source (repeatable)")
    q.add_argument("--block-rows", type=int, default=None)
    q.add_argument("--dtype", default="float32")
    q.add_argument("--overwrite", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "plan":
        sources = _parse_sources(args.source)
        anchor = read_anchor(args.anchor) if args.anchor else None
        scored = read_anchor(args.scored) if args.scored else anchor
        axes = {k: read_gene_axis(v) for k, v in sources.items()}
        for k, v in axes.items():
            print(f"  {k:16s} {len(v):>7,} genes  {sources[k]}")
        plan = build_axis(axes, mode=args.mode, anchor=anchor, sources=sources)
        print(f"\naxis: {len(plan.genes):,} genes (mode {plan.mode})")
        print(f"{'source':16s} {'on axis':>9s} {'dropped':>9s} {'scored':>9s}")
        for row in coverage_rows(plan, scored):
            sc = "-" if row["scored_covered"] is None else f"{row['scored_covered']:,}"
            print(f"{row['source']:16s} {row['measured_on_axis']:>9,} {row['dropped_off_axis']:>9,} {sc:>9s}")
        out = write_manifest(plan, args.manifest, scored=scored)
        print(f"\nmanifest -> {out}")
        return 0

    plan = plan_from_manifest(args.manifest)
    keys = args.only or list(plan.measured)
    for key in keys:
        src = plan.sources.get(key)
        if src is None:
            raise SystemExit(f"manifest records no path for source {key!r}; re-run `plan`")
        # One directory per source: cell_load names every h5ad in a directory after the
        # TOML key, and that name is the mask lookup.
        dst = args.out_dir.expanduser() / key / Path(src).name
        if dst.exists() and not args.overwrite:
            print(f"{key}: {dst} exists; skipping (--overwrite to rebuild)")
            continue
        print(f"{key}: {src.name} -> {dst}")
        info = project_h5ad(src, dst, plan, key, block_rows=args.block_rows,
                            dtype=args.dtype, progress=True)
        print(f"  {info['cells']:,} cells x {info['genes']:,} genes | {info['nnz']:,} nonzeros | "
              f"measured {info['measured']:,} | dropped off axis {info['dropped_off_axis']:,}")
    print(f"\nTOML [datasets] keys must be exactly: {', '.join(keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
