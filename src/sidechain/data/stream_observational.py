"""Stream an OBSERVATIONAL corpus into per-context expression profiles, without landing it.

The fourth streaming reader, and the first that is not a `*_pseudobulk`. The other three
(`stream_pseudobulk`, `stream_parquet_pseudobulk`, `stream_lamindb_pseudobulk`) all group rows
by a **perturbation** label and emit `PseudobulkSums`, which is the integration contract into
`submit.build.pooled_delta` and `eval.loco`. An observational corpus has no such label -- that
is what `kind: observational` means in `configs/datasets.yaml` -- so there is no delta to pool
and nothing here may ever reach `pooled_delta`.

    uv run python -m sidechain.data.stream_observational --dataset scbasecount_human_gene \
        --exclude-experiments ~/data/sidechain/derived/scbasecount/pooled_srx.txt \
        --min-cells 2000 --limit-contexts 400 \
        --out ~/data/sidechain/derived/scbasecount/independent_context_profiles.npz

**What it emits and why it is a different type.** `ContextProfiles`, keyed by CONTEXT (a cell
line or other biological label) rather than by perturbation, with a per-gene detection count
that `PseudobulkSums` has no slot for. It is deliberately NOT the same dataclass and its npz
uses different key names, so feeding one to `PseudobulkSums.load` raises `KeyError: labels`
rather than loading a context panel as if it were a perturbation panel. The corpus has no
deltas; a silent load would invent some.

**Why this is worth streaming at all** (measured 2026-08-29, T31; private
`research/ideas/scbasecount-expression-prior.md`). Our curated Replogle K562 file carries 8,248
genes and covers 7,679 of the 18,533 challenge genes. The 10,854 it lacks are not lost signal --
in the same cells they sit at 0.0001 median UMI/cell and 47.5 % are exactly zero, because K562
does not express them. But 2,867 of those ARE expressed in challenge context A, carrying 13.6 %
of its UMI mass and containing 69 of the 300 panel targets, and six unrelated contexts recover
95.8 % of them. So the object worth building is a **gene x context** matrix over many cell
identities -- not a within-line block, which is where this started.

**CSC in, CSR out, and the reason is not the one the recon gave.** scBaseCount stores `X` as a
`csc_matrix`, which the T31 recon correctly called the better layout for a gene-major pass such
as a gene x gene Gram accumulation. That is no longer the target. Both operations here -- the
per-cell UMI floor and CPM normalisation -- are ROW operations, so each file is converted once
with `.tocsr()`. One conversion per ~67 MB experiment is far cheaper than repeated CSC row
slicing, and the whole file is the streaming block: one experiment is one SRA accession, small
enough to hold, so there is no row-block loop at all.

**The content filter is not optional and is read from the config, never hardcoded.** On labelled
data a bad row is caught downstream by a control arm that does not match or a delta that comes
out zero. Unlabelled data has no such check, so the QC rule is declared at the front door in
`configs/datasets.yaml` and applied here. For scBaseCount that matters enormously: 20.8 % of its
human cells are guide-capture sub-libraries quantified against the gene reference, and corpus-wide
only 71.8 % of cells reach 500 UMI.

Writes `<out>` plus `<out>.lineage.json` beside it -- which `PROVENANCE.json` it came from, the
filter that was applied, which experiments survived it, and the code SHA (ADR 0003's LINEAGE
shape).
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:  # anndata >= 0.11
    from anndata.io import read_elem
except ImportError:  # pragma: no cover - older anndata
    from anndata.experimental import read_elem

GCS = "https://storage.googleapis.com/"


# --------------------------------------------------------------- the output --


@dataclass
class ContextProfiles:
    """Per-context expression profiles from an unlabelled corpus.

    One row per CONTEXT, not per perturbation. Every field is a sum, so two runs over
    disjoint experiment sets can be added; nothing here is a mean until asked for one.

    `detect_count` is the field `PseudobulkSums` has no room for and the reason this is a
    separate type rather than a reuse: how often a gene is seen AT ALL in a context is the
    quantity that separates "this line does not express it" from "we did not sequence deep
    enough", and it is what the `jac` metric charges for.
    """

    contexts: list[str]
    genes: np.ndarray           # (G,) the axis these profiles are ON
    count_sum: np.ndarray       # (C, G) float64 -- raw counts
    cpm_sum: np.ndarray         # (C, G) float64 -- sum over cells of per-cell CPM
    cpm_sq_sum: np.ndarray      # (C, G) float64 -- and of its square
    detect_count: np.ndarray    # (C, G) int64   -- cells with a nonzero count
    n_cells: np.ndarray         # (C,) int64
    libsize_sum: np.ndarray     # (C,) float64
    n_experiments: np.ndarray   # (C,) int64
    sources: list[str] = field(default_factory=list)

    def mean_cpm(self) -> np.ndarray:
        return self.cpm_sum / np.maximum(self.n_cells, 1)[:, None]

    def var_cpm(self) -> np.ndarray:
        n = np.maximum(self.n_cells, 1)[:, None]
        m = self.cpm_sum / n
        return np.maximum(self.cpm_sq_sum / n - m * m, 0.0)

    def detect_rate(self) -> np.ndarray:
        return self.detect_count / np.maximum(self.n_cells, 1)[:, None]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path).expanduser(),
            contexts=np.asarray(self.contexts, dtype=object),
            genes=np.asarray(self.genes, dtype=object),
            count_sum=self.count_sum, cpm_sum=self.cpm_sum, cpm_sq_sum=self.cpm_sq_sum,
            detect_count=self.detect_count, n_cells=self.n_cells,
            libsize_sum=self.libsize_sum, n_experiments=self.n_experiments,
            sources=np.asarray(self.sources, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> ContextProfiles:
        z = np.load(Path(path).expanduser(), allow_pickle=True)
        return cls(
            contexts=[str(x) for x in z["contexts"]], genes=z["genes"].astype(str),
            count_sum=z["count_sum"], cpm_sum=z["cpm_sum"], cpm_sq_sum=z["cpm_sq_sum"],
            detect_count=z["detect_count"], n_cells=z["n_cells"],
            libsize_sum=z["libsize_sum"], n_experiments=z["n_experiments"],
            sources=[str(x) for x in z["sources"]],
        )


class Accumulator:
    """Running sums for a fixed gene axis, growing its context list as it goes."""

    def __init__(self, genes: np.ndarray):
        self.genes = np.asarray(genes, dtype=str)
        self._idx: dict[str, int] = {}
        self._rows: list[dict] = []
        self.sources: list[str] = []

    def _row(self, context: str) -> dict:
        if context not in self._idx:
            self._idx[context] = len(self._rows)
            g = len(self.genes)
            self._rows.append({
                "count_sum": np.zeros(g), "cpm_sum": np.zeros(g), "cpm_sq_sum": np.zeros(g),
                "detect_count": np.zeros(g, dtype=np.int64),
                "n_cells": 0, "libsize_sum": 0.0, "n_experiments": 0,
            })
        return self._rows[self._idx[context]]

    def add(self, context: str, counts: sp.csr_matrix, cols: np.ndarray) -> int:
        """Fold one experiment's already-filtered cells into `context`.

        `cols[j]` is the position on THIS accumulator's gene axis of the file's column j, or
        -1 if the gene is off-axis. Resolved per file rather than once, because nothing
        guarantees two experiments were quantified against the same reference -- assuming it
        would misalign every gene silently, which is the failure `loaders.gene_index` is
        strict about.
        """
        if counts.shape[0] == 0:
            return 0
        keep = cols >= 0
        sub = counts[:, keep]
        dest = cols[keep]
        lib = np.asarray(counts.sum(axis=1)).ravel()          # CPM uses the FULL library
        scale = 1e6 / np.maximum(lib, 1.0)
        cpm = sp.diags(scale) @ sub

        row = self._row(context)
        np.add.at(row["count_sum"], dest, np.asarray(sub.sum(axis=0)).ravel())
        np.add.at(row["cpm_sum"], dest, np.asarray(cpm.sum(axis=0)).ravel())
        np.add.at(row["cpm_sq_sum"], dest, np.asarray(cpm.multiply(cpm).sum(axis=0)).ravel())
        np.add.at(row["detect_count"], dest, np.diff(sub.tocsc().indptr))
        row["n_cells"] += counts.shape[0]
        row["libsize_sum"] += float(lib.sum())
        row["n_experiments"] += 1
        return counts.shape[0]

    def result(self) -> ContextProfiles:
        order = sorted(self._idx, key=lambda c: self._idx[c])
        stack = lambda k: np.array([r[k] for r in self._rows])
        return ContextProfiles(
            contexts=order, genes=self.genes,
            count_sum=stack("count_sum"), cpm_sum=stack("cpm_sum"),
            cpm_sq_sum=stack("cpm_sq_sum"), detect_count=stack("detect_count"),
            n_cells=np.array([r["n_cells"] for r in self._rows], dtype=np.int64),
            libsize_sum=np.array([r["libsize_sum"] for r in self._rows]),
            n_experiments=np.array([r["n_experiments"] for r in self._rows], dtype=np.int64),
            sources=list(self.sources),
        )


# ------------------------------------------------------------- the selection --


def select_experiments(manifest: pd.DataFrame, content_filter: dict, *,
                       sizes: dict[str, int] | None = None,
                       exclude: set[str] | None = None,
                       drop_contexts: set[str] | None = None,
                       keep_contexts: set[str] | None = None,
                       context_col: str | None = None,
                       max_per_context: int | None = None,
                       min_cells_per_experiment: int = 0,
                       id_col: str = "srx_accession") -> pd.DataFrame:
    """Apply a block's declared `content_filter` to the experiment manifest.

    Every rule is read from the config, so the filter that ran is the filter that is written
    down. `min_bytes_per_cell` needs the per-file sizes the gate already recorded in
    PROVENANCE.json; when they are not supplied that rule is SKIPPED LOUDLY by the caller
    rather than silently, because a missing filter on an unlabelled corpus is invisible
    downstream.
    """
    m = manifest
    for col in ("cell_prep", "tech_10x"):
        allowed = content_filter.get(col)
        if allowed:
            m = m[m[col].isin(list(allowed))]
    mbpc = content_filter.get("min_bytes_per_cell")
    if mbpc and sizes:
        n = m["obs_count"].astype(float).clip(lower=1)
        bpc = m[id_col].map(sizes).astype(float) / n
        m = m[bpc >= float(mbpc)]
    if exclude:
        m = m[~m[id_col].isin(exclude)]
    if keep_contexts and context_col:
        # An allow-list, and the reason it exists is measured. `cell_line` is 5,644 free-text
        # strings, so "the top N contexts by cell mass" is dominated by primary tissue and by
        # descriptive sentences ("Patient 23, age 67, female"), and none of the four rotation
        # lines appears in the top 150 at all. Naming the contexts is the cheap version of the
        # Cellosaurus resolver the idea file asks for, and it is what makes a positive control
        # possible. Matched case-insensitively; the SPELLINGS stay distinct rows on purpose,
        # because whether `MCF7` and `MCF-7` land on each other is itself a control.
        norm = m[context_col].astype(str).str.strip().str.lower()
        m = m[norm.isin({str(x).strip().lower() for x in keep_contexts})]
    if drop_contexts and context_col:
        # Free-text context labels include non-labels -- scBaseCount's `cell_line` has
        # `unsure`, `none`, `not_applicable` and case variants over 4,358 experiments. Pooling
        # those into one row would manufacture exactly the composition artefact the
        # within-line restriction was originally meant to avoid, so they are named and dropped
        # by the CALLER rather than defaulted here, and the list lands in LINEAGE.json.
        norm = m[context_col].astype(str).str.strip().str.lower()
        m = m[~norm.isin({str(x).strip().lower() for x in drop_contexts})]
    if min_cells_per_experiment:
        m = m[m["obs_count"].astype(float) >= float(min_cells_per_experiment)]
    if max_per_context and context_col:
        # Breadth, not bulk: a context panel wants many contexts each measured well, not one
        # context measured 4,000 times.
        #
        # WHICH experiment the cap keeps matters, and the obvious sort is the wrong one. Most
        # CELLS is not best measured -- an experiment with 50,000 barcodes at 300 UMI each is
        # worse for a profile than 5,000 barcodes at 20,000, and it is also ten times the
        # bytes. So the cap ranks by per-cell content (bytes per cell, the proxy validated
        # against true depth on 2026-08-29: rejected experiments sit at a median 7.0 UMI/cell,
        # kept ones at 3,887) and takes cell count only as a floor. Falls back to obs_count
        # when no sizes are available, which is the only ranking left.
        if sizes:
            rank = -(m[id_col].map(sizes).astype(float)
                     / m["obs_count"].astype(float).clip(lower=1))
        else:
            rank = -m["obs_count"].astype(float)
        m = (m.assign(_rank=rank).sort_values("_rank")
               .groupby(context_col, sort=False).head(max_per_context).drop(columns="_rank"))
    return m


def gene_axis_map(symbols: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Position of each file column on `axis`, or -1. Built per file, never assumed."""
    pos = {g: i for i, g in enumerate(np.asarray(axis, dtype=str))}
    return np.array([pos.get(str(s), -1) for s in symbols], dtype=np.int64)


def read_counts(handle, *, symbol_col: str, min_umi: float) -> tuple[sp.csr_matrix, np.ndarray, int]:
    """One experiment's matrix, CSR and cell-filtered, plus its gene symbols.

    Returns `(counts, symbols, n_dropped)`. The whole file is one block: an experiment is one
    SRA accession and holds comfortably, so there is no row-block loop.
    """
    var = read_elem(handle["var"])
    if symbol_col not in var.columns:
        raise KeyError(
            f"var has no {symbol_col!r} column (found {list(var.columns)}). The gene axis is "
            "declared in configs/datasets.yaml as gene_symbol_col; a wrong one would key the "
            "profiles on an axis nothing else shares."
        )
    symbols = var[symbol_col].astype(str).to_numpy()
    X = handle["X"]
    enc = X.attrs.get("encoding-type")
    shape = tuple(X.attrs["shape"])
    if enc == "csc_matrix":
        M = sp.csc_matrix((X["data"][:], X["indices"][:], X["indptr"][:]), shape=shape).tocsr()
    elif enc == "csr_matrix":
        M = sp.csr_matrix((X["data"][:], X["indices"][:], X["indptr"][:]), shape=shape)
    else:
        raise ValueError(f"unsupported X encoding {enc!r}")
    before = M.shape[0]
    if min_umi:
        lib = np.asarray(M.sum(axis=1)).ravel()
        M = M[lib >= float(min_umi)]
    return M, symbols, before - M.shape[0]


# ------------------------------------------------------------------ the run --


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 -- lineage records "unknown", it does not crash a stream
        return "unknown"


def _open_url(url: str, retries: int = 3):
    last = None
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(url, timeout=300)
        except Exception as exc:  # noqa: BLE001 -- one flaky object must not end a long stream
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not read {url}: {last}")


def _key_to_url(path_or_key: str, bucket: str) -> str:
    """`gs://bucket/key` or a bare key -> an anonymous HTTPS URL.

    The access finding this rests on (T31, re-probed 2026-08-29): anonymous LISTING of the
    bucket is 401 but anonymous GET of an object whose exact key you already have is 200 with
    working range requests, and the manifest hands over every key. The route is re-probed at
    the start of every run rather than assumed, because Arc's README says Requester Pays and
    it is Arc's setting to change.
    """
    if path_or_key.startswith("gs://"):
        return GCS + path_or_key[len("gs://"):]
    return f"{GCS}{bucket}/{path_or_key.lstrip('/')}"


def stream_observational(
    block: dict,
    *,
    root: Path,
    gene_axis: np.ndarray,
    exclude: set[str] | None = None,
    drop_contexts: set[str] | None = None,
    keep_contexts: set[str] | None = None,
    max_per_context: int | None = None,
    min_cells_per_experiment: int = 0,
    min_cells: int = 0,
    limit_contexts: int | None = None,
    limit_experiments: int | None = None,
    bucket: str = "arc-institute-virtual-cell-atlas",
    progress_every: int = 25,
) -> tuple[ContextProfiles, dict]:
    """Stream one `kind: observational` block into per-context profiles.

    Reads the manifest the block names in `sample_from`, applies the declared
    `content_filter`, and folds every surviving experiment into the context its
    `context_col` names. Nothing lands.
    """
    import h5py

    spec = next(f["spec"] for f in block["files"] if f.get("kind") == "observational")
    manifest_key = spec["sample_from"]
    id_col, ctx_col = spec["sample_id_col"], spec["context_col"]
    cf = dict(spec.get("content_filter") or {})
    if not cf:
        raise ValueError("block declares an empty content_filter; refusing to build a prior "
                         "out of whatever the corpus happens to contain")

    prov_path = root / block["dest"] / "PROVENANCE.json"
    sizes: dict[str, int] = {}
    if prov_path.exists():
        prov = json.loads(prov_path.read_text())
        sizes = {Path(f["name"]).stem: int(f["size_bytes"])
                 for f in prov["selected"] if f["name"].endswith(".h5ad")}
    elif cf.get("min_bytes_per_cell"):
        raise FileNotFoundError(
            f"{prov_path} is missing, so the declared min_bytes_per_cell rule cannot run. "
            "Run `python -m sidechain.ingest.fetch --dataset "
            f"{block['name']}` first -- an unlabelled corpus has no downstream check that "
            "would notice the filter silently not happening."
        )

    with _open_url(_key_to_url(manifest_key, bucket)) as fh:
        manifest = pd.read_parquet(io.BytesIO(fh.read()))
    picked = select_experiments(manifest, cf, sizes=sizes, exclude=exclude,
                                drop_contexts=drop_contexts, keep_contexts=keep_contexts,
                                context_col=ctx_col,
                                max_per_context=max_per_context,
                                min_cells_per_experiment=min_cells_per_experiment,
                                id_col=id_col)

    counts_by_ctx = picked.groupby(ctx_col)["obs_count"].sum()
    if min_cells:
        keep_ctx = set(counts_by_ctx[counts_by_ctx >= min_cells].index)
        picked = picked[picked[ctx_col].isin(keep_ctx)]
    if limit_contexts:
        top = counts_by_ctx.loc[sorted(set(picked[ctx_col]))].nlargest(limit_contexts).index
        picked = picked[picked[ctx_col].isin(set(top))]
    if limit_experiments:
        picked = picked.head(limit_experiments)

    acc = Accumulator(gene_axis)
    t0, dropped_cells, failed = time.time(), 0, []
    for i, (_, row) in enumerate(picked.iterrows(), start=1):
        url = _key_to_url(str(row["file_path"]), bucket)
        try:
            with _open_url(url) as fh, h5py.File(io.BytesIO(fh.read())) as h5:
                M, symbols, ndrop = read_counts(
                    h5, symbol_col=spec.get("gene_symbol_col", "gene_symbols"),
                    min_umi=cf.get("min_umi_per_cell", 0))
                acc.add(str(row[ctx_col]), M, gene_axis_map(symbols, gene_axis))
                dropped_cells += ndrop
        except Exception as exc:  # noqa: BLE001 -- record and continue; a stream is long
            failed.append({"experiment": str(row[id_col]), "error": str(exc)[:200]})
        if progress_every and (i % progress_every == 0 or i == len(picked)):
            print(f"  {i}/{len(picked)} experiments  {len(acc._idx)} contexts  "
                  f"{time.time() - t0:.0f}s", flush=True)

    acc.sources = [f"{block['name']}:{block['record']}"]
    profiles = acc.result()
    lineage = {
        "produced_by": "sidechain.data.stream_observational",
        "code_sha": _git_sha(),
        "dataset": block["name"], "host": block["host"], "record": block["record"],
        "provenance": str(prov_path), "license": block.get("license"),
        "resolution": f"per {ctx_col}",
        "content_filter": cf,
        "drop_contexts": sorted(drop_contexts or ()),
        "keep_contexts": sorted(keep_contexts or ()),
        "max_per_context": max_per_context,
        "min_cells_per_experiment": min_cells_per_experiment,
        "min_cells_per_context": min_cells,
        "experiments_in_manifest": len(manifest),
        "experiments_selected": len(picked),
        "experiments_failed": failed,
        "experiments_excluded_by_caller": len(exclude or ()),
        "cells_dropped_by_min_umi": int(dropped_cells),
        "cells_kept": int(profiles.n_cells.sum()),
        "contexts": len(profiles.contexts),
        "genes": len(gene_axis),
        "seconds": round(time.time() - t0),
    }
    return profiles, lineage


def main(argv: list[str] | None = None) -> int:
    from sidechain.ingest.fetch import DEFAULT_ROOT, select_dataset

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="a kind: observational block")
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--genes", type=Path, help="one gene symbol per line; default is the "
                                               "2026 challenge axis under --root")
    ap.add_argument("--exclude-experiments", type=Path,
                    help="file of experiment ids to drop -- how the independent part of a "
                         "corpus is taken when some of it re-quantifies screens we pool")
    ap.add_argument("--drop-contexts", default="",
                    help="comma-separated context labels that are not contexts (scBaseCount's "
                         "cell_line carries `unsure`, `none`, `not_applicable`); matched "
                         "case-insensitively and recorded in the lineage file")
    ap.add_argument("--keep-contexts", type=Path,
                    help="file of context labels to keep, one per line -- the cheap version of "
                         "a Cellosaurus resolver, and what makes a positive control possible")
    ap.add_argument("--max-per-context", type=int,
                    help="cap experiments per context -- a context panel wants breadth; the "
                         "cap keeps the ones with the most content PER CELL, not the most cells")
    ap.add_argument("--min-cells-per-experiment", type=int, default=0,
                    help="floor on an experiment's cell count, so the per-cell ranking cannot "
                         "pick a deep but tiny run")
    ap.add_argument("--min-cells", type=int, default=0,
                    help="drop contexts with fewer than this many cells after filtering")
    ap.add_argument("--limit-contexts", type=int)
    ap.add_argument("--limit-experiments", type=int)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    block = select_dataset(args.dataset, args.config)
    if not any(f.get("kind") == "observational" for f in block["files"]):
        raise SystemExit(f"{args.dataset} declares no `kind: observational` file; this reader "
                         "is for corpora with no per-cell perturbation label")

    if args.genes:
        axis = np.array([l.strip() for l in args.genes.read_text().splitlines() if l.strip()])
    else:
        g = pd.read_csv(args.root / "vcc2026" / "gene_names.csv")   # HAS a header in 2026
        axis = g.iloc[:, 0].astype(str).to_numpy()
    exclude = None
    if args.exclude_experiments:
        exclude = {l.strip() for l in args.exclude_experiments.read_text().splitlines() if l.strip()}

    print(f"{args.dataset}: gene axis {len(axis):,} | excluding {len(exclude or ()):,} experiments")
    profiles, lineage = stream_observational(
        block, root=args.root, gene_axis=axis, exclude=exclude,
        drop_contexts={c.strip() for c in args.drop_contexts.split(",") if c.strip()},
        keep_contexts=({l.strip() for l in args.keep_contexts.read_text().splitlines() if l.strip()}
                       if args.keep_contexts else None),
        max_per_context=args.max_per_context,
        min_cells_per_experiment=args.min_cells_per_experiment,
        min_cells=args.min_cells, limit_contexts=args.limit_contexts,
        limit_experiments=args.limit_experiments)

    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    profiles.save(out)
    lineage["out"] = str(out)
    out.with_name(out.name + ".lineage.json").write_text(json.dumps(lineage, indent=1) + "\n")
    print(json.dumps({k: v for k, v in lineage.items() if k != "experiments_failed"}, indent=1))
    if lineage["experiments_failed"]:
        print(f"  {len(lineage['experiments_failed'])} experiment(s) failed; see the lineage file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
