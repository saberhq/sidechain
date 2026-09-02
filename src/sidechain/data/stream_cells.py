"""Stream X-Atlas/Orion parquet into an h5ad of CELLS -- a LOCO fold receiver.

    uv run python -m sidechain.data.stream_cells --context hct116 \
        --keep ~/data/sidechain/vcc2026/panels_union.csv \
        --qc ~/data/sidechain/derived/xatlas-orion/hct116_full.qc.npz \
        --cap 300 --control-cells 20000 \
        --out ~/data/sidechain/cache/vcc2026/loco_hct116_real.h5ad

WHY THIS EXISTS, given the corpus has already been streamed twice. Both previous passes
emitted `PseudobulkSums` -- per-perturbation sums, which is all a *source* needs. A fold
RECEIVER needs something a sum cannot provide: the emitter builds its `ContextProfile` from
the receiver's own control CELLS, and cell-eval2 runs its differential-expression call on
cells. So HCT116 can be a source forever and a receiver never, until this pass exists.

What it buys: a SECOND challenge-shaped fold. `loco_k562gwps` is currently the only fold
whose panel is the challenge panel (the four local folds share **0** of the challenge 300),
so every knob priced this season rests on one fold.

THREE DECISIONS, each of which has a wrong version that has already cost us something.

1. **The gene axis is never narrowed.** Library size is summed over the EMITTED axis
   (`count_emitters.ContextProfile.from_controls`), so dropping genes silently rescales every
   cell's CPM -- on the challenge axis, to 0.71x, and not by a constant, so it does not even
   cancel in a ratio. That is why the 2026-08-23 panel artifacts were archived. This writer
   emits the full native axis and refuses `--keep`-style gene restriction outright.

2. **The LABEL scope is the union of every panel we might ever score, not just the challenge
   300.** This is a one-way door: the corpus is `route: stream`, so a label we drop costs
   another 46 GB to recover. HCT116 covers 300/300 of the challenge panel, 290/300 of Jurkat's,
   290/300 of K562-essential's and 272/272 of the gwps fold's -- 830 of a union of 849. Taking
   the union costs ~1 GB more and makes this line a receiver for the essential-gene panels
   too, which is the only way to test whether a calibration fitted on one corpus transfers.

3. **Cells are drawn ACROSS batches by quota, never "the first N".** Files arrive
   lexicographically (`Batch1, Batch10, Batch100`), so a naive cap would draw a systematic
   subset of GEM batches and confound the fold with batch structure. The per-(label, batch)
   cell counts already sit in the full run's QC sidecar, so the quota is computed exactly,
   offline, by largest-remainder allocation before a byte moves.

LICENCE. X-Atlas/Orion is CC-BY-NC-SA-4.0 and `redistribution_encumbered: true`. Unlike every
other artifact we derive from it, this one is a row subset of verbatim Xaira cells rather than
a transformative aggregate. It belongs on the Mac and in the private lamin bucket ONLY -- never
the public repo, the site, or a published bundle. (Saber's explicit sign-off, 2026-08-31.)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from sidechain.data.stream_parquet_pseudobulk import (
    GeneAxis,
    build_gene_axis,
    decode_lists,
    hf_opener,
    label_column,
    read_gene_names,
    tokens_to_csr,
)

CONTROL_IN_CORPUS = "Non-Targeting"
CONTROL_HARMONISED = "non-targeting"   # what every fold receiver spells it (loco.py --control)


def plan_quota(batch_cells: np.ndarray, labels: list[str], batches: list[str],
               want: dict[str, int]) -> dict[tuple[str, str], int]:
    """Allocate each label's cell budget across the batches that actually hold its cells.

    Largest-remainder (Hamilton) apportionment on the per-(label, batch) counts, so the
    sample mirrors the corpus's own batch composition instead of whichever files happen to
    be read first. A label with fewer cells than its budget contributes all of them.

    `batch_cells` is (labels x batches) from the full run's QC sidecar -- counts we already
    paid 126 GB for, reused here so the plan costs nothing.
    """
    pos = {lab: i for i, lab in enumerate(labels)}
    quota: dict[tuple[str, str], int] = {}
    for lab, budget in want.items():
        i = pos.get(lab)
        if i is None:
            continue
        row = batch_cells[i].astype(np.int64)
        total = int(row.sum())
        if total == 0:
            continue
        take = min(int(budget), total)
        if take == total:                      # everything this label has
            alloc = row.copy()
        else:
            exact = row * (take / total)
            alloc = np.floor(exact).astype(np.int64)
            short = take - int(alloc.sum())
            if short > 0:
                # largest remainders first; ties broken by the bigger batch, then by name,
                # so the plan is deterministic and reproducible from the sidecar alone
                order = np.lexsort((np.arange(len(row)), -row, -(exact - alloc)))
                for j in order[:short]:
                    alloc[j] += 1
            alloc = np.minimum(alloc, row)
        for j, n in enumerate(alloc):
            if n:
                quota[(lab, batches[j])] = int(n)
    return quota


class CellSink:
    """Collects up to a quota of cells per (label, batch) as CSR shards."""

    def __init__(self, axis: GeneAxis, quota: dict[tuple[str, str], int],
                 relabel: dict[str, str] | None = None, seed: int = 20260831):
        self.axis = axis
        self.quota = dict(quota)
        self.relabel = relabel or {}
        self.blocks: list[sp.csr_matrix] = []
        self.obs: list[pd.DataFrame] = []
        self.kept = 0
        self.by_label: dict[str, int] = {}
        self.seed = seed
        # Remaining budget is drawn down across the several record-batches a file is read
        # in, so it lives on the sink rather than in `fold`.
        self.left: dict[tuple[str, str], int] = dict(quota)

    def wanted_labels(self) -> set[str]:
        return {lab for lab, _ in self.quota}

    def fold(self, frame: pd.DataFrame, batch: str) -> None:
        """Take this batch's share of the cells it holds, then decode only those."""
        if frame.empty:
            return
        targets = label_column(frame, "gene_target")
        present = set(targets.tolist())
        if not any(self.left.get((lab, batch), 0) for lab in present):
            return
        # Choose rows first, decode second: the sparse work is the expensive part and this
        # pass wants a few hundred of a batch's ~18,500 cells.
        #
        # WITHIN a batch the pick is randomised, not "the first N rows". Across batches the
        # quota is already exact (`plan_quota`), so the only ordering left to worry about is
        # the corpus's own row order inside a file -- unknown, and free to correlate with
        # something. A seeded shuffle costs nothing and removes the question; the seed makes
        # the selection reproducible from the sidecar plus this constant.
        take = np.zeros(len(targets), dtype=bool)
        for lab in present:
            budget = self.left.get((lab, batch), 0)
            if budget <= 0:
                continue
            rows = np.flatnonzero(targets == lab)
            if len(rows) > budget:
                # crc32, not hash(): Python randomises string hashing per process, which
                # would make the selection unreproducible between runs of the same command.
                key = f"{self.seed}|{batch}|{lab}|{len(self.blocks)}".encode()
                rng = np.random.default_rng(zlib.crc32(key))
                rows = rng.choice(rows, size=budget, replace=False)
            take[rows] = True
            self.left[(lab, batch)] = budget - len(rows)
        if not take.any():
            return
        sub_frame = frame[take]
        _, tokens, values, rows = decode_lists(sub_frame)
        mat = tokens_to_csr(self.axis, tokens, values, rows, n_cells=len(sub_frame))
        lib = np.asarray(mat.sum(axis=1)).ravel()
        alive = lib > 0
        if not alive.any():
            return
        mat = mat[alive]
        sub_frame = sub_frame[alive]
        labs = label_column(sub_frame, "gene_target")
        pert = np.array([self.relabel.get(str(x), str(x)) for x in labs])
        obs = pd.DataFrame({
            "perturbation": pert,
            "gene_target": labs.astype(str),
            "batch": batch,
        })
        for col in ("guide_target", "sample", "pct_counts_mt", "total_counts",
                    "n_genes_by_counts"):
            if col in sub_frame.columns:
                obs[col] = sub_frame[col].to_numpy()
        self.blocks.append(mat.astype(np.float32))
        self.obs.append(obs)
        self.kept += int(mat.shape[0])
        for lab, n in zip(*np.unique(pert, return_counts=True), strict=True):
            self.by_label[str(lab)] = self.by_label.get(str(lab), 0) + int(n)

    def write(self, out: Path, gene_ids: np.ndarray | None = None) -> dict:
        import anndata as ad

        X = sp.vstack(self.blocks, format="csr")
        obs = pd.concat(self.obs, ignore_index=True)
        obs.index = [f"cell_{i}" for i in range(len(obs))]
        var = pd.DataFrame(index=pd.Index(self.axis.genes, name=None))
        if gene_ids is not None:
            var["gene_id"] = gene_ids
        adata = ad.AnnData(X=X, obs=obs, var=var)
        out.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out, compression="gzip")
        return {"cells": int(X.shape[0]), "genes": int(X.shape[1]), "nnz": int(X.nnz),
                "file_gb": round(out.stat().st_size / 1e9, 3)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="xatlas_orion")
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--challenge-config", default="challenges/vcc2026/config.yaml")
    ap.add_argument("--root", type=Path, default=Path.home() / "data" / "sidechain")
    ap.add_argument("--context", required=True)
    ap.add_argument("--keep", required=True,
                    help="CSV of perturbation labels to keep (column target_gene or first)")
    ap.add_argument("--qc", required=True,
                    help="the full run's .qc.npz -- supplies the per-(label, batch) counts "
                         "the quota is computed from")
    ap.add_argument("--cap", type=int, default=400,
                    help="max cells per perturbation. 400 mirrors the challenge, which asks "
                         "for 400 predicted cells per perturbation per context, and costs "
                         "only ~3%% more than 300 because the corpus's median target has 149")
    ap.add_argument("--control-cells", type=int, default=20000)
    ap.add_argument("--limit-files", type=int, help="stream only the first N files (smoke run)")
    ap.add_argument("--batch-rows", type=int, default=2048)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    from sidechain.data.loaders import load_challenge_config
    from sidechain.data.stream_parquet_pseudobulk import _dataset_block
    from sidechain.ingest.provenance import read_provenance

    block = _dataset_block(args.dataset, args.config)
    dest = args.root / block["dest"]
    provenance = read_provenance(dest)
    if provenance is None:
        raise SystemExit(f"no PROVENANCE.json at {dest}: run sidechain.ingest.fetch first")
    revision = provenance["record"]["version"]
    repo = provenance["record"]["record_id"]
    entry = next((f for f in block["files"]
                  if (f.get("spec") or {}).get("context") == args.context), None)
    if entry is None:
        raise SystemExit(f"no context {args.context!r} in {args.dataset}")

    import fnmatch
    names = sorted(fnmatch.filter([f["name"] for f in provenance["selected"]], entry["name"]))
    if args.limit_files:
        names = names[: args.limit_files]

    opener = hf_opener(repo, revision)
    # The gene axis: ALL native symbols. `build_gene_axis(gene_map, None)` is the --all-genes
    # branch -- see decision 1 in the module docstring; there is deliberately no flag here.
    with opener("metadata/gene_metadata.parquet") as handle:
        gene_map = pq.read_table(handle).to_pandas()
    axis = build_gene_axis(gene_map, None)

    qc = np.load(Path(args.qc).expanduser(), allow_pickle=True)
    qc_labels = [str(x) for x in qc["labels"]]
    qc_batches = [str(x) for x in qc["batches"]]
    df = pd.read_csv(args.keep)
    col = "target_gene" if "target_gene" in df.columns else df.columns[0]
    want = {str(t): args.cap for t in dict.fromkeys(df[col].astype(str))}
    want[CONTROL_IN_CORPUS] = args.control_cells
    quota = plan_quota(qc["batch_cells"], qc_labels, qc_batches, want)
    planned = sum(quota.values())
    labels_planned = len({lab for lab, _ in quota})
    print(json.dumps({"labels_requested": len(want), "labels_with_cells": labels_planned,
                      "cells_planned": planned, "files": len(names),
                      "genes": len(axis.genes), "revision": revision}), flush=True)

    sink = CellSink(axis, quota, relabel={CONTROL_IN_CORPUS: CONTROL_HARMONISED})
    columns = ["gene_token_id", "gene_expression", "gene_target", "guide_target",
               "pass_guide_filter", "sample", "pct_counts_mt", "total_counts",
               "n_genes_by_counts"]
    t0 = time.time()
    for n, path in enumerate(names, 1):
        batch = Path(path).stem
        with opener(path) as handle:
            pf = pq.ParquetFile(handle)
            for rb in pf.iter_batches(batch_size=args.batch_rows, columns=columns):
                frame = rb.to_pandas()
                # pass_guide_filter is int64, not bool: `== 1`, not `is True`.
                frame = frame[frame["pass_guide_filter"].to_numpy() == 1]
                sink.fold(frame, batch)
        print(f"  [{n}/{len(names)}] {batch}  kept {sink.kept:,}  "
              f"{time.time() - t0:6.0f}s", flush=True)

    if not sink.blocks:
        raise SystemExit("no cells kept -- check --keep against the corpus's labels")
    ids = None
    if "gene_id" in gene_map.columns:
        by_symbol = dict(zip(gene_map["gene_symbol"].astype(str),
                             gene_map["gene_id"].astype(str), strict=False)) \
            if "gene_symbol" in gene_map.columns else {}
        if by_symbol:
            ids = np.array([by_symbol.get(g, "") for g in axis.genes])
    info = sink.write(args.out.expanduser(), gene_ids=ids)
    info.update(planned=planned, kept=sink.kept, labels=len(sink.by_label),
                control_cells=sink.by_label.get(CONTROL_HARMONISED, 0),
                seconds=round(time.time() - t0))
    print(json.dumps(info), flush=True)
    counts = args.out.expanduser().with_suffix(".cells_per_label.csv")
    pd.DataFrame(sorted(sink.by_label.items()),
                 columns=["perturbation", "cells"]).to_csv(counts, index=False)
    print(f"per-label counts -> {counts}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
