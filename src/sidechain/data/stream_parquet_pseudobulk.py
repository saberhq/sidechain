"""Stream long-format parquet (X-Atlas/Orion) into the same `PseudobulkSums` the h5ad streamer emits.

    uv run python -m sidechain.data.stream_parquet_pseudobulk --dataset xatlas_orion --context hct116 \
        [--keep PANEL.csv | --all-genes] [--guide-agreement] --out ~/data/sidechain/derived/xatlas-orion/hct116_full

Input: one row per cell, its non-zero genes as two parallel list columns (`gene_token_id`, `gene_expression`: raw
counts as float64), control `Non-Targeting`, `pass_guide_filter` int64 1/0. Reads the corpus over HF at its
PROVENANCE.json revision; writes OUT.npz, the OUT.qc.npz sidecar (per-(label, batch) cell counts, mt fraction),
OUT.guide.npz with --guide-agreement, and an entry in the directory's LINEAGE.json.
Notes: private/code-notes/src/sidechain/data/stream_parquet_pseudobulk.py.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.ingest import checks
from sidechain.utils.logging import code_sha
from sidechain.utils.paths import resolve_config

CONTROL_LABEL = "Non-Targeting"

# The columns this reader fetches; `_columns_for` adds `guide_target` only when a keying or the coverage tally
# needs it.
COLUMNS = [
    "gene_token_id",      # list<int64>  -- the non-zero genes for this cell
    "gene_expression",    # list<double> -- their raw counts, parallel to the above
    "gene_target",        # the perturbation label; `Non-Targeting` is the control
    "sample",             # the GEM batch
    "pass_guide_filter",  # int64 1/0, NOT bool
    "pct_counts_mt",      # mitochondrial read fraction
    "total_counts",       # the cell's library size over the corpus's whole gene axis
]


@dataclass
class Keying:
    """One way of grouping cells into labels: `gene_target` (the silenced gene) or `guide_target` (the construct).

    `name` suffixes the artifacts and the LINEAGE.json entry: empty for the primary keying, `guide` for
    `<out>.guide.npz`.
    """

    name: str
    column: str
    labels: list[str]
    scope: str
    resolution: str


def label_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    """The per-cell label under one keying; under `guide_target` the control cells keep `Non-Targeting`."""
    if column == "gene_target":
        return frame["gene_target"].to_numpy()
    gene = frame["gene_target"].to_numpy()
    return np.where(gene == CONTROL_LABEL, CONTROL_LABEL, frame[column].to_numpy())


def _columns_for(keyings: list[Keying], *, guide_coverage: bool = False) -> list[str]:
    """`COLUMNS` plus whatever the requested keyings key on. Order is stable.

    `guide_coverage` adds `guide_target` on its own account: counting cells per construct
    needs the column even when no accumulator is keyed by it.
    """
    cols = list(COLUMNS)
    for keying in keyings:
        if keying.column not in cols:
            cols.append(keying.column)
    if guide_coverage and "guide_target" not in cols:
        cols.append("guide_target")
    return cols


@dataclass
class StreamQC:
    """What the stream saw, beside the sums; saved as the `.qc.npz` sidecar, never inside `PseudobulkSums`."""

    labels: list[str]
    batches: list[str]
    batch_cells: np.ndarray        # (L, B) int32 -- per-(perturbation, batch) cell counts
    pct_mt_sum: np.ndarray         # (L,) float64
    pct_mt_sq_sum: np.ndarray      # (L,) float64
    total_counts_sum: np.ndarray   # (L,) float64 -- the corpus's own `total_counts`, not libsize_sum
    n_cells: np.ndarray            # (L,) int64
    cells_seen: int = 0
    cells_dropped_filter: int = 0
    cells_dropped_zero: int = 0
    labels_in_corpus: int = 0   # distinct gene_target values seen, kept or not

    # ---- screen-coverage QC ----------------------------------------------------------------
    # `gene_cells[g]`: cells with a non-zero count for emitted gene g, over every cell kept.
    gene_cells: np.ndarray | None = None        # (G,) int64
    # Cells per construct, over every construct the corpus carries; controls pooled under one label.
    guide_labels: list[str] = field(default_factory=list)
    guide_cells: np.ndarray | None = None       # (len(guide_labels),) int64

    def mean_pct_mt(self) -> np.ndarray:
        return self.pct_mt_sum / np.maximum(self.n_cells, 1)

    def sd_pct_mt(self) -> np.ndarray:
        n = np.maximum(self.n_cells, 1)
        m = self.pct_mt_sum / n
        return np.sqrt(np.maximum(self.pct_mt_sq_sum / n - m * m, 0.0))

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path).expanduser(),
            labels=np.asarray(self.labels, dtype=object),
            batches=np.asarray(self.batches, dtype=object),
            batch_cells=self.batch_cells,
            pct_mt_sum=self.pct_mt_sum, pct_mt_sq_sum=self.pct_mt_sq_sum,
            total_counts_sum=self.total_counts_sum, n_cells=self.n_cells,
            cells_seen=self.cells_seen, cells_dropped_filter=self.cells_dropped_filter,
            cells_dropped_zero=self.cells_dropped_zero,
            labels_in_corpus=self.labels_in_corpus,
            gene_cells=(self.gene_cells if self.gene_cells is not None
                        else np.zeros(0, dtype=np.int64)),
            guide_labels=np.asarray(self.guide_labels, dtype=object),
            guide_cells=(self.guide_cells if self.guide_cells is not None
                         else np.zeros(0, dtype=np.int64)),
        )


@dataclass
class GeneAxis:
    """Token id -> emitted column, and the record of what did not map.

    `col_of_token[t]` is the column gene token `t` feeds, or -1 to drop it.
    """

    genes: np.ndarray              # the emitted axis, in challenge order
    col_of_token: np.ndarray       # (max_token+1,) int32, -1 = not on the emitted axis
    unmapped_challenge: list[str]  # challenge genes X-Atlas does not carry -- recorded, not dropped silently
    collided_symbols: list[str]    # emitted symbols fed by more than one token
    # Whether the axis was restricted to the challenge genes; carried, not inferred from `unmapped_challenge`.
    restricted: bool = True

    @property
    def n_mapped(self) -> int:
        return len(self.genes)


def read_gene_names(path: str | Path, *, expect: int | None = None) -> list[str]:
    """The 2026 `gene_names.csv`, which has a header row (`gene_name`); `expect` checks the gene count."""
    genes = pd.read_csv(Path(path).expanduser()).iloc[:, 0].astype(str).tolist()
    if expect is not None and len(genes) != expect:
        raise ValueError(
            f"{path} read as {len(genes)} genes but the config says {expect}. This is the "
            "header trap: the 2026 file has a header row and the 2025 file does not. Do not "
            '"fix" it by switching header=None -- check which file this is.'
        )
    return genes


def build_gene_axis(gene_map: pd.DataFrame, challenge_genes: list[str] | None) -> GeneAxis:
    """Map X-Atlas gene tokens onto the emitted axis: the challenge symbols it carries, in challenge order, or with
    `challenge_genes=None` every symbol it carries, sorted.

    `gene_map` is `metadata/gene_metadata.parquet` (`ensembl_id`, `gene_name`, `gene_token_id`). Tokens sharing a
    symbol are summed into one column; absent challenge genes and collided symbols are recorded on the axis.
    """
    symbols = gene_map["gene_name"].astype(str).to_numpy()
    tokens = gene_map["gene_token_id"].astype(np.int64).to_numpy()
    present = set(symbols.tolist())

    if challenge_genes is None:
        genes = sorted(present)
        unmapped: list[str] = []
    else:
        genes = [g for g in challenge_genes if g in present]
        unmapped = [g for g in challenge_genes if g not in present]

    col_of_symbol = {g: i for i, g in enumerate(genes)}
    col_of_token = np.full(int(tokens.max()) + 1, -1, dtype=np.int32)

    hits: dict[int, int] = {}
    for token, symbol in zip(tokens, symbols, strict=True):
        col = col_of_symbol.get(symbol)
        if col is None:
            continue
        col_of_token[token] = col
        hits[col] = hits.get(col, 0) + 1

    collided = sorted(genes[c] for c, n in hits.items() if n > 1)
    return GeneAxis(np.asarray(genes, dtype=object), col_of_token, unmapped, collided,
                    restricted=challenge_genes is not None)


class _Accumulator:
    """The (labels x genes) sums, plus the sidecar counters. Allocated once, up front."""

    def __init__(self, labels: list[str], genes: np.ndarray, column: str = "gene_target",
                 *, track_coverage: bool = False):
        self.labels = labels
        self.genes = genes
        # The column this accumulator keys on (see `Keying`).
        self.column = column
        self.code_of = {lab: i for i, lab in enumerate(labels)}
        L, G = len(labels), len(genes)
        self.count_sum = np.zeros((L, G))
        self.cpm_sum = np.zeros((L, G))
        self.cpm_sq_sum = np.zeros((L, G))
        self.n_cells = np.zeros(L, dtype=np.int64)
        self.libsize_sum = np.zeros(L)
        # sidecar
        self.pct_mt_sum = np.zeros(L)
        self.pct_mt_sq_sum = np.zeros(L)
        self.total_counts_sum = np.zeros(L)
        self.batch_cells: dict[str, np.ndarray] = {}
        self.cells_seen = 0
        self.dropped_filter = 0
        self.dropped_zero = 0
        self.sources: list[str] = []
        # Every distinct label the corpus contains under this keying, accumulated or not.
        self.seen_labels: set[str] = set()

        # Screen-coverage tallies, off by default (see `StreamQC`).
        self.track_coverage = track_coverage
        self.gene_cells = np.zeros(len(genes), dtype=np.int64) if track_coverage else None
        self.guide_cells: dict[str, int] = {}

    def batch_row(self, batch: str) -> np.ndarray:
        if batch not in self.batch_cells:
            self.batch_cells[batch] = np.zeros(len(self.labels), dtype=np.int32)
        return self.batch_cells[batch]

    def to_pseudobulk(self) -> PseudobulkSums:
        """The sums, with labels that never got a cell pruned (as `stream_pseudobulk` does)."""
        seen = np.flatnonzero(self.n_cells > 0)
        return PseudobulkSums(
            labels=[self.labels[i] for i in seen], genes=self.genes,
            count_sum=self.count_sum[seen], cpm_sum=self.cpm_sum[seen],
            cpm_sq_sum=self.cpm_sq_sum[seen], n_cells=self.n_cells[seen],
            libsize_sum=self.libsize_sum[seen], sources=list(self.sources),
        )

    def to_qc(self) -> StreamQC:
        """The sidecar, keeping every requested label, including those that never got a cell."""
        batches = sorted(self.batch_cells)
        table = (np.stack([self.batch_cells[b] for b in batches], axis=1)
                 if batches else np.zeros((len(self.labels), 0), dtype=np.int32))
        guide_labels = sorted(self.guide_cells)
        return StreamQC(
            labels=list(self.labels), batches=batches, batch_cells=table,
            pct_mt_sum=self.pct_mt_sum, pct_mt_sq_sum=self.pct_mt_sq_sum,
            total_counts_sum=self.total_counts_sum, n_cells=self.n_cells,
            cells_seen=self.cells_seen, cells_dropped_filter=self.dropped_filter,
            cells_dropped_zero=self.dropped_zero, labels_in_corpus=len(self.seen_labels),
            gene_cells=self.gene_cells,
            guide_labels=guide_labels,
            guide_cells=(np.array([self.guide_cells[g] for g in guide_labels], dtype=np.int64)
                         if guide_labels else None),
        )


def _scatter_add(target: np.ndarray, rows: np.ndarray, product) -> None:
    """`target[rows] += product` over the sparse product's coordinates, never a dense (rows x genes) block.

    Bit-identical to the dense form: scipy sums duplicate coordinates, so each (row, col) is unique.
    """
    coo = product.tocoo()
    if coo.nnz == 0:
        return
    target[rows[coo.row], coo.col] += coo.data


def _consume_batch(accs: list[_Accumulator], axis: GeneAxis, frame: pd.DataFrame, *,
                   check_counts: bool) -> bool:
    """Apply the guide filter once, then fold the surviving cells into every accumulator."""
    # pass_guide_filter is int64, not bool: `== 1`, not `is True`.
    keep = frame["pass_guide_filter"].to_numpy() == 1
    dropped = int((~keep).sum())
    for acc in accs:
        acc.dropped_filter += dropped
    frame = frame[keep]
    if frame.empty:
        return False

    # The raw-counts check runs on whichever accumulator first reaches real values, and
    # then only once for the batch -- it is a question about `gene_expression`, which no
    # keying changes.
    ran = False
    for acc in accs:
        ran = _fold_batch(acc, axis, frame, check_counts=check_counts and not ran) or ran
    return ran


def decode_lists(frame: pd.DataFrame):
    """The two parallel list columns -> flat COO `(lengths, tokens, values, rows)`; shared with `stream_cells`."""
    tok_lists = frame["gene_token_id"].to_numpy()
    val_lists = frame["gene_expression"].to_numpy()
    lengths = np.fromiter((len(t) for t in tok_lists), count=len(tok_lists), dtype=np.int64)
    tokens = np.concatenate(tok_lists) if len(tok_lists) else np.empty(0, dtype=np.int64)
    values = np.concatenate(val_lists) if len(val_lists) else np.empty(0, dtype=np.float64)
    rows = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    return lengths, tokens, values, rows


def tokens_to_csr(axis: GeneAxis, tokens, values, rows, *, n_cells: int) -> sp.csr_matrix:
    """Map gene tokens onto the emitted axis and build the (cells x genes) CSR.

    Tokens off the axis or out of range drop; a negative id would otherwise index `col_of_token` from the end.
    """
    in_range = (tokens >= 0) & (tokens < len(axis.col_of_token))
    cols = np.full(tokens.shape, -1, dtype=np.int64)
    cols[in_range] = axis.col_of_token[tokens[in_range]]
    on_axis = cols >= 0
    return sp.csr_matrix(
        (values[on_axis], (rows[on_axis], cols[on_axis])),
        shape=(n_cells, len(axis.genes)),
    )


def _fold_batch(acc: _Accumulator, axis: GeneAxis, frame: pd.DataFrame, *,
                check_counts: bool) -> bool:
    """Fold one guide-filtered frame into one accumulator: decode only the wanted labels' cells, then sum them per
    label through an indicator-matrix product."""
    targets = label_column(frame, acc.column)
    acc.seen_labels.update(targets.tolist())
    codes = np.array([acc.code_of.get(t, -1) for t in targets], dtype=np.int64)
    wanted = codes >= 0
    if not wanted.any():
        return False
    frame = frame[wanted]
    codes = codes[wanted]

    lengths, tokens, values, rows = decode_lists(frame)

    ran_check = False
    if check_counts and values.size:
        ran_check = True
        # Raw counts, checked on real rows: `counts_state` needs row structure, rebuilt here as a padded probe.
        probe_cells = min(64, len(lengths))
        width = int(lengths[:probe_cells].max()) if probe_cells else 0
        if probe_cells and width:
            probe = np.zeros((probe_cells, width))
            start = 0
            for r in range(probe_cells):
                n = int(lengths[r])
                probe[r, :n] = values[start : start + n]
                start += n
        else:
            probe = values[:200_000].reshape(1, -1)
        state = checks.counts_state(probe)
        if state != checks.RAW_COUNTS:
            raise ValueError(
                f"gene_expression is {state!r}, not raw counts. The card and the paper both "
                "say raw UMI counts; the data disagrees, so stop rather than pseudobulking a "
                "transformed matrix."
            )

    sub = tokens_to_csr(axis, tokens, values, rows, n_cells=len(codes))

    lib = np.asarray(sub.sum(axis=1)).ravel()
    alive = lib > 0
    acc.dropped_zero += int((~alive).sum())
    if not alive.any():
        return ran_check
    sub = sub[alive]
    codes = codes[alive]
    lib = lib[alive]
    frame = frame[alive]

    if acc.track_coverage:
        # Per-gene DETECTION, not counts: how many of the cells we just kept had a non-zero
        # value for each emitted gene. `getnnz` counts stored entries per column, and the
        # CSR build already summed duplicate (cell, gene) coordinates -- so a gene fed by two
        # colliding tokens in one cell is counted once, which is the intended meaning.
        acc.gene_cells += sub.getnnz(axis=0).astype(np.int64)
        if "guide_target" in frame.columns:
            constructs, counts = np.unique(label_column(frame, "guide_target"),
                                           return_counts=True)
            for name, count in zip(constructs, counts, strict=True):
                acc.guide_cells[str(name)] = acc.guide_cells.get(str(name), 0) + int(count)

    cpm = sp.diags(1e6 / lib) @ sub

    # Only the labels this batch touched are densified: (labels in batch x genes).
    uniq, local = np.unique(codes, return_inverse=True)
    ind = sp.csr_matrix(
        (np.ones(len(local)), (local, np.arange(len(local)))), shape=(len(uniq), len(local))
    )
    _scatter_add(acc.count_sum, uniq, ind @ sub)
    _scatter_add(acc.cpm_sum, uniq, ind @ cpm)
    _scatter_add(acc.cpm_sq_sum, uniq, ind @ cpm.multiply(cpm))

    np.add.at(acc.n_cells, codes, 1)
    np.add.at(acc.libsize_sum, codes, lib)
    pct = frame["pct_counts_mt"].to_numpy(dtype=np.float64)
    np.add.at(acc.pct_mt_sum, codes, pct)
    np.add.at(acc.pct_mt_sq_sum, codes, pct * pct)
    # The corpus's own total_counts (its whole gene axis), not `lib` (the emitted axis).
    np.add.at(acc.total_counts_sum, codes,
              frame["total_counts"].to_numpy(dtype=np.float64))
    acc.cells_seen += len(codes)

    for batch, sub_codes in pd.Series(codes).groupby(frame["sample"].to_numpy()):
        np.add.at(acc.batch_row(str(batch)), sub_codes.to_numpy(), 1)
    return ran_check


def stream_files(
    paths: list[str],
    *,
    opener,
    axis: GeneAxis,
    labels: list[str],
    batch_rows: int = 2048,
    progress: bool = True,
) -> tuple[PseudobulkSums, StreamQC]:
    """One pass over `paths` under the single `gene_target` keying (see `stream_keyings`)."""
    (pb, qc), = stream_keyings(
        paths, opener=opener, axis=axis, batch_rows=batch_rows, progress=progress,
        keyings=[Keying(name="", column="gene_target", labels=labels,
                        scope=f"{len(labels)} labels", resolution="per-perturbation")],
    )
    return pb, qc


def stream_keyings(
    paths: list[str],
    *,
    opener,
    axis: GeneAxis,
    keyings: list[Keying],
    batch_rows: int = 2048,
    progress: bool = True,
    guide_coverage: bool = False,
) -> list[tuple[PseudobulkSums, StreamQC]]:
    """One pass over `paths`, accumulating every keying at once; results in keying order.

    Peak RAM is the accumulators, about one file's compressed column chunks (pyarrow buffers a whole row group
    before the first batch), and per-batch (labels touched x genes) temporaries.
    """
    # Coverage is tallied on the primary accumulator only.
    accs = [_Accumulator(k.labels, axis.genes, column=k.column,
                         track_coverage=guide_coverage and i == 0)
            for i, k in enumerate(keyings)]
    columns = _columns_for(keyings, guide_coverage=guide_coverage)
    t0 = time.time()
    for n, path in enumerate(paths, 1):
        for acc in accs:
            acc.sources.append(path)
        first = True
        with opener(path) as handle:
            pf = pq.ParquetFile(handle)
            for record_batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
                # `first` clears only once the check has actually RUN. An early
                # return (every cell failed the guide filter, or none carried a
                # kept label) would otherwise disarm the raw-counts gate for the
                # whole file, making it depend on --batch-rows and row order
                # rather than on the data.
                ran = _consume_batch(accs, axis, record_batch.to_pandas(),
                                     check_counts=first)
                first = first and not ran
        if progress:
            cells = "  ".join(f"{k.name or 'perturbation'}={a.cells_seen:,}"
                              for k, a in zip(keyings, accs, strict=True))
            print(f"  [{n}/{len(paths)}] {Path(path).name}  "
                  f"cells {cells}  {time.time() - t0:6.0f}s", flush=True)

    out = []
    for keying, acc in zip(keyings, accs, strict=True):
        pb, qc = acc.to_pseudobulk(), acc.to_qc()
        _require_control(pb, acc, keying, n_paths=len(paths))
        out.append((pb, qc))
    return out


def _require_control(pb: PseudobulkSums, acc: _Accumulator, keying: Keying, *,
                     n_paths: int) -> None:
    """Raise if the control arm was asked for and yielded no cells, naming near-miss labels the data carried."""
    if CONTROL_LABEL not in keying.labels or CONTROL_LABEL in pb.labels:
        return
    # Near-misses come from the labels the DATA carried, not the ones we asked for --
    # the whole point is that what we declared is absent and something adjacent is
    # present. Matching case-insensitively here is safe: it only builds an error
    # message. The accumulator itself stays exact-match, always.
    low = CONTROL_LABEL.lower()
    near = sorted(lab for lab in acc.seen_labels
                  if lab != CONTROL_LABEL and (low in lab.lower() or lab.lower() in low))
    hint = (f" Labels present that resemble it: {near[:5]} -- if one of those is the "
            "real control, declare it exactly rather than matching loosely."
            if near else f" Labels that DID yield cells: {pb.labels[:5]}")
    keyed = f" (keyed by {keying.column})" if keying.column != "gene_target" else ""
    raise ValueError(
        f"control label {CONTROL_LABEL!r} accumulated zero cells across "
        f"{n_paths} file(s){keyed}. Every delta is measured against this arm, so an "
        f"aggregate without it is unusable.{hint}"
    )


# ------------------------------------------------------------------ lineage --


def write_lineage(out_dir: Path, *, provenance: Path, dataset: str, context: str,
                  axis: GeneAxis, pb: PseudobulkSums, qc: StreamQC,
                  scope: str, artifacts: dict[str, str], entry: str | None = None,
                  resolution: str = "per-perturbation",
                  keyed_by: str = "gene_target") -> Path:
    """Add this aggregate's entry, keyed `dataset/entry`, to the directory's LINEAGE.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "LINEAGE.json"
    payload = {"schema_version": 2, "entries": {}}
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("schema_version") == 2 and isinstance(existing.get("entries"), dict):
            payload = existing
        else:
            # An unrecognised file is moved aside, never overwritten.
            path.replace(path.with_suffix(".json.bak"))
    payload.setdefault("entries", {})

    record = {
        "dataset": dataset,
        "context": context,
        "derives_from": str(provenance),
        "code_sha": code_sha(),
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "accumulator": {
            "resolution": resolution,
            # Which column the labels ARE. Recorded because `resolution` is prose and this
            # is the fact a reader needs to know whether `labels` are gene symbols or
            # `sgRNA1|sgRNA2` construct strings before loading a multi-GB array.
            "keyed_by": keyed_by,
            "not_per_perturbation_batch": (
                "109 GEM batches x 18,330 targets is ~1.7 cells per bucket and a ~290 GB "
                "dense tensor; per-(perturbation, batch) CELL COUNTS are kept in the sidecar "
                "instead (report 07 section 2.3, overturning ADR 0003 on this point)"
            ),
            "scope": scope,
            "labels": len(pb.labels),
            "genes": int(axis.n_mapped),
            "gene_axis": "challenge-symbols" if axis.restricted else "all-symbols",
            "control_label": CONTROL_LABEL,
            "guide_filter": "pass_guide_filter == 1",
            "libsize": "sum over the EMITTED gene axis (mirrors stream_pseudobulk)",
        },
        "coverage": {
            "cells_accumulated": int(qc.cells_seen),
            # Counted BEFORE the label filter, so this spans every label in the file,
            # not just the ones accumulated. Named so the denominator is unambiguous.
            "cells_dropped_guide_filter_all_labels": int(qc.cells_dropped_filter),
            "cells_dropped_zero_libsize": int(qc.cells_dropped_zero),
            "batches_seen": len(qc.batches),
            "labels_in_corpus": int(qc.labels_in_corpus),
            "challenge_genes_mapped": int(axis.n_mapped),
            "challenge_genes_unmapped": len(axis.unmapped_challenge),
            "unmapped_genes": axis.unmapped_challenge,
            "collided_symbols": axis.collided_symbols,
        },
        "artifacts": artifacts,
        "source_files": len(pb.sources),
    }
    key = f"{dataset}/{entry or context}"
    payload["entries"][key] = record
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------- CLI --


def hf_opener(repo: str, revision: str):
    """Open a repo-relative path over HTTP range requests. Nothing lands on disk."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()

    def _open(name: str):
        return fs.open(f"datasets/{repo}@{revision}/{name}")

    return _open


def local_opener(root: Path):
    def _open(name: str):
        return open(Path(root).expanduser() / name, "rb")

    return _open


def _dataset_block(name: str, config: str) -> dict:
    import yaml

    cfg = yaml.safe_load(resolve_config(config).read_text())
    for block in cfg.get("datasets", []):
        if block["name"] == name:
            return block
    raise SystemExit(f"no dataset {name!r} in {config}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="xatlas_orion")
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--challenge-config", default="challenges/vcc2026/config.yaml")
    ap.add_argument("--root", type=Path, default=Path.home() / "data" / "sidechain")
    ap.add_argument("--context", required=True, help="which file group, e.g. hct116")
    ap.add_argument("--keep", help="CSV of labels to accumulate (the 300 panel); default all")
    ap.add_argument("--all-genes", action="store_true",
                    help="emit every X-Atlas symbol rather than only the challenge axis")
    ap.add_argument("--guide-coverage", action="store_true",
                    help="screen-QC tallies in the sidecar: cells per CRISPR construct over "
                         "EVERY construct, and cells-per-gene detection counts. Costs the "
                         "guide_target column. Implied by --guide-agreement")
    ap.add_argument("--guide-agreement", action="store_true",
                    help="also emit a construct-keyed pseudobulk over the multi-construct "
                         "targets (<out>.guide.npz). Costs the guide_target column on every "
                         "file, a construct-discovery pass, and a second accumulator")
    ap.add_argument("--limit-files", type=int, help="stream only the first N files (proving run)")
    ap.add_argument("--batch-rows", type=int, default=2048)
    ap.add_argument("--out", required=True, help="output prefix; .npz/.qc.npz land beside it")
    args = ap.parse_args(argv)

    from sidechain.data.loaders import load_challenge_config
    from sidechain.ingest.provenance import read_provenance

    block = _dataset_block(args.dataset, args.config)
    dest = args.root / block["dest"]
    provenance = read_provenance(dest)
    if provenance is None:
        raise SystemExit(
            f"no PROVENANCE.json at {dest}. The gate runs BEFORE any bytes move:\n"
            f"  uv run python -m sidechain.ingest.fetch --dataset {args.dataset}"
        )
    revision = provenance["record"]["version"]
    repo = provenance["record"]["record_id"]

    entry = next((f for f in block["files"]
                  if (f.get("spec") or {}).get("context") == args.context), None)
    if entry is None:
        contexts = [(f.get("spec") or {}).get("context") for f in block["files"]]
        raise SystemExit(f"no context {args.context!r} in {args.dataset}; have {contexts}")

    import fnmatch
    names = sorted(fnmatch.filter([f["name"] for f in provenance["selected"]], entry["name"]))
    if args.limit_files:
        names = names[: args.limit_files]
    if not names:
        raise SystemExit(f"no files matched {entry['name']!r} in the recorded provenance")

    opener = hf_opener(repo, revision)

    gene_map_name = next(f["name"] for f in block["files"] if f.get("kind") == "gene_map")
    with opener(gene_map_name) as handle:
        gene_map = pq.read_table(handle).to_pandas()

    cfg = load_challenge_config(args.challenge_config)
    data_dir = Path(cfg["data_dir"]).expanduser()
    challenge_genes = None if args.all_genes else read_gene_names(
        data_dir / cfg["gene_names_file"], expect=cfg.get("n_genes"))
    axis = build_gene_axis(gene_map, challenge_genes)

    pairs: set[tuple[str, str]] = set()
    if args.keep:
        keep = pd.read_csv(Path(args.keep).expanduser())
        col = "target_gene" if "target_gene" in keep.columns else keep.columns[0]
        labels = sorted(set(keep[col].astype(str)) | {CONTROL_LABEL})
        scope = f"panel+control ({len(labels)} labels)"
        if args.guide_agreement:
            # The keep list names targets, and the constructs behind them are only in the
            # data -- so this pass happens even though the label set did not need it.
            _, pairs = _discover_labels(names, opener, with_constructs=True)
    else:
        found, pairs = _discover_labels(names, opener,
                                        with_constructs=args.guide_agreement)
        labels = sorted(found)
        scope = f"all labels ({len(labels)})"

    keyings = [Keying(name="", column="gene_target", labels=labels, scope=scope,
                      resolution="per-perturbation")]
    if args.guide_agreement:
        guide_labels, n_multi = multi_construct_labels(pairs, keep=set(labels))
        if n_multi == 0:
            raise SystemExit(
                "--guide-agreement found no target with more than one construct among the "
                f"{len(labels)} labels being accumulated. There is nothing to compare, so "
                "the construct-keyed aggregate would be an expensive copy of the one beside "
                "it. Drop the flag, or widen --keep."
            )
        keyings.append(Keying(
            name="guide", column="guide_target", labels=guide_labels,
            scope=(f"multi-construct targets only ({n_multi} targets, "
                   f"{len(guide_labels) - 1} constructs) + control"),
            resolution="per-construct (multi-construct targets only; control pooled)",
        ))

    print(f"{args.dataset}:{args.context}  {len(names)} files  {scope}")
    print(f"  gene axis: {axis.n_mapped} mapped"
          + (f", {len(axis.unmapped_challenge)} challenge genes absent" if challenge_genes else "")
          + (f", {len(axis.collided_symbols)} symbol collisions" if axis.collided_symbols else ""))
    total_ram = 0.0
    for keying in keyings:
        ram = len(keying.labels) * axis.n_mapped * 8 * 3 / 1e9
        total_ram += ram
        print(f"  accumulator [{keying.name or 'perturbation'}]: {len(keying.labels)} x "
              f"{axis.n_mapped} x 3 float64 = {ram:.2f} GB  -- {keying.scope}")
    print(f"  accumulators total: {total_ram:.2f} GB\n")

    # --guide-agreement already pays for the guide_target column, so the coverage tally on
    # top of it is free; asking for it separately is the cheap QC-only path.
    guide_coverage = args.guide_coverage or args.guide_agreement
    results = stream_keyings(names, opener=opener, axis=axis, keyings=keyings,
                             batch_rows=args.batch_rows, guide_coverage=guide_coverage)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    lineage = None
    for keying, (pb, qc) in zip(keyings, results, strict=True):
        stem = out.name + (f".{keying.name}" if keying.name else "")
        npz = out.with_name(stem + ".npz")
        qc_npz = out.with_name(stem + ".qc.npz")
        pb.save(npz)
        qc.save(qc_npz)
        lineage = write_lineage(
            out.parent, provenance=dest / "PROVENANCE.json", dataset=args.dataset,
            context=args.context, axis=axis, pb=pb, qc=qc, scope=keying.scope,
            artifacts={"pseudobulk": npz.name, "qc_sidecar": qc_npz.name},
            entry=stem, resolution=keying.resolution, keyed_by=keying.column,
        )
        covered = int((pb.n_cells > 0).sum())
        print(f"\n  [{keying.name or 'perturbation'}] keyed by {keying.column}")
        print(f"  labels with >=1 cell : {covered}/{len(keying.labels)}")
        print(f"  cells accumulated    : {qc.cells_seen:,}")
        print(f"  labels in corpus     : {qc.labels_in_corpus:,}")
        print(f"  batches seen         : {len(qc.batches)}")
        if qc.gene_cells is not None and qc.gene_cells.size:
            never = int((qc.gene_cells == 0).sum())
            print(f"  genes detected in <10 cells : "
                  f"{int((qc.gene_cells < 10).sum()):,} of {qc.gene_cells.size:,}"
                  f"  (never detected: {never:,})")
        if qc.guide_cells is not None and qc.guide_cells.size:
            g = qc.guide_cells
            print(f"  constructs                  : {g.size:,}"
                  f"   median {int(np.median(g)):,} cells"
                  f"   <10 cells: {int((g < 10).sum()):,}"
                  f"   <25: {int((g < 25).sum()):,}")
        print(f"  -> {npz}")
    print(f"  -> {lineage}")
    return 0


def _discover_labels(names: list[str], opener, *, with_constructs: bool = False,
                     progress: bool = True) -> tuple[set[str], set[tuple[str, str]]]:
    """A first pass over the label column(s) only; with `with_constructs`, also the distinct (target, construct)
    pairs, the control excluded."""
    found: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    columns = ["gene_target", "guide_target"] if with_constructs else ["gene_target"]
    t0 = time.time()
    for n, name in enumerate(names, 1):
        with opener(name) as handle:
            table = pq.ParquetFile(handle).read(columns=columns)
        found.update(table.column("gene_target").to_pylist())
        if with_constructs:
            frame = table.to_pandas()
            frame = frame[frame["gene_target"] != CONTROL_LABEL].drop_duplicates()
            pairs.update((str(t), str(g)) for t, g in frame.to_numpy())
        if progress and (n % 10 == 0 or n == len(names)):
            print(f"  discover [{n}/{len(names)}] labels={len(found):,}"
                  + (f"  constructs={len(pairs):,}" if with_constructs else "")
                  + f"  {time.time() - t0:6.0f}s", flush=True)
    return found, pairs


def multi_construct_labels(
    pairs: set[tuple[str, str]], *, keep: set[str] | None = None,
) -> tuple[list[str], int]:
    """The constructs of every target with more than one in the corpus, plus the control.

    `keep` restricts to targets the primary accumulator keeps. Returns (sorted labels, number of multi-construct
    targets).
    """
    by_target: dict[str, set[str]] = {}
    for target, construct in pairs:
        by_target.setdefault(target, set()).add(construct)

    labels: set[str] = set()
    n_targets = 0
    for target, constructs in by_target.items():
        if len(constructs) < 2 or (keep is not None and target not in keep):
            continue
        n_targets += 1
        labels |= constructs
    return sorted(labels | {CONTROL_LABEL}), n_targets


if __name__ == "__main__":
    sys.exit(main())
