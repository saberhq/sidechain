"""Stream long-format parquet (X-Atlas/Orion) into the same `PseudobulkSums` the h5ad streamer emits.

    uv run python -m sidechain.data.stream_parquet_pseudobulk --dataset xatlas_orion \
        --context hct116 --keep ~/data/sidechain/vcc2026/pert_counts.csv \
        --out ~/data/sidechain/derived/xatlas-orion/hct116_panel

    uv run python -m sidechain.data.stream_parquet_pseudobulk --dataset xatlas_orion \
        --context hct116 --all-genes --guide-agreement \
        --out ~/data/sidechain/derived/xatlas-orion/hct116_full

Why this module exists as a separate reader: X-Atlas/Orion is 126.26 GB of parquet on
Hugging Face that we read ONCE and never store (`configs/datasets.yaml` → `route: stream`).
Its on-disk shape is nothing like an h5ad -- one row per cell, with the non-zero genes held
as two parallel *list* columns -- so `stream_pseudobulk` cannot read it.

**Why it emits `PseudobulkSums` and not something better suited.** That object is the entire
integration contract. `submit.build.pooled_delta` and `eval.loco` consume it and nothing else,
so emitting it makes X-Atlas a new `--source` on the existing command lines with no change to
the model, the emitter or the scorer. If this module ever needed something downstream to
change, the shape would be wrong.

RESOLUTION: PER PERTURBATION, NOT PER (PERTURBATION, BATCH).
ADR 0003 instructed the opposite -- budget the aggregate at per-(perturbation, batch)
resolution because re-streaming is expensive. The arithmetic refutes it and this is a one-way
door at 126 GB per re-run, so it is stated here rather than left in a report: HCT116 has 109
GEM batches over 18,330 targets, which is ~1.7 cells per (perturbation, batch) bucket. A
1.7-cell pseudobulk is not a measurement, and the dense tensor would be ~290 TB. X-Atlas
reaches its median 141 cells per perturbation precisely BY pooling across batches.
What is kept instead is a per-(perturbation, batch) *cell-count* table -- ~8 MB, preserving
every batch-structure question we might want to ask, for nothing.

CAPTURED HERE OR LOST: `pct_counts_mt` and those per-batch cell counts. Free while the cells
stream past, another 126 GB afterwards. Mitochondrial read fraction rises in stressed and
dying cells, and knocking down an essential gene kills cells -- so a perturbation whose
"effect" is largely a dying-cell signature would transfer to the challenge contexts as if it
were biology. Nothing consumes these yet. They are taken anyway, into a sidecar file, because
the alternative is re-reading the corpus to get them.

Same argument, second instance: `--guide-agreement` accumulates a SECOND pseudobulk in the
same pass, keyed by `guide_target` (the dual-guide construct) instead of `gene_target` (the
gene it silences). It answers one question nothing else in the corpus can: do two
independently delivered constructs against the same gene produce the same profile? That is
the empirical size of the construct-level noise we are currently ignoring when a target is
measured through one construct only -- which is 16,832 of 18,330 targets. See
`Keying` and `multi_construct_labels` for what it is restricted to and why.

The four facts about this corpus that are hardcoded, each verified against the data rather
than the README (2026-08-23, `data/HCT116_Batch106.parquet`):

  * control label is exactly `Non-Targeting` -- matched with `checks.control_mask`, which is
    an equality test by design. Never a substring: `"NT" in value` once reported INTS1 and
    DOT1L_INTS1 as controls;
  * `pass_guide_filter` is stored as **int64, not bool** -- 1/0, so it is compared to 1;
  * counts are raw integers stored as float64 (min 1, max 1631, all integral) -- confirmed
    with `checks.counts_state` on the first block of every file, not assumed;
  * each file is ONE row group, so "read one row group at a time" would not bound memory
    at all. `iter_batches` streams the DECODE within that row group, which is what keeps the
    dense temporaries small -- but pyarrow still buffers the requested column chunks for the
    whole row group before yielding the first batch. Measured on an 82.8 MB single-row-group
    file: 82.9 MB had been read before batch one. So peak is **the accumulator plus roughly
    one file's compressed column chunks**, not "plus one batch". Real files run 0.23-0.72 GB
    (mean 0.43 GB), so budget ~1 GB of headroom on top of the accumulator.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.ingest import checks
from sidechain.utils.paths import resolve_config

CONTROL_LABEL = "Non-Targeting"

# The columns this reader touches. Named explicitly because parquet is columnar: asking for
# seven of thirteen columns means the other six are never fetched over the network, which on
# a 126 GB corpus is most of the saving.
#
# `guide_target` is deliberately NOT here: it is appended by `_columns_for` only when a
# construct-keyed accumulator asks for it, so a run that does not want guide agreement pays
# nothing for the column.
COLUMNS = [
    "gene_token_id",      # list<int64>  -- the non-zero genes for this cell
    "gene_expression",    # list<double> -- their raw counts, parallel to the above
    "gene_target",        # the perturbation label; `Non-Targeting` is the control
    "sample",             # the GEM batch
    "pass_guide_filter",  # int64 1/0, NOT bool
    "pct_counts_mt",      # captured now or re-read later at 126 GB
    "total_counts",       # the cell's TRUE library size, over all 38,606 genes
]


@dataclass
class Keying:
    """One way of grouping cells into labels, and the labels to allocate for it.

    Two exist, and both are accumulated in the SAME pass because the pass costs 126 GB of
    network and the second accumulator costs 2.8 GB of RAM.

      * `gene_target` -- the gene that was silenced. This is what every downstream consumer
        means by a perturbation, and the emitted `PseudobulkSums` under this keying is the
        one `pooled_delta` and `eval.loco` read.
      * `guide_target` -- the dual-guide CONSTRUCT that did the silencing, e.g.
        `USP22_P1P2-1|USP22_P1P2-2`. Nothing consumes this yet; it exists so that
        "how much is a single-construct estimate worth relative to a two-construct one?"
        can be answered without re-streaming the corpus.

    `name` is the suffix the artifacts and the `LINEAGE.json` entry take -- empty for the
    primary keying, so `hct116_full.npz` keeps its name and the construct-keyed pair lands
    beside it as `hct116_full.guide.npz`.
    """

    name: str
    column: str
    labels: list[str]
    scope: str
    resolution: str


def label_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    """The per-cell label under one keying.

    Under `guide_target` the CONTROL is deliberately still keyed by gene. X-Atlas pairs its
    1,026 non-targeting sgRNAs at random, so `Non-Targeting` cells carry tens of thousands
    of distinct construct strings, most of them seen once or twice. Keying those by
    construct would replace the single control arm every delta is measured against with a
    cloud of one-cell labels -- and `stream_keyings` would then stop the run, correctly, for
    having accumulated no control. Control cells therefore keep the label `Non-Targeting`
    under both keyings, and only the targeting cells split by construct.
    """
    if column == "gene_target":
        return frame["gene_target"].to_numpy()
    gene = frame["gene_target"].to_numpy()
    return np.where(gene == CONTROL_LABEL, CONTROL_LABEL, frame[column].to_numpy())


def _columns_for(keyings: list[Keying]) -> list[str]:
    """`COLUMNS` plus whatever the requested keyings key on. Order is stable."""
    cols = list(COLUMNS)
    for keying in keyings:
        if keying.column not in cols:
            cols.append(keying.column)
    return cols


@dataclass
class StreamQC:
    """What the stream saw, beside the sums. Saved as a sidecar, never inside `PseudobulkSums`.

    A sidecar rather than extra fields because `PseudobulkSums` is the integration contract:
    widening it would mean every producer and consumer of it moves together, to carry
    something only this corpus has. These are recorded so the question "was that big delta a
    dying-cell signature?" can be asked without re-streaming 126 GB.
    """

    labels: list[str]
    batches: list[str]
    batch_cells: np.ndarray        # (L, B) int32 -- per-(perturbation, batch) cell counts
    pct_mt_sum: np.ndarray         # (L,) float64
    pct_mt_sq_sum: np.ndarray      # (L,) float64
    total_counts_sum: np.ndarray   # (L,) float64 -- the corpus's own `total_counts`, over ALL
    #                                38,606 genes. NOT the same as PseudobulkSums.libsize_sum,
    #                                which sums the emitted axis only; keeping both is what
    #                                makes the alternative CPM normalization recoverable
    #                                without re-streaming 126 GB.
    n_cells: np.ndarray            # (L,) int64
    cells_seen: int = 0
    cells_dropped_filter: int = 0
    cells_dropped_zero: int = 0
    labels_in_corpus: int = 0   # distinct gene_target values seen, kept or not

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
    # Whether the axis was RESTRICTED to the challenge genes. Carried rather than inferred
    # from `unmapped_challenge` being empty: a source covering every challenge gene would
    # otherwise be recorded as an unrestricted all-symbols axis, which is a different and
    # much larger axis.
    restricted: bool = True

    @property
    def n_mapped(self) -> int:
        return len(self.genes)


def read_gene_names(path: str | Path, *, expect: int | None = None) -> list[str]:
    """The 2026 `gene_names.csv`, read the way that file actually needs.

    It HAS a header row (`gene_name`) -- the opposite of the 2025 file. A `header=None` read
    returns 18,534 rows whose first "gene" is the literal string `gene_name`, and every gene
    after it is off by one. Nothing downstream would raise; the axis would simply be wrong.
    So the count is checked against what the config declares rather than trusted.
    """
    genes = pd.read_csv(Path(path).expanduser()).iloc[:, 0].astype(str).tolist()
    if expect is not None and len(genes) != expect:
        raise ValueError(
            f"{path} read as {len(genes)} genes but the config says {expect}. This is the "
            "header trap: the 2026 file has a header row and the 2025 file does not. Do not "
            '"fix" it by switching header=None -- check which file this is.'
        )
    return genes


def build_gene_axis(gene_map: pd.DataFrame, challenge_genes: list[str] | None) -> GeneAxis:
    """Map X-Atlas gene tokens onto the axis we emit.

    `gene_map` is `metadata/gene_metadata.parquet`: 38,606 rows of
    (`ensembl_id`, `gene_name`, `gene_token_id`). The long-format data files carry only
    `gene_token_id`, so without this table the counts sit on an unnamed axis.

    When `challenge_genes` is given the emitted axis is restricted to the challenge symbols
    X-Atlas actually carries, in challenge order. That is not a shortcut: `pooled_delta`
    remaps every source onto the challenge axis by symbol anyway, so genes outside it are
    dropped downstream regardless -- carrying all 38,606 would only double the accumulator.
    The challenge genes X-Atlas lacks are RECORDED rather than silently dropped, because
    "this source is blind to these 427 genes" is a fact about our coverage, not a bug.

    Duplicate symbols (38,606 ids over 38,584 names) are summed into one column. Resolving
    by Ensembl id would be more principled, but the emitted axis has to be symbols -- that is
    what the challenge axis is keyed on, and the 2026 bundle ships no Ensembl ids at all.
    Which symbols collided is recorded so the cost is visible rather than assumed to be zero.
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

    def __init__(self, labels: list[str], genes: np.ndarray, column: str = "gene_target"):
        self.labels = labels
        self.genes = genes
        # Which column this accumulator keys on -- see `Keying`. Held here rather than
        # passed in per batch so that one `_fold_batch` serves both keyings and the primary
        # arithmetic stays literally the code that produced the panel artifacts.
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
        # Every distinct label the corpus actually contains UNDER THIS KEYING, whether or
        # not we accumulate it. Bounded by the guide library (~18.3k targets or ~20.9k
        # constructs, short strings, well under a megabyte) and it answers two questions
        # nothing else can: "did you mean this label?" when a declared one matches nothing,
        # and "what else is in here?" without re-reading 126 GB.
        self.seen_labels: set[str] = set()

    def batch_row(self, batch: str) -> np.ndarray:
        if batch not in self.batch_cells:
            self.batch_cells[batch] = np.zeros(len(self.labels), dtype=np.int32)
        return self.batch_cells[batch]

    def to_pseudobulk(self) -> PseudobulkSums:
        """The sums, with never-observed labels PRUNED.

        The accumulator is allocated for every requested label up front, because growing a
        (301 x 18,106) array mid-stream is not an option. But a label the corpus never
        actually perturbed must not survive into the emitted object, and this is not
        tidiness -- it is a correctness fix with a specific failure behind it:

        `pooled_delta` decides whether a source votes on a target with `target not in
        pb.labels`. A label that is present with zero cells passes that test, and
        `_log2fc_with_var` then reads mean_cpm = 0 / max(0, 1) = 0 for every gene and
        returns log2(1 / (ctrl + 1)) -- i.e. "this perturbation silenced the entire
        transcriptome", around -7 to -10 log2 per gene, carrying real (if modest) weight
        into the inverse-variance pool. A source that never saw the target would be voting
        confidently and wrongly on it.

        `stream_pseudobulk` never had this problem because it intersects the keep list with
        the labels actually present in the file. This restores that parity.
        """
        seen = np.flatnonzero(self.n_cells > 0)
        return PseudobulkSums(
            labels=[self.labels[i] for i in seen], genes=self.genes,
            count_sum=self.count_sum[seen], cpm_sum=self.cpm_sum[seen],
            cpm_sq_sum=self.cpm_sq_sum[seen], n_cells=self.n_cells[seen],
            libsize_sum=self.libsize_sum[seen], sources=list(self.sources),
        )

    def to_qc(self) -> StreamQC:
        """The sidecar keeps EVERY requested label, including the never-observed ones.

        Deliberately the opposite choice to `to_pseudobulk`: "the panel asked for 300 targets
        and this line yielded cells for 287" is the coverage number, and it is only
        answerable if the zeros are still here.
        """
        batches = sorted(self.batch_cells)
        table = (np.stack([self.batch_cells[b] for b in batches], axis=1)
                 if batches else np.zeros((len(self.labels), 0), dtype=np.int32))
        return StreamQC(
            labels=list(self.labels), batches=batches, batch_cells=table,
            pct_mt_sum=self.pct_mt_sum, pct_mt_sq_sum=self.pct_mt_sq_sum,
            total_counts_sum=self.total_counts_sum, n_cells=self.n_cells,
            cells_seen=self.cells_seen, cells_dropped_filter=self.dropped_filter,
            cells_dropped_zero=self.dropped_zero, labels_in_corpus=len(self.seen_labels),
        )


def _consume_batch(accs: list[_Accumulator], axis: GeneAxis, frame: pd.DataFrame, *,
                   check_counts: bool) -> bool:
    """Apply the guide filter once, then fold the surviving cells into every accumulator.

    The filter is shared because it is a property of the cell, not of the keying: a cell
    whose guide call did not pass is not a cell under either keying. Everything after it is
    per accumulator, because the two disagree about which label a cell belongs to and about
    which labels are wanted at all.
    """
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


def _fold_batch(acc: _Accumulator, axis: GeneAxis, frame: pd.DataFrame, *,
                check_counts: bool) -> bool:
    """Fold one guide-filtered frame into one accumulator.

    The long -> sparse step is the whole of it: each cell carries two parallel lists, so the
    batch becomes one COO of (row=cell within batch, col=emitted gene) and the per-label sums
    fall out of an indicator matrix product -- the same shape `stream_pseudobulk` uses, for
    the same reason (it is a sparse matmul rather than a Python loop over cells).

    Everything expensive happens AFTER the label filter, which is what keeps a second
    accumulator cheap: the construct-keyed one wants ~8% of the cells, so it pays ~8% of the
    sparse work rather than doing the whole batch again.
    """
    targets = label_column(frame, acc.column)
    acc.seen_labels.update(targets.tolist())
    codes = np.array([acc.code_of.get(t, -1) for t in targets], dtype=np.int64)
    wanted = codes >= 0
    if not wanted.any():
        return False
    frame = frame[wanted]
    codes = codes[wanted]

    tok_lists = frame["gene_token_id"].to_numpy()
    val_lists = frame["gene_expression"].to_numpy()
    lengths = np.fromiter((len(t) for t in tok_lists), count=len(tok_lists), dtype=np.int64)
    tokens = np.concatenate(tok_lists) if len(tok_lists) else np.empty(0, dtype=np.int64)
    values = np.concatenate(val_lists) if len(val_lists) else np.empty(0, dtype=np.float64)
    rows = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)

    ran_check = False
    if check_counts and values.size:
        ran_check = True
        # Answer the normalization question from the data, not from the card.
        #
        # Give `counts_state` REAL ROWS, not one flattened vector. Two of its three signals
        # need row structure: constant row totals is how it catches a cp10k matrix whose
        # values happen to land back on integers, and that signal is dead on a single row.
        # Long format has no row structure, so it is rebuilt here from the `lengths` already
        # computed above -- a ragged set of cells padded into a rectangle, which is enough
        # for "are these integers" and for "is every library size identical".
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

    # Tokens outside the emitted axis (and any id beyond the map) are dropped here.
    # `tokens >= 0` is not paranoia: a negative id would index from the END of
    # col_of_token and land silently in some unrelated gene's column. Gene
    # misalignment is this project's most expensive recurring bug and it never
    # announces itself.
    in_range = (tokens >= 0) & (tokens < len(axis.col_of_token))
    cols = np.full(tokens.shape, -1, dtype=np.int64)
    cols[in_range] = axis.col_of_token[tokens[in_range]]
    on_axis = cols >= 0

    n_cells, n_genes = len(codes), len(acc.genes)
    sub = sp.csr_matrix(
        (values[on_axis], (rows[on_axis], cols[on_axis])),
        shape=(n_cells, n_genes),
    )

    lib = np.asarray(sub.sum(axis=1)).ravel()
    alive = lib > 0
    acc.dropped_zero += int((~alive).sum())
    if not alive.any():
        return ran_check
    sub = sub[alive]
    codes = codes[alive]
    lib = lib[alive]
    frame = frame[alive]

    cpm = sp.diags(1e6 / lib) @ sub

    # Only the labels this batch actually touched get densified, so the transient is
    # (labels in batch x genes) rather than (all labels x genes) -- which is what lets the
    # 18,331-label run stay inside the box's memory.
    uniq, local = np.unique(codes, return_inverse=True)
    ind = sp.csr_matrix(
        (np.ones(len(local)), (local, np.arange(len(local)))), shape=(len(uniq), len(local))
    )
    acc.count_sum[uniq] += (ind @ sub).toarray()
    acc.cpm_sum[uniq] += (ind @ cpm).toarray()
    acc.cpm_sq_sum[uniq] += (ind @ cpm.multiply(cpm)).toarray()

    np.add.at(acc.n_cells, codes, 1)
    np.add.at(acc.libsize_sum, codes, lib)
    pct = frame["pct_counts_mt"].to_numpy(dtype=np.float64)
    np.add.at(acc.pct_mt_sum, codes, pct)
    np.add.at(acc.pct_mt_sq_sum, codes, pct * pct)
    # The corpus's OWN total_counts, not `lib`. `lib` sums the emitted axis (18,106 genes);
    # this is the cell's whole transcriptome (38,606). Recording `lib` here would have made
    # this field a duplicate of libsize_sum under a name promising something else, and the
    # alternative normalization would then need another 126 GB to recover.
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
    """One pass over `paths`, accumulating per perturbation. The ordinary case.

    A thin call into `stream_keyings` with the single `gene_target` keying, kept because
    that is what every caller but the guide-agreement run wants and the two-value return
    reads better than indexing a list of one.
    """
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
) -> list[tuple[PseudobulkSums, StreamQC]]:
    """One pass over `paths`, accumulating every keying at once. Results are in order.

    Several keyings share a pass because the pass is the expensive thing: 126 GB of network
    against a few GB of RAM per extra accumulator. They are NOT separate runs of this
    function, because two runs would read the corpus twice and could disagree -- the second
    read is not guaranteed to see the same revision, and even where it does, an aggregate
    pair that came from two passes cannot be compared cell for cell.

    Peak RAM is the accumulators plus about one file's column chunks (see the module
    docstring -- pyarrow buffers the whole row group before the first batch, and each
    X-Atlas file is a single row group of 0.23-0.72 GB), plus the per-batch dense
    temporaries, which are (labels touched by the batch x genes) per accumulator.
    """
    accs = [_Accumulator(k.labels, axis.genes, column=k.column) for k in keyings]
    columns = _columns_for(keyings)
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
    """Stop the run if the control arm was asked for and yielded nothing.

    The control is the arm every delta is measured against. If it was asked for and yielded
    nothing, the prune removes it, `.npz` and `LINEAGE.json` still get written, and the
    failure surfaces much later inside `pooled_delta` as
    `ValueError: 'Non-Targeting' is not in list` -- a message that names neither this corpus
    nor this run. Stop here instead, with the labels we did see.

    This is what `checks.control_mask` does for an in-memory matrix (it refuses to return an
    all-False mask, and offers near-miss labels). It cannot be called on a stream that never
    holds the labels at once, so the guarantee is reproduced.
    """
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


def code_sha() -> str:
    """The commit this aggregate was built at, or `dirty:<sha>` / `unknown`."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True, cwd=Path(__file__).resolve().parent).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True,
                               cwd=Path(__file__).resolve().parent).stdout.strip()
        return f"dirty:{sha}" if dirty else sha
    except Exception:  # noqa: BLE001 - lineage must never be the thing that fails a 4h stream
        return "unknown"


def write_lineage(out_dir: Path, *, provenance: Path, dataset: str, context: str,
                  axis: GeneAxis, pb: PseudobulkSums, qc: StreamQC,
                  scope: str, artifacts: dict[str, str], entry: str | None = None,
                  resolution: str = "per-perturbation",
                  keyed_by: str = "gene_target") -> Path:
    """Record this aggregate in the directory's LINEAGE.json.

    ADR 0003 section 1 sketched this file and nothing wrote it until now. It answers the
    question that otherwise costs 126 GB to re-answer: which PROVENANCE.json this derives
    from, which code built it, and -- the one that matters most -- what the accumulator's
    resolution was. An aggregate whose resolution is unknown cannot be trusted or extended.

    One file per derived DIRECTORY, holding one entry per artifact, because a directory
    holds several: X-Atlas alone puts HCT116 and HEK293T side by side, and a panel-scope run
    beside a full-corpus one. An earlier version wrote the whole file per run, so streaming
    the second line silently erased the first line's lineage -- leaving two aggregates and a
    record for one of them, which is worse than no record because it looks complete.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "LINEAGE.json"
    payload = {"schema_version": 2, "entries": {}}
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("schema_version") == 2 and isinstance(existing.get("entries"), dict):
            payload = existing
        else:
            # An unrecognised file is someone else's record, not ours to discard. Move it
            # aside rather than overwriting: losing lineage is worse than an extra file,
            # because the aggregate it described stays on disk looking trustworthy.
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
                "109 GEM batches x 18,330 targets is ~1.7 cells per bucket and a ~290 TB "
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

    results = stream_keyings(names, opener=opener, axis=axis, keyings=keyings,
                             batch_rows=args.batch_rows)

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
        print(f"  -> {npz}")
    print(f"  -> {lineage}")
    return 0


def _discover_labels(names: list[str], opener, *, with_constructs: bool = False,
                     progress: bool = True) -> tuple[set[str], set[tuple[str, str]]]:
    """A cheap first pass over the label column(s) only.

    Parquet is columnar, so reading `gene_target` alone touches a small fraction of each
    file. Worth it because the accumulator has to be allocated once, up front, at its final
    size -- growing an (18,331 x 38,584) array mid-stream is not an option.

    With `with_constructs`, `guide_target` is read alongside and the distinct
    (target, construct) pairs come back too. **`Non-Targeting` is excluded from the pairs**:
    X-Atlas combines its 1,026 non-targeting sgRNAs at random, so the control's distinct
    construct strings run into the tens of thousands and grow with the corpus, while every
    targeting construct is a designed pair from the library and the whole set is bounded by
    ~20,890. Keeping the control's pairs would make this set unbounded for no use -- nothing
    keys the control by construct (`label_column`).
    """
    found: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    columns = ["gene_target", "guide_target"] if with_constructs else ["gene_target"]
    # Reported every few files. Without it this phase is a long silent wait over 223 files
    # before the first stream line appears, which is indistinguishable from a hung job --
    # and on a box that bills by the hour, "is it stuck?" is an expensive question.
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
    """The constructs of every target measured through MORE THAN ONE, plus the control.

    Guide agreement asks whether two independently delivered constructs against the same
    gene produce the same profile. That is only answerable where a target has more than one,
    so a single-construct target contributes nothing: its construct profile IS its gene
    profile, already accumulated under the primary keying. Restricting is therefore not
    thrift -- carrying all ~20,890 constructs on the 38,584-gene axis would cost ~19 GB of
    RAM to store a copy of numbers we already have.

    The set is derived from the CORPUS, not from the published guide library, because the
    question is which constructs this context actually yielded cells for. A target the
    library gives two constructs but HCT116 only ever delivered one of belongs in neither
    accumulator's numerator, and reading the library would have put it there.

    `keep` restricts to targets the primary accumulator is also keeping, so the construct
    aggregate is always a refinement of the aggregate beside it rather than a different
    population that happens to share a directory.

    Returns the labels (constructs, sorted, plus `Non-Targeting`) and the number of
    multi-construct targets behind them.
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
