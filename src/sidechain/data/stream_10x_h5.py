"""Read Cell Ranger feature-barcode h5 lanes into the same `PseudobulkSums` every streamer emits.

    D=~/data/sidechain/derived/pan2026-h1
    uv run python -m sidechain.data.stream_10x_h5 --dataset pan2026_h1_crispri \
        --guide-level --groups $D/lane_groups.tsv --axis-aliases $D/axis_aliases.tsv \
        --target-aliases $D/target_aliases.tsv --out $D/h1_pan2026

Why a separate reader: a Cell Ranger `filtered_feature_bc_matrix.h5` is not an h5ad. It has no
obs table and no perturbation column. The matrix is stored features x barcodes, and when the
library was captured directly the guide counts are ROWS OF THAT SAME MATRIX (feature type
`CRISPR Guide Capture`) beside the genes (`Gene Expression`). So which cell got which guide is
not read from the file, it is DERIVED from it, and the rule that derives it is a fact about
the experiment. Every such rule is declared in the dataset's block in `configs/datasets.yaml`
(`min_genes`, `guide_min_umi`, `guide_rule`, the two regexes, `n_control_guides`) and read
here from the block; none is a default in this module. Pan 2026's H1 screen (GEO GSE295214,
T114) is the corpus this was written for.

What a run writes, beside `<out>`:

  <out>.npz              per-target sums over every lane -- the `--source` a pool reads
  <out>.<group>.npz      the same per lane group, when `--groups` names groups. Lanes are the
                         replicate unit of a screen like this one, so a split-half is a merge
                         of groups; one gene-level accumulator is resident at a time (the
                         per-guide one, when asked for, stays for the whole run and is the
                         larger of the two)
  <out>.guide.npz        per-guide sums over every lane (`--guide-level`), controls pooled
  <out>.qc.npz           the `StreamQC` sidecar X-Atlas writes, lanes as its batches
  <out>.cells.parquet    one row per barcode in the filtered matrices, KEPT OR NOT: lane,
                         barcode, the guide call with its top two guide UMI counts, and why a
                         cell was dropped. The only place the cell <-> guide link is written
  <out>.control_targets.npz
                         control cells x target genes, raw counts: what a per-target test
                         against the control arm needs (the paper's own knockdown filter is
                         a Mann-Whitney U against these cells)
  <out>.lanes.tsv        one row per lane: size, sha256, cells kept and dropped by reason
  LINEAGE.json           one entry per artifact, in the directory

`libsize` is the sum over the EMITTED gene axis, as in the other two streamers; the cell's
total over all gene features is kept in the per-cell table and in the sidecar.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sidechain.data.gene_aliases import RETIRED_SYMBOLS
from sidechain.data.stream_parquet_pseudobulk import (
    StreamQC,
    _dataset_block,
    _scatter_add,
    read_gene_names,
)
from sidechain.data.stream_pseudobulk import PseudobulkSums, merge
from sidechain.ingest import checks
from sidechain.utils.logging import code_sha

# The keys an expression file's spec must carry for this reader, beyond the registry's own
# required set. Missing ones are a refusal: a filter threshold that defaulted silently is a
# cell arm nobody declared.
RULE_KEYS = ("control_label", "expression_feature_type", "guide_feature_type", "min_genes",
             "guide_min_umi", "guide_rule", "guide_target_regex", "control_guide_regex",
             "n_control_guides")

# Guides per cell are also counted at these UMI thresholds, whatever the calling rule is:
# many guides per cell at low thresholds is what a corpus that cannot be read by label
# looks like (GSE249595, 2026-10-09), and it is cheap here and a re-read afterwards.
GUIDE_SHAPE_UMIS = (2, 5, 10)

# A group's sums are saved as <out>.<group>.npz, beside these three.
RESERVED_GROUPS = ("guide", "qc", "control_targets")

DROP_LOW_GENES = "low_genes"
DROP_NO_GUIDE = "no_guide"
DROP_MULTI_GUIDE = "multi_guide"
DROP_ZERO_LIBSIZE = "zero_libsize"


@dataclass(frozen=True)
class LaneRules:
    """How a cell gets its label, as the dataset's block declares it."""

    control_label: str
    expression_feature_type: str
    guide_feature_type: str
    min_genes: int
    guide_min_umi: int
    guide_target_regex: str
    control_guide_regex: str
    n_control_guides: int

    @classmethod
    def from_spec(cls, spec: dict) -> LaneRules:
        missing = [k for k in RULE_KEYS if spec.get(k) is None]
        if missing:
            raise ValueError(
                f"spec declares no {missing}. A Cell Ranger h5 carries no perturbation "
                "column: the cell filter and the guide call are rules read from the paper's "
                "methods, and they are declared in configs/datasets.yaml, never defaulted here.")
        if spec["guide_rule"] != "exactly_one":
            raise ValueError(f"guide_rule {spec['guide_rule']!r} is not implemented; "
                             "only 'exactly_one' (one guide at or above guide_min_umi) is")
        if isinstance(spec["control_label"], (list, tuple)):
            raise TypeError("control_label is a list; this reader collapses every control "
                            "guide to ONE label and needs a single string")
        if "target" not in re.compile(spec["guide_target_regex"]).groupindex:
            raise ValueError("guide_target_regex must define a named group `target`")
        return cls(
            control_label=str(spec["control_label"]),
            expression_feature_type=str(spec["expression_feature_type"]),
            guide_feature_type=str(spec["guide_feature_type"]),
            min_genes=int(spec["min_genes"]), guide_min_umi=int(spec["guide_min_umi"]),
            guide_target_regex=str(spec["guide_target_regex"]),
            control_guide_regex=str(spec["control_guide_regex"]),
            n_control_guides=int(spec["n_control_guides"]),
        )


@dataclass
class Lane:
    """One lane as stored, turned the way the rest of the code reads a matrix."""

    name: str
    barcodes: np.ndarray       # (n_cells,) str
    X: sp.csr_matrix           # (n_cells, n_features) raw counts
    feature_ids: np.ndarray    # (n_features,) str -- a gene's Ensembl id, a guide's name
    feature_names: np.ndarray  # (n_features,) str -- the symbol for a gene
    feature_types: np.ndarray  # (n_features,) str


def _decode(values) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object)


def read_lane(path: str | Path, name: str | None = None) -> Lane:
    """One `filtered_feature_bc_matrix.h5` -> cells x features CSR.

    Cell Ranger stores the matrix compressed by BARCODE with the feature index in
    `indices` and `shape = (features, barcodes)`: a CSC of features x barcodes. The same
    three arrays read as a CSR are barcodes x features, which is the orientation every
    other reader here uses, so nothing is transposed (the index array is narrowed to
    int32, which is the one copy). The lengths
    are checked because the opposite reading also builds without error whenever the two
    counts happen to allow it, and yields a matrix of the right size with genes for cells.
    """
    path = Path(path).expanduser()
    with h5py.File(path, "r") as f:
        if "matrix" not in f:
            raise ValueError(f"{path.name}: no `matrix` group; not a Cell Ranger "
                             "feature-barcode h5 (version 3 layout)")
        m = f["matrix"]
        n_features, n_cells = (int(v) for v in m["shape"][:])
        indptr = m["indptr"][:]
        barcodes = _decode(m["barcodes"][:])
        feats = m["features"]
        ids, names, types = (_decode(feats[k][:]) for k in ("id", "name", "feature_type"))
        if len(indptr) != n_cells + 1 or len(barcodes) != n_cells or len(ids) != n_features:
            raise ValueError(
                f"{path.name}: shape says {n_features} features x {n_cells} barcodes but "
                f"indptr has {len(indptr)} entries, barcodes {len(barcodes)}, features "
                f"{len(ids)}. The matrix is not compressed by barcode as expected.")
        data = m["data"][:]
        indices = m["indices"][:]
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError(f"{path.name}: counts are stored as {data.dtype}, not integers")
    if data.size and int(data.min()) < 0:
        raise ValueError(f"{path.name}: negative counts")
    X = sp.csr_matrix((data, indices.astype(np.int32, copy=False), indptr),
                      shape=(n_cells, n_features))
    X.sum_duplicates()
    X.eliminate_zeros()
    return Lane(name or path.name, barcodes, X, ids, names, types)


@dataclass
class GuideLibrary:
    """The guide features of the library and what each one means.

    Built from the first lane and required identical on every later one: a lane whose
    feature table differs was quantified against another reference, and its rows would be
    added into the wrong columns without a word.
    """

    guide_ids: np.ndarray        # (n_guides,) str, in feature order
    guide_label: np.ndarray      # (n_guides,) str -- the target gene, or the control label
    is_control: np.ndarray       # (n_guides,) bool
    targets: list[str]           # sorted target genes, control excluded

    @property
    def gene_labels(self) -> list[str]:
        return sorted([*self.targets, *{str(x) for x in self.guide_label[self.is_control]}])

    def guide_level_labels(self, control_label: str) -> list[str]:
        """Targeting guides by id, the control guides pooled under the one control label."""
        return sorted([*self.guide_ids[~self.is_control].tolist(), control_label])


def build_guide_library(guide_ids: np.ndarray, rules: LaneRules) -> GuideLibrary:
    """Guide name -> label, by the two declared patterns and nothing looser.

    Each pattern must match the WHOLE name. A guide that matches neither is a refusal, not
    a dropped row: an unparsed guide would leave its cells looking unassigned, and they
    would be counted as carrying no guide. The number of control guides is checked against
    the number the methods state, which is what ties the pattern to the experiment.
    """
    target_re = re.compile(rules.guide_target_regex)
    control_re = re.compile(rules.control_guide_regex)
    labels, is_control, unparsed = [], [], []
    for gid in guide_ids.tolist():
        if control_re.fullmatch(gid):
            labels.append(rules.control_label)
            is_control.append(True)
            continue
        hit = target_re.fullmatch(gid)
        if hit is None:
            unparsed.append(gid)
            continue
        labels.append(hit.group("target"))
        is_control.append(False)
    if unparsed:
        raise ValueError(f"{len(unparsed)} guide name(s) match neither control_guide_regex nor "
                         f"guide_target_regex, first few {unparsed[:5]}")
    is_control_arr = np.asarray(is_control, dtype=bool)
    label_arr = np.asarray(labels, dtype=object)
    n_control = int(is_control_arr.sum())
    if n_control != rules.n_control_guides:
        raise ValueError(
            f"{n_control} guide(s) match control_guide_regex but the block declares "
            f"n_control_guides: {rules.n_control_guides}. The control arm is what the methods "
            "say it is; a pattern that finds a different number is matching something else.")
    targets = sorted({str(x) for x in label_arr[~is_control_arr]})
    if rules.control_label in targets:
        raise ValueError(f"a target gene is spelled {rules.control_label!r}, the control label")
    if len(set(guide_ids.tolist())) != len(guide_ids):
        raise ValueError("duplicate guide names in the feature table")
    return GuideLibrary(guide_ids, label_arr, is_control_arr, targets)


@dataclass
class SymbolAxis:
    """Gene feature -> emitted column, and the record of what did not map."""

    genes: np.ndarray                 # the emitted axis
    col_of_feature: np.ndarray        # (n_gene_features,) int32, -1 = not emitted
    unmapped_challenge: list[str]
    collided_symbols: list[str]
    alias_recovered: list[dict] = field(default_factory=list)
    restricted: bool = True

    @property
    def n_mapped(self) -> int:
        return len(self.genes)

    def projector(self) -> sp.csr_matrix:
        """(n_gene_features x emitted genes) of ones: `X @ projector` sums features into columns."""
        feat = np.flatnonzero(self.col_of_feature >= 0)
        return sp.csr_matrix((np.ones(len(feat)), (feat, self.col_of_feature[feat])),
                             shape=(len(self.col_of_feature), len(self.genes)))


def build_symbol_axis(names: np.ndarray, ids: np.ndarray, challenge_genes: list[str] | None,
                      *, aliases: dict[str, str] | None = None) -> SymbolAxis:
    """Map a lane's gene features onto the axis we emit.

    With `challenge_genes` the axis is the challenge symbols this reference carries, in
    challenge order, as the X-Atlas streamer's is and for the same reason (`pooled_delta`
    remaps every source onto the challenge axis by symbol anyway).

    THE ALIAS TABLE IS READ IN REVERSE HERE. `gene_aliases.RETIRED_SYMBOLS` maps an old
    spelling onto the current one, and its docstring warns against applying it onto the 2026
    axis, because that axis speaks the OLD dialect. A Cell Ranger 2024 reference speaks the
    current one. So for a challenge symbol the reference lacks, the feature carrying its
    CURRENT symbol feeds its column: `AARS` on the axis is fed by the feature named `AARS1`.
    It is never applied when the current symbol is itself a challenge gene (one feature
    would feed two columns), and every pair used is recorded with the feature's Ensembl id
    so it can be checked against the alias table's own evidence. The 48-pair table was
    built for another purpose and finds 38 of the 427 challenge symbols a 2024 reference
    lacks by name; `aliases` takes a wider map in the same direction (`--axis-aliases`, a
    table built from a symbol authority before the read).

    Duplicate symbols (38,606 ids over 38,584 names on the 2024 reference) are summed into
    one column and recorded, as in `stream_parquet_pseudobulk.build_gene_axis`.
    """
    aliases = RETIRED_SYMBOLS if aliases is None else aliases
    present: dict[str, list[int]] = {}
    for i, n in enumerate(names.tolist()):
        present.setdefault(n, []).append(i)

    recovered: list[dict] = []
    if challenge_genes is None:
        genes = sorted(present)
        feeds = {g: present[g] for g in genes}
        unmapped: list[str] = []
    else:
        if len(set(challenge_genes)) != len(challenge_genes):
            raise ValueError("duplicate symbols on the challenge axis")
        on_axis = set(challenge_genes)
        feeds = {}
        unmapped = []
        for g in challenge_genes:
            if g in present:
                feeds[g] = present[g]
                continue
            current = aliases.get(g)
            if current is not None and current in present and current not in on_axis:
                feeds[g] = present[current]
                recovered.append({"axis_symbol": g, "source_symbol": current,
                                  "ensembl_ids": [str(ids[i]) for i in present[current]]})
            else:
                unmapped.append(g)
        genes = [g for g in challenge_genes if g in feeds]

    col_of_feature = np.full(len(names), -1, dtype=np.int32)
    for col, g in enumerate(genes):
        col_of_feature[feeds[g]] = col
    collided = sorted(g for g in genes if len(feeds[g]) > 1)
    return SymbolAxis(np.asarray(genes, dtype=object), col_of_feature, unmapped, collided,
                      recovered, restricted=challenge_genes is not None)


def own_gene_projector(names: np.ndarray, targets: list[str], *,
                       aliases: dict[str, str] | None = None,
                       ) -> tuple[sp.csr_matrix, np.ndarray]:
    """(n_gene_features x targets) of ones selecting each target's OWN gene, and which have one.

    The library names its targets in whatever symbols it was designed with; the reference
    names genes in current ones (181 of Pan 2026's 2,978 targets are not a reference
    symbol: `ATP5A1`, `C10orf2`). Exact symbol first, then `aliases`, target -> reference symbol:
    the alias table read FORWARD (old onto current), which is its documented direction,
    under whatever the caller adds from a symbol authority (`--target-aliases`). A target
    with no gene feature keeps an empty column and is reported as not self-measurable.
    Labels are never renamed by this: it only finds a target's own column.
    """
    aliases = RETIRED_SYMBOLS if aliases is None else aliases
    present: dict[str, list[int]] = {}
    for i, n in enumerate(names.tolist()):
        present.setdefault(n, []).append(i)
    rows, cols = [], []
    has = np.zeros(len(targets), dtype=bool)
    for j, t in enumerate(targets):
        feats = present.get(t) or present.get(aliases.get(t, ""), [])
        has[j] = bool(feats)
        rows.extend(feats)
        cols.extend([j] * len(feats))
    proj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(names), len(targets)))
    return proj, has


@dataclass
class CellCalls:
    n_genes: np.ndarray          # genes with a non-zero count, over every gene feature
    total_umi: np.ndarray        # gene UMIs, over every gene feature
    guide_umi: np.ndarray        # guide UMIs, over every guide
    n_guides_detected: np.ndarray
    n_guides_at: dict[int, np.ndarray]  # guides at or above 2, 5 and 10 UMIs: the shape of
    #                              the capture (ambient guides against the one a cell carries)
    n_guides_called: np.ndarray  # guides at or above guide_min_umi
    top_guide: np.ndarray        # index into the guide features, -1 when the cell has none
    top_guide_umi: np.ndarray
    second_guide_umi: np.ndarray
    drop_reason: np.ndarray      # '' for a kept cell

    @property
    def kept(self) -> np.ndarray:
        return self.drop_reason == ""


def call_cells(ge: sp.csr_matrix, guides: sp.csr_matrix, rules: LaneRules) -> CellCalls:
    """The cell filter and the guide call, per barcode.

    A cell is kept when it has at least `min_genes` genes with a non-zero count AND exactly
    one guide with at least `guide_min_umi` UMIs; it takes that guide. Both thresholds are
    inclusive, as the methods word them ("at least"). The gene filter is read first, so a
    cell failing both is counted under the gene filter.
    """
    n = ge.shape[0]
    n_genes = ge.getnnz(axis=1).astype(np.int64)
    total_umi = np.asarray(ge.sum(axis=1)).ravel().astype(np.int64)
    guide_umi = np.asarray(guides.sum(axis=1)).ravel().astype(np.int64)
    n_detected = guides.getnnz(axis=1).astype(np.int64)

    coo = guides.tocoo()
    n_called = np.bincount(coo.row[coo.data >= rules.guide_min_umi], minlength=n).astype(np.int64)
    n_at = {k: np.bincount(coo.row[coo.data >= k], minlength=n).astype(np.int64)
            for k in GUIDE_SHAPE_UMIS}
    top = np.full(n, -1, dtype=np.int64)
    top_umi = np.zeros(n, dtype=np.int64)
    second_umi = np.zeros(n, dtype=np.int64)
    if coo.nnz:
        order = np.lexsort((-coo.data.astype(np.int64), coo.row))
        r, c, d = coo.row[order], coo.col[order], coo.data[order]
        first = np.r_[True, r[1:] != r[:-1]]
        top[r[first]] = c[first]
        top_umi[r[first]] = d[first]
        second = np.r_[False, first[:-1] & (r[1:] == r[:-1])]
        second_umi[r[second]] = d[second]

    reason = np.full(n, "", dtype=object)
    low = n_genes < rules.min_genes
    reason[low] = DROP_LOW_GENES
    reason[~low & (n_called == 0)] = DROP_NO_GUIDE
    reason[~low & (n_called > 1)] = DROP_MULTI_GUIDE
    return CellCalls(n_genes, total_umi, guide_umi, n_detected, n_at, n_called, top, top_umi,
                     second_umi, reason)


@dataclass
class LaneResult:
    """Everything one lane contributes, before it is added to anything."""

    name: str
    cells: pd.DataFrame           # every barcode, kept or not
    sub: sp.csr_matrix            # (kept cells x emitted genes) raw counts
    lib: np.ndarray               # (kept,) emitted-axis library size
    gene_label: np.ndarray        # (kept,) target gene or the control label
    guide_label: np.ndarray       # (kept,) guide name, controls pooled under the control label
    control_targets: sp.csr_matrix  # (control cells x targets) raw counts of each target's own gene
    control_rows: np.ndarray      # row of each control cell within `cells`


@dataclass
class LaneContext:
    """What every lane of one dataset shares: the rules, the feature table, and what is
    built from the two."""

    rules: LaneRules
    feature_ids: np.ndarray
    feature_names: np.ndarray
    feature_types: np.ndarray
    library: GuideLibrary
    axis: SymbolAxis
    axis_proj: sp.csr_matrix      # (gene features x emitted genes)
    own_proj: sp.csr_matrix       # (gene features x targets): each target's own gene
    target_has_gene: np.ndarray
    ge_idx: np.ndarray
    guide_idx: np.ndarray
    mt_idx: np.ndarray

    @classmethod
    def from_lane(cls, lane: Lane, rules: LaneRules, challenge_genes: list[str] | None,
                  *, axis_aliases: dict[str, str] | None = None,
                  target_aliases: dict[str, str] | None = None) -> LaneContext:
        types = lane.feature_types
        ge_idx = np.flatnonzero(types == rules.expression_feature_type)
        guide_idx = np.flatnonzero(types == rules.guide_feature_type)
        if not len(ge_idx) or not len(guide_idx):
            raise ValueError(
                f"{lane.name}: {len(ge_idx)} features of type {rules.expression_feature_type!r} "
                f"and {len(guide_idx)} of type {rules.guide_feature_type!r}; feature types "
                f"present: {sorted(set(types.tolist()))}")
        ge_names = lane.feature_names[ge_idx]
        try:
            library = build_guide_library(lane.feature_ids[guide_idx], rules)
        except ValueError as exc:
            raise ValueError(f"{lane.name}: {exc}") from exc
        axis = build_symbol_axis(ge_names, lane.feature_ids[ge_idx], challenge_genes,
                                 aliases={**RETIRED_SYMBOLS, **(axis_aliases or {})})
        own_proj, has = own_gene_projector(
            ge_names, library.targets, aliases={**RETIRED_SYMBOLS, **(target_aliases or {})})
        mt_idx = np.flatnonzero(np.char.startswith(ge_names.astype(str), "MT-"))
        return cls(rules, lane.feature_ids, lane.feature_names, lane.feature_types, library,
                   axis, axis.projector(), own_proj, has, ge_idx, guide_idx, mt_idx)

    def require_same_features(self, lane: Lane) -> None:
        for mine, theirs, what in ((self.feature_ids, lane.feature_ids, "ids"),
                                   (self.feature_names, lane.feature_names, "names"),
                                   (self.feature_types, lane.feature_types, "types")):
            if len(mine) != len(theirs) or not np.array_equal(mine, theirs):
                raise ValueError(
                    f"{lane.name}: feature {what} differ from the first lane's. Lanes are "
                    "added column for column, so a lane on another reference or another "
                    "guide library cannot be folded in.")


def process_lane(lane: Lane, ctx: LaneContext) -> LaneResult:
    """Filter, call and project one lane. Pure: nothing is accumulated here."""
    ctx.require_same_features(lane)
    rules, library = ctx.rules, ctx.library
    ge = lane.X[:, ctx.ge_idx].tocsr()
    guides = lane.X[:, ctx.guide_idx].tocsr()
    if not ge.shape[0]:
        raise ValueError(f"{lane.name}: the matrix holds no barcode")
    if checks.counts_state(ge[:256]) != checks.RAW_COUNTS:
        raise ValueError(f"{lane.name}: gene counts do not read as raw counts")
    calls = call_cells(ge, guides, rules)

    kept = np.flatnonzero(calls.kept)
    sub = (ge[kept] @ ctx.axis_proj).tocsr()
    lib = np.asarray(sub.sum(axis=1)).ravel()
    dead = lib <= 0
    if dead.any():
        calls.drop_reason[kept[dead]] = DROP_ZERO_LIBSIZE
        kept, sub, lib = kept[~dead], sub[~dead], lib[~dead]

    top = calls.top_guide
    gene_label_all = np.full(len(lane.barcodes), "", dtype=object)
    gene_label_all[kept] = library.guide_label[top[kept]]
    guide_label = np.where(library.is_control[top[kept]], rules.control_label,
                           library.guide_ids[top[kept]]).astype(object)

    mt = (np.asarray(ge[:, ctx.mt_idx].sum(axis=1)).ravel() if len(ctx.mt_idx)
          else np.zeros(ge.shape[0]))
    pct_mt = np.where(calls.total_umi > 0, 100.0 * mt / np.maximum(calls.total_umi, 1), 0.0)

    # Each kept cell's count of its OWN target gene: the on-target readout, per cell.
    own = (ge[kept] @ ctx.own_proj).tocsr()
    target_pos = {t: j for j, t in enumerate(library.targets)}
    own_count = np.full(len(lane.barcodes), -1, dtype=np.int64)
    is_ctrl_kept = library.is_control[top[kept]]
    tcol = np.array([target_pos.get(lab, -1) for lab in gene_label_all[kept]], dtype=np.int64)
    measurable = (~is_ctrl_kept) & (tcol >= 0)
    measurable[measurable] &= ctx.target_has_gene[tcol[measurable]]
    if measurable.any():
        own_count[kept[measurable]] = np.asarray(
            own[np.flatnonzero(measurable), tcol[measurable]]).ravel().astype(np.int64)

    libsize_all = np.zeros(len(lane.barcodes))
    libsize_all[kept] = lib
    top_name = np.full(len(lane.barcodes), "", dtype=object)
    has_top = top >= 0
    top_name[has_top] = library.guide_ids[top[has_top]]
    cells = pd.DataFrame({
        "lane": lane.name, "barcode": lane.barcodes,
        "n_genes": calls.n_genes, "total_umi": calls.total_umi, "pct_mt": pct_mt,
        "guide_umi": calls.guide_umi, "n_guides_detected": calls.n_guides_detected,
        **{f"n_guides_ge{k}": v for k, v in calls.n_guides_at.items()},
        "n_guides_called": calls.n_guides_called, "top_guide": top_name,
        "top_guide_umi": calls.top_guide_umi, "second_guide_umi": calls.second_guide_umi,
        "kept": calls.drop_reason == "", "drop_reason": calls.drop_reason,
        "target_gene": gene_label_all, "own_target_count": own_count, "libsize": libsize_all,
    })
    return LaneResult(
        name=lane.name, cells=cells, sub=sub, lib=lib,
        gene_label=gene_label_all[kept], guide_label=guide_label,
        control_targets=own[np.flatnonzero(is_ctrl_kept)].tocsr(),
        control_rows=kept[is_ctrl_kept],
    )


class Sums:
    """The (labels x genes) sums of one keying. Allocated once; `add` folds one lane in."""

    def __init__(self, labels: list[str], genes: np.ndarray):
        self.labels = list(labels)
        self.genes = genes
        self.code_of = {lab: i for i, lab in enumerate(self.labels)}
        L, G = len(self.labels), len(genes)
        self.count_sum = np.zeros((L, G))
        self.cpm_sum = np.zeros((L, G))
        self.cpm_sq_sum = np.zeros((L, G))
        self.n_cells = np.zeros(L, dtype=np.int64)
        self.libsize_sum = np.zeros(L)
        self.sources: list[str] = []

    def add(self, labels: np.ndarray, sub: sp.csr_matrix, lib: np.ndarray, source: str) -> None:
        self.sources.append(source)
        if not len(labels):
            return
        missing = sorted({str(x) for x in labels} - set(self.code_of))
        if missing:
            raise KeyError(f"{source}: labels with no row in the accumulator: {missing[:5]}")
        codes = np.array([self.code_of[lab] for lab in labels], dtype=np.int64)
        sub = sp.csr_matrix(sub, dtype=np.float64)
        cpm = sp.diags(1e6 / lib) @ sub
        uniq, local = np.unique(codes, return_inverse=True)
        ind = sp.csr_matrix((np.ones(len(local)), (local, np.arange(len(local)))),
                            shape=(len(uniq), len(local)))
        _scatter_add(self.count_sum, uniq, ind @ sub)
        _scatter_add(self.cpm_sum, uniq, ind @ cpm)
        _scatter_add(self.cpm_sq_sum, uniq, ind @ cpm.multiply(cpm))
        np.add.at(self.n_cells, codes, 1)
        np.add.at(self.libsize_sum, codes, lib)

    def to_pseudobulk(self) -> PseudobulkSums:
        """Never-observed labels are pruned, for the reason `_Accumulator.to_pseudobulk`
        in the parquet streamer gives: a label present with zero cells votes."""
        seen = np.flatnonzero(self.n_cells > 0)
        return PseudobulkSums(
            labels=[self.labels[i] for i in seen], genes=self.genes,
            count_sum=self.count_sum[seen], cpm_sum=self.cpm_sum[seen],
            cpm_sq_sum=self.cpm_sq_sum[seen], n_cells=self.n_cells[seen],
            libsize_sum=self.libsize_sum[seen], sources=list(self.sources))


def collapse_to_genes(guide_pb: PseudobulkSums, library: GuideLibrary,
                      control_label: str) -> PseudobulkSums:
    """Guide-level sums -> gene-level sums. Every member is additive over disjoint cells,
    and a kept cell carries exactly one guide, so this is a row sum and nothing else."""
    label_of = dict(zip(library.guide_ids.tolist(), library.guide_label.tolist(), strict=True))
    label_of[control_label] = control_label
    gene_of_row = [label_of[lab] for lab in guide_pb.labels]
    labels = sorted(set(gene_of_row))
    pos = {lab: i for i, lab in enumerate(labels)}
    rows = np.array([pos[g] for g in gene_of_row], dtype=np.int64)
    L, G = len(labels), len(guide_pb.genes)
    out = PseudobulkSums(labels, guide_pb.genes, np.zeros((L, G)), np.zeros((L, G)),
                         np.zeros((L, G)), np.zeros(L, dtype=np.int64), np.zeros(L),
                         list(guide_pb.sources))
    np.add.at(out.count_sum, rows, guide_pb.count_sum)
    np.add.at(out.cpm_sum, rows, guide_pb.cpm_sum)
    np.add.at(out.cpm_sq_sum, rows, guide_pb.cpm_sq_sum)
    np.add.at(out.n_cells, rows, guide_pb.n_cells)
    np.add.at(out.libsize_sum, rows, guide_pb.libsize_sum)
    return out


def read_groups(path: str | Path, lane_names: list[str]) -> dict[str, str]:
    """A two-column TSV, lane <tab> group, `#` lines ignored. Every lane must be named once.

    The file is written BEFORE the read and kept beside the artifacts: which lanes sit in
    which half is a choice, and it is made without sight of the data.
    """
    out: dict[str, str] = {}
    for n, line in enumerate(Path(path).expanduser().read_text().splitlines(), start=1):
        if not line.strip() or line.strip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(f"{path}: line {n} is not lane<TAB>group")
        lane, group = parts[:2]
        if lane in out:
            raise ValueError(f"{path}: lane {lane!r} listed twice")
        if not re.fullmatch(r"[A-Za-z0-9_]+", group):
            raise ValueError(f"{path}: group {group!r} must be letters, digits or underscores "
                             "(it becomes part of a file name)")
        if group in RESERVED_GROUPS:
            raise ValueError(f"{path}: group {group!r} would be written over <out>.{group}.npz, "
                             "a file name this reader uses for something else")
        out[lane] = group
    missing = [n for n in lane_names if n not in out]
    extra = [n for n in out if n not in set(lane_names)]
    if missing or extra:
        raise ValueError(f"{path}: {len(missing)} lane(s) have no group (first {missing[:3]}) "
                         f"and {len(extra)} listed lane(s) are not in the run (first {extra[:3]})")
    return out


def read_aliases(path: str | Path, key: str) -> dict[str, str]:
    """A TSV with a header and the columns `key` and `reference_symbol`: our spelling -> the
    symbol the reference uses. Rows with an empty reference symbol (a symbol the authority
    could not resolve) are skipped. `key` is `target` for the library's targets and
    `axis_symbol` for the emitted axis."""
    frame = pd.read_csv(Path(path).expanduser(), sep="\t", dtype=str).fillna("")
    for col in (key, "reference_symbol"):
        if col not in frame.columns:
            raise ValueError(f"{path}: no column {col!r}; have {list(frame.columns)}")
    if frame[key].duplicated().any():
        raise ValueError(f"{path}: a {key} is listed twice")
    return {t: r for t, r in zip(frame[key], frame["reference_symbol"], strict=True) if r}


def read_target_aliases(path: str | Path) -> dict[str, str]:
    return read_aliases(path, "target")


def read_feature_table(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A lane's feature ids, names and types, without its matrix."""
    with h5py.File(path, "r") as f:
        feats = f["matrix"]["features"]
        return tuple(_decode(feats[k][:]) for k in ("id", "name", "feature_type"))


def require_one_feature_table(lanes: list[tuple[str, Path]]) -> None:
    """Every lane's feature table equals the first lane's, checked BEFORE the pass: a lane
    on another reference then fails in seconds, not after the lanes before it were read."""
    first = None
    for name, path in lanes:
        table = read_feature_table(path)
        if first is None:
            first = table
            continue
        for mine, theirs, what in zip(first, table, ("ids", "names", "types"), strict=True):
            if len(mine) != len(theirs) or not np.array_equal(mine, theirs):
                raise ValueError(f"{name}: feature {what} differ from the first lane's")


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


@dataclass
class StreamResult:
    ctx: LaneContext
    gene_pb: PseudobulkSums
    guide_pb: PseudobulkSums | None
    group_pbs: dict[str, Path]
    cells: pd.DataFrame
    qc: StreamQC
    control_targets: sp.csr_matrix
    control_cell_rows: np.ndarray
    lanes: pd.DataFrame


def stream_lanes(lanes: list[tuple[str, Path]], rules: LaneRules, *,
                 challenge_genes: list[str] | None, groups: dict[str, str] | None = None,
                 guide_level: bool = False, group_out: Path | None = None,
                 axis_aliases: dict[str, str] | None = None,
                 target_aliases: dict[str, str] | None = None,
                 progress: bool = False) -> StreamResult:
    """One pass over the lanes, in group order, one gene-level accumulator resident at a time.

    With `groups`, each group's sums are saved to `<group_out>.<group>.npz` as its last lane
    finishes and the accumulator is freed; the all-lane sums are their merge. Without, there
    is one accumulator and nothing is written here.
    """
    if groups is not None and group_out is None:
        raise ValueError("groups need group_out: a group's sums are written as it finishes")
    order = sorted(lanes, key=lambda item: ((groups or {}).get(item[0], ""), item[0]))
    group_of = (lambda name: groups[name]) if groups else (lambda name: "")

    ctx: LaneContext | None = None
    gene_acc: Sums | None = None
    guide_acc: Sums | None = None
    gene_pb: PseudobulkSums | None = None
    group_pbs: dict[str, Path] = {}
    cells_parts: list[pd.DataFrame] = []
    ct_parts: list[sp.csr_matrix] = []
    ct_rows: list[np.ndarray] = []
    lane_rows: list[dict] = []
    gene_cells: np.ndarray | None = None
    row0 = 0
    t0 = time.time()

    def flush(group: str) -> None:
        nonlocal gene_acc, gene_pb
        assert gene_acc is not None
        pb = gene_acc.to_pseudobulk()
        gene_acc = None
        if groups is None:
            gene_pb = pb
            return
        path = group_out.with_name(f"{group_out.name}.{group}.npz")
        pb.save(path)
        group_pbs[group] = path

    current: str | None = None
    for i, (name, path) in enumerate(order):
        digest = sha256_file(path)
        lane = read_lane(path, name)
        if ctx is None:
            ctx = LaneContext.from_lane(lane, rules, challenge_genes,
                                        axis_aliases=axis_aliases,
                                        target_aliases=target_aliases)
            if guide_level:
                guide_acc = Sums(ctx.library.guide_level_labels(rules.control_label),
                                 ctx.axis.genes)
            gene_cells = np.zeros(ctx.axis.n_mapped, dtype=np.int64)
        group = group_of(name)
        if current is not None and group != current:
            flush(current)
        if gene_acc is None:
            gene_acc = Sums(ctx.library.gene_labels, ctx.axis.genes)
        current = group

        res = process_lane(lane, ctx)
        del lane
        gene_acc.add(res.gene_label, res.sub, res.lib, name)
        if guide_acc is not None:
            guide_acc.add(res.guide_label, res.sub, res.lib, name)
        gene_cells += res.sub.getnnz(axis=0).astype(np.int64)
        cells_parts.append(res.cells)
        ct_parts.append(res.control_targets.astype(np.int32))
        ct_rows.append(res.control_rows + row0)
        row0 += len(res.cells)

        reasons = res.cells["drop_reason"].value_counts()
        kept_cells = res.cells[res.cells["kept"]]
        lane_rows.append({
            "lane": name, "group": group, "bytes": path.stat().st_size, "sha256": digest,
            "barcodes": len(res.cells), "kept": int(res.cells["kept"].sum()),
            DROP_LOW_GENES: int(reasons.get(DROP_LOW_GENES, 0)),
            DROP_NO_GUIDE: int(reasons.get(DROP_NO_GUIDE, 0)),
            DROP_MULTI_GUIDE: int(reasons.get(DROP_MULTI_GUIDE, 0)),
            DROP_ZERO_LIBSIZE: int(reasons.get(DROP_ZERO_LIBSIZE, 0)),
            "control_cells": int((res.gene_label == rules.control_label).sum()),
            "targets_seen": len(set(res.gene_label.tolist()) - {rules.control_label}),
            "median_umi_kept": float(kept_cells["total_umi"].median()) if len(kept_cells) else 0.0,
            "median_genes_kept": float(kept_cells["n_genes"].median()) if len(kept_cells) else 0.0,
        })
        if progress:
            row = lane_rows[-1]
            print(f"  [{i + 1:>3}/{len(order)}] {name.rsplit('/', 1)[-1][:44]:44s} "
                  f"{row['barcodes']:>6} barcodes  {row['kept']:>6} kept  "
                  f"{row['control_cells']:>5} control  {time.time() - t0:6.0f}s", flush=True)
    if ctx is None or current is None:
        raise ValueError("no lanes to read")
    flush(current)

    if groups is not None:
        for path in group_pbs.values():
            part = PseudobulkSums.load(path)
            gene_pb = part if gene_pb is None else merge(gene_pb, part)
    assert gene_pb is not None
    if rules.control_label not in gene_pb.labels:
        raise ValueError(
            f"control label {rules.control_label!r} accumulated zero cells across "
            f"{len(order)} lane(s). Every delta is measured against this arm, so an aggregate "
            "without it is unusable.")

    cells = pd.concat(cells_parts, ignore_index=True)
    for col in ("lane", "top_guide", "drop_reason", "target_gene"):
        cells[col] = cells[col].astype("category")
    qc = _build_qc(cells, ctx, gene_cells)
    return StreamResult(
        ctx=ctx, gene_pb=gene_pb,
        guide_pb=guide_acc.to_pseudobulk() if guide_acc is not None else None,
        group_pbs=group_pbs, cells=cells, qc=qc,
        control_targets=sp.vstack(ct_parts, format="csr"),
        control_cell_rows=np.concatenate(ct_rows), lanes=pd.DataFrame(lane_rows))


def _build_qc(cells: pd.DataFrame, ctx: LaneContext, gene_cells: np.ndarray) -> StreamQC:
    """The sidecar X-Atlas writes, from the per-cell table: lanes are its batches, and it
    keeps every label of the library, the never-observed ones included (the coverage
    number is only answerable if the zeros are still there)."""
    rules, library = ctx.rules, ctx.library
    labels = library.gene_labels
    kept = cells[cells["kept"]]
    code = pd.Categorical(kept["target_gene"].astype(str), categories=labels).codes
    lanes = sorted(cells["lane"].astype(str).unique())
    lane_code = pd.Categorical(kept["lane"].astype(str), categories=lanes).codes
    batch_cells = np.zeros((len(labels), len(lanes)), dtype=np.int32)
    np.add.at(batch_cells, (code, lane_code), 1)
    L = len(labels)
    pct = kept["pct_mt"].to_numpy(dtype=np.float64)
    guide_name = np.where(kept["target_gene"].astype(str) == rules.control_label,
                          rules.control_label, kept["top_guide"].astype(str))
    guide_labels, guide_counts = np.unique(guide_name, return_counts=True)
    return StreamQC(
        labels=labels, batches=lanes, batch_cells=batch_cells,
        pct_mt_sum=np.bincount(code, weights=pct, minlength=L),
        pct_mt_sq_sum=np.bincount(code, weights=pct * pct, minlength=L),
        total_counts_sum=np.bincount(code, weights=kept["total_umi"].to_numpy(np.float64),
                                     minlength=L),
        n_cells=np.bincount(code, minlength=L).astype(np.int64),
        cells_seen=len(kept),
        cells_dropped_filter=int(cells["drop_reason"].isin(
            [DROP_LOW_GENES, DROP_NO_GUIDE, DROP_MULTI_GUIDE]).sum()),
        cells_dropped_zero=int((cells["drop_reason"] == DROP_ZERO_LIBSIZE).sum()),
        labels_in_corpus=len(labels), gene_cells=gene_cells,
        guide_labels=[str(g) for g in guide_labels], guide_cells=guide_counts.astype(np.int64))


# ------------------------------------------------------------------ lineage --


def write_lineage(out_dir: Path, *, provenance: Path, dataset: str, context: str, entry: str,
                  ctx: LaneContext, pb: PseudobulkSums, result: StreamResult, scope: str,
                  resolution: str, keyed_by: str, artifacts: dict[str, str],
                  extra: dict | None = None) -> Path:
    """One entry in the directory's LINEAGE.json, in the file shape the parquet streamer
    writes (schema 2, `entries` keyed `<dataset>/<entry>`), so a directory reads the same
    whichever reader filled it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "LINEAGE.json"
    payload = {"schema_version": 2, "entries": {}}
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("schema_version") == 2 and isinstance(existing.get("entries"), dict):
            payload = existing
        else:
            path.replace(path.with_suffix(".json.bak"))
    rules, axis = ctx.rules, ctx.axis
    in_entry = set(pb.sources)
    lanes = result.lanes[result.lanes["lane"].isin(in_entry)]
    record = {
        "dataset": dataset,
        "context": context,
        "derives_from": str(provenance),
        "code_sha": code_sha(),
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reader": "sidechain.data.stream_10x_h5",
        "accumulator": {
            "resolution": resolution,
            "keyed_by": keyed_by,
            "scope": scope,
            "labels": len(pb.labels),
            "genes": int(axis.n_mapped),
            "gene_axis": "challenge-symbols" if axis.restricted else "all-symbols",
            "control_label": rules.control_label,
            "cell_filter": f"genes with a non-zero count >= {rules.min_genes}",
            "guide_rule": f"exactly one guide with >= {rules.guide_min_umi} UMIs",
            "control_guides": f"{rules.n_control_guides} guides matching "
                              f"{rules.control_guide_regex!r}, pooled",
            "libsize": "sum over the EMITTED gene axis (mirrors stream_pseudobulk)",
            # The rules as the block declared them when this was built. PROVENANCE.json's
            # copy of the spec is frozen at the first gate; this one is the build's.
            "rules": dataclasses.asdict(rules),
        },
        "coverage": {
            "cells_accumulated": int(pb.n_cells.sum()),
            "barcodes_read": int(lanes["barcodes"].sum()),
            "cells_dropped_low_genes": int(lanes[DROP_LOW_GENES].sum()),
            "cells_dropped_no_guide": int(lanes[DROP_NO_GUIDE].sum()),
            "cells_dropped_multi_guide": int(lanes[DROP_MULTI_GUIDE].sum()),
            "cells_dropped_zero_libsize": int(lanes[DROP_ZERO_LIBSIZE].sum()),
            "lanes": len(pb.sources),
            "labels_in_library": len(ctx.library.gene_labels),
            "targets_with_no_gene_feature": int((~ctx.target_has_gene).sum()),
            "challenge_genes_mapped": int(axis.n_mapped),
            "challenge_genes_unmapped": len(axis.unmapped_challenge),
            "unmapped_genes": axis.unmapped_challenge,
            "collided_symbols": axis.collided_symbols,
            "alias_recovered": axis.alias_recovered,
        },
        "artifacts": artifacts,
        "source_files": len(pb.sources),
        # A web server publishes no checksum, so the digest of what landed is recorded
        # here, by lane (configs/datasets.yaml, `allow_missing_checksum`).
        "source_sha256": {row.lane: row.sha256 for row in lanes.itertuples()},
    }
    if extra:
        record.update(extra)
    payload["entries"][f"{dataset}/{entry}"] = record
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------- CLI --


def _input_record(path: str | Path) -> dict:
    """A table this run read beside the lanes, with its digest: what makes "written before
    the read" checkable afterwards."""
    path = Path(path).expanduser()
    return {"path": str(path), "sha256": sha256_file(path)}


def save_control_targets(path: Path, result: StreamResult) -> None:
    m = result.control_targets
    np.savez_compressed(
        path, data=m.data.astype(np.int32), indices=m.indices.astype(np.int32),
        indptr=m.indptr.astype(np.int64), shape=np.asarray(m.shape, dtype=np.int64),
        targets=np.asarray(result.ctx.library.targets, dtype=object),
        target_has_gene=result.ctx.target_has_gene,
        cell_row=result.control_cell_rows.astype(np.int64))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="name from configs/datasets.yaml")
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--challenge-config", default="challenges/vcc2026/config.yaml")
    ap.add_argument("--root", type=Path, default=Path.home() / "data" / "sidechain")
    ap.add_argument("--all-genes", action="store_true",
                    help="emit every symbol of the reference rather than only the challenge axis")
    ap.add_argument("--guide-level", action="store_true",
                    help="also accumulate per guide (<out>.guide.npz), controls pooled")
    ap.add_argument("--groups", help="TSV, lane<TAB>group: one <out>.<group>.npz per group, "
                                     "written before the read (the split is fixed blind)")
    ap.add_argument("--axis-aliases",
                    help="TSV (axis_symbol, reference_symbol): a challenge symbol the "
                         "reference spells differently, and the symbol it has there. Widens "
                         "the alias table read in reverse; needs the challenge axis")
    ap.add_argument("--target-aliases",
                    help="TSV (target, reference_symbol): where the library spells a target "
                         "differently from the reference, the symbol its own gene has there. "
                         "Only finds the on-target column; labels are never renamed")
    ap.add_argument("--limit-files", type=int, help="read only the first N lanes (proving run)")
    ap.add_argument("--out", required=True, help="output prefix; the artifacts land beside it")
    args = ap.parse_args(argv)

    from sidechain.data.loaders import load_challenge_config
    from sidechain.ingest.provenance import read_provenance

    block = _dataset_block(args.dataset, args.config)
    dest = args.root / block["dest"]
    provenance = read_provenance(dest)
    if provenance is None:
        raise SystemExit(
            f"no PROVENANCE.json at {dest}. The gate runs BEFORE any bytes move:\n"
            f"  uv run python -m sidechain.ingest.fetch --dataset {args.dataset}")
    entries = [f for f in block["files"] if f.get("kind", "expression") == "expression"]
    specs = {json.dumps(f.get("spec") or {}, sort_keys=True) for f in entries}
    if len(specs) != 1:
        raise SystemExit(f"{args.dataset}: its expression files declare {len(specs)} different "
                         "specs; lanes are read under one set of rules")
    spec = entries[0]["spec"]
    rules = LaneRules.from_spec(spec)

    recorded = {f["name"]: f for f in provenance["selected"]}
    names = [f["name"] for f in entries]
    absent = [n for n in names if n not in recorded]
    if absent:
        raise SystemExit(f"{len(absent)} file(s) of the block are not in the recorded "
                         f"provenance, first {absent[:3]}; re-run the gate")
    if args.limit_files is not None:
        if args.limit_files < 1:
            raise SystemExit("--limit-files must be at least 1")
        names = names[: args.limit_files]
    lanes = []
    for name in names:
        path = dest / name
        if not path.exists() or path.stat().st_size != recorded[name]["size_bytes"]:
            raise SystemExit(f"{path} is missing or not the recorded size; run\n"
                             f"  uv run python -m sidechain.ingest.fetch --dataset "
                             f"{args.dataset} --check")
        lanes.append((name, path))

    try:
        require_one_feature_table(lanes)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.all_genes and args.axis_aliases:
        raise SystemExit("--axis-aliases maps challenge symbols; it has no meaning with --all-genes")
    challenge_genes = None
    if not args.all_genes:
        cfg = load_challenge_config(args.challenge_config)
        challenge_genes = read_gene_names(
            Path(cfg["data_dir"]).expanduser() / cfg["gene_names_file"], expect=cfg.get("n_genes"))

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    groups = None
    if args.groups:
        all_groups = read_groups(args.groups, [f["name"] for f in entries])
        groups = {n: all_groups[n] for n in names}

    print(f"{args.dataset}  {len(lanes)} lanes  -> {out}")
    target_aliases = read_target_aliases(args.target_aliases) if args.target_aliases else None
    axis_aliases = read_aliases(args.axis_aliases, "axis_symbol") if args.axis_aliases else None
    result = stream_lanes(lanes, rules, challenge_genes=challenge_genes, groups=groups,
                          guide_level=args.guide_level, group_out=out,
                          axis_aliases=axis_aliases, target_aliases=target_aliases,
                          progress=True)
    ctx = result.ctx
    scope = f"all labels ({len(ctx.library.gene_labels)})"

    npz = out.with_name(out.name + ".npz")
    result.gene_pb.save(npz)
    qc_npz = out.with_name(out.name + ".qc.npz")
    result.qc.save(qc_npz)
    cells_path = out.with_name(out.name + ".cells.parquet")
    result.cells.to_parquet(cells_path, index=False)
    ct_path = out.with_name(out.name + ".control_targets.npz")
    save_control_targets(ct_path, result)
    lanes_path = out.with_name(out.name + ".lanes.tsv")
    result.lanes.to_csv(lanes_path, sep="\t", index=False)

    # The one positive control (see `checks.require_on_target_knockdown`), recorded in the
    # directory whatever it reads: a failed one must not look like a clean run.
    try:
        on_target = checks.require_on_target_knockdown(result.gene_pb, rules.control_label)
    except ValueError as exc:
        on_target = {"status": "FAILED", "detail": str(exc)}
        print(f"!! {exc}", flush=True)

    shared = {"qc_sidecar": qc_npz.name, "cells": cells_path.name,
              "control_targets": ct_path.name, "lanes": lanes_path.name}
    inputs = {key: _input_record(value) for key, value in (
        ("groups_file", args.groups), ("axis_aliases_file", args.axis_aliases),
        ("target_aliases_file", args.target_aliases)) if value}
    context = str(spec.get("context"))
    prov_path = dest / "PROVENANCE.json"
    lineage = write_lineage(
        out.parent, provenance=prov_path, dataset=args.dataset, context=context,
        entry=out.name, ctx=ctx, pb=result.gene_pb, result=result, scope=scope,
        resolution="per-perturbation", keyed_by="target_gene",
        artifacts={"pseudobulk": npz.name, **shared},
        extra={"on_target": on_target, **inputs,
               **({"groups": {g: p.name for g, p in result.group_pbs.items()}}
                  if groups else {})})
    for group, path in result.group_pbs.items():
        pb = PseudobulkSums.load(path)
        write_lineage(
            out.parent, provenance=prov_path, dataset=args.dataset, context=context,
            entry=f"{out.name}.{group}", ctx=ctx, pb=pb, result=result,
            scope=f"{scope}, lane group {group} ({len(pb.sources)} lanes)",
            resolution="per-perturbation", keyed_by="target_gene",
            artifacts={"pseudobulk": path.name}, extra=inputs)
    if result.guide_pb is not None:
        guide_npz = out.with_name(out.name + ".guide.npz")
        result.guide_pb.save(guide_npz)
        write_lineage(
            out.parent, provenance=prov_path, dataset=args.dataset, context=context,
            entry=f"{out.name}.guide", ctx=ctx, pb=result.guide_pb, result=result,
            scope=f"every targeting guide + control ({len(result.guide_pb.labels)} labels)",
            resolution="per-guide (control guides pooled)", keyed_by="top_guide",
            artifacts={"pseudobulk": guide_npz.name, **shared}, extra=inputs)

    meta = {"labels": len(result.gene_pb.labels), "genes": int(ctx.axis.n_mapped),
            "cells": int(result.gene_pb.n_cells.sum()), "lanes": len(lanes),
            "on_target": on_target}
    lanes_df = result.lanes
    print(f"\n  barcodes read        : {int(lanes_df['barcodes'].sum()):,}")
    print(f"  cells kept           : {int(lanes_df['kept'].sum()):,}")
    for reason in (DROP_LOW_GENES, DROP_NO_GUIDE, DROP_MULTI_GUIDE, DROP_ZERO_LIBSIZE):
        print(f"  dropped {reason:13s}: {int(lanes_df[reason].sum()):,}")
    print(f"  labels with >=1 cell : {len(result.gene_pb.labels)}/{len(ctx.library.gene_labels)}")
    print(f"  gene axis            : {ctx.axis.n_mapped} mapped, "
          f"{len(ctx.axis.unmapped_challenge)} challenge genes not mapped, "
          f"{len(ctx.axis.alias_recovered)} found through an alias")
    print(f"  -> {npz}\n  -> {lineage}")
    print(json.dumps(meta))
    return 1 if on_target.get("status") == "FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())
