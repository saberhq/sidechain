"""Ingest-time checks that make two specific mistakes impossible to repeat.

Both were made for real on 2026-08-16 while QC-ing the first external corpus,
and both produced a confident wrong answer rather than an error. They live here
as functions that raise, rather than as a note in a doc, because a rule the
environment enforces is paid for once and a rule in prose is paid for every
session.

1. CONTROL LABELS MATCH EXACTLY, NEVER AS SUBSTRINGS.
   A control check written as `"NT" in value` reported the perturbations
   INTS1, DOT1L_INTS1 and three others as controls, because "INTS1" contains
   "NT". Deltas computed against that pool would silently be perturbation-minus-
   perturbation, biased toward looking like no effect. `control_mask` does the
   exact comparison and refuses to return an all-False mask.

2. NORMALIZATION STATE IS DETECTED, NEVER ASSUMED.
   `expm1` was applied to a matrix that was already raw counts, on the strength
   of a doc sentence saying cell-eval wants log1p. That sentence is about the
   *submission*, not about stored data: the 2025 training matrix ships as raw
   integer UMI counts. The result overflowed float32 to `inf` and produced two
   `nan` summary statistics. `counts_state` answers the question from the data,
   and `to_cp10k` refuses to transform something it cannot confirm is counts.

3. THE CONTROL ARM IS WHATEVER THE AUTHORS DEFINED, NOT WHAT THE LABEL READS.
   Rule 1 says match the control label exactly. It does not say WHICH labels
   are controls, and on 2026-08-23 that gap produced a wrong conclusion in a
   report. Feng 2026's genome-wide arm has 48 cells whose guide call reads
   `NonTarget_*`; its control arm is 499,998 cells, because the paper defines
   controls as "either no guide or a non-targeting gRNA" -- so `unassigned` is
   a control too, not a discarded cell. Reading the label instead of the design
   undercounted the arm by four orders of magnitude and led to "these counts
   cannot recompute the contrast", which is the opposite of the truth.
   `control_mask` therefore accepts a LIST, and every entry must match
   something. Whether a corpus's controls are one label or five is a fact about
   the experiment that has to be read out of the methods and declared -- it can
   never be inferred from the values in the column.

None of these checks is clever. All are cheap, and all three failed in a way
that looked like a finding rather than a bug -- which is the reason they are
here.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

# A matrix of counts is integral. Sampling beats scanning a 2M-cell file, and
# 2000 rows is enough to catch a transform: log1p/cp10k values are non-integral
# almost everywhere, so a transformed matrix fails on the first row.
_STATE_SAMPLE_ROWS = 2000

RAW_COUNTS = "raw_counts"
TRANSFORMED = "transformed"


def _as_dense_sample(X, rows: int = _STATE_SAMPLE_ROWS) -> np.ndarray:
    chunk = X[: min(rows, X.shape[0])]
    return np.asarray(chunk.todense()) if sp.issparse(chunk) else np.asarray(chunk)


def counts_state(X) -> str:
    """Return RAW_COUNTS or TRANSFORMED by looking at the data.

    Call this instead of believing a README, a docstring, or a previous
    session's summary. The answer is cheap and the assumption is not.

    Three signals, in order of strength:

    * negative values      -> scaled or centred, certainly not counts;
    * constant row totals  -> already scaled to a fixed library size. This one
      matters because integrality alone is NOT sufficient: cp10k of small
      integers can land back on integers (cp10k of [[1,9],[4,6]] is
      [[1000,9000],[4000,6000]]), and the first version of this function
      cheerfully called that raw counts. Real UMI totals vary per cell -- H1's
      span roughly 24k-75k -- so identical totals mean someone normalized;
    * non-integral values  -> transformed.

    Downsampled-to-equal-depth counts would be misreported as TRANSFORMED. That
    is the safe direction to be wrong in: it refuses to transform rather than
    transforming twice.
    """
    sample = _as_dense_sample(X)
    if sample.size == 0:
        raise ValueError("empty matrix: cannot determine normalization state")
    if np.any(sample < 0):
        return TRANSFORMED
    totals = sample.sum(axis=1)
    if sample.shape[0] > 1 and totals[0] > 0 and np.allclose(totals, totals[0]):
        return TRANSFORMED
    return RAW_COUNTS if np.allclose(sample, np.round(sample)) else TRANSFORMED


def require_raw_counts(X, *, where: str) -> None:
    """Raise unless `X` is raw integer counts.

    `where` names the caller so the error says which step refused, not just
    that something did.
    """
    state = counts_state(X)
    if state != RAW_COUNTS:
        raise ValueError(
            f"{where}: expected raw integer counts, found {state!r}. Transforming "
            "already-transformed data is the expm1-on-counts bug: it does not error, "
            "it produces plausible nonsense. If the source really is pre-transformed, "
            "set HarmonizedDataset.counts_are_raw=False and say so in notes."
        )


def to_cp10k(X) -> np.ndarray:
    """Counts -> counts-per-10k. Refuses anything that is not counts.

    The refusal is the point: this is the exact call site where assuming the
    input's state produced `inf` and two `nan`s.
    """
    require_raw_counts(X, where="to_cp10k")
    dense = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    totals = dense.sum(axis=1, keepdims=True)
    if np.any(totals == 0):
        raise ValueError(
            "cells with zero total counts cannot be scaled to cp10k; filter them first"
        )
    return dense / totals * 1e4


def control_mask(labels, control_label: str | Sequence[str]) -> np.ndarray:
    """Boolean mask of control cells, by EXACT label match.

    Deliberately not a substring, prefix, case-insensitive or fuzzy test. The
    contract carries `control_label` as a declared field precisely so this can
    be an equality check -- inferring it is what went wrong.

    **`control_label` may be a LIST, because a control arm is not always one
    label.** Feng 2026 is the case that forced this: it defines its controls as
    cells "assigned to a donor but either no guide or a non-targeting gRNA", so
    its control arm is `NonTarget` AND `unassigned` together -- 499,998 cells.
    Taking only the label that reads like a control gives 48, and a delta built
    on 48 cells against 219,206 perturbed would be noise presented as signal.
    Until this accepted a list, that corpus could not be declared correctly at
    all, so the config would have had to lie or drop 99.99 % of its controls.

    Every declared label must match at least one row. A set where one label
    matches nothing is the silently-partial failure this module exists to
    prevent -- the same strictness as `HostRecord.select` on a glob that hits
    no files.
    """
    arr = np.asarray(labels).astype(str)
    wanted = [control_label] if isinstance(control_label, str) else list(control_label)
    if not wanted:
        raise ValueError("control_label is empty; declare at least one control label")

    present = sorted(set(arr))
    missing = [w for w in wanted if w not in set(arr)]
    if missing:
        near = [v for p in missing for v in present if p.lower() in v.lower()][:5]
        hint = (
            f" Labels containing one as a substring: {near} -- if one of those is the "
            "real control, declare it exactly rather than matching loosely."
            if near else f" First few labels present: {present[:5]}"
        )
        raise ValueError(
            f"no cells match control_label(s) {missing!r} exactly.{hint}"
        )
    return np.isin(arr, wanted)
