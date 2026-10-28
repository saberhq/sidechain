"""Ingest-time checks that raise: exact control labels, normalization state read from the data, on-target knockdown.

    mask = control_mask(adata.obs["perturbation"], ["NonTarget", "unassigned"])   # one label or a list
    X = to_cp10k(adata.X)                                                         # refuses non-counts
    reading = require_on_target_knockdown(pb, "non-targeting")                    # a pseudobulk's sums

Reads the arrays it is given; writes nothing. Notes: private/code-notes/src/sidechain/ingest/checks.py.md
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

# rows `counts_state` reads
_STATE_SAMPLE_ROWS = 2000

RAW_COUNTS = "raw_counts"
TRANSFORMED = "transformed"


def _as_dense_sample(X, rows: int = _STATE_SAMPLE_ROWS) -> np.ndarray:
    chunk = X[: min(rows, X.shape[0])]
    return np.asarray(chunk.todense()) if sp.issparse(chunk) else np.asarray(chunk)


def counts_state(X) -> str:
    """RAW_COUNTS or TRANSFORMED from the first rows: negative, equal-total or non-integral rows mean transformed."""
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
    """Raise unless `X` is raw integer counts; `where` names the calling step in the error."""
    state = counts_state(X)
    if state != RAW_COUNTS:
        raise ValueError(
            f"{where}: expected raw integer counts, found {state!r}. Transforming "
            "already-transformed data is the expm1-on-counts bug: it does not error, "
            "it produces plausible nonsense. If the source really is pre-transformed, "
            "set HarmonizedDataset.counts_are_raw=False and say so in notes."
        )


def to_cp10k(X) -> np.ndarray:
    """Counts -> counts-per-10k; refuses anything that is not raw counts."""
    require_raw_counts(X, where="to_cp10k")
    dense = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    totals = dense.sum(axis=1, keepdims=True)
    if np.any(totals == 0):
        raise ValueError(
            "cells with zero total counts cannot be scaled to cp10k; filter them first"
        )
    return dense / totals * 1e4


def control_mask(labels, control_label: str | Sequence[str]) -> np.ndarray:
    """Boolean mask of control cells by exact label match; `control_label` is one label or a list, each must match."""
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


NOT_APPLICABLE = "not_applicable"


def require_on_target_knockdown(pb, control_label: str | Sequence[str], *,
                                max_median_log2fc: float = -0.20,
                                min_arms: int = 50,
                                pseudocount: float = 1.0) -> dict:
    """Raise unless the median self log2FC, over labels whose own gene is on the axis, is at most `max_median_log2fc`.

    Returns `{"status", "n_self_measurable", "median_self_log2fc", "frac_positive", "bar"}`; fewer than `min_arms`
    such labels returns status `not_applicable`.
    """
    labels = np.asarray(pb.labels).astype(str)
    ctrl = control_mask(labels, control_label)
    pos = {g: i for i, g in enumerate(np.asarray(pb.genes).astype(str))}
    n = np.maximum(np.asarray(pb.n_cells), 1)
    ctrl_mean = pb.cpm_sum[ctrl].sum(axis=0) / max(int(np.asarray(pb.n_cells)[ctrl].sum()), 1)

    lfcs = []
    for i, lab in enumerate(labels):
        if ctrl[i]:
            continue
        j = pos.get(lab)
        if j is None:                      # this label's own gene is not on the axis
            continue
        m = pb.cpm_sum[i, j] / n[i]
        lfcs.append(np.log2((m + pseudocount) / (ctrl_mean[j] + pseudocount)))

    out = {"n_self_measurable": len(lfcs),
           "median_self_log2fc": float(np.median(lfcs)) if lfcs else None,
           "frac_positive": float(np.mean([x > 0 for x in lfcs])) if lfcs else None,
           "bar": max_median_log2fc}
    if len(lfcs) < min_arms:
        return out | {"status": NOT_APPLICABLE}
    if out["median_self_log2fc"] > max_median_log2fc:
        raise ValueError(
            f"on-target knockdown check FAILED: median self log2FC over "
            f"{len(lfcs)} self-measurable arms is {out['median_self_log2fc']:+.3f}, "
            f"which is not at or below {max_median_log2fc:+.2f}. CRISPRi silences, so "
            f"this aggregate does not look like the screen it claims to be -- check the "
            f"perturbation column, the control arm, the gene axis and the orientation of "
            f"the matrix before trusting any delta built from it.")
    return out | {"status": "ok"}
