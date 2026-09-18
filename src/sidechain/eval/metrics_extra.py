"""Extra DEV metrics (not scored by Arc) to understand *why* a model wins/fails.

An `e_distance` stub lived here from 2026-08-01 as a local stand-in for PDS, from
before the mirror existed. It was never implemented and never called, and there is
no longer a question for it to answer: `sidechain.eval.mirror2026` scores the real
cell-eval2 PDS on every fold. Deleted 2026-09-17 with T21, because a stub reads as
work owed, then revived properly under T94 as `sidechain.eval.e_distance` -- not
here, because it needed its own preprocessing recipe and permutation-test machinery,
not a one-line stand-in.

**E-distance was never a PDS proxy**, so nothing about a PDS question is settled by
reasoning about it. `pds_cosine` ranks *pseudobulk deltas* -- one vector per
perturbation -- and asks whether our predicted delta is nearer that target's real
delta than any other target's, discarding within-group spread entirely. E-distance
compares *two clouds of cells* in PCA space, half its formula IS the within-group
spread, and it needs no prediction at all: it is a property of real data. Its use is
on the input side, grading the corpora we pool deltas from -- see
`sidechain.eval.e_distance` and `research/ideas/e-test-source-perturbation-gate.md`.

**Do not reach for pertpy's E-distance.** Measured against 1.0.3, not inferred: its
`Edistance` is plain Euclidean rather than squared, applies no bias correction, and
ignores `cell_wise_metric` on the `__call__` path -- it does not reproduce the
paper. `sidechain.eval.e_distance` wraps the authors' own `scperturb` package
instead, which does.
"""
from __future__ import annotations


def generalization_gap(train_scores: dict, holdout_scores: dict) -> dict:
    """Per-metric (train - holdout) — the overfitting alarm."""
    return {k: train_scores[k] - holdout_scores.get(k, float("nan")) for k in train_scores}
