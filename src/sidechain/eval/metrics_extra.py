"""Extra DEV metrics (not scored by Arc) to understand *why* a model wins/fails.

An `e_distance` stub lived here from 2026-08-01 as a local stand-in for PDS, from
before the mirror existed. It was never implemented and never called, and there is
no longer a question for it to answer: `sidechain.eval.mirror2026` scores the real
cell-eval2 PDS on every fold. Deleted 2026-09-17 with T21, because a stub reads as
work owed. If a distributional distance is ever wanted for a diagnostic, take
pertpy's rather than re-deriving one here.
"""
from __future__ import annotations


def generalization_gap(train_scores: dict, holdout_scores: dict) -> dict:
    """Per-metric (train - holdout) — the overfitting alarm."""
    return {k: train_scores[k] - holdout_scores.get(k, float("nan")) for k in train_scores}
