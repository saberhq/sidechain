"""Extra DEV metrics (not scored by Arc) to understand *why* a model wins/fails:
E-distance (scPerturb/pertpy), calibration, per-pathway accuracy, and the
train-vs-held-out-context generalization gap.
"""
from __future__ import annotations


def e_distance(pred, truth):
    """E-distance between predicted and true perturbation distributions (PDS proxy)."""
    raise NotImplementedError


def generalization_gap(train_scores: dict, holdout_scores: dict) -> dict:
    """Per-metric (train - holdout) — the overfitting alarm."""
    return {k: train_scores[k] - holdout_scores.get(k, float("nan")) for k in train_scores}
