"""Rung 4 — metric-aware fusion. Combine backbone + STATE + residual, and train the
final under a fused loss (BioMap-style: PDS + DES + small-weight MAE). Emit multiple
replicate outputs per perturbation so DES has dispersion to score.
"""
from __future__ import annotations


def fused_loss(pred, truth, weights: dict[str, float]):
    """weights e.g. {'pds':1.0,'des':1.0,'mae':0.2}. Terms mirror the official metrics."""
    raise NotImplementedError


def assemble_final(backbone, state, residual, n_replicate_outputs: int = 20):
    """Ensemble the rungs and expand to replicate outputs for DES."""
    raise NotImplementedError
