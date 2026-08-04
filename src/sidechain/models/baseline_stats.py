"""Rung 1 — the workhorse. Pseudobulk delta + explicit statistical features
(DEG frequency, mean expression, gene variance, effect-size prior). This is the
bar every fancier rung must clear on the local mirror.
"""
from __future__ import annotations


class StatisticalBackbone:
    def fit(self, adata) -> "StatisticalBackbone":
        raise NotImplementedError

    def predict_delta(self, perturbations):
        """Return per-gene delta vectors (+ dispersion for DES) per perturbation."""
        raise NotImplementedError
