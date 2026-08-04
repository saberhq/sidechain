"""Rung 3 — residual head. A 2-3 layer message-passing GNN over the multi-relational
prior graph (from graph_builder) that predicts the RESIDUAL left by Rungs 1-2, not
raw expression. Residual-gated: only kept if it beats the backbone on the local eval.
Node features = frozen expression embeddings + cis 3'UTR sequence embeddings.
"""
from __future__ import annotations


class ResidualGNN:
    def __init__(self, graph, layers: int = 2, hidden_dim: int = 256):
        self.graph = graph
        self.layers = layers
        self.hidden_dim = hidden_dim

    def fit_residual(self, backbone_pred, truth):
        """Train to predict (truth - backbone_pred)."""
        raise NotImplementedError

    def predict_residual(self, perturbations):
        raise NotImplementedError
