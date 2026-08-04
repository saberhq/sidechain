"""Rung 2 — wrap Arc's STATE (State Transition). Fine-tune via `state tx` on the
challenge data; expose predict() in the same delta interface as the backbone so the
two can be ensembled.
"""
from __future__ import annotations


class StateAdapter:
    def __init__(self, checkpoint: str | None = None):
        self.checkpoint = checkpoint

    def finetune(self, data_dir: str) -> None:
        # TODO(dev): shell out to `state tx` per Arc's VCC Colab; log to lamindb.
        raise NotImplementedError

    def predict_delta(self, perturbations):
        raise NotImplementedError
