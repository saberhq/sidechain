"""Rungs 0, 0b and 1 of the 2025 ladder: predictors that shift drawn control cells and emit them as predicted cells.

    model = StatisticalBackbone(pert_col="target_gene", control_label="non-targeting").fit(train)
    pred = model.predict({"GENE": 100}, include_control=500)     # an AnnData of predicted cells

Rung 0 emits controls; 0b adds the mean training shift; 1 adds the mean pseudobulk delta gated per gene by DEG
frequency and sets the target gene to the mean self-knockdown. Reads the AnnData given; writes nothing.
Notes: private/code-notes/src/sidechain/models/baseline_stats.py.md
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def _dense(x) -> np.ndarray:
    return np.asarray(x.todense()) if sp.issparse(x) else np.asarray(x)


def _col_moments(x, chunk: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene (mean, variance), accumulated in float64 over `chunk`-row blocks without densifying."""
    if not sp.issparse(x):
        arr = np.asarray(x, dtype=np.float64)
        return arr.mean(axis=0), arr.var(axis=0)

    n = x.shape[0]
    total = np.zeros(x.shape[1], dtype=np.float64)
    total_sq = np.zeros(x.shape[1], dtype=np.float64)
    for start in range(0, n, chunk):
        blk = x[start : start + chunk].astype(np.float64)
        total += np.asarray(blk.sum(axis=0)).ravel()
        total_sq += np.asarray(blk.multiply(blk).sum(axis=0)).ravel()

    mean = total / n
    return mean, np.maximum(total_sq / n - mean**2, 0.0)


def _col_mean(x) -> np.ndarray:
    return _col_moments(x)[0]


def _min_nonzero(x) -> float:
    """Smallest stored nonzero value, read from `.data` for a sparse matrix."""
    data = x.data if sp.issparse(x) else np.asarray(x)
    positive = data[data > 0]
    return float(positive.min()) if positive.size else 0.0


class PerturbationPredictor(ABC):
    """Common interface: fit on training cells, emit predicted cells."""

    def __init__(self, pert_col: str = "target_gene", control_label: str = "non-targeting"):
        self.pert_col = pert_col
        self.control_label = control_label
        self.var: pd.DataFrame | None = None
        self.control_cells_: np.ndarray | None = None
        self.control_mean_: np.ndarray | None = None

    # -- fitting --

    def fit(self, adata: ad.AnnData) -> PerturbationPredictor:
        labels = adata.obs[self.pert_col].astype(str).to_numpy()
        ctrl_mask = labels == self.control_label
        if not ctrl_mask.any():
            raise ValueError(
                f"No control cells ({self.pert_col}=={self.control_label!r}) in the "
                f"training data; the whole ladder is defined relative to controls."
            )
        self.var = adata.var.copy()
        # The control pool keeps its layout; predict() densifies only the rows it draws.
        self.control_cells_ = adata.X[ctrl_mask]
        self.control_mean_, self.control_var_ = _col_moments(self.control_cells_)
        # The smallest value the assay represents; predict() snaps values below it to zero.
        self.min_nonzero_ = _min_nonzero(self.control_cells_)
        logger.info("%s: fit on %d control cells", type(self).__name__, int(ctrl_mask.sum()))
        self._fit(adata, labels, ctrl_mask)
        return self

    def _fit(self, adata: ad.AnnData, labels: np.ndarray, ctrl_mask: np.ndarray) -> None:
        """Hook for subclass-specific fitting."""

    # -- prediction --

    @abstractmethod
    def delta_for(self, perturbation: str) -> np.ndarray:
        """Per-gene shift to apply to the control profile for this perturbation."""

    def predict(
        self,
        perturbations: dict[str, int],
        *,
        seed: int = 0,
        include_control: int | None = None,
        sparse_output: bool = True,
    ) -> ad.AnnData:
        """Emit predicted cells: `perturbations` maps label -> cell count; `include_control` adds that many controls.

        With `sparse_output`, values below the controls' smallest nonzero are snapped to zero and X is CSR.
        """
        if self.control_cells_ is None:
            raise RuntimeError("predict() called before fit()")
        rng = np.random.default_rng(seed)
        pool = self.control_cells_

        floor = getattr(self, "min_nonzero_", 0.0) if sparse_output else 0.0

        def _emit(rows: np.ndarray, delta: np.ndarray | None):
            """Densify one perturbation's draw, shift it, and re-sparsify."""
            block = _dense(pool[rows]).astype(np.float32, copy=True)
            if delta is not None:
                block += delta
            # Expression is log1p of a non-negative quantity; a shift can push a
            # gene below zero, which is not a representable expression value.
            np.clip(block, 0, None, out=block)
            if floor > 0:
                block[block < floor] = 0.0
            return sp.csr_matrix(block) if sparse_output else block

        blocks: list = []
        labels: list[str] = []
        for pert, n in perturbations.items():
            if n <= 0:
                continue
            blocks.append(_emit(rng.integers(0, pool.shape[0], size=n), self.delta_for(pert)))
            labels.extend([pert] * n)

        if include_control:
            blocks.append(_emit(rng.integers(0, pool.shape[0], size=include_control), None))
            labels.extend([self.control_label] * include_control)

        if not blocks:
            raise ValueError("predict() got no perturbations with a positive cell count")

        X = sp.vstack(blocks, format="csr") if sparse_output else np.vstack(blocks)

        obs = pd.DataFrame({self.pert_col: labels})
        obs.index = [f"pred_{i}" for i in range(len(labels))]
        return ad.AnnData(X=X, obs=obs, var=self.var.copy())


class PredictControl(PerturbationPredictor):
    """Rung 0. Every perturbation looks exactly like a control cell."""

    def delta_for(self, perturbation: str) -> np.ndarray:
        return np.zeros_like(self.control_mean_)


class PredictMeanPerturbation(PerturbationPredictor):
    """Rung 0b. Every perturbation gets the average training perturbation shift."""

    def _fit(self, adata: ad.AnnData, labels: np.ndarray, ctrl_mask: np.ndarray) -> None:
        # Sparse .mean(0): the column means without densifying.
        block = adata.X[~ctrl_mask]
        pert_mean = (
            np.asarray(block.mean(axis=0)).ravel() if sp.issparse(block) else block.mean(axis=0)
        )
        self.mean_delta_ = pert_mean - self.control_mean_

    def delta_for(self, perturbation: str) -> np.ndarray:
        return self.mean_delta_


class StatisticalBackbone(PerturbationPredictor):
    """Rung 1. Generic response gated by DEG frequency, plus target knockdown.

    Fits per gene `mean_delta_`, `deg_frequency_` (share of perturbations with |z| > deg_z), `gene_variance_`
    (controls) and `effect_size_` (mean |delta|); `knockdown_` is the scalar mean self-delta of a targeted gene.
    """

    def __init__(
        self,
        pert_col: str = "target_gene",
        control_label: str = "non-targeting",
        *,
        deg_z: float = 2.0,
        response_scale: float = 1.0,
        min_cells: int = 10,
        apply_knockdown: bool = True,
    ):
        super().__init__(pert_col, control_label)
        self.deg_z = deg_z
        self.response_scale = response_scale
        self.min_cells = min_cells
        self.apply_knockdown = apply_knockdown

    def _fit(self, adata: ad.AnnData, labels: np.ndarray, ctrl_mask: np.ndarray) -> None:
        gene_names = np.asarray(adata.var_names.astype(str))
        gene_pos = {g: i for i, g in enumerate(gene_names)}

        self.gene_variance_ = self.control_var_
        n_ctrl = self.control_cells_.shape[0]

        deltas: list[np.ndarray] = []
        zscores: list[np.ndarray] = []
        self_knockdowns: list[float] = []
        for pert in sorted(set(labels) - {self.control_label}):
            mask = labels == pert
            n_pert = int(mask.sum())
            if n_pert < self.min_cells:
                continue
            block = _dense(adata.X[mask])
            delta = block.mean(axis=0) - self.control_mean_
            deltas.append(delta)

            # z is the delta over the standard error of a difference of means.
            se = np.sqrt(self.gene_variance_ / n_ctrl + block.var(axis=0) / n_pert)
            zscores.append(delta / np.maximum(se, 1e-9))

            if pert in gene_pos:
                self_knockdowns.append(float(delta[gene_pos[pert]]))

        if not deltas:
            raise ValueError(
                f"No training perturbation had >= min_cells={self.min_cells} cells."
            )

        D = np.vstack(deltas)
        Z = np.vstack(zscores)
        self.n_train_perts_ = D.shape[0]
        self.mean_delta_ = D.mean(axis=0)
        self.deg_frequency_ = (np.abs(Z) > self.deg_z).mean(axis=0)
        self.effect_size_ = np.abs(D).mean(axis=0)
        self.knockdown_ = float(np.mean(self_knockdowns)) if self_knockdowns else 0.0
        self._gene_pos = gene_pos

        # DEG frequency gates the generic response per gene.
        self._weighted_delta = self.response_scale * self.deg_frequency_ * self.mean_delta_

        logger.info(
            "StatisticalBackbone: %d train perts | %d genes with DEG freq > 0.1 | "
            "mean self-knockdown %.3f",
            self.n_train_perts_,
            int((self.deg_frequency_ > 0.1).sum()),
            self.knockdown_,
        )

    def delta_for(self, perturbation: str) -> np.ndarray:
        delta = self._weighted_delta.copy()
        if self.apply_knockdown:
            pos = self._gene_pos.get(perturbation)
            if pos is not None:
                # CRISPRi drives the targeted transcript down; this replaces the
                # generic response for that one gene rather than adding to it.
                delta[pos] = self.knockdown_
        return delta

    def feature_table(self) -> pd.DataFrame:
        """The explicit statistical features, for inspection and as Rung 3 input."""
        return pd.DataFrame(
            {
                "mean_delta": self.mean_delta_,
                "deg_frequency": self.deg_frequency_,
                "mean_expression": self.control_mean_,
                "gene_variance": self.gene_variance_,
                "effect_size_prior": self.effect_size_,
            },
            index=self.var.index if self.var is not None else None,
        )
