"""Rung 0 and Rung 1 of the ladder.

Rung 0 -- naive controls. Predict the control profile for every perturbation.
Establishes the floor: a model that cannot beat this is not modelling anything.

Rung 1 -- the statistical backbone. Pseudobulk delta plus explicit statistical
features (DEG frequency, mean expression, per-gene variance, effect-size prior).
This is the bar every fancier rung must clear on the local mirror.

The prediction task is response to *unseen* perturbations, so the backbone cannot
look up a per-perturbation delta at predict time. What it can exploit, and what
carried the 2025 winners, is that responses share a large common component:

  1. a generic transcriptional response shared across perturbations, weighted
     per-gene by how often that gene responds at all (DEG frequency), and
  2. the target gene's own knockdown -- CRISPRi drives the targeted transcript
     down, the single most reliable per-perturbation signal available without
     having seen that perturbation.

Both models emit *cells*, not one mean vector: DES needs dispersion to score.
Cells are drawn from the control pool and shifted, so predicted cell-to-cell
variation matches the real assay's rather than being invented.
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
    """Per-gene (mean, variance) without densifying the whole matrix.

    Two precision traps, both of which showed up as real disagreement against a
    dense float64 recompute:

      * scipy's sparse reductions accumulate in the matrix dtype, so summing 38k
        float32 values drifts by ~1e-6 relative.
      * variance via E[X^2] - E[X]^2 loses further precision to catastrophic
        cancellation, which in float32 reached ~1e-3 relative.

    Accumulating both moments in float64 over row chunks fixes both while never
    holding more than `chunk` rows densely.
    """
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
    """Smallest stored nonzero value. For CSR this reads .data directly, so it
    never materializes the matrix."""
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
        # The control pool is kept in whatever layout the data arrived in. On the
        # full 2025 file there are 38,176 control cells, so densifying it here
        # would cost 2.8 GB held for the entire run; predict() densifies only the
        # rows it draws instead.
        self.control_cells_ = adata.X[ctrl_mask]
        self.control_mean_, self.control_var_ = _col_moments(self.control_cells_)
        # The smallest expression value the assay can actually represent. Adding a
        # delta to a dropped-out (zero) gene produces values far below this, which
        # are numerical artifacts rather than predictions -- see predict().
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
        """Emit predicted cells. `perturbations` maps label -> number of cells.

        `include_control` adds that many control cells under the control label;
        cell-eval requires the control to be present in both pred and real.

        `sparse_output` matters more than it looks. Real single-cell data is ~54%
        zeros from dropout, and because the deltas are small our predictions are
        ~50% zeros too -- but built naively they are stored DENSE. At the full
        2025 scale that is 5.2 GB against 2.4 GB for the equivalent sparse
        matrix, which was enough to drive a 17 GB machine into swap thrashing.
        Values below the assay's smallest representable level are snapped to zero
        first: adding a delta to a dropped-out gene yields ~1e-3 where the real
        data's floor is ~6e-2, so those entries are artifacts of the arithmetic,
        not predictions.
        """
        if self.control_cells_ is None:
            raise RuntimeError("predict() called before fit()")
        rng = np.random.default_rng(seed)
        pool = self.control_cells_

        floor = getattr(self, "min_nonzero_", 0.0) if sparse_output else 0.0

        def _emit(rows: np.ndarray, delta: np.ndarray | None):
            """Densify ONE perturbation's draw, shift it, and re-sparsify.

            The pool itself stays in its original (sparse) layout -- densifying it
            whole would cost 2.8 GB on the full 2025 file, held for the entire run.
            """
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
    """Rung 0b. Every perturbation gets the average training perturbation shift.

    A more honest floor than PredictControl: it already captures the "cells
    respond to being perturbed at all" effect, so beating it requires genuinely
    perturbation-specific signal.
    """

    def _fit(self, adata: ad.AnnData, labels: np.ndarray, ctrl_mask: np.ndarray) -> None:
        # Sparse .mean(0) avoids materializing every perturbed cell at once --
        # 40k x 18k float32 is ~3 GB, and we only need the column means.
        block = adata.X[~ctrl_mask]
        pert_mean = (
            np.asarray(block.mean(axis=0)).ravel() if sp.issparse(block) else block.mean(axis=0)
        )
        self.mean_delta_ = pert_mean - self.control_mean_

    def delta_for(self, perturbation: str) -> np.ndarray:
        return self.mean_delta_


class StatisticalBackbone(PerturbationPredictor):
    """Rung 1. Generic response gated by DEG frequency, plus target knockdown.

    Fitted quantities (per-gene unless noted):
      mean_delta_      mean pseudobulk delta across training perturbations
      deg_frequency_   fraction of training perturbations where |z| > deg_z
      gene_variance_   variance across control cells
      effect_size_     mean |delta| across training perturbations
      knockdown_       scalar: mean self-delta of a targeted gene (negative)
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

            # The quantity being tested is a DIFFERENCE OF MEANS, so its noise is
            # the standard error of that difference -- not the per-cell SD, which
            # overstates it by ~sqrt(n) and gates away nearly every gene.
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

        # DEG frequency is the gate: genes that respond to many perturbations get
        # the generic response, genes that respond to none stay put. Without it
        # the mean delta smears a weak shift across all 18k genes and MAE suffers.
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
