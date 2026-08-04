"""Pseudobulk strategies (Rung 1 backbone + replicate generation for DES).

Knobs mirror configs/model.yaml -> rung1_statistical.pseudobulk:
  aggregate : 'sum' (count-native, DESeq2/edgeR-friendly) | 'mean' | 'median'
  group_by  : e.g. [perturbation] (coarse) or [perturbation, replicate] (gives variance)
  min_cells : drop groups below this many cells

Optionally run Mixscape (pertpy) first to drop non-perturbed escaper cells, then
bootstrap cells within a group to manufacture pseudo-replicates -> a distribution
DES can score without a full generative model.
"""
from __future__ import annotations
import anndata as ad
import numpy as np


def pseudobulk(
    adata: ad.AnnData,
    group_by: list[str],
    aggregate: str = "sum",
    min_cells: int = 25,
    n_bootstrap: int = 0,
) -> ad.AnnData:
    """Collapse cells to pseudobulk profiles. If n_bootstrap>0, resample cells
    within each group to emit multiple pseudo-replicates per group."""
    raise NotImplementedError("dev: implement sum/mean/median + bootstrap replicates")


def delta_vs_control(pb: ad.AnnData, control_key: str = "control_type") -> np.ndarray:
    """Per-gene delta of each perturbation pseudobulk vs matched NTC/safe-targeting."""
    raise NotImplementedError("dev: subtract matched-control pseudobulk")
