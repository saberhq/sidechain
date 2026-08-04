"""Pseudobulk strategies (Rung 1 backbone + replicate generation for DES).

Knobs mirror configs/model.yaml -> rung1_statistical.pseudobulk:
  aggregate : 'sum' (count-native, DESeq2/edgeR-friendly) | 'mean' | 'median'
  group_by  : e.g. [target_gene] (coarse) or [target_gene, batch] (gives variance)
  min_cells : drop groups below this many cells

Bootstrapping cells within a group manufactures pseudo-replicates, which gives
DES a dispersion to score without needing a full generative model.
"""
from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

_AGGREGATORS = ("sum", "mean", "median")


def _to_dense(x) -> np.ndarray:
    return np.asarray(x.todense()) if sp.issparse(x) else np.asarray(x)


def _aggregate(block, how: str) -> np.ndarray:
    """Collapse a (cells, genes) block to a single (genes,) profile."""
    if how == "sum":
        # Stays sparse: .sum(0) on a CSR block never densifies the whole block.
        return np.asarray(block.sum(axis=0)).ravel() if sp.issparse(block) else block.sum(0)
    if how == "mean":
        return np.asarray(block.mean(axis=0)).ravel() if sp.issparse(block) else block.mean(0)
    if how == "median":
        # No sparse median; densify this group only (one perturbation at a time).
        return np.median(_to_dense(block), axis=0)
    raise ValueError(f"Unknown aggregate {how!r}; expected one of {_AGGREGATORS}")


def pseudobulk(
    adata: ad.AnnData,
    group_by: list[str],
    aggregate: str = "sum",
    min_cells: int = 25,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> ad.AnnData:
    """Collapse cells to pseudobulk profiles.

    If `n_bootstrap` > 0, resample cells with replacement within each group to
    emit that many pseudo-replicates per group instead of one profile.

    Returns an AnnData with one row per (group [, replicate]); the grouping keys
    are preserved as obs columns, plus `n_cells` and `replicate`.
    """
    if aggregate not in _AGGREGATORS:
        raise ValueError(f"Unknown aggregate {aggregate!r}; expected one of {_AGGREGATORS}")
    missing = [c for c in group_by if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"group_by columns not in obs: {missing}. Have: {list(adata.obs.columns)}")

    rng = np.random.default_rng(seed)
    keys = adata.obs[group_by].astype(str)
    # Single string key keeps grouping cheap and order deterministic.
    joined = keys.agg("||".join, axis=1) if len(group_by) > 1 else keys.iloc[:, 0]

    profiles: list[np.ndarray] = []
    meta: list[dict] = []
    n_dropped = 0

    for gval, positions in sorted(pd.Series(np.arange(adata.n_obs), index=joined.to_numpy())
                                  .groupby(level=0)):
        pos = positions.to_numpy()
        if len(pos) < min_cells:
            n_dropped += 1
            continue
        block = adata.X[pos]
        parts = dict(zip(group_by, str(gval).split("||")))

        if n_bootstrap <= 0:
            profiles.append(_aggregate(block, aggregate))
            meta.append({**parts, "n_cells": len(pos), "replicate": 0})
        else:
            for r in range(n_bootstrap):
                draw = rng.integers(0, len(pos), size=len(pos))
                profiles.append(_aggregate(block[draw], aggregate))
                meta.append({**parts, "n_cells": len(pos), "replicate": r})

    if n_dropped:
        logger.info("pseudobulk: dropped %d groups below min_cells=%d", n_dropped, min_cells)
    if not profiles:
        raise ValueError(
            f"pseudobulk produced no groups: every group had < min_cells={min_cells} cells."
        )

    obs = pd.DataFrame(meta)
    obs.index = [
        "|".join(str(row[c]) for c in group_by) + f"|rep{row['replicate']}"
        for _, row in obs.iterrows()
    ]
    out = ad.AnnData(X=np.vstack(profiles).astype(np.float32), obs=obs, var=adata.var.copy())
    logger.info("pseudobulk: %d profiles x %d genes", out.n_obs, out.n_vars)
    return out


def control_profile(
    pb: ad.AnnData,
    pert_col: str = "target_gene",
    control_label: str = "non-targeting",
) -> np.ndarray:
    """Mean pseudobulk profile of the control group(s)."""
    mask = pb.obs[pert_col].astype(str).to_numpy() == control_label
    if not mask.any():
        raise ValueError(
            f"No control rows: {pert_col}=={control_label!r} matched nothing. "
            f"Present: {sorted(pb.obs[pert_col].astype(str).unique())[:10]}"
        )
    return _to_dense(pb.X[mask]).mean(axis=0)


def delta_vs_control(
    pb: ad.AnnData,
    pert_col: str = "target_gene",
    control_label: str = "non-targeting",
) -> pd.DataFrame:
    """Per-gene delta of each perturbation pseudobulk vs the matched control.

    Returns a (n_perturbations, n_genes) frame indexed by perturbation. Controls
    are excluded from the rows -- their delta is zero by construction.
    """
    ctrl = control_profile(pb, pert_col, control_label)
    labels = pb.obs[pert_col].astype(str).to_numpy()
    X = _to_dense(pb.X)

    rows, index = [], []
    for pert in sorted(set(labels) - {control_label}):
        rows.append(X[labels == pert].mean(axis=0) - ctrl)
        index.append(pert)
    return pd.DataFrame(np.vstack(rows), index=index, columns=list(pb.var_names))
