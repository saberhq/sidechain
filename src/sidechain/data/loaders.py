"""Thin wrappers around Arc's cell-load so the rest of Sidechain speaks AnnData.

TODO(dev): wire to ArcInstitute/cell-load once the 2026 dataset format is known.
Keep this module the ONLY place that touches the challenge file format, so a
format change at kickoff is a one-file edit.
"""
from __future__ import annotations
from pathlib import Path
import anndata as ad


def load_challenge_split(path: str | Path, split: str) -> ad.AnnData:
    """Load a challenge split ('train' | 'public_test' | 'private_local').

    Returns AnnData with .obs['perturbation'] and matched controls tagged
    (NTC / safe-targeting) in .obs['control_type'].
    """
    raise NotImplementedError("wire to cell-load at kickoff")


def gene_index(adata: ad.AnnData, id_type: str = "ensembl_gene_id") -> dict[str, int]:
    """Canonical {gene_id -> row position}. Every PriorSource aligns to this."""
    ids = adata.var[id_type] if id_type in adata.var else adata.var_names
    return {str(g): i for i, g in enumerate(ids)}
