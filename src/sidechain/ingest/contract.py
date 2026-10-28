"""The shape every external perturbation dataset is adapted to: `HarmonizedDataset` and its `Provenance`.

`harmonize(path, spec=...)` and `qc_report(ds)` are stubs that raise NotImplementedError; nothing is read or written.
Notes: private/code-notes/src/sidechain/ingest/contract.py.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad


@dataclass(frozen=True)
class Provenance:
    """Where a dataset came from, and what its terms allow; `license` may be "unknown"."""

    source: str      # e.g. "GEO", "ArrayExpress", "author-hosted"
    accession: str   # e.g. "GSE90546"; "" when none exists
    license: str     # SPDX identifier where possible, else the stated terms
    retrieved: str   # ISO date the bytes were fetched


@dataclass
class HarmonizedDataset:
    """The one shape in which external perturbation data enters Sidechain.

    `adata`: var resolves via `loaders.gene_index`; obs[`pert_col`] names the silenced gene and holds
    `control_label`; obs[`context_col`] names the line; X is raw integer counts unless `counts_are_raw` is
    False, when `notes` names the transform. `granularity` is "cell" or "pseudobulk" (a sum of raw counts).
    """

    adata: ad.AnnData
    pert_col: str
    control_label: str
    context_col: str
    counts_are_raw: bool
    provenance: Provenance
    granularity: str = "cell"   # "cell" | "pseudobulk" -- set by the adapter, never inferred
    notes: dict = field(default_factory=dict)


def harmonize(path: str | Path, *, spec: dict) -> HarmonizedDataset:
    """Adapt one raw download into the contract; `spec` is a per-dataset block (renames, control label, context)."""
    # TODO(dev): per-source adapters.
    raise NotImplementedError


def qc_report(ds: HarmonizedDataset) -> dict:
    """Admission QC for a candidate corpus; thresholds are not set yet."""
    # TODO(dev): gates + report schema.
    raise NotImplementedError
