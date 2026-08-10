"""Ingestion of external perturbation datasets — the contract, ahead of any fetcher.

The 2026 task scores unseen cell lines and Arc ships no net-new training data, so
external corpora are the only way this model ever sees a second context. Every
dataset enters through ONE shape: upstream formats stay in their per-source
adapters, and the rest of Sidechain only ever consumes a HarmonizedDataset.

Deliberately a stub, like the unbuilt rungs: the interface is the part worth
committing early. Fetchers and QC thresholds land dataset by dataset behind it,
and nothing downstream changes when they do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad


@dataclass(frozen=True)
class Provenance:
    """Where a dataset came from, and what its terms allow.

    `license` is a first-class field, not metadata trivia: a source whose terms
    we cannot meet is unusable regardless of its biology, and that is cheapest
    to discover at ingest time. "unknown" is a legal value, and a loud one.
    """

    source: str      # e.g. "GEO", "ArrayExpress", "author-hosted"
    accession: str   # e.g. "GSE90546"; "" when none exists
    license: str     # SPDX identifier where possible, else the stated terms
    retrieved: str   # ISO date the bytes were fetched


@dataclass
class HarmonizedDataset:
    """The one shape in which external perturbation data enters Sidechain.

    Contract on `adata`:
      var  -- resolvable to Ensembl gene IDs via `loaders.gene_index` (strict,
              so an ID-space mismatch raises instead of matching no prior);
      obs  -- `pert_col` names the silenced gene, `control_label` occurs in it,
              and `context_col` names the cell line / context;
      X    -- raw integer counts unless `counts_are_raw` is False, in which
              case `notes` must say what transformation the source applied.
    """

    adata: ad.AnnData
    pert_col: str
    control_label: str
    context_col: str
    counts_are_raw: bool
    provenance: Provenance
    notes: dict = field(default_factory=dict)


def harmonize(path: str | Path, *, spec: dict) -> HarmonizedDataset:
    """Adapt one raw download into the contract above.

    `spec` is a per-dataset block (column renames, control label, context name),
    so adapters stay declarative — the same pattern as prior sources.
    """
    # TODO(dev): per-source adapters. Which sources come first is decided by the
    # dataset shortlist, not here.
    raise NotImplementedError


def qc_report(ds: HarmonizedDataset) -> dict:
    """Admission QC for a candidate corpus.

    Thresholds are deliberately not set here yet; once settled they belong in
    this function, applied identically to every candidate.
    """
    # TODO(dev): gates + report schema.
    raise NotImplementedError
