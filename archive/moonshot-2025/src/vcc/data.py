"""Data loading utilities for the Virtual Cell Challenge datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import anndata as ad
import pandas as pd

from .config import get_data_dir

DEFAULT_DATASETS: Mapping[str, str] = {
    # Only the training AnnData matrix is distributed ahead of time.
    "train": "adata_Training.h5ad",
}
GENE_NAMES_FILE = "gene_names.csv"
PERT_COUNTS_VALIDATION_FILE = "pert_counts_Validation.csv"


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a requested dataset cannot be located."""


def _resolve_dataset_path(name: str, data_dir: Path) -> Path:
    try:
        relative = DEFAULT_DATASETS[name.lower()]
    except KeyError as exc:
        known = ", ".join(sorted(DEFAULT_DATASETS))
        raise DatasetNotFoundError(
            f"Unknown dataset '{name}'. Expected one of: {known}. "
            "If you intended to load validation/test data, pass the explicit file path once it is released."
        ) from exc
    return (data_dir / relative).resolve()


def load_dataset(name: str, *, data_dir: Optional[Path] = None, backed: bool = False) -> ad.AnnData:
    """Load a named dataset (currently only 'train') from the configured data directory.

    Other splits can be loaded by passing an explicit path to ``load_anndata`` once released.
    """
    directory = data_dir or get_data_dir()
    path = _resolve_dataset_path(name, directory)
    return load_anndata(path, backed=backed)


def load_anndata(path: Path | str, *, backed: bool = False) -> ad.AnnData:
    """Load an AnnData object from disk.

    When ``backed=True``, the data matrix is not fully loaded into memory (read-only).
    """
    ad_path = Path(path).expanduser().resolve()
    if not ad_path.exists():
        raise DatasetNotFoundError(f"Dataset {ad_path} does not exist")
    mode = "r" if backed else None
    return ad.read_h5ad(ad_path, backed=mode)


def load_gene_names(*, data_dir: Optional[Path] = None) -> pd.Series:
    """Load the ordered list of gene names available in the challenge data."""
    directory = data_dir or get_data_dir()
    path = (directory / GENE_NAMES_FILE).resolve()
    if not path.exists():
        raise DatasetNotFoundError(f"Gene name file {path} does not exist")
    df = pd.read_csv(path, header=None, names=["gene_name"])
    return df["gene_name"]


def load_validation_perturbations(*, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load metadata for validation perturbations supplied by the organizers."""
    directory = data_dir or get_data_dir()
    path = (directory / PERT_COUNTS_VALIDATION_FILE).resolve()
    if not path.exists():
        raise DatasetNotFoundError(f"Validation perturbation file {path} does not exist")
    return pd.read_csv(path)


def compare_gene_sets(adata: ad.AnnData, gene_names: pd.Series) -> dict[str, object]:
    """Compare AnnData variable names with the provided gene name list."""
    adata_genes = pd.Index(adata.var_names.astype(str))
    ref_genes = pd.Index(gene_names.astype(str))

    missing_in_ref = adata_genes.difference(ref_genes)
    missing_in_adata = ref_genes.difference(adata_genes)
    overlap = adata_genes.intersection(ref_genes)

    return {
        "n_adata": len(adata_genes),
        "n_reference": len(ref_genes),
        "n_overlap": len(overlap),
        "missing_in_reference": tuple(missing_in_ref[:5]),
        "missing_in_adata": tuple(missing_in_adata[:5]),
        "is_aligned": len(missing_in_ref) == 0 and len(missing_in_adata) == 0,
    }


def top_obs_counts(adata: ad.AnnData, column: str, *, limit: int = 10) -> pd.Series:
    """Return the most frequent values for an obs column."""
    if column not in adata.obs.columns:
        available = ", ".join(sorted(adata.obs.columns))
        raise KeyError(f"Column '{column}' not found in adata.obs. Available: {available}")
    return adata.obs[column].value_counts().head(limit)


@dataclass(frozen=True)
class Summary:
    """Lightweight metadata summary of an AnnData object."""

    path: Path
    n_obs: int
    n_vars: int
    obs_columns: tuple[str, ...]
    var_columns: tuple[str, ...]
    layers: tuple[str, ...]
    uns_keys: tuple[str, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_obs, self.n_vars


def summarize(adata: ad.AnnData, *, source: Path | None = None) -> Summary:
    """Return basic structural information about the given AnnData object."""
    if source is None:
        raw_path = adata.filename
        source_path = Path(raw_path) if raw_path is not None else Path("unknown")
    else:
        source_path = Path(source)

    return Summary(
        path=source_path.resolve(),
        n_obs=adata.n_obs,
        n_vars=adata.n_vars,
        obs_columns=tuple(sorted(adata.obs.columns)),
        var_columns=tuple(sorted(adata.var.columns)),
        layers=tuple(sorted(adata.layers.keys())),
        uns_keys=tuple(sorted(adata.uns.keys())),
    )
