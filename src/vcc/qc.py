"""Reusable QC helpers following Scanpy/scverse best practices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


@dataclass(frozen=True)
class QCThresholds:
    """Common cell/gene QC thresholds (best-practices defaults)."""

    min_genes: int = 200
    max_genes: Optional[int] = 6000
    min_counts: int = 1000
    max_counts: Optional[int] = None
    max_mt_percent: Optional[float] = 20.0
    min_cells_per_gene: int = 3

    def as_dict(self) -> dict[str, object]:
        return {
            "min_genes": self.min_genes,
            "max_genes": self.max_genes,
            "min_counts": self.min_counts,
            "max_counts": self.max_counts,
            "max_mt_percent": self.max_mt_percent,
            "min_cells_per_gene": self.min_cells_per_gene,
        }


def _starts_with_any(names: Sequence[str], prefixes: Iterable[str]) -> pd.Series:
    """Vectorized prefix check for gene names."""
    upper_names = pd.Index([str(n).upper() for n in names])
    upper_prefixes = tuple(p.upper() for p in prefixes)
    return pd.Series([name.startswith(upper_prefixes) for name in upper_names], index=names)


def annotate_qc_metrics(
    adata: sc.AnnData,
    *,
    mito_prefix: str = "MT-",
    ribo_prefixes: Sequence[str] = ("RPS", "RPL"),
    ercc_prefix: str = "ERCC-",
) -> None:
    """Add QC flags (mito/ribo/ERCC) and compute standard qc metrics."""
    adata.var["mt"] = _starts_with_any(adata.var_names, (mito_prefix,))
    adata.var["ribo"] = _starts_with_any(adata.var_names, ribo_prefixes)
    adata.var["ercc"] = _starts_with_any(adata.var_names, (ercc_prefix,))

    qc_vars = [var for var in ("mt", "ribo", "ercc") if adata.var[var].any()]
    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, inplace=True)


def filter_genes(adata: sc.AnnData, *, min_cells: int) -> int:
    """Filter genes expressed in fewer than min_cells; returns genes removed."""
    if min_cells <= 0:
        return 0
    n_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    return n_before - adata.n_vars


def cell_filter_mask(adata: sc.AnnData, thresholds: QCThresholds) -> pd.Series:
    """Return a boolean mask of cells passing QC thresholds."""
    obs = adata.obs
    mask = pd.Series(True, index=adata.obs_names)

    if thresholds.min_genes:
        mask &= obs["n_genes_by_counts"] >= thresholds.min_genes
    if thresholds.max_genes:
        mask &= obs["n_genes_by_counts"] <= thresholds.max_genes
    if thresholds.min_counts:
        mask &= obs["total_counts"] >= thresholds.min_counts
    if thresholds.max_counts:
        mask &= obs["total_counts"] <= thresholds.max_counts
    if thresholds.max_mt_percent and "pct_counts_mt" in obs:
        mask &= obs["pct_counts_mt"] <= thresholds.max_mt_percent

    return mask


def summarize_qc(
    adata_raw: sc.AnnData,
    adata_filtered: sc.AnnData,
    *,
    thresholds: QCThresholds,
    genes_dropped: int,
) -> pd.DataFrame:
    """Return a tidy summary table of QC decisions."""
    return pd.DataFrame(
        [
            {
                "stage": "raw",
                "cells": adata_raw.n_obs,
                "genes": adata_raw.n_vars,
                "median_genes_per_cell": adata_raw.obs["n_genes_by_counts"].median(),
                "median_counts_per_cell": adata_raw.obs["total_counts"].median(),
                "median_pct_mt": adata_raw.obs.get("pct_counts_mt", pd.Series(dtype=float)).median(),
            },
            {
                "stage": "filtered",
                "cells": adata_filtered.n_obs,
                "genes": adata_filtered.n_vars,
                "median_genes_per_cell": adata_filtered.obs["n_genes_by_counts"].median(),
                "median_counts_per_cell": adata_filtered.obs["total_counts"].median(),
                "median_pct_mt": adata_filtered.obs.get("pct_counts_mt", pd.Series(dtype=float)).median(),
            },
        ]
    ).assign(**thresholds.as_dict(), genes_dropped=genes_dropped)


def _downsample_obs(obs: pd.DataFrame, *, max_points: int = 50000, random_state: int = 0) -> pd.DataFrame:
    """Downsample obs rows for plotting to keep figures light."""
    if len(obs) <= max_points:
        return obs
    return obs.sample(max_points, random_state=random_state)


def plot_basic_qc(
    adata: sc.AnnData,
    *,
    outdir: Path,
    max_points: int = 50000,
) -> list[Path]:
    """Create basic QC figures (violins + scatters) and return their paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    obs = adata.obs
    sampled = _downsample_obs(obs, max_points=max_points)

    # Violin plot of counts/genes/mt%
    metrics = [col for col in ("n_genes_by_counts", "total_counts", "pct_counts_mt") if col in obs.columns]
    if metrics:
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            ax.violinplot(obs[metric], showmeans=True)
            ax.set_title(metric)
        plt.tight_layout()
        path = outdir / "qc_violin.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # Scatter: counts vs genes
    if {"total_counts", "n_genes_by_counts"} <= set(sampled.columns):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(sampled["total_counts"], sampled["n_genes_by_counts"], s=2, alpha=0.4)
        ax.set_xlabel("total_counts")
        ax.set_ylabel("n_genes_by_counts")
        ax.set_title("Counts vs Genes per cell")
        plt.tight_layout()
        path = outdir / "qc_counts_vs_genes.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # Scatter: counts vs mt%
    if {"total_counts", "pct_counts_mt"} <= set(sampled.columns):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(sampled["total_counts"], sampled["pct_counts_mt"], s=2, alpha=0.4, color="tab:red")
        ax.set_xlabel("total_counts")
        ax.set_ylabel("pct_counts_mt")
        ax.set_title("Mito % vs counts")
        plt.tight_layout()
        path = outdir / "qc_counts_vs_mt.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    return saved


def write_markdown_report(
    *,
    outdir: Path,
    summary: pd.DataFrame,
    figure_paths: Sequence[Path],
    source: Path,
) -> Path:
    """Write a lightweight Markdown report linking saved plots."""
    outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir / "qc_report.md"
    rows = summary.to_dict(orient="records")

    lines = [
        f"# QC report for {source.name}",
        "",
        "## Summary",
    ]
    for row in rows:
        stage = row.pop("stage")
        lines.append(f"- **{stage}**: cells={row['cells']:,}, genes={row['genes']:,}")
        lines.append(
            f"  median genes/cell={row['median_genes_per_cell']:.0f}, "
            f"median counts/cell={row['median_counts_per_cell']:.0f}, "
            f"median pct_mt={row['median_pct_mt']:.2f}"
        )
    lines.extend(
        [
            "",
            "### Thresholds",
        ]
    )
    thresholds = summary.drop(columns=["stage", "cells", "genes", "median_genes_per_cell", "median_counts_per_cell", "median_pct_mt"]).iloc[0]
    for key, value in thresholds.items():
        lines.append(f"- {key}: {value}")

    if figure_paths:
        lines.extend(["", "## Figures"])
        for path in figure_paths:
            rel = path.name
            lines.append(f"![{rel}]({rel})")

    md_path.write_text("\n".join(lines))
    return md_path
