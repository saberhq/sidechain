#!/usr/bin/env python
"""QC script following scverse best practices with minimal arguments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vcc.config import ConfigurationError
from vcc.data import DatasetNotFoundError, load_anndata, load_dataset
from vcc.qc import (
    QCThresholds,
    annotate_qc_metrics,
    cell_filter_mask,
    filter_genes,
    plot_basic_qc,
    summarize_qc,
    write_markdown_report,
)


def _default_output_dir(source_path: Path) -> Path:
    """Place QC outputs under the repo-level results directory."""
    return REPO_ROOT / "results" / f"{source_path.stem}_qc"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        "-d",
        default="train",
        help="Dataset key to load when no path is provided (defaults to 'train').",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=Path,
        help="Explicit AnnData file to QC. Overrides --dataset if set.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing challenge data files. Overrides VCC_DATA_DIR if provided.",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        type=Path,
        help="Directory to store QC outputs (defaults to results/<stem>_qc in the repo).",
    )
    parser.add_argument(
        "--min-genes",
        type=int,
        default=None,
        help="Minimum genes per cell to keep (defaults to best practices).",
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=None,
        help="Maximum genes per cell to keep (defaults to best practices).",
    )
    parser.add_argument(
        "--min-counts",
        type=int,
        default=None,
        help="Minimum total counts per cell (defaults to best practices).",
    )
    parser.add_argument(
        "--max-counts",
        type=int,
        default=None,
        help="Maximum total counts per cell (optional, defaults to disabled).",
    )
    parser.add_argument(
        "--max-mt",
        type=float,
        default=None,
        help="Maximum mitochondrial percent to keep (defaults to best practices).",
    )
    parser.add_argument(
        "--min-cells-per-gene",
        type=int,
        default=None,
        help="Drop genes detected in fewer than this many cells (defaults to best practices).",
    )
    parser.add_argument(
        "--backed",
        action="store_true",
        help="Load AnnData in backed (read-only) mode before computing QC (will be materialized to memory for filtering).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    thresholds = QCThresholds(
        min_genes=args.min_genes if args.min_genes is not None else QCThresholds.min_genes,
        max_genes=args.max_genes if args.max_genes is not None else QCThresholds.max_genes,
        min_counts=args.min_counts if args.min_counts is not None else QCThresholds.min_counts,
        max_counts=args.max_counts if args.max_counts is not None else QCThresholds.max_counts,
        max_mt_percent=args.max_mt if args.max_mt is not None else QCThresholds.max_mt_percent,
        min_cells_per_gene=(
            args.min_cells_per_gene if args.min_cells_per_gene is not None else QCThresholds.min_cells_per_gene
        ),
    )

    try:
        if args.path is not None:
            adata = load_anndata(args.path, backed=args.backed)
            source_path = Path(args.path).resolve()
        else:
            adata = load_dataset(args.dataset, data_dir=args.data_dir, backed=args.backed)
            source_path = Path(adata.filename) if adata.filename is not None else Path(args.dataset)
    except (ConfigurationError, DatasetNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.backed:
        print("Loaded in backed mode; materializing into memory for QC/filtering.")
        adata = adata.to_memory()

    outdir = args.outdir or _default_output_dir(source_path)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Computing QC metrics for {adata.n_obs:,} cells × {adata.n_vars:,} genes.")
    annotate_qc_metrics(adata)

    # Capture raw snapshot for summary before mutating.
    adata_raw = adata.copy()

    genes_dropped = filter_genes(adata, min_cells=thresholds.min_cells_per_gene)
    print(f"Dropped {genes_dropped} genes detected in fewer than {thresholds.min_cells_per_gene} cells.")

    # Recompute QC metrics after gene filtering for accurate per-cell stats.
    annotate_qc_metrics(adata)

    mask = cell_filter_mask(adata, thresholds)
    kept = int(mask.sum())
    print(f"Filtering cells: keeping {kept:,} of {adata.n_obs:,}.")
    adata_filtered = adata[mask].copy()

    # Save filtered AnnData
    filtered_path = outdir / f"{source_path.stem}_qc_filtered.h5ad"
    adata_filtered.write(filtered_path)
    print(f"Filtered AnnData written to: {filtered_path}")

    summary = summarize_qc(
        adata_raw,
        adata_filtered,
        thresholds=thresholds,
        genes_dropped=genes_dropped,
    )
    summary_path = outdir / "qc_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"QC summary saved to: {summary_path}")

    figures = plot_basic_qc(adata_raw, outdir=outdir)
    if figures:
        print(f"QC figures saved: {', '.join(p.name for p in figures)}")

    report_path = write_markdown_report(outdir=outdir, summary=summary, figure_paths=figures, source=source_path)
    print(f"Markdown report written to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
