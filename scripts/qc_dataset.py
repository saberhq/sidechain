"""Lightweight QC script using scanpy/scverse conventions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vcc.config import ConfigurationError
from vcc.data import DatasetNotFoundError, load_anndata, load_dataset


def _mito_mask(var_names) -> list[bool]:
    """Detect mitochondrial genes (assumes var_names are gene symbols)."""
    return [str(g).upper().startswith("MT-") for g in var_names]


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
        "--min-genes",
        type=int,
        default=200,
        help="Minimum genes per cell to keep (0 to disable).",
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=6000,
        help="Maximum genes per cell to keep (0 to disable).",
    )
    parser.add_argument(
        "--max-mt",
        type=float,
        default=20.0,
        help="Maximum mitochondrial percent (pct_counts_mt) to keep (0 to disable).",
    )
    parser.add_argument(
        "--min-counts",
        type=int,
        default=0,
        help="Minimum total counts per cell to keep (0 to disable).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output .h5ad path. Defaults to <data_dir>/<stem>_filtered.h5ad.",
    )
    parser.add_argument(
        "--backed",
        action="store_true",
        help="Load AnnData in backed (read-only) mode before computing QC (will be materialized to memory for filtering).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

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
    else:
        print("Loaded in-memory; computing QC metrics and applying basic filters.")

    adata.var["mt"] = _mito_mask(adata.var_names)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    print(
        f"Cells: {adata.n_obs:,}, Genes: {adata.n_vars:,}, "
        f"median genes/cell: {adata.obs['n_genes_by_counts'].median():.0f}, "
        f"median counts/cell: {adata.obs['total_counts'].median():.0f}"
    )
    if "pct_counts_mt" in adata.obs:
        print(f"pct_counts_mt median: {adata.obs['pct_counts_mt'].median():.2f}")
    else:
        print("pct_counts_mt median: n/a")

    to_keep: pd.Series = pd.Series(True, index=adata.obs_names)
    if args.min_genes > 0:
        to_keep &= adata.obs["n_genes_by_counts"] >= args.min_genes
    if args.max_genes > 0:
        to_keep &= adata.obs["n_genes_by_counts"] <= args.max_genes
    if args.min_counts > 0:
        to_keep &= adata.obs["total_counts"] >= args.min_counts
    if args.max_mt > 0 and "pct_counts_mt" in adata.obs:
        to_keep &= adata.obs["pct_counts_mt"] <= args.max_mt

    kept = int(to_keep.sum())
    print(f"Filtering: keeping {kept:,} of {adata.n_obs:,} cells.")
    adata = adata[to_keep].copy()

    if args.output:
        out_path = Path(args.output)
    else:
        stem = source_path.stem.replace(".h5ad", "")
        out_path = source_path.with_name(f"{stem}_filtered.h5ad")

    adata.write(out_path)
    print(f"Filtered AnnData written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
