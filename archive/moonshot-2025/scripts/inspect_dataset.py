"""CLI helpers to validate that Virtual Cell Challenge datasets load correctly."""

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
from vcc.data import (
    DatasetNotFoundError,
    compare_gene_sets,
    load_anndata,
    load_dataset,
    load_gene_names,
    load_validation_perturbations,
    summarize,
    top_obs_counts,
)


def _format_list(values: tuple[str, ...], *, limit: int = 8) -> str:
    if not values:
        return "none"
    if len(values) <= limit:
        return ", ".join(values)
    head = ", ".join(values[:limit])
    return f"{head}, … (+{len(values) - limit} more)"


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
        help="Explicit AnnData file to inspect. Overrides --dataset if set.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing challenge data files. Overrides VCC_DATA_DIR if provided.",
    )
    parser.add_argument(
        "--show-obs-head",
        type=int,
        default=0,
        help="Show the first N rows of adata.obs (0 to skip).",
    )
    parser.add_argument(
        "--show-var-head",
        type=int,
        default=0,
        help="Show the first N rows of adata.var (0 to skip).",
    )
    parser.add_argument(
        "--count-by",
        type=str,
        default=None,
        help="obs column to tally value counts (top 10 by default).",
    )
    parser.add_argument(
        "--count-limit",
        type=int,
        default=10,
        help="How many value-count entries to display when using --count-by.",
    )
    parser.add_argument(
        "--check-genes",
        action="store_true",
        help="Compare adata.var_names to gene_names.csv in the data directory.",
    )
    parser.add_argument(
        "--show-validation-metadata",
        action="store_true",
        help="Print the header and first rows of pert_counts_Validation.csv.",
    )
    parser.add_argument(
        "--backed",
        action="store_true",
        help="Load AnnData in backed (read-only) mode to reduce memory usage.",
    )
    return parser.parse_args(argv)


def dataset_info(
    *,
    dataset: str,
    path: Optional[Path],
    data_dir: Optional[Path],
    show_obs_head: int,
    show_var_head: int,
    count_by: Optional[str],
    count_limit: int,
    check_genes: bool,
    show_validation_metadata: bool,
    backed: bool,
) -> int:
    """Load an AnnData file and print high-level metadata."""
    try:
        if path is not None:
            adata = load_anndata(path, backed=backed)
            summary = summarize(adata, source=path)
        else:
            adata = load_dataset(dataset, data_dir=data_dir, backed=backed)
            summary = summarize(adata)
    except (ConfigurationError, DatasetNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Path         : {summary.path}")
    print(f"Shape        : {summary.n_obs:,} cells × {summary.n_vars:,} genes")
    print(f"obs columns  : {_format_list(summary.obs_columns)}")
    print(f"var columns  : {_format_list(summary.var_columns)}")
    print(f"layers       : {_format_list(summary.layers)}")
    print(f"uns keys     : {_format_list(summary.uns_keys)}")

    if show_obs_head > 0:
        print(f"\nobs head (first {show_obs_head} rows):")
        print(adata.obs.head(show_obs_head))

    if show_var_head > 0:
        print(f"\nvar head (first {show_var_head} rows):")
        print(adata.var.head(show_var_head))

    if count_by:
        try:
            counts = top_obs_counts(adata, count_by, limit=count_limit)
            print(f"\nTop {count_limit} value counts for obs['{count_by}']:")
            print(counts)
        except KeyError as exc:
            print(f"\nCould not compute counts: {exc}", file=sys.stderr)

    if check_genes:
        try:
            genes = load_gene_names(data_dir=data_dir)
            cmp = compare_gene_sets(adata, genes)
            print("\nGene set comparison against gene_names.csv:")
            print(
                f"- adata genes    : {cmp['n_adata']:,}\n"
                f"- reference genes: {cmp['n_reference']:,}\n"
                f"- overlap        : {cmp['n_overlap']:,}"
            )
            if not cmp["is_aligned"]:
                if cmp["missing_in_reference"]:
                    print(f"- in adata not in reference (sample): {list(cmp['missing_in_reference'])}")
                if cmp["missing_in_adata"]:
                    print(f"- in reference not in adata (sample): {list(cmp['missing_in_adata'])}")
            else:
                print("- Gene sets align.")
        except DatasetNotFoundError as exc:
            print(f"\nGene check skipped: {exc}", file=sys.stderr)

    if show_validation_metadata:
        try:
            df = load_validation_perturbations(data_dir=data_dir)
            print("\nValidation perturbation metadata (first 10 rows):")
            print(df.head(10))
        except DatasetNotFoundError as exc:
            print(f"\nValidation metadata not found: {exc}", file=sys.stderr)

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    return dataset_info(
        dataset=args.dataset,
        path=args.path,
        data_dir=args.data_dir,
        show_obs_head=args.show_obs_head,
        show_var_head=args.show_var_head,
        count_by=args.count_by,
        count_limit=args.count_limit,
        check_genes=args.check_genes,
        show_validation_metadata=args.show_validation_metadata,
        backed=args.backed,
    )


if __name__ == "__main__":
    sys.exit(main())
