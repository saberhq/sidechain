"""Profile a challenge dataset: shape, labels, ID spaces, count statistics.

Reads in backed mode, so it touches metadata without pulling a 15 GB matrix into memory.

    uv run python -m sidechain.data.profile
    uv run python -m sidechain.data.profile --challenge-config challenges/vcc2026/config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sidechain.data.loaders import load_challenge_config
from sidechain.utils.paths import resolve_config


def profile_file(path: Path, pert_col: str, control_label: str) -> dict:
    """Metadata-only profile of one h5ad."""
    a = ad.read_h5ad(path, backed="r")
    labels = a.obs[pert_col].astype(str)
    counts = labels.value_counts()
    perts = counts.drop(control_label, errors="ignore")
    measured = set(map(str, a.var_names))

    out = {
        "file": path.name,
        "size_gb": round(path.stat().st_size / 1e9, 2),
        "n_cells": int(a.n_obs),
        "n_genes": int(a.n_vars),
        "obs_columns": list(a.obs.columns),
        "var_columns": list(a.var.columns),
        "n_perturbations": len(perts),
        "n_control_cells": int(counts.get(control_label, 0)),
        "control_fraction": round(float(counts.get(control_label, 0)) / a.n_obs, 4),
        "cells_per_pert": {
            "min": int(perts.min()),
            "p25": int(perts.quantile(0.25)),
            "median": int(perts.median()),
            "p75": int(perts.quantile(0.75)),
            "max": int(perts.max()),
        },
        "perts_with_min_cells": {
            str(t): int((perts >= t).sum()) for t in (10, 30, 50, 100)
        },
        "perturbed_genes_also_measured": f"{sum(p in measured for p in perts.index)}/{len(perts)}",
    }
    for col in a.obs.columns:
        if col != pert_col:
            out[f"n_unique_{col}"] = int(a.obs[col].nunique())
    return out


def profile_matrix(path: Path, n_cells: int = 3000) -> dict:
    """Count statistics from the first `n_cells` rows."""
    a = ad.read_h5ad(path, backed="r")
    X = a[: min(n_cells, a.n_obs)].to_memory().X
    dense = X.toarray() if sp.issparse(X) else np.asarray(X)
    lib = dense.sum(axis=1)
    return {
        "storage": type(X).__name__,
        "dtype": str(dense.dtype),
        "integral_counts": bool(np.allclose(dense, np.round(dense))),
        "pct_zeros": round(100 * float((dense == 0).mean()), 1),
        "value_max": float(dense.max()),
        "umi_per_cell": {
            "min": int(lib.min()),
            "median": int(np.median(lib)),
            "max": int(lib.max()),
        },
        "median_genes_detected": int(np.median((dense > 0).sum(axis=1))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--challenge-config", default="challenges/vcc2025/config.yaml")
    args = ap.parse_args()

    cfg = load_challenge_config(resolve_config(args.challenge_config))
    data_dir = Path(cfg["data_dir"]).expanduser()
    pert_col = cfg.get("pert_col", "target_gene")
    control = cfg.get("control_label", "non-targeting")

    print(f"# Dataset profile — {cfg.get('year')} · {data_dir}\n")

    files = [f for f in (cfg.get("source_file"), cfg.get("dev_file")) if f]
    profiles = []
    for fname in files:
        p = data_dir / fname
        if not p.exists():
            print(f"  (missing: {fname})")
            continue
        prof = profile_file(p, pert_col, control)
        profiles.append(prof)
        print(f"## {prof['file']}  ({prof['size_gb']} GB)")
        for k, v in prof.items():
            if k in ("file", "size_gb"):
                continue
            print(f"  {k:34s} {v}")
        print()

    main_file = data_dir / files[0] if files else None
    if main_file and main_file.exists():
        print("## count matrix")
        for k, v in profile_matrix(main_file).items():
            print(f"  {k:34s} {v}")
        print()

    # The training/validation overlap is the fact that defines the task.
    val_csv = data_dir / "pert_counts_Validation.csv"
    if val_csv.exists() and main_file and main_file.exists():
        a = ad.read_h5ad(main_file, backed="r")
        train = set(a.obs[pert_col].astype(str).unique()) - {control}
        val = set(pd.read_csv(val_csv)["target_gene"].astype(str))
        print("## task structure")
        print(f"  training perturbations             {len(train)}")
        print(f"  validation targets                 {len(val)}")
        print(f"  overlap                            {len(train & val)}")
        print(
            "  => generalization to UNSEEN perturbations"
            if not (train & val)
            else "  => validation overlaps training"
        )
        print()

    # gene_names.csv ships without a header row; the default read silently drops
    # the first gene and shifts every downstream index by one.
    gn = data_dir / "gene_names.csv"
    if gn.exists() and profiles:
        n_default = pd.read_csv(gn).shape[0]
        n_nohdr = pd.read_csv(gn, header=None).shape[0]
        print("## gene_names.csv")
        print(f"  rows with default read()           {n_default}")
        print(f"  rows with header=None              {n_nohdr}  <- correct")
        print(f"  n_genes in h5ad                    {profiles[0]['n_genes']}")


if __name__ == "__main__":
    main()
