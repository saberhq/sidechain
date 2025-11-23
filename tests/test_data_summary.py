from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from vcc.data import (
    compare_gene_sets,
    load_gene_names,
    load_validation_perturbations,
    summarize,
    top_obs_counts,
)


def test_summarize_returns_basic_metadata(tmp_path):
    matrix = np.eye(3)
    obs = pd.DataFrame({"target_gene": ["A", "B", "C"]}, index=[f"cell_{i}" for i in range(3)])
    var = pd.DataFrame({"gene_id": [f"g_{i}" for i in range(3)]}, index=[f"gene_{i}" for i in range(3)])

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.layers["raw"] = matrix
    adata.uns["info"] = {"source": "synthetic"}

    summary = summarize(adata, source=tmp_path / "synthetic.h5ad")

    assert summary.n_obs == 3
    assert summary.n_vars == 3
    assert summary.obs_columns == ("target_gene",)
    assert summary.var_columns == ("gene_id",)
    assert summary.layers == ("raw",)
    assert summary.uns_keys == ("info",)


def test_load_gene_names_reads_single_column(tmp_path):
    data_dir = tmp_path
    (data_dir / "gene_names.csv").write_text("A\nB\nC\n", encoding="utf-8")

    series = load_gene_names(data_dir=data_dir)

    assert list(series) == ["A", "B", "C"]


def test_load_validation_perturbations_reads_csv(tmp_path):
    data_dir = tmp_path
    (data_dir / "pert_counts_Validation.csv").write_text(
        "target_gene,n_cells,median_umi_per_cell\nGENE1,10,100.0\n",
        encoding="utf-8",
    )

    df = load_validation_perturbations(data_dir=data_dir)

    assert list(df.columns) == ["target_gene", "n_cells", "median_umi_per_cell"]
    assert df.iloc[0]["target_gene"] == "GENE1"


def test_compare_gene_sets_reports_alignment():
    adata = ad.AnnData(X=np.eye(2))
    adata.var_names = ["A", "B"]
    genes = pd.Series(["A", "B"])

    result = compare_gene_sets(adata, genes)

    assert result["is_aligned"] is True
    assert result["n_adata"] == 2
    assert result["n_reference"] == 2
    assert result["n_overlap"] == 2
    assert result["missing_in_reference"] == ()
    assert result["missing_in_adata"] == ()


def test_top_obs_counts_returns_frequency():
    adata = ad.AnnData(X=np.eye(3))
    adata.obs["target_gene"] = ["G1", "G1", "G2"]

    counts = top_obs_counts(adata, "target_gene", limit=2)

    assert list(counts.index) == ["G1", "G2"]
    assert list(counts.values) == [2, 1]
