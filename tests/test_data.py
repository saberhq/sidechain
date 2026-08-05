"""Tests for the data layer: gene-ID resolution, holdout carving, pseudobulk.

The headline case is `gene_index`. Its old failure was silent: the 2025 data ships
HGNC symbols in var_names while the prior registry keys on Ensembl IDs, so falling
back to var_names produced an index matching NO prior -- every source returned
zero edges and nothing errored.
"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sidechain.data import loaders
from sidechain.data.pseudobulk import delta_vs_control, pseudobulk


def _adata(n_cells=120, n_genes=6, with_ensembl=True, seed=0):
    """A miniature of the real 2025 layout: symbols in var_names, Ensembl in var."""
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i}" for i in range(n_genes)]
    var = pd.DataFrame(index=pd.Index(symbols, name=None))
    if with_ensembl:
        var["gene_id"] = [f"ENSG{i:011d}" for i in range(n_genes)]

    labels = ["non-targeting"] * (n_cells // 2)
    for i in range(n_cells - len(labels)):
        labels.append(f"SYM{i % 3}")
    X = sp.csr_matrix(rng.poisson(5, size=(len(labels), n_genes)).astype(np.float32))
    obs = pd.DataFrame({"target_gene": labels}, index=[f"c{i}" for i in range(len(labels))])
    return ad.AnnData(X=X, obs=obs, var=var)


# ------------------------------------------------------------- gene_index --


def test_gene_index_resolves_ensembl_from_var_column():
    a = _adata()
    gi = loaders.gene_index(a, "ensembl_gene_id")
    assert len(gi) == a.n_vars
    assert "ENSG00000000000" in gi
    assert gi["ENSG00000000000"] == 0


def test_gene_index_raises_when_ensembl_absent_instead_of_falling_back():
    """The silent-failure guard: symbols only, Ensembl requested."""
    a = _adata(with_ensembl=False)
    with pytest.raises(loaders.GeneIDSpaceError, match="No var column carries"):
        loaders.gene_index(a, "ensembl_gene_id")


def test_gene_index_fallback_is_opt_in_and_explicit():
    a = _adata(with_ensembl=False)
    gi = loaders.gene_index(a, "ensembl_gene_id", allow_symbol_fallback=True)
    assert "SYM0" in gi  # symbols, not Ensembl -- caller asked for this


def test_gene_index_rejects_a_column_whose_values_are_the_wrong_shape():
    """A column literally named gene_id but holding symbols must not satisfy a
    request for Ensembl IDs -- name alone is not evidence."""
    a = _adata(with_ensembl=False)
    a.var["gene_id"] = [f"SYM{i}" for i in range(a.n_vars)]
    with pytest.raises(loaders.GeneIDSpaceError):
        loaders.gene_index(a, "ensembl_gene_id")


def test_gene_index_resolves_symbols():
    gi = loaders.gene_index(_adata(), "hgnc_symbol")
    assert "SYM0" in gi


# --------------------------------------------------------------- coverage --


def test_coverage_and_assert_coverage():
    gi = {"a": 0, "b": 1}
    assert loaders.coverage(gi, ["a", "b"]) == 1.0
    assert loaders.coverage(gi, ["a", "zzz"]) == 0.5
    with pytest.raises(loaders.GeneIDSpaceError, match="ID-space mismatch"):
        loaders.assert_coverage(gi, ["x", "y"], source="fake_prior")
    # A partial map warns but does not raise.
    assert loaders.assert_coverage(gi, ["a", "zzz"], source="fake_prior") == 0.5


# ----------------------------------------------------------------- splits --


def test_carve_holdout_is_deterministic_and_excludes_control():
    a = _adata()
    t1, h1 = loaders.carve_holdout(a, 1, seed=7, min_cells=1)
    t2, h2 = loaders.carve_holdout(a, 1, seed=7, min_cells=1)
    assert h1 == h2 and t1 == t2
    assert "non-targeting" not in h1 and "non-targeting" not in t1
    assert not set(t1) & set(h1)


def test_carve_holdout_skips_perturbations_too_small_to_score():
    a = _adata()
    _, holdout = loaders.carve_holdout(a, 3, seed=0, min_cells=10_000)
    assert holdout == [], "no perturbation clears the threshold, so none is held out"


# ------------------------------------------------------------- pseudobulk --


def test_pseudobulk_collapses_to_one_row_per_group():
    a = _adata()
    pb = pseudobulk(a, ["target_gene"], aggregate="mean", min_cells=1)
    assert pb.n_obs == a.obs["target_gene"].nunique()
    assert pb.n_vars == a.n_vars
    assert "n_cells" in pb.obs.columns


def test_pseudobulk_min_cells_drops_small_groups():
    a = _adata()
    counts = a.obs["target_gene"].value_counts()
    threshold = int(counts.min()) + 1
    pb = pseudobulk(a, ["target_gene"], min_cells=threshold)
    assert pb.n_obs < a.obs["target_gene"].nunique()


def test_pseudobulk_bootstrap_emits_replicates_for_des_dispersion():
    a = _adata()
    pb = pseudobulk(a, ["target_gene"], min_cells=1, n_bootstrap=4, seed=1)
    assert pb.n_obs == 4 * a.obs["target_gene"].nunique()
    assert set(pb.obs["replicate"].astype(int)) == {0, 1, 2, 3}
    # Replicates must actually differ, or DES has no dispersion to score.
    first = pb[pb.obs["target_gene"] == "non-targeting"].X
    assert not np.allclose(first[0], first[1])


def test_pseudobulk_rejects_unknown_aggregate_and_missing_column():
    a = _adata()
    with pytest.raises(ValueError, match="Unknown aggregate"):
        pseudobulk(a, ["target_gene"], aggregate="geometric")
    with pytest.raises(KeyError, match="group_by columns"):
        pseudobulk(a, ["no_such_column"])


def test_delta_vs_control_excludes_control_and_matches_gene_axis():
    a = _adata()
    pb = pseudobulk(a, ["target_gene"], aggregate="mean", min_cells=1)
    d = delta_vs_control(pb)
    assert "non-targeting" not in d.index
    assert list(d.columns) == list(a.var_names)
    assert d.shape[0] == a.obs["target_gene"].nunique() - 1


def test_delta_vs_control_raises_when_control_label_absent():
    a = _adata()
    pb = pseudobulk(a, ["target_gene"], aggregate="mean", min_cells=1)
    with pytest.raises(ValueError, match="No control rows"):
        delta_vs_control(pb, control_label="not-a-label")


# ------------------------------------------------------------ normalizing --


def test_is_discrete_and_normalize_counts():
    a = _adata()
    assert loaders.is_discrete(a)
    n = loaders.normalize_counts(a)
    assert not loaders.is_discrete(n)
    assert n.X.max() < a.X.max()


def test_normalize_counts_default_leaves_the_original_untouched():
    a = _adata()
    before = a.X.max()
    loaders.normalize_counts(a)
    assert a.X.max() == before, "default must copy, not mutate the caller's AnnData"


def test_normalize_counts_inplace_mutates_and_returns_the_same_object():
    """The full 2025 file is ~3 GB; the defensive copy doubles peak memory and is
    the slowest step in the pipeline. inplace is how the dry run avoids it."""
    a = _adata()
    out = loaders.normalize_counts(a, inplace=True)
    assert out is a
    assert not loaders.is_discrete(a)
