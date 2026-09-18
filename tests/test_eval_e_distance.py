"""Tests for `sidechain.eval.e_distance` (T94).

These pin two things: that our wrapper's fixed arguments (`sqeuclidean`, bias-corrected)
reproduce the paper's formula by hand, not just by trusting `scperturb`'s defaults; and that
`prep_for_edistance` follows the paper's literal drop-then-equalize recipe rather than
`scperturb.equal_subsampling`'s own `N_min=` shortcut, which was measured to undershoot it
(module docstring).
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from sidechain.eval.e_distance import e_distance, e_test, prep_for_edistance

CONTROL = "non-targeting"


def _manual_e_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """The Methods formula, independent of `scperturb`: E(X,Y) = 2*delta_XY - sigma_X - sigma_Y,
    squared Euclidean, sigma bias-corrected by N(N-1)."""
    n, m = len(X), len(Y)
    delta_xy = cdist(X, Y, metric="sqeuclidean").mean()
    sigma_x = cdist(X, X, metric="sqeuclidean").sum() / (n * (n - 1))
    sigma_y = cdist(Y, Y, metric="sqeuclidean").sum() / (m * (m - 1))
    return 2 * delta_xy - sigma_x - sigma_y


def _pca_adata(
    group_sizes: dict[str, int], n_dims: int = 5, seed: int = 0
) -> ad.AnnData:
    """A tiny AnnData with `X_pca` already set -- skips `prep_for_edistance` for tests that
    only exercise the statistic, not the preprocessing recipe."""
    rng = np.random.default_rng(seed)
    labels, chunks = [], []
    for label, (n, offset) in group_sizes.items():
        chunks.append(rng.normal(loc=offset, scale=1.0, size=(n, n_dims)))
        labels += [label] * n
    X = np.concatenate(chunks)
    obs = pd.DataFrame({"pert": labels})
    obs.index = [f"cell{i}" for i in range(len(labels))]
    out = ad.AnnData(X=np.zeros((len(labels), 1)), obs=obs)
    out.obsm["X_pca"] = X
    return out


# --------------------------------------------------- e_distance matches the formula --


def test_e_distance_matches_the_papers_formula_by_hand():
    adata = _pca_adata({CONTROL: (80, 0.0), "target_a": (60, 3.0)})
    result = e_distance(adata, "pert", control=CONTROL)

    X = adata.obsm["X_pca"][adata.obs["pert"] == "target_a"]
    Y = adata.obsm["X_pca"][adata.obs["pert"] == CONTROL]
    expected = _manual_e_distance(X, Y)

    assert result["target_a"] == pytest.approx(expected, rel=1e-9)


def test_e_distance_of_control_to_itself_is_near_zero():
    adata = _pca_adata({CONTROL: (100, 0.0)})
    result = e_distance(adata, "pert", control=CONTROL)
    assert result[CONTROL] == pytest.approx(0.0, abs=1e-8)


def test_e_distance_is_larger_for_a_further_separated_group():
    adata = _pca_adata({CONTROL: (80, 0.0), "near": (60, 0.5), "far": (60, 5.0)})
    result = e_distance(adata, "pert", control=CONTROL)
    assert result["far"] > result["near"]


# ------------------------------------------------------------------------- e_test --


def test_e_test_separated_group_is_significant_same_distribution_is_not():
    adata = _pca_adata({CONTROL: (100, 0.0), "null": (80, 0.0), "real": (80, 6.0)})
    result = e_test(adata, "pert", control=CONTROL, n_permutations=200)

    assert result.loc["null", "pvalue"] > 0.05
    assert result.loc["real", "pvalue"] < 0.05
    # Holm-Sidak only ever loosens a p-value.
    assert (result["pvalue_adj"] >= result["pvalue"]).all()


# ------------------------------------------------------------- prep_for_edistance --


def _raw_counts_adata(
    group_sizes: dict[str, int], n_genes: int = 80, seed: int = 0
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    labels = [label for label, n in group_sizes.items() for _ in range(n)]
    n_obs = len(labels)
    # Mean counts per gene high enough that every cell clears the 1,000-UMI floor.
    X = rng.poisson(lam=30.0, size=(n_obs, n_genes)).astype(np.float32)
    obs = pd.DataFrame({"pert": labels})
    obs.index = [f"cell{i}" for i in range(n_obs)]
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_prep_for_edistance_drops_thin_groups_and_equalizes_to_the_survivor_minimum():
    # 'ghost' is under the 50-cell floor and must be dropped entirely; the paper's own
    # pipeline deletes it before computing anything, rather than folding it into the count
    # that sets the equalized size (module docstring's measured scperturb discrepancy).
    adata = _raw_counts_adata({CONTROL: 200, "ghost": 10, "a": 90, "b": 120})

    out = prep_for_edistance(adata, "pert", seed=0)

    groups = set(out.obs["pert"].unique())
    assert "ghost" not in groups
    assert groups == {CONTROL, "a", "b"}

    counts = out.obs["pert"].value_counts()
    assert counts.nunique() == 1  # every surviving group equalized to the same size
    assert (
        counts.iloc[0] == 90
    )  # the smallest SURVIVING group, not scperturb's N_min=50

    assert "X_pca" in out.obsm
    assert out.obsm["X_pca"].shape[1] == 50


def test_prep_for_edistance_does_not_leak_the_global_numpy_rng():
    adata = _raw_counts_adata({CONTROL: 200, "a": 90, "b": 120})

    np.random.seed(12345)
    before = np.random.randn(5)
    np.random.seed(12345)
    prep_for_edistance(adata, "pert", seed=0)
    after = np.random.randn(5)

    assert np.array_equal(before, after)


def test_prep_for_edistance_is_reproducible_given_the_same_seed():
    adata = _raw_counts_adata({CONTROL: 200, "a": 90, "b": 120})

    out1 = prep_for_edistance(adata, "pert", seed=7)
    out2 = prep_for_edistance(adata, "pert", seed=7)

    assert list(out1.obs_names) == list(out2.obs_names)
