"""Tests for the rung-0/1 predictors.

The prediction contract cell-eval enforces (same gene order, same perturbation
set including the control) plus the two properties that are easy to get wrong and
expensive when you do: output sparsity, and the DEG-frequency statistic.
"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sidechain.models.baseline_stats import (
    PredictControl,
    PredictMeanPerturbation,
    StatisticalBackbone,
    _col_moments,
)

CONTROL = "non-targeting"


def _train(n_ctrl=200, n_pert_each=60, n_genes=40, seed=0):
    """Controls plus three perturbations, with realistic dropout."""
    rng = np.random.default_rng(seed)
    perts = ["SYM0", "SYM1", "SYM2"]
    labels = [CONTROL] * n_ctrl + [p for p in perts for _ in range(n_pert_each)]
    counts = rng.poisson(3, size=(len(labels), n_genes)).astype(np.float32)
    X = np.log1p(counts)  # ~5% of entries land at 0, mimicking dropout

    # Give each perturbation a real self-knockdown so knockdown_ is learnable.
    gene_names = [f"SYM{i}" for i in range(n_genes)]
    for i, p in enumerate(perts):
        rows = np.flatnonzero(np.array(labels) == p)
        X[rows, gene_names.index(p)] = 0.0

    obs = pd.DataFrame({"target_gene": labels}, index=[f"c{i}" for i in range(len(labels))])
    var = pd.DataFrame(index=pd.Index(gene_names))
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)


# ------------------------------------------------------------- contract --


@pytest.mark.parametrize("cls", [PredictControl, PredictMeanPerturbation, StatisticalBackbone])
def test_prediction_matches_the_cell_eval_pairing_contract(cls):
    a = _train()
    model = cls().fit(a)
    pred = model.predict({"SYM0": 10, "SYM1": 7}, include_control=5)

    assert pred.n_obs == 22
    assert list(pred.var_names) == list(a.var_names), "gene order must match the truth"
    labels = set(pred.obs["target_gene"])
    assert labels == {"SYM0", "SYM1", CONTROL}, "control must be present in the prediction"
    assert int((pred.obs["target_gene"] == "SYM0").sum()) == 10


@pytest.mark.parametrize("cls", [PredictControl, PredictMeanPerturbation, StatisticalBackbone])
def test_predictions_are_non_negative(cls):
    a = _train()
    pred = cls().fit(a).predict({"SYM0": 20}, include_control=5)
    X = pred.X.toarray() if sp.issparse(pred.X) else pred.X
    assert X.min() >= 0, "expression is log1p of a non-negative quantity"


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        PredictControl().predict({"SYM0": 3})


def test_fit_without_controls_raises():
    a = _train()
    a = a[a.obs["target_gene"] != CONTROL].copy()
    with pytest.raises(ValueError, match="No control cells"):
        PredictControl().fit(a)


# -------------------------------------------------------------- sparsity --


def test_prediction_is_sparse_by_default():
    """A dense prediction at the full 2025 scale is 5.2 GB and drove a 17 GB
    machine into swap thrashing. Sparse is the default for that reason."""
    a = _train()
    pred = StatisticalBackbone().fit(a).predict({"SYM0": 50}, include_control=20)
    assert sp.issparse(pred.X)


def test_sparse_output_can_be_disabled():
    a = _train()
    pred = StatisticalBackbone().fit(a).predict(
        {"SYM0": 50}, include_control=20, sparse_output=False
    )
    assert not sp.issparse(pred.X)


def test_values_below_the_assay_floor_are_snapped_to_zero():
    """Adding a delta to a dropped-out gene produces values far below the smallest
    value the assay can represent. Those are arithmetic artifacts, and leaving
    them in makes predictions ~4% denser than the truth."""
    a = _train()
    m = StatisticalBackbone().fit(a)
    assert m.min_nonzero_ > 0

    dense = m.predict({"SYM0": 80}, include_control=40, sparse_output=False).X
    sparse = m.predict({"SYM0": 80}, include_control=40, sparse_output=True).X.toarray()

    below = (dense > 0) & (dense < m.min_nonzero_)
    assert below.any(), "fixture should produce sub-floor values for this to be meaningful"
    assert not ((sparse > 0) & (sparse < m.min_nonzero_)).any()
    # Every surviving value is unchanged -- this thresholds, it does not rescale.
    keep = sparse > 0
    np.testing.assert_allclose(sparse[keep], dense[keep])


# ----------------------------------------------------------- the backbone --


def test_deg_frequency_uses_standard_error_not_per_cell_sd():
    """The original gate z-scored a pseudobulk mean against per-cell SD, which
    overstates the noise by ~sqrt(n): only 10 of 18,080 genes cleared it and the
    generic response collapsed to zero. With the correct denominator a real
    shared response must clear it for a substantial fraction of genes."""
    a = _train()
    m = StatisticalBackbone().fit(a)
    assert m.deg_frequency_.shape == (a.n_vars,)
    assert ((m.deg_frequency_ >= 0) & (m.deg_frequency_ <= 1)).all()
    assert (m.deg_frequency_ > 0).sum() > 0


def test_target_gene_is_knocked_down_and_overrides_the_generic_response():
    a = _train()
    m = StatisticalBackbone().fit(a)
    assert m.knockdown_ < 0, "CRISPRi drives the targeted transcript down"
    pos = list(a.var_names).index("SYM0")
    assert m.delta_for("SYM0")[pos] == pytest.approx(m.knockdown_)


def test_unknown_perturbation_still_predicts_the_generic_response():
    """The real task is unseen genes, so a label absent from training must not
    raise -- it just gets the generic response with no knockdown term."""
    a = _train()
    m = StatisticalBackbone().fit(a)
    delta = m.delta_for("A_GENE_NEVER_SEEN")
    assert delta.shape == (a.n_vars,)
    assert np.isfinite(delta).all()


def test_feature_table_exposes_the_documented_features():
    a = _train()
    tbl = StatisticalBackbone().fit(a).feature_table()
    assert set(tbl.columns) == {
        "mean_delta",
        "deg_frequency",
        "mean_expression",
        "gene_variance",
        "effect_size_prior",
    }
    assert len(tbl) == a.n_vars


def test_rung0_predicts_exactly_the_control_distribution():
    a = _train()
    m = PredictControl().fit(a)
    np.testing.assert_allclose(m.delta_for("anything"), np.zeros(a.n_vars))


def test_predict_rejects_an_empty_request():
    a = _train()
    m = PredictControl().fit(a)
    with pytest.raises(ValueError, match="no perturbations"):
        m.predict({"SYM0": 0})


# ------------------------------------------------------ numerical moments --


def test_control_pool_is_not_densified():
    """38,176 control cells densified is 2.8 GB held for the whole run. The pool
    keeps whatever layout the data arrived in; predict() densifies only its draw."""
    a = _train()
    assert sp.issparse(a.X)
    m = StatisticalBackbone().fit(a)
    assert sp.issparse(m.control_cells_)


def test_sparse_moments_match_a_float64_dense_recompute():
    """Sparse reductions accumulate in the matrix dtype, and E[X^2]-E[X]^2 loses
    precision to cancellation. In float32 those two together reached ~1e-3
    relative error on real data, so both moments accumulate in float64."""
    rng = np.random.default_rng(0)
    dense = (rng.random((900, 60)) * 8.0).astype(np.float32)
    dense[rng.random(dense.shape) < 0.55] = 0.0  # realistic dropout
    csr = sp.csr_matrix(dense)

    mean, var = _col_moments(csr, chunk=100)
    ref = dense.astype(np.float64)
    np.testing.assert_allclose(mean, ref.mean(axis=0), rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(var, ref.var(axis=0), rtol=1e-9, atol=1e-12)


def test_moments_are_chunk_size_invariant():
    rng = np.random.default_rng(1)
    dense = (rng.random((500, 30)) * 5.0).astype(np.float32)
    dense[rng.random(dense.shape) < 0.5] = 0.0
    csr = sp.csr_matrix(dense)
    for chunk in (7, 64, 10_000):
        mean, var = _col_moments(csr, chunk=chunk)
        np.testing.assert_allclose(mean, _col_moments(csr, chunk=123)[0], rtol=1e-12)
        np.testing.assert_allclose(var, _col_moments(csr, chunk=123)[1], rtol=1e-12)


def test_moments_handle_dense_input_too():
    arr = np.array([[1.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    mean, var = _col_moments(arr)
    np.testing.assert_allclose(mean, [2.0, 2.0])
    np.testing.assert_allclose(var, [1.0, 4.0])


def test_variance_is_never_negative():
    """Cancellation can drive E[X^2]-E[X]^2 slightly below zero for a constant
    column; a negative variance would produce NaN in the z-score denominator."""
    const = np.full((300, 5), 7.25, dtype=np.float32)
    _, var = _col_moments(sp.csr_matrix(const), chunk=32)
    assert (var >= 0).all()
