"""Contract tests for the two-amplitude emission (T84, `PoissonEmitter.emit_dual`).

What is pinned: the per-cell mean follows one profile and the column sums another, both to
the routine's own tolerance; every cell keeps its depth and every column its total; the
identity call (same profile on both channels) is the pin's own control; lambda 0 is refused;
`eval.loco` records the knob and stays bit-identical without it.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from sidechain.models.count_emitters import (
    ContextProfile,
    PoissonEmitter,
    dual_moment_counts,
    repair_bulk_totals,
)

G = 60


def _profile(rng, g=G, n_ctrl=300):
    frac = rng.gamma(0.7, 1.0, size=g)
    frac /= frac.sum()
    libs = rng.lognormal(np.log(4000), 0.45, size=n_ctrl).round()
    return ContextProfile(name="A", genes=np.array([f"g{i}" for i in range(g)]), fraction=frac,
                          libsizes=libs, n_cells=n_ctrl)


def _moments(M):
    X = M.toarray().astype(np.float64)
    depth = X.sum(axis=1)
    per_cell = (X / depth[:, None]).mean(axis=0)
    bulk = X.sum(axis=0) / X.sum()
    return X, depth, per_cell, bulk


def test_two_channels_follow_two_profiles_and_the_integers_hold(tmp_path):
    rng = np.random.default_rng(0)
    prof = _profile(rng)
    em = PoissonEmitter(prof, seed=3, lam=0.5)
    d_cell = rng.normal(0, 0.4, size=G)
    d_bulk = 1.25 * d_cell        # inside what a lambda 0.5 depth spread can carry (see below)
    M = em.emit_dual(400, d_cell, d_bulk)
    assert sp.isspmatrix_csr(M) and M.shape == (400, G)
    X, depth, per_cell, bulk = _moments(M)
    assert np.array_equal(X, np.round(X)) and X.min() >= 0
    p_cell, p_bulk = em._fraction(d_cell), em._fraction(d_bulk)
    # the two moments hold to a few thousandths (L1 over the genes), and they differ
    assert np.abs(per_cell - p_cell).sum() < 0.01
    assert np.abs(bulk - p_bulk).sum() < 0.01
    assert np.abs(p_bulk - p_cell).sum() > 0.05
    # depths are the template's draws, not the median: a real spread
    assert depth.std() / depth.mean() > 0.05


def test_the_identity_call_is_the_pin_and_keeps_the_template_depths(tmp_path):
    rng = np.random.default_rng(1)
    prof = _profile(rng)
    d = rng.normal(0, 0.4, size=G)
    em_a = PoissonEmitter(prof, seed=5, lam=0.5)
    ref = em_a.emit(300, d).toarray().astype(np.float64)
    em_b = PoissonEmitter(prof, seed=5, lam=0.5)
    M = em_b.emit_dual(300, d, d)
    X, depth, per_cell, bulk = _moments(M)
    # same RNG stream, so the template IS the reference emission: depths agree cell for cell
    assert np.array_equal(depth, ref.sum(axis=1))
    # and the column sums now sit exactly on their expectation (the pin)
    expect = ref.sum() * em_a._fraction(d)
    assert np.abs(X.sum(axis=0) - expect).max() <= 1.0
    assert np.abs(bulk - em_a._fraction(d)).sum() < 1e-3


def test_lambda_zero_is_refused_and_an_infeasible_bulk_is_refused():
    rng = np.random.default_rng(2)
    prof = _profile(rng)
    d = rng.normal(0, 0.4, size=G)
    with pytest.raises(ValueError, match="depth spread"):
        PoissonEmitter(prof, seed=0, lam=0.0).emit_dual(100, d, d)
    em = PoissonEmitter(prof, seed=0, lam=0.5)
    wild = d.copy()
    wild[:20] += 6.0          # a bulk profile far outside what the depth envelope can carry
    with pytest.raises(ValueError, match="depth envelope"):
        em.emit_dual(100, d, wild, max_projection=0.03)
    # a bulk profile that passes the per-gene envelope check but cannot be met JOINTLY (every
    # strong gene at its edge at once) fails the fit itself rather than returning wrong moments
    with pytest.raises(ValueError, match="moment fitting failed"):
        em.emit_dual(400, d, 2.0 * d, max_projection=1.0)
    # and with on_fail="fallback" the same call returns the single-amplitude template and counts it
    em2 = PoissonEmitter(prof, seed=0, lam=0.5)
    M = em2.emit_dual(400, d, 2.0 * d, max_projection=1.0, on_fail="fallback")
    assert M.shape == (400, G) and em2.dual_fallbacks == 1
    with pytest.raises(ValueError, match="on_fail"):
        em2.emit_dual(10, d, d, on_fail="ignore")


def test_repair_bulk_totals_moves_counts_within_columns_only():
    rng = np.random.default_rng(3)
    expected = rng.gamma(2.0, 3.0, size=(50, 30))
    integer = np.floor(expected).astype(np.int64)
    # scatter each row's fractional residual onto some columns so row depths are whole
    for i in range(50):
        r = int(round(expected[i].sum())) - int(integer[i].sum())
        if r > 0:
            integer[i, rng.choice(30, size=r, replace=False)] += 1
    rows_before = integer.sum(axis=1).copy()
    out = repair_bulk_totals(integer.copy(), expected)
    assert np.array_equal(out.sum(axis=1), rows_before)
    target = np.floor(expected.sum(axis=0)).astype(np.int64)
    extra = int(rows_before.sum() - target.sum())
    assert extra >= 0 and np.abs(out.sum(axis=0) - target).max() <= 1
    assert (out >= 0).all()


def test_dual_moment_counts_rejects_bad_inputs():
    rng = np.random.default_rng(4)
    template = rng.poisson(3.0, size=(20, 10)).astype(float) + 0.1
    p = np.full(10, 0.1)
    depths = np.full(20, 40)
    with pytest.raises(ValueError):
        dual_moment_counts(template, p * 2, p, depths=depths, seed=0)
    with pytest.raises(ValueError):
        dual_moment_counts(template, p, p, depths=depths.astype(float) + 0.5, seed=0)
    with pytest.raises(ValueError):
        dual_moment_counts(template[:, :5], p, p, depths=depths, seed=0)


def test_loco_records_alpha_bulk_and_stays_identical_without_it(monkeypatch, tmp_path):
    import anndata as ad
    import pandas as pd

    from sidechain.data.stream_pseudobulk import PseudobulkSums
    from sidechain.eval import loco

    rng = np.random.default_rng(6)
    genes = np.array([f"g{i}" for i in range(G)], dtype=object)
    basal = rng.uniform(100, 2000, size=G)
    labels_src = ["ctrl", "g0", "g1"]
    # effects the size of a real pooled delta (mean |log2FC| ~0.1-0.15), so alpha_bulk 1.6 sits
    # inside the lambda 0.5 depth envelope on every gene
    mean = np.stack([basal, basal * np.exp2(rng.normal(0, 0.15, G)), basal * np.exp2(rng.normal(0, 0.15, G))])
    n = np.full(3, 1000, dtype=np.int64)
    src = PseudobulkSums(labels=labels_src, genes=genes.copy(), count_sum=mean * n[:, None],
                         cpm_sum=mean * n[:, None], cpm_sq_sum=(mean**2 + mean) * n[:, None],
                         n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["t"])
    rows, labels = [], []
    for lab, k in (("non-targeting", 40), ("g0", 12), ("g1", 12)):
        for _ in range(k):
            rows.append(rng.poisson(basal / basal.sum() * rng.integers(3000, 6000))); labels.append(lab)
    real = ad.AnnData(X=sp.csr_matrix(np.asarray(rows, dtype=np.float32)),
                      obs=pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(len(rows))]),
                      var=pd.DataFrame(index=genes.astype(str)))
    real_path = tmp_path / "real.h5ad"
    real.write_h5ad(real_path)
    calls = []
    orig = loco.PoissonEmitter.emit_dual
    monkeypatch.setattr(loco.PoissonEmitter, "emit_dual",
                        lambda self, *a, **k: calls.append(1) or orig(self, *a, **k))
    one = loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "one.h5ad",
                                         pert_col="perturbation", control="non-targeting",
                                         shrinkage=False, var_floor="poisson", emit_lambda=0.5,
                                         alpha=1.35, min_libsize=0.0)
    assert calls == [] and one["alpha_bulk"] is None and one["dual_fallbacks"] is None
    two = loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "two.h5ad",
                                         pert_col="perturbation", control="non-targeting",
                                         shrinkage=False, var_floor="poisson", emit_lambda=0.5,
                                         alpha=1.35, alpha_bulk=1.6, min_libsize=0.0)
    assert calls == [1, 1] and two["alpha_bulk"] == 1.6 and two["dual_fallbacks"] == 0
    a = ad.read_h5ad(tmp_path / "one.h5ad"); b = ad.read_h5ad(tmp_path / "two.h5ad")
    assert a.shape == b.shape
    # the FIRST target's cells share the RNG stream up to the template, so their depths agree
    # cell for cell (the dual call then draws one extra seed, so later targets diverge); the
    # column sums do not agree -- the bulk moved
    first = a.obs["perturbation"].to_numpy() == a.obs["perturbation"].iloc[0]
    assert np.array_equal(np.asarray(a.X[first].sum(axis=1)).ravel(), np.asarray(b.X[first].sum(axis=1)).ravel())
    assert not np.allclose(np.asarray(a.X[first].sum(axis=0)).ravel(), np.asarray(b.X[first].sum(axis=0)).ravel())
    with pytest.raises(SystemExit, match="depth spread"):
        loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "bad.h5ad",
                                       pert_col="perturbation", control="non-targeting",
                                       dispersion="even", alpha_bulk=2.0, min_libsize=0.0)
