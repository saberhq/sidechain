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


# ── T84 round 2: the pooled anchor (private research/ideas/board-methods-survey.md D9) ───────


def _coupled_controls(tmp_path, rng, n=400, g=G):
    """Control cells whose composition tracks depth, as cycling cells do: gene 0 is enriched in
    deep cells, so the pooled profile holds more of it than the mean of per-cell CPM does."""
    import anndata as ad
    import pandas as pd

    base = rng.gamma(0.7, 1.0, size=g) + 0.05
    base /= base.sum()
    depth = rng.lognormal(np.log(4000), 0.45, size=n)
    comp = np.tile(base, (n, 1))
    comp[:, 0] *= (depth / np.median(depth)) ** 0.5
    comp /= comp.sum(axis=1, keepdims=True)
    X = rng.poisson(comp * depth[:, None]).astype(np.float32)
    a = ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                   var=pd.DataFrame(index=[f"g{j}" for j in range(g)]))
    path = tmp_path / "coupled_ctrl.h5ad"
    a.write_h5ad(path)
    return path, X


def test_from_controls_carries_the_pooled_profile_beside_the_mean_cpm_one(tmp_path):
    rng = np.random.default_rng(11)
    path, X = _coupled_controls(tmp_path, rng)
    prof = ContextProfile.from_controls(path, "A")
    pooled = X.sum(axis=0) / X.sum()
    assert np.allclose(prof.bulk_fraction, pooled) and np.isclose(prof.bulk_fraction.sum(), 1)
    # depth-coupled gene 0 is heavier in the pooled profile than in the mean of per-cell CPM
    assert prof.bulk_fraction[0] > prof.fraction[0] * 1.02
    # the floor applies to both profiles alike
    kept = ContextProfile.from_controls(path, "A", min_libsize=float(np.median(X.sum(axis=1))))
    deep = X[X.sum(axis=1) > np.median(X.sum(axis=1))]
    assert np.allclose(kept.bulk_fraction, deep.sum(axis=0) / deep.sum())


def test_the_pooled_anchor_moves_the_pseudobulk_and_nothing_else():
    rng = np.random.default_rng(12)
    prof = _profile(rng)
    tilt = np.exp(rng.uniform(-0.08, 0.08, size=G))   # inside a lambda 0.5 depth envelope
    prof.bulk_fraction = prof.fraction * tilt / (prof.fraction * tilt).sum()
    d = rng.normal(0, 0.3, size=G)
    em = PoissonEmitter(prof, seed=4, lam=0.5, bulk_anchor="pooled")
    X, _depth, per_cell, bulk = _moments(em.emit_dual(400, d, d))
    shifted = lambda f: f * np.exp2(d) / (f * np.exp2(d)).sum()
    assert np.abs(per_cell - shifted(prof.fraction)).sum() < 0.01      # per-cell channel unmoved
    assert np.abs(bulk - shifted(prof.bulk_fraction)).sum() < 0.01     # bulk on the pooled profile
    assert np.abs(bulk - shifted(prof.fraction)).sum() > 0.02          # and it did move
    assert np.array_equal(X, np.round(X))


def test_the_default_anchor_is_bit_identical_and_bad_anchors_are_refused():
    rng = np.random.default_rng(13)
    prof = _profile(rng)
    prof.bulk_fraction = prof.fraction.copy()
    d = rng.normal(0, 0.3, size=G)
    a = PoissonEmitter(prof, seed=5, lam=0.5).emit_dual(200, d, 1.2 * d)
    b = PoissonEmitter(prof, seed=5, lam=0.5, bulk_anchor="mean_cpm").emit_dual(200, d, 1.2 * d)
    assert (a != b).nnz == 0
    with pytest.raises(ValueError, match="bulk_anchor must be"):
        PoissonEmitter(prof, lam=0.5, bulk_anchor="median")
    prof.bulk_fraction = None
    with pytest.raises(ValueError, match="needs a profile with bulk_fraction"):
        PoissonEmitter(prof, lam=0.5, bulk_anchor="pooled")


def test_loco_routes_every_target_through_the_dual_emitter_under_the_pooled_anchor(monkeypatch, tmp_path):
    import anndata as ad
    import pandas as pd

    from sidechain.data.stream_pseudobulk import PseudobulkSums
    from sidechain.eval import loco

    rng = np.random.default_rng(14)
    genes = np.array([f"g{i}" for i in range(G)], dtype=object)
    basal = rng.uniform(100, 2000, size=G)
    mean = np.stack([basal, basal * np.exp2(rng.normal(0, 0.15, G))])
    n = np.full(2, 1000, dtype=np.int64)
    src = PseudobulkSums(labels=["ctrl", "g0"], genes=genes.copy(), count_sum=mean * n[:, None],
                         cpm_sum=mean * n[:, None], cpm_sq_sum=(mean**2 + mean) * n[:, None],
                         n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["t"])
    rows, labels = [], []
    # g0 is covered by the source; g1 is not, and must still get the pooled bulk
    for lab, k in (("non-targeting", 60), ("g0", 12), ("g1", 12)):
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
                        lambda self, *a, **k: calls.append(a[1] is None) or orig(self, *a, **k))
    kw = dict(pert_col="perturbation", control="non-targeting", shrinkage=False,
              var_floor="poisson", emit_lambda=0.5, alpha=1.35, min_libsize=0.0)
    base = loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "base.h5ad", **kw)
    assert calls == [] and base["bulk_anchor"] == "mean_cpm" and base["dual_fallbacks"] is None
    got = loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "pooled.h5ad",
                                         bulk_anchor="pooled", **kw)
    assert calls == [False, True]          # the covered target, then the uncovered one
    assert got["bulk_anchor"] == "pooled" and got["alpha_bulk"] is None and got["dual_fallbacks"] == 0
    with pytest.raises(SystemExit, match="depth spread"):
        loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "bad.h5ad",
                                       pert_col="perturbation", control="non-targeting",
                                       dispersion="even", bulk_anchor="pooled", min_libsize=0.0)


def _pooled_profile(rng):
    prof = _profile(rng)
    tilt = np.exp(rng.uniform(-0.08, 0.08, size=G))   # inside a lambda 0.5 depth envelope
    prof.bulk_fraction = prof.fraction * tilt / (prof.fraction * tilt).sum()
    return prof


def test_the_anchor_fallback_keeps_the_pooled_profile_and_the_template_one_loses_it():
    """T84 sweep, 2026-10-01: a target whose two amplitudes cannot both be met used to be emitted
    from the one-amplitude template, which is built on the mean-per-cell-CPM profile -- so under
    the pooled anchor it lost the anchor as well. `on_fail="anchor"` keeps the summed profile on
    the anchor at the per-cell amplitude; `"fallback"` is unchanged, bit for bit."""
    rng = np.random.default_rng(2)
    prof = _pooled_profile(rng)
    d = rng.normal(0, 0.4, size=G)
    shifted = lambda f: f * np.exp2(d) / (f * np.exp2(d)).sum()
    kw = dict(max_projection=1.0)                      # 2.0 * d passes the envelope and fails the fit
    old = PoissonEmitter(prof, seed=0, lam=0.5, bulk_anchor="pooled")
    A = old.emit_dual(400, d, 2.0 * d, on_fail="fallback", **kw)
    _X, _depth, per_cell, bulk = _moments(A)
    assert old.dual_fallbacks == 1 and not hasattr(old, "dual_fallbacks_anchor")
    assert (old.last_dual, old.last_dual_reason) == ("template", "fit")
    assert np.abs(bulk - shifted(prof.bulk_fraction)).sum() > 0.02     # off the pooled anchor
    assert (A != PoissonEmitter(prof, seed=0, lam=0.5).emit(400, d)).nnz == 0   # the template, exactly
    new = PoissonEmitter(prof, seed=0, lam=0.5, bulk_anchor="pooled")
    B = new.emit_dual(400, d, 2.0 * d, on_fail="anchor", **kw)
    X, depth, per_cell, bulk = _moments(B)
    assert new.dual_fallbacks == 1 and new.dual_fallbacks_anchor == 1   # one amplitude, anchor kept
    assert (new.last_dual, new.last_dual_reason) == ("anchor", "fit")
    assert np.abs(bulk - shifted(prof.bulk_fraction)).sum() < 0.01     # summed profile on the anchor
    assert np.abs(per_cell - shifted(prof.fraction)).sum() < 0.01      # per-cell channel where it was
    assert np.array_equal(X, np.round(X)) and np.array_equal(depth, A.toarray().sum(axis=1))
    # the retry draws nothing from the emitter's stream: the next target is the same cells either way
    assert (old.emit_dual(200, 0.5 * d, 0.5 * d) != new.emit_dual(200, 0.5 * d, 0.5 * d)).nnz == 0
    assert (new.last_dual, new.last_dual_reason) == ("dual", None)


def test_the_anchor_fallback_ends_on_the_template_when_one_amplitude_fails_too():
    rng = np.random.default_rng(3)
    prof = _profile(rng)
    wild = np.exp(rng.uniform(-3, 3, size=G))          # an anchor no depth spread can carry
    prof.bulk_fraction = prof.fraction * wild / (prof.fraction * wild).sum()
    d = rng.normal(0, 0.4, size=G)
    em = PoissonEmitter(prof, seed=0, lam=0.5, bulk_anchor="pooled")
    # both rungs are refused before the fit, and the template is returned and counted once
    M = em.emit_dual(300, d, 2.0 * d, on_fail="anchor")
    assert em.dual_fallbacks == 1 and not hasattr(em, "dual_fallbacks_anchor")
    assert (em.last_dual, em.last_dual_reason) == ("template", "envelope")
    assert (M != PoissonEmitter(prof, seed=0, lam=0.5).emit(300, d)).nnz == 0
    # a request that already IS the one-amplitude rung is not fitted twice
    calls = []
    import sidechain.models.count_emitters as ce
    orig = ce.dual_moment_counts
    try:
        ce.dual_moment_counts = lambda *a, **k: calls.append(1) or orig(*a, **k)
        em.emit_dual(300, d, d, on_fail="anchor")
    finally:
        ce.dual_moment_counts = orig
    assert calls == [1] and em.last_dual == "template" and em.dual_fallbacks == 2


def test_loco_passes_the_fallback_rung_and_names_the_targets_that_fell_back(monkeypatch, tmp_path):
    import anndata as ad
    import pandas as pd

    from sidechain.data.stream_pseudobulk import PseudobulkSums
    from sidechain.eval import loco

    rng = np.random.default_rng(15)
    genes = np.array([f"g{i}" for i in range(G)], dtype=object)
    basal = rng.uniform(100, 2000, size=G)
    mean = np.stack([basal, basal * np.exp2(rng.normal(0, 0.15, G))])
    n = np.full(2, 1000, dtype=np.int64)
    src = PseudobulkSums(labels=["ctrl", "g0"], genes=genes.copy(), count_sum=mean * n[:, None],
                         cpm_sum=mean * n[:, None], cpm_sq_sum=(mean**2 + mean) * n[:, None],
                         n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["t"])
    rows, labels = [], []
    for lab, k in (("non-targeting", 60), ("g0", 12)):
        for _ in range(k):
            rows.append(rng.poisson(basal / basal.sum() * rng.integers(3000, 6000))); labels.append(lab)
    real = ad.AnnData(X=sp.csr_matrix(np.asarray(rows, dtype=np.float32)),
                      obs=pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(len(rows))]),
                      var=pd.DataFrame(index=genes.astype(str)))
    real_path = tmp_path / "real.h5ad"
    real.write_h5ad(real_path)
    import sidechain.models.count_emitters as ce
    seen, fits = [], []
    orig, orig_fit = loco.PoissonEmitter.emit_dual, ce.dual_moment_counts

    def spy(self, *a, **k):
        seen.append(k["on_fail"])
        fits.clear()
        return orig(self, *a, **k)

    def refuse_the_pair(*a, **k):     # the requested pair fails by fiat, so the rung is exercised
        # (max_projection=0.0 used to do this: it needed the projection error to sit above zero by
        # rounding noise, and on some CI runners it is exactly zero)
        fits.append(1)
        if len(fits) == 1:
            raise ValueError("requested bulk profile lies 1.0000 (L1) outside the depth envelope, "
                             "above max_projection=0.03")
        return orig_fit(*a, **k)
    kw = dict(pert_col="perturbation", control="non-targeting", shrinkage=False, var_floor="poisson",
              emit_lambda=0.5, alpha=1.35, alpha_bulk=1.6, bulk_anchor="pooled", min_libsize=0.0)
    monkeypatch.setattr(loco.PoissonEmitter, "emit_dual", spy)
    monkeypatch.setattr(ce, "dual_moment_counts", refuse_the_pair)
    old = loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "old.h5ad", **kw)
    assert seen == ["fallback"] and old["dual_fallback"] == "template" and old["dual_fallbacks"] == 1
    assert old["dual_fallback_targets"] == {"anchor": [], "template": [["g0", "envelope"]]}
    seen.clear()
    new = loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "new.h5ad",
                                         dual_fallback="anchor", **kw)
    assert seen == ["anchor"] and new["dual_fallback"] == "anchor" and new["dual_fallbacks"] == 1
    # the requested pair is refused, one amplitude on the anchor is met: named under that rung
    assert new["dual_fallback_targets"] == {"anchor": [["g0", "envelope"]], "template": []}
    with pytest.raises(SystemExit, match="dual_fallback"):
        loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "bad.h5ad",
                                       dual_fallback="pooled", **kw)
    one = loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "one.h5ad",
                                         pert_col="perturbation", control="non-targeting",
                                         shrinkage=False, emit_lambda=0.5, alpha=1.35, min_libsize=0.0)
    assert one["dual_fallback"] is None and one["dual_fallback_targets"] is None
    with pytest.raises(SystemExit, match="two-channel"):           # one channel: nothing to fall back from
        loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "inert.h5ad",
                                       pert_col="perturbation", control="non-targeting", shrinkage=False,
                                       emit_lambda=0.5, alpha=1.35, min_libsize=0.0, dual_fallback="anchor")


# ── T85: the per-gene scatter dial (private research/ideas/reach-call-set-emitter.md) ─────────


def _cpm(M):
    X = M.toarray().astype(np.float64)
    return X, X / X.sum(axis=1, keepdims=True)


def test_the_per_gene_dial_is_off_by_default_and_all_ones_is_the_same_call():
    rng = np.random.default_rng(11)
    prof = _profile(rng)
    d = rng.normal(0, 0.3, size=G)
    plain = PoissonEmitter(prof, seed=4, lam=0.5)
    ones = PoissonEmitter(prof, seed=4, lam=0.5)
    a = plain.emit_dual(300, d, 1.2 * d).toarray()
    b = ones.emit_dual(300, d, 1.2 * d, scatter=np.ones(G)).toarray()
    assert np.array_equal(a, b)
    assert plain.last_sharpened is None and ones.last_sharpened == 0
    # and a plain emit resets the record
    plain.emit(10, d)
    assert plain.last_sharpened is None


def test_a_sharpened_gene_keeps_both_moments_and_every_depth_and_loses_its_scatter():
    rng = np.random.default_rng(12)
    prof = _profile(rng)
    d = rng.normal(0, 0.3, size=G)
    head = np.zeros(G, dtype=bool)
    head[np.argsort(-prof.fraction)[:20:2]] = True      # ten well-expressed genes
    scatter = np.where(head, 0.0, 1.0)
    base_em, sharp_em = PoissonEmitter(prof, seed=9, lam=0.5), PoissonEmitter(prof, seed=9, lam=0.5)
    base, base_comp = _cpm(base_em.emit_dual(400, d, d))
    sharp, sharp_comp = _cpm(sharp_em.emit_dual(400, d, d, scatter=scatter))
    assert sharp_em.last_sharpened == 10 and sharp_em.last_dual == "dual"
    assert np.array_equal(sharp, np.round(sharp)) and sharp.min() >= 0
    # the cells are the same cells: every depth agrees, and so does (to the pin's rounding) every column total
    assert np.array_equal(base.sum(axis=1), sharp.sum(axis=1))
    assert np.abs(base.sum(axis=0) - sharp.sum(axis=0)).max() <= 2.0
    # the per-cell mean of every gene, sharpened or not, stays on the profile
    p_cell = sharp_em._fraction(d)
    assert np.abs(sharp_comp.mean(axis=0) - p_cell).sum() < 0.01
    assert np.abs(sharp_comp.mean(axis=0) - base_comp.mean(axis=0))[head].max() < 0.02 * p_cell[head].max()
    # what moved is the spread: a sharpened gene's composition varies far less from cell to cell
    ratio = sharp_comp.std(axis=0)[head] / base_comp.std(axis=0)[head]
    assert np.median(ratio) < 0.5 and ratio.max() < 0.9
    # and the genes that were not asked for keep their spread (their counts move only by rounding)
    other = ~head
    assert np.median(sharp_comp.std(axis=0)[other] / base_comp.std(axis=0)[other]) > 0.9
    moved = np.abs(base[:, other] - sharp[:, other])
    assert (moved == 0).mean() > 0.6 and moved.mean() < 0.5      # here the head is 39 % of the counts; on real folds it is a few %


def test_an_interior_scatter_sits_between_the_template_and_the_sharp_gene():
    rng = np.random.default_rng(13)
    prof = _profile(rng)
    d = rng.normal(0, 0.3, size=G)
    g = int(np.argmax(prof.fraction))
    spread = []
    for s in (1.0, 0.5, 0.0):
        sc = np.ones(G)
        sc[g] = s
        _, comp = _cpm(PoissonEmitter(prof, seed=2, lam=0.5).emit_dual(400, d, d, scatter=sc))
        spread.append(comp[:, g].std())
    assert spread[0] > spread[1] > spread[2]


def test_a_scatter_above_one_widens_a_gene_and_keeps_its_two_moments():
    rng = np.random.default_rng(16)
    prof = _profile(rng)
    d = rng.normal(0, 0.3, size=G)
    order = np.argsort(-prof.fraction)
    wide = np.ones(G)
    wide[order[:6]] = 3.0           # well-expressed genes
    wide[order[-6:]] = 3.0          # sparse genes: here the floor at zero does the work
    base_em, wide_em = PoissonEmitter(prof, seed=8, lam=0.5), PoissonEmitter(prof, seed=8, lam=0.5)
    base, base_comp = _cpm(base_em.emit_dual(400, d, d))
    out, comp = _cpm(wide_em.emit_dual(400, d, d, scatter=wide))
    assert wide_em.last_sharpened == 12 and wide_em.last_dual == "dual"
    assert np.array_equal(out, np.round(out)) and out.min() >= 0
    assert np.array_equal(base.sum(axis=1), out.sum(axis=1))
    p_cell = wide_em._fraction(d)
    assert np.abs(comp.mean(axis=0) - p_cell).sum() < 0.01
    assert np.abs(out.sum(axis=0) - base.sum(axis=0)).max() <= 2.0
    top = order[:6]
    assert (comp[:, top].std(axis=0) > 1.5 * base_comp[:, top].std(axis=0)).all()
    # a widened sparse gene has more empty cells than the template gave it
    low = order[-6:]
    assert (out[:, low] == 0).sum() > (base[:, low] == 0).sum()


def test_the_dial_never_changes_what_is_drawn_next():
    rng = np.random.default_rng(14)
    prof = _profile(rng)
    d1, d2 = rng.normal(0, 0.3, size=G), rng.normal(0, 0.3, size=G)
    a, b = PoissonEmitter(prof, seed=7, lam=0.5), PoissonEmitter(prof, seed=7, lam=0.5)
    a.emit_dual(200, d1, d1)
    b.emit_dual(200, d1, d1, scatter=np.zeros(G))
    assert np.array_equal(a.emit_dual(200, d2, d2).toarray(), b.emit_dual(200, d2, d2).toarray())


def test_scatter_is_checked_and_the_template_rung_returns_the_cells_as_drawn():
    rng = np.random.default_rng(15)
    prof = _profile(rng)
    d = rng.normal(0, 0.3, size=G)
    em = PoissonEmitter(prof, seed=1, lam=0.5)
    for bad in (np.ones(G - 1), np.full(G, -0.1), np.full(G, np.nan), np.full(G, np.inf)):
        with pytest.raises(ValueError, match="scatter"):
            em.emit_dual(50, d, d, scatter=bad)
    # a bulk profile far outside the depth envelope cannot be met: the last rung is the template,
    # and it comes back unsharpened and integral
    far = PoissonEmitter(prof, seed=1, lam=0.5)
    ref = PoissonEmitter(prof, seed=1, lam=0.5)
    M = far.emit_dual(120, d, 6.0 * d, on_fail="fallback", scatter=np.zeros(G))
    assert far.last_dual == "template" and far.last_sharpened == 0
    assert np.array_equal(M.toarray(), ref.emit(120, d).toarray())


def test_loco_reads_a_scatter_table_and_touches_only_what_it_lists(tmp_path):
    import anndata as ad
    import pandas as pd

    from sidechain.data.stream_pseudobulk import PseudobulkSums
    from sidechain.eval import loco

    rng = np.random.default_rng(21)
    genes = np.array([f"g{i}" for i in range(G)], dtype=object)
    basal = rng.uniform(100, 2000, size=G)
    mean = np.stack([basal, basal * np.exp2(rng.normal(0, 0.15, G)), basal * np.exp2(rng.normal(0, 0.15, G))])
    n = np.full(3, 1000, dtype=np.int64)
    src = PseudobulkSums(labels=["ctrl", "g0", "g1"], genes=genes.copy(), count_sum=mean * n[:, None],
                         cpm_sum=mean * n[:, None], cpm_sq_sum=(mean**2 + mean) * n[:, None],
                         n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["t"])
    rows, labels = [], []
    for lab, k in (("non-targeting", 60), ("g0", 12), ("g1", 12)):
        for _ in range(k):
            rows.append(rng.poisson(basal / basal.sum() * rng.integers(3000, 6000))); labels.append(lab)
    real = ad.AnnData(X=sp.csr_matrix(np.asarray(rows, dtype=np.float32)),
                      obs=pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(len(rows))]),
                      var=pd.DataFrame(index=genes.astype(str)))
    real_path = tmp_path / "real.h5ad"
    real.write_h5ad(real_path)
    top = [f"g{j}" for j in np.argsort(-basal)[:6] if j not in (0, 1)][:4]      # four well-expressed genes
    table = tmp_path / "scatter.parquet"
    pd.DataFrame({"target": ["g0"] * 4 + ["g0", "g0", "zz", "g0"],
                  "feature": top + ["g0", "nope", "g5", top[0] + "x"],
                  "scatter": [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]}).to_parquet(table)
    kw = dict(pert_col="perturbation", control="non-targeting", shrinkage=False, var_floor="poisson",
              emit_lambda=0.5, alpha=1.35, alpha_bulk=1.35, min_libsize=0.0, cells_per_pert=300)
    plain = loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "plain.h5ad", **kw)
    dialed = loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "dialed.h5ad",
                                            scatter_table=table, **kw)
    assert "scatter_table" not in plain
    rec = dialed["scatter_table"]
    assert rec["rows"] == 8 and rec["pairs_applied"] == 4 and rec["rows_at_scatter_zero"] == 3
    assert rec["rows_dropped"] == {"target_not_in_this_file": 1, "gene_not_on_the_axis": 2, "the_targets_own_gene": 1}
    assert rec["targets_with_a_pair"] == 1 and rec["targets_carrying_it"] == 1
    assert rec["targets_listed_but_not_carrying"] == [] and len(rec["sha256"]) == 64
    a, b = ad.read_h5ad(tmp_path / "plain.h5ad"), ad.read_h5ad(tmp_path / "dialed.h5ad")
    lab = a.obs["perturbation"].to_numpy()
    A, B_ = a.X.toarray().astype(np.float64), b.X.toarray().astype(np.float64)
    # the target the table does not list is the same cells, bit for bit
    assert np.array_equal(A[lab == "g1"], B_[lab == "g1"])
    # the listed target keeps every depth; its listed genes lose spread, the fully sharpened ones most
    g0 = lab == "g0"
    assert np.array_equal(A[g0].sum(axis=1), B_[g0].sum(axis=1))
    comp_a, comp_b = A[g0] / A[g0].sum(axis=1, keepdims=True), B_[g0] / B_[g0].sum(axis=1, keepdims=True)
    cols = [int(g[1:]) for g in top]
    ratio = comp_b[:, cols].std(axis=0) / comp_a[:, cols].std(axis=0)
    assert (ratio[:3] < 0.8).all() and ratio[3] < 1.0 and ratio[:3].mean() < ratio[3]
    assert np.abs(comp_b[:, cols].mean(axis=0) / comp_a[:, cols].mean(axis=0) - 1).max() < 0.01
    # refusals: one channel, a pair listed twice, a value outside the dial
    one = {k: v for k, v in kw.items() if k != "alpha_bulk"}
    with pytest.raises(SystemExit, match="two-channel"):
        loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "x.h5ad", scatter_table=table, **one)
    twice = tmp_path / "twice.parquet"
    pd.DataFrame({"target": ["g0", "g0"], "feature": [top[0], top[0]], "scatter": [0.0, 0.5]}).to_parquet(twice)
    with pytest.raises(SystemExit, match="listed twice"):
        loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "x.h5ad", scatter_table=twice, **kw)
    wild = tmp_path / "wild.parquet"
    pd.DataFrame({"target": ["g0"], "feature": [top[0]], "scatter": [-0.5]}).to_parquet(wild)
    with pytest.raises(SystemExit, match=">= 0"):
        loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "x.h5ad", scatter_table=wild, **kw)
