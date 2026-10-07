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


# ── T85: the controls' own shape (private research/ideas/reach-call-set-emitter.md) ──────────────────

GS = 300      # the shape tests need expression classes: sparse, depth-tracking, on/off and minority-carried genes
KINDS = ("plain", "plain", "plain", "depth-tracking", "on/off", "minority")


def _gene_laws(g=GS):
    """One fixed set of genes: mean counts per cell from 0.02 to 40 at depth 8,000; half over-dispersed
    gamma genes, a sixth tracking the cell's depth (so the pooled profile is not the mean-CPM one), a
    sixth off in 60 % of cells, a sixth carried by a minority (5 % of cells at eight times the rest)."""
    rng = np.random.default_rng(7)
    base = np.exp(rng.uniform(np.log(2.5e-6), np.log(5e-3), size=g))
    return base / base.sum(), rng.uniform(1.0, 4.0, size=g), np.array([KINDS[j % 6] for j in range(g)])


def _cells_like_real(rng, n, log2fc=None, g=GS):
    """Cells of those genes; with `log2fc` the same cells at shifted rates, which is the truth a
    shaped gene is held against."""
    base, k, kind = _gene_laws(g)
    depth = rng.lognormal(np.log(8000), 0.45, size=n)
    rate = base[None, :] * rng.gamma(k[None, :], 1.0 / k[None, :], size=(n, g))
    rate[:, kind == "depth-tracking"] *= ((depth / 8000) ** 0.5)[:, None]
    rate[:, kind == "on/off"] *= rng.random((n, int((kind == "on/off").sum()))) < 0.4
    few = kind == "minority"
    rate[:, few] = base[None, few] * rng.gamma(25.0, 0.04, size=(n, int(few.sum()))) * np.where(
        rng.random((n, int(few.sum()))) < 0.05, 8.0, 1.0)
    if log2fc is not None:
        rate = rate * np.exp2(log2fc)[None, :]
    rate /= rate.sum(axis=1, keepdims=True)
    return rng.poisson(rate * depth[:, None]).astype(np.float32)


def _kept_profile(tmp_path, X, **kw):
    import anndata as ad
    import pandas as pd

    path = tmp_path / f"ctrl_{len(X)}_{X.shape[1]}.h5ad"
    ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame(index=[f"c{i}" for i in range(len(X))]),
               var=pd.DataFrame(index=[f"g{j}" for j in range(X.shape[1])])).write_h5ad(path)
    return ContextProfile.from_controls(path, "A", **kw)


def _rank_test(M, X_ctrl, alpha=0.05):
    """The scorer's kind of test: a two-sided rank-sum test on per-cell CPM of the emitted cells
    against every control cell, Benjamini-Hochberg over the genes. Returns (signed z, called)."""
    from scipy.stats import mannwhitneyu, norm

    cpm = lambda A: A / A.sum(axis=1, keepdims=True) * 1e6
    a, b = cpm(np.asarray(M, dtype=np.float64)), cpm(X_ctrl.astype(np.float64))
    r = mannwhitneyu(a, b, axis=0, method="asymptotic", use_continuity=False)
    z = np.sign(r.statistic - len(a) * len(b) / 2.0) * norm.isf(np.clip(r.pvalue, 1e-300, 1.0) / 2.0)
    order = np.argsort(r.pvalue)
    q = np.minimum.accumulate((r.pvalue[order] * len(z) / np.arange(1, len(z) + 1))[::-1])[::-1]
    called = np.zeros(len(z), dtype=bool)
    called[order] = q < alpha
    return z, called


def _classes(X):
    per_cell = X.mean(axis=0)
    kind = _gene_laws(X.shape[1])[2]
    plain = kind == "plain"
    cls = {"sparse": plain & (per_cell < 0.3), "middle": plain & (per_cell >= 0.3) & (per_cell < 3),
           "high": plain & (per_cell >= 3), "depth-tracking": kind == "depth-tracking", "on/off": kind == "on/off",
           "minority": (kind == "minority") & (per_cell >= 1)}
    assert all(m.sum() >= 12 for m in cls.values()), {k: int(m.sum()) for k, m in cls.items()}
    return cls


def test_from_controls_keeps_the_cells_only_when_asked_and_reads_each_genes_spread(tmp_path):
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(31)
    X = _cells_like_real(rng, 3000)
    bare = _kept_profile(tmp_path, X)
    assert bare.cells is None and bare.dispersion is None and bare.zero_rate is None
    floor = float(np.median(X.sum(axis=1)))
    prof = _kept_profile(tmp_path, X, min_libsize=floor, keep_cells=True)
    assert sp.isspmatrix_csr(prof.cells) and prof.cells.shape == (prof.n_cells, GS)
    assert np.array_equal(np.asarray(prof.cells.sum(axis=1)).ravel(), prof.libsizes)
    assert np.array_equal(prof.cells.toarray(), X[X.sum(axis=1) > floor])
    # the spread a Poisson draw does not explain: a gamma rate of shape 1 to 4 is a squared CV of
    # 0.25 to 1, and a gene that is off in most cells carries more than any gamma here
    whole = _kept_profile(tmp_path, X, keep_cells=True)
    cls = _classes(X)
    assert whole.dispersion.shape == (GS,) and (whole.dispersion >= 0).all()
    assert 0.2 < np.median(whole.dispersion[cls["high"]]) < 1.2
    assert np.median(whole.dispersion[cls["on/off"]]) > np.median(whole.dispersion[cls["high"]])
    poisson = _kept_profile(tmp_path, rng.poisson(np.tile(np.linspace(0.5, 20, 30), (1500, 1))).astype(np.float32),
                            keep_cells=True)
    assert np.median(poisson.dispersion) < 0.02
    # a gene the controls hold a few counts of has no spread to read (the moment estimate is zero
    # for a third of such genes and several times too large for most others): it takes the
    # well-measured genes' typical value
    small = _kept_profile(tmp_path, X[:400], keep_cells=True)
    few = X[:400].sum(axis=0) < 100
    typical = np.median(small.dispersion[X[:400].sum(axis=0) >= 400])
    assert few.sum() >= 10 and (small.dispersion[few] > 0.3 * typical).all()
    assert 0.6 < np.median(small.dispersion[few]) / typical < 1.8
    # what a cell with no count of a gene may expect of it, against the gamma reading: the same for
    # a gamma-Poisson gene (and a Poisson one), a small part of it for a gene that is off in most cells
    assert 0.7 < np.median(whole.zero_rate[cls["middle"]]) < 1.3
    assert np.median(whole.zero_rate[cls["on/off"] & (X.mean(axis=0) > 1)]) < 0.4
    assert 0.8 < np.median(poisson.zero_rate[:8]) < 1.25
    # a zero stored in the file is not a count: the same profile with and without stored zeros
    stored = sp.csr_matrix(X[:400])
    stored.data[::7] = 0.0
    clean = stored.copy()
    clean.eliminate_zeros()
    profs = []
    for name, M in (("stored", stored), ("clean", clean)):
        path = tmp_path / f"{name}.h5ad"
        ad.AnnData(X=M, obs=pd.DataFrame(index=[f"c{i}" for i in range(400)]),
                   var=pd.DataFrame(index=[f"g{j}" for j in range(GS)])).write_h5ad(path)
        profs.append(ContextProfile.from_controls(path, "A", keep_cells=True))
    assert profs[0].cells.nnz == profs[1].cells.nnz and np.array_equal(profs[0].zero_rate, profs[1].zero_rate)


@pytest.mark.parametrize("anchor", ["mean_cpm", "pooled"])
def test_shape_is_off_by_default_and_false_is_the_same_call(tmp_path, anchor):
    rng = np.random.default_rng(32)
    X = _cells_like_real(rng, 900)
    prof, bare = _kept_profile(tmp_path, X, keep_cells=True), _kept_profile(tmp_path, X)
    d = rng.normal(0, 0.3, size=GS)
    sc = np.where(np.arange(GS) % 3 == 0, 0.5, 1.0)
    em = lambda p: PoissonEmitter(p, seed=4, lam=0.5, bulk_anchor=anchor)
    plain, off = em(bare), em(prof)
    a = plain.emit_dual(300, d, d).toarray()
    assert np.array_equal(a, off.emit_dual(300, d, d, shape=False).toarray())     # keeping the cells changes nothing
    assert plain.last_shaped is None and off.last_shaped is None
    assert np.array_equal(em(bare).emit_dual(300, d, d, scatter=sc).toarray(),
                          em(prof).emit_dual(300, d, d, scatter=sc, shape=False).toarray())
    on = em(prof)
    on.emit_dual(300, d, d, shape=True)
    assert on.last_shaped is True
    on.emit(10, d)
    assert on.last_shaped is None
    # refusals: a profile that kept no cells, or whose cells are not its depths' cells
    with pytest.raises(ValueError, match="keep_cells=True"):
        plain.emit_dual(50, d, d, shape=True)
    short = _kept_profile(tmp_path, X, keep_cells=True)
    short.cells = short.cells[:-10]
    with pytest.raises(ValueError, match="one row per depth"):
        em(short).emit_dual(50, d, d, shape=True)


def test_asked_for_every_control_cell_and_nothing_predicted_the_block_is_the_control_cells(tmp_path):
    """The construction in one line. With as many cells asked for as the profile holds, the draw
    is every control cell once (no cell twice), their two moments are the prediction already, and
    nothing is thinned or topped up: what comes back is the control cells."""
    rng = np.random.default_rng(33)
    X = _cells_like_real(rng, 500)
    prof = _kept_profile(tmp_path, X, keep_cells=True)
    out = PoissonEmitter(prof, seed=3, lam=0.5, bulk_anchor="pooled").emit_dual(len(X), None, None, shape=True).toarray()
    assert np.array_equal(np.sort(out.sum(axis=1)), np.sort(X.sum(axis=1)))
    ours, theirs = np.sort(out, axis=0), np.sort(X, axis=0)
    # what differs is the fit's integer rounding: a count in two thousand moved, by one or two,
    # nearly all of it on counts in the tens and hundreds
    assert np.abs(ours - theirs).sum() < 0.002 * theirs.sum() and np.abs(ours - theirs).max() <= 4
    assert (ours == theirs)[theirs <= 5].mean() > 0.99 and abs((out == 0).mean() - (X == 0).mean()) < 0.004
    # and a depth window is kept: cells drawn inside it, at their own depths
    lo, hi = np.quantile(prof.libsizes, [0.25, 0.75])
    mid = PoissonEmitter(prof, seed=3, lam=0.5, bulk_anchor="pooled", libsize_quantiles=(0.25, 0.75))
    depth = mid.emit_dual(120, None, None, shape=True).toarray().sum(axis=1)
    assert depth.min() > 0.9 * lo and depth.max() < 1.1 * hi


def test_a_shaped_block_is_real_cells_at_real_depths_and_sits_on_both_moments(tmp_path):
    rng = np.random.default_rng(34)
    X = _cells_like_real(rng, 4000)
    prof = _kept_profile(tmp_path, X, keep_cells=True)
    cls = _classes(X)
    well = cls["high"] | cls["middle"] | cls["depth-tracking"]
    delta = lambda b: np.log2((b + 1e-9) / (prof.bulk_fraction + 1e-9))[well]
    for size in (0.05, 0.4):                       # a weak knockdown and a strong one
        d = rng.normal(0, size, size=GS)
        em = PoissonEmitter(prof, seed=9, lam=0.5, bulk_anchor="pooled")
        ref = PoissonEmitter(prof, seed=9, lam=0.5, bulk_anchor="pooled")
        _, depth_t, _, bulk_t = _moments(ref.emit_dual(400, d, d))
        out, depth, per_cell, bulk = _moments(em.emit_dual(400, d, d, shape=True))
        assert em.last_shaped is True and em.last_dual == "dual"
        assert np.array_equal(out, np.round(out)) and out.min() >= 0
        # the cells take real cells' depths: as wide as the controls', where the template's are a quarter of that
        cv = lambda v: v.std() / v.mean()
        assert cv(depth) > 0.8 * cv(prof.libsizes) and cv(depth_t) < 0.4 * cv(prof.libsizes)
        # both moments on the prediction, as for any block
        assert np.abs(per_cell - em._fraction(d)).sum() < 0.01
        assert np.abs(bulk - em._fraction(d, bulk=True)).sum() < 0.01
        # so the summed profile's change is the unshaped call's (the weak knockdown is the hard case)
        assert np.corrcoef(delta(bulk), delta(bulk_t))[0, 1] > (0.995 if size > 0.1 else 0.9)
        # and the per-cell-mean fold change is the one asked for, down to the genes that are mostly zeros
        asked = np.log2(em._fraction(d) / prof.fraction)
        got = np.log2((per_cell + 1e-12) / (prof.fraction + 1e-12))
        assert np.median(np.abs(got - asked)[~cls["sparse"]]) < 0.01 and np.median(np.abs(got - asked)[cls["sparse"]]) < 0.08


@pytest.mark.parametrize("anchor", ["mean_cpm", "pooled"])
def test_with_nothing_predicted_a_shaped_block_reads_as_control_cells(tmp_path, anchor):
    """The actuator's own control. A template gene is called for being narrower than a real cell's,
    whatever is predicted; a shaped block with nothing predicted must be control cells to the rank
    test, in every expression class, under either anchor. Two earlier constructions failed it: a
    control cell's composition scaled to the template's depth (z near -0.8, about twenty calls a
    knockdown), and a re-rating that took the drawn cells' sampling error out of every cell alike
    (calls on genes a few cells carry)."""
    rng = np.random.default_rng(35)
    X = _cells_like_real(rng, 5000)
    prof = _kept_profile(tmp_path, X, keep_cells=True)
    cls = _classes(X)
    draws = 8
    em = lambda s: PoissonEmitter(prof, seed=s, lam=0.5, bulk_anchor=anchor)
    as_emitted = _rank_test(em(0).emit_dual(400, None, None).toarray(), X)
    assert as_emitted[1].sum() > 0.4 * GS and as_emitted[0][cls["high"]].mean() > 2
    blocks = [em(s).emit_dual(400, None, None, shape=True).toarray() for s in range(draws)]
    tests = [_rank_test(b, X) for b in blocks]
    raw = np.array([_rank_test(X[np.random.default_rng(50 + s).choice(len(X), 400, replace=False)], X)[0] for s in range(draws)])
    z = np.array([t[0] for t in tests])
    assert max(int(t[1].sum()) for t in tests) <= 1                 # raw control cells give 0 or 1
    zeros = np.mean([(b == 0).mean(axis=0) for b in blocks], axis=0) - (X == 0).mean(axis=0)
    for name, m in cls.items():
        # a gene the block holds about ten counts of sits a little low (the fit rescales it, and a
        # rescaled count never fills a zero): the one class with a one-sided allowance
        lo = -0.45 if name == "sparse" else -0.35
        assert lo < z[:, m].mean() < 0.35, (name, z[:, m].mean())
        assert abs(zeros[m].mean()) < 0.012, (name, zeros[m].mean())
        # never a wider null than a draw of real cells, class by class
        assert z[:, m].std(axis=0, ddof=1).mean() < 1.15 * raw[:, m].std(axis=0, ddof=1).mean(), name


def test_a_predicted_shift_moves_a_shaped_gene_as_it_moves_a_real_one(tmp_path):
    rng = np.random.default_rng(36)
    X = _cells_like_real(rng, 5000)
    prof = _kept_profile(tmp_path, X, keep_cells=True)
    cls = _classes(X)
    order = np.random.default_rng(3).permutation(GS)
    d = np.zeros(GS)
    up, down = order[:110], order[110:220]
    d[up], d[down] = 1.0, -1.0
    # what is asked of the emitter is the per-cell-mean fold change truly shifted cells show
    comp = lambda A: (A / A.sum(axis=1, keepdims=True)).mean(axis=0)
    asked = np.log2((comp(_cells_like_real(np.random.default_rng(99), 20000, d)) + 1e-12) / (comp(X) + 1e-12))
    draws = 4
    truth = [_cells_like_real(np.random.default_rng(1000 + s), 400, d) for s in range(draws)]
    ours = [PoissonEmitter(prof, seed=s, lam=0.5, bulk_anchor="pooled").emit_dual(400, asked, asked, shape=True).toarray()
            for s in range(draws)]
    z_true = np.mean([_rank_test(b, X)[0] for b in truth], axis=0)
    z_ours = np.mean([_rank_test(b, X)[0] for b in ours], axis=0)
    zero = lambda blocks: np.mean([(b == 0).mean(axis=0) for b in blocks], axis=0) - (X == 0).mean(axis=0)
    for name, m in cls.items():
        for idx in (up, down):
            both = idx[m[idx]]
            assert len(both) >= 3, (name, len(both))                 # an empty class is a broken fixture, not a pass
            assert 0.7 < z_ours[both].mean() / z_true[both].mean() < 1.4, (name, z_ours[both].mean(), z_true[both].mean())
    # an up-shift fills zeros and a down-shift makes them, as in the truly shifted cells; a
    # composition scaled by a factor could never do the first
    low = cls["sparse"] | cls["middle"] | cls["on/off"]
    low_up, low_down = up[low[up]], down[low[down]]
    assert zero(ours)[low_up].mean() < -0.03 and abs(zero(ours)[low_up].mean() - zero(truth)[low_up].mean()) < 0.02
    assert zero(ours)[low_down].mean() > 0.03 and abs(zero(ours)[low_down].mean() - zero(truth)[low_down].mean()) < 0.02


@pytest.mark.parametrize("gap", [1.0, 1.26])
def test_the_fit_is_handed_cells_that_already_sit_on_both_moments(tmp_path, gap):
    """Why the re-rating solves for two numbers a gene: the fit can only rescale a count, and a
    rescaled count never fills a zero, so the fit must have next to nothing left to do, also when
    the summed profile is asked at another amplitude than the per-cell mean."""
    import sidechain.models.count_emitters as ce

    rng = np.random.default_rng(37)
    X = _cells_like_real(rng, 4000)
    prof = _kept_profile(tmp_path, X, keep_cells=True)
    d = rng.normal(0, 0.3, size=GS)
    em = PoissonEmitter(prof, seed=6, lam=0.5, bulk_anchor="pooled")
    seen, orig = [], ce.dual_moment_counts
    try:
        ce.dual_moment_counts = lambda start, *a, **k: seen.append((start.copy(), k["depths"])) or orig(start, *a, **k)
        out = em.emit_dual(400, d, gap * d, shape=True).toarray()
    finally:
        ce.dual_moment_counts = orig
    assert em.last_dual == "dual" and len(seen) == 1
    start, depths = seen[0]
    assert np.array_equal(start, np.round(start)) and np.array_equal(depths, start.sum(axis=1))    # counts, not rescaled ones
    per_cell = (start / depths[:, None]).mean(axis=0)
    bulk = start.sum(axis=0) / start.sum()
    assert np.abs(per_cell - em._fraction(d)).sum() < 0.008
    assert np.abs(bulk - em._fraction(gap * d, bulk=True)).sum() < 0.008
    # what the fit then moves: about one count in 150, and one zero-or-not in 400
    assert np.abs(out - start).sum() < 0.012 * start.sum() and ((out == 0) != (start == 0)).mean() < 0.006


def test_two_amplitudes_are_carried_as_a_lean_with_depth_and_no_cell_is_emptied(tmp_path):
    rng = np.random.default_rng(38)
    X = _cells_like_real(rng, 4000)
    prof = _kept_profile(tmp_path, X, keep_cells=True)
    cls = _classes(X)
    order = np.random.default_rng(4).permutation(np.flatnonzero(cls["high"]))
    d = np.zeros(GS)
    up, down = order[:15], order[15:30]
    d[up], d[down] = 1.0, -1.0
    one, two = (PoissonEmitter(prof, seed=2, lam=0.5, bulk_anchor="pooled") for _ in range(2))
    A = one.emit_dual(400, d, d, shape=True).toarray()
    B = two.emit_dual(400, d, 1.26 * d, shape=True).toarray()
    assert two.last_dual == "dual"
    lean = lambda M, g: np.mean([np.corrcoef(M.sum(axis=1), M[:, j] / M.sum(axis=1))[0, 1] for j in g])
    assert abs(lean(A, up)) < 0.12 and lean(B, up) > 0.2 and lean(B, down) < -0.2      # the gap, as a lean with depth
    zeros = lambda M, g: (M[:, g] == 0).mean()
    assert abs(zeros(B, down) - zeros(A, down)) < 0.03 and abs(zeros(B, up) - zeros(A, up)) < 0.03


def _cells_with_a_long_tail(rng, n, g=GS):
    """`_cells_like_real` with what broke two box arms on HEK293T: a few cells eight to ten times the
    mean depth, depth-tracking genes that rise faster than depth, and a few cells holding fifteen times
    their share of a sixth of the genes."""
    base, k, kind = _gene_laws(g)
    depth = rng.lognormal(np.log(8000), 0.45, size=n)
    deep = rng.choice(n, size=max(n // 100, 4), replace=False)
    depth[deep] = 8000 * rng.uniform(8.0, 10.0, size=len(deep))
    rate = base[None, :] * rng.gamma(k[None, :], 1.0 / k[None, :], size=(n, g))
    rate[:, kind == "depth-tracking"] *= ((depth / 8000) ** 1.0)[:, None]
    rate[:, kind == "on/off"] *= rng.random((n, int((kind == "on/off").sum()))) < 0.4
    odd = rng.choice(n, size=max(n // 100, 4), replace=False)
    rate[np.ix_(odd, np.flatnonzero(kind == "minority"))] *= 15.0
    rate /= rate.sum(axis=1, keepdims=True)
    return rng.poisson(rate * depth[:, None]).astype(np.float32)


@pytest.mark.parametrize("shift", [False, True])
def test_no_shaped_cell_runs_away_from_its_own_depth(tmp_path, shift):
    """T85, 2026-10-06: two box arms died on the scorer's 1,000,000-count limit because a re-rated
    HEK293T cell reached 1,684,750 counts. The solve that moves the drawn cells onto the two moments
    diverged where a few very deep cells carry a gene, and threw those cells to its 64-fold limit.
    On cells with that long tail, predicted shift or none: the re-rated cells end no further from
    the two moments than the drawn cells began, to within 0.05 (a solve that takes every step ends
    far further), and no cell needs holding to the depth cap."""
    X = _cells_with_a_long_tail(np.random.default_rng(3), 2500)
    prof = _kept_profile(tmp_path, X, keep_cells=True)
    em = PoissonEmitter(prof, seed=0, lam=0.5, bulk_anchor="pooled")
    d = np.zeros(GS)
    if shift:
        d[::3] = np.where(np.arange(GS)[::3] % 2 == 0, 1.0, -1.0)
    p_cell, p_bulk = em._fraction(d), em._fraction(d, bulk=True)
    gain, deep_in, held = [], 0, 0
    for s in range(11, 23):
        rows = em._depth_spread_rows(400, np.random.default_rng([s, 1]))
        own = prof.libsizes[rows]
        start, depths = em._shaped_start(400, p_cell, p_bulk, s)
        assert np.array_equal(depths, np.maximum(np.rint(start.sum(axis=1)), 1).astype(np.int64))
        gain.append(float((start.sum(axis=1) / own).max()))
        deep_in += int((own > 6 * prof.libsizes.mean()).sum())
        held += em.last_shape_held
        # the largest relative miss of either moment over the well-counted genes, after and before
        drawn = prof.cells[rows].toarray().astype(np.float64) * (p_cell / prof.fraction)[None, :]
        off = lambda M: max(np.abs((M / own[:, None]).mean(axis=0) / p_cell - 1.0)[prof.fraction > 2e-4].max(),
                            np.abs(M.sum(axis=0) / (p_bulk * own.sum()) - 1.0)[prof.fraction > 2e-4].max())
        assert off(start) <= off(drawn) + 0.05, (off(start), off(drawn))
    assert deep_in >= 12                                   # the deep cells are in the draws
    assert held == 0 and max(gain) < 2.0, (gain, held)      # the old solve wrote a cell of over two million counts here


def test_a_shaped_block_holds_the_controls_own_spread_of_depths(tmp_path):
    """A block is one cell from each of 400 slices of the control cells in order of depth: distinct
    cells, every slice represented, so the deep tail the summed profile leans on is in every block
    (a plain random draw of these cells leaves it out of about one block in eighty)."""
    X = _cells_with_a_long_tail(np.random.default_rng(3), 2500)
    prof = _kept_profile(tmp_path, X, keep_cells=True)
    em = PoissonEmitter(prof, seed=0, lam=0.5, bulk_anchor="pooled")
    order = np.sort(prof.libsizes)
    deep = prof.libsizes > 6 * prof.libsizes.mean()
    for s in range(40):
        rows = em._depth_spread_rows(400, np.random.default_rng([s, 1]))
        assert len(np.unique(rows)) == 400
        got = np.sort(prof.libsizes[rows])
        lo, hi = order[np.floor(np.linspace(0, 2500, 401)).astype(int)[:-1]], order[np.floor(np.linspace(0, 2500, 401)).astype(int)[1:] - 1]
        assert (got >= lo).all() and (got <= hi).all()          # one cell from every slice
        assert 3 <= deep[rows].sum() <= 5                       # 25 deep cells in 2,500: four a block
    assert sum(deep[np.random.default_rng([s, 1]).choice(2500, 400, replace=False)].sum() == 0 for s in range(400)) > 0
    every = np.concatenate([em._depth_spread_rows(400, np.random.default_rng([s, 1])) for s in range(40)])
    assert len(np.unique(every)) >= 2400                    # any cell of a slice can be the one drawn (2,497 of 2,500 here)
    # the same draw for the same seed, another for another; too few cells: with replacement, as before
    a, b = em._depth_spread_rows(400, np.random.default_rng([1, 1])), em._depth_spread_rows(400, np.random.default_rng([1, 1]))
    assert np.array_equal(a, b) and not np.array_equal(a, em._depth_spread_rows(400, np.random.default_rng([2, 1])))
    assert len(em._depth_spread_rows(3000, np.random.default_rng(0))) == 3000


def test_a_cell_asked_for_more_than_twice_its_depth_is_held_to_that(tmp_path, monkeypatch):
    """The guarantee under the solve: whatever is predicted, no re-rated cell leaves with more than
    twice the depth it came with. A sixty-four-fold rise of the genes the deep cells hold most of
    asks the deepest cells for two to three times their depth; every such cell is held to exactly
    twice, counted, and loses only counts that were added to it; a cell under the cap is untouched;
    and the emitted cells keep those depths."""
    import sidechain.models.count_emitters as ce

    X = _cells_with_a_long_tail(np.random.default_rng(3), 2500)
    prof = _kept_profile(tmp_path, X, keep_cells=True)
    em = PoissonEmitter(prof, seed=0, lam=0.5, bulk_anchor="pooled")
    up = np.zeros(GS)
    up[_gene_laws(GS)[2] == "depth-tracking"] = 6.0
    p_cell, p_bulk = em._fraction(up), em._fraction(up, bulk=True)
    rows = em._depth_spread_rows(400, np.random.default_rng([5, 1]))
    own, came = prof.libsizes[rows], prof.cells[rows].toarray().astype(np.float64)
    start, _ = em._shaped_start(400, p_cell, p_bulk, 5)
    held = em.last_shape_held
    monkeypatch.setattr(ce, "SHAPE_DEPTH_GAIN_CAP", 1e30)
    free, _ = em._shaped_start(400, p_cell, p_bulk, 5)       # the same draws, no cap
    assert em.last_shape_held == 0
    monkeypatch.undo()
    over = free.sum(axis=1) > 2.0 * own
    assert over[np.argsort(own)[-4:]].sum() >= 2             # the deepest cells are among those asked for more
    assert held == int(over.sum()) > 0
    assert np.array_equal(start.sum(axis=1)[over], np.floor(2.0 * own[over]))      # exactly twice, the literal
    assert np.array_equal(start[~over], free[~over])                               # under the cap: untouched
    assert (start >= np.minimum(free, came)).all() and (start <= free).all()       # only added counts are taken
    assert (start >= 0).all() and np.array_equal(start, np.rint(start))
    # through the public call the fit keeps every row's total, so each emitted cell is within twice ITS depth
    seen, rows_of = [], em._depth_spread_rows
    monkeypatch.setattr(em, "_depth_spread_rows", lambda *a, **k: seen.append(rows_of(*a, **k)) or seen[-1])
    block = em.emit_dual(400, up, up, on_fail="raise", shape=True).toarray()
    assert em.last_shaped is True and em.last_shape_held >= 1
    assert (block.sum(axis=1) <= 2.0 * prof.libsizes[seen[-1]]).all()


def test_no_shaped_cell_passes_the_scorers_count_limit_whatever_its_own_depth(tmp_path, monkeypatch):
    """Twice a cell's own depth is not the scorer's limit: a control cell over half of
    max_counts_per_cell could still leave above it. The second cap is absolute, and it is the
    number the scorer, the submission writer and `emit` use. With the limit set inside this
    fixture's depths (30,000, where its deep cells hold 64,000 to 80,000), no cell leaves over it,
    a control cell that is itself over it keeps that many of its own counts, and a cell under both
    caps is untouched by either."""
    import sidechain.models.count_emitters as ce
    from sidechain.submit.writer import MAX_COUNTS_PER_CELL

    assert (ce.SHAPE_MAX_COUNTS_PER_CELL == 1_000_000 == MAX_COUNTS_PER_CELL
            == PoissonEmitter.emit.__kwdefaults__["max_counts_per_cell"])
    assert ce.SHAPE_DEPTH_GAIN_CAP == 2.0

    X = _cells_with_a_long_tail(np.random.default_rng(3), 2500)
    prof = _kept_profile(tmp_path, X, keep_cells=True)
    em = PoissonEmitter(prof, seed=0, lam=0.5, bulk_anchor="pooled")
    p_cell, p_bulk = em._fraction(None), em._fraction(None, bulk=True)
    free, _ = em._shaped_start(400, p_cell, p_bulk, 7)
    monkeypatch.setattr(ce, "SHAPE_MAX_COUNTS_PER_CELL", 30_000)
    start, depths = em._shaped_start(400, p_cell, p_bulk, 7)
    own = prof.libsizes[em._depth_spread_rows(400, np.random.default_rng([7, 1]))]
    assert (own > 30_000).sum() >= 3                        # control cells that are themselves over the limit
    assert start.sum(axis=1).max() == 30_000 and em.last_shape_held >= (own > 30_000).sum()
    assert (start >= 0).all() and np.array_equal(start, np.rint(start))
    low = free.sum(axis=1) <= 30_000                        # the same draws up to the caps: those cells are the same cells
    assert low.sum() > 380 and np.array_equal(start[low], free[low])
    block = em.emit_dual(400, None, None, on_fail="raise", shape=True).toarray()
    assert block.sum(axis=1).max() <= 30_000


def test_a_gene_predicted_to_nothing_is_met_and_the_solve_runs_to_its_end(tmp_path):
    """Two things the solve's record must not get wrong. A gene whose predicted fraction is exactly
    zero is delivered by its ratio alone (every count thinned away) and is not a miss. And with two
    amplitudes on ordinary cells the solve has the steps to finish: all but a gene or two a block
    end within 1 % of both moments."""
    X = _cells_like_real(np.random.default_rng(4), 2500)
    prof = _kept_profile(tmp_path, X, keep_cells=True)
    em = PoissonEmitter(prof, seed=0, lam=0.5, bulk_anchor="pooled")
    gone = np.arange(GS) % 2 == 0
    p_cell = np.where(gone, 0.0, prof.fraction); p_cell = p_cell / p_cell.sum()
    p_bulk = np.where(gone, 0.0, prof.bulk_fraction); p_bulk = p_bulk / p_bulk.sum()
    start, _ = em._shaped_start(400, p_cell, p_bulk, 3)
    assert start[:, gone].sum() == 0 and start[:, ~gone].sum() > 0
    assert em.last_shape_unmet <= 5, em.last_shape_unmet
    rng = np.random.default_rng(9)
    d = np.where(rng.random(GS) < 0.4, rng.normal(0, 0.6, GS), 0.0)
    unmet = []
    for s in range(6):
        e = PoissonEmitter(prof, seed=s, lam=0.5, bulk_anchor="pooled")
        e.emit_dual(400, d, 1.4 * d, on_fail="anchor", shape=True)
        unmet.append(e.last_shape_unmet)
    assert np.mean(unmet) <= 3, unmet


def test_on_long_tailed_depths_a_shaped_block_with_nothing_predicted_still_reads_as_control_cells(tmp_path):
    """The fix must not buy safety with a worse null: on the long-tailed cells a shaped block with
    nothing predicted is called no more than control cells are, and the emitted cells sit on both
    moments."""
    X = _cells_with_a_long_tail(np.random.default_rng(3), 2500)
    prof = _kept_profile(tmp_path, X, keep_cells=True)
    em = PoissonEmitter(prof, seed=0, lam=0.5, bulk_anchor="pooled")
    calls = []
    for _ in range(6):
        block = em.emit_dual(400, None, None, on_fail="raise", shape=True).toarray()
        assert em.last_shaped is True and em.last_shape_held == 0
        calls.append(int(_rank_test(block, X)[1].sum()))
        depth = block.sum(axis=1)
        assert np.abs((block / depth[:, None]).mean(axis=0) - prof.fraction).sum() < 2e-3      # per-cell mean
        assert np.abs(block.sum(axis=0) / depth.sum() - em._fraction(None, bulk=True)).sum() < 2e-3   # summed profile
        assert depth.max() <= 2.0 * prof.libsizes.max()
    raw = [int(_rank_test(X[np.random.default_rng(50 + s).choice(len(X), 400, replace=False)], X)[1].sum()) for s in range(6)]
    assert max(raw) == 0 and max(calls) <= 1, (calls, raw)  # one gene in one block of 48 over eight emitter seeds; the old solve: five a block


@pytest.mark.parametrize("anchor", ["mean_cpm", "pooled"])
def test_on_ordinary_cells_the_solve_leaves_no_gene_unmet_and_holds_no_cell(tmp_path, anchor):
    Z = _cells_like_real(np.random.default_rng(35), 5000)
    prof = _kept_profile(tmp_path, Z, keep_cells=True)
    for s in range(8):
        em = PoissonEmitter(prof, seed=s, lam=0.5, bulk_anchor=anchor)
        em.emit_dual(400, None, None, shape=True)
        assert em.last_shape_unmet == 0 and em.last_shape_held == 0


def test_the_unmet_count_counts_what_the_solve_leaves(tmp_path, monkeypatch):
    """`last_shape_unmet` is a count, not a ceiling: on the long-tailed cells the solve leaves about
    one gene in eleven more than 1 % off, and with no steps at all several times as many."""
    import sidechain.models.count_emitters as ce

    X = _cells_with_a_long_tail(np.random.default_rng(3), 2500)
    prof = _kept_profile(tmp_path, X, keep_cells=True)
    em = PoissonEmitter(prof, seed=0, lam=0.5, bulk_anchor="pooled")
    p_cell, p_bulk = em._fraction(None), em._fraction(None, bulk=True)
    unmet = []
    for s in range(11, 23):
        em._shaped_start(400, p_cell, p_bulk, s)
        unmet.append(em.last_shape_unmet)
    assert 250 <= sum(unmet) <= 380, unmet                  # 326 here; without trust halving 456, at one step 839
    solved = unmet[0]
    monkeypatch.setattr(ce, "SHAPE_SOLVE_STEPS", 0)
    em._shaped_start(400, p_cell, p_bulk, 11)
    assert 10 <= solved <= 35 and em.last_shape_unmet > 3 * solved, (solved, em.last_shape_unmet)


@pytest.mark.parametrize("shift", [False, True])
def test_at_a_budget_of_one_step_the_solve_still_ends_no_worse_than_it_began(tmp_path, monkeypatch, shift):
    """The point used is the best one seen, not the last one tried: with a single step allowed the
    re-rated cells still end no further from the two moments than the drawn cells began."""
    import sidechain.models.count_emitters as ce

    X = _cells_with_a_long_tail(np.random.default_rng(3), 2500)
    prof = _kept_profile(tmp_path, X, keep_cells=True)
    em = PoissonEmitter(prof, seed=0, lam=0.5, bulk_anchor="pooled")
    monkeypatch.setattr(ce, "SHAPE_SOLVE_STEPS", 1)
    d = np.zeros(GS)
    if shift:
        d[::3] = np.where(np.arange(GS)[::3] % 2 == 0, 1.0, -1.0)
    p_cell, p_bulk = em._fraction(d), em._fraction(d, bulk=True)
    for s in range(11, 23):
        rows = em._depth_spread_rows(400, np.random.default_rng([s, 1]))
        own = prof.libsizes[rows]
        start, _ = em._shaped_start(400, p_cell, p_bulk, s)
        drawn = prof.cells[rows].toarray().astype(np.float64) * (p_cell / prof.fraction)[None, :]
        off = lambda M: max(np.abs((M / own[:, None]).mean(axis=0) / p_cell - 1.0)[prof.fraction > 2e-4].max(),
                            np.abs(M.sum(axis=0) / (p_bulk * own.sum()) - 1.0)[prof.fraction > 2e-4].max())
        assert off(start) <= off(drawn) + 0.05, (off(start), off(drawn))


def test_a_plain_emit_and_the_template_rung_clear_the_shape_counters(tmp_path):
    Z = _cells_like_real(np.random.default_rng(35), 1500)
    prof = _kept_profile(tmp_path, Z, keep_cells=True)
    em = PoissonEmitter(prof, seed=1, lam=0.5, bulk_anchor="pooled")
    em.emit_dual(100, None, None, shape=True)
    assert em.last_shape_held == 0 and em.last_shape_unmet is not None
    em.emit(10)
    assert em.last_shape_held is None and em.last_shape_unmet is None
    d = np.random.default_rng(40).normal(0, 0.3, size=GS)
    far = PoissonEmitter(prof, seed=5, lam=0.5)
    far.emit_dual(120, d, 9.0 * d, on_fail="fallback", shape=True)      # a pair the fit cannot carry: the template rung
    assert far.last_dual == "template" and far.last_shape_held is None and far.last_shape_unmet is None


def _loco_shaped_record(tmp_path, name):
    """`eval.loco`'s `emit_shape` record for three small targets emitted in the controls' shape."""
    import anndata as ad
    import pandas as pd

    from sidechain.data.stream_pseudobulk import PseudobulkSums
    from sidechain.eval import loco

    rng = np.random.default_rng(41)
    X_ctrl = _cells_like_real(rng, 600, g=G)
    genes = np.array([f"g{i}" for i in range(G)], dtype=object)
    basal = X_ctrl.mean(axis=0) + 1.0
    mean = np.stack([basal, basal * np.exp2(rng.normal(0, 0.15, G)), basal * np.exp2(rng.normal(0, 0.15, G))])
    n = np.full(3, 1000, dtype=np.int64)
    src = PseudobulkSums(labels=["ctrl", "g0", "g1"], genes=genes.copy(), count_sum=mean * n[:, None],
                         cpm_sum=mean * n[:, None], cpm_sq_sum=(mean**2 + mean) * n[:, None],
                         n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["t"])
    real = ad.AnnData(X=sp.csr_matrix(np.vstack([X_ctrl, X_ctrl[:36]])),
                      obs=pd.DataFrame({"perturbation": ["non-targeting"] * len(X_ctrl) + ["a00"] * 12 + ["g0"] * 12 + ["g1"] * 12},
                                       index=[f"c{i}" for i in range(len(X_ctrl) + 36)]),
                      var=pd.DataFrame(index=genes.astype(str)))
    real_path = tmp_path / f"real_{name}.h5ad"
    real.write_h5ad(real_path)
    kw = dict(pert_col="perturbation", control="non-targeting", shrinkage=False, var_floor="poisson",
              emit_lambda=0.5, alpha=1.35, alpha_bulk=1.35, bulk_anchor="pooled", min_libsize=0.0, cells_per_pert=300)
    return loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / f"{name}.h5ad", emit_shape="controls", **kw)["emit_shape"]


def test_the_loco_record_counts_held_cells_and_unmet_genes_when_there_are_some(tmp_path, monkeypatch):
    import sidechain.models.count_emitters as ce

    monkeypatch.setattr(ce, "SHAPE_DEPTH_GAIN_CAP", 1.2)
    rec = _loco_shaped_record(tmp_path, "held")
    assert rec["cells_held_to_the_caps"] >= 1 and rec["targets_with_a_held_cell"], rec
    monkeypatch.undo()
    monkeypatch.setattr(ce, "SHAPE_SOLVE_STEPS", 0)
    rec = _loco_shaped_record(tmp_path, "unsolved")
    assert rec["genes_left_unmet_a_target"]["median"] >= 5, rec


def test_shape_never_changes_what_is_drawn_next_and_scatter_composes_with_it(tmp_path):
    rng = np.random.default_rng(39)
    X = _cells_like_real(rng, 1500)
    prof = _kept_profile(tmp_path, X, keep_cells=True)
    d1, d2 = rng.normal(0, 0.3, size=GS), rng.normal(0, 0.3, size=GS)
    a, b = PoissonEmitter(prof, seed=7, lam=0.5), PoissonEmitter(prof, seed=7, lam=0.5)
    first = a.emit_dual(200, d1, d1).toarray()
    shaped_first = b.emit_dual(200, d1, d1, shape=True).toarray()
    assert not np.array_equal(first, shaped_first)
    assert np.array_equal(a.emit_dual(200, d2, d2).toarray(), b.emit_dual(200, d2, d2).toarray())
    # the same call twice is the same cells: every draw of the shape is seeded by the call
    c = PoissonEmitter(prof, seed=7, lam=0.5)
    assert np.array_equal(shaped_first, c.emit_dual(200, d1, d1, shape=True).toarray())
    # scatter acts on the re-rated column: 1 is the controls' shape, 0 the predicted count in every
    # cell, and halfway the spread is half
    g = int(np.argmax(prof.fraction))
    spread = []
    for s in (1.0, 0.5, 0.0):
        sc = np.ones(GS)
        sc[g] = s
        _, comp = _cpm(PoissonEmitter(prof, seed=7, lam=0.5).emit_dual(200, d1, d1, scatter=sc, shape=True))
        spread.append(comp[:, g].std() / comp[:, g].mean())
    assert 0.4 < spread[1] / spread[0] < 0.6 and spread[2] < 0.05


def test_the_rungs_under_shape_are_control_cells_on_the_anchor_rung_and_the_template_on_the_last(tmp_path):
    rng = np.random.default_rng(40)
    X = _cells_like_real(rng, 1500)
    prof = _kept_profile(tmp_path, X, keep_cells=True)
    d = rng.normal(0, 0.3, size=GS)
    # the last rung returns the template as drawn
    far, ref = PoissonEmitter(prof, seed=1, lam=0.5), PoissonEmitter(prof, seed=1, lam=0.5)
    M = far.emit_dual(120, d, 9.0 * d, on_fail="fallback", shape=True)
    assert far.last_dual == "template" and far.last_shaped is False
    assert np.array_equal(M.toarray(), ref.emit(120, d).toarray())
    # the anchor rung re-rates the same control cells for its own pair of profiles: under the default
    # anchor that is the one-amplitude shaped block, bit for bit
    rung, one = PoissonEmitter(prof, seed=1, lam=0.5), PoissonEmitter(prof, seed=1, lam=0.5)
    R = rung.emit_dual(120, d, 9.0 * d, on_fail="anchor", shape=True).toarray()
    assert rung.last_dual == "anchor" and rung.last_shaped is True
    assert np.array_equal(R, one.emit_dual(120, d, d, shape=True).toarray())
    # and under the pooled anchor it is control cells at control depths, on the anchor's two profiles
    pooled = PoissonEmitter(prof, seed=1, lam=0.5, bulk_anchor="pooled")
    _, depth, per_cell, bulk = _moments(pooled.emit_dual(300, d, 9.0 * d, on_fail="anchor", shape=True))
    assert pooled.last_dual == "anchor" and pooled.last_shaped is True
    assert np.abs(per_cell - pooled._fraction(d)).sum() < 0.01 and np.abs(bulk - pooled._fraction(d, bulk=True)).sum() < 0.01
    assert 0.8 < (depth.std() / depth.mean()) / (prof.libsizes.std() / prof.libsizes.mean()) < 1.2


def test_loco_emits_control_cells_on_request_and_keeps_the_runs_draws(monkeypatch, tmp_path):
    import anndata as ad
    import pandas as pd

    from sidechain.data.stream_pseudobulk import PseudobulkSums
    from sidechain.eval import loco

    rng = np.random.default_rng(41)
    X_ctrl = _cells_like_real(rng, 600, g=G)
    genes = np.array([f"g{i}" for i in range(G)], dtype=object)
    basal = X_ctrl.mean(axis=0) + 1.0
    mean = np.stack([basal, basal * np.exp2(rng.normal(0, 0.15, G)), basal * np.exp2(rng.normal(0, 0.15, G))])
    n = np.full(3, 1000, dtype=np.int64)
    src = PseudobulkSums(labels=["ctrl", "g0", "g1"], genes=genes.copy(), count_sum=mean * n[:, None],
                         cpm_sum=mean * n[:, None], cpm_sq_sum=(mean**2 + mean) * n[:, None],
                         n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["t"])
    # the held-out file: the controls, two covered targets, and one no source covers that is emitted FIRST
    real = ad.AnnData(X=sp.csr_matrix(np.vstack([X_ctrl, X_ctrl[:36]])),
                      obs=pd.DataFrame({"perturbation": ["non-targeting"] * len(X_ctrl) + ["a00"] * 12 + ["g0"] * 12 + ["g1"] * 12},
                                       index=[f"c{i}" for i in range(len(X_ctrl) + 36)]),
                      var=pd.DataFrame(index=genes.astype(str)))
    real_path = tmp_path / "real.h5ad"
    real.write_h5ad(real_path)
    drawn = []
    orig = loco.PoissonEmitter.emit
    monkeypatch.setattr(loco.PoissonEmitter, "emit", lambda self, *a, **k: drawn.append(orig(self, *a, **k)) or drawn[-1])
    kw = dict(pert_col="perturbation", control="non-targeting", shrinkage=False, var_floor="poisson",
              emit_lambda=0.5, alpha=1.35, alpha_bulk=1.35, bulk_anchor="pooled", min_libsize=0.0, cells_per_pert=300)
    plain = loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "plain.h5ad", **kw)
    templates, drawn[:] = [t.toarray() for t in drawn], []
    shaped = loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "shaped.h5ad",
                                            emit_shape="controls", **kw)
    assert "emit_shape" not in plain
    unmet = shaped["emit_shape"].pop("genes_left_unmet_a_target")
    assert set(unmet) == {"median", "max"} and 0 <= unmet["median"] <= unmet["max"]      # what the solve left to the fit
    assert shaped["emit_shape"] == {"shape": "controls", "control_cells_kept": len(X_ctrl),
                                    "targets_in_the_controls_shape": 3, "targets_left_on_the_template": [],
                                    "cells_held_to_the_caps": 0, "targets_with_a_held_cell": []}
    # under the pooled anchor the flag moves no draw: every target's template is the run's without it,
    # the targets after an uncovered one included
    assert len(drawn) == 3 and all(np.array_equal(t, s.toarray()) for t, s in zip(templates, drawn))
    a, b = ad.read_h5ad(tmp_path / "plain.h5ad"), ad.read_h5ad(tmp_path / "shaped.h5ad")
    lab = a.obs["perturbation"].to_numpy()
    A, B_ = a.X.toarray().astype(np.float64), b.X.toarray().astype(np.float64)
    cv = lambda C: C.std(axis=0) / np.maximum(C.mean(axis=0), 1e-12)
    comp = lambda M, rows: M[rows] / M[rows].sum(axis=1, keepdims=True)
    busy = X_ctrl.mean(axis=0) > 3
    for t in ("a00", "g0", "g1"):                                    # the uncovered target too
        rows = lab == t
        assert np.median(cv(comp(B_, rows))[busy] / cv(comp(A, rows))[busy]) > 1.5      # real cells' spread
        assert np.abs(comp(B_, rows).mean(axis=0) - comp(A, rows).mean(axis=0)).sum() < 0.03   # the same prediction
        depth = B_[rows].sum(axis=1)
        assert depth.std() / depth.mean() > 2.0 * A[rows].sum(axis=1).std() / A[rows].sum(axis=1).mean()
    # the scatter table composes with it on a covered target, and is not applied to an uncovered one
    top = [f"g{j}" for j in np.argsort(-basal)[:8] if j not in (0, 1)][:2]
    table = tmp_path / "pairs.parquet"
    pd.DataFrame({"target": ["g0", "g0", "a00"], "feature": [top[0], top[1], top[0]], "scatter": [0.0, 0.5, 0.0]}).to_parquet(table)
    both = loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "both.h5ad",
                                          emit_shape="controls", scatter_table=table, **kw)
    assert both["scatter_table"]["targets_carrying_it"] == 1 and both["scatter_table"]["targets_listed_but_uncovered"] == ["a00"]
    c = ad.read_h5ad(tmp_path / "both.h5ad").X.toarray().astype(np.float64)
    assert np.array_equal(B_[lab == "a00"], c[lab == "a00"]) and np.array_equal(B_[lab == "g1"], c[lab == "g1"])
    j0, j1 = int(top[0][1:]), int(top[1][1:])
    g0 = lab == "g0"
    # (at 0 what is left is the depth tilt the pooled anchor asks of the gene)
    assert cv(comp(c, g0))[j0] < 0.35 * cv(comp(B_, g0))[j0] and 0.3 < cv(comp(c, g0))[j1] / cv(comp(B_, g0))[j1] < 0.75
    # under the default anchor a target no source covers is control cells too (it leaves the one-channel path)
    flat = loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "flat.h5ad", emit_shape="controls",
                                          **{**kw, "bulk_anchor": "mean_cpm"})
    assert flat["emit_shape"]["targets_in_the_controls_shape"] == 3
    f = ad.read_h5ad(tmp_path / "flat.h5ad").X.toarray().astype(np.float64)[lab == "a00"]
    assert f.sum(axis=1).std() / f.sum(axis=1).mean() > 0.3
    # refusals: one channel, an unknown shape
    single = {k: v for k, v in kw.items() if k not in ("alpha_bulk", "bulk_anchor")}
    with pytest.raises(SystemExit, match="two-channel"):
        loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "x.h5ad", emit_shape="controls", **single)
    with pytest.raises(SystemExit, match="emit_shape must be"):
        loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "x.h5ad", emit_shape="real", **kw)
