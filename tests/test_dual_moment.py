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
