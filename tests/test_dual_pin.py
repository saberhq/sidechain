"""`emit_dual(pin_bulk=True)`, `clamp_to_band`, the shrink rung, and their `eval.loco` flags.

    .venv/bin/python -m pytest -q tests/test_dual_pin.py

Writes temporary files only. Notes: private/code-notes/tests/test_dual_pin.py.md
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import sidechain.models.count_emitters as ce
from sidechain.models.count_emitters import ContextProfile, PoissonEmitter, clamp_to_band, dual_moment_counts

G = 1000
FIT = ce.dual_moment_counts


L4_SHARE = 1.35 / 2.6        # L4's per-cell strength over its summed one


def _profile(rng, g=G, n_ctrl=600):
    frac = rng.gamma(0.5, 1.0, size=g)
    frac /= frac.sum()
    bulk = frac * np.exp(rng.normal(0, 0.05, size=g))
    libs = rng.lognormal(np.log(8000), 0.6, size=n_ctrl).round()
    return ContextProfile(name="A", genes=np.array([f"g{i}" for i in range(g)]), fraction=frac, libsizes=libs,
                          n_cells=n_ctrl, bulk_fraction=bulk / bulk.sum())


def _request(rng, prof, times, moved=150):
    """A summed request on `moved` genes of low share, and a per-cell one `times` L4's per-cell share of it."""
    d_bulk = np.zeros(len(prof.fraction))
    low = np.flatnonzero(prof.fraction < 3 * np.median(prof.fraction))
    at = rng.choice(low, size=moved, replace=False)
    d_bulk[at] = rng.choice([-1.0, 1.0], size=moved) * rng.uniform(0.8, 1.9, size=moved)
    d_cell = d_bulk.copy()
    d_cell[at] *= times * L4_SHARE
    return d_cell, d_bulk


def _moments(M):
    X = M.toarray().astype(np.float64)
    depth = X.sum(axis=1)
    return X, depth, (X / depth[:, None]).mean(axis=0)


@pytest.mark.parametrize("times", [2.0, 3.0, 4.0])
def test_pinned_blocks_hold_the_summed_request_to_one_count_per_gene(times):
    rng = np.random.default_rng(int(times))
    prof = _profile(rng)
    d_cell, d_bulk = _request(rng, prof, times)
    em = PoissonEmitter(prof, seed=1, lam=1.0, bulk_anchor="pooled")
    M = em.emit_dual(400, d_cell, d_bulk, pin_bulk=True)
    X, depth, per_cell = _moments(M)
    p_bulk = em._fraction(d_bulk, bulk=True)
    assert np.array_equal(X, np.round(X)) and X.min() >= 0
    assert np.abs(X.sum(axis=0) - p_bulk * X.sum()).max() <= 1.0
    assert em.last_dual == "dual" and em.last_dual_clamp["depth_moved_max"] < 0.05
    clamped, _, _ = clamp_to_band(em._fraction(d_cell), p_bulk, depth / depth.mean(), 0.01)
    assert np.abs(per_cell - clamped).sum() < 0.01
    if times >= 4.0:
        assert em.last_dual_clamp["genes_clamped"] > 0 and em.last_dual_clamp["cell_l1_moved"] > 0


def test_unpinned_the_same_request_fails_or_leaves_the_summed_profile_off():
    rng = np.random.default_rng(4)
    prof = _profile(rng)
    d_cell, d_bulk = _request(rng, prof, 4.0)
    em = PoissonEmitter(prof, seed=1, lam=1.0, bulk_anchor="pooled")
    try:
        M = em.emit_dual(400, d_cell, d_bulk)
    except ValueError:
        return
    X, _, _ = _moments(M)
    assert np.abs(X.sum(axis=0) - em._fraction(d_bulk, bulk=True) * X.sum()).max() > 1.0


def test_inside_the_band_the_clamp_moves_nothing():
    rng = np.random.default_rng(5)
    prof = _profile(rng)
    d_cell, d_bulk = _request(rng, prof, 1.2)
    em = PoissonEmitter(prof, seed=1, lam=1.0, bulk_anchor="pooled")
    M = em.emit_dual(400, 0.3 * d_cell, 0.3 * d_bulk, pin_bulk=True)
    X, _, per_cell = _moments(M)
    assert em.last_dual_clamp["genes_clamped"] == 0 and em.last_dual_clamp["cell_l1_moved"] < 1e-9
    assert np.abs(per_cell - em._fraction(0.3 * d_cell)).sum() < 0.01
    assert np.abs(X.sum(axis=0) - em._fraction(0.3 * d_bulk, bulk=True) * X.sum()).max() <= 1.0


@pytest.mark.parametrize("shape", [False, True])
def test_off_by_default_and_false_is_the_same_call(tmp_path, shape):
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(6)
    prof = _profile(rng)
    if shape:
        X = rng.poisson(prof.fraction[None, :] * prof.libsizes[:, None]).astype(np.float32)
        a = ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame(index=[f"c{i}" for i in range(len(X))]),
                       var=pd.DataFrame(index=prof.genes))
        a.write_h5ad(tmp_path / "ctrl.h5ad")
        prof = ContextProfile.from_controls(tmp_path / "ctrl.h5ad", "A", keep_cells=True)
    d_cell, d_bulk = _request(rng, prof, 0.5)
    blocks = []
    for kw in ({}, {"pin_bulk": False}, {"pin_bulk": False, "band": 0.2}):
        em = PoissonEmitter(prof, seed=2, lam=0.5, bulk_anchor="pooled")
        blocks.append(em.emit_dual(300, d_cell, d_bulk, on_fail="anchor", shape=shape, **kw))
        assert em.last_dual_clamp is None and em.last_dual_keep is None
    assert all((b != blocks[0]).nnz == 0 for b in blocks[1:])


def test_clamp_to_band_sums_to_one_and_holds_every_ratio_inside_the_band():
    rng = np.random.default_rng(7)
    p = rng.gamma(0.5, 1.0, size=G)
    p /= p.sum()
    bulk = p * np.exp2(rng.normal(0, 2.0, size=G))
    bulk /= bulk.sum()
    z = rng.lognormal(0, 0.6, size=400)
    z /= z.mean()
    out, moved, clipped = clamp_to_band(p, bulk, z, 0.05)
    lo, hi = np.quantile(z, [0.05, 0.95])
    ratio = bulk / out
    assert np.isclose(out.sum(), 1.0, atol=1e-12) and moved > 0 and clipped > 0
    assert ratio.min() >= lo * (1 - 1e-9) and ratio.max() <= hi * (1 + 1e-9)
    same, none, zero = clamp_to_band(p, p, z, 0.05)
    assert np.allclose(same, p, atol=1e-15) and none < 1e-12 and zero == 0
    with pytest.raises(ValueError, match="band"):
        clamp_to_band(p, bulk, z, 0.5)


def test_pin_bulk_refuses_a_block_without_a_depth_spread():
    rng = np.random.default_rng(8)
    template = rng.poisson(3.0, size=(20, 10)).astype(float) + 0.1
    p = np.full(10, 0.1)
    with pytest.raises(ValueError, match="depth spread"):
        dual_moment_counts(template, p, p, depths=np.full(20, 40), seed=0, pin_bulk=True)


def _fail_first(monkeypatch, n):
    """The first `n` fits of each emit_dual call refused as a failed fit."""
    fits = []
    orig = FIT

    def fit(*a, **k):
        fits.append(1)
        if len(fits) <= n:
            raise ValueError("moment fitting failed: per-cell L1 0.01, bulk L1 0.01")
        return orig(*a, **k)
    monkeypatch.setattr(ce, "dual_moment_counts", fit)
    return fits


def test_the_shrink_rung_keeps_the_summed_request_and_halves_the_per_cell_excess(monkeypatch):
    rng = np.random.default_rng(9)
    prof = _profile(rng)
    d_cell, d_bulk = _request(rng, prof, 2.0)
    p_bulk = PoissonEmitter(prof, seed=0, lam=1.0, bulk_anchor="pooled")._fraction(d_bulk, bulk=True)
    fits = _fail_first(monkeypatch, 2)
    em = PoissonEmitter(prof, seed=3, lam=1.0, bulk_anchor="pooled")
    M = em.emit_dual(400, d_cell, d_bulk, on_fail="shrink", pin_bulk=True)
    X, _, per_cell = _moments(M)
    assert (em.last_dual, em.last_dual_reason, em.last_dual_keep) == ("shrink", "fit", 0.25) and len(fits) == 3
    assert em.dual_fallbacks == 1 and em.dual_fallbacks_shrink == 1
    assert np.abs(X.sum(axis=0) - p_bulk * X.sum()).max() <= 1.0
    assert np.abs(per_cell - em._fraction(d_bulk + 0.25 * (d_cell - d_bulk))).sum() < 0.01
    # the anchor rung, on the same refusal, sets the summed profile to the per-cell request
    _fail_first(monkeypatch, 1)
    an = PoissonEmitter(prof, seed=3, lam=1.0, bulk_anchor="pooled")
    Y, _, _ = _moments(an.emit_dual(400, d_cell, d_bulk, on_fail="anchor"))
    assert an.last_dual == "anchor" and np.abs(Y.sum(axis=0) - p_bulk * Y.sum()).max() > 10.0


def test_the_shrink_rung_ends_on_the_template_when_every_try_fails(monkeypatch):
    rng = np.random.default_rng(10)
    prof = _profile(rng)
    d_cell, d_bulk = _request(rng, prof, 2.0)
    fits = _fail_first(monkeypatch, 99)
    em = PoissonEmitter(prof, seed=4, lam=1.0, bulk_anchor="pooled")
    M = em.emit_dual(300, d_cell, d_bulk, on_fail="shrink")
    assert len(fits) == 1 + ce.DUAL_SHRINK_STEPS + 1 and em.last_dual == "template" and em.last_dual_keep is None
    assert (M != PoissonEmitter(prof, seed=4, lam=1.0).emit(300, d_cell)).nnz == 0
    with pytest.raises(ValueError, match="on_fail"):
        em.emit_dual(10, d_cell, d_bulk, on_fail="shrunk")


def test_loco_passes_the_pin_and_the_shrink_rung_and_records_them(monkeypatch, tmp_path):
    import anndata as ad
    import pandas as pd

    from sidechain.data.stream_pseudobulk import PseudobulkSums
    from sidechain.eval import loco

    rng = np.random.default_rng(15)
    g = 60
    genes = np.array([f"g{i}" for i in range(g)], dtype=object)
    basal = rng.uniform(100, 2000, size=g)
    mean = np.stack([basal, basal * np.exp2(rng.normal(0, 0.15, g))])
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
    real.write_h5ad(tmp_path / "real.h5ad")
    kw = dict(pert_col="perturbation", control="non-targeting", shrinkage=False, var_floor="poisson",
              emit_lambda=0.6, alpha=1.2, alpha_bulk=1.6, bulk_anchor="pooled", min_libsize=0.0)
    off = loco.build_transfer_prediction(tmp_path / "real.h5ad", [(src, "ctrl")], tmp_path / "off.h5ad", **kw)
    assert "dual_pin" not in off
    seen = []
    orig = loco.PoissonEmitter.emit_dual
    monkeypatch.setattr(loco.PoissonEmitter, "emit_dual", lambda self, *a, **k: seen.append(k) or orig(self, *a, **k))
    on = loco.build_transfer_prediction(tmp_path / "real.h5ad", [(src, "ctrl")], tmp_path / "on.h5ad",
                                        dual_pin="bulk", dual_band=0.02, **kw)
    assert seen[-1]["pin_bulk"] is True and seen[-1]["band"] == 0.02
    assert on["dual_pin"]["pin"] == "bulk" and set(on["dual_pin"]["targets"]) == {"g0"}
    assert set(on["dual_pin"]["targets"]["g0"]) == {"cell_l1_moved", "genes_clamped", "cell_l1_residual", "depth_moved_max"}
    _fail_first(monkeypatch, 1)
    sh = loco.build_transfer_prediction(tmp_path / "real.h5ad", [(src, "ctrl")], tmp_path / "sh.h5ad",
                                        dual_fallback="shrink", **kw)
    assert seen[-1]["on_fail"] == "shrink" and sh["dual_fallback_targets"]["shrink"] == [["g0", "fit", 0.5]]
    with pytest.raises(SystemExit, match="dual_pin"):
        loco.build_transfer_prediction(tmp_path / "real.h5ad", [(src, "ctrl")], tmp_path / "bad.h5ad",
                                       dual_pin="exact", **kw)
    with pytest.raises(SystemExit, match="two-channel"):
        loco.build_transfer_prediction(tmp_path / "real.h5ad", [(src, "ctrl")], tmp_path / "one.h5ad",
                                       pert_col="perturbation", control="non-targeting", shrinkage=False,
                                       emit_lambda=0.6, alpha=1.2, min_libsize=0.0, dual_pin="bulk")
