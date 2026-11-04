"""`sidechain.models.emit_pool` and `--emit-workers` in `eval.loco`: the pooled emission equals the serial one.

    .venv/bin/python -m pytest -q tests/test_emit_pool.py

Writes temporary files only. Notes: private/code-notes/tests/test_emit_pool.py.md
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.eval import loco
from sidechain.models.count_emitters import ContextProfile, PoissonEmitter
from sidechain.models.emit_pool import COUNTERS, EmitPool

G, N_LAB = 120, 12
KW = dict(pert_col="perturbation", control="non-targeting", var_floor="poisson", emit_lambda=0.4,
          alpha=1.0, bulk_anchor="pooled", min_libsize=0.0, shrink_stage="pooled", shrink_rule="adaptive",
          emit_shape="controls")


@pytest.fixture
def fold(tmp_path):
    rng = np.random.default_rng(5)
    genes = np.array([f"g{i}" for i in range(G)], dtype=object)
    labs = [f"g{i}" for i in range(N_LAB)]
    basal = rng.uniform(50, 3000, size=G)
    mean = np.stack([basal] + [basal * np.exp2(rng.normal(0, 0.8, G)) for _ in labs])
    n = np.full(len(labs) + 1, 300, dtype=np.int64)
    src = PseudobulkSums(labels=["ctrl", *labs], genes=genes.copy(), count_sum=mean * n[:, None],
                         cpm_sum=mean * n[:, None], cpm_sq_sum=(mean**2 + mean) * n[:, None],
                         n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["t"])
    rows, obs = [], []
    for lab, k in [("non-targeting", 120)] + [(x, 25) for x in labs + ["unseen1", "unseen2"]]:
        for _ in range(k):
            rows.append(rng.poisson(basal / basal.sum() * rng.integers(2000, 9000)))
            obs.append(lab)
    real = ad.AnnData(X=sp.csr_matrix(np.asarray(rows, dtype=np.float32)),
                      obs=pd.DataFrame({"perturbation": obs}, index=[f"c{i}" for i in range(len(rows))]),
                      var=pd.DataFrame(index=genes.astype(str)))
    real.write_h5ad(tmp_path / "real.h5ad")
    return tmp_path, tmp_path / "real.h5ad", [(src, "ctrl")]


def _same_file(a, b):
    x, y = ad.read_h5ad(a), ad.read_h5ad(b)
    assert x.X.dtype == y.X.dtype and x.X.shape == y.X.shape
    assert np.array_equal(x.X.indptr, y.X.indptr) and np.array_equal(x.X.indices, y.X.indices)
    assert np.array_equal(x.X.data, y.X.data)
    assert x.obs.equals(y.obs) and list(x.obs_names) == list(y.obs_names)
    assert list(x.var_names) == list(y.var_names)


@pytest.mark.parametrize("extra", [
    dict(alpha_bulk=6.0, alpha_bulk_target=0.5, dual_fallback="anchor"),
    dict(alpha_bulk=6.0, alpha_bulk_target=0.5, dual_fallback="shrink", dual_pin="bulk"),
    dict(alpha_bulk=6.0, dual_fallback="template"),
    dict(alpha_bulk=2.0, dual_fallback="anchor", emit_shape="template"),
])
def test_the_pooled_prediction_is_the_serial_prediction(fold, extra):
    tmp, real, sources = fold
    one = loco.build_transfer_prediction(real, sources, tmp / "one.h5ad", **{**KW, **extra}, seed=3)
    three = loco.build_transfer_prediction(real, sources, tmp / "three.h5ad", **{**KW, **extra}, seed=3,
                                           emit_workers=3)
    _same_file(tmp / "one.h5ad", tmp / "three.h5ad")
    assert three.pop("emit_workers") == 3 and "emit_workers" not in one
    one.pop("pred"), three.pop("pred")
    assert one == three                                           # every record, the knee's reads included


def test_the_fixture_reaches_the_fallback_rungs(fold):
    tmp, real, sources = fold
    info = loco.build_transfer_prediction(real, sources, tmp / "f.h5ad", **KW, alpha_bulk=6.0,
                                          dual_fallback="anchor", seed=3, emit_workers=2)
    assert info["dual_fallbacks"] > 0


def _profile(rng, g=30, n=80):
    cells = sp.csr_matrix(rng.poisson(rng.gamma(0.6, 3.0, size=g)[None, :]
                                      * rng.uniform(4, 12, size=(n, 1))).astype(np.float32))
    lib = np.asarray(cells.sum(axis=1)).ravel()
    frac = np.asarray(cells.sum(axis=0)).ravel()
    frac = frac / frac.sum()
    return ContextProfile(name="t", genes=np.array([f"g{i}" for i in range(g)]), fraction=frac, libsizes=lib,
                          n_cells=n, bulk_fraction=frac.copy())


def _calls(rng, g, k=10):
    out = []
    for i in range(k):
        d = rng.normal(0, 1.5, g)
        out.append(("emit", 15, d) if i % 4 == 3 else ("dual", 15, d, d * 2.5, {}))
    return out


def test_the_pool_yields_the_serial_blocks_counters_and_stream():
    rng = np.random.default_rng(0)
    prof = _profile(rng)
    calls = _calls(rng, len(prof.genes))
    kw = dict(on_fail="anchor")
    serial = PoissonEmitter(prof, seed=11, lam=0.5, bulk_anchor="pooled")
    want = [serial.emit(c[1], c[2]) if c[0] == "emit" else serial.emit_dual(c[1], c[2], c[3], **kw) for c in calls]
    em = PoissonEmitter(prof, seed=11, lam=0.5, bulk_anchor="pooled")
    with EmitPool(em, 3, **kw) as pool:
        got = list(pool.blocks(calls))
        walked = pool._walk
        again = list(pool.blocks(calls, PoissonEmitter(prof, seed=11, lam=0.5, bulk_anchor="pooled")))
        assert pool._walk is walked                               # the same start and calls reuse the walk
    for w, a, b in zip(want, got, again):
        assert (w != a).nnz == 0 and (w != b).nnz == 0 and w.dtype == a.dtype
    assert {c: getattr(em, c, 0) for c in COUNTERS} == {c: getattr(serial, c, 0) for c in COUNTERS}
    assert em.rng.bit_generator.state == serial.rng.bit_generator.state


def test_a_call_that_draws_off_the_walk_is_refused(monkeypatch):
    rng = np.random.default_rng(1)
    prof = _profile(rng)
    calls = [c for c in _calls(rng, len(prof.genes)) if c[0] == "dual"]
    real = PoissonEmitter.emit_dual

    def extra_draw(self, *a, **k):
        self.rng.random()
        return real(self, *a, **k)

    monkeypatch.setattr(PoissonEmitter, "emit_dual", extra_draw)
    with EmitPool(PoissonEmitter(prof, seed=2, lam=0.5, bulk_anchor="pooled"), 2, on_fail="anchor") as pool:
        with pytest.raises(RuntimeError, match="off the serial walk"):
            list(pool.blocks(calls))


def test_one_worker_is_refused():
    with pytest.raises(ValueError, match="2 or more"):
        EmitPool(None, 1)
