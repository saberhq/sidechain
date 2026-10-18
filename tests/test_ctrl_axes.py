"""`--ctrl-axes` (T125): the context's control-cell axes out of the summed-profile channel.

Pinned here: the axes are the centred log1p-CP50K principal axes (against a dense SVD), nested over K and
cached by their inputs; the projection removes eta of the emitted change's along-axis part and holds the
target's own gene; eta 0 and no axes are no-ops; on `submit.build` the per-cell channel is bit-identical with
the flag on while the summed-profile channel moves, the record is written, and off the build writes what it
wrote before (no record, no new keys in args.json); on `eval.loco` the same, and the flag needs the pooled
anchor and refuses a model-named build stem.
"""
from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sidechain.models.count_emitters import PoissonEmitter
from sidechain.submit import build
from sidechain.submit import ctrl_axes as ca

from test_alpha_table import _argv, challenge  # noqa: F401  (the fixture)


def _controls(path, n=400, g=60, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 20, size=g)
    prog = rng.standard_normal((3, g))                          # three planted programmes
    load = rng.standard_normal((n, 3)) * np.array([1.0, 0.6, 0.3])
    rate = base[None, :] * np.exp(0.4 * load @ prog)
    depth = rng.uniform(0.5, 2.0, size=n)[:, None]
    X = rng.poisson(rate * depth * 30).astype(np.float32)
    genes = [f"g{i}" for i in range(g)]
    ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
               var=pd.DataFrame(index=genes)).write_h5ad(path)
    return X, genes


def test_axes_are_the_centred_principal_axes_nested_and_cached(tmp_path):
    X, genes = _controls(tmp_path / "ctrl.h5ad")
    lib = X.sum(axis=1)
    Y = np.log1p(X * (ca.BULK_TARGET_SUM / lib)[:, None])
    Y -= Y.mean(axis=0)
    vt = np.linalg.svd(Y, full_matrices=False)[2]
    axes, rec = ca.control_axes(tmp_path / "ctrl.h5ad", k=5, min_libsize=0, cache_dir=tmp_path / "c")
    assert axes.shape == (5, len(genes))
    assert np.allclose(axes @ axes.T, np.eye(5), atol=1e-10)
    # the same span as the dense SVD's top five
    assert np.allclose(axes.T @ axes, vt[:5].T @ vt[:5], atol=1e-4)
    assert rec["cache"] and (tmp_path / "c" / f"{rec['key']}.npz").exists()
    assert 0 < rec["variance_share_first_axis"] < rec["variance_share_on_axes"] < 1
    again, rec2 = ca.control_axes(tmp_path / "ctrl.h5ad", k=3, min_libsize=0, cache_dir=tmp_path / "c")
    assert rec2["key"] == rec["key"] and np.array_equal(again, axes[:3])          # nested, read from the cache
    other = ca.control_axes(tmp_path / "ctrl.h5ad", k=3, min_libsize=float(np.median(lib)),
                            cache_dir=tmp_path / "c")[1]
    assert other["key"] != rec["key"] and other["control_cells"] < rec["control_cells"]
    rnd, rrec = ca.control_axes(tmp_path / "ctrl.h5ad", k=5, min_libsize=0, basis="random", cache_dir=None)
    assert np.allclose(rnd @ rnd.T, np.eye(5), atol=1e-10)
    assert rrec["variance_share_on_axes"] < rec["variance_share_on_axes"]
    with pytest.raises(ValueError, match="genes differ"):
        ca.control_axes(tmp_path / "ctrl.h5ad", k=3, min_libsize=0, cache_dir=None, genes=genes[::-1])


def _toy(g=80, k=4, seed=1):
    rng = np.random.default_rng(seed)
    anchor = rng.uniform(1, 100, size=g)
    anchor[3] = 0.0                                             # a gene the controls never show
    anchor /= anchor.sum()
    axes = np.linalg.qr(rng.standard_normal((g, k)))[0].T
    l = rng.standard_normal(g) * 0.8
    return anchor, axes, l


@pytest.mark.parametrize("eta", [1.0, 0.5])
def test_the_projection_removes_eta_of_the_emitted_along_axis_change(eta):
    anchor, axes, l = _toy()
    held = 7
    l[held] = -2.32
    out, rec = ca.deflate_bulk(l, anchor, axes, eta, held=(held,))
    free = anchor > 0
    free[held] = False
    V = axes * free[None, :]
    before, after = V @ ca.scorer_change(l, anchor), V @ ca.scorer_change(out, anchor)
    assert np.linalg.norm(before) > 0.1                           # the contrast is not vacuous
    assert np.allclose(after, (1 - eta) * before, atol=1e-6 * np.linalg.norm(before))
    assert rec["residual"] <= ca.TOLERANCE and 0 < rec["share_on_axes"] < 1
    assert out[held] == l[held] and out[3] == l[3]                # the pin held, the unseen gene untouched
    assert not np.array_equal(out, l)


def test_eta_zero_and_no_axes_are_no_ops_and_the_floor_holds():
    anchor, axes, l = _toy()
    assert np.array_equal(ca.deflate_bulk(l, anchor, axes, 0.0)[0], l)
    assert np.array_equal(ca.deflate_bulk(l, anchor, axes[:0], 1.0)[0], l)
    with pytest.raises(ValueError, match="eta"):
        ca.deflate_bulk(l, anchor, axes, 1.5)
    # a change the projection would push below zero counts is floored, never sent as minus infinity
    big = l.copy()
    big[np.argmax(np.abs(axes[0]))] = 12.0 * np.sign(axes[0][np.argmax(np.abs(axes[0]))])
    out, rec = ca.deflate_bulk(big, anchor, axes, 1.0)
    assert np.isfinite(out[anchor > 0]).all() and out.min() >= min(big.min(), ca.LOG2FC_FLOOR)


# ---------------------------------------------------------------------------------- submit.build

def _spy(monkeypatch):
    seen = {}

    def spy(self, n, log2fc_cell, log2fc_bulk, **kw):
        seen.setdefault(PoissonEmitter.last_stem, []).append((np.array(log2fc_cell), np.array(log2fc_bulk)))
        return self.emit(n, log2fc_cell)
    monkeypatch.setattr(PoissonEmitter, "emit_dual", spy)
    return seen


def test_build_moves_only_the_summed_profile_and_records_it(challenge, monkeypatch, tmp_path):  # noqa: F811
    seen = _spy(monkeypatch)
    pooled = ["--bulk-anchor", "pooled"]
    PoissonEmitter.last_stem = "plain"
    assert build.main(_argv(challenge, "plain", pooled)) == 0
    PoissonEmitter.last_stem = "axes"
    assert build.main(_argv(challenge, "axes", pooled + ["--ctrl-axes", "1", "--ctrl-axes-cache",
                                                          str(tmp_path / "c")])) == 0
    assert len(seen["plain"]) == len(seen["axes"]) == 2
    for (c0, b0), (c1, b1) in zip(seen["plain"], seen["axes"]):
        assert np.array_equal(c0, c1)                             # the per-cell channel, bit-identical
        assert not np.array_equal(b0, b1)                         # the summed profile moved
    assert seen["axes"][0][1][0] == seen["plain"][0][1][0]          # A's own gene keeps its pin
    rec = json.loads((challenge["out"] / "axes.ctrl_axes.json").read_text())
    assert rec["k"] == 1 and rec["contexts"]["X"]["deflation"]["targets"] == 2
    assert rec["contexts"]["X"]["axes"]["key"] and "dual_fallbacks" in rec
    assert not (challenge["out"] / "plain.ctrl_axes.json").exists()
    plain_args = json.loads((challenge["out"] / "plain.args.json").read_text())
    assert not any(k.startswith("ctrl_axes") for k in plain_args)           # off: the record as before
    assert json.loads((challenge["out"] / "axes.args.json").read_text())["ctrl_axes"] == 1


def test_build_refuses_without_the_pooled_anchor_and_under_a_model_name(challenge):  # noqa: F811
    with pytest.raises(SystemExit):
        build.main(_argv(challenge, "x", ["--ctrl-axes", "1"]))
    with pytest.raises(SystemExit):
        build.main(_argv(challenge, "x", ["--bulk-anchor", "pooled", "--ctrl-axes-eta", "0.5"]))
    with pytest.raises(SystemExit):
        build.main(_argv(challenge, "ser-99a_delta_test_v1", ["--bulk-anchor", "pooled", "--ctrl-axes", "1"]))


# ------------------------------------------------------------------------------------ eval.loco

def test_loco_moves_only_the_summed_profile_and_off_is_the_historical_path(tmp_path, monkeypatch):
    from sidechain.data.stream_pseudobulk import PseudobulkSums
    from sidechain.eval import loco

    genes = np.array([f"g{i}" for i in range(30)], dtype=object)
    targets = [f"g{i}" for i in range(4)]

    def line(seed):
        r = np.random.default_rng(seed)
        labels = ["ctrl"] + targets
        mean = r.uniform(100, 2000, size=(len(labels), len(genes)))
        n = np.full(len(labels), 2000, dtype=np.int64)
        return (PseudobulkSums(labels=labels, genes=genes.copy(), count_sum=mean * n[:, None],
                               cpm_sum=mean * n[:, None], cpm_sq_sum=(mean**2 + mean) * n[:, None],
                               n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["test"]), "ctrl")
    srcs = [line(1), line(2)]
    rng = np.random.default_rng(3)
    basal = rng.uniform(100, 2000, size=len(genes))
    rows, labels = [], []
    for i in range(60):
        rows.append(rng.poisson(basal * (20 + i) / basal.sum() * 1000)); labels.append("non-targeting")
    for t in targets[:2]:
        for _ in range(8):
            rows.append(rng.poisson(basal * 40 / basal.sum() * 1000)); labels.append(t)
    real = ad.AnnData(X=sp.csr_matrix(np.asarray(rows, dtype=np.float32)),
                      obs=pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(len(rows))]),
                      var=pd.DataFrame(index=genes.astype(str)))
    real.write_h5ad(tmp_path / "real.h5ad")
    kw = dict(pert_col="perturbation", control="non-targeting", shrinkage=False, var_floor="poisson",
              min_libsize=0.0, seed=0, alpha=1.35, emit_lambda=0.5, bulk_anchor="pooled")
    off = loco.build_transfer_prediction(tmp_path / "real.h5ad", srcs, tmp_path / "off.h5ad", **kw)
    zero = loco.build_transfer_prediction(tmp_path / "real.h5ad", srcs, tmp_path / "zero.h5ad", ctrl_axes=0, **kw)
    assert "ctrl_axes" not in off and "ctrl_axes" not in zero
    a, b = ad.read_h5ad(tmp_path / "off.h5ad"), ad.read_h5ad(tmp_path / "zero.h5ad")
    assert (a.X != b.X).nnz == 0
    seen = {}

    def spy(self, n, log2fc_cell, log2fc_bulk, **kw):
        seen.setdefault(PoissonEmitter.last_stem, []).append((np.array(log2fc_cell), np.array(log2fc_bulk)))
        return self.emit(n, log2fc_cell)
    monkeypatch.setattr(PoissonEmitter, "emit_dual", spy)
    PoissonEmitter.last_stem = "plain"
    loco.build_transfer_prediction(tmp_path / "real.h5ad", srcs, tmp_path / "p.h5ad", **kw)
    PoissonEmitter.last_stem = "axes"
    on = loco.build_transfer_prediction(tmp_path / "real.h5ad", srcs, tmp_path / "q.h5ad", ctrl_axes=2,
                                        ctrl_axes_cache=None, **kw)
    assert on["ctrl_axes"]["k"] == 2 and on["ctrl_axes"]["deflation"]["targets"] == 2
    for (c0, b0), (c1, b1) in zip(seen["plain"], seen["axes"]):
        assert np.array_equal(c0, c1) and not np.array_equal(b0, b1)
    PoissonEmitter.last_stem = "half"
    half = loco.build_transfer_prediction(tmp_path / "real.h5ad", srcs, tmp_path / "h.h5ad", ctrl_axes=2,
                                          ctrl_axes_share=0.5, ctrl_axes_cache=None, **kw)
    assert half["ctrl_axes"]["deflation"]["targets"] == 1
    assert half["ctrl_axes"]["deflation"]["targets_left_as_they_were"] == 1
    assert half["covered_by_sources"] == on["covered_by_sources"] == 2
    pairs = list(zip(seen["plain"], seen["half"]))
    assert all(np.array_equal(c0, c1) for (c0, _), (c1, _) in pairs)
    assert sum(not np.array_equal(b0, b1) for (_, b0), (_, b1) in pairs) == 1
    with pytest.raises(SystemExit, match="pooled"):
        loco.build_transfer_prediction(tmp_path / "real.h5ad", srcs, tmp_path / "r.h5ad", ctrl_axes=2,
                                       ctrl_axes_cache=None, **{**kw, "bulk_anchor": "mean_cpm",
                                                                "alpha_bulk": 1.35})


def test_the_weakest_share_deflates_only_the_smallest_changes(challenge, monkeypatch, tmp_path):  # noqa: F811
    assert ca.weakest({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, 0.5) == {"a", "b"}
    assert ca.weakest({"a": 1.0, "b": 2.0}, 1.0) == {"a", "b"}
    seen = _spy(monkeypatch)
    pooled = ["--bulk-anchor", "pooled", "--ctrl-axes-cache", str(tmp_path / "c")]
    PoissonEmitter.last_stem = "plain"
    assert build.main(_argv(challenge, "plain", pooled)) == 0
    PoissonEmitter.last_stem = "half"
    assert build.main(_argv(challenge, "half", pooled + ["--ctrl-axes", "1", "--ctrl-axes-share", "0.5"])) == 0
    moved = [not np.array_equal(b0, b1) for (_, b0), (_, b1) in zip(seen["plain"], seen["half"])]
    assert sum(moved) == 1
    rec = json.loads((challenge["out"] / "half.ctrl_axes.json").read_text())["contexts"]["X"]["deflation"]
    assert rec["targets"] == 1 and rec["targets_left_as_they_were"] == 1 and rec["share"] == 0.5
