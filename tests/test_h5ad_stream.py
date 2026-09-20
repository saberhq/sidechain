"""The streamed h5ad writer, and the two mirror paths that were rewritten onto it.

Both `loco.build_transfer_prediction` and `mirror2026.attach_controls` used to hold the whole
matrix: one through `scipy.sparse.vstack`, the other through `anndata.concat`. On the X-Atlas
folds (726 M nonzeros on the truth side, ~800 M on the emitted side) each peaks at roughly twice
its result, which is what kept every arm on those folds box-only. They now append row blocks to
HDF5 by hand.

That is a plumbing change to code whose outputs are recorded numbers, so what these tests pin is
**equivalence, not the plumbing**: the file the streamed path writes must carry the same matrix,
the same obs labels and index, and the same gene axis as the in-memory construction it replaced.
The in-memory forms are written out here longhand rather than imported, so a future edit to
either function cannot quietly move both sides of the comparison.
"""
from __future__ import annotations

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sidechain.eval import loco
from sidechain.eval.mirror2026 import attach_controls
from sidechain.utils.h5ad_stream import (
    CsrWriter,
    load_rows_csr,
    open_anndata_h5,
    write_frame,
)


def _toy(tmp_path, n_ctrl=40, n_pert=25, n_genes=30, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(20, size=(n_ctrl + 2 * n_pert, n_genes)).astype(np.float32))
    obs = pd.DataFrame({"perturbation": ["non-targeting"] * n_ctrl + ["T1"] * n_pert + ["T2"] * n_pert})
    real = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]))
    path = tmp_path / "real.h5ad"
    real.write_h5ad(path)
    return path, real


def test_writer_round_trips_a_matrix_anndata_reads_back(tmp_path):
    rng = np.random.default_rng(1)
    X = sp.csr_matrix((rng.random((11, 7)) < 0.4) * rng.integers(1, 9, (11, 7)).astype(np.float32))
    out = tmp_path / "x.h5ad"
    with open_anndata_h5(out, "w") as h:
        w = CsrWriter(h, X.shape[1], chunk=4)
        for lo in range(0, X.shape[0], 3):
            w.append_csr(X[lo:lo + 3])
        n = w.close()
        write_frame(h, "obs", np.array([f"c{i}" for i in range(n)], dtype=object),
                    {"perturbation": np.array(["p"] * n, dtype=object)})
        write_frame(h, "var", np.array([f"g{i}" for i in range(X.shape[1])], dtype=object), {})
    back = ad.read_h5ad(out)
    assert back.shape == X.shape
    assert (back.X.toarray() == X.toarray()).all()
    assert back.X.dtype == np.float32


def test_load_rows_csr_returns_scattered_rows_in_order(tmp_path):
    rng = np.random.default_rng(2)
    X = sp.csr_matrix((rng.random((20, 6)) < 0.5) * rng.integers(1, 5, (20, 6)).astype(np.float32))
    p = tmp_path / "x.h5ad"
    ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(20)]),
               var=pd.DataFrame(index=[f"g{i}" for i in range(6)])).write_h5ad(p)
    rows = np.array([3, 4, 9, 17])
    with h5py.File(p, "r") as f:
        d, i, ptr = load_rows_csr(f, rows)
    got = sp.csr_matrix((d, i, ptr), shape=(len(rows), 6))
    assert (got.toarray() == X[rows].toarray()).all()


def test_streamed_prediction_matches_the_in_memory_build(tmp_path, monkeypatch):
    """`build_transfer_prediction`'s file, against the `vstack` + AnnData form it replaced."""
    real_path, _ = _toy(tmp_path)
    deltas = {}

    def fake_pool(target, sources, axis, **kw):
        d = np.full(len(axis), 0.25 if target == "T1" else -0.5)
        deltas[target] = d
        return d

    monkeypatch.setattr(loco, "pooled_delta", fake_pool)
    # The toy's control cells are shallower than the project floor, so the floor is
    # named here rather than inherited: this test is about the streamed writer, and
    # the two sides below have to keep the same control pool.
    info = loco.build_transfer_prediction(
        real_path, [], tmp_path / "pred.h5ad", pert_col="perturbation", control="non-targeting",
        dispersion="even", seed=7, min_libsize=0.0,
    )
    got = ad.read_h5ad(tmp_path / "pred.h5ad")

    # the same emitter, the same seed, assembled the old way
    from sidechain.models.count_emitters import ContextProfile, PoissonEmitter
    prof = ContextProfile.from_controls(tmp_path / "pred.controls.h5ad", "real", min_libsize=0.0)
    em = PoissonEmitter(prof, seed=7, dispersion="even")
    axis = got.var_names.astype(str).to_numpy()
    gene_pos = {g: i for i, g in enumerate(axis)}
    blocks, labels = [], []
    for p in ("T1", "T2"):
        d = deltas[p].copy()
        if p in gene_pos:
            d[gene_pos[p]] = -2.32
        blocks.append(em.emit(25, d))
        labels += [p] * 25
    want = ad.AnnData(X=sp.vstack(blocks, format="csr"),
                      obs=pd.DataFrame({"perturbation": labels},
                                       index=[f"pred_{i}" for i in range(len(labels))]),
                      var=pd.DataFrame(index=axis))

    assert (got.X.toarray() == want.X.toarray()).all()
    assert list(got.obs["perturbation"].astype(str)) == list(want.obs["perturbation"])
    assert list(got.obs_names) == list(want.obs_names)
    assert list(got.var_names) == list(want.var_names)
    assert info["cells"] == want.n_obs and info["genes"] == want.n_vars
    assert info["nonzeros"] == want.X.nnz


def test_attach_controls_matches_the_concat_it_replaced(tmp_path):
    real_path, real = _toy(tmp_path, seed=3)
    rng = np.random.default_rng(4)
    pred = ad.AnnData(X=sp.csr_matrix(rng.poisson(5, size=(50, 30)).astype(np.float32)),
                      obs=pd.DataFrame({"perturbation": ["T1"] * 25 + ["T2"] * 25}),
                      var=pd.DataFrame(index=list(real.var_names)))
    pred_path = tmp_path / "pred.h5ad"
    pred.write_h5ad(pred_path)

    out = attach_controls(pred_path, real_path, tmp_path / "with_ctrl.h5ad",
                          pert_col="perturbation", control="non-targeting", block_rows=7)
    got = ad.read_h5ad(out)

    ctrl = real[real.obs["perturbation"].astype(str) == "non-targeting"]
    want_X = sp.vstack([pred.X, ctrl.X], format="csr").toarray()
    assert (got.X.toarray() == want_X).all()
    assert list(got.obs["perturbation"].astype(str)) == ["T1"] * 25 + ["T2"] * 25 + ["non-targeting"] * 40
    assert list(got.obs_names) == ([f"pred_{i}" for i in range(50)] + [f"ctrl_{i}" for i in range(40)])
    assert list(got.var_names) == list(real.var_names)


def test_attach_controls_still_refuses_a_different_gene_axis(tmp_path):
    real_path, _ = _toy(tmp_path, seed=5)
    pred = ad.AnnData(X=sp.csr_matrix(np.ones((4, 30), dtype=np.float32)),
                      obs=pd.DataFrame({"perturbation": ["T1"] * 4}),
                      var=pd.DataFrame(index=[f"H{i}" for i in range(30)]))
    pred.write_h5ad(tmp_path / "pred.h5ad")
    with pytest.raises(ValueError, match="gene axes differ"):
        attach_controls(tmp_path / "pred.h5ad", real_path, tmp_path / "out.h5ad",
                        pert_col="perturbation", control="non-targeting")
