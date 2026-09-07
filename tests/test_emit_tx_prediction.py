"""Contract tests for scripts/emit_tx_prediction.py — the count space and the CSR writer.

Two things here can be wrong without anything crashing, which is the whole reason for the file.

**The count space.** A model trained on `prep_tx_training.py --log1p` emits shifted logarithms;
cell-eval2 scores against raw counts. `test_inverse_round_trips_prep_tx_training` asserts the
undo is exact against the prep script's own transform rather than against a re-derivation of it,
so the two cannot drift apart. And `test_rescale_bounds_a_cell_that_inverse_does_not` pins the
measurement that made `rescale` the default: `expm1` of an unbounded prediction is unbounded, and
on real PHE-2 output that produced single cells of 2.4e10 counts, each one enough to *be* its
perturbation's pseudobulk.

**The file.** The writer appends CSR blocks to HDF5 by hand, because holding the matrix would
cost 22 GB. `test_written_file_reads_back_identically` is what says the hand-written layout is
the one AnnData reads.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.sparse as sp

_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


emit = _load("emit_tx_prediction")
prep = _load("prep_tx_training")


def counts(n_cells=30, n_genes=18, seed=0):
    rng = np.random.default_rng(seed)
    depth = rng.integers(3, 40, size=(n_cells, 1))
    dense = rng.poisson(rng.random((n_cells, n_genes)) * depth).astype(np.float32)
    dense[:, 0] = 0.0
    dense[0] = np.maximum(dense[0], 1.0)  # no empty first row, so indptr[1] > 0
    return sp.csr_matrix(dense)


# --------------------------------------------------------------------- the count space


def test_inverse_round_trips_prep_tx_training(tmp_path):
    """`--x-space log1p --depth inverse` is the exact undo of `prep_tx_training.py --log1p`.

    Asserted against the prep script itself, not against a second implementation of the same
    formula — the two live in different files and would otherwise be free to drift.
    """
    import anndata as ad

    X = counts()
    ad.AnnData(X=X).write_h5ad(tmp_path / "raw.h5ad")
    with h5py.File(tmp_path / "raw.h5ad", "r+") as h:
        stats = prep.shifted_log(h, None, block_nnz=13)
    logged = ad.read_h5ad(tmp_path / "raw.h5ad").X.toarray()
    depths = np.asarray(X.sum(axis=1)).ravel()

    back = emit.to_counts(logged, depths, stats["target_sum"], "log1p", "inverse")
    assert np.allclose(back, X.toarray(), atol=1e-3)


def test_counts_space_is_a_passthrough():
    """A model trained on raw counts emits raw counts; nothing should touch them."""
    preds = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    got = emit.to_counts(preds, np.array([100.0, 200.0]), 150.0, "counts", "rescale")
    assert np.array_equal(got, preds)


def test_rescale_pins_every_cell_to_its_basal_depth():
    preds = np.log1p(np.array([[10.0, 90.0], [500.0, 500.0]], dtype=np.float32))
    depths = np.array([1_000.0, 4_000.0])
    got = emit.to_counts(preds, depths, 100.0, "log1p", "rescale")
    assert np.allclose(got.sum(1), depths)
    # the composition is untouched: cell 0 stays 10:90
    assert np.allclose(got[0] / got[0].sum(), [0.1, 0.9])


def test_size_factor_mode_leaves_every_cell_on_l():
    """No depth variation at all — the shape the earlier diagnostic reported, kept measurable."""
    preds = np.log1p(np.array([[10.0, 90.0], [40.0, 60.0]], dtype=np.float32))
    got = emit.to_counts(preds, np.array([1.0, 99_999.0]), 100.0, "log1p", "size-factor")
    assert np.allclose(got.sum(1), [100.0, 100.0])


def test_rescale_bounds_a_cell_that_inverse_does_not():
    """The measurement that made `rescale` the default.

    One log-space value of 24 is 2.6e10 counts under the literal inverse — on real PHE-2 output
    that happened in 588 of 122,046 cells, and a pseudobulk is a sum over cells, so one such cell
    *is* that perturbation's profile. `rescale` keeps the same (bad) composition but cannot let
    the cell outweigh its neighbours.
    """
    preds = np.array([[24.0, 1.0, 1.0]], dtype=np.float32)
    depths = np.array([10_000.0])
    blown = emit.to_counts(preds, depths, 10_000.0, "log1p", "inverse")
    assert blown.sum() > 1e10
    tamed = emit.to_counts(preds, depths, 10_000.0, "log1p", "rescale")
    assert np.isclose(tamed.sum(), 10_000.0)


# ------------------------------------------------------------------------- the emission


def test_round_is_deterministic_and_poisson_is_not():
    counts_ = np.array([[0.4, 1.6, 2.5]], dtype=np.float32)
    rng = np.random.default_rng(0)
    assert np.array_equal(emit.emit(counts_, "round", rng), np.array([[0.0, 2.0, 2.0]]))
    a = emit.emit(np.full((200, 3), 5.0), "poisson", np.random.default_rng(1))
    assert a.std() > 0


def test_emission_clips_negatives():
    """`predict_step` is softplussed, but a negative would become a negative count silently."""
    got = emit.emit(np.array([[-3.0, 2.4]], dtype=np.float32), "round", np.random.default_rng(0))
    assert got.min() >= 0


# ---------------------------------------------------------------------------- the file


def test_written_file_reads_back_identically(tmp_path):
    import anndata as ad

    X = counts(n_cells=12, n_genes=7)
    labels = ["A"] * 5 + ["B"] * 4 + ["ctrl"] * 3
    with h5py.File(tmp_path / "out.h5ad", "w") as h:
        h.attrs["encoding-type"] = "anndata"
        h.attrs["encoding-version"] = "0.1.0"
        w = emit.CsrWriter(h, X.shape[1], chunk=4)
        for lo in range(0, X.shape[0], 5):  # several appends, so the offsets are exercised
            blk = X[lo:lo + 5]
            w.append(blk.data, blk.indices, blk.indptr)
        n = w.close()
        emit.write_frame(h, "obs", np.array([f"c{i}" for i in range(n)], dtype=object),
                         {"perturbation": np.array(labels, dtype=object)})
        emit.write_frame(h, "var", np.array([f"g{i}" for i in range(X.shape[1])], dtype=object), {})

    a = ad.read_h5ad(tmp_path / "out.h5ad")
    assert a.shape == X.shape
    assert np.array_equal(a.X.toarray(), X.toarray())
    assert list(a.obs["perturbation"]) == labels
    assert list(a.var_names) == [f"g{i}" for i in range(X.shape[1])]


def test_writer_keeps_an_empty_row_empty(tmp_path):
    """An all-zero predicted cell is a hole in indptr, not a copy of its neighbour."""
    import anndata as ad

    dense = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    X = sp.csr_matrix(dense)
    with h5py.File(tmp_path / "out.h5ad", "w") as h:
        h.attrs["encoding-type"] = "anndata"
        h.attrs["encoding-version"] = "0.1.0"
        w = emit.CsrWriter(h, 2, chunk=2)
        w.append(X.data, X.indices, X.indptr)
        n = w.close()
        emit.write_frame(h, "obs", np.array([f"c{i}" for i in range(n)], dtype=object),
                         {"perturbation": np.array(["A"] * n, dtype=object)})
        emit.write_frame(h, "var", np.array(["g0", "g1"], dtype=object), {})
    assert np.array_equal(ad.read_h5ad(tmp_path / "out.h5ad").X.toarray(), dense)


def test_load_rows_csr_returns_the_rows_in_the_order_asked(tmp_path):
    """The basal pool is sampled, so the row order is not the file's order."""
    import anndata as ad

    X = counts(n_cells=9, n_genes=5)
    ad.AnnData(X=X).write_h5ad(tmp_path / "a.h5ad")
    rows = np.array([7, 0, 3, 3])
    with h5py.File(tmp_path / "a.h5ad", "r") as h:
        d, i, p = emit.load_rows_csr(h, rows)
    got = sp.csr_matrix((d, i, p), shape=(len(rows), X.shape[1])).toarray()
    assert np.array_equal(got, X.toarray()[rows])


# ------------------------------------------------------------------------------ the CLI


def _fake_fold(tmp_path, name, labels, n_genes=6, seed=0):
    """A raw-count fold h5ad with the two obs columns the real ones carry."""
    import anndata as ad
    import pandas as pd

    X = counts(n_cells=len(labels), n_genes=n_genes, seed=seed)
    obs = pd.DataFrame(
        {"perturbation": [lab.lower() if lab == "Non-Targeting" else lab for lab in labels],
         "gene_target": labels,
         "total_counts": np.asarray(X.sum(axis=1)).ravel().astype(float)},
        index=[f"cell_{i}" for i in range(len(labels))])
    obs["perturbation"] = obs["perturbation"].replace({"non-targeting": "non-targeting"})
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    a.write_h5ad(tmp_path / name)
    return tmp_path / name


def test_a_target_missing_from_the_model_map_is_refused(tmp_path, monkeypatch):
    """`state tx infer` substitutes the CONTROL one-hot for an unknown perturbation and warns
    only when not quiet — a silent wrong answer. Refuse instead."""
    import pickle

    import torch

    labels = ["Non-Targeting"] * 3 + ["AAA"] * 2 + ["BBB"] * 2
    fold = _fake_fold(tmp_path, "fold.h5ad", labels)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    torch.save({"Non-Targeting": torch.eye(2)[0], "AAA": torch.eye(2)[1]},
               model_dir / "pert_onehot_map.pt")  # BBB is missing
    with open(model_dir / "var_dims.pkl", "wb") as f:
        pickle.dump({"input_dim": 6}, f)

    with pytest.raises(SystemExit, match="not in the model's map"):
        emit.main(["--model-dir", str(model_dir), "--checkpoint", str(tmp_path / "x.ckpt"),
                   "--basal", str(fold), "--truth", str(fold), "--out", str(tmp_path / "o.h5ad")])


def test_string_datasets_carry_anndata_encoding_metadata(tmp_path):
    """Without these two attributes anndata reads the file through its legacy path and warns;
    a later release is free to drop that path."""
    with h5py.File(tmp_path / "out.h5ad", "w") as h:
        emit.write_frame(h, "obs", np.array(["c0", "c1"], dtype=object),
                         {"perturbation": np.array(["A", "B"], dtype=object)})
        assert h["obs/_index"].attrs["encoding-type"] == "string-array"
        assert h["obs/perturbation/categories"].attrs["encoding-type"] == "string-array"
        assert h["obs/perturbation/codes"].attrs["encoding-type"] == "array"
