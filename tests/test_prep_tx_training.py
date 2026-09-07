"""Contract tests for scripts/prep_tx_training.py --log1p — the shifted logarithm.

The claim the change rests on is `test_matches_scanpy_exactly`: the streamed rewrite is
`sc.pp.normalize_total(target_sum=None)` + `sc.pp.log1p`, value for value, on a matrix scanpy
never sees. It exists streamed because the fold files are 1.5-2.4 GB of CSR on a 17 GB Mac, and
a rewrite that is only *approximately* scanpy is a silent divergence from the recipe Arc's own
training file uses.

The other thing worth a test of its own is the size factor. `target_sum=None` means the
dataset's **median raw count depth**, not CP10k and not CPM, and the two are not close: a
median depth of 17,138 against 10,000 is a 1.7x difference in every logged value.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

_SPEC = importlib.util.spec_from_file_location(
    "prep_tx_training", Path(__file__).resolve().parent.parent / "scripts" / "prep_tx_training.py"
)
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)


def counts(n_cells=40, n_genes=25, seed=0):
    """A CSR count matrix with an uneven depth distribution, so the size factor matters."""
    rng = np.random.default_rng(seed)
    depth = rng.integers(1, 12, size=(n_cells, 1))
    dense = rng.poisson(rng.random((n_cells, n_genes)) * depth).astype(np.float32)
    dense[:, 0] = 0.0  # a gene nobody measured, to prove structural zeros stay zero
    return sp.csr_matrix(dense)


def write_h5ad(path, X):
    import anndata as ad

    ad.AnnData(X=X).write_h5ad(path)
    return path


# ------------------------------------------------------------------------ the row totals


def test_row_totals_matches_a_dense_sum(tmp_path):
    import h5py

    X = counts()
    write_h5ad(tmp_path / "a.h5ad", X)
    with h5py.File(tmp_path / "a.h5ad", "r") as h:
        got = prep.row_totals(h, block_nnz=17)  # a block size that splits mid-matrix
    assert np.allclose(got, np.asarray(X.sum(axis=1)).ravel())


def test_row_totals_handles_a_cell_with_no_counts(tmp_path):
    """An empty row is a hole in indptr; the cumulative-sum form must return 0, not the
    neighbouring row's total."""
    import h5py

    X = counts().tolil()
    X[3, :] = 0
    X[0, :] = 0
    X = X.tocsr()
    write_h5ad(tmp_path / "a.h5ad", X)
    with h5py.File(tmp_path / "a.h5ad", "r") as h:
        got = prep.row_totals(h, block_nnz=5)
    assert got[0] == 0.0 and got[3] == 0.0
    assert np.allclose(got, np.asarray(X.sum(axis=1)).ravel())


# --------------------------------------------------------------- the transform itself


def test_matches_scanpy_exactly(tmp_path):
    """The whole point: streamed, in blocks, on h5py — and identical to the library call."""
    import anndata as ad
    import h5py
    import scanpy as sc

    X = counts(n_cells=60, n_genes=30, seed=7)
    write_h5ad(tmp_path / "a.h5ad", X)

    want = ad.AnnData(X=X.copy())
    sc.pp.normalize_total(want, target_sum=None)
    sc.pp.log1p(want)

    with h5py.File(tmp_path / "a.h5ad", "a") as h:
        stats = prep.shifted_log(h, None, block_nnz=13)
    got = ad.read_h5ad(tmp_path / "a.h5ad")

    assert np.allclose(got.X.toarray(), want.X.toarray(), atol=1e-6)
    assert stats["target_sum"] == pytest.approx(np.median(np.asarray(X.sum(1)).ravel()))


def test_the_size_factor_is_the_median_depth_not_cp10k(tmp_path):
    import h5py

    X = counts(n_cells=50, seed=3)
    write_h5ad(tmp_path / "a.h5ad", X)
    with h5py.File(tmp_path / "a.h5ad", "a") as h:
        stats = prep.shifted_log(h, None, block_nnz=1_000)
    depths = np.asarray(X.sum(1)).ravel()
    assert stats["target_sum"] == pytest.approx(np.median(depths))
    assert stats["target_sum"] != 1e4
    assert stats["depth_min"] == depths.min() and stats["depth_max"] == depths.max()


def test_an_explicit_target_sum_is_honoured(tmp_path):
    """CP10k stays reachable, but only by asking for it."""
    import anndata as ad
    import h5py
    import scanpy as sc

    X = counts(seed=11)
    write_h5ad(tmp_path / "a.h5ad", X)
    want = ad.AnnData(X=X.copy())
    sc.pp.normalize_total(want, target_sum=1e4)
    sc.pp.log1p(want)
    with h5py.File(tmp_path / "a.h5ad", "a") as h:
        stats = prep.shifted_log(h, 1e4, block_nnz=1_000)
    assert stats["target_sum"] == 1e4
    assert np.allclose(ad.read_h5ad(tmp_path / "a.h5ad").X.toarray(), want.X.toarray(), atol=1e-6)


def test_structural_zeros_stay_zero(tmp_path):
    """Scaling a row and log1p both map 0 to 0, which is why the sparsity never changes and
    the rewrite can be done in place on X/data alone."""
    import anndata as ad
    import h5py

    X = counts()
    write_h5ad(tmp_path / "a.h5ad", X)
    with h5py.File(tmp_path / "a.h5ad", "a") as h:
        prep.shifted_log(h, None, block_nnz=29)
    got = ad.read_h5ad(tmp_path / "a.h5ad")
    assert got.X.nnz == X.nnz
    assert (got.X.toarray()[:, 0] == 0).all()


def test_values_stop_being_integral(tmp_path):
    import anndata as ad
    import h5py

    X = counts()
    write_h5ad(tmp_path / "a.h5ad", X)
    with h5py.File(tmp_path / "a.h5ad", "a") as h:
        prep.shifted_log(h, None, block_nnz=29)
    data = ad.read_h5ad(tmp_path / "a.h5ad").X.data
    assert (data != np.rint(data)).mean() > 0.5


# -------------------------------------------------------------------- what it refuses


def test_uns_log1p_is_written_in_the_shape_anndata_reads(tmp_path):
    """cell_load only checks the key's presence, but anndata still has to read the file."""
    import anndata as ad
    import h5py

    write_h5ad(tmp_path / "a.h5ad", counts())
    with h5py.File(tmp_path / "a.h5ad", "a") as h:
        prep.shifted_log(h, None)
    with h5py.File(tmp_path / "a.h5ad", "r") as h:
        assert "log1p" in h["uns"]
    assert ad.read_h5ad(tmp_path / "a.h5ad").uns["log1p"] == {"base": None}


def test_logging_a_file_twice_is_refused(tmp_path):
    """The unrecoverable mistake: log1p of a log1p, silent and structure-preserving."""
    import h5py

    write_h5ad(tmp_path / "a.h5ad", counts())
    with h5py.File(tmp_path / "a.h5ad", "a") as h:
        prep.shifted_log(h, None)
        with pytest.raises(SystemExit, match="already present"):
            prep.shifted_log(h, None)


def test_a_matrix_that_is_not_counts_is_refused(tmp_path):
    """A median-depth size factor is only meaningful on counts, so fractional values stop it
    even when uns/log1p was never written."""
    import h5py

    X = counts()
    X.data = X.data / 3.0
    write_h5ad(tmp_path / "a.h5ad", X)
    with h5py.File(tmp_path / "a.h5ad", "a") as h, \
            pytest.raises(SystemExit, match="not raw counts"):
        prep.shifted_log(h, None)


def test_a_dense_matrix_is_refused(tmp_path):
    import h5py

    write_h5ad(tmp_path / "a.h5ad", counts().toarray())
    with h5py.File(tmp_path / "a.h5ad", "a") as h, \
            pytest.raises(SystemExit, match="csr_matrix"):
        prep.csr_group(h)


def test_an_integer_payload_is_refused_rather_than_truncated(tmp_path):
    """Writing 0.83 into an int32 dataset stores 0, with no error anywhere."""
    import h5py

    X = counts()
    X.data = X.data.astype(np.int32)
    write_h5ad(tmp_path / "a.h5ad", X)
    with h5py.File(tmp_path / "a.h5ad", "a") as h, \
            pytest.raises(SystemExit, match="truncate"):
        prep.csr_group(h)


def test_a_zero_target_sum_is_refused(tmp_path):
    import h5py

    write_h5ad(tmp_path / "a.h5ad", counts())
    with h5py.File(tmp_path / "a.h5ad", "a") as h, \
            pytest.raises(SystemExit, match="must be positive"):
        prep.shifted_log(h, 0.0)


# --------------------------------------------------------------------- through the CLI


def _obs_ready(path, X, pert, batch):
    """An h5ad shaped the way the fold files are: a categorical pert column and a batch."""
    import anndata as ad
    import pandas as pd

    obs = pd.DataFrame({"gene_target": pd.Categorical(pert), "batch": pd.Categorical(batch)})
    ad.AnnData(X=X, obs=obs).write_h5ad(path)
    return path


def test_the_cli_logs_and_stamps_every_output(tmp_path, monkeypatch, capsys):
    import anndata as ad

    X = counts(n_cells=8, n_genes=6)
    labels = ["Non-Targeting"] * 4 + ["TP53"] * 4
    _obs_ready(tmp_path / "src.h5ad", X, labels, ["b1"] * 8)
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", [
        "prep", "--src", str(tmp_path / "src.h5ad"), "--cell-type", "HCT116",
        "--out-dir", str(out), "--log1p",
    ])
    assert prep.main() == 0
    got = ad.read_h5ad(out / "src.h5ad")
    assert got.uns["log1p"] == {"base": None}
    assert list(got.obs["cell_type"].unique()) == ["HCT116"]
    assert "shifted log" in capsys.readouterr().out


def test_the_cli_leaves_x_alone_without_the_flag(tmp_path, monkeypatch):
    import anndata as ad

    X = counts(n_cells=8, n_genes=6)
    _obs_ready(tmp_path / "src.h5ad", X, ["Non-Targeting"] * 8, ["b1"] * 8)
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", [
        "prep", "--src", str(tmp_path / "src.h5ad"), "--cell-type", "HCT116",
        "--out-dir", str(out),
    ])
    assert prep.main() == 0
    got = ad.read_h5ad(out / "src.h5ad")
    assert "log1p" not in got.uns
    assert np.array_equal(got.X.toarray(), X.toarray())


def test_target_sum_without_log1p_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "prep", "--src", str(tmp_path / "src.h5ad"), "--cell-type", "HCT116",
        "--out-dir", str(tmp_path / "o"), "--target-sum", "10000",
    ])
    with pytest.raises(SystemExit, match="only means something with --log1p"):
        prep.main()


def test_mixed_outputs_are_called_out(tmp_path, monkeypatch, capsys):
    """cell_load raises outright on a mix for output_space='all'; say so at prep time."""
    import h5py

    X = counts(n_cells=8, n_genes=6)
    a = _obs_ready(tmp_path / "a.h5ad", X, ["Non-Targeting"] * 8, ["b1"] * 8)
    b = _obs_ready(tmp_path / "b.h5ad", X, ["Non-Targeting"] * 8, ["b1"] * 8)
    with h5py.File(a, "a") as h:  # a is already logged; b is not
        prep.shifted_log(h, None)
    monkeypatch.setattr("sys.argv", [
        "prep", "--src", str(a), "--cell-type", "HCT116", "--src", str(b),
        "--cell-type", "HEK293T", "--in-place",
    ])
    assert prep.main() == 0
    assert "disagree on uns/log1p" in capsys.readouterr().out
