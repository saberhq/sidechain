"""The streaming aggregator must reproduce in-memory pseudobulks exactly, on
both on-disk layouts (CSR group and dense dataset), and merge without
double-counting."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sidechain.data import stream_pseudobulk as spb


def _toy(rng, n=57, g=11, dense=False):
    X = rng.poisson(1.5, size=(n, g)).astype(np.float32)
    X[3] = 0  # an empty cell must be skipped, not divide by zero
    labels = rng.choice(["ctrl", "A", "B", "C"], size=n)
    obs = pd.DataFrame({"pert": labels}, index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"g{j}" for j in range(g)])
    a = ad.AnnData(X=X if dense else sp.csr_matrix(X), obs=obs, var=var)
    return a, X, labels


@pytest.mark.parametrize("dense", [False, True])
def test_matches_in_memory_sums(tmp_path, dense):
    rng = np.random.default_rng(0)
    a, X, labels = _toy(rng, dense=dense)
    p = tmp_path / "toy.h5ad"
    a.write_h5ad(p)
    out = spb.stream_pseudobulk(p, "pert", {"A", "B", "ctrl"}, block_rows=10)
    assert out.labels == ["A", "B", "ctrl"]
    X = X.astype(np.float64)
    lib = X.sum(1)
    for i, lab in enumerate(out.labels):
        rows = (labels == lab) & (lib > 0)
        np.testing.assert_allclose(out.count_sum[i], X[rows].sum(0))
        cpm = X[rows] / lib[rows][:, None] * 1e6
        np.testing.assert_allclose(out.cpm_sum[i], cpm.sum(0), rtol=1e-9)
        np.testing.assert_allclose(out.cpm_sq_sum[i], (cpm**2).sum(0), rtol=1e-9)
        assert out.n_cells[i] == rows.sum()
    assert "C" not in out.labels


def test_merge_unions_labels_and_control_once(tmp_path):
    rng = np.random.default_rng(1)
    a, _X, _labels = _toy(rng)
    p = tmp_path / "t.h5ad"
    a.write_h5ad(p)
    first = spb.stream_pseudobulk(p, "pert", {"A", "ctrl"})
    second = spb.stream_pseudobulk(p, "pert", {"B", "ctrl"}, skip_labels={"ctrl"})
    m = spb.merge(first, second)
    assert m.labels == ["A", "B", "ctrl"]
    i = m.labels.index("ctrl")
    assert m.n_cells[i] == first.n_cells[first.labels.index("ctrl")]  # not doubled
    q = tmp_path / "m.npz"
    m.save(q)
    back = spb.PseudobulkSums.load(q)
    np.testing.assert_allclose(back.cpm_sum, m.cpm_sum)
    assert back.labels == m.labels
