import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sidechain.data.stream_subset import extract_cells


@pytest.mark.parametrize("dense", [False, True])
def test_extracts_exactly_the_wanted_cells_with_caps(tmp_path, dense):
    rng = np.random.default_rng(0)
    n, g = 120, 9
    X = rng.poisson(1.0, size=(n, g)).astype(np.float32)
    labels = np.array(["ctrl"] * 60 + ["A"] * 30 + ["B"] * 20 + ["C"] * 10)
    a = ad.AnnData(X=X if dense else sp.csr_matrix(X),
                   obs=pd.DataFrame({"pert": labels, "batch": rng.integers(0, 3, n)}, index=[f"c{i}" for i in range(n)]),
                   var=pd.DataFrame({"chr": ["1"] * g}, index=[f"g{j}" for j in range(g)]))
    p = tmp_path / "big.h5ad"
    a.write_h5ad(p)
    sub = extract_cells(p, "pert", {"A", "B"}, control="ctrl", max_per_label=12, max_control=25, seed=1, block_rows=17)
    assert "C" not in list(sub.obs["pert"].cat.categories) if hasattr(sub.obs["pert"], "cat") else True
    vc = sub.obs["pert"].astype(str).value_counts()
    assert vc["A"] == 12 and vc["B"] == 12 and vc["ctrl"] == 25 and "C" not in vc
    assert list(sub.var.columns) == ["chr"] and "batch" in sub.obs
    # values match the source rows, in source order
    src = a[sub.obs_names].X
    src = src.toarray() if sp.issparse(src) else src
    np.testing.assert_array_equal(sub.X.toarray(), src)


def test_relabel_control(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.poisson(1.0, size=(30, 5)).astype(np.float32)
    labels = np.array(["ctrl"] * 15 + ["A"] * 15)
    a = ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame({"pert": labels}, index=[f"c{i}" for i in range(30)]),
                   var=pd.DataFrame(index=[f"g{j}" for j in range(5)]))
    p = tmp_path / "r.h5ad"
    a.write_h5ad(p)
    sub = extract_cells(p, "pert", {"A"}, control="ctrl", relabel_control="non-targeting")
    vc = sub.obs["pert"].astype(str).value_counts()
    assert vc["non-targeting"] == 15 and vc["A"] == 15 and "ctrl" not in vc.index
