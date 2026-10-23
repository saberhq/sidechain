"""`sidechain.eval.loco --alpha-bulk-target` (T84's knee rule on a fold, T93).

What is pinned: the knee run's prediction is, cell for cell, a plain run at the strength it chose; its record
carries every read, the ceiling's first; a ratio already under the target stops at the ceiling; and the flag
is refused where it cannot act.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.eval import loco

G, N_LAB = 120, 12
KW = dict(pert_col="perturbation", control="non-targeting", var_floor="poisson", emit_lambda=0.5,
          alpha=1.0, bulk_anchor="pooled", min_libsize=0.0, shrink_stage="pooled", shrink_rule="adaptive",
          dual_fallback="anchor", emit_shape="controls")


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
    for lab, k in [("non-targeting", 120)] + [(x, 25) for x in labs]:
        for _ in range(k):
            rows.append(rng.poisson(basal / basal.sum() * rng.integers(2000, 9000)))
            obs.append(lab)
    real = ad.AnnData(X=sp.csr_matrix(np.asarray(rows, dtype=np.float32)),
                      obs=pd.DataFrame({"perturbation": obs}, index=[f"c{i}" for i in range(len(rows))]),
                      var=pd.DataFrame(index=genes.astype(str)))
    real.write_h5ad(tmp_path / "real.h5ad")
    return tmp_path, tmp_path / "real.h5ad", [(src, "ctrl")]


def test_the_knee_run_is_a_plain_run_at_the_strength_it_chose(fold):
    tmp, real, sources = fold
    info = loco.build_transfer_prediction(real, sources, tmp / "knee.h5ad", **KW, alpha_bulk=6.0,
                                          alpha_bulk_target=0.5, seed=3)
    k = info["alpha_bulk_knee"]
    assert k["tries"][0]["alpha_bulk"] == 6.0 and k["reads"] == len(k["tries"]) >= 2
    assert k["alpha_bulk"] == info["alpha_bulk"] < 6.0
    assert [t["ratio_budget_over_claim"] for t in k["tries"]] == sorted(
        (t["ratio_budget_over_claim"] for t in k["tries"]), reverse=True)
    assert not list(tmp.glob("knee.knee*.h5ad"))                  # the reads' files are gone
    plain = loco.build_transfer_prediction(real, sources, tmp / "plain.h5ad", **KW, alpha_bulk=k["alpha_bulk"],
                                           seed=3)
    a, b = ad.read_h5ad(tmp / "knee.h5ad"), ad.read_h5ad(tmp / "plain.h5ad")
    assert (a.X != b.X).nnz == 0 and list(a.obs["perturbation"]) == list(b.obs["perturbation"])
    info.pop("pred"), plain.pop("pred"), info.pop("alpha_bulk_knee")
    assert info == plain                                          # every other record is the plain run's


def test_a_ratio_under_the_target_stops_at_the_ceiling(fold):
    tmp, real, sources = fold
    info = loco.build_transfer_prediction(real, sources, tmp / "k.h5ad", **KW, alpha_bulk=1.05,
                                          alpha_bulk_target=0.99, seed=1)
    k = info["alpha_bulk_knee"]
    if k["tries"][0]["ratio_budget_over_claim"] <= 0.99:
        assert k["reads"] == 1 and k["alpha_bulk"] == 1.05 and k["met"]
    else:                                                         # the floor is alpha: never below it
        assert k["alpha_bulk"] >= 1.0


@pytest.mark.parametrize("extra, msg", [
    ({"alpha_bulk": None}, "pass --alpha-bulk"),
    ({"alpha_bulk": 2.0, "alpha_bulk_target": 1.5}, "in \\(0, 1\\)"),
    ({"alpha_bulk": 0.9}, "must be above --alpha"),
])
def test_the_flag_is_refused_where_it_cannot_act(fold, extra, msg):
    tmp, real, sources = fold
    kw = {**KW, "alpha_bulk_target": 0.9, **extra}
    with pytest.raises(SystemExit, match=msg):
        loco.build_transfer_prediction(real, sources, tmp / "r.h5ad", **kw)
