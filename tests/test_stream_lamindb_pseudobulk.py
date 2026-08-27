"""Label derivation for lamin-streamed corpora: the three curated-obs shapes
(stringified multi-lists, guide-name control rules, condition row filters) must
resolve to exactly the cells the design says -- and the override path through
the shared accumulator must count only those cells."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sidechain.data import stream_pseudobulk as spb
from sidechain.data.stream_lamindb_pseudobulk import DROP, _derive_labels


def test_multi_label_keeps_singles_and_drops_combos_and_empties():
    obs = pd.DataFrame({"pert_target_multi": [
        "['VAC14']", "['CEBPE' 'RUNX1T1']", "None", "nan", "['LDLR']", "PLAIN"]})
    labels, stats = _derive_labels(obs, "pert_target_multi", multi_label=True)
    assert labels.tolist() == ["VAC14", DROP, DROP, DROP, "LDLR", "PLAIN"]
    assert stats["rows_dropped"] == 3


def test_control_rule_claims_by_prefix_and_unclaimed_nan_stays_dropped():
    obs = pd.DataFrame({
        "pert_target": ["GENE1", "nan", "nan", "GENE2"],
        "pert_name": ["GENE1_g1", "non-targeting_01", "NO_SITE_9", "GENE2_g1"],
    })
    labels, stats = _derive_labels(
        obs, "pert_target",
        control_rule=("pert_name", ("non-targeting", "NO_SITE"), "control"))
    assert labels.tolist() == ["GENE1", "control", "control", "GENE2"]
    assert stats["control_cells"] == 2

    # a NaN target the rule does not claim is dropped, never pooled as control
    obs2 = pd.DataFrame({
        "pert_target": ["nan", "nan"],
        "pert_name": ["non-targeting_01", "unassigned"],
    })
    labels2, _ = _derive_labels(
        obs2, "pert_target", control_rule=("pert_name", ("non-targeting",), "control"))
    assert labels2.tolist() == ["control", DROP]


def test_control_rule_matching_nothing_is_an_error_not_an_empty_arm():
    obs = pd.DataFrame({"pert_target": ["A"], "pert_name": ["A_g1"]})
    with pytest.raises(SystemExit, match="matched 0 rows"):
        _derive_labels(obs, "pert_target",
                       control_rule=("pert_name", ("non-targeting",), "control"))


def test_row_filter_keeps_one_condition_arm():
    obs = pd.DataFrame({
        "pert_target": ["A", "A", "B", "nan"],
        "pert_name": ["A_g1", "A_g2", "B_g1", "NO_SITE_1"],
        "perturbation_2": ["Control", "IFN", "Control", "Control"],
    })
    labels, stats = _derive_labels(
        obs, "pert_target",
        control_rule=("pert_name", ("NO_SITE",), "control"),
        row_filters={"perturbation_2": "Control"})
    assert labels.tolist() == ["A", DROP, "B", "control"]
    assert stats["rows_kept_perturbation_2=Control"] == 3


def test_accumulator_override_counts_only_derived_labels(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.poisson(2.0, size=(6, 5)).astype(np.float32) + 1  # no empty cells
    a = ad.AnnData(X=sp.csr_matrix(X),
                   obs=pd.DataFrame(index=[f"c{i}" for i in range(6)]),
                   var=pd.DataFrame(index=[f"g{j}" for j in range(5)]))
    p = tmp_path / "t.h5ad"
    a.write_h5ad(p)
    derived = np.array(["A", DROP, "control", "A", DROP, "control"])
    import h5py
    with h5py.File(p, "r") as f:
        out = spb.stream_pseudobulk_file(
            f, "ignored", labels_all=derived, skip_labels={DROP}, block_rows=2)
    assert out.labels == ["A", "control"]
    X = X.astype(np.float64)
    np.testing.assert_allclose(out.count_sum[0], X[[0, 3]].sum(0))
    np.testing.assert_allclose(out.count_sum[1], X[[2, 5]].sum(0))
    assert out.n_cells.tolist() == [2, 2]


def test_gene_axis_override_is_recorded_verbatim(tmp_path):
    rng = np.random.default_rng(8)
    X = rng.poisson(2.0, size=(4, 3)).astype(np.float32) + 1
    a = ad.AnnData(X=sp.csr_matrix(X),
                   obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
                   var=pd.DataFrame(index=["ENSG1", "ENSG2", "ENSG3"]))
    p = tmp_path / "t.h5ad"
    a.write_h5ad(p)
    import h5py
    with h5py.File(p, "r") as f:
        out = spb.stream_pseudobulk_file(
            f, "ignored", labels_all=np.array(["A"] * 4),
            genes=np.array(["SYM1", "", "SYM3"]))
    assert out.genes.tolist() == ["SYM1", "", "SYM3"]
