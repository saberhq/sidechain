"""Contract tests for the shared gene axis: the plan, the manifest, and the projection.

The failure this guards against is silent: a corpus written onto the wrong columns still
trains, still scores, and is simply wrong -- the same class of error as re-pairing edges from
two independently filtered endpoint lists.
"""
from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from sidechain.data.union_axis import (
    AxisPlan,
    build_axis,
    coverage_rows,
    load_manifest,
    plan_from_manifest,
    project_h5ad,
    read_anchor,
    read_gene_axis,
    write_manifest,
)

A = ["A", "B", "C"]  # narrow corpus
B = ["B", "C", "D", "E"]  # wider, overlapping
ANCHOR = ["E", "D", "C", "B", "A", "Z"]  # scored axis, its own order, one gene nobody has


def _axes():
    return {"a": A, "b": B}


# ------------------------------------------------------------------------ the plan


def test_anchor_mode_keeps_anchor_order_and_length():
    plan = build_axis(_axes(), mode="anchor", anchor=ANCHOR)
    assert list(plan.genes) == ANCHOR
    assert list(plan.mask("a")) == [False, False, True, True, True, False]
    assert list(plan.mask("b")) == [True, True, True, True, False, False]


def test_union_mode_is_anchor_order_then_sorted_extras():
    plan = build_axis({"a": A, "b": B + ["Q"]}, mode="union", anchor=["C", "B"])
    # anchor genes first in anchor order, then everything else any source measures, sorted.
    assert list(plan.genes) == ["C", "B", "A", "D", "E", "Q"]


def test_union_without_anchor_is_sorted():
    plan = build_axis(_axes(), mode="union")
    assert list(plan.genes) == ["A", "B", "C", "D", "E"]


def test_intersection_reproduces_the_status_quo():
    plan = build_axis(_axes(), mode="intersection")
    assert list(plan.genes) == ["B", "C"]
    assert plan.mask("a").all() and plan.mask("b").all()
    # ...and this is the cost the union mode exists to avoid.
    assert len(build_axis(_axes(), mode="union").genes) > len(plan.genes)


def test_dropped_counts_genes_the_axis_does_not_carry():
    plan = build_axis(_axes(), mode="anchor", anchor=["A", "B"])
    assert plan.dropped == {"a": 1, "b": 3}


def test_duplicate_symbols_are_refused_not_resolved():
    with pytest.raises(ValueError, match="repeats"):
        build_axis({"a": ["A", "B", "A"]}, mode="union")


def test_a_source_sharing_no_gene_raises():
    """An Ensembl-id file against a symbol axis: a crash, not an all-zero corpus."""
    with pytest.raises(ValueError, match="shares no gene"):
        build_axis({"a": A, "z": ["ENSG1", "ENSG2"]}, mode="anchor", anchor=ANCHOR)


def test_anchor_mode_requires_an_anchor():
    with pytest.raises(ValueError, match="anchor"):
        build_axis(_axes(), mode="anchor")


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        build_axis(_axes(), mode="widest")


def test_coverage_rows_report_scored_coverage():
    plan = build_axis(_axes(), mode="anchor", anchor=ANCHOR)
    rows = {r["source"]: r for r in coverage_rows(plan, scored=["A", "B", "C", "D", "E"])}
    assert rows["a"]["scored_covered"] == 3
    assert rows["b"]["scored_covered"] == 4
    assert rows["AXIS"]["scored_covered"] == 5  # Z is on the axis but not scored


# --------------------------------------------------------------------- the manifest


def test_manifest_round_trips(tmp_path):
    plan = build_axis(_axes(), mode="anchor", anchor=ANCHOR, sources={"a": tmp_path / "a.h5ad"})
    p = write_manifest(plan, tmp_path / "m.json", scored=["A", "B"])
    back = plan_from_manifest(p)
    assert back.genes == plan.genes
    assert back.mode == plan.mode
    for k in plan.measured:
        assert np.array_equal(back.measured[k], plan.measured[k])


def test_manifest_detects_an_edited_gene_list(tmp_path):
    plan = build_axis(_axes(), mode="anchor", anchor=ANCHOR)
    p = write_manifest(plan, tmp_path / "m.json")
    payload = json.loads(p.read_text())
    payload["genes"][0] = "TAMPERED"
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sha256"):
        load_manifest(p)


def test_read_anchor_handles_both_gene_name_files(tmp_path):
    with_header = tmp_path / "2026.csv"
    with_header.write_text("gene_name\nA1BG\nA1CF\n")
    without = tmp_path / "2025.csv"
    without.write_text("A1BG\nA1CF\n")
    assert read_anchor(with_header) == ["A1BG", "A1CF"]
    assert read_anchor(without) == ["A1BG", "A1CF"]


# -------------------------------------------------------------------- the projection


def _write(tmp_path, name, genes, X, obs=None):
    import pandas as pd

    n = X.shape[0]
    obs = obs if obs is not None else pd.DataFrame(
        {"perturbation": ["p"] * n}, index=[f"c{i}" for i in range(n)]
    )
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=list(genes)))
    path = tmp_path / name
    a.write_h5ad(path)
    return path


@pytest.mark.parametrize("sparse", [False, True])
def test_projection_scatters_values_onto_the_right_columns(tmp_path, sparse):
    X = np.array([[1.0, 2.0, 3.0], [0.0, 5.0, 0.0]], dtype=np.float32)
    src = _write(tmp_path, "a.h5ad", A, sp.csr_matrix(X) if sparse else X)
    plan = build_axis({"a": read_gene_axis(src)}, mode="anchor", anchor=ANCHOR,
                      sources={"a": src})
    info = project_h5ad(src, tmp_path / "out.h5ad", plan, "a")

    out = ad.read_h5ad(tmp_path / "out.h5ad")
    assert list(out.var_names) == ANCHOR
    dense = out.X.toarray()
    # A B C -> positions 4 3 2 of the anchor; E D and Z are structural zeros.
    assert dense[0, 4] == 1.0 and dense[0, 3] == 2.0 and dense[0, 2] == 3.0
    assert dense[:, [0, 1, 5]].sum() == 0.0
    assert info["nnz"] == int((X != 0).sum())
    assert info["measured"] == 3


def test_projection_preserves_obs_and_records_the_mask(tmp_path):
    import pandas as pd

    obs = pd.DataFrame({"perturbation": ["p", "control"]}, index=["c0", "c1"])
    src = _write(tmp_path, "a.h5ad", A, np.eye(3, dtype=np.float32)[:2], obs=obs)
    plan = build_axis({"a": A}, mode="anchor", anchor=ANCHOR, sources={"a": src})
    project_h5ad(src, tmp_path / "out.h5ad", plan, "a")

    out = ad.read_h5ad(tmp_path / "out.h5ad")
    assert list(out.obs["perturbation"]) == ["p", "control"]
    assert list(out.obs_names) == ["c0", "c1"]
    mask = np.asarray(out.uns["sidechain_axis"]["mask"], dtype=bool)
    assert list(mask) == list(plan.mask("a"))
    assert out.uns["sidechain_axis"]["genes_sha256"] == plan.genes_sha256


def test_two_corpora_land_on_one_axis_and_stay_distinguishable(tmp_path):
    """The whole point: different gene lists, same output axis, masks that differ."""
    a = _write(tmp_path, "a.h5ad", A, np.ones((2, 3), dtype=np.float32))
    b = _write(tmp_path, "b.h5ad", B, np.ones((2, 4), dtype=np.float32))
    plan = build_axis({"a": A, "b": B}, mode="anchor", anchor=ANCHOR, sources={"a": a, "b": b})
    project_h5ad(a, tmp_path / "pa.h5ad", plan, "a")
    project_h5ad(b, tmp_path / "pb.h5ad", plan, "b")

    oa, ob = ad.read_h5ad(tmp_path / "pa.h5ad"), ad.read_h5ad(tmp_path / "pb.h5ad")
    assert list(oa.var_names) == list(ob.var_names) == ANCHOR
    # A is measured only by 'a', E only by 'b'; both are real zeros nowhere.
    assert oa.X.toarray()[:, 4].sum() == 2.0 and ob.X.toarray()[:, 4].sum() == 0.0
    assert ob.X.toarray()[:, 0].sum() == 2.0 and oa.X.toarray()[:, 0].sum() == 0.0


def test_projecting_a_source_the_plan_does_not_know_raises(tmp_path):
    src = _write(tmp_path, "a.h5ad", A, np.ones((2, 3), dtype=np.float32))
    plan = build_axis({"a": A}, mode="anchor", anchor=ANCHOR, sources={"a": src})
    with pytest.raises(KeyError):
        project_h5ad(src, tmp_path / "out.h5ad", plan, "b")


def test_block_streaming_gives_the_same_matrix(tmp_path):
    rng = np.random.default_rng(0)
    X = (rng.random((37, 3)) < 0.4) * rng.integers(1, 9, (37, 3))
    src = _write(tmp_path, "a.h5ad", A, sp.csr_matrix(X.astype(np.float32)))
    plan = build_axis({"a": A}, mode="anchor", anchor=ANCHOR, sources={"a": src})
    project_h5ad(src, tmp_path / "one.h5ad", plan, "a")
    project_h5ad(src, tmp_path / "many.h5ad", plan, "a", block_rows=5)
    one = ad.read_h5ad(tmp_path / "one.h5ad").X.toarray()
    many = ad.read_h5ad(tmp_path / "many.h5ad").X.toarray()
    assert np.array_equal(one, many)
    assert np.array_equal(one[:, [4, 3, 2]], X)


def test_axis_plan_mask_is_boolean_over_the_whole_axis():
    plan: AxisPlan = build_axis(_axes(), mode="anchor", anchor=ANCHOR)
    assert plan.mask("a").dtype == bool and plan.mask("a").size == len(ANCHOR)
