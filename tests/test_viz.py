"""sidechain.viz: the dataset colour plan and the per-cell QC numbers the figure builders share."""
import json

import numpy as np
import pytest
import scipy.sparse as sp

from sidechain.viz import palette
from sidechain.viz.qc import cell_qc, compact, gene_masks, gene_stats, program_score


def test_every_context_and_line_has_a_colour_and_marker_in_both_themes():
    for key in [*palette.CONTEXTS, *palette.LINES]:
        for theme in ("light", "dark"):
            c = palette.color(key, theme)
            assert c.startswith("#") and len(c) == 7
        assert palette.marker(key) in palette._MPL
        assert palette.mpl_marker(key)
    with pytest.raises(KeyError):
        palette.color("nonexistent")


def test_validation_and_test_contexts_differ_by_marker_and_lines_share_their_studys_colour():
    assert {palette.marker(c) for c in "ABC"} == {"circle"}
    assert {palette.marker(c) for c in "DEF"} == {"diamond"}
    assert len({palette.color(c) for c in "ABCDEF"}) == 6
    assert palette.color("replogle22_k562_ess") == palette.color("replogle22_rpe1") == palette.color("replogle22")
    # a corpus colour never repeats a context's
    assert not {palette.color(k) for k in palette.LINES} & {palette.color(c) for c in "ABCDEF"}


def test_as_json_resolves_every_line_and_is_serialisable():
    j = palette.as_json()
    json.dumps(j)
    for k, v in j["lines"].items():
        assert v["light"] == palette.color(k) and v["label"] == palette.label(k)


def test_gene_masks_keep_ribosomal_proteins_and_drop_the_rps6k_kinases():
    m = gene_masks(["MT-CO1", "RPL10A", "RPS4X", "RPLP0", "RPS6KA1", "RPL22L1", "ACTB"])
    assert m["mt"].tolist() == [True, False, False, False, False, False, False]
    assert m["ribo"].tolist() == [False, True, True, True, False, True, False]


def test_cell_qc_on_a_small_matrix():
    genes = ["MT-CO1", "RPL10A", "G1", "G2"]
    X = sp.csr_matrix(np.array([[10, 0, 30, 60], [0, 0, 0, 0], [5, 5, 5, 5]], dtype=float))
    q = cell_qc(X, genes, top_n=2)
    assert q["total_umi"].tolist() == [100, 0, 20]
    assert q["n_genes"].tolist() == [3, 0, 4]
    assert q["pct_mt"][0] == pytest.approx(10.0) and np.isnan(q["pct_mt"][1]) and q["pct_mt"][2] == pytest.approx(25.0)
    assert q["pct_ribo"][2] == pytest.approx(25.0)
    assert q["pct_top50"][0] == pytest.approx(90.0)          # top 2 of the cell: 60 + 30
    assert q["complexity"][0] == pytest.approx(np.log10(3) / 2)


def test_gene_stats_match_a_dense_computation():
    rng = np.random.default_rng(0)
    D = rng.poisson(2.0, size=(50, 6)).astype(float)
    g = gene_stats(sp.csr_matrix(D))
    lib = D.sum(axis=1, keepdims=True)
    N = D / lib * 1e4
    L = np.log1p(N)
    assert np.allclose(g["mean_count"], D.mean(axis=0))
    assert np.allclose(g["detect"], (D > 0).mean(axis=0))
    assert np.allclose(g["mean_cp10k"], N.mean(axis=0))
    assert np.allclose(g["mean_log"], L.mean(axis=0))
    assert np.allclose(g["var_log"], L.var(axis=0), atol=1e-9)


def test_program_score_is_the_programme_minus_a_control_set():
    L = sp.csr_matrix(np.array([[5.0, 5.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))
    s = program_score(L, ["P1", "P2", "C1", "C2"], ["P1", "P2", "absent"], ctrl_size=2)
    assert s.tolist() == [5.0, 0.0]
    assert np.isnan(program_score(L, ["P1", "P2", "C1", "C2"], ["absent"])).all()


def test_compact_rounds_floats_keeps_ints_and_nulls_nans():
    out = compact({"f": np.array([1.23456, np.nan, 0.0, 123456.7]), "i": np.array([1, 2]), "s": np.array(["a"]),
                   "x": np.float32(2.5), "n": float("nan")}, digits=3)
    assert out["f"] == [1.23, None, 0, 123000]
    assert out["i"] == [1, 2] and out["s"] == ["a"] and out["x"] == 2.5 and out["n"] is None
    json.dumps(out, allow_nan=False)
