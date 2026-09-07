"""Contract tests for scripts/emit_tx_submission.py — the two gene axes and the fill.

The model's axis is 38,584 genes; the challenge's is 18,533, and only 18,106 are on both. Every
mistake available here is silent: a submission built from a mis-mapped axis has the right shape,
the right dtype, the right cell counts and passes `vcc prep`, and scores like a bad model. So the
mapping is asserted by SYMBOL against deliberately shuffled axes (`test_projection_follows_symbols
_not_positions`), which is the same rule `PriorSource.to_edge_index` exists to enforce on the
prior side — filter two lists independently and the survivors get re-paired into pairs that were
never in the data.

The second group is the arithmetic that has to hold for the file to be scorable at all: every
emitted row totals exactly the library size it was given (before integerisation), and the 427
genes the model cannot emit take the controls' share of it rather than having their mass
silently redistributed onto the genes it can.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sub = _load("emit_tx_submission")
diag = _load("diagnose_tx_arm")


def _axes(model_genes: list[str], chal_genes: list[str]):
    """The two index maps main() builds, from two symbol lists."""
    pos = {g: i for i, g in enumerate(model_genes)}
    chal_to_model = np.array([pos.get(g, -1) for g in chal_genes], dtype=np.int64)
    model_to_chal = np.full(len(model_genes), -1, dtype=np.int64)
    for j, m in enumerate(chal_to_model):
        if m >= 0:
            model_to_chal[m] = j
    return chal_to_model, model_to_chal, np.where(chal_to_model < 0)[0]


# -- the projection out of the model's axis ------------------------------------------------

def test_every_row_totals_exactly_its_target_depth():
    """cell-eval2 scores CPM-like quantities and `--depth rescale` exists to make the library
    size come from the control cell rather than from an unbounded expm1. If the row total is
    not the number we asked for, that guarantee is gone."""
    model_genes = [f"g{i}" for i in range(6)]
    chal_genes = ["g1", "g3", "g5", "zz"]
    _, m2c, fill = _axes(model_genes, chal_genes)
    comp = np.abs(np.random.default_rng(0).normal(size=(4, 6))) + 0.1
    fill_frac = np.array([0.05])
    depths = np.array([1000.0, 2000.0, 500.0, 12345.0])
    out = sub.project_to_submission(comp, m2c, fill, fill_frac, len(chal_genes), depths)
    assert np.allclose(out.sum(1), depths)


def test_the_unpredictable_genes_take_the_controls_share_and_no_more():
    """The 427 get the context's own control mean. If they got zero instead, their mass would
    move onto the genes the model does emit and every one of those would read high."""
    model_genes = ["a", "b", "c"]
    chal_genes = ["a", "c", "x", "y"]
    _, m2c, fill = _axes(model_genes, chal_genes)
    assert list(fill) == [2, 3]
    comp = np.array([[1.0, 99.0, 3.0]])          # b is model-only and must not appear
    fill_frac = np.array([0.02, 0.03])            # q = 0.05
    out = sub.project_to_submission(comp, m2c, fill, fill_frac, 4, np.array([10_000.0]))
    assert out[0, 2] == pytest.approx(200.0)      # 0.02 * 10,000
    assert out[0, 3] == pytest.approx(300.0)      # 0.03 * 10,000
    # a and c split the remaining 95% in the model's own ratio, 1 : 3
    assert out[0, 0] == pytest.approx(9_500.0 * 0.25)
    assert out[0, 1] == pytest.approx(9_500.0 * 0.75)


def test_zero_fill_gives_the_whole_cell_to_the_predicted_genes():
    model_genes = ["a", "c"]
    chal_genes = ["a", "c", "x"]
    _, m2c, fill = _axes(model_genes, chal_genes)
    out = sub.project_to_submission(np.array([[1.0, 1.0]]), m2c, fill, np.zeros(1), 3,
                                    np.array([100.0]))
    assert out[0, 2] == 0.0
    assert out[0, :2].sum() == pytest.approx(100.0)


def test_projection_follows_symbols_not_positions():
    """Shuffle the challenge axis and the emitted values must follow their symbols. Two
    independently-filtered index lists would silently re-pair here and nothing downstream
    would notice."""
    model_genes = ["a", "b", "c", "d"]
    chal_genes = ["d", "b", "a"]                   # shuffled, and `c` is dropped
    _, m2c, fill = _axes(model_genes, chal_genes)
    assert fill.size == 0
    comp = np.array([[1.0, 2.0, 500.0, 4.0]])      # a=1, b=2, c=500 (unemittable), d=4
    out = sub.project_to_submission(comp, m2c, fill, np.zeros(0), 3, np.array([7.0]))
    # totals 1 + 2 + 4 = 7 among the emittable three, so each lands on its own count
    assert out[0, 0] == pytest.approx(4.0)         # d
    assert out[0, 1] == pytest.approx(2.0)         # b
    assert out[0, 2] == pytest.approx(1.0)         # a


def test_a_model_gene_the_challenge_lacks_never_reaches_the_output():
    model_genes = ["a", "hidden", "b"]
    chal_genes = ["a", "b"]
    _, m2c, fill = _axes(model_genes, chal_genes)
    out = sub.project_to_submission(np.array([[1.0, 1e6, 1.0]]), m2c, fill, np.zeros(0), 2,
                                    np.array([100.0]))
    assert out[0].tolist() == pytest.approx([50.0, 50.0])


# -- the projection into the model's axis --------------------------------------------------

def _pool(rows: list[list[float]]) -> tuple[sp.csr_matrix, np.ndarray]:
    X = sp.csr_matrix(np.asarray(rows, dtype=np.float32))
    return X, np.asarray(X.sum(axis=1)).ravel()


def test_basal_window_zeroes_the_genes_the_challenge_never_measured():
    """53.1 % of the model's input axis is a structural zero under this projection. That is the
    distribution shift the whole pre-test is about; here we only assert it is what happens."""
    chal_genes = ["a", "b"]
    model_genes = ["a", "x", "b", "y"]
    c2m, _, _ = _axes(model_genes, chal_genes)
    pool, depths = _pool([[3.0, 1.0]])
    X = sub.basal_window(pool, depths, np.array([0]), c2m, 4, size_factor=4.0,
                         on_axis_share=1.0, x_space="counts")
    assert X[0].tolist() == pytest.approx([3.0, 0.0, 1.0, 0.0])


def test_basal_window_scales_by_l_times_share_over_the_measured_depth():
    """`D_obs / share` estimates the library the training prep divided by, from the part of it
    the challenge measured. Getting this wrong inflates every surviving value by ~1/0.70."""
    chal_genes = ["a", "b"]
    model_genes = ["a", "b"]
    c2m, _, _ = _axes(model_genes, chal_genes)
    pool, depths = _pool([[30.0, 70.0]])           # D_obs = 100
    X = sub.basal_window(pool, depths, np.array([0]), c2m, 2, size_factor=1000.0,
                         on_axis_share=0.7, x_space="counts")
    assert X[0].tolist() == pytest.approx([210.0, 490.0])   # * 1000 * 0.7 / 100
    assert X[0].sum() == pytest.approx(700.0)

    full = sub.basal_window(pool, depths, np.array([0]), c2m, 2, size_factor=1000.0,
                            on_axis_share=1.0, x_space="counts")
    assert full[0].sum() == pytest.approx(1000.0)


def test_basal_window_applies_log1p_when_the_model_consumes_it():
    chal_genes = ["a"]
    model_genes = ["a", "z"]
    c2m, _, _ = _axes(model_genes, chal_genes)
    pool, depths = _pool([[10.0]])
    X = sub.basal_window(pool, depths, np.array([0]), c2m, 2, size_factor=10.0,
                         on_axis_share=1.0, x_space="log1p")
    assert X[0].tolist() == pytest.approx([np.log1p(10.0), 0.0])


def test_basal_window_returns_the_rows_in_the_order_asked():
    chal_genes = ["a", "b"]
    c2m, _, _ = _axes(["a", "b"], chal_genes)
    pool, depths = _pool([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    X = sub.basal_window(pool, depths, np.array([2, 0, 1]), c2m, 2, size_factor=1.0,
                         on_axis_share=1.0, x_space="counts")
    assert (X > 0).sum(1).tolist() == [2, 1, 1]
    assert X[1, 0] > 0 and X[2, 1] > 0


def test_control_mean_fraction_sums_to_one_and_drops_shallow_cells():
    pool, depths = _pool([[1.0, 1.0], [500.0, 1500.0], [1000.0, 1000.0]])
    frac = sub.control_mean_fraction(pool, depths, min_libsize=10.0)
    assert frac.sum() == pytest.approx(1.0)
    # the 2-UMI cell is excluded; the other two average CPM 0.375 / 0.625
    assert frac.tolist() == pytest.approx([0.375, 0.625])


# -- the mask the pre-test uses, which is the same decision seen from the training side ----

def test_apply_gene_mask_zeroes_exactly_the_genes_outside_the_narrower_assay():
    X = np.log1p(np.array([[10.0, 20.0, 30.0]], dtype=np.float32))
    out, info = diag.apply_gene_mask(X, np.array(["a", "b", "c"]), {"a", "c"}, "log1p", False)
    assert np.expm1(out)[0].tolist() == pytest.approx([10.0, 0.0, 30.0])
    assert info["genes_kept"] == 2 and info["genes_zeroed"] == 1
    assert info["library_share_kept_median"] == pytest.approx(40 / 60)


def test_apply_gene_mask_renorm_gives_the_survivors_the_whole_library():
    X = np.log1p(np.array([[10.0, 20.0, 30.0]], dtype=np.float32))
    out, info = diag.apply_gene_mask(X, np.array(["a", "b", "c"]), {"a", "c"}, "log1p", True)
    assert np.expm1(out)[0].sum() == pytest.approx(60.0, rel=1e-5)
    assert info["renormalised"] is True


def test_read_symbol_list_survives_both_header_conventions(tmp_path):
    """gene_names.csv HAS a header in 2026 and had none in 2025 (challenges/vcc2026/CLAUDE.md,
    trap 11). Either mistake misaligns every gene silently."""
    with_header = tmp_path / "h.csv"
    with_header.write_text("gene_name\nAAA\nBBB\n")
    without = tmp_path / "n.csv"
    without.write_text("AAA\nBBB\n")
    assert diag.read_symbol_list(with_header) == ["AAA", "BBB"]
    assert diag.read_symbol_list(without) == ["AAA", "BBB"]


# -- the two data-defined constants --------------------------------------------------------

def _log1p_fold(tmp_path, rows: list[list[float]]) -> Path:
    """A tiny h5ad in the model's own shifted-log space, as prep_tx_training --log1p writes."""
    import anndata as ad
    import pandas as pd

    X = sp.csr_matrix(np.log1p(np.asarray(rows, dtype=np.float32)))
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])]),
                   var=pd.DataFrame(index=[f"g{i}" for i in range(X.shape[1])]))
    a.write_h5ad(tmp_path / "fold.h5ad")
    return tmp_path / "fold.h5ad"


def test_measure_reference_reads_the_on_axis_share_in_count_space(tmp_path):
    """The share is a property of the RAW library, and the file holds shifted logarithms — so it
    has to be measured after expm1. Taking it on the log values would report a different number
    and silently mis-scale every basal cell."""
    path = _log1p_fold(tmp_path, [[30.0, 70.0, 0.0], [10.0, 10.0, 80.0]])
    on_axis = np.array([True, True, False])
    ref = sub.measure_reference(path, on_axis, block=1)
    assert ref["on_axis_share_median"] == pytest.approx(np.median([1.0, 0.2]))
    assert ref["n_cells"] == 2


def test_measure_reference_bounds_on_the_per_cell_maximum_not_the_corpus_one(tmp_path):
    """One extreme cell must not license every cell to be that extreme — the p99.9 of the
    per-cell maximum is the statistic, and it sits well below the corpus maximum."""
    rows = [[float(v), 1.0] for v in range(1, 1001)]
    path = _log1p_fold(tmp_path, rows)
    ref = sub.measure_reference(path, np.array([True, True]))
    assert ref["corpus_max"] == pytest.approx(np.log1p(1000.0), rel=1e-5)
    assert ref["per_cell_max_p999"] < ref["corpus_max"]
    assert ref["per_cell_max_median"] == pytest.approx(np.log1p(500.5), rel=1e-3)


def test_measure_reference_survives_an_all_zero_cell(tmp_path):
    """A CSR row with no stored entries has no maximum; reduceat on its start index would read
    the next row's data instead."""
    path = _log1p_fold(tmp_path, [[5.0, 5.0], [0.0, 0.0], [1.0, 0.0]])
    ref = sub.measure_reference(path, np.array([True, False]), block=2)
    assert ref["n_cells"] == 3
    assert ref["corpus_max"] == pytest.approx(np.log1p(5.0), rel=1e-5)
