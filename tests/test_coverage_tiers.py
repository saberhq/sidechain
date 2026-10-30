"""`PseudobulkSums.n_eff` and the builder's `--coverage-tiers` pooling weights.

    .venv/bin/python -m pytest -q tests/test_coverage_tiers.py

Writes nothing. Notes: private/code-notes/tests/test_coverage_tiers.py.md
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from sidechain.data.lfc_table import LfcTable
from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.submit.build import (
    coverage_factor,
    parse_coverage_tiers,
    pooled_delta,
)

AXIS = np.array(["A", "B", "C"])


def _pb_from_cells(cells_by_label, genes=("A", "B", "C")):
    """A `PseudobulkSums` summed from per-cell CPMs."""
    labels, cpm_sum, cpm_sq_sum, n_cells, libsize = [], [], [], [], []
    for label, cells in cells_by_label.items():
        X = np.asarray(cells, dtype=float)
        lib = X.sum(axis=1)
        cpm = X / np.maximum(lib, 1)[:, None] * 1e6
        labels.append(label)
        cpm_sum.append(cpm.sum(axis=0))
        cpm_sq_sum.append((cpm**2).sum(axis=0))
        n_cells.append(len(X))
        libsize.append(float(lib.sum()))
    return PseudobulkSums(
        labels=labels, genes=np.array(genes),
        count_sum=np.stack(cpm_sum), cpm_sum=np.stack(cpm_sum),
        cpm_sq_sum=np.stack(cpm_sq_sum), n_cells=np.array(n_cells, dtype=np.int64),
        libsize_sum=np.array(libsize), sources=["test"],
    )


# ------------------------------------------------------------------- n_eff --


def test_equal_contributors_read_as_their_own_count():
    """k cells contributing equally read as n_eff = k."""
    pb = _pb_from_cells({"X": [[10, 0, 0], [10, 0, 0], [10, 0, 0], [10, 0, 0]]})
    assert pb.n_eff(0)[0] == pytest.approx(4.0)


def test_one_cell_carrying_everything_reads_as_one():
    """One cell carrying all of a gene's counts reads as n_eff = 1."""
    pb = _pb_from_cells({"X": [[40, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]]})
    assert pb.n_eff(0)[0] == pytest.approx(1.0)


def test_never_exceeds_the_number_of_detecting_cells():
    """n_eff never exceeds the number of cells that detected the gene, on random zero-inflated arms."""
    rng = np.random.default_rng(20260829)
    for _ in range(50):
        X = rng.poisson(0.4, size=(60, 40)) * (rng.random((60, 40)) < 0.3)
        X[:, 0] = 5                                  # keep every library non-empty
        pb = _pb_from_cells({"X": X.tolist()}, genes=tuple(f"g{i}" for i in range(40)))
        true_k = (X > 0).sum(axis=0)
        assert np.all(pb.n_eff(0) <= true_k + 1e-9)


def test_a_gene_no_cell_expressed_reads_as_zero():
    pb = _pb_from_cells({"X": [[10, 0, 1], [10, 0, 1]]})
    assert pb.n_eff(0)[1] == 0.0


def test_capped_at_the_arms_own_cell_count():
    pb = _pb_from_cells({"X": [[5, 5, 5], [5, 5, 5]]})
    assert np.all(pb.n_eff(0) <= 2.0)


# ------------------------------------------------------------ the tier rule --


def test_parse_rejects_a_zero_factor():
    """A tier factor of zero is refused."""
    with pytest.raises(SystemExit, match="silences genes"):
        parse_coverage_tiers("3:0.0")


def test_parse_rejects_unordered_cuts():
    with pytest.raises(SystemExit, match="strictly increase"):
        parse_coverage_tiers("10:0.5,3:0.1")


def test_parse_rejects_a_factor_above_one():
    with pytest.raises(SystemExit, match=r"must be in \(0, 1\]"):
        parse_coverage_tiers("3:1.5")


def test_parse_round_trip():
    assert parse_coverage_tiers("3:0.10,10:0.50") == ((3.0, 0.10), (10.0, 0.50))
    assert parse_coverage_tiers(None) is None


def test_factor_reads_first_matching_cut():
    tiers = ((3.0, 0.10), (10.0, 0.50))
    got = coverage_factor(np.array([0.0, 2.9, 3.0, 9.9, 10.0, 1e6]), tiers)
    assert list(got) == [0.10, 0.10, 0.50, 0.50, 1.0, 1.0]


# --------------------------------------------------------- inside the pool --


def test_default_is_bit_identical_to_the_historical_call():
    """`coverage_tiers=None` gives the same floats as omitting it."""
    pb = _pb_from_cells({"control": [[100, 100, 100]] * 8, "TP53": [[200, 100, 50]] * 8})
    before = pooled_delta("TP53", [(pb, "control")], AXIS, var_floor="poisson")
    after = pooled_delta("TP53", [(pb, "control")], AXIS, var_floor="poisson",
                         coverage_tiers=None)
    assert np.array_equal(before, after)


def test_a_thin_source_loses_ground_to_a_dense_one():
    """Tiering moves a gene's pooled value from the thin source's toward the dense source's."""
    thin = _pb_from_cells({
        "control": [[1, 50, 50]] * 6,
        "TP53": [[600, 50, 50]] + [[0, 50, 50]] * 5,     # gene A: one cell, huge
    })
    dense = _pb_from_cells({
        "control": [[100, 50, 50]] * 6,
        "TP53": [[120, 50, 50]] * 6,                     # gene A: every cell, modest
    })
    srcs = [(thin, "control"), (dense, "control")]
    plain = pooled_delta("TP53", srcs, AXIS, var_floor="poisson", shrinkage=False)
    tiered = pooled_delta("TP53", srcs, AXIS, var_floor="poisson", shrinkage=False,
                          coverage_tiers=((3.0, 0.10), (10.0, 0.50)))
    dense_only = pooled_delta("TP53", [(dense, "control")], AXIS, var_floor="poisson",
                              shrinkage=False)
    assert abs(tiered[0] - dense_only[0]) < abs(plain[0] - dense_only[0])


def test_no_tier_can_silence_a_gene():
    """A gene that every source covers thinly keeps a nonzero value."""
    weak = _pb_from_cells({
        "control": [[300, 50, 50]] + [[0, 50, 50]] * 5,
        "TP53": [[600, 50, 50]] + [[0, 50, 50]] * 5,
    })
    out = pooled_delta("TP53", [(weak, "control")], AXIS, var_floor="poisson",
                       shrinkage=False, coverage_tiers=((3.0, 0.10), (10.0, 0.50)))
    assert out[0] != 0.0


def test_a_source_with_no_cells_keeps_full_weight_and_is_counted():
    """An `LfcTable` source keeps full weight and is counted as unweighted."""
    lfc = LfcTable(labels=["TP53"], genes=AXIS, lfc=np.array([[1.0, 0.0, 0.0]]),
                   var=np.array([[0.25, 0.25, 0.25]]), source="test-lfc")
    stats: dict = {}
    out = pooled_delta("TP53", [lfc], AXIS, shrinkage=False,
                       coverage_tiers=((3.0, 0.10),), stats=stats)
    assert out[0] == pytest.approx(1.0)
    assert stats["coverage_sources_unweighted"] == 1
    assert "coverage_gene_arms" not in stats


def test_the_stats_count_what_was_demoted():
    pb = _pb_from_cells({
        "control": [[100, 100, 100]] * 8,
        "TP53": [[200, 100, 50]] + [[0, 100, 50]] * 7,   # gene A thin, B and C dense
    })
    stats: dict = {}
    pooled_delta("TP53", [(pb, "control")], AXIS, var_floor="poisson",
                 coverage_tiers=((3.0, 0.10),), stats=stats)
    assert stats["coverage_gene_arms"] == 3
    assert stats["coverage_gene_arms_demoted"] == 1


def test_n_eff_matches_the_true_count_on_a_real_sparse_matrix():
    """On a sparse matrix, n_eff is at most the detecting-cell count and positive wherever a cell detected the gene."""
    rng = np.random.default_rng(7)
    X = sp.random(200, 30, density=0.2, random_state=rng, data_rvs=lambda n: rng.integers(1, 20, n))
    dense = np.asarray(X.todense(), dtype=float)
    dense[:, 0] = 7                                     # no empty libraries
    pb = _pb_from_cells({"X": dense.tolist()}, genes=tuple(f"g{i}" for i in range(30)))
    true_k = (dense > 0).sum(axis=0)
    ne = pb.n_eff(0)
    assert np.all(ne <= true_k + 1e-9)
    assert np.all(ne[true_k > 0] > 0)
