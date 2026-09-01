"""`n_eff` and the coverage-tiered pooling weights.

`n_eff` answers the question `count_sum` cannot: a gene whose counts add up to 100 might
be 100 cells at 1 each or one cell at 100. The properties worth pinning are the ones the
weighting rule leans on -- that it never overstates the number of contributing cells, that
it reads the two endpoints exactly, and that no tier can silence a gene.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from sidechain.data.lfc_table import LfcTable
from sidechain.data.screen_coverage import (
    _poisson_floored_weight,
    arm_coverage,
    gene_coverage,
)
from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.submit.build import (
    _log2fc_with_var,
    coverage_factor,
    parse_coverage_tiers,
    pooled_delta,
)

AXIS = np.array(["A", "B", "C"])


def _pb_from_cells(cells_by_label, genes=("A", "B", "C")):
    """Build a PseudobulkSums by actually summing per-cell CPMs, so `n_eff` is checked
    against the arithmetic it claims to summarise rather than against a fixture."""
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
    """k cells contributing equally is the equality case: n_eff is exactly k."""
    pb = _pb_from_cells({"X": [[10, 0, 0], [10, 0, 0], [10, 0, 0], [10, 0, 0]]})
    assert pb.n_eff(0)[0] == pytest.approx(4.0)


def test_one_cell_carrying_everything_reads_as_one():
    """The failure this exists to catch: gene A's counts add up to 40 either way, and
    only n_eff separates 'four cells agreed' from 'one cell, once'."""
    pb = _pb_from_cells({"X": [[40, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]]})
    assert pb.n_eff(0)[0] == pytest.approx(1.0)


def test_never_exceeds_the_number_of_detecting_cells():
    """The property the whole rule rests on: n_eff is a FLOOR on the cell count, so a
    thin gene can never be promoted by it. Checked against the true count on random
    zero-inflated arms."""
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
    """A zero factor would zero the denominator on genes every source calls weak, and
    the pooled delta comes out 0 -- which the emitter replays as 'no change' and `fid`
    charges as silence. Downweighting is the whole point; refuse the gate."""
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
    """`coverage_tiers=None` must not perturb a single float -- every scored run to date
    was produced without this knob and has to stay reproducible."""
    pb = _pb_from_cells({"control": [[100, 100, 100]] * 8, "TP53": [[200, 100, 50]] * 8})
    before = pooled_delta("TP53", [(pb, "control")], AXIS, var_floor="poisson")
    after = pooled_delta("TP53", [(pb, "control")], AXIS, var_floor="poisson",
                         coverage_tiers=None)
    assert np.array_equal(before, after)


def test_a_thin_source_loses_ground_to_a_dense_one():
    """The behaviour the knob exists for. Gene A: `thin` saw it in one cell and shouts,
    `dense` saw it in many and does not. Tiering must move the pooled value toward
    `dense` without silencing anything."""
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
    """Even where EVERY source is weak, the gene keeps a value. A hard filter would
    return 0 here, and 0 is an assertion of 'no change', not an abstention."""
    weak = _pb_from_cells({
        "control": [[300, 50, 50]] + [[0, 50, 50]] * 5,
        "TP53": [[600, 50, 50]] + [[0, 50, 50]] * 5,
    })
    out = pooled_delta("TP53", [(weak, "control")], AXIS, var_floor="poisson",
                       shrinkage=False, coverage_tiers=((3.0, 0.10), (10.0, 0.50)))
    assert out[0] != 0.0


def test_a_source_with_no_cells_keeps_full_weight_and_is_counted():
    """An LfcTable publishes a contrast with no cells behind it, so it defines no n_eff.
    It must pass through unweighted -- and the run must say so rather than looking
    fully tiered."""
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


# --------------------------------------------------------- the QC reporter --


def test_report_weight_matches_the_pooling_weight_it_claims_to_share():
    """`screen_coverage` recomputes the poisson-floored weight rather than importing the
    submission path. Pin the two together so the report cannot drift into describing a
    weighting the pool does not use."""
    pb = _pb_from_cells({"control": [[100, 90, 80]] * 5, "TP53": [[200, 90, 40]] * 5})
    _, var = _log2fc_with_var(pb, "TP53", "control", var_floor="poisson")
    expected = 1.0 / np.maximum(var, 1e-12)
    got = _poisson_floored_weight(pb, pb.labels.index("TP53"), pb.labels.index("control"))
    assert np.allclose(got, expected)


def test_report_excludes_the_control_arm_from_the_arm_summary():
    """A pooled control is tens of thousands of cells; leaving it in would move every
    summary statistic and hide the thin tail the report exists to show."""
    pb = _pb_from_cells({"control": [[10, 10, 10]] * 40, "A": [[10, 10, 10]] * 3,
                         "B": [[10, 10, 10]] * 5})
    a = arm_coverage(pb, "control")
    assert a["arms"] == 2
    assert a["cells_max"] == 5


def test_report_tiers_cover_every_gene_arm_exactly_once():
    pb = _pb_from_cells({"control": [[100, 100, 100]] * 6,
                         "A": [[200, 0, 50]] + [[0, 0, 50]] * 5,
                         "B": [[100, 100, 100]] * 6})
    g = gene_coverage(pb, "control", (3.0, 10.0))
    assert sum(t["gene_arms"] for t in g["tiers"]) == g["arms_measured"] * g["genes"]


def test_report_survives_a_sparse_arm_without_dividing_by_zero():
    """A gene no cell of either arm expressed has no weight and no fold change; the
    report must still produce a valid row for it."""
    pb = _pb_from_cells({"control": [[10, 0, 0]] * 4, "A": [[10, 0, 0]] * 4})
    g = gene_coverage(pb, "control", (3.0, 10.0))
    assert all(np.isfinite(t["weight_pct"]) for t in g["tiers"])


def test_n_eff_matches_the_true_count_on_a_real_sparse_matrix():
    """End to end on a CSR matrix built the way a corpus stores one, to check the
    accessor against the count computed from the cells themselves."""
    rng = np.random.default_rng(7)
    X = sp.random(200, 30, density=0.2, random_state=rng, data_rvs=lambda n: rng.integers(1, 20, n))
    dense = np.asarray(X.todense(), dtype=float)
    dense[:, 0] = 7                                     # no empty libraries
    pb = _pb_from_cells({"X": dense.tolist()}, genes=tuple(f"g{i}" for i in range(30)))
    true_k = (dense > 0).sum(axis=0)
    ne = pb.n_eff(0)
    assert np.all(ne <= true_k + 1e-9)
    assert np.all(ne[true_k > 0] > 0)
