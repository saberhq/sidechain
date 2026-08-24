"""Tests for the (lfc, var) source type -- a corpus that ships the contrast taken.

The assertions worth keeping here are all about the variance, because the fold
changes are simply read off the file. A source with no cells behind it has to
have its weight derived, and the derivation has exactly two ways to go wrong:
give a thin source too much weight, or give a real effect none.
"""
from __future__ import annotations

import gzip

import numpy as np
import pytest

from sidechain.data.lfc_table import (
    LfcTable,
    _nominal_from_bh,
    read_feng_lfc,
    variance_from_pvalue,
)

HEADER = "Target\tExpressed_Gene_Symbol\tExpressed_Gene_Ens_ID\tlfc\tpval_adj\n"


def _write(tmp_path, rows, name="lfc.tsv.gz", header=HEADER):
    path = tmp_path / name
    body = header + "".join("\t".join(str(c) for c in r) + "\n" for r in rows)
    if name.endswith(".gz"):
        path.write_bytes(gzip.compress(body.encode()))
    else:
        path.write_text(body)
    return path


# ------------------------------------------------------- the variance itself --


def test_a_saturated_pvalue_abstains_rather_than_voting_toward_zero():
    """The failure this whole module is shaped around.

    `se = |lfc| / z` on a near-zero lfc gives a near-zero standard error and so
    an enormous weight -- Feng would dominate the pool on precisely the genes it
    measured nothing on. 98.9 % of its rows are that case. Infinite variance
    makes the weight exactly 0, so it abstains instead.
    """
    lfc = np.array([0.001, 0.02, -0.005])
    p = np.array([0.999999927, 0.9999, 1.0])
    var = variance_from_pvalue(lfc, p)
    assert np.isinf(var).all()
    assert (1.0 / var == 0).all()          # the weight, which is what pooling uses


def test_a_significant_row_gets_a_finite_variance_that_scales_with_the_effect():
    """A real effect must keep a usable weight, or the source contributes nothing
    at all and there was no point ingesting it.

    Computed as two families of one, deliberately. Put both rows in a single
    family and the BH inversion ranks them 1 and 2, giving equal adjusted
    p-values *different* nominal ones -- which is the inversion working, but it
    would confound the thing under test here.
    """
    one = variance_from_pvalue(np.array([1.0]), np.array([1e-8]))
    two = variance_from_pvalue(np.array([2.0]), np.array([1e-8]))
    assert np.isfinite(one).all() and np.isfinite(two).all()
    assert one[0] > 0 and two[0] > 0
    # same p, twice the effect -> twice the standard error
    assert np.isclose(np.sqrt(two[0]) / np.sqrt(one[0]), 2.0, rtol=1e-9)


def test_a_weaker_pvalue_at_the_same_effect_size_means_more_variance():
    """Monotonicity is the property that makes this a weighting at all."""
    lfc = np.array([1.0, 1.0, 1.0])
    # distinct ranks so the BH inversion does not tie them
    p = np.array([1e-10, 1e-4, 0.5])
    var = variance_from_pvalue(lfc, p)
    assert var[0] < var[1] < var[2]


def test_a_zero_effect_never_produces_a_weight():
    """`se = |lfc|/z` is 0 when lfc is 0, which would be an infinite weight on a
    gene the source says does not move."""
    var = variance_from_pvalue(np.array([0.0]), np.array([1e-12]))
    assert np.isinf(var[0])


def test_bh_inversion_recovers_nominal_p_where_the_step_up_did_not_bind():
    """BH with a non-binding running minimum is p_adj_(i) = (m/i) * p_(i)."""
    m = 5
    nominal = np.array([0.001, 0.010, 0.030, 0.060, 0.500])
    adj = np.array([min(1.0, nominal[i] * m / (i + 1)) for i in range(m)])
    # the step-up's cumulative min from the top, so this is a real BH output
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    assert np.allclose(_nominal_from_bh(adj), nominal, rtol=1e-9)


def test_bh_inversion_is_order_independent():
    """The file is sorted by target then gene, not by p, so the inversion has to
    rank internally rather than assume the rows arrive sorted."""
    adj = np.array([0.9, 0.02, 0.5, 0.004])
    shuffled = adj[[3, 1, 2, 0]]
    got = _nominal_from_bh(adj)[[3, 1, 2, 0]]
    assert np.allclose(got, _nominal_from_bh(shuffled))


# ------------------------------------------------------------- reading a file --


def test_reads_a_feng_shaped_table_into_a_matrix(tmp_path):
    path = _write(tmp_path, [
        ("AAMP", "TSPAN6", "ENSG00000000003", 1.5, 1e-9),
        ("AAMP", "DPM1", "ENSG00000000419", -0.2, 0.999999927),
        ("BRCA1", "TSPAN6", "ENSG00000000003", 0.4, 1e-5),
        ("BRCA1", "DPM1", "ENSG00000000419", 2.0, 1e-7),
    ])
    t = read_feng_lfc(path)
    assert t.labels == ["AAMP", "BRCA1"]
    assert sorted(t.genes) == ["DPM1", "TSPAN6"]
    assert t.lfc.shape == (2, 2)

    fc, var = t.effect("AAMP")
    g = {name: i for i, name in enumerate(t.genes)}
    assert fc[g["TSPAN6"]] == 1.5
    assert np.isfinite(var[g["TSPAN6"]])
    assert np.isinf(var[g["DPM1"]])          # saturated -> abstain
    assert t.effect("NOT_A_TARGET") is None


def test_keep_filters_while_streaming(tmp_path):
    """The genome-wide table is 43 M rows; holding all 6,673 targets is ~691 MB
    against ~19 MB for the 182 that touch the panel. Filtering afterwards would
    defeat the point, so it has to happen during the read."""
    path = _write(tmp_path, [
        ("AAMP", "TSPAN6", "ENSG00000000003", 1.5, 1e-9),
        ("BRCA1", "TSPAN6", "ENSG00000000003", 0.4, 1e-5),
        ("ZZZ3", "TSPAN6", "ENSG00000000003", 0.1, 1e-2),
    ])
    t = read_feng_lfc(path, keep={"AAMP", "ZZZ3"})
    assert t.labels == ["AAMP", "ZZZ3"]


def test_per_line_table_selects_one_context(tmp_path):
    """`Cell_Line` makes context a per-ROW property in the targeted table. Reading
    it without selecting a line would pool 19 lines into one context -- the exact
    mistake report 07 caught the genome-wide arm being described with."""
    header = ("Target\tExpressed_Gene_Symbol\tExpressed_Gene_Ens_ID\t"
              "Cell_Line\twt_expr\tlfc\tpval_adj\n")
    path = _write(tmp_path, [
        ("AAMP", "TSPAN6", "ENSG00000000003", "eipl_1", 0.69, 1.5, 1e-9),
        ("AAMP", "TSPAN6", "ENSG00000000003", "kolf_2", 0.70, -0.9, 1e-9),
    ], header=header)

    t = read_feng_lfc(path, context_col="Cell_Line", context="eipl_1")
    assert t.effect("AAMP")[0][0] == 1.5
    assert t.notes["rows_skipped_other_context"] == 1

    other = read_feng_lfc(path, context_col="Cell_Line", context="kolf_2")
    assert other.effect("AAMP")[0][0] == -0.9


def test_a_renamed_column_raises_rather_than_returning_an_empty_table(tmp_path):
    """Same strictness as `loaders.gene_index`: a silently empty source looks
    exactly like a source that legitimately covers nothing."""
    path = _write(tmp_path, [("AAMP", "TSPAN6", "ENSG00000000003", 1.5, 1e-9)],
                  header="Target\tGene\tEns\tlogFC\tFDR\n")
    with pytest.raises(KeyError, match="lacks column"):
        read_feng_lfc(path)


def test_the_bh_family_is_the_rows_the_target_actually_has(tmp_path):
    """A target measured on fewer genes must not be given a family it never had:
    the rank/m inversion would then read every one of its p-values as smaller
    than it is, and inflate its weight."""
    # G0 is the SMALLEST adjusted p in WIDE's family, so rank 1 of 6. The other
    # five sit at 0.5, which keeps the sequence non-decreasing and so a valid
    # BH output rather than an impossible one.
    rows = [("WIDE", "G0", "ENSG00000000000", 1.0, 1e-2)]
    rows += [("WIDE", f"G{i}", f"ENSG{i:011d}", 1.0, 0.5) for i in range(1, 6)]
    rows += [("NARROW", "G0", "ENSG00000000000", 1.0, 1e-2)]
    t = read_feng_lfc(_write(tmp_path, rows))
    g = {name: i for i, name in enumerate(t.genes)}
    wide_var = t.effect("WIDE")[1][g["G0"]]
    narrow_var = t.effect("NARROW")[1][g["G0"]]
    # NARROW's single test is its whole family (m=1, rank=1) -> nominal p = 0.01.
    # WIDE's G0 is rank 1 of 6 -> nominal p = 0.01/6, a smaller p, a larger z,
    # and so a SMALLER variance. Same lfc, same adjusted p, different families.
    assert wide_var < narrow_var


def test_roundtrips_through_npz(tmp_path):
    path = _write(tmp_path, [
        ("AAMP", "TSPAN6", "ENSG00000000003", 1.5, 1e-9),
        ("AAMP", "DPM1", "ENSG00000000419", -0.2, 0.9999999),
    ])
    t = read_feng_lfc(path, source="feng", context="ipsc_pooled")
    t.save(tmp_path / "c.npz")
    back = LfcTable.load(tmp_path / "c.npz")
    assert back.labels == t.labels
    assert np.array_equal(back.genes, t.genes)
    assert np.allclose(back.lfc, t.lfc)
    assert np.array_equal(np.isinf(back.var), np.isinf(t.var))
    assert back.source == "feng" and back.context == "ipsc_pooled"


def test_n_usable_counts_genes_carrying_a_weight(tmp_path):
    path = _write(tmp_path, [
        ("AAMP", "A", "E1", 1.5, 1e-9),
        ("AAMP", "B", "E2", 0.1, 0.9999999),
        ("AAMP", "C", "E3", 2.0, 1e-6),
    ])
    t = read_feng_lfc(path)
    assert t.n_usable.tolist() == [2]


def test_a_tiny_pvalue_never_produces_an_infinite_weight():
    """The numerical door into this module's central failure.

    `1 - p/2` rounds to exactly 1.0 in float64 below p ~ 1e-16, so computing z
    as `ndtri(1 - p/2)` returns inf, se becomes 0, and the weight 1/var is
    INFINITE -- on precisely the genes the source is most confident about. On
    the real Feng cache that hit 35 genes across 34 of 182 targets and each one
    silently overrode every other source on that gene.
    """
    lfc = np.array([1.0, 1.0, 1.0, 1.0])
    p = np.array([1e-2, 1e-20, 1e-100, 1e-300])
    var = variance_from_pvalue(lfc, p)
    assert np.isfinite(var).all(), "a tiny p must not yield an infinite weight"
    assert (var > 0).all(), "a zero variance IS an infinite weight"
    # and it stays monotone: a smaller p is still more certain
    assert var[0] > var[1] > var[2] > var[3]


def test_no_finite_variance_is_ever_zero(tmp_path):
    """A zero variance is an infinite weight wearing a finite mask -- the pooling
    site checks `isfinite(var)`, which a 0.0 passes."""
    rows = [("T", f"G{i}", f"E{i}", 2.5, 10.0 ** -(3 * i + 3)) for i in range(8)]
    t = read_feng_lfc(_write(tmp_path, rows))
    fin = np.isfinite(t.var)
    assert fin.any()
    assert (t.var[fin] > 0).all()
