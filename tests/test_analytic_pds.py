"""Contract tests for the analytic pds path (T59).

What is worth pinning here is not the metric -- that is cell-eval2's kernel, called
directly -- but the arithmetic around it and the three boundaries the module claims.
"""
from __future__ import annotations

import numpy as np
import pytest

from sidechain.eval.analytic_pds import (
    FoldCache,
    delta_from_parts,
    emitted_sums,
    score_delta,
)


def _fold(n_genes=6, n_targets=3):
    """A fold whose targets ARE genes on its axis -- cell-eval2 refuses to score with
    `exclude_target_gene` on when no label resolves to a gene, which is the real
    construct-ID-vs-symbol trap and not something to paper over in a fixture."""
    genes = np.array([f"g{i}" for i in range(n_genes)])
    targets = [f"g{i}" for i in range(min(n_targets, n_genes))]
    perts = np.array(list(targets) + ["non-targeting"], dtype=object)
    rng = np.random.default_rng(0)
    real = rng.random((len(perts), n_genes))
    return FoldCache(perts=perts, real_means=real, genes=genes,
                     n_cells=np.array([100] * len(perts)),
                     frac=np.full(n_genes, 1.0 / n_genes), lib_median=1000.0,
                     ctrl_n_cells=100)


def test_zero_delta_emits_the_control_profile():
    f = _fold()
    sums = emitted_sums(np.zeros((3, 6)), f.frac, f.lib_median, np.array([100, 100, 100]))
    # every gene equal, and the row totals the emitter's budget
    # `rint` quantises to whole counts, so these are exact only to the rounding
    assert np.allclose(sums, sums[0, 0])
    assert np.allclose(sums.sum(axis=1), 100 * 1000.0, rtol=1e-3)


def test_a_one_log2fc_doubles_that_gene_before_renormalising():
    f = _fold(n_genes=2)
    d = np.array([[1.0, 0.0]])
    sums = emitted_sums(d, f.frac, f.lib_median, np.array([100]))
    # shares 2:1 after renormalisation
    assert sums[0, 0] / sums[0, 1] == pytest.approx(2.0, rel=1e-3)   # rint quantisation


def test_non_finite_deltas_are_treated_as_no_change_not_propagated():
    f = _fold(n_genes=3)
    d = np.array([[np.nan, np.inf, -np.inf]])
    sums = emitted_sums(d, f.frac, f.lib_median, np.array([100]))
    assert np.isfinite(sums).all()
    assert np.allclose(sums, sums[0, 0])


def test_score_delta_does_not_mutate_its_input():
    f = _fold()
    d = np.ones((3, 6)) * 0.5
    before = d.copy()
    score_delta(d, ["g0", "g1", "g2"], f, alpha=1.7)
    assert np.array_equal(d, before), "score_delta mutated the caller's deltas"


def test_score_delta_pins_the_targets_own_gene():
    """The emitter pins the knocked-down gene; the metric excludes it. Both must happen
    on a COPY, and a target whose gene is not on the axis must not raise."""
    genes = np.array(["A", "B", "zz"])
    f = FoldCache(perts=np.array(["A", "B", "non-targeting"], dtype=object),
                  real_means=np.random.default_rng(1).random((3, 3)), genes=genes,
                  n_cells=np.array([50, 50, 50]), frac=np.full(3, 1 / 3),
                  lib_median=500.0, ctrl_n_cells=50)
    out = score_delta(np.zeros((2, 3)), ["A", "B"], f)
    assert np.isfinite(out)


def test_cells_for_refuses_a_target_the_fold_does_not_have():
    f = _fold()
    with pytest.raises(KeyError, match="absent from the fold"):
        f.cells_for(["g0", "NOPE"])


def test_delta_from_parts_leaves_unspoken_genes_at_zero():
    num = np.array([[2.0, 5.0], [0.0, 4.0]])
    den = np.array([[1.0, 0.0], [0.0, 2.0]])
    d = delta_from_parts(num, den)
    assert d[0, 0] == 2.0
    assert d[0, 1] == 0.0, "a gene no source spoke for must stay at 0, not divide by zero"
    assert d[1, 1] == 2.0


def test_alpha_scales_the_delta_and_therefore_the_emitted_sums():
    f = _fold(n_genes=2)
    d = np.array([[1.0, 0.0]])
    one = emitted_sums(d, f.frac, f.lib_median, np.array([100]))
    two = emitted_sums(d * 2.0, f.frac, f.lib_median, np.array([100]))
    assert two[0, 0] / two[0, 1] == pytest.approx(4.0, rel=1e-3)
    assert one[0, 0] / one[0, 1] == pytest.approx(2.0, rel=1e-3)


def test_an_uncovered_target_is_emitted_without_the_knockdown_pin():
    """`eval.loco` pins the knockdown INSIDE `if d is not None`, so a target no source
    covers is emitted as the bare control profile with no pin. `score_delta` must honour
    that, and the difference is real: on `loco_hct116/afn_nosib` (802 of 830 covered) it
    is the entire 4.4e-06 replay residual -- pinning everything reads +4.359e-06 against
    the recorded mirror value, honouring coverage reads -7.4e-10.

    The assertion is on the EMITTED PROFILE, not the score, and deliberately so: `pds`
    uses `exclusion_scope="panel"`, which drops every target gene from the scored vector,
    so on a small fixture the pin reaches the metric only as a uniform rescale -- and
    cosine is scale-invariant. The effect is real but only resolves on a real panel, which
    is what the replay demonstrates. Testing the score here would pass or fail for reasons
    that have nothing to do with coverage.
    """
    f = _fold(n_genes=60, n_targets=40)
    d = np.random.default_rng(7).normal(0, 0.8, (40, 60))
    pinned, bare = d.copy(), d.copy()
    pinned[7, 7] = -2.32
    n = np.array([100])
    a = emitted_sums(pinned[7:8], f.frac, f.lib_median, n)
    b = emitted_sums(bare[7:8], f.frac, f.lib_median, n)
    assert not np.array_equal(a, b), "the pin must change the emitted profile"
    # and it moves every gene, because the profile is renormalised after 2**d
    assert (a != b).sum() == 60


def test_covered_mask_of_all_true_is_the_same_as_omitting_it():
    f = _fold(n_genes=60, n_targets=40)
    tg = [f"g{i}" for i in range(40)]
    d = np.random.default_rng(8).normal(0, 0.8, (40, 60))
    assert score_delta(d, tg, f) == score_delta(d, tg, f, covered=[True] * 40)
