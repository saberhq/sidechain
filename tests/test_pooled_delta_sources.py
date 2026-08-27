"""`pooled_delta` taking two kinds of source, and what must not change.

Feng is the first corpus that publishes a contrast instead of cells, so pooling
had to stop assuming every source is a `PseudobulkSums`. Two properties matter
enough to pin: every existing call site keeps working untouched, and a source
that abstains contributes nothing rather than contributing a zero.
"""
from __future__ import annotations

import numpy as np

from sidechain.data.lfc_table import LfcTable
from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.submit.build import as_delta_source, pooled_delta

AXIS = np.array(["A", "B", "C"])


def _pb(labels, cpm, n_cells=1000, genes=("A", "B", "C")):
    """A PseudobulkSums with the given mean CPM per label and a fixed spread."""
    m = np.asarray(cpm, dtype=float)
    n = np.full(len(labels), n_cells, dtype=np.int64)
    cpm_sum = m * n[:, None]
    # var_cpm = cpm_sq/n - mean^2; pick cpm_sq so the per-gene variance is 1.0
    cpm_sq_sum = (m**2 + 1.0) * n[:, None]
    return PseudobulkSums(
        labels=list(labels), genes=np.array(genes),
        count_sum=cpm_sum.copy(), cpm_sum=cpm_sum, cpm_sq_sum=cpm_sq_sum,
        n_cells=n, libsize_sum=n.astype(float) * 1e4, sources=["test"],
    )


def _lfc(labels, lfc, var, genes=("A", "B", "C")):
    return LfcTable(labels=list(labels), genes=np.array(genes),
                    lfc=np.asarray(lfc, float), var=np.asarray(var, float),
                    source="test-lfc")


# --------------------------------------------------- nothing existing changes --


def test_the_historical_tuple_form_still_works():
    """`loco.py` and `build.py` both construct `[(PseudobulkSums, control), ...]`
    and neither was touched. If this breaks, every scored run breaks."""
    pb = _pb(["control", "TP53"], [[100.0, 100.0, 100.0], [200.0, 100.0, 50.0]])
    out = pooled_delta("TP53", [(pb, "control")], AXIS, shrinkage=False)
    assert out is not None
    assert out[0] > 0        # 100 -> 200 CPM
    assert abs(out[1]) < 1e-9
    assert out[2] < 0        # 100 -> 50 CPM


def test_a_target_no_source_covers_is_still_none():
    """None is what makes `build.py` fall back to the mean shift. A source type
    that returned zeros instead would silently replace that fallback with a
    prediction of 'nothing happens'."""
    pb = _pb(["control", "TP53"], [[100.0, 100.0, 100.0], [200.0, 100.0, 50.0]])
    assert pooled_delta("BRCA1", [(pb, "control")], AXIS) is None


def test_as_delta_source_rejects_something_that_is_neither():
    import pytest

    with pytest.raises(TypeError, match="not a delta source"):
        as_delta_source(object())


# --------------------------------------------------------- the new source type --


def test_an_lfc_table_pools_alongside_a_pseudobulk():
    """The point of the whole exercise: one call, two kinds of evidence."""
    pb = _pb(["control", "TP53"], [[100.0, 100.0, 100.0], [200.0, 100.0, 100.0]])
    tab = _lfc(["TP53"], [[2.0, 0.0, 0.0]], [[0.01, np.inf, np.inf]])

    both = pooled_delta("TP53", [(pb, "control"), tab], AXIS, shrinkage=False)
    only_pb = pooled_delta("TP53", [(pb, "control")], AXIS, shrinkage=False)
    assert both is not None and only_pb is not None
    # gene A: the table is far more certain (var 0.01) so it pulls the pool up
    assert both[0] > only_pb[0]
    # genes B and C: the table abstains, so the pool is unchanged
    assert np.allclose(both[1:], only_pb[1:])


def test_an_abstaining_gene_contributes_no_weight_at_all():
    """`var = inf` must mean 'no vote', not 'a vote for zero'. If it voted, a
    thin source would drag a well-powered source's real effect toward 0 using
    evidence it does not have -- which at Feng's median 25 cells per target is
    the difference between a useful prior and a harmful one."""
    pb = _pb(["control", "TP53"], [[100.0, 100.0, 100.0], [400.0, 400.0, 400.0]])
    abstains = _lfc(["TP53"], [[0.0, 0.0, 0.0]], [[np.inf, np.inf, np.inf]])

    with_it = pooled_delta("TP53", [(pb, "control"), abstains], AXIS, shrinkage=False)
    without = pooled_delta("TP53", [(pb, "control")], AXIS, shrinkage=False)
    assert np.allclose(with_it, without)


def test_a_target_only_a_fully_abstaining_source_covers_returns_zeros_not_none():
    """A real distinction: the source HAS this target and reports no detectable
    change anywhere. That is different from no source covering it, and the two
    must not collapse -- None routes to the mean-shift fallback, which would be
    a fabricated effect."""
    tab = _lfc(["TP53"], [[0.1, 0.2, 0.3]], [[np.inf, np.inf, np.inf]])
    out = pooled_delta("TP53", [tab], AXIS)
    assert out is not None
    assert np.allclose(out, 0.0)


def test_pooling_is_inverse_variance_across_two_lfc_tables():
    """Two tables, same gene, different certainty -> the certain one dominates,
    and the pooled value is the exact inverse-variance mean."""
    a = _lfc(["TP53"], [[1.0, 0.0, 0.0]], [[1.0, np.inf, np.inf]])
    b = _lfc(["TP53"], [[3.0, 0.0, 0.0]], [[1.0 / 3.0, np.inf, np.inf]])
    out = pooled_delta("TP53", [a, b], AXIS, shrinkage=False)
    expected = (1.0 * 1.0 + 3.0 * 3.0) / (1.0 + 3.0)
    assert np.isclose(out[0], expected)


def test_a_source_on_a_different_gene_axis_is_remapped_not_misaligned():
    """Feng's axis is 6,472 genes against the challenge's 18,533. Pooling by
    position rather than by name would silently pair unrelated genes."""
    tab = LfcTable(labels=["TP53"], genes=np.array(["C", "A"]),
                   lfc=np.array([[5.0, 1.0]]), var=np.array([[1.0, 1.0]]),
                   source="reordered")
    out = pooled_delta("TP53", [tab], AXIS, shrinkage=False)
    assert np.isclose(out[0], 1.0)     # A
    assert np.isclose(out[1], 0.0)     # B, absent from the source
    assert np.isclose(out[2], 5.0)     # C


def test_shrinkage_on_an_abstaining_gene_does_not_produce_nan():
    """`shrink` divides by fc^2 and subtracts var/fc^2; with var = inf that is
    -inf before the clip. It must clip to zero, not propagate a nan into the
    submission."""
    tab = _lfc(["TP53"], [[1.0, 2.0, 0.0]], [[np.inf, 0.01, np.inf]])
    out = pooled_delta("TP53", [tab], AXIS, shrinkage=True)
    assert np.isfinite(out).all()
    assert np.isclose(out[0], 0.0)
    assert out[1] > 0


# ----------------------------------------------------- the variance floor --


def _pb_var(labels, cpm, var, n_cells=150, lib_per_cell=20000.0, genes=("A", "B", "C")):
    """A PseudobulkSums with chosen per-gene observed variance per label.

    `n_cells` may be a scalar or per-label; `lib_per_cell` sets the mean library
    size the Poisson floor reads.
    """
    m = np.asarray(cpm, dtype=float)
    v = np.asarray(var, dtype=float)
    n = np.asarray(n_cells if np.ndim(n_cells) else [n_cells] * len(labels), dtype=np.int64)
    return PseudobulkSums(
        labels=list(labels), genes=np.array(genes),
        count_sum=m * n[:, None], cpm_sum=m * n[:, None],
        cpm_sq_sum=(m**2 + v) * n[:, None],
        n_cells=n, libsize_sum=n.astype(float) * lib_per_cell,
    )


def test_without_the_floor_a_zero_spread_gene_pins_the_pool():
    """The measured pathology (research idea: inverse-variance weight flooring),
    pinned as the DEFAULT so the shipped arms stay bit-for-bit reproducible: a
    gene whose observed spread is exactly 0 hits the 1e-6 clamp, takes weight
    1e6, and erases another source's real +2 effect down to ~nothing."""
    flat = _pb_var(["control", "TP53"], [[100.0] * 3, [100.0] * 3], [[0.0] * 3, [0.0] * 3])
    # Poisson-realistic spread at 20k UMI: var_cpm = mean * 1e6/libsize = mean * 50
    real = _pb_var(["control", "TP53"], [[100.0] * 3, [400.0] * 3],
                   [[5000.0] * 3, [20000.0] * 3])
    stats: dict = {}
    out = pooled_delta("TP53", [(flat, "control"), (real, "control")], AXIS,
                       shrinkage=False, stats=stats)
    assert abs(out[0]) < 0.05                     # ~99.8 % erasure of a ~2.0 log2FC
    assert stats["gene_weights_var_le_1e-6"] == 3  # every zero-spread gene sat at the clamp


def test_the_poisson_floor_unpins_zero_spread_genes():
    """Same two sources with `var_floor='poisson'`: the flat source keeps a vote
    sized by its sampling variance instead of a 1e6 veto, so the pooled value
    lands between the sources -- and nothing sits at the old clamp any more."""
    flat = _pb_var(["control", "TP53"], [[100.0] * 3, [100.0] * 3], [[0.0] * 3, [0.0] * 3])
    real = _pb_var(["control", "TP53"], [[100.0] * 3, [400.0] * 3],
                   [[5000.0] * 3, [20000.0] * 3])
    stats: dict = {}
    out = pooled_delta("TP53", [(flat, "control"), (real, "control")], AXIS,
                       shrinkage=False, var_floor="poisson", stats=stats)
    fc_real = pooled_delta("TP53", [(real, "control")], AXIS, shrinkage=False,
                           var_floor="poisson")
    assert 0.5 < out[0] < fc_real[0]
    assert stats["gene_weights_var_le_1e-6"] == 0


def test_a_single_cell_arm_dominates_without_the_floor_and_abstains_with_it():
    """A label seen once computes observed variance identically 0 (population
    form), so without the floor its pure-noise fold change outvotes a 150-cell
    source; with the floor the arm abstains and the pool is the other source
    alone. This is the 150x-weight arm from the idea file's measurements."""
    once = _pb_var(["control", "TP53"], [[100.0] * 3, [800.0] * 3], [[0.0] * 3, [0.0] * 3],
                   n_cells=[150, 1])
    real = _pb_var(["control", "TP53"], [[100.0] * 3, [200.0] * 3], [[1.0] * 3, [1.0] * 3])

    unfloored = pooled_delta("TP53", [(once, "control"), (real, "control")], AXIS, shrinkage=False)
    only_real = pooled_delta("TP53", [(real, "control")], AXIS, shrinkage=False)
    assert unfloored[0] > only_real[0] + 0.5      # the 1-cell arm dragged the pool its way

    floored = pooled_delta("TP53", [(once, "control"), (real, "control")], AXIS,
                           shrinkage=False, var_floor="poisson")
    floored_real = pooled_delta("TP53", [(real, "control")], AXIS, shrinkage=False,
                                var_floor="poisson")
    assert np.allclose(floored, floored_real)


def test_a_target_covered_only_by_a_single_cell_arm_reports_zero_weight_not_fallback():
    """Abstention must not collapse into the None -> mean-shift fallback: the
    source covers the target, so the pool returns a genuine all-zero delta and
    the stats count the target, which is how the A/B reports coverage cost."""
    once = _pb_var(["control", "TP53"], [[100.0] * 3, [800.0] * 3], [[0.0] * 3, [0.0] * 3],
                   n_cells=[150, 1])
    stats: dict = {}
    out = pooled_delta("TP53", [(once, "control")], AXIS, shrinkage=False,
                       var_floor="poisson", stats=stats)
    assert out is not None and np.allclose(out, 0.0)
    assert stats["source_arms_abstained"] == 1
    assert stats["targets_zero_weight"] == 1


def test_var_floor_none_leaves_every_existing_number_untouched():
    """The knob defaults off, and off means bit-identical: the same pool with
    `var_floor='none'` passed explicitly equals the historical call."""
    pb = _pb(["control", "TP53"], [[100.0, 100.0, 100.0], [200.0, 100.0, 50.0]])
    a = pooled_delta("TP53", [(pb, "control")], AXIS, shrinkage=False)
    b = pooled_delta("TP53", [(pb, "control")], AXIS, shrinkage=False, var_floor="none")
    assert np.array_equal(a, b)


def test_an_unknown_var_floor_is_refused_not_ignored():
    import pytest

    pb = _pb(["control", "TP53"], [[100.0] * 3, [200.0] * 3])
    with pytest.raises(ValueError, match="var_floor"):
        pooled_delta("TP53", [(pb, "control")], AXIS, var_floor="possion")


# ------------------------------------------------- the loco entry point --


def test_loco_requires_at_least_one_source_but_not_a_pseudobulk_one():
    """An LfcTable is a complete source, so an arm built only from published
    contrasts is a legitimate run -- and scoring one is how you learn what that
    corpus is worth alone. `--source` was `required=True`, which made that arm
    unrunnable; it was caught only when the GPU box refused it mid-run.

    An arm with NO sources at all stays an error: it would silently score the
    fallback shift and look like a model.
    """
    import pytest

    from sidechain.eval import loco

    with pytest.raises(SystemExit) as exc:
        loco.main(["--real", "r.h5ad", "--bundle", "b", "--out", "o"])
    assert exc.value.code == 2      # argparse error, not a traceback


def test_row_wise_log2fc_matches_the_whole_matrix_form():
    """`_log2fc_with_var` stopped materialising `mean_cpm()`/`var_cpm()` because on a
    full-corpus artifact (18,294 x 38,584) each was 5.6 GB for two rows. The saving is only
    allowed if the numbers do not move -- bit-identical, since these sums are pooled against
    results produced by the matrix form."""
    from sidechain.models.count_emitters import log2fc_from_cpm
    from sidechain.submit.build import LN2_SQ, _log2fc_with_var

    rng = np.random.default_rng(7)
    n_labels, n_genes = 6, 40
    pb = PseudobulkSums(
        labels=[f"T{i}" for i in range(n_labels - 1)] + ["ctrl"],
        genes=np.array([f"g{j}" for j in range(n_genes)], dtype=object),
        count_sum=rng.random((n_labels, n_genes)) * 100,
        cpm_sum=rng.random((n_labels, n_genes)) * 1e6,
        cpm_sq_sum=rng.random((n_labels, n_genes)) * 1e12,
        n_cells=rng.integers(1, 500, n_labels),
        libsize_sum=rng.random(n_labels) * 1e5,
    )

    def matrix_form(label, control, pseudocount=1.0):
        i, c = pb.labels.index(label), pb.labels.index(control)
        m = pb.mean_cpm(); v = pb.var_cpm(); n = np.maximum(pb.n_cells, 1)
        fc = log2fc_from_cpm(m[i], m[c], pseudocount)
        var = (v[i] / n[i]) / (m[i] + pseudocount) ** 2 + (v[c] / n[c]) / (m[c] + pseudocount) ** 2
        return fc, var / LN2_SQ

    for label in pb.labels[:-1]:
        fc_new, var_new = _log2fc_with_var(pb, label, "ctrl")
        fc_old, var_old = matrix_form(label, "ctrl")
        assert np.array_equal(fc_new, fc_old), label
        assert np.array_equal(var_new, var_old), label
