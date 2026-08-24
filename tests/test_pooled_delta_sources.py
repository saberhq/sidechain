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
