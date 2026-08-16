"""Regression tests for two mistakes made during the first external ingest.

Each test reproduces the original failure with the data that caused it, so a
future rewrite that reintroduces the bug fails here rather than in a findings
document six weeks later.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from sidechain.ingest.checks import (
    RAW_COUNTS,
    TRANSFORMED,
    control_mask,
    counts_state,
    require_raw_counts,
    to_cp10k,
)

# The real labels from WesselsSatija2023 that a substring test misread.
WESSELS_LABELS = np.array(
    ["INTS1", "DOT1L_INTS1", "INTS1_RING1", "IKZF1_INTS1", "GATA2_INTS1", "control"] * 3
)

# Raw counts with DIFFERENT per-cell totals (10 and 12), as real UMI data has.
# Equal totals are the signature of already-normalized data, so a fixture with
# equal row sums cannot stand in for counts.
COUNTS = np.array([[1.0, 9.0], [4.0, 8.0]])


# ------------------------------------------------- control labels: exact --


def test_substring_control_match_does_not_happen():
    """The original bug: "NT" matched "INTS1", so five perturbations were
    reported as controls."""
    mask = control_mask(WESSELS_LABELS, "control")
    assert mask.sum() == 3
    assert set(WESSELS_LABELS[mask]) == {"control"}
    # every INTS1-bearing perturbation must be OUTSIDE the control pool
    assert not any("INTS1" in v for v in WESSELS_LABELS[mask])


def test_nt_is_not_a_control_label_here():
    """`NT` is a real control name in some corpora, which is exactly why it
    must be matched exactly rather than searched for."""
    with pytest.raises(ValueError, match="no cells match"):
        control_mask(WESSELS_LABELS, "NT")


def test_absent_control_error_names_substring_near_misses():
    """The error should point at the trap it just avoided."""
    with pytest.raises(ValueError) as exc:
        control_mask(np.array(["INTS1", "DOT1L_INTS1"]), "INTS")
    assert "substring" in str(exc.value)
    assert "INTS1" in str(exc.value)


def test_control_label_is_case_sensitive():
    with pytest.raises(ValueError, match="no cells match"):
        control_mask(np.array(["control", "TP53"]), "Control")


def test_control_mask_works_on_a_pandas_series():
    pd = pytest.importorskip("pandas")
    mask = control_mask(pd.Series(["control", "TP53", "control"]), "control")
    assert mask.tolist() == [True, False, True]


# ------------------------------------------- normalization state: detect --


def test_raw_counts_are_detected():
    X = np.array([[0.0, 3.0, 12.0], [5.0, 0.0, 1.0]])
    assert counts_state(X) == RAW_COUNTS


def test_cp10k_is_detected_as_transformed():
    X = np.array([[0.0, 3.0, 12.0], [5.0, 0.0, 1.0]])
    assert counts_state(to_cp10k(X)) == TRANSFORMED


def test_log1p_is_detected_as_transformed():
    X = np.log1p(np.array([[0.0, 3.0, 12.0], [5.0, 0.0, 1.0]]))
    assert counts_state(X) == TRANSFORMED


def test_negative_values_are_transformed():
    """Scaled/centred data is never counts, even if it rounds cleanly."""
    assert counts_state(np.array([[-1.0, 2.0], [3.0, -4.0]])) == TRANSFORMED


def test_sparse_input_is_handled():
    X = sp.csr_matrix(np.array([[0.0, 3.0], [5.0, 0.0]]))
    assert counts_state(X) == RAW_COUNTS


def test_double_transform_is_refused():
    """The original bug: transforming data that was already transformed.

    Applied to real values it overflowed float32 to inf and produced nan
    summary statistics -- no exception, just wrong numbers.

    Note the fixture has UNEQUAL row totals (10 and 12). An earlier version of
    this test used [[1,9],[4,6]], whose rows both sum to 10 -- which the
    detector now correctly calls already-normalized, because constant library
    size is what normalization produces.
    """
    once = to_cp10k(COUNTS)
    with pytest.raises(ValueError, match="already-transformed"):
        to_cp10k(once)


def test_require_raw_counts_names_the_caller():
    with pytest.raises(ValueError, match="harmonize:"):
        require_raw_counts(np.array([[0.5, 1.5]]), where="harmonize")


def test_require_raw_counts_passes_on_counts():
    require_raw_counts(np.array([[0, 3], [5, 1]]), where="harmonize")  # no raise


def test_zero_count_cells_are_refused_rather_than_nan():
    """Dividing by a zero row total yields nan, which is the silent-wrong
    failure mode this module exists to prevent."""
    with pytest.raises(ValueError, match="zero total counts"):
        to_cp10k(np.array([[0.0, 0.0], [1.0, 2.0]]))


def test_cp10k_rows_sum_to_10k():
    out = to_cp10k(COUNTS)
    assert np.allclose(out.sum(axis=1), 1e4)
