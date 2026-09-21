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


# ------------------------------- a control arm can be more than one label --


def test_control_mask_accepts_several_labels():
    """Feng 2026 defines its controls as "either no guide or a non-targeting
    gRNA", so its control arm is two labels. Until this accepted a list the
    corpus could not be declared correctly at all."""
    labels = ["TP53", "NonTarget", "unassigned", "BRCA1", "unassigned"]
    mask = control_mask(labels, ["NonTarget", "unassigned"])
    assert mask.tolist() == [False, True, True, False, True]


def test_a_single_string_still_works():
    """Every existing block declares one label; none of them may change."""
    mask = control_mask(["a", "control", "b"], "control")
    assert mask.tolist() == [False, True, False]


def test_taking_only_the_control_looking_label_undercounts_the_arm():
    """The 2026-08-23 mistake, in miniature. `NonTarget` alone is 1 of 4 cells
    here; Feng's real ratio was 48 against 499,998, and a delta anchored on the
    48 would have been noise presented as signal."""
    labels = ["NonTarget"] + ["unassigned"] * 3 + ["TP53"] * 6
    narrow = control_mask(labels, "NonTarget")
    full = control_mask(labels, ["NonTarget", "unassigned"])
    assert narrow.sum() == 1
    assert full.sum() == 4


def test_a_declared_label_matching_nothing_raises_even_if_others_match():
    """Silently partial is the failure mode of the whole module: three of four
    labels matching looks exactly like a control arm that is simply smaller."""
    labels = ["TP53", "NonTarget", "BRCA1"]
    with pytest.raises(ValueError, match="unassigned"):
        control_mask(labels, ["NonTarget", "unassigned"])


def test_an_empty_control_label_list_raises():
    with pytest.raises(ValueError, match="empty"):
        control_mask(["a", "b"], [])


# ── the on-target positive control (T18 check 4) ────────────────────────────────


def _screen(n_targets=120, n_genes=200, knockdown_log2=-2.0, seed=0):
    """A CRISPRi screen in pseudobulk form: every target's own gene is silenced, the rest
    of the axis is noise. Targets are the first `n_targets` genes, so each is self-measurable."""
    import numpy as np

    from sidechain.data.stream_pseudobulk import PseudobulkSums

    rng = np.random.default_rng(seed)
    genes = np.array([f"g{i}" for i in range(n_genes)])
    labels = [f"g{i}" for i in range(n_targets)] + ["non-targeting"]
    base = rng.uniform(20, 200, n_genes)
    mean = np.tile(base, (len(labels), 1)) * rng.lognormal(0, 0.05, (len(labels), n_genes))
    for i in range(n_targets):
        mean[i, i] = base[i] * 2.0**knockdown_log2
    n = np.full(len(labels), 200, dtype=np.int64)
    return PseudobulkSums(labels=labels, genes=genes, count_sum=mean * n[:, None],
                          cpm_sum=mean * n[:, None], cpm_sq_sum=(mean**2) * n[:, None] * 2,
                          n_cells=n, libsize_sum=n * 20_000.0, sources=["s"])


def test_a_real_screen_passes_the_on_target_control():
    from sidechain.ingest.checks import require_on_target_knockdown

    got = require_on_target_knockdown(_screen(), "non-targeting")
    assert got["status"] == "ok"
    assert got["n_self_measurable"] == 120
    assert got["median_self_log2fc"] < -1.5 and got["frac_positive"] == 0.0


def test_a_shuffled_perturbation_column_is_caught():
    """The failure this exists for: a well-formed aggregate whose labels name the wrong rows.
    Every other check in this module passes on it."""
    import numpy as np
    import pytest

    from sidechain.ingest.checks import require_on_target_knockdown

    pb = _screen()
    rng = np.random.default_rng(1)
    perts = pb.labels[:-1]
    pb.labels = list(rng.permutation(perts)) + ["non-targeting"]
    with pytest.raises(ValueError, match="on-target knockdown check FAILED"):
        require_on_target_knockdown(pb, "non-targeting")


def test_a_gene_axis_off_by_one_is_caught():
    """A misaligned axis keeps every number and moves every meaning."""
    import numpy as np
    import pytest

    from sidechain.ingest.checks import require_on_target_knockdown

    pb = _screen()
    pb.genes = np.roll(pb.genes, 1)
    with pytest.raises(ValueError, match="on-target knockdown check FAILED"):
        require_on_target_knockdown(pb, "non-targeting")


def test_an_activation_screen_is_caught_by_the_sign():
    """CRISPRi silences. A positive median is not this screen, whatever else is right."""
    import pytest

    from sidechain.ingest.checks import require_on_target_knockdown

    with pytest.raises(ValueError, match=r"is \+"):
        require_on_target_knockdown(_screen(knockdown_log2=+2.0), "non-targeting")


def test_too_few_measurable_arms_is_not_applicable_rather_than_a_pass():
    """A pre-filtered axis need not carry the perturbed genes. Three arms cannot support a
    median, and a check that cannot see must not report agreement."""
    from sidechain.ingest.checks import NOT_APPLICABLE, require_on_target_knockdown

    got = require_on_target_knockdown(_screen(n_targets=3), "non-targeting")
    assert got["status"] == NOT_APPLICABLE and got["n_self_measurable"] == 3
    # and it does not raise even though three arms would have passed on their median
    assert got["median_self_log2fc"] < 0


def test_labels_whose_gene_is_off_the_axis_are_skipped_not_counted():
    """K562 genome-wide perturbs 272 fold targets and carries the on-target row for 229."""
    from sidechain.ingest.checks import require_on_target_knockdown

    pb = _screen(n_targets=120, n_genes=200)
    pb.labels = pb.labels[:-1] + ["GENE_NOT_ON_AXIS", "non-targeting"]
    pb.count_sum = pb.count_sum[[*range(120), 0, 120]]
    pb.cpm_sum = pb.cpm_sum[[*range(120), 0, 120]]
    pb.cpm_sq_sum = pb.cpm_sq_sum[[*range(120), 0, 120]]
    pb.n_cells = pb.n_cells[[*range(120), 0, 120]]
    pb.libsize_sum = pb.libsize_sum[[*range(120), 0, 120]]
    assert require_on_target_knockdown(pb, "non-targeting")["n_self_measurable"] == 120


def test_the_control_arm_may_be_several_labels():
    """Same rule as `control_mask`: Feng 2026's arm is two labels."""
    import numpy as np

    from sidechain.ingest.checks import require_on_target_knockdown

    pb = _screen()
    pb.labels = pb.labels[:-1] + ["NonTarget"]
    pb.labels = pb.labels + ["unassigned"]
    for a in ("count_sum", "cpm_sum", "cpm_sq_sum"):
        setattr(pb, a, np.vstack([getattr(pb, a), getattr(pb, a)[-1]]))
    pb.n_cells = np.append(pb.n_cells, pb.n_cells[-1])
    pb.libsize_sum = np.append(pb.libsize_sum, pb.libsize_sum[-1])
    assert require_on_target_knockdown(pb, ["NonTarget", "unassigned"])["status"] == "ok"
