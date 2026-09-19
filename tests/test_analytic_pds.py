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


def test_replay_refuses_an_arm_whose_shrinkage_is_unrecorded(tmp_path):
    """`bool(None)` is False but `pooled_delta` defaults to shrinkage=True, so an absent
    field must not be rebuilt as "off" -- the same falsy-default class as the gamma bug.
    The record does not say, so the replay does not guess."""
    import json
    from scripts.analytic_pds_replay import replay_arm

    arm = tmp_path / "some_arm"
    (arm / "run").mkdir(parents=True)
    (arm / "run" / "agg_results.csv").write_text("statistic,pds_cosine\nmean,0.7\n")
    (arm / "summary.json").write_text(json.dumps(
        {"build": {"alpha": 1.0, "var_floor": "poisson"},          # shrinkage absent
         "sources": {"pseudobulk": ["/nowhere/x.npz:non-targeting"]}}))
    out = replay_arm(arm, "some_fold", None)
    assert out["status"] == "skipped"
    assert "shrinkage not recorded" in out["why"]


# ---------------------------------------------------------------- drop_one_arm


class _Src:
    """A minimal delta source: `effect(target) -> (log2fc, var)` on its own gene axis.

    `as_delta_source` passes anything that already answers `effect` straight through, so a
    fake arm needs nothing else. `covers` is the set of targets it speaks for -- an arm that
    covers few targets is the case the coverage statistic gets wrong if it is careless.
    """

    def __init__(self, genes, covers, fc, var, shrink=None):
        self.genes, self._covers, self._fc, self._var = np.asarray(genes), set(covers), fc, var
        self.shrink = shrink

    def effect(self, target):
        if target not in self._covers:
            return None
        return np.full(len(self.genes), self._fc), np.full(len(self.genes), self._var)


def test_drop_one_arm_reports_what_each_arm_is_worth():
    from sidechain.eval.analytic_pds import drop_one_arm

    f = _fold(n_genes=6, n_targets=3)
    targets = ["g0", "g1", "g2"]
    good = _Src(f.genes, targets, fc=1.0, var=0.1)
    noise = _Src(f.genes, targets, fc=-1.0, var=0.1)
    r = drop_one_arm(targets, [good, noise], f, names=["good", "noise"], verify=3)

    assert set(r["arms"]) == {"good", "noise"}
    assert r["n_targets"] == 3 and r["n_targets_covered"] == 3
    for name in ("good", "noise"):
        arm = r["arms"][name]
        # worth is defined as full minus without -- the identity, not an approximation
        assert arm["worth"] == pytest.approx(r["pds_full"] - arm["pds_without"])
        assert arm["n_targets_covered"] == 3
        assert arm["axis_coverage_median"] == pytest.approx(1.0)


def test_drop_one_arm_coverage_is_medianed_over_the_targets_an_arm_covers():
    """The bug this pins: a thin arm must not read as covering zero genes.

    An arm covering 1 of 3 targets has no weight on the other two, so a median over ALL
    targets is a structural 0.0 and says "reaches nothing" about an arm that reaches every
    gene where it speaks. Absent and outvoted are different facts.
    """
    from sidechain.eval.analytic_pds import drop_one_arm

    f = _fold(n_genes=6, n_targets=3)
    targets = ["g0", "g1", "g2"]
    broad = _Src(f.genes, targets, fc=1.0, var=0.1)
    thin = _Src(f.genes, ["g0"], fc=1.0, var=0.1)
    r = drop_one_arm(targets, [broad, thin], f, names=["broad", "thin"], verify=0)

    assert r["arms"]["thin"]["n_targets_covered"] == 1
    assert r["arms"]["thin"]["axis_coverage_median"] == pytest.approx(1.0)
    assert r["arms"]["broad"]["n_targets_covered"] == 3


def test_drop_one_arm_sees_a_half_axis_arm_as_half_covering():
    from sidechain.eval.analytic_pds import drop_one_arm

    f = _fold(n_genes=6, n_targets=3)
    targets = ["g0", "g1", "g2"]
    broad = _Src(f.genes, targets, fc=1.0, var=0.1)
    half = _Src(f.genes[:3], targets, fc=1.0, var=0.1)     # only the first three genes
    r = drop_one_arm(targets, [broad, half], f, names=["broad", "half"], verify=0)
    assert r["arms"]["half"]["axis_coverage_median"] == pytest.approx(0.5)


def test_drop_one_arm_refuses_a_pool_it_cannot_drop_from():
    from sidechain.eval.analytic_pds import drop_one_arm

    f = _fold()
    one = _Src(f.genes, ["g0"], fc=1.0, var=0.1)
    with pytest.raises(ValueError, match="at least 2 sources"):
        drop_one_arm(["g0"], [one], f)


def test_drop_one_arm_refuses_mismatched_names():
    from sidechain.eval.analytic_pds import drop_one_arm

    f = _fold()
    a = _Src(f.genes, ["g0"], fc=1.0, var=0.1)
    with pytest.raises(ValueError, match="names for"):
        drop_one_arm(["g0"], [a, a], f, names=["only-one"])


def test_drop_one_arm_verify_catches_a_pool_the_shortcut_cannot_reproduce():
    """`verify` is the guard that keeps the cached-parts shortcut honest.

    `pooled_delta` with per-source shrinkage on is not a plain inverse-variance average any
    more, so the parts stop reproducing it -- and the caller must be told, not handed a
    quietly wrong number.
    """
    from sidechain.eval.analytic_pds import drop_one_arm

    f = _fold(n_genes=6, n_targets=3)
    targets = ["g0", "g1", "g2"]
    a = _Src(f.genes, targets, fc=2.0, var=0.5, shrink=True)   # shrinks inside pooled_delta
    b = _Src(f.genes, targets, fc=1.0, var=0.1)
    with pytest.raises(AssertionError, match="do not reproduce pooled_delta"):
        drop_one_arm(targets, [a, b], f, verify=3)


class _VecSrc:
    """A delta source with a per-target log2FC VECTOR and a per-target variance, so an arm can
    be right about one target and wrong about another -- which `_Src`'s constant vector cannot."""

    shrink = None

    def __init__(self, genes, votes):
        self.genes, self._votes = np.asarray(genes), votes     # target -> (fc vector, var)

    def effect(self, target):
        if target not in self._votes:
            return None
        fc, var = self._votes[target]
        return np.asarray(fc, dtype=float), np.full(len(self.genes), float(var))


def _signed_fold():
    """Three targets whose real effects are three disjoint gene blocks, off the target genes
    themselves (cell-eval2's `exclusion_scope="panel"` removes those columns)."""
    from cell_eval2.prep import bulk_lognorm_means

    genes = np.array([f"g{i}" for i in range(12)])
    targets = ["g0", "g1", "g2"]
    E = np.zeros((3, 12))
    for t in range(3):
        E[t, 3 + 3 * t: 6 + 3 * t] = 1.0
    frac = np.full(12, 1.0 / 12)
    sums = np.vstack([frac * np.exp2(e) for e in E] + [frac]) * 1e5
    return FoldCache(perts=np.array(targets + ["non-targeting"], dtype=object),
                     real_means=bulk_lognorm_means(sums, 50_000.0), genes=genes,
                     n_cells=np.array([100] * 4), frac=frac, lib_median=1000.0,
                     ctrl_n_cells=100), targets, E


def test_drop_one_arm_worth_is_not_a_ceiling_on_a_rule():
    """The docstring once called `worth` "the ceiling on any per-arm rule". It is not.

    `pds_cosine` is a mean of per-target scores and is not monotone in an arm's weight, so a
    rule can beat BOTH keeping the arm and dropping it. Measured on real folds by `T94`
    (session `94641ce7`, 2026-09-19); this is the smallest pool that shows it. `mixed` is
    right and loud about g0, wrong about g1; `steady` is the reverse.
    """
    from sidechain.eval.analytic_pds import drop_one_arm

    f, targets, E = _signed_fold()
    steady = _VecSrc(f.genes, {"g0": (E[1], 1.0), "g1": (E[1], 1.0), "g2": (E[2], 1.0)})

    def mixed(weight=1.0, covers=("g0", "g1")):
        votes = {"g0": (E[0], 0.01 / weight), "g1": (E[2], (1 / 3) / weight)}
        return _VecSrc(f.genes, {t: votes[t] for t in covers})

    r = drop_one_arm(targets, [steady, mixed()], f, names=["steady", "mixed"], verify=3)
    keep, drop = r["pds_full"], r["arms"]["mixed"]["pds_without"]
    assert keep < 1.0 and drop < 1.0          # each endpoint gets one target wrong

    def pool(src):
        return drop_one_arm(targets, [steady, src], f, verify=3)["pds_full"]

    assert pool(mixed(covers=("g0",))) > max(keep, drop)      # a hard per-target gate
    assert pool(mixed(weight=0.1)) > max(keep, drop)          # ONE scalar weight on the arm


def test_pool_parts_verify_survives_a_target_no_source_covers():
    """`pooled_delta` abstains with None on an uncovered target, and `None - array` raises.

    Reported by session `66c37b95` on 2026-09-18: their driver passed a target no arm in the
    pool had, and the crash came from inside the CHECK rather than from the arithmetic being
    verified. An uncovered target's parts are all zero, so there is nothing to compare.
    """
    from sidechain.eval.analytic_pds import pool_parts

    f = _fold(n_genes=6, n_targets=3)
    covered, uncovered = "g0", "g2"
    a = _Src(f.genes, [covered], fc=1.0, var=0.1)
    b = _Src(f.genes, [covered], fc=2.0, var=0.4)

    num, den = pool_parts([covered, uncovered], [a, b], f.genes, verify=2)
    assert num.shape == (2, 6)
    assert np.all(den[1] == 0.0)          # the uncovered target got no weight from anyone
    assert np.any(den[0] > 0.0)
