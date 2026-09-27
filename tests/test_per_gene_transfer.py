"""Contract tests for `sidechain.eval.per_gene_transfer` (T102).

What each one pins, and why it is worth a test:

* `lfc_ce2` is cell-eval2's formula, not the pipeline's pseudocount-1 one -- the two differ
  by up to a third of a log2 unit on a 5-CPM gene, and the whole point of the module is to
  read errors in the metric's units.
* `pred_lfc` equals what the emitter's per-cell mean would produce: a target no source
  covered is the bare profile and reads exactly 0 against a matching control, and the
  target's own gene is pinned when covered.
* `per_gene_stats` drops perturbations whose gate is below `min_gate_size` -- nmae scores
  nothing on those, so their cells must not leak into a per-gene number -- and its slope is
  the least-squares amplitude through the origin.
* `gate_wald` is per-perturbation BH over the universe with the own gene off.
* `correlate`'s shuffle null is calibrated: a feature unrelated to the statistic gets a
  uniform-ish `p_perm`, and partialling removes a confound-driven correlation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sidechain.eval.per_gene_transfer import (
    FoldTruth,
    correlate,
    gate_from_de_real,
    gate_wald,
    lfc_ce2,
    per_gene_stats,
    pred_lfc,
)


def test_lfc_ce2_is_the_epsilon_formula_not_a_pseudocount():
    assert lfc_ce2(np.array([10.0]), np.array([5.0]))[0] == pytest.approx(1.0)
    # pseudocount 1 would give log2(11/6) = 0.874; epsilon 1e-9 gives ~1.0
    assert lfc_ce2(np.array([0.0]), np.array([5.0]))[0] < -30      # zero mean -> a large negative


def test_pred_lfc_uncovered_target_is_zero_and_covered_target_is_pinned():
    genes = np.array(["A", "B", "C"])
    frac = np.array([0.5, 0.3, 0.2])
    ctrl = 1e6 * frac                                   # a control profile that matches the emitter's
    deltas = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    out = pred_lfc(deltas, ["B", "C"], genes, frac, ctrl, alpha=2.0, covered=[True, False])
    # row 1 is uncovered: bare profile, no pin, exactly zero log2FC against the matching control
    assert np.allclose(out[1], 0.0, atol=1e-9)
    # row 0: B is the target, pinned at -2.32 before renormalisation; A carries alpha*delta = 2
    f = frac * np.exp2(np.array([2.0, -2.32, 0.0]))
    f = f / f.sum()
    assert np.allclose(out[0], np.log2(1e6 * f / ctrl), atol=1e-9)


def test_per_gene_stats_slope_and_gate_size_rule():
    genes = np.array(["g0", "g1", "g2"])
    # 12 perturbations; gene g0 gated on all of them with pred = 0.5 * truth exactly
    P = 12
    truth = np.zeros((P, 3)); truth[:, 0] = np.linspace(-2, 2, P) + 0.1
    pred = 0.5 * truth
    gate = np.zeros((P, 3), dtype=bool); gate[:, 0] = True
    # min_gate_size 1 so every perturbation counts
    s = per_gene_stats(pred, truth, gate, genes, min_targets=8, min_gate_size=1)
    assert list(s["gene"]) == ["g0"]
    assert s["slope"].iloc[0] == pytest.approx(0.5)
    assert s["ratio"].iloc[0] == pytest.approx(0.5)
    assert s["nmae_g"].iloc[0] == pytest.approx(0.5)
    assert s["sign_agree"].iloc[0] == pytest.approx(1.0)
    # with nmae's own min_gate_size of 10, a one-gene gate scores nothing: no rows
    assert per_gene_stats(pred, truth, gate, genes, min_targets=8).empty


def _truth(P=6, G=5, seed=0):
    rng = np.random.default_rng(seed)
    ctrl = np.array([50.0, 20.0, 10.0, 6.0, 1.0])            # last gene below the 5-CPM floor
    mean = np.tile(ctrl, (P, 1)) * np.exp2(rng.normal(0, 0.5, (P, G)))
    mean[0, 1] = ctrl[1] * 8                                  # one strong, certain effect
    return FoldTruth(targets=[f"t{i}" for i in range(P)], genes=np.array(["a", "b", "c", "d", "e"]),
                     mean_cpm=mean, var_cpm=np.full((P, G), 4.0), n_cells=np.full(P, 400),
                     ctrl_mean_cpm=ctrl, ctrl_var_cpm=np.full(G, 4.0), ctrl_n_cells=10000,
                     control="non-targeting")


def test_gate_wald_respects_universe_and_own_gene():
    tr = _truth()
    tr = FoldTruth(**{**tr.__dict__, "targets": ["b"] + tr.targets[1:]})   # first target IS gene b
    gate, info = gate_wald(tr)
    assert info["kind"] == "wald_z_approx"
    assert not gate[:, 4].any()                    # below the CPM floor: never gated
    assert not gate[0, 1]                          # own gene off, however strong its effect
    assert tr.universe.sum() == 4


def test_gate_wald_uses_the_null_variance_so_a_zero_arm_is_not_certainty():
    """A target arm whose every cell reads 0 has var 0; under a Welch form its standard error
    came from the controls alone and the cell was always significant. The null form asks
    whether a mean of 0 is far from the control mean GIVEN the controls' own spread."""
    ctrl = np.array([6.0, 6.0])
    P = 2
    mean = np.array([[0.0, 6.0], [0.0, 6.0]])          # gene 0 is all-zero in both targets
    var = np.zeros((P, 2))
    tr = FoldTruth(targets=["t0", "t1"], genes=np.array(["a", "b"]), mean_cpm=mean, var_cpm=var,
                   n_cells=np.array([20, 20]), ctrl_mean_cpm=ctrl,
                   ctrl_var_cpm=np.array([2000.0, 2000.0]),   # controls are very dispersed (sd ~45 on a mean of 6)
                   ctrl_n_cells=10000, control="non-targeting")
    gate, _ = gate_wald(tr)
    # z = 6 / sqrt(2000 * (1/20 + 1/10000)) = 0.6: not significant under the null variance
    assert not gate[:, 0].any()
    tr2 = FoldTruth(**{**tr.__dict__, "ctrl_var_cpm": np.array([1.0, 1.0])})   # tight controls
    gate2, _ = gate_wald(tr2)
    assert gate2[:, 0].all()                          # z = 6 / sqrt(1 * 0.0501) ~ 27: significant


def test_per_gene_stats_l1_scale_is_the_weighted_median_and_truth_se_is_filled():
    genes = np.array(["g0", "g1"])
    P = 12
    truth = np.zeros((P, 2)); truth[:, 0] = np.linspace(1, 2, P)
    pred = 0.5 * truth                               # exact rescale 2.0 on every cell
    gate = np.zeros((P, 2), dtype=bool); gate[:, 0] = True
    tr = FoldTruth(targets=[f"t{i}" for i in range(P)], genes=genes,
                   mean_cpm=np.full((P, 2), 10.0), var_cpm=np.full((P, 2), 4.0),
                   n_cells=np.full(P, 100), ctrl_mean_cpm=np.array([10.0, 10.0]),
                   ctrl_var_cpm=np.array([4.0, 4.0]), ctrl_n_cells=10000, control="non-targeting")
    s = per_gene_stats(pred, truth, gate, genes, min_targets=8, min_gate_size=1, truth=tr)
    assert s["l1_scale"].iloc[0] == pytest.approx(2.0)
    assert np.isfinite(s["truth_se"].iloc[0]) and s["truth_se"].iloc[0] > 0


def test_gate_from_de_real_reads_the_table_and_drops_own_gene(tmp_path):
    df = pd.DataFrame({"target": ["t0", "t0", "t1", "b"], "feature": ["a", "b", "a", "b"],
                       "log2_fold_change": [1.0, -1.0, np.inf, 2.0],
                       "p_adj": [0.01, 0.2, 0.01, 0.001]})
    pq = tmp_path / "de_real.parquet"; df.to_parquet(pq)
    gate, info = gate_from_de_real(pq, ["t0", "t1", "b"], np.array(["a", "b"]))
    assert gate[0, 0] and not gate[0, 1]           # p_adj 0.2 is out
    assert not gate[1, 0]                          # infinite real lfc leaves the gate
    assert not gate[2, 1]                          # own gene off
    assert info["kind"] == "wilcoxon_de_real"


def test_correlate_null_is_calibrated_and_partial_removes_a_confound():
    rng = np.random.default_rng(1)
    n = 400
    expr = rng.normal(size=n)
    stat = expr + rng.normal(scale=0.5, size=n)            # the error tracks expression
    feature = expr + rng.normal(scale=0.5, size=n)         # so does the feature
    raw = correlate(stat, feature, n_perm=300, seed=0)
    assert raw["rho"] > 0.4 and raw["p_perm"] < 0.01
    part = correlate(stat, feature, covariates=expr[None, :], n_perm=300, seed=0)
    assert abs(part["rho_partial"]) < 0.15                  # the confound carried it
    noise = correlate(stat, rng.normal(size=n), n_perm=300, seed=0)
    assert noise["p_perm"] > 0.05


def test_nmae_closed_form_matches_the_definition_and_drops_small_gates():
    from sidechain.eval.per_gene_transfer import nmae_closed_form
    P, G = 3, 12
    truth = np.ones((P, G)); pred = np.full((P, G), 0.5)
    gate = np.zeros((P, G), dtype=bool)
    gate[0, :] = True                     # 12 genes: scored, nmae 0.5
    gate[1, :5] = True                    # 5 genes: below min_gate_size, scores nothing
    m, per = nmae_closed_form(pred, truth, gate)
    assert per.shape == (1,) and m == pytest.approx(0.5)


def test_binned_amplitude_redistributes_and_maps_back_onto_an_axis():
    from sidechain.eval.per_gene_transfer import amplitude_on_axis, fit_binned_amplitude
    rng = np.random.default_rng(0)
    feature = rng.uniform(size=200)
    slope = np.where(feature > 0.5, 0.4, 0.8) + rng.normal(scale=0.01, size=200)   # high feature -> under-shoot
    fit = fit_binned_amplitude(feature, slope, n_bins=2)
    assert fit["factors"][1] > fit["factors"][0]                # the under-shooting bin is boosted
    assert np.exp(np.mean(np.log(fit["factors"]))) == pytest.approx(1.0)   # geometric mean 1
    a = amplitude_on_axis(np.array([0.1, 0.9, np.nan]), fit)
    assert a[0] == pytest.approx(fit["factors"][0]) and a[1] == pytest.approx(fit["factors"][1])
    assert a[2] == 1.0                                          # no feature: untouched


def test_binned_amplitude_ties_share_a_bin_and_a_floored_bin_refuses():
    from sidechain.eval.per_gene_transfer import amplitude_on_axis, fit_binned_amplitude
    rng = np.random.default_rng(1)
    # 40 % zeros (no sites), the rest 1..20: a rank split would put zeros in two bins
    feature = np.concatenate([np.zeros(80), rng.integers(1, 21, size=120)]).astype(float)
    stat = np.where(feature == 0, 0.9, 0.5) + rng.normal(scale=0.01, size=200)
    fit = fit_binned_amplitude(feature, stat, n_bins=5)
    assert fit["edges"][0] == 0.0 and fit["bin_n"][0] == 80          # every zero in bin 0
    a = amplitude_on_axis(np.array([0.0, 0.0, 1.0]), fit)
    assert a[0] == a[1] == pytest.approx(fit["factors"][0])         # ties get one factor
    assert a[2] != a[0]
    with pytest.raises(ValueError, match="floor"):
        fit_binned_amplitude(feature, np.full(200, 0.001), n_bins=2, floor=0.01)
