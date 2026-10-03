"""Contract tests for the shrinkage rule's three knobs (T84): hardness, stage and rule.

What is pinned: the defaults are the historical rule, bit for bit; `k` moves the garrote's
threshold and nothing else; the pooled stage shrinks the pooled vector with the pooled
variance and is the per-source rule when one source measures the gene; the settings that
name no model are refused, in the function and at both entry points; the adaptive rule
recovers a known mixture and never zeroes a gene; and `eval.loco` and `submit.build` hand
`pooled_delta` the same keyword arguments for the same flags.
"""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml

from sidechain.data.lfc_table import LfcTable
from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.eval import loco
from sidechain.models import adaptive_shrink as ash
from sidechain.submit import build
from sidechain.submit.build import pooled_delta, shrink

AXIS = np.array(["A", "B", "C", "D"])
KNOBS = ("shrink_k", "shrink_stage", "shrink_rule")


def _lfc(lfc, var, genes=AXIS, label="T"):
    return LfcTable(labels=[label], genes=np.array(genes), lfc=np.asarray([lfc], float),
                    var=np.asarray([var], float), source="test-lfc")


# ----------------------------------------------------------------------- the garrote --


def test_k_one_is_the_historical_rule_bit_for_bit():
    rng = np.random.default_rng(0)
    fc = rng.normal(0, 1, 5000)
    fc[:50] = 0.0
    var = rng.gamma(1.0, 0.3, 5000)
    with np.errstate(divide="ignore", invalid="ignore"):
        old = fc * np.clip(np.where(fc != 0, 1.0 - var / np.maximum(fc**2, 1e-12), 0.0), 0.0, 1.0)
    assert np.array_equal(shrink(fc, var), old)
    assert np.array_equal(shrink(fc, var, 1.0), old)


def test_k_sets_the_threshold_in_standard_errors():
    se = 0.25
    z = np.array([0.5, 1.0, 2.0, 2.82, 2.84, 4.0, 10.0])
    fc, var = z * se, np.full(z.size, se**2)
    assert np.allclose(shrink(fc, var, 1.0) / fc, [0, 0, 0.75, 1 - 1 / 2.82**2, 1 - 1 / 2.84**2, 0.9375, 0.99])
    k8 = shrink(fc, var, 8.0) / fc
    assert np.array_equal(k8[:4], np.zeros(4))                     # below sqrt(8) = 2.83: zero
    assert np.allclose(k8[4:], [1 - 8 / 2.84**2, 0.5, 0.92])
    # a harder rule never keeps more of any gene, and the sign never flips
    for lo, hi in ((1.0, 4.0), (4.0, 8.0), (8.0, 16.0)):
        a, b = shrink(fc, var, lo), shrink(fc, var, hi)
        assert np.all(np.abs(b) <= np.abs(a)) and np.all(a * fc >= 0) and np.all(b * fc >= 0)
    assert np.array_equal(shrink(-fc, var, 8.0), -shrink(fc, var, 8.0))


# ---------------------------------------------------------------------- pooled_delta --


def test_the_defaults_are_the_historical_call_bit_for_bit():
    a = _lfc([0.5, -2.0, 0.1, 3.0], [0.3, 0.1, 0.2, 4.0])
    b = _lfc([0.7, -1.0, 0.4, -0.2], [0.1, 0.4, 0.2, 0.5])
    for on in (True, False):
        plain = pooled_delta("T", [a, b], AXIS, shrinkage=on)
        named = pooled_delta("T", [a, b], AXIS, shrinkage=on, shrink_k=1.0,
                             shrink_stage="source", shrink_rule="garrote")
        assert np.array_equal(plain, named)
    # and the default is the per-source garrote at k 1, written out by hand
    w1, w2 = 1 / a.var[0], 1 / b.var[0]
    by_hand = (shrink(a.lfc[0], a.var[0]) * w1 + shrink(b.lfc[0], b.var[0]) * w2) / (w1 + w2)
    assert np.allclose(pooled_delta("T", [a, b], AXIS), by_hand, rtol=1e-12, atol=0)


def test_k_reaches_each_source_at_the_source_stage():
    a = _lfc([0.5, -2.0, 0.1, 3.0], [0.01, 0.1, 0.2, 0.5])
    b = _lfc([0.7, -1.0, 0.4, 2.0], [0.02, 0.4, 0.2, 0.5])
    w1, w2 = 1 / a.var[0], 1 / b.var[0]
    by_hand = (shrink(a.lfc[0], a.var[0], 4.0) * w1 + shrink(b.lfc[0], b.var[0], 4.0) * w2) / (w1 + w2)
    got = pooled_delta("T", [a, b], AXIS, shrink_k=4.0)
    assert np.allclose(got, by_hand, rtol=1e-12, atol=0)
    assert not np.allclose(got, pooled_delta("T", [a, b], AXIS))


def test_the_pooled_stage_shrinks_the_pooled_vector_with_the_pooled_variance():
    a = _lfc([0.5, -2.0, 0.1, 3.0], [0.01, 0.1, 0.2, 0.5])
    b = _lfc([0.7, -1.0, 0.4, 2.0], [0.02, 0.4, 0.2, 0.5])
    w1, w2 = 1 / a.var[0], 1 / b.var[0]
    raw = (a.lfc[0] * w1 + b.lfc[0] * w2) / (w1 + w2)
    for k in (1.0, 8.0):
        got = pooled_delta("T", [a, b], AXIS, shrink_stage="pooled", shrink_k=k)
        assert np.allclose(got, shrink(raw, 1 / (w1 + w2), k), rtol=1e-12, atol=0)
    # with the rule off the stage is nothing: the plain pool, bit for bit
    off = pooled_delta("T", [a, b], AXIS, shrinkage=False)
    assert np.array_equal(off, raw)
    assert np.array_equal(pooled_delta("T", [a, b], AXIS, shrinkage=False, shrink_stage="pooled"), off)


def test_two_sources_under_one_standard_error_survive_only_when_pooled_first():
    """The stage's whole point: 0.9 standard errors twice is 1.27 standard errors once."""
    se = 0.2
    a = _lfc([0.9 * se] * 4, [se**2] * 4)
    b = _lfc([0.9 * se] * 4, [se**2] * 4)
    assert np.array_equal(pooled_delta("T", [a, b], AXIS), np.zeros(4))
    pooled = pooled_delta("T", [a, b], AXIS, shrink_stage="pooled")
    assert np.allclose(pooled, 0.9 * se * (1 - (se**2 / 2) / (0.9 * se) ** 2))
    assert np.all(pooled > 0)


def test_one_source_reads_the_same_at_either_stage():
    a = _lfc([0.5, -2.0, 0.1, 3.0], [0.01, 0.1, 0.2, 0.5])
    for k in (1.0, 4.0, 8.0):
        assert np.allclose(pooled_delta("T", [a], AXIS, shrink_k=k),
                           pooled_delta("T", [a], AXIS, shrink_k=k, shrink_stage="pooled"),
                           rtol=1e-12, atol=1e-15)
    # a gene only one of two sources measures is that source's gene at either stage
    b = _lfc([0.7, -1.0], [0.02, 0.4], genes=["A", "B"])
    per_source = pooled_delta("T", [a, b], AXIS, shrink_k=4.0)
    after = pooled_delta("T", [a, b], AXIS, shrink_k=4.0, shrink_stage="pooled")
    assert np.allclose(per_source[2:], after[2:], rtol=1e-12, atol=1e-15)


def test_the_pooled_stage_covers_abstention_and_an_uncovered_target():
    a = _lfc([0.5, -2.0, 0.1, 3.0], [0.01, np.inf, 0.2, 0.5])
    out = pooled_delta("T", [a], AXIS, shrink_stage="pooled", shrink_k=4.0)
    assert out[1] == 0.0 and np.all(np.isfinite(out))            # the abstained gene has no weight
    assert pooled_delta("NOPE", [a], AXIS, shrink_stage="pooled") is None


def test_settings_that_name_no_model_are_refused():
    a = _lfc([0.5, -2.0, 0.1, 3.0], [0.01, 0.1, 0.2, 0.5])
    for kw in ({"shrink_stage": "after"}, {"shrink_rule": "ashr"}, {"shrink_k": 0.0},
               {"shrink_k": -1.0}, {"shrink_k": float("nan")},
               {"shrink_rule": "adaptive"},                                  # per source: not wired
               {"shrink_rule": "adaptive", "shrink_stage": "pooled", "shrink_k": 8.0}):
        with pytest.raises(ValueError):
            pooled_delta("T", [a], AXIS, **kw)
    # refused even with the rule off: a bad setting is a bad setting
    with pytest.raises(ValueError):
        pooled_delta("T", [a], AXIS, shrinkage=False, shrink_k=0.0)


def test_the_pooled_stage_refuses_weights_that_are_not_inverse_variances():
    m = np.array([[100.0, 100.0, 100.0, 100.0], [200.0, 100.0, 50.0, 120.0]])
    n = np.full(2, 1000, dtype=np.int64)
    pb = PseudobulkSums(labels=["control", "T"], genes=AXIS.copy(), count_sum=m * n[:, None],
                        cpm_sum=m * n[:, None], cpm_sq_sum=(m**2 + 1.0) * n[:, None], n_cells=n,
                        libsize_sum=n.astype(float) * 1e4, sources=["t"])
    kw = dict(shrink_stage="pooled", var_floor="poisson")
    assert pooled_delta("T", [(pb, "control")], AXIS, **kw) is not None
    with pytest.raises(ValueError, match="gamma"):
        pooled_delta("T", [(pb, "control")], AXIS, gamma=0.5, ctrl_tgt_cpm=m[0], **kw)
    with pytest.raises(ValueError, match="coverage"):
        pooled_delta("T", [(pb, "control")], AXIS, coverage_tiers=((3.0, 0.1),), **kw)
    with pytest.raises(ValueError, match="similarity"):
        pooled_delta("T", [(pb, "control")], AXIS, similarity_beta=2.0, ctrl_tgt_cpm=m[0], **kw)
    with pytest.raises(ValueError, match="override"):
        pooled_delta("T", [(pb, "control", True)], AXIS, **kw)
    with pytest.raises(ValueError, match="override"):            # with the global rule off too:
        pooled_delta("T", [(pb, "control", True)], AXIS, shrinkage=False, **kw)   # never per source
    # each of those is still accepted at the source stage
    assert pooled_delta("T", [(pb, "control", True)], AXIS, var_floor="poisson", shrink_k=4.0) is not None
    pb.sidechain_name = "t"
    floored = build.apply_transfer_floors([(pb, "control")], {"t": 0.05})
    assert pooled_delta("T", floored, AXIS, var_floor="poisson", shrink_k=4.0) is not None
    with pytest.raises(ValueError, match="transfer floor"):
        pooled_delta("T", floored, AXIS, **kw)


# ------------------------------------------------------------------ the adaptive rule --


def _mixture(n=20000, seed=7):
    r = np.random.default_rng(seed)
    comp = r.choice(3, size=n, p=[0.7, 0.2, 0.1])
    beta = r.normal(size=n) * np.array([0.0, 0.3, 1.5])[comp]
    s = np.exp(r.uniform(np.log(0.05), np.log(1.0), size=n))
    return beta, beta + r.normal(size=n) * s, s


def test_adaptive_recovers_a_known_mixture():
    beta, x, s = _mixture()
    stats = {}
    got = ash.adaptive_shrink(x, s * s, stats=stats)
    oracle = ash.posterior_mean(x, s, np.array([1e-9, 0.3, 1.5]), np.array([0.7, 0.2, 0.1]))
    mse = lambda est: float(np.mean((est - beta) ** 2))
    assert mse(got) < 1.02 * mse(oracle) < 0.5 * mse(x)          # within 2 % of knowing the prior
    assert stats == {"adaptive_fits": 1, "adaptive_cycles": stats["adaptive_cycles"],
                     "adaptive_fits_at_cap": 0}
    assert 1 <= stats["adaptive_cycles"] < ash.MAX_CYCLES


def test_adaptive_shrinks_toward_zero_and_zeroes_nothing():
    _, x, s = _mixture(5000, seed=1)
    got = ash.adaptive_shrink(x, s * s)
    assert np.all(np.abs(got) <= np.abs(x) * (1 + 1e-12)) and np.all(got * x > 0)   # same sign, never larger
    assert np.count_nonzero(got) == x.size
    # a gene far outside its noise is nearly kept; one inside it is mostly removed
    assert np.median(got[np.abs(x) > 6 * s] / x[np.abs(x) > 6 * s]) > 0.9
    assert np.median(got[np.abs(x) < 0.5 * s] / x[np.abs(x) < 0.5 * s]) < 0.3


def test_adaptive_fit_agrees_with_plain_em():
    _, x, s = _mixture(3000, seed=2)
    sd = ash.prior_grid(x, s)
    lik = ash.likelihood(x, s, sd)
    pi, cycles = ash.fit_weights(lik)
    assert 1 <= cycles < ash.MAX_CYCLES
    p = np.full(len(sd), 1.0 / len(sd))
    for _ in range(20000):
        nk = p * (lik.T @ (1.0 / (lik @ p)))
        p = nk / nk.sum()
    assert np.isclose(pi.sum(), 1.0) and pi.min() >= 0
    # the stop is a plateau, not the optimum: within a millionth of a nat per gene of 20,000 steps
    assert float(np.log(lik @ pi).sum()) >= float(np.log(lik @ p).sum()) - 1e-6 * x.size
    a, b = ash.posterior_mean(x, s, sd, pi), ash.posterior_mean(x, s, sd, p)
    assert np.sqrt(np.mean((a - b) ** 2) / np.mean(b**2)) < 0.01
    # the cap is honoured and reported
    stats = {}
    ash.adaptive_shrink(x, s * s, stats=stats, max_cycles=2)
    assert stats["adaptive_cycles"] == 2 and stats["adaptive_fits_at_cap"] == 1


def test_adaptive_constants_and_a_reference_fit_are_pinned():
    """The stopping constants are part of the model a scored arm was built with: moving one
    must fail here, not pass quietly. The reference values hold to 1e-3, not to the bit --
    the fit is reproducible to about 1e-4 across BLAS builds (the module's docstring)."""
    assert (ash.MAX_CYCLES, ash.TOL_PER_GENE, ash.CALM_CYCLES, ash.MIN_GENES) == (5000, 1e-9, 3, 50)
    _, x, s = _mixture(2000, seed=11)
    stats = {}
    got = ash.adaptive_shrink(x, s * s, stats=stats)
    assert np.allclose(got[:5], [-0.000564, -0.015739, 0.033192, -0.009072, 0.007138], rtol=5e-3, atol=2e-5)
    assert np.isclose(np.abs(got).sum(), 278.6678, rtol=1e-3)
    assert 2000 < stats["adaptive_cycles"] < 3500 and stats["adaptive_fits_at_cap"] == 0


def test_adaptive_on_a_vector_that_is_all_noise():
    """Nothing but noise is the flat-likelihood case real knockdowns sit near: the fit runs to
    its cap, says so, stays finite, and removes nearly everything."""
    r = np.random.default_rng(3)
    s = np.exp(r.uniform(np.log(0.05), np.log(1.0), 400))
    x = r.normal(size=400) * s
    stats = {}
    got = ash.adaptive_shrink(x, s * s, stats=stats)
    assert np.all(np.isfinite(got)) and np.count_nonzero(got) == 400
    assert np.abs(got).sum() < 0.05 * np.abs(x).sum()
    assert stats["adaptive_fits_at_cap"] == 1 and stats["adaptive_cycles"] == ash.MAX_CYCLES
    assert np.array_equal(ash.adaptive_shrink(np.zeros(100), np.full(100, 0.04)), np.zeros(100))


def test_adaptive_refuses_a_grid_that_means_a_broken_variance():
    _, x, s = _mixture(200, seed=5)
    var = s * s
    var[0] = 1e-300
    with pytest.raises(ValueError, match="prior grid"):
        ash.adaptive_shrink(x, var)
    x2 = x.copy()
    x2[0] = 1e160
    with pytest.raises(ValueError, match="prior grid"):
        ash.adaptive_shrink(x2, s * s)


def test_adaptive_leaves_unusable_genes_and_thin_vectors_alone():
    _, x, s = _mixture(400, seed=3)
    var = s * s
    var[:5] = np.inf
    var[5:8] = 0.0
    x2 = x.copy()
    x2[8] = np.nan
    got = ash.adaptive_shrink(x2, var)
    assert np.array_equal(got[:8], x2[:8]) and np.isnan(got[8])
    assert not np.array_equal(got[9:], x2[9:])
    stats = {}
    thin_in = x[:49].copy()
    thin = ash.adaptive_shrink(thin_in, (s * s)[9:58], stats=stats)        # 49 usable genes: no fit
    assert np.array_equal(thin, thin_in) and thin is not thin_in and stats == {"adaptive_too_few_genes": 1}
    fits = {}
    fifty = ash.adaptive_shrink(x[9:59], (s * s)[9:59], stats=fits)          # 50: fitted
    assert fits["adaptive_fits"] == 1 and not np.array_equal(fifty, x[9:59])


def test_adaptive_runs_after_pooling_inside_pooled_delta():
    _, x1, s1 = _mixture(300, seed=4)
    _, x2, s2 = _mixture(300, seed=5)
    genes = np.array([f"g{i}" for i in range(300)])
    a, b = _lfc(x1, s1 * s1, genes=genes), _lfc(x2, s2 * s2, genes=genes)
    w1, w2 = 1 / (s1 * s1), 1 / (s2 * s2)
    raw = (x1 * w1 + x2 * w2) / (w1 + w2)
    stats = {}
    got = pooled_delta("T", [a, b], genes, shrink_stage="pooled", shrink_rule="adaptive", stats=stats)
    assert np.allclose(got, ash.adaptive_shrink(raw, 1 / (w1 + w2)), rtol=1e-10, atol=1e-14)
    assert stats["adaptive_fits"] == 1 and np.count_nonzero(got) == 300
    assert np.array_equal(pooled_delta("T", [a, b], genes, shrinkage=False, shrink_stage="pooled",
                                       shrink_rule="adaptive"), raw)


# ----------------------------------------------------------------- the two entry points --


def _parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-shrink", action="store_true")
    ap.add_argument("--shrink-source", action="append", default=[])
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--coverage-tiers", default=None)
    build.add_shrink_args(ap, twin="x")
    args = ap.parse_args(argv)
    build.check_shrink_args(ap, args)
    return args


def test_the_flags_become_pooled_deltas_keyword_arguments():
    assert build.shrink_kwargs(_parse([])) == {"shrinkage": True, "shrink_k": 1.0,
                                               "shrink_stage": "source", "shrink_rule": "garrote"}
    assert build.shrink_kwargs(_parse(["--shrink-k", "8", "--shrink-stage", "pooled"])) == {
        "shrinkage": True, "shrink_k": 8.0, "shrink_stage": "pooled", "shrink_rule": "garrote"}
    assert build.shrink_kwargs(_parse(["--no-shrink"]))["shrinkage"] is False
    # the depth-aware arm: the rule is off globally and pinned on for one source, so k is live
    deep = build.shrink_kwargs(_parse(["--no-shrink", "--shrink-source", "x.npz:c", "--shrink-k", "4"]))
    assert deep == {"shrinkage": False, "shrink_k": 4.0, "shrink_stage": "source", "shrink_rule": "garrote"}
    # and they are exactly pooled_delta's own names
    a = _lfc([0.5, -2.0, 0.1, 3.0], [0.01, 0.1, 0.2, 0.5])
    kw = build.shrink_kwargs(_parse(["--shrink-k", "4", "--shrink-stage", "pooled"]))
    assert np.array_equal(pooled_delta("T", [a], AXIS, **kw),
                          pooled_delta("T", [a], AXIS, shrink_k=4.0, shrink_stage="pooled"))


@pytest.mark.parametrize("argv", [
    ["--no-shrink", "--shrink-k", "8"],                       # inert: the rule is off
    ["--no-shrink", "--shrink-source", "x.npz:c", "--shrink-stage", "pooled"],
    ["--no-shrink", "--shrink-stage", "pooled"],
    ["--no-shrink", "--shrink-stage", "pooled", "--shrink-rule", "adaptive"],
    ["--shrink-k", "0"],
    ["--shrink-rule", "adaptive"],                            # adaptive is after pooling only
    ["--shrink-rule", "adaptive", "--shrink-stage", "pooled", "--shrink-k", "8"],
    ["--shrink-stage", "pooled", "--shrink-source", "x.npz:c"],
    ["--shrink-stage", "pooled", "--gamma", "0.5"],
    ["--shrink-stage", "pooled", "--coverage-tiers", "3:0.1"],
])
def test_inert_and_undefined_flag_settings_are_refused(argv):
    with pytest.raises(SystemExit):
        _parse(argv)


def _fold(tmp_path, g=60):
    rng = np.random.default_rng(6)
    genes = np.array([f"g{i}" for i in range(g)], dtype=object)
    basal = rng.uniform(100, 2000, size=g)
    mean = np.stack([basal, basal * np.exp2(rng.normal(0, 0.15, g)), basal * np.exp2(rng.normal(0, 0.15, g))])
    n = np.full(3, 1000, dtype=np.int64)
    src = PseudobulkSums(labels=["ctrl", "g0", "g1"], genes=genes.copy(), count_sum=mean * n[:, None],
                         cpm_sum=mean * n[:, None], cpm_sq_sum=(mean**2 + mean) * n[:, None],
                         n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["t"])
    rows, labels = [], []
    for lab, k in (("non-targeting", 40), ("g0", 12), ("g1", 12)):
        for _ in range(k):
            rows.append(rng.poisson(basal / basal.sum() * rng.integers(3000, 6000)))
            labels.append(lab)
    real = ad.AnnData(X=sp.csr_matrix(np.asarray(rows, dtype=np.float32)),
                      obs=pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(len(rows))]),
                      var=pd.DataFrame(index=genes.astype(str)))
    real.write_h5ad(tmp_path / "real.h5ad")
    return tmp_path / "real.h5ad", src


def test_loco_hands_the_rule_to_every_pool_and_records_it(monkeypatch, tmp_path):
    real_path, src = _fold(tmp_path)
    seen = []
    orig = loco.pooled_delta
    monkeypatch.setattr(loco, "pooled_delta", lambda *a, **k: seen.append(
        {x: k.get(x) for x in ("shrinkage", "shrink_k", "shrink_stage", "shrink_rule")}) or orig(*a, **k))
    kw = dict(pert_col="perturbation", control="non-targeting", var_floor="poisson",
              emit_lambda=0.5, alpha=1.35, min_libsize=0.0)
    plain = loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "plain.h5ad", **kw)
    assert seen and all(s == {"shrinkage": True, "shrink_k": 1.0, "shrink_stage": "source",
                              "shrink_rule": "garrote"} for s in seen)
    assert (plain["shrink_k"], plain["shrink_stage"], plain["shrink_rule"]) == (1.0, "source", "garrote")
    seen.clear()
    hard = loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "hard.h5ad",
                                          shrink_k=8.0, shrink_stage="pooled", **kw)
    assert seen and all(s == {"shrinkage": True, "shrink_k": 8.0, "shrink_stage": "pooled",
                              "shrink_rule": "garrote"} for s in seen)
    assert (hard["shrink_k"], hard["shrink_stage"], hard["shrink_rule"]) == (8.0, "pooled", "garrote")
    assert "adaptive_fit" not in plain and "adaptive_fit" not in hard
    with pytest.raises(ValueError):                                # refused before any file is written
        loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "bad.h5ad",
                                       shrink_rule="adaptive", **kw)
    assert not (tmp_path / "bad.h5ad").exists()
    soft = loco.build_transfer_prediction(real_path, [(src, "ctrl")], tmp_path / "soft.h5ad",
                                          shrink_stage="pooled", shrink_rule="adaptive", **kw)
    assert soft["adaptive_fit"]["max_cycles"] == ash.MAX_CYCLES and soft["adaptive_fit"]["min_genes"] == 50
    assert soft["pool_stats"]["adaptive_fits"] == 2               # one fit per target, counted
    a, b = ad.read_h5ad(tmp_path / "plain.h5ad"), ad.read_h5ad(tmp_path / "hard.h5ad")
    assert a.shape == b.shape and (a.X != b.X).nnz > 0             # the rule reached the cells


def test_loco_refuses_an_inert_flag_before_any_work(tmp_path, capsys):
    argv = ["--real", str(tmp_path / "no.h5ad"), "--bundle", str(tmp_path / "b"),
            "--out", str(tmp_path / "arm"), "--source", "x.npz:c", "--no-shrink", "--shrink-k", "8"]
    with pytest.raises(SystemExit):
        loco.main(argv)
    assert "--shrink-k with --no-shrink" in capsys.readouterr().err


# the board builder: the same fixture shape as test_submit_build_alpha_bulk.py
GENES3 = ["A", "B", "C"]


@pytest.fixture
def challenge(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "gene_names.csv").write_text("gene_name\n" + "\n".join(GENES3) + "\n")
    (data / "pert_counts.csv").write_text("target_gene\nTP53\n")
    depths = [1500, 2000, 2500, 3000, 1800, 2200, 2700, 1600, 2400, 2900, 2100, 1900]
    for name, base in (("ctx_x.h5ad", [10, 1000, 990]), ("ctx_y.h5ad", [1000, 505, 495])):
        b = np.asarray(base, float)
        X = sp.csr_matrix(np.stack([np.rint(b * d / b.sum()) for d in depths]))
        ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(len(depths))]),
                   var=pd.DataFrame(index=GENES3)).write_h5ad(data / name)
    for name, label in (("h1.npz", "non-targeting"), ("gwps.npz", "control")):
        m = np.asarray([[100000.0, 450000.0, 450000.0], [200000.0, 400000.0, 400000.0]])
        n = np.full(2, 1000, dtype=np.int64)
        PseudobulkSums(labels=[label, "TP53"], genes=np.array(GENES3, dtype=object),
                       count_sum=m * n[:, None], cpm_sum=m * n[:, None],
                       cpm_sq_sum=(m**2 + 1.0) * n[:, None], n_cells=n,
                       libsize_sum=n.astype(float) * 2e4, sources=["test"]).save(data / name)
    cfg = {"data_dir": str(data), "gene_names_file": "gene_names.csv", "n_genes": 3,
           "pert_counts_file": "pert_counts.csv", "pert_col": "target_gene",
           "context_col": "context", "control_label": "non-targeting",
           "phase": "p1", "phases": {"p1": {"contexts": ["X", "Y"]}},
           "control_files": {"X": "ctx_x.h5ad", "Y": "ctx_y.h5ad"},
           "submission": {"cells_per_pert": 6, "max_counts_per_cell": 1_000_000,
                          "max_cells": 100_000, "max_stored_entries": 10_000_000}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    return {"cfg": tmp_path / "config.yaml", "data": data, "out": tmp_path / "out"}


def _argv(ch, stem, extra, emitter="delta-transfer"):
    return ["--challenge-config", str(ch["cfg"]), "--emitter", emitter,
            "--h1-cache", str(ch["data"] / "h1.npz"), "--gwps-cache", str(ch["data"] / "gwps.npz"),
            "--out", str(ch["out"] / stem), "--no-pack", "--min-libsize", "100", *extra]


def test_build_hands_pooled_delta_what_loco_does_and_records_the_flags(challenge, monkeypatch):
    seen = []
    orig = build.pooled_delta
    monkeypatch.setattr(build, "pooled_delta", lambda *a, **k: seen.append(
        {x: k.get(x) for x in ("shrinkage", "shrink_k", "shrink_stage", "shrink_rule")}) or orig(*a, **k))
    assert build.main(_argv(challenge, "plain", [])) == 0
    assert seen and all(s == {"shrinkage": True, "shrink_k": 1.0, "shrink_stage": "source",
                              "shrink_rule": "garrote"} for s in seen)
    seen.clear()
    assert build.main(_argv(challenge, "hard", ["--shrink-k", "8", "--shrink-stage", "pooled"])) == 0
    assert seen and all(s == {"shrinkage": True, "shrink_k": 8.0, "shrink_stage": "pooled",
                              "shrink_rule": "garrote"} for s in seen)
    rec = json.loads((challenge["out"] / "hard.args.json").read_text())
    assert (rec["shrink_k"], rec["shrink_stage"], rec["shrink_rule"]) == (8.0, "pooled", "garrote")


def test_build_refuses_the_flags_where_they_cannot_act(challenge):
    with pytest.raises(SystemExit):
        build.main(_argv(challenge, "inert", ["--no-shrink", "--shrink-k", "8"]))
    with pytest.raises(SystemExit):                              # no pooled delta to shrink
        build.main(_argv(challenge, "null", ["--shrink-k", "8"], emitter="control-null"))
    with pytest.raises(SystemExit):
        build.main(_argv(challenge, "ash", ["--shrink-rule", "adaptive"]))
    assert build.main(_argv(challenge, "ok", ["--shrink-rule", "adaptive", "--shrink-stage", "pooled"])) == 0
    # a moved rule is knob letter s, so a stem named like a model without it is refused
    with pytest.raises(SystemExit):
        build.main(_argv(challenge, "ser-99aefkw_k8pool_v1", ["--shrink-k", "8", "--shrink-stage", "pooled"]))


# ------------------------------------------------- paths a mutation review found unwalked --


def _spy(monkeypatch, module):
    seen = []
    orig = module.pooled_delta
    monkeypatch.setattr(module, "pooled_delta", lambda *a, **k: seen.append(
        {x: k.get(x) for x in KNOBS}) or orig(*a, **k))
    return seen


def test_build_hands_the_adaptive_rule_through(challenge, monkeypatch):
    """An arm named adaptive that silently ran the garrote could never be renamed (ADR 0005)."""
    seen = _spy(monkeypatch, build)
    assert build.main(_argv(challenge, "ash2", ["--shrink-rule", "adaptive", "--shrink-stage", "pooled"])) == 0
    assert seen and all(s == {"shrink_k": 1.0, "shrink_stage": "pooled", "shrink_rule": "adaptive"}
                        for s in seen)


def test_build_carries_the_rule_through_the_per_context_gamma_pool(challenge, monkeypatch):
    seen = _spy(monkeypatch, build)
    assert build.main(_argv(challenge, "g", ["--gamma", "0.5", "--shrink-k", "8"])) == 0
    assert seen and all(s == {"shrink_k": 8.0, "shrink_stage": "source", "shrink_rule": "garrote"}
                        for s in seen)


def test_loco_hands_the_rule_to_the_neighbour_pool_too(monkeypatch, tmp_path):
    """The pool is pooled with every knob the targets get, or the two live in different spaces."""
    real_path, src = _fold(tmp_path)
    seen = _spy(monkeypatch, loco)

    def fake_arm(args, delta_of, axis):
        assert delta_of("g0") is not None            # the pool pools through delta_of
        return SimpleNamespace(fuse=lambda p, d: d, summary=lambda: {}), {}
    monkeypatch.setattr(loco, "neighbour_arm_for", fake_arm)
    loco.build_transfer_prediction(
        real_path, [(src, "ctrl")], tmp_path / "nb.h5ad", pert_col="perturbation",
        control="non-targeting", var_floor="poisson", emit_lambda=0.5, alpha=1.35,
        min_libsize=0.0, neighbour_table=[tmp_path / "t.pt"], neighbour_w=[0.3],
        shrink_k=8.0, shrink_stage="pooled")
    assert len(seen) == 3 and all(s == {"shrink_k": 8.0, "shrink_stage": "pooled",
                                        "shrink_rule": "garrote"} for s in seen)


def test_build_neighbour_pool_pools_with_the_same_rule(monkeypatch, tmp_path):
    seen = _spy(monkeypatch, build)

    def fake_arm(args, delta_of, axis):
        delta_of("g0")
        return (SimpleNamespace(fuse=lambda p, d: d, summary=lambda: {}),
                {"k": 1, "w": 1, "pool_used": 1, "pool_requested": 1, "targets_fused": 0})
    monkeypatch.setattr(build, "neighbour_arm_for", fake_arm)
    a = _lfc([0.5, -2.0, 0.1, 3.0], [0.01, 0.1, 0.2, 0.5], label="g0")
    args = SimpleNamespace(no_shrink=False, shrink_k=8.0, shrink_stage="pooled",
                           shrink_rule="garrote", log_bias_correct=False, var_floor="none")
    build.fuse_neighbours(args, {"g0": np.zeros(4)}, [], [a], AXIS, None, tmp_path / "rec.json")
    assert seen and all(s == {"shrink_k": 8.0, "shrink_stage": "pooled", "shrink_rule": "garrote"}
                        for s in seen)


def test_adaptive_prior_grid_spans_a_tenth_of_the_smallest_error_to_twice_the_largest_effect():
    s = np.array([0.05, 0.2, 1.0])
    x = np.array([0.0, 0.0, 3.0])
    sd = ash.prior_grid(x, s)
    assert sd[0] == pytest.approx(s.min() / 10.0)
    assert sd[-1] >= 2.0 * np.sqrt(np.max(x * x - s * s)) > sd[-2]
    assert np.allclose(sd[1:] / sd[:-1], np.sqrt(2.0))
    # each row of the likelihood is scaled to a maximum of exactly 1
    assert np.allclose(ash.likelihood(x, s, sd).max(axis=1), 1.0)


def test_the_adaptive_stop_waits_for_several_quiet_cycles(monkeypatch):
    _, x, s = _mixture(3000, seed=2)
    lik = ash.likelihood(x, s, ash.prior_grid(x, s))
    n_three = ash.fit_weights(lik)[1]
    monkeypatch.setattr(ash, "CALM_CYCLES", 1)
    assert ash.fit_weights(lik)[1] < n_three


# --------------------------------------------- the letter s and the build's shrinkage record --


def test_a_model_named_build_with_a_moved_rule_needs_the_letter_s(challenge, capsys):
    """ADR 0005 (2026-10-03): `s` is the fold-change shrinkage rule moved off the baseline garrote,
    with the rule named in the slug; `n` is the same object switched off; no letter is the baseline."""
    flags = ["--shrink-k", "8", "--shrink-stage", "pooled"]
    assert build.main(_argv(challenge, "ser-99aefksw_k8pool_v1", flags)) == 0
    rec = json.loads((challenge["out"] / "ser-99aefksw_k8pool_v1.shrink.json").read_text())
    assert rec == {"shrink_k": 8.0, "shrink_stage": "pooled", "shrink_rule": "garrote"}
    capsys.readouterr()
    for stem, extra, why in (
            ("ser-99aefkw_k8pool_v1", flags, "must carry the letter s"),          # moved rule, no s
            ("ser-99aefksw_shrink_v1", [], "carries the letter s"),               # s, but the baseline rule
            ("ser-99aefknsw_k8pool_v1", flags, "both n"),                         # n and s together
            # the depth-aware arm at a harder threshold: a real model with no name yet
            ("ser-99dn_k8_v1", ["--no-shrink", "--shrink-source", "x.npz:c", "--shrink-k", "8"], "no registered name"),
            ("ser-99ds_k8_v1", ["--no-shrink", "--shrink-source", "x.npz:c", "--shrink-k", "8"], "no registered name")):
        with pytest.raises(SystemExit):
            build.main(_argv(challenge, stem, extra))
        assert why in capsys.readouterr().err, stem
    # the baseline rule writes no shrinkage record: its records keep their shipped shape
    assert build.main(_argv(challenge, "ser-99aefkw_shrink_v1", [])) == 0
    assert not (challenge["out"] / "ser-99aefkw_shrink_v1.shrink.json").exists()


def test_the_adaptive_build_records_its_fit(challenge):
    assert build.main(_argv(challenge, "ser-99aefksw_ashpool_v1",
                            ["--shrink-stage", "pooled", "--shrink-rule", "adaptive"])) == 0
    rec = json.loads((challenge["out"] / "ser-99aefksw_ashpool_v1.shrink.json").read_text())
    assert (rec["shrink_rule"], rec["shrink_stage"]) == ("adaptive", "pooled")
    assert rec["adaptive_fit"] == {"max_cycles": ash.MAX_CYCLES, "tol_per_gene": ash.TOL_PER_GENE,
                                   "calm_cycles": ash.CALM_CYCLES, "min_genes": ash.MIN_GENES,
                                   "max_components": ash.MAX_COMPONENTS}
    # three genes is too thin for a prior: the one target is counted as unshrunk, not silently passed
    assert rec["fits_targets"] == {"adaptive_too_few_genes": 1}
    assert rec["fits_neighbour_pool"] is None                        # this fixture has no neighbour arm
    assert set(rec["numerics"]) == {"numpy", "blas", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"}


def test_the_neighbour_pool_fits_are_counted_apart(monkeypatch, tmp_path):
    seen = {}

    def fake_arm(args, delta_of, axis):
        delta_of("g0")
        return (SimpleNamespace(fuse=lambda p, d: d, summary=lambda: {}),
                {"k": 1, "w": 1, "pool_used": 1, "pool_requested": 1, "targets_fused": 0})
    monkeypatch.setattr(build, "neighbour_arm_for", fake_arm)
    genes = np.array([f"g{i}" for i in range(300)])
    _, x, s = _mixture(300, seed=9)
    a = _lfc(x, s * s, genes=genes, label="g0")
    args = SimpleNamespace(no_shrink=False, shrink_k=1.0, shrink_stage="pooled", shrink_rule="adaptive",
                           log_bias_correct=False, var_floor="none")
    build.fuse_neighbours(args, {}, [], [a], genes, None, tmp_path / "rec.json", fit_stats=seen)
    assert seen.get("adaptive_fits") == 1
