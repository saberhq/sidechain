"""The variance behind the pooling weight as a knob (T98): what each model is, and what must not move.

Three properties carry the A/B: the default is bit-identical to every historical call; each
model's variance is the formula its docstring states (checked against a hand-computed form on
a synthetic Gamma-Poisson artifact whose dispersion is known); and the controls -- a shuffle, a
flat multiplier, the rule held on the shipped variance -- are the same code path with one thing
moved. A fourth: a fit attached to the wrong source, or a factor naming no source, is refused.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sidechain.data.dispersion import (
    GeneDispersion,
    fit_gene_dispersion,
    fit_gene_dispersion_file,
    moment_dispersion,
    moment_dispersion_file,
)
from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.submit.build import LN2_SQ, _log2fc_with_var, pooled_delta, shrink
from sidechain.submit.variance_model import (
    SHUFFLE_BINS,
    STRATUM_EDGES,
    VarianceModel,
    apply_dispersion_fits,
    check_variance_args,
    expression_bins,
    parse_dispersion_fits,
    permutation_within_bins,
    read_gene_list,
)
from tests.test_dispersion import synthetic

PC = 1.0


def _fitted(n_genes=120, n_groups=60, n_cells=40, seed=3, df0=None):
    """A synthetic arm (known dispersion curve), named like a loaded source, with its fit attached."""
    pb, theta_true = synthetic(n_genes=n_genes, n_groups=n_groups, n_cells=n_cells, seed=seed)
    pb.labels[0] = "control"
    pb.sidechain_name = "arm"
    pb.dispersion_fit = fit_gene_dispersion(pb, df0=df0)
    return pb, theta_true


def _row(pb, label):
    i = pb.labels.index(label)
    n = max(int(pb.n_cells[i]), 1)
    m = pb.cpm_sum[i] / n
    v = np.maximum(pb.cpm_sq_sum[i] / n - m * m, 0.0)
    s = 1e6 / (pb.libsize_sum[i] / n)
    v = np.maximum(v, (m + PC) * s)          # the Poisson floor, as shipped
    return n, m, v, s


def _term(n, m, v):
    return (v / n) / (m + PC) ** 2


# ---------------------------------------------------------------- the default does not move


def test_the_default_and_shipped_are_bit_identical():
    pb, _ = _fitted()
    axis = np.asarray(pb.genes)
    plain = pooled_delta("T5", [(pb, "control")], axis, shrinkage=False, var_floor="poisson")
    for vm in (VarianceModel(), VarianceModel.parse("shipped"), VarianceModel.parse(None)):
        assert np.array_equal(plain, pooled_delta("T5", [(pb, "control")], axis, shrinkage=False,
                                                  var_floor="poisson", variance_model=vm))
    # and the shipped path never reads the fit, so a source WITHOUT one pools the same
    bare, _ = synthetic(n_genes=120, n_groups=60, n_cells=40, seed=3)
    bare.labels[0] = "control"
    assert np.array_equal(plain, pooled_delta("T5", [(bare, "control")], axis, shrinkage=False,
                                              var_floor="poisson"))


# ---------------------------------------------------------------- the spec grammar


@pytest.mark.parametrize("spec", [
    "shipped", "trend", "trend:shuffle=7", "own", "own:shuffle=20261009", "sql", "sql:shuffle=1",
    "category:list=/tmp/x.txt", "category:list=/tmp/x.txt,shuffle=3", "flat", "flat:theta=0.3",
    "flat:h1_pseudobulk=0.042,hct116_full_union849=0.112",
    "multiplier:h1_pseudobulk=0.95/1.1/1.26,hct116_full=1.08", "multiplier:a=2",
])
def test_parse_round_trips_through_spec(spec):
    vm = VarianceModel.parse(spec)
    assert VarianceModel.parse(vm.spec()) == vm
    assert vm.is_shipped == (spec == "shipped")


@pytest.mark.parametrize("spec, why", [
    ("trended", "unknown model"), ("trend:seed=3", "unknown option"), ("own:shuffle=x", "integer seed"),
    ("category", "needs list="), ("category:shuffle=2", "needs list="),
    ("multiplier", "at least one"), ("multiplier:a", "not NAME=K"), ("multiplier:a=1/2", "one factor"),
    ("multiplier:a=0", "finite and > 0"), ("multiplier:a=1,a=2", "given twice"),
    ("multiplier:a=nan", "finite and > 0"), ("trend:shuffle=1,shuffle=2", "given twice"),
])
def test_parse_refuses_what_it_cannot_mean(spec, why):
    with pytest.raises(ValueError, match=why):
        VarianceModel.parse(spec)


def test_the_flat_multiplier_is_three_equal_factors():
    vm = VarianceModel.parse("multiplier:a=1.5,b=1/2/3")
    assert vm.multiplier_for("a") == (1.5, 1.5, 1.5)
    assert vm.multiplier_for("b") == (1.0, 2.0, 3.0)
    assert vm.multiplier_for("c") is None
    assert vm.spec() == "multiplier:a=1.5,b=1/2/3"


# ---------------------------------------------------------------- what each model computes


def test_trend_is_the_curve_at_the_rows_own_mean_plus_the_poisson_term():
    """`(m + c) s + theta_trend(mu_row) m^2` on BOTH arms, the curve interpolated at the row's own
    mean raw count -- T84's `alt_var(kind='trend')`, which the wiring gate reproduces."""
    pb, _ = _fitted()
    gd = pb.dispersion_fit
    vm = VarianceModel.parse("trend")
    fc, var = _log2fc_with_var(pb, "T7", "control", var_floor="poisson", variance=vm.for_source(pb, "control"))
    ok = gd.mean_count > 0
    o = np.argsort(gd.mean_count[ok], kind="stable")
    lx, ty = np.log(gd.mean_count[ok][o]), gd.theta_trend[ok][o]
    want = 0.0
    for lab in ("T7", "control"):
        n, m, _, s = _row(pb, lab)
        with np.errstate(divide="ignore"):
            th = np.interp(np.log(m / s), lx, ty)
        want = want + _term(n, m, (m + PC) * s + th * m * m)
    assert np.allclose(var, want / LN2_SQ, rtol=1e-12, atol=0)
    assert np.isfinite(var).all() and (var > 0).all()
    # the Poisson floor is inside the model, so the max never binds for the curve
    stats: dict = {}
    _log2fc_with_var(pb, "T7", "control", var_floor="poisson", variance=vm.for_source(pb, "control"), stats=stats)
    assert stats["variance_floor_bound_gene_arms"] == 0


def test_own_is_the_genes_dispersion_and_sql_is_variance_cpm_as_built():
    pb, _ = _fitted(df0=0.0)                       # df0 = 0: theta_sql IS theta_ql, so the algebra below is exact
    gd = pb.dispersion_fit
    own = VarianceModel.parse("own").for_source(pb, "control")
    sql = VarianceModel.parse("sql").for_source(pb, "control")
    n, m, v, s = _row(pb, "T3")
    assert np.allclose(own.cell_variance(m, s, v, PC, perturbed=True), (m + PC) * s + gd.theta_ml * m * m)
    assert np.allclose(sql.cell_variance(m, s, v, PC, perturbed=True),
                       gd.theta_sql * ((m + PC) * s + gd.theta_trend * m * m))


def test_at_the_arm_mean_own_and_sql_differ_only_by_the_pseudocounts_share():
    """Step 5 of the hand-off: `theta_QL (mu + theta_trend mu^2) = mu + theta_ML mu^2` at the gene's
    arm-level mean, so `sql` equals `own` there up to `(theta_sql - 1) * c * s` -- the pseudocount's
    term, which the quasi-likelihood form also multiplies. Off the arm mean they part."""
    pb, _ = _fitted(df0=0.0)
    gd = pb.dispersion_fit
    own = VarianceModel.parse("own").for_source(pb, "control")
    sql = VarianceModel.parse("sql").for_source(pb, "control")
    s = 1e6 / (pb.libsize_sum[1] / pb.n_cells[1])
    m = gd.mean_count * s                          # a row sitting exactly at the arm mean
    dummy = np.zeros_like(m)
    diff = sql.cell_variance(m, s, dummy, PC, perturbed=True) - own.cell_variance(m, s, dummy, PC, perturbed=True)
    assert np.allclose(diff, (gd.theta_sql - 1.0) * PC * s, rtol=1e-9, atol=1e-9)
    # the same identity in count units, pseudocount-free, is exact
    mu = gd.mean_count
    assert np.allclose(gd.theta_sql * (mu + gd.theta_trend * mu * mu), mu + gd.theta_ml * mu * mu, rtol=1e-10)
    # and off the arm mean the two forms disagree (the knockdown's own gene, say, at a tenth)
    m10 = m / 10.0
    off = sql.cell_variance(m10, s, dummy, PC, perturbed=True) - own.cell_variance(m10, s, dummy, PC, perturbed=True)
    assert not np.allclose(off, (gd.theta_sql - 1.0) * PC * s, rtol=1e-3)


def test_the_floor_max_binds_for_sql_where_theta_sql_is_under_one_and_is_counted():
    pb, _ = _fitted(df0=0.0)
    gd = pb.dispersion_fit
    assert (gd.theta_sql < 1).any(), "the synthetic fit should have genes under their trend"
    vm = VarianceModel.parse("sql")
    stats: dict = {}
    _, var = _log2fc_with_var(pb, "T3", "control", var_floor="poisson", variance=vm.for_source(pb, "control"), stats=stats)
    n, m, v, s = _row(pb, "T3")
    nc, mc, vc, sc = _row(pb, "control")
    raw_i = gd.theta_sql * ((m + PC) * s + gd.theta_trend * m * m)
    raw_c = gd.theta_sql * ((mc + PC) * sc + gd.theta_trend * mc * mc)
    bound = (raw_i < (m + PC) * s) | (raw_c < (mc + PC) * sc)
    assert stats["variance_floor_bound_gene_arms"] == int(bound.sum())
    want = _term(n, m, np.maximum(raw_i, (m + PC) * s)) + _term(nc, mc, np.maximum(raw_c, (mc + PC) * sc))
    assert np.allclose(var, want / LN2_SQ, rtol=1e-12, atol=0)


def test_a_shuffle_keeps_each_expression_bins_summed_overdispersion_and_moves_theta_to_other_genes():
    """The own and sql shuffles are expression- and size-matched: theta moves only among genes of the same
    expression bin (the dispersion falls 10- to 300-fold with expression, so a shuffle over the whole axis would
    hand a highly expressed gene a low-expression theta), and each bin is rescaled so its summed `theta mu^2`
    equals the real one (within a bin theta still falls with expression, so a bare permutation inflated the
    summed variance 1.4 to 2.1x on three pool sources)."""
    pb, _ = _fitted(n_genes=1500, n_groups=40, n_cells=30)     # enough genes for 100 bins of a dozen or more
    gd = pb.dispersion_fit
    ok = gd.mean_count > 0
    mu2 = gd.mean_count ** 2
    own = VarianceModel.parse("own:shuffle=11").for_source(pb, "control")
    assert not np.array_equal(own.theta, gd.theta_ml)
    p = permutation_within_bins(np.random.default_rng(11), gd.mean_count, ok)
    bins = expression_bins(gd.mean_count, ok)
    assert len(bins) == SHUFFLE_BINS == 100
    for chunk in bins:
        # the same values up to one factor per bin, and the bin's summed overdispersion kept
        ratio = own.theta[chunk] / np.where(gd.theta_ml[p][chunk] > 0, gd.theta_ml[p][chunk], np.nan)
        finite = np.isfinite(ratio)
        assert finite.sum() == 0 or np.allclose(ratio[finite], ratio[finite][0])
        assert np.isclose((own.theta[chunk] * mu2[chunk]).sum(), (gd.theta_ml[chunk] * mu2[chunk]).sum(), rtol=1e-9)
    assert np.isclose((own.theta[ok] * mu2[ok]).sum(), (gd.theta_ml[ok] * mu2[ok]).sum(), rtol=1e-9)
    lo, hi = own.record["shuffle_bin_scale_range"]
    assert 0.2 < lo <= 1.0 <= hi < 5.0 and "within 100 quantile bins" in own.record["shuffle"]
    sql = VarianceModel.parse("sql:shuffle=11").for_source(pb, "control")
    assert np.array_equal(sql.theta_sql, gd.theta_sql[p])                        # the pair travels together...
    for chunk in bins:                                                           # ...and its term is rescaled per bin
        assert np.isclose((sql.theta_sql[chunk] * sql.theta[chunk] * mu2[chunk]).sum(),
                          (gd.theta_sql[chunk] * gd.theta_trend[chunk] * mu2[chunk]).sum(), rtol=1e-9)
    # the trend's shuffle scrambles the curve over the whole axis and says so: a diagnostic, not the control
    trend = VarianceModel.parse("trend:shuffle=11").for_source(pb, "control")
    assert "not size-preserving" in trend.record["shuffle"]
    # two seeds, two permutations; the same seed, the same permutation
    a = VarianceModel.parse("own:shuffle=1").for_source(pb, "control").theta
    b = VarianceModel.parse("own:shuffle=2").for_source(pb, "control").theta
    assert not np.array_equal(a, b)
    assert np.array_equal(a, VarianceModel.parse("own:shuffle=1").for_source(pb, "control").theta)
    plain = VarianceModel.parse("trend").for_source(pb, "control")
    assert np.array_equal(trend.curve[0], plain.curve[0])                         # same expression axis
    assert np.array_equal(np.sort(trend.curve[1]), np.sort(plain.curve[1]))       # same values, scrambled
    assert not np.array_equal(trend.curve[1], plain.curve[1])


def test_flat_is_one_dispersion_for_every_gene_matched_to_the_trends_summed_overdispersion():
    """`flat`: the Poisson term plus one theta for every gene, the no-curve control the trend is read against. The
    default theta is the mean-count-squared-weighted mean of the trend over expressed genes, so the summed
    `theta mu^2` equals the trend's (a median would sit on the barely-expressed tail, 10 to 100x the curve at
    high expression). Per source or for every source, a value can be given instead."""
    pb, _ = _fitted(n_genes=700, n_groups=40, n_cells=30)
    gd = pb.dispersion_fit
    ok = gd.mean_count > 0
    mu2 = gd.mean_count[ok] ** 2
    matched = float((gd.theta_trend[ok] * mu2).sum() / mu2.sum())
    flat = VarianceModel.parse("flat").for_source(pb, "control")
    assert np.allclose(flat.theta, matched) and flat.record["flat_theta_size_matched"] == matched
    assert np.isclose((flat.theta[ok] * mu2).sum(), (gd.theta_trend[ok] * mu2).sum())
    assert flat.record["flat_theta_from"].startswith("the mean-count-squared-weighted mean")
    n, m, v, s = _row(pb, "T3")
    assert np.allclose(flat.cell_variance(m, s, v, PC, perturbed=True), (m + PC) * s + matched * m * m)
    given = VarianceModel.parse("flat:theta=0.05")
    assert given.spec() == "flat:theta=0.05" and VarianceModel.parse(given.spec()) == given
    assert np.allclose(given.for_source(pb, "control").theta, 0.05)
    per = VarianceModel.parse("flat:arm=0.11,other=0.2")
    assert per.spec() == "flat:arm=0.11,other=0.2" and VarianceModel.parse(per.spec()) == per
    assert np.allclose(per.for_source(pb, "control").theta, 0.11) and per.flat_theta_for("nobody") is None
    with pytest.raises(ValueError, match="finite and >= 0"):
        VarianceModel.parse("flat:theta=-1")
    with pytest.raises(ValueError, match="nothing to shuffle"):
        VarianceModel.parse("flat:shuffle=3")                 # a constant has nothing to shuffle
    with pytest.raises(ValueError, match="not both"):
        VarianceModel.parse("flat:theta=0.1,arm=0.2")
    assert VarianceModel.parse("flat").needs_fit and VarianceModel.parse("flat").floor_max


def test_the_multiplier_scales_the_perturbed_arms_term_by_the_controls_stratum():
    pb, _ = _fitted()
    nc, mc, vc, sc = _row(pb, "control")
    code = np.digitize(mc, STRATUM_EDGES)
    assert len(set(code.tolist())) >= 3, "the synthetic control should span several strata"
    vm = VarianceModel.parse("multiplier:arm=2/3/4")
    fc, var = _log2fc_with_var(pb, "T3", "control", var_floor="poisson", variance=vm.for_source(pb, "control"))
    fc0, var0 = _log2fc_with_var(pb, "T3", "control", var_floor="poisson")
    assert np.array_equal(fc, fc0)                                     # the fold change is never touched
    n, m, v, s = _row(pb, "T3")
    k = np.array([1.0, 2.0, 3.0, 4.0])[code]
    want = k * _term(n, m, v) + _term(nc, mc, vc)
    assert np.allclose(var, want / LN2_SQ, rtol=1e-12, atol=0)
    assert np.array_equal(var[code == 0], var0[code == 0])             # below 1 CPM: untouched
    # the flat form is one factor from 1 CPM up
    flat = VarianceModel.parse("multiplier:arm=1.5")
    _, varf = _log2fc_with_var(pb, "T3", "control", var_floor="poisson", variance=flat.for_source(pb, "control"))
    assert np.allclose(varf, (np.where(code == 0, 1.0, 1.5) * _term(n, m, v) + _term(nc, mc, vc)) / LN2_SQ)
    # a factor under 1 deflates below the floored shipped variance: no max for the multiplier
    low = VarianceModel.parse("multiplier:arm=0.5/0.5/0.5")
    _, varl = _log2fc_with_var(pb, "T3", "control", var_floor="poisson", variance=low.for_source(pb, "control"))
    assert (varl[code > 0] < var0[code > 0]).all()


def test_a_multiplier_leaves_an_unnamed_source_on_the_shipped_variance_and_counts_it():
    pb, _ = _fitted()
    other, _ = synthetic(n_genes=120, n_groups=60, n_cells=40, seed=4)
    other.labels[0] = "control"
    other.sidechain_name = "other"
    axis = np.asarray(pb.genes)
    vm = VarianceModel.parse("multiplier:arm=3")
    stats: dict = {}
    out = pooled_delta("T3", [(pb, "control"), (other, "control")], axis, shrinkage=False,
                       var_floor="poisson", variance_model=vm, stats=stats)
    assert stats["variance_model_sources_unmodelled"] == 1
    # by hand: arm's weight from 3x its term, other's from its shipped variance
    fa, va = _log2fc_with_var(pb, "T3", "control", var_floor="poisson", variance=vm.for_source(pb, "control"))
    fo, vo = _log2fc_with_var(other, "T3", "control", var_floor="poisson")
    wa, wo = 1.0 / np.maximum(va, 1e-12), 1.0 / np.maximum(vo, 1e-12)
    assert np.allclose(out, (fa * wa + fo * wo) / (wa + wo))
    # naming a source nobody loaded is refused once the sources are known
    with pytest.raises(SystemExit, match="match no pseudobulk source"):
        VarianceModel.parse("multiplier:nobody=2").check_sources([(pb, "control"), (other, "control")])
    # and a per-source flat value on a name nobody loaded, which would otherwise fall back to the default
    with pytest.raises(SystemExit, match="flat names .* match no pseudobulk source"):
        VarianceModel.parse("flat:nobody=0.1").check_sources([(pb, "control"), (other, "control")])


def test_rule_variance_shipped_holds_the_pooled_rule_on_the_shipped_den_while_the_weights_move():
    """The second knob: the weights follow the model, the rule's `1/sum(w)` stays the shipped one."""
    pb, _ = _fitted()
    other, _ = synthetic(n_genes=120, n_groups=60, n_cells=40, seed=4)
    other.labels[0] = "control"
    other.sidechain_name = "other"
    axis = np.asarray(pb.genes)
    srcs = [(pb, "control"), (other, "control")]
    vm = VarianceModel.parse("multiplier:arm=4,other=0.5")
    fa, va, va0 = _log2fc_with_var(pb, "T3", "control", var_floor="poisson", variance=vm.for_source(pb, "control"), return_shipped=True)
    fo, vo, vo0 = _log2fc_with_var(other, "T3", "control", var_floor="poisson", variance=vm.for_source(other, "control"), return_shipped=True)
    wa, wo = 1.0 / np.maximum(va, 1e-12), 1.0 / np.maximum(vo, 1e-12)
    pooled = (fa * wa + fo * wo) / (wa + wo)
    den_model, den_shipped = wa + wo, 1.0 / np.maximum(va0, 1e-12) + 1.0 / np.maximum(vo0, 1e-12)
    held = pooled_delta("T3", srcs, axis, shrinkage=True, shrink_stage="pooled", var_floor="poisson",
                        variance_model=vm, rule_variance="shipped")
    follow = pooled_delta("T3", srcs, axis, shrinkage=True, shrink_stage="pooled", var_floor="poisson",
                          variance_model=vm, rule_variance="model")
    assert np.allclose(held, shrink(pooled, 1.0 / den_shipped, 1.0))
    assert np.allclose(follow, shrink(pooled, 1.0 / den_model, 1.0))
    assert not np.allclose(held, follow)
    # at the source stage the garrote reads the held variance too
    held_src = pooled_delta("T3", srcs, axis, shrinkage=True, var_floor="poisson", variance_model=vm, rule_variance="shipped")
    fa_s, fo_s = shrink(fa, va0, 1.0), shrink(fo, vo0, 1.0)
    assert np.allclose(held_src, (fa_s * wa + fo_s * wo) / (wa + wo))
    # inert settings are refused rather than ignored
    with pytest.raises(ValueError, match="without a variance model"):
        pooled_delta("T3", srcs, axis, shrinkage=False, var_floor="poisson", rule_variance="shipped")
    with pytest.raises(ValueError, match="unknown rule_variance"):
        pooled_delta("T3", srcs, axis, shrinkage=False, var_floor="poisson", variance_model=vm, rule_variance="both")


def test_category_fits_one_curve_per_group_and_its_shuffle_keeps_the_group_sizes(tmp_path):
    pb, _ = _fitted()
    gd = pb.dispersion_fit
    listed = [str(g) for g in gd.genes[::3]]
    lst = tmp_path / "ess.txt"
    lst.write_text("# the list\n" + "\n".join(listed) + "\n\nNOT_A_GENE\n")
    assert read_gene_list(lst) == set(listed) | {"NOT_A_GENE"}
    sv = VarianceModel.parse(f"category:list={lst}").for_source(pb, "control")
    assert int(sv.category.sum()) == len(listed)
    assert sv.record["genes_in_list"] == len(listed)
    n, m, v, s = _row(pb, "T3")
    out = sv.cell_variance(m, s, v, PC, perturbed=True)
    with np.errstate(divide="ignore"):
        th_in = np.interp(np.log(m / s), *sv.curve_in)
        th_out = np.interp(np.log(m / s), *sv.curve)
    want = (m + PC) * s + np.where(sv.category == 1, th_in, th_out) * m * m
    assert np.allclose(out, want)
    # the two curves differ (a list that is a third of the genes gives a different window median)
    assert not np.array_equal(sv.curve_in[1], np.interp(sv.curve_in[0], *sv.curve))
    shuf = VarianceModel.parse(f"category:list={lst},shuffle=5").for_source(pb, "control")
    assert int(shuf.category.sum()) == len(listed)
    assert not np.array_equal(shuf.category, sv.category)
    # the labels move within expression bins: every bin keeps its in-group count
    ok = gd.mean_count > 0
    for chunk in expression_bins(gd.mean_count, ok):
        assert int(shuf.category[chunk].sum()) == int(sv.category[chunk].sum())
    assert "within 100 quantile bins" in shuf.record["shuffle"]
    # a list covering every gene, or none, is no split
    (tmp_path / "all.txt").write_text("\n".join(str(g) for g in gd.genes))
    with pytest.raises(ValueError, match="both groups"):
        VarianceModel.parse(f"category:list={tmp_path / 'all.txt'}").for_source(pb, "control")


# ---------------------------------------------------------------- refusals and provenance


def test_a_dispersion_model_needs_the_poisson_floor_and_a_fit():
    pb, _ = _fitted()
    axis = np.asarray(pb.genes)
    vm = VarianceModel.parse("trend")
    with pytest.raises(ValueError, match="var_floor='poisson'"):
        pooled_delta("T3", [(pb, "control")], axis, shrinkage=False, variance_model=vm)
    bare, _ = synthetic(n_genes=120, n_groups=60, n_cells=40, seed=3)
    bare.labels[0] = "control"
    bare.sidechain_name = "bare"
    with pytest.raises(ValueError, match="needs the dispersion fit"):
        pooled_delta("T3", [(bare, "control")], axis, shrinkage=False, var_floor="poisson", variance_model=vm)
    with pytest.raises(SystemExit, match="carry no dispersion fit"):
        vm.check_sources([(bare, "control")])
    vm.check_sources([(pb, "control")])            # the fitted one passes


def test_a_fit_is_attached_only_to_the_source_it_was_made_from(tmp_path):
    pb, _ = synthetic(n_genes=120, n_groups=60, n_cells=40, seed=3)
    pb.labels[0] = "control"
    src = tmp_path / "arm.npz"
    pb.save(src)
    gd = fit_gene_dispersion_file(src)
    assert gd.source == "arm" and len(gd.fitted_on_sha256) == 64
    fit = tmp_path / "arm.fit.npz"
    gd.save(fit)
    loaded = GeneDispersion.load(fit)
    assert np.array_equal(loaded.theta_sql, gd.theta_sql) and loaded.fitted_on_sha256 == gd.fitted_on_sha256
    pb2 = PseudobulkSums.load(src)
    pb2.sidechain_name, pb2.sidechain_path = "arm", src
    out = apply_dispersion_fits([(pb2, "control")], parse_dispersion_fits([f"arm={fit}"]))
    assert out[0][0].dispersion_fit.n_groups_used == gd.n_groups_used
    assert len(out[0][0].dispersion_fit_sha256) == 64
    with pytest.raises(SystemExit, match="match no source"):
        apply_dispersion_fits([(pb2, "control")], {"elsewhere": fit})
    other, _ = synthetic(n_genes=50, n_groups=20, n_cells=40, seed=9)
    other.sidechain_name = "arm"
    with pytest.raises(SystemExit, match="gene axis"):
        apply_dispersion_fits([(other, "control")], {"arm": fit})
    # the same stem, another file of another corpus: the recorded sha256 catches it
    pb3, _ = synthetic(n_genes=120, n_groups=60, n_cells=40, seed=8)
    pb3.labels[0] = "control"
    pb3.sources = ["another/corpus.h5ad"]
    src3 = tmp_path / "arm_v2.npz"
    pb3.save(src3)
    pb3.sidechain_name, pb3.sidechain_path = "arm", src3
    with pytest.raises(SystemExit, match="refit it"):
        apply_dispersion_fits([(pb3, "control")], {"arm": fit})
    # a LABEL SUBSET of the fitted artifact (the same `sources`, fewer labels) takes the full fit:
    # that is how a fold's subset file carries the full arm's dispersion
    full = PseudobulkSums.load(src)
    sub = PseudobulkSums(labels=full.labels[:20], genes=full.genes, count_sum=full.count_sum[:20],
                         cpm_sum=full.cpm_sum[:20], cpm_sq_sum=full.cpm_sq_sum[:20], n_cells=full.n_cells[:20],
                         libsize_sum=full.libsize_sum[:20], sources=list(full.sources))   # as subset_pseudobulk_labels.py cuts it
    src_sub = tmp_path / "arm_sub.npz"
    sub.save(src_sub)
    sub.sidechain_name, sub.sidechain_path = "arm", src_sub
    out = apply_dispersion_fits([(sub, "control")], {"arm": fit})
    assert out[0][0].dispersion_fit.n_groups_used == gd.n_groups_used
    assert out[0][0].dispersion_fit_attached_as.startswith("a label subset (20 of")
    # but a fit on the SUBSET cannot be attached to the full arm (labels the fit never saw)
    small = fit_gene_dispersion_file(src_sub)
    small.save(tmp_path / "small.fit.npz")
    assert small.fitted_on_sources_sha256 == gd.fitted_on_sources_sha256
    with pytest.raises(SystemExit, match="refit it"):
        apply_dispersion_fits([(pb2, "control")], {"arm": tmp_path / "small.fit.npz"})
    # another accumulation over the SAME corpus files -- other labels (a construct-level file), or the same
    # labels with other cell counts (another QC) -- is refused even though the sources list matches
    other_labels = PseudobulkSums(labels=[f"{l}_P1-1" for l in full.labels[:20]], genes=full.genes,
                                  count_sum=full.count_sum[:20], cpm_sum=full.cpm_sum[:20], cpm_sq_sum=full.cpm_sq_sum[:20],
                                  n_cells=full.n_cells[:20], libsize_sum=full.libsize_sum[:20], sources=list(full.sources))
    other_labels.save(tmp_path / "guide.npz")
    other_labels.sidechain_name, other_labels.sidechain_path = "arm", tmp_path / "guide.npz"
    with pytest.raises(SystemExit, match="refit it"):
        apply_dispersion_fits([(other_labels, "control")], {"arm": fit})
    other_qc = PseudobulkSums(labels=full.labels[:20], genes=full.genes, count_sum=full.count_sum[:20],
                              cpm_sum=full.cpm_sum[:20], cpm_sq_sum=full.cpm_sq_sum[:20], n_cells=full.n_cells[:20] - 1,
                              libsize_sum=full.libsize_sum[:20], sources=list(full.sources))
    other_qc.save(tmp_path / "qc.npz")
    other_qc.sidechain_name, other_qc.sidechain_path = "arm", tmp_path / "qc.npz"
    with pytest.raises(SystemExit, match="refit it"):
        apply_dispersion_fits([(other_qc, "control")], {"arm": fit})
    # a fit with no provenance (made in memory) is refused on any source that has a path
    bare = fit_gene_dispersion(full)
    bare.save(tmp_path / "bare.fit.npz")
    with pytest.raises(SystemExit, match="no provenance"):
        apply_dispersion_fits([(pb2, "control")], {"arm": tmp_path / "bare.fit.npz"})
    with pytest.raises(SystemExit, match="not NAME=PATH"):
        parse_dispersion_fits(["arm"])


def test_the_streamed_fit_equals_the_in_memory_one_and_refuses_a_subset_product(tmp_path):
    pb, _ = synthetic(n_genes=200, n_groups=80, n_cells=30, seed=5)
    src = tmp_path / "arm.npz"
    pb.save(src)
    mc, th, df = moment_dispersion(pb)
    mc2, th2, df2, n_used, genes = moment_dispersion_file(src, block_rows=7)
    assert np.allclose(mc, mc2, rtol=1e-12) and np.allclose(th, th2, rtol=1e-9, atol=1e-12)
    assert df == df2 and n_used == 80 and np.array_equal(genes, pb.genes.astype(str))
    sub = PseudobulkSums.load_subset(src, pb.labels, pb.genes.tolist())
    sub.save(tmp_path / "subset.npz")
    with pytest.raises(ValueError, match="count_sum is all zero"):
        moment_dispersion_file(tmp_path / "subset.npz")


def test_the_cli_check_refuses_inert_and_underspecified_settings(tmp_path):
    import argparse

    class _AP(argparse.ArgumentParser):
        def error(self, message):
            raise SystemExit(message)

    ap = _AP()
    mk = lambda **kw: argparse.Namespace(**{"variance_model": "shipped", "rule_variance": "model",
                                             "dispersion_fit": [], "var_floor": "poisson", **kw})
    assert check_variance_args(ap, mk()).is_shipped
    with pytest.raises(SystemExit, match="rule-variance"):
        check_variance_args(ap, mk(rule_variance="shipped"))
    with pytest.raises(SystemExit, match="never be read"):
        check_variance_args(ap, mk(dispersion_fit=["a=b"]))
    with pytest.raises(SystemExit, match="var-floor poisson"):
        check_variance_args(ap, mk(variance_model="trend", dispersion_fit=["a=b"], var_floor="none"))
    with pytest.raises(SystemExit, match="dispersion-fit NAME=PATH"):
        check_variance_args(ap, mk(variance_model="trend"))
    with pytest.raises(SystemExit, match="never be read"):
        check_variance_args(ap, mk(variance_model="multiplier:a=2", dispersion_fit=["a=b"]))
    with pytest.raises(SystemExit, match="does not exist"):
        check_variance_args(ap, mk(variance_model=f"category:list={tmp_path / 'none.txt'}", dispersion_fit=["a=b"]))
    vm = check_variance_args(ap, mk(variance_model="multiplier:h1_pseudobulk=0.95/1.1/1.26"))
    assert vm.kind == "multiplier"


def test_loco_threads_the_knob_through_and_records_it(monkeypatch, tmp_path):
    """The CLI contract, heavy stages stubbed: the model and the rule setting reach
    build_transfer_prediction and the log_run payload, with the fit specs."""
    from sidechain.eval import loco
    from tests.test_pooled_delta_sources import _StubPB

    monkeypatch.setattr(PseudobulkSums, "load", classmethod(lambda cls, p: _StubPB(f"PB:{p}")))
    monkeypatch.setattr(loco, "apply_dispersion_fits", lambda sources, fits: sources)
    captured, logged = {}, {}

    def fake_build(real, sources, out_path, **kw):
        captured.update(kw)
        return {}

    monkeypatch.setattr(loco, "build_transfer_prediction", fake_build)
    monkeypatch.setattr(loco, "attach_controls", lambda pred, real, out, **kw: out)
    monkeypatch.setattr(loco, "score", lambda *a, **kw: {"overall": 0.0, "members": {}})
    monkeypatch.setattr(loco, "log_run", lambda params, results, artifacts=None: logged.update(params))
    rc = loco.main(["--real", "r.h5ad", "--bundle", "b", "--out", str(tmp_path / "arm"),
                    "--source", "plain.npz:ctl", "--var-floor", "poisson",
                    "--variance-model", "multiplier:plain=1.1/1.2/1.3", "--rule-variance", "shipped"])
    assert rc == 0
    assert captured["variance_model"].spec() == "multiplier:plain=1.1/1.2/1.3"
    assert captured["rule_variance"] == "shipped"
    assert logged["variance_model"] == "multiplier:plain=1.1/1.2/1.3" and logged["rule_variance"] == "shipped"
    # the default records `shipped`, and the shipped path refuses an inert --rule-variance
    rc = loco.main(["--real", "r.h5ad", "--bundle", "b", "--out", str(tmp_path / "arm2"),
                    "--source", "plain.npz:ctl"])
    assert rc == 0 and logged["variance_model"] == "shipped" and captured["variance_model"].is_shipped
    with pytest.raises(SystemExit):
        loco.main(["--real", "r.h5ad", "--bundle", "b", "--out", str(tmp_path / "arm3"),
                   "--source", "plain.npz:ctl", "--rule-variance", "shipped"])
    with pytest.raises(SystemExit):     # a dispersion model without --var-floor poisson
        loco.main(["--real", "r.h5ad", "--bundle", "b", "--out", str(tmp_path / "arm4"),
                   "--source", "plain.npz:ctl", "--variance-model", "trend", "--dispersion-fit", "plain=x"])


def test_the_run_record_names_the_fit_the_factors_and_the_list(tmp_path):
    pb, _ = _fitted()
    pb.dispersion_fit_path, pb.dispersion_fit_sha256 = "fits/arm.fit.npz", "ab" * 32
    rec = VarianceModel.parse("trend:shuffle=2").record([(pb, "control")])
    assert rec["spec"] == "trend:shuffle=2" and rec["shuffle_seed"] == 2 and rec["shuffle_bins"] == SHUFFLE_BINS
    assert rec["sources"]["arm"]["dispersion_fit_sha256"] == "ab" * 32
    rec = VarianceModel.parse("multiplier:arm=1/2/3").record([(pb, "control")])
    assert rec["sources"]["arm"]["factors"] == [1.0, 2.0, 3.0]
    lst = tmp_path / "ess.txt"
    lst.write_text("\n".join(str(g) for g in pb.dispersion_fit.genes[::3]))
    vm = VarianceModel.parse(f"category:list={lst}")
    vm.for_source(pb, "control")                   # built once, so the record carries the split's counts
    rec = vm.record([(pb, "control")])
    assert rec["category_list_sha256"] and rec["category_list_genes"] == len(pb.dispersion_fit.genes[::3])
    assert rec["sources"]["arm"]["built"]["genes_in_list"] == len(pb.dispersion_fit.genes[::3])
    before = vm.identity()
    assert before == f"{vm.spec()}#list_sha256={rec['category_list_sha256']}"
    lst.write_text("ONLY_ONE\n")                   # the same path with other contents is another model
    assert VarianceModel.parse(f"category:list={lst}").identity() != before
    assert VarianceModel.parse("trend").identity() == "trend"
    json.dumps(rec)                                 # JSON-able, as a run record must be
