"""Contract tests for `sidechain.data.dispersion` -- the numpy port of glmGamPoi's step 8.

The estimator is checked against **synthetic Gamma-Poisson counts with a known theta(mu)
curve**, because that is the only setting where "did it recover the truth" is answerable. The
generator here is the model's own definition -- a Poisson whose rate is a Gamma draw -- so a
failure is the estimator's, never the simulator's.
"""
from __future__ import annotations

import numpy as np
import pytest

from sidechain.data.dispersion import (
    MIN_DISPERSION,
    dispersion_trend,
    fit_gene_dispersion,
    fit_variance_prior,
    moment_dispersion,
    quasi_dispersion,
    shrink_quasi_dispersion,
)
from sidechain.data.stream_pseudobulk import PseudobulkSums

LIBSIZE = 20_000.0


def synthetic(n_genes=300, n_groups=200, n_cells=45, seed=0, libsize=LIBSIZE,
              theta_curve=lambda base: 0.05 + 2.0 / (1.0 + base), population_form=False):
    """A pseudobulk artifact whose true per-gene dispersion is known.

    `population_form=True` builds the second moment so that the *population* variance is
    recovered exactly -- used only by the Bessel test, which needs the biased input on purpose.
    """
    rng = np.random.default_rng(seed)
    base_cpm = np.exp(rng.normal(np.log(30.0), 1.2, size=n_genes))
    theta_true = theta_curve(base_cpm)
    mu = base_cpm * libsize / 1e6                      # mean RAW count per cell
    cpm_sum = np.zeros((n_groups, n_genes))
    cpm_sq_sum = np.zeros((n_groups, n_genes))
    count_sum = np.zeros((n_groups, n_genes))
    for g in range(n_groups):
        lam = rng.gamma(1.0 / theta_true, theta_true * mu, size=(n_cells, n_genes))
        counts = rng.poisson(lam)
        cpm = counts * 1e6 / libsize
        cpm_sum[g] = cpm.sum(axis=0)
        cpm_sq_sum[g] = (cpm ** 2).sum(axis=0)
        count_sum[g] = counts.sum(axis=0)
    pb = PseudobulkSums(
        labels=[f"T{i}" for i in range(n_groups)],
        genes=np.array([f"G{i}" for i in range(n_genes)]),
        count_sum=count_sum, cpm_sum=cpm_sum, cpm_sq_sum=cpm_sq_sum,
        n_cells=np.full(n_groups, n_cells, dtype=np.int64),
        libsize_sum=np.full(n_groups, n_cells * libsize, dtype=np.float64),
    )
    return pb, theta_true


def test_moment_dispersion_recovers_a_known_dispersion():
    pb, theta_true = synthetic()
    _, theta_ml, _ = moment_dispersion(pb)
    ratio = theta_ml / theta_true
    assert 0.90 < float(np.median(ratio)) < 1.10, float(np.median(ratio))
    # It also has to ORDER the genes, because that is all the trend reads. Ranks, not values:
    # a handful of genes clamp to exactly zero (a negative slope is undersampling, not negative
    # dispersion), and on a log scale those few dominate any correlation of the values. Measured
    # here at 200 groups of 45 cells: Spearman 0.73, and our thinnest real arm has 2,393 groups.
    ranks = lambda a: np.argsort(np.argsort(a))
    assert np.corrcoef(ranks(theta_ml), ranks(theta_true))[0, 1] > 0.6


def test_the_bessel_correction_is_what_makes_it_unbiased():
    """The reason `moment_dispersion` rescales the cached variance, held as a test.

    `PseudobulkSums.var_cpm` is the population form. Its expectation is (n-1)/n of the truth,
    which is a 2% error on the variance and a ~35% error on the *excess over Poisson* that the
    dispersion is estimated from -- because the subtraction cancels the large term and leaves
    the small bias standing. If someone ever "simplifies" this back to `var_cpm()`, this fails.
    """
    pb, theta_true = synthetic()
    _, theta_corrected, _ = moment_dispersion(pb)

    n = pb.n_cells.astype(np.float64)
    m = pb.cpm_sum / n[:, None]
    v_pop = np.maximum(pb.cpm_sq_sum / n[:, None] - m * m, 0.0)      # NO Bessel
    scale = (1e6 / (pb.libsize_sum / n))[:, None]
    w = (n - 1.0)[:, None]
    theta_uncorrected = np.maximum(
        (w * (v_pop - m * scale)).sum(axis=0) / (w * m * m).sum(axis=0), 0.0)

    corrected = float(np.median(theta_corrected / theta_true))
    uncorrected = float(np.median(theta_uncorrected / theta_true))
    assert uncorrected < 0.80, uncorrected          # measured ~0.66 at 45 cells
    assert abs(corrected - 1.0) < abs(uncorrected - 1.0)


def test_moment_dispersion_needs_more_than_one_usable_group():
    pb, _ = synthetic(n_groups=3)
    pb.n_cells[:] = 1                                # no group has any observable spread
    with pytest.raises(ValueError, match="at least 2 groups"):
        moment_dispersion(pb)


def test_dispersion_trend_beats_the_raw_estimate_at_recovering_the_curve():
    """The whole claim of step 8: borrowing across genes of similar expression helps."""
    pb, theta_true = synthetic(n_genes=600, n_groups=40)   # few groups => noisy per-gene theta
    mean_count, theta_ml, _ = moment_dispersion(pb)
    trend = dispersion_trend(mean_count, theta_ml, window=101)
    err_raw = np.abs(np.log(np.maximum(theta_ml, 1e-6)) - np.log(theta_true))
    err_trend = np.abs(np.log(np.maximum(trend, 1e-6)) - np.log(theta_true))
    assert float(np.median(err_trend)) < float(np.median(err_raw))


def test_dispersion_trend_is_a_median_so_one_wild_gene_cannot_move_it():
    mean_count = np.linspace(1.0, 100.0, 201)
    theta_ml = np.full(201, 0.5)
    clean = dispersion_trend(mean_count, theta_ml, window=51)
    theta_ml[100] = 1e6
    spiked = dispersion_trend(mean_count, theta_ml, window=51)
    assert np.allclose(clean, spiked)


def test_dispersion_trend_never_returns_zero():
    """A zero trend would divide by zero in `quasi_dispersion`, so it is clamped, not allowed."""
    trend = dispersion_trend(np.array([0.0, 0.0, 5.0, 9.0]), np.zeros(4), window=3)
    assert np.all(trend >= MIN_DISPERSION)


def test_unexpressed_genes_inherit_the_low_end_rather_than_a_zero():
    mean_count = np.array([0.0, 0.0, 1.0, 5.0, 50.0])
    theta_ml = np.array([0.0, 0.0, 0.9, 0.5, 0.1])
    trend = dispersion_trend(mean_count, theta_ml, window=1)
    assert trend[0] == trend[1] == trend[2]          # the lowest EXPRESSED gene's value
    assert trend[2] == pytest.approx(0.9)


def test_quasi_dispersion_is_one_exactly_on_the_trend():
    mu = np.array([0.5, 5.0, 50.0])
    theta = np.array([0.4, 0.2, 0.05])
    assert np.allclose(quasi_dispersion(mu, theta, theta), 1.0)


def test_quasi_dispersion_orders_genes_by_departure_from_the_trend():
    mu = np.full(3, 10.0)
    trend = np.full(3, 0.2)
    ql = quasi_dispersion(mu, np.array([0.1, 0.2, 0.4]), trend)
    assert ql[0] < 1.0 < ql[2]
    assert ql[1] == pytest.approx(1.0)


def test_shrink_is_the_identity_when_the_prior_has_no_strength():
    ql = np.array([0.5, 1.0, 2.0])
    assert np.allclose(shrink_quasi_dispersion(ql, df=10.0, df0=0.0, tau0_sq=1.0), ql)


def test_shrink_moves_every_gene_toward_the_prior_and_never_past_it():
    ql = np.array([0.25, 1.0, 4.0])
    out = shrink_quasi_dispersion(ql, df=4.0, df0=4.0, tau0_sq=1.0)
    assert np.all(np.abs(out - 1.0) < np.abs(ql - 1.0) + 1e-12)
    assert out[0] < 1.0 < out[2]                      # nothing crosses the prior
    assert np.allclose(out, (4.0 * 1.0 + 4.0 * ql) / 8.0)


def test_a_stronger_prior_shrinks_harder():
    ql = np.array([4.0])
    weak = shrink_quasi_dispersion(ql, df=10.0, df0=1.0, tau0_sq=1.0)[0]
    strong = shrink_quasi_dispersion(ql, df=10.0, df0=100.0, tau0_sq=1.0)[0]
    assert abs(strong - 1.0) < abs(weak - 1.0)


def test_fit_variance_prior_returns_no_prior_when_there_is_nothing_to_fit():
    assert fit_variance_prior(np.array([1.0]), df=10.0) == (0.0, 1.0)
    assert fit_variance_prior(np.array([np.nan, -1.0, 0.0]), df=10.0) == (0.0, 1.0)
    assert fit_variance_prior(np.array([1.0, 1.1, 0.9]), df=0.0) == (0.0, 1.0)


def test_fit_variance_prior_finds_a_scale_near_the_centre_of_the_data():
    rng = np.random.default_rng(3)
    df = 20.0
    ql = rng.chisquare(df, size=2000) / df * 1.7      # centred on 1.7, df degrees of freedom
    df0, tau0_sq = fit_variance_prior(ql, df=df)
    assert df0 > 0.0
    assert 1.2 < tau0_sq < 2.4, tau0_sq


def test_variance_function_is_poisson_when_the_dispersion_is_zero():
    pb, _ = synthetic(n_genes=50, n_groups=20)
    gd = fit_gene_dispersion(pb, window=11)
    object.__setattr__(gd, "theta_trend", np.zeros_like(gd.theta_trend))
    object.__setattr__(gd, "theta_sql", np.ones_like(gd.theta_sql))
    mu = np.full(gd.mean_count.shape, 3.0)
    assert np.allclose(gd.variance_counts(mu), 3.0)


def test_variance_cpm_is_variance_counts_rescaled():
    pb, _ = synthetic(n_genes=40, n_groups=20)
    gd = fit_gene_dispersion(pb, window=11)
    scale = 1e6 / LIBSIZE
    mu = np.full(gd.mean_count.shape, 2.0)
    assert np.allclose(gd.variance_cpm(mu * scale, LIBSIZE),
                       gd.variance_counts(mu) * scale * scale)


def test_fit_gene_dispersion_carries_every_stage_and_the_prior_it_fitted():
    pb, theta_true = synthetic(n_genes=200, n_groups=60)
    gd = fit_gene_dispersion(pb, window=51)
    assert gd.genes.shape == theta_true.shape
    assert gd.n_groups_used == 60
    assert gd.df == pytest.approx(60 * 44.0)
    for name in ("theta_ml", "theta_trend", "theta_ql", "theta_sql"):
        arr = getattr(gd, name)
        assert arr.shape == theta_true.shape and np.all(np.isfinite(arr)), name
    assert np.all(gd.theta_trend > 0.0)


def test_excluding_a_label_drops_it_from_the_fit():
    pb, _ = synthetic(n_genes=40, n_groups=12)
    full = fit_gene_dispersion(pb, window=11)
    without = fit_gene_dispersion(pb, exclude=("T0",), window=11)
    assert without.n_groups_used == full.n_groups_used - 1
    assert not np.allclose(without.theta_ml, full.theta_ml)


def test_df0_can_be_forced_because_the_fitted_one_is_inert_on_our_arms():
    """Saber, 2026-09-18: can the shrinkage strength be dialled? Not in glmGamPoi; here, yes.

    The point is falsifiability. On every real arm the fitted `df0` is three orders of
    magnitude under the residual `df`, so `theta_sql` equals `theta_ql` and the empirical-Bayes
    step is inert. Forcing `df0` is what turns that from an observation into a claim someone
    can break.
    """
    pb, _ = synthetic(n_genes=200, n_groups=60)
    fitted = fit_gene_dispersion(pb, window=51)

    weak = fit_gene_dispersion(pb, window=51, df0=fitted.df / 100.0)
    strong = fit_gene_dispersion(pb, window=51, df0=fitted.df * 100.0)
    moved = lambda g: float(np.abs(g.theta_sql - g.theta_ql).max())
    assert np.allclose(fitted.theta_ql, weak.theta_ql)            # only the shrinkage differs
    assert moved(weak) < moved(strong)                            # the knob is monotone
    # df0 = 0 is the far end: no prior, so shrinkage is the identity.
    assert np.allclose(fit_gene_dispersion(pb, window=51, df0=0.0).theta_sql, fitted.theta_ql)

    # NOTE the synthetic fit goes the OTHER way from the real arms, and that is the estimator
    # being right in both cases. Here every gene is drawn from one exact theta(mu) curve, so
    # they agree with their trend and the prior is correctly strong -- fitted df0 lands ~11x
    # ABOVE df. Real arms carry genuine gene-to-gene dispersion heterogeneity, so the same fit
    # puts df0 three orders of magnitude BELOW df and steps aside. df0 is a measurement of how
    # well the genes agree with their own trend, not a constant of the method.
    assert fitted.df0 > fitted.df


def test_a_negative_prior_strength_is_refused_not_clamped():
    pb, _ = synthetic(n_genes=40, n_groups=20)
    with pytest.raises(ValueError, match="df0 must be non-negative"):
        fit_gene_dispersion(pb, window=11, df0=-1.0)
