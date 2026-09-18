"""Trended, empirically shrunk Gamma-Poisson dispersion for a pseudobulk artifact.

A numpy port of **step 8 of glmGamPoi's algorithm** (Ahlmann-Eltze & Huber 2021,
*Bioinformatics* 36(24):5701-5702, doi 10.1093/bioinformatics/btaa1009, Supplement S1) --
written from the paper's equations, which are CC-BY, and not from the R package, which is
GPL-3. No R, no rpy2, no new dependency beyond scipy, which is already installed.

**What problem this solves.** `submit.build._log2fc_with_var` estimates each gene's variance
from one target's cells alone -- 45 of them on HepG2 at the median. A variance estimated from
45 cells is a noisy denominator, and it is used as a pooling weight, so the noise propagates
into the pooled delta. `shrink()`'s own docstring records the obvious fix failing: *"A single
global normal prior was tried first and erased the real effects too ... any one-variance prior
is dominated by the nulls."* glmGamPoi's answer is that the prior must be a **curve over mean
expression**, not one number, and that its strength must be fitted rather than chosen -- which
is exactly the shape a one-variance prior lacks.

**The four equations, verbatim from the supplement**, and what this module calls each:

    sigma^2 = theta_QL * (mu + theta_trend * mu^2)          -> `variance_counts`
    theta_QL  = (1 + mu*theta_ML) / (1 + mu*theta_trend)    -> `quasi_dispersion`
    theta_SQL = (df0*tau0^2 + df*theta_QL) / (df0 + df)     -> `shrink_quasi_dispersion`
    (df0, tau0^2) = ML fit of an inverse-chi-squared prior  -> `fit_variance_prior`

**Two departures from the paper, both forced by what our caches hold, both measured.**

1. **The dispersion is a method-of-moments estimate (the supplement's step 2), never the
   maximum-likelihood one (its step 7).** A `PseudobulkSums` stores per-group *sums* --
   `cpm_sum`, `cpm_sq_sum`, `count_sum`, `n_cells`, `libsize_sum` -- and a Gamma-Poisson
   likelihood needs the individual counts, which are gone. So the ML fit, its `nlminb` call and
   the frequency-table speedup that the paper's whole abstract is about are **not computable
   here at all**, and pretending otherwise would be the wrong kind of faithful. Getting step 7
   would mean a re-stream over cells, not a change to this file. What survives the sums is the
   mean-variance relation `sigma^2 = mu + theta*mu^2`, which is all step 2 uses, and step 8,
   which reads only a per-gene dispersion and a per-gene mean.
2. **The trend is a sliding-window median over genes ordered by mean expression**, not R's
   `locfit`-based local median regression. Same intent -- a robust local centre -- and the
   window is the one knob. `dispersion_trend` takes it explicitly rather than guessing.

**This module estimates; it does not yet vote.** Nothing in `submit.build` calls it. Wiring it
into the pooling weights is an A/B over the fold rotation with a shuffled control, and that is
a separate, pre-registered experiment (`research/ideas/gamma-poisson-delta-variance.md`).

**Why the per-gene estimate is worth having even though it is only step 2:** it pools across
every group in the arm. `_log2fc_with_var` reads one target's cells; `moment_dispersion` reads
all 2,393 targets' groups at once to estimate one dispersion per gene, so its input is three
orders of magnitude larger for the same cached bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import f as f_dist

from sidechain.data.stream_pseudobulk import PseudobulkSums

# A dispersion is non-negative; exactly zero is the Poisson case and breaks the ratios in
# `quasi_dispersion`, so the trend is clamped here rather than at each call site.
MIN_DISPERSION = 1e-8
# Groups thinner than this contribute no usable variance estimate (n = 1 has none at all).
MIN_CELLS_PER_GROUP = 2


@dataclass(frozen=True)
class GeneDispersion:
    """Per-gene dispersion for one arm, at the three stages the paper distinguishes.

    `theta_ml` is raw and noisy, `theta_trend` is what genes of that expression level look
    like, and `theta_sql` is the shrunken quasi-dispersion -- a dimensionless multiplier
    centred near 1 that says how far this gene departs from its own trend. Keep all three:
    the point of the method is the distance between them, and collapsing to one number is
    how a reader loses the ability to tell a real outlier from a noisy one.
    """

    genes: np.ndarray        # (G,) gene names, the arm's own axis
    mean_count: np.ndarray   # (G,) cell-weighted mean RAW count per cell
    theta_ml: np.ndarray     # (G,) method-of-moments dispersion (supplement step 2)
    theta_trend: np.ndarray  # (G,) the local median curve through theta_ml
    theta_ql: np.ndarray     # (G,) theta_ML expressed relative to the trend
    theta_sql: np.ndarray    # (G,) theta_QL shrunk toward the fitted prior
    df: float                # residual degrees of freedom behind each theta_ML
    df0: float               # the prior's degrees of freedom -- its strength, fitted
    tau0_sq: float           # the prior's scale, fitted
    n_groups_used: int

    def variance_counts(self, mean_count: np.ndarray) -> np.ndarray:
        """`sigma^2 = theta_QL * (mu + theta_trend * mu^2)` -- the paper's variance function.

        `mean_count` is a mean RAW count per cell, on this arm's gene axis; the shrunken
        `theta_sql` plays the paper's `theta_QL`, which is the whole point of shrinking it.
        """
        mu = np.asarray(mean_count, dtype=np.float64)
        return self.theta_sql * (mu + self.theta_trend * mu * mu)

    def variance_cpm(self, mean_cpm: np.ndarray, mean_libsize: float) -> np.ndarray:
        """The same variance in the CPM units `_log2fc_with_var` works in.

        One count is `1e6 / mean_libsize` CPM, and a variance scales by the square of that,
        so the Poisson term becomes `mean_cpm * 1e6 / mean_libsize` -- which is precisely the
        expression `var_floor="poisson"` already floors at, with the pseudocount dropped.
        The dispersion term is scale-free by construction and carries over unchanged.
        """
        m = np.asarray(mean_cpm, dtype=np.float64)
        scale = 1e6 / float(mean_libsize)
        return self.theta_sql * (m * scale + self.theta_trend * m * m)


def moment_dispersion(pb: PseudobulkSums, *, exclude: tuple[str, ...] = (),
                      min_cells: int = MIN_CELLS_PER_GROUP
                      ) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-gene dispersion from the mean-variance relation, pooled over every group.

    The supplement's step 2: `sigma^2 = mu + theta*mu^2`, so the part of a group's observed
    variance that Poisson sampling does not explain should grow as `theta * mu^2`. Fitting
    one slope per gene through every group's `(mu^2, excess variance)` pair gives `theta`.

    Done in CPM space because that is what the cache stores, then converted: a group's CPM
    variance is `mu*s + theta*(mu*s)^2` where `s = 1e6/mean_libsize`, so dividing the excess
    by `(mu*s)^2` returns the same scale-free `theta` either way.

    Weighted by `n - 1`, each group's degrees of freedom. Optimal weights would also divide
    by the square of the group's own variance, which needs `theta` to compute -- the paper
    reaches its estimate by iterating to ML instead, and we cannot (see the module docstring),
    so this stays the one-pass rough estimate it is honestly named after.

    Returns `(mean_count, theta_ml, df)`. `df` is the pooled residual degrees of freedom,
    `sum(n_l - 1)` over the groups used, which `fit_variance_prior` needs.
    """
    keep = [i for i, lab in enumerate(pb.labels)
            if lab not in exclude and int(pb.n_cells[i]) >= min_cells]
    if len(keep) < 2:
        raise ValueError(
            f"need at least 2 groups with >= {min_cells} cells to fit a dispersion, "
            f"got {len(keep)} -- a single group cannot separate sampling noise from dispersion")
    rows = np.asarray(keep)
    n = pb.n_cells[rows].astype(np.float64)
    m_cpm = pb.cpm_sum[rows] / n[:, None]
    # BESSEL, and it is not cosmetic here. `PseudobulkSums.var_cpm` is the POPULATION form
    # (divides by n), whose expectation is (n-1)/n of the true variance. Subtracting the
    # Poisson term then leaves a bias of -mu/n, which is small against the variance itself
    # and large against the *excess* this estimator is built from: measured on synthetic
    # Gamma-Poisson at 45 cells per group, the population form recovers 0.66 of the true
    # dispersion and the sample form recovers 0.98. Same class of correction as scPerturb's
    # N(N-1) on the E-distance, and the same reason -- the statistic is a difference.
    v_cpm = np.maximum(pb.cpm_sq_sum[rows] / n[:, None] - m_cpm * m_cpm, 0.0)
    v_cpm = v_cpm * (n / np.maximum(n - 1.0, 1.0))[:, None]
    scale = (1e6 / (pb.libsize_sum[rows] / n))[:, None]      # CPM per raw count, per group

    # Var_cpm = mu*scale^2 + theta*(mu*scale)^2, and mu*scale is m_cpm, so the part Poisson
    # does not explain is exactly theta * m_cpm^2. One ratio per gene, pooled over groups and
    # weighted by each group's degrees of freedom.
    excess = v_cpm - m_cpm * scale
    w = np.maximum(n - 1.0, 0.0)[:, None]
    num = (w * excess).sum(axis=0)
    den = (w * m_cpm * m_cpm).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.where(den > 0, num / den, 0.0)
    # A negative slope means the group variances came in under Poisson -- undersampling, not
    # negative dispersion, which does not exist. Clamped to zero, exactly as the supplement
    # clamps its own ML fit when the derivative says there is no interior maximum.
    theta = np.maximum(np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    mean_count = (pb.count_sum[rows].sum(axis=0) / n.sum())
    df = float(np.maximum(n - 1.0, 0.0).sum())
    return mean_count, theta, df


def dispersion_trend(mean_count: np.ndarray, theta_ml: np.ndarray, *,
                     window: int = 301, min_expressed: float = 0.0) -> np.ndarray:
    """A sliding-window median of `theta_ml` over genes ordered by mean expression.

    The supplement fits *"a local median regression through the mean gene expression values
    and maximum likelihood overdispersion estimates"*. This is that, with a rectangular
    window instead of `locfit`'s kernel: sort by mean count, take the median of each gene's
    `window` nearest neighbours in that order, and hold the edges flat.

    Median, not mean, and that is the load-bearing choice -- the low-expression end is where
    half the individual estimates are noise, and a mean there would track the noise.

    Genes at or below `min_expressed` are given the trend of the lowest expressed gene above
    it rather than their own: a gene nobody detected has no information about dispersion, and
    letting it into the window would drag the curve down at exactly the end that is already
    hardest to estimate.
    """
    mean_count = np.asarray(mean_count, dtype=np.float64)
    theta_ml = np.asarray(theta_ml, dtype=np.float64)
    if mean_count.shape != theta_ml.shape:
        raise ValueError(f"shape mismatch: {mean_count.shape} vs {theta_ml.shape}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    out = np.full(theta_ml.shape, MIN_DISPERSION, dtype=np.float64)
    usable = np.flatnonzero(mean_count > min_expressed)
    if usable.size == 0:
        return out
    order = usable[np.argsort(mean_count[usable], kind="stable")]
    vals = theta_ml[order]
    k = int(min(window, vals.size))
    half = k // 2
    # Cumulative-median by explicit windows: G is ~9k-18k here, so an O(G*k) pass with
    # np.median over slices is milliseconds and needs no rolling-median machinery.
    smoothed = np.empty(vals.size, dtype=np.float64)
    for i in range(vals.size):
        lo = max(0, min(i - half, vals.size - k))
        smoothed[i] = np.median(vals[lo:lo + k])
    out[order] = np.maximum(smoothed, MIN_DISPERSION)
    # Genes with no expression inherit the curve's low-expression end rather than a zero.
    out[mean_count <= min_expressed] = out[order[0]]
    return out


def quasi_dispersion(mean_count: np.ndarray, theta_ml: np.ndarray,
                     theta_trend: np.ndarray) -> np.ndarray:
    """`theta_QL = (1 + mu*theta_ML) / (1 + mu*theta_trend)` -- the supplement, verbatim.

    It re-expresses a gene's own dispersion as a ratio against what its expression level
    predicts, which is what makes the genes comparable enough to share one prior. A gene
    exactly on the trend gets 1.0.
    """
    mu = np.asarray(mean_count, dtype=np.float64)
    num = 1.0 + mu * np.asarray(theta_ml, dtype=np.float64)
    den = 1.0 + mu * np.maximum(np.asarray(theta_trend, dtype=np.float64), MIN_DISPERSION)
    return num / np.maximum(den, 1e-300)


def fit_variance_prior(theta_ql: np.ndarray, df: float) -> tuple[float, float]:
    """ML fit of the inverse-chi-squared prior's `(df0, tau0^2)`, as the supplement specifies.

    If each gene's `theta_QL` has `df` degrees of freedom and the true per-gene value is drawn
    from a scaled inverse-chi-squared with `df0` degrees of freedom and scale `tau0^2`, then
    `theta_QL / tau0^2` is F-distributed with `df` and `df0` degrees of freedom. So the fit is
    a two-parameter maximum likelihood on that F density -- which is what the supplement means
    by *"using a maximum likelihood procedure"*, stated there without the density.

    `df0` is the answer that matters: it is the prior's strength in the same units as the
    data's own, so `df0 >> df` means the genes agree with their trend and the prior should
    dominate, and `df0 -> 0` means it should step aside. Fitted, never chosen.

    Optimised over logs so both parameters stay positive without a constrained solver.
    Returns `(0.0, 1.0)` -- a prior with no strength, so `shrink_quasi_dispersion` becomes the
    identity -- when there is nothing to fit, rather than raising into a caller's pipeline.
    """
    x = np.asarray(theta_ql, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 2 or df <= 0:
        return 0.0, 1.0

    def nll(p):
        df0, tau0_sq = np.exp(p)
        if not np.isfinite(df0) or not np.isfinite(tau0_sq) or df0 <= 0 or tau0_sq <= 0:
            return np.inf
        # density of x = tau0_sq * F(df, df0), so the Jacobian contributes -log(tau0_sq)
        ll = f_dist.logpdf(x / tau0_sq, df, df0) - np.log(tau0_sq)
        return -float(np.sum(ll)) if np.all(np.isfinite(ll)) else np.inf

    start = np.log([max(df, 1.0), max(float(np.median(x)), 1e-6)])
    res = minimize(nll, start, method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
    if not np.isfinite(res.fun):
        return 0.0, 1.0
    df0, tau0_sq = (float(v) for v in np.exp(res.x))
    return df0, tau0_sq


def shrink_quasi_dispersion(theta_ql: np.ndarray, df: float, df0: float,
                            tau0_sq: float) -> np.ndarray:
    """`theta_SQL = (df0*tau0^2 + df*theta_QL) / (df0 + df)` -- the supplement, verbatim.

    A weighted mean of the gene's own estimate and the prior, weighted by their degrees of
    freedom. `df0 = 0` returns `theta_QL` unchanged, which is the honest behaviour when the
    prior fit found nothing to borrow.
    """
    total = float(df0) + float(df)
    if total <= 0:
        return np.asarray(theta_ql, dtype=np.float64).copy()
    return (df0 * tau0_sq + df * np.asarray(theta_ql, dtype=np.float64)) / total


def fit_gene_dispersion(pb: PseudobulkSums, *, exclude: tuple[str, ...] = (),
                        window: int = 301, min_cells: int = MIN_CELLS_PER_GROUP,
                        df0: float | None = None) -> GeneDispersion:
    """The whole of step 8 on one arm: moments, trend, quasi-dispersion, prior, shrinkage.

    `exclude` drops label rows before fitting -- pass the control label when the dispersion
    should describe the perturbed groups only. Left empty by default, because the control is
    the deepest group in every arm we hold and it carries real information about the trend.

    **`df0` forces the prior's strength instead of fitting it, and glmGamPoi has no such
    knob** -- its `overdispersion_shrinkage` is TRUE/FALSE, and the `ridge_penalty` argument
    people reach for regularizes the COEFFICIENTS, not the dispersion (checked against
    `glm_gp`'s signature, 2026-09-18). It exists here for one reason: measured on our arms the
    fitted `df0` lands between 59 and 133 against a residual `df` of 125,183 to 414,393, so the
    shrinkage does nothing, and a claim like that should be falsifiable rather than merely
    observed. Pass `df0=df` to weight the prior equally with the data and see what moves.

    Forcing it is no longer empirical Bayes -- it is a hand-set shrinkage wearing the same
    formula -- so anything found that way is a tuned knob and owes a letter at ADR 0005, not a
    citation to the paper.
    """
    mean_count, theta_ml, df = moment_dispersion(pb, exclude=exclude, min_cells=min_cells)
    theta_trend = dispersion_trend(mean_count, theta_ml, window=window)
    theta_ql = quasi_dispersion(mean_count, theta_ml, theta_trend)
    fitted_df0, tau0_sq = fit_variance_prior(theta_ql, df)
    if df0 is None:
        df0 = fitted_df0
    elif df0 < 0:
        raise ValueError(f"df0 must be non-negative, got {df0}")
    theta_sql = shrink_quasi_dispersion(theta_ql, df, df0, tau0_sq)
    n_used = int(sum(1 for i, lab in enumerate(pb.labels)
                     if lab not in exclude and int(pb.n_cells[i]) >= min_cells))
    return GeneDispersion(genes=np.asarray(pb.genes), mean_count=mean_count,
                          theta_ml=theta_ml, theta_trend=theta_trend, theta_ql=theta_ql,
                          theta_sql=theta_sql, df=df, df0=df0, tau0_sq=tau0_sq,
                          n_groups_used=n_used)
