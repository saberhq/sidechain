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
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import f as f_dist

from sidechain.data.stream_pseudobulk import PseudobulkSums, iter_npz_row_blocks

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
    # Provenance, filled by `fit_gene_dispersion_file` and carried through `save`/`load`: the
    # source artifact's stem and its sha256, so a fit handed to the pooling knob
    # (`submit.variance_model`) can be checked against the source it is attached to rather
    # than trusted by file name. Empty for a fit made from an in-memory object.
    source: str = ""
    fitted_on_sha256: str = ""
    # sha256 of the artifact's `sources` member (the corpus files it was accumulated from): a label
    # subset of the same artifact (`scripts/subset_pseudobulk_labels.py`) carries the same list, so a
    # full-arm fit can be attached to the subset a fold pools from, while a fit from another corpus
    # or another gene axis cannot.
    fitted_on_sources_sha256: str = ""
    # every label of the fitted artifact with its cell count: a label subset of that artifact (the file
    # a fold pools from) is recognised by these, and a different accumulation over the same corpus files
    # (a construct-level file, another QC) is not
    fitted_labels: np.ndarray | None = None
    fitted_n_cells: np.ndarray | None = None

    def save(self, path: str | Path) -> None:
        """One `.npz` holding the arrays and the scalars; `load` is its exact inverse."""
        np.savez_compressed(
            Path(path).expanduser(),
            genes=np.asarray(self.genes, dtype=object), mean_count=self.mean_count,
            theta_ml=self.theta_ml, theta_trend=self.theta_trend, theta_ql=self.theta_ql,
            theta_sql=self.theta_sql,
            scalars=np.array([self.df, self.df0, self.tau0_sq, float(self.n_groups_used)]),
            provenance=np.array([self.source, self.fitted_on_sha256, self.fitted_on_sources_sha256],
                                dtype=object),
            fitted_labels=np.asarray([] if self.fitted_labels is None else self.fitted_labels, dtype=object),
            fitted_n_cells=np.asarray([] if self.fitted_n_cells is None else self.fitted_n_cells, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> GeneDispersion:
        z = np.load(Path(path).expanduser(), allow_pickle=True)
        df, df0, tau0_sq, n_used = (float(v) for v in z["scalars"])
        prov = [str(v) for v in z["provenance"]] + ["", "", ""]
        labels = z["fitted_labels"].astype(str) if "fitted_labels" in z.files and z["fitted_labels"].size else None
        n_cells = z["fitted_n_cells"] if "fitted_n_cells" in z.files and z["fitted_n_cells"].size else None
        return cls(genes=z["genes"].astype(str), mean_count=z["mean_count"],
                   theta_ml=z["theta_ml"], theta_trend=z["theta_trend"], theta_ql=z["theta_ql"],
                   theta_sql=z["theta_sql"], df=df, df0=df0, tau0_sq=tau0_sq,
                   n_groups_used=int(n_used), source=prov[0], fitted_on_sha256=prov[1],
                   fitted_on_sources_sha256=prov[2], fitted_labels=labels, fitted_n_cells=n_cells)

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


# ------------------------------------------------------------- the same fit, streamed from disk


def _small_members(path: Path) -> dict:
    """labels, genes, n_cells, libsize_sum, sources of a saved `PseudobulkSums` -- the cheap members."""
    import io
    import zipfile

    out = {}
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        for name in ("labels", "genes", "n_cells", "libsize_sum", "sources"):
            if f"{name}.npy" not in names:
                out[name] = np.array([], dtype=object)
                continue
            with z.open(f"{name}.npy") as fh:
                out[name] = np.load(io.BytesIO(fh.read()), allow_pickle=True)
    return out


def sources_sha256(sources) -> str:
    """The identity of the corpus files an artifact was accumulated from (its `sources` member)."""
    import hashlib

    return hashlib.sha256("\n".join(str(s) for s in sources).encode()).hexdigest()


def moment_dispersion_file(path: str | Path, *, exclude: tuple[str, ...] = (),
                           min_cells: int = MIN_CELLS_PER_GROUP, block_rows: int = 128,
                           progress: bool = False) -> tuple[np.ndarray, np.ndarray, float, int, np.ndarray]:
    """`moment_dispersion` on a saved artifact, one row block at a time, never loading it whole.

    Every quantity the in-memory estimator needs is a SUM over groups -- `sum_l w_l * excess_lg`,
    `sum_l w_l * m_lg^2`, `sum_l count_sum_lg`, `sum_l (n_l - 1)` -- so the three (L, G) members
    are walked in lockstep (`stream_pseudobulk.iter_npz_row_blocks`) and only the per-gene
    accumulators are kept. On the full X-Atlas arm (18,294 labels x 38,584 genes) the in-memory
    form wants about five 5.65 GB arrays and does not fit a 16 GB Mac; this form peaks under
    200 MB and is I/O bound.

    Equal to `moment_dispersion(PseudobulkSums.load(path))` to floating-point rounding (held by
    `tests/test_variance_model.py`). Refuses an artifact whose `count_sum` is all zero -- the
    signature of a `load_subset` product, whose fit would silently flatten the trend to
    `MIN_DISPERSION` because `mean_count` reads that member.

    Returns `(mean_count, theta_ml, df, n_groups_used, genes)`.
    """
    path = Path(path).expanduser()
    small = _small_members(path)
    labels = [str(x) for x in small["labels"]]
    genes = small["genes"].astype(str)
    n_all = small["n_cells"].astype(np.float64)
    lib_all = small["libsize_sum"].astype(np.float64)
    keep = np.array([lab not in exclude and int(n_all[i]) >= min_cells
                     for i, lab in enumerate(labels)])
    if int(keep.sum()) < 2:
        raise ValueError(
            f"need at least 2 groups with >= {min_cells} cells to fit a dispersion, "
            f"got {int(keep.sum())} -- a single group cannot separate sampling noise from dispersion")
    G = len(genes)
    num = np.zeros(G); den = np.zeros(G); count_total = np.zeros(G)
    n_total = 0.0; df = 0.0; count_any = False
    members = ("cpm_sum", "cpm_sq_sum", "count_sum")
    for r0, blk in iter_npz_row_blocks(path, [f"{m}.npy" for m in members], block_rows=block_rows):
        k = blk["cpm_sum.npy"].shape[0]
        rows = np.flatnonzero(keep[r0:r0 + k])
        if progress and (r0 // block_rows) % 20 == 0:
            print(f"  rows {r0}/{len(labels)}", flush=True)
        if rows.size == 0:
            continue
        n = n_all[r0 + rows]
        cs = blk["cpm_sum.npy"][rows]
        cq = blk["cpm_sq_sum.npy"][rows]
        ct = blk["count_sum.npy"][rows]
        if not count_any and ct.any():
            count_any = True
        m_cpm = cs / n[:, None]
        # the same Bessel correction as `moment_dispersion`, for the same reason
        v_cpm = np.maximum(cq / n[:, None] - m_cpm * m_cpm, 0.0)
        v_cpm = v_cpm * (n / np.maximum(n - 1.0, 1.0))[:, None]
        scale = (1e6 / (lib_all[r0 + rows] / n))[:, None]
        excess = v_cpm - m_cpm * scale
        w = np.maximum(n - 1.0, 0.0)[:, None]
        num += (w * excess).sum(axis=0)
        den += (w * m_cpm * m_cpm).sum(axis=0)
        count_total += ct.sum(axis=0)
        n_total += float(n.sum())
        df += float(np.maximum(n - 1.0, 0.0).sum())
    if not count_any:
        raise ValueError(f"{path.name}: count_sum is all zero -- a load_subset product, not a "
                         "fit-able artifact (the trend would flatten to MIN_DISPERSION)")
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.where(den > 0, num / den, 0.0)
    theta = np.maximum(np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    mean_count = count_total / n_total
    return mean_count, theta, df, int(keep.sum()), genes


def file_sha256(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).expanduser().open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fit_gene_dispersion_file(path: str | Path, *, exclude: tuple[str, ...] = (), window: int = 301,
                             min_cells: int = MIN_CELLS_PER_GROUP, df0: float | None = None,
                             block_rows: int = 128, progress: bool = False,
                             sha256: str | None = None) -> GeneDispersion:
    """`fit_gene_dispersion` on a saved artifact through `moment_dispersion_file`, with provenance.

    The fit is stamped with the artifact's stem and sha256 (`sha256` may be passed in when the
    caller has already hashed the file), which is what lets the pooling knob refuse a fit
    attached to the wrong source.
    """
    path = Path(path).expanduser()
    mean_count, theta_ml, df, n_used, genes = moment_dispersion_file(
        path, exclude=exclude, min_cells=min_cells, block_rows=block_rows, progress=progress)
    theta_trend = dispersion_trend(mean_count, theta_ml, window=window)
    theta_ql = quasi_dispersion(mean_count, theta_ml, theta_trend)
    fitted_df0, tau0_sq = fit_variance_prior(theta_ql, df)
    if df0 is None:
        df0 = fitted_df0
    elif df0 < 0:
        raise ValueError(f"df0 must be non-negative, got {df0}")
    theta_sql = shrink_quasi_dispersion(theta_ql, df, df0, tau0_sq)
    small = _small_members(path)
    return GeneDispersion(genes=np.asarray(genes), mean_count=mean_count, theta_ml=theta_ml,
                          theta_trend=theta_trend, theta_ql=theta_ql, theta_sql=theta_sql,
                          df=df, df0=df0, tau0_sq=tau0_sq, n_groups_used=n_used,
                          source=path.stem, fitted_on_sha256=sha256 or file_sha256(path),
                          fitted_on_sources_sha256=sources_sha256(small["sources"]),
                          fitted_labels=np.asarray([str(x) for x in small["labels"]], dtype=object),
                          fitted_n_cells=np.asarray(small["n_cells"], dtype=np.int64))


def main(argv: list[str] | None = None) -> int:
    """Fit one artifact and save the fit: `python -m sidechain.data.dispersion SRC.npz --out FIT.npz`."""
    import argparse
    import json
    import time

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("source", type=Path, help="a saved PseudobulkSums (.npz), read in row blocks")
    ap.add_argument("--out", type=Path, required=True, help="where the fit (.npz) is written")
    ap.add_argument("--exclude", action="append", default=[], metavar="LABEL",
                    help="label rows left out of the fit (repeatable); default: none")
    ap.add_argument("--window", type=int, default=301)
    ap.add_argument("--min-cells", type=int, default=MIN_CELLS_PER_GROUP)
    ap.add_argument("--block-rows", type=int, default=128)
    args = ap.parse_args(argv)
    t0 = time.time()
    gd = fit_gene_dispersion_file(args.source, exclude=tuple(args.exclude), window=args.window,
                                  min_cells=args.min_cells, block_rows=args.block_rows, progress=True)
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    gd.save(args.out)
    ok = gd.mean_count > 0
    q = np.quantile(gd.mean_count[ok], [0.1, 0.5, 0.9])
    trend_at = [float(np.interp(np.log(x), np.log(np.sort(gd.mean_count[ok])),
                                gd.theta_trend[ok][np.argsort(gd.mean_count[ok], kind="stable")]))
                for x in q]
    rec = {"source": str(args.source), "source_sha256": gd.fitted_on_sha256, "out": str(args.out),
           "exclude": args.exclude, "window": args.window, "min_cells": args.min_cells,
           "genes": int(len(gd.genes)), "genes_expressed": int(ok.sum()),
           "n_groups_used": gd.n_groups_used, "df": gd.df, "df0": gd.df0, "tau0_sq": gd.tau0_sq,
           "theta_trend_at_p10_p50_p90_mean_count": trend_at,
           "theta_sql_median": float(np.median(gd.theta_sql[ok])),
           "theta_sql_max_shift_from_ql": float(np.max(np.abs(gd.theta_sql[ok] - gd.theta_ql[ok]))),
           "seconds": round(time.time() - t0, 1)}
    args.out.expanduser().with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
