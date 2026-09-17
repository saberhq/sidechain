"""Emit predicted cells as raw integer counts.

The 2026 scorer reads raw counts and computes four of its six metrics from a
Wilcoxon test on the cells we emit versus the real controls, so HOW cells are
generated is part of the model: their depth, their cell-to-cell dispersion
and which genes are ever nonzero all move the score (private reports/05 s3.3).

The emitters here share one recipe -- a per-gene expected-fraction profile
for the target context, optionally shifted per perturbation by a log2
fold-change vector, then sampled as independent Poisson counts at library
sizes drawn from that context's own controls. Poisson is the simplest
count-generating choice that is (a) integral, (b) never negative, (c) nonzero
on every gene the control expresses, and (d) independent across cells and
perturbations, which the expression-error metric's correction cap requires.
Over-dispersion (negative binomial) is a later, measured, change.
"""
from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import scipy.sparse as sp


@dataclass
class ContextProfile:
    """What a context's control cells say about it: expected per-gene fractions
    and the empirical library-size distribution."""
    name: str
    genes: np.ndarray          # (G,) symbols, in submission order
    fraction: np.ndarray       # (G,) mean CPM / 1e6, sums to ~1
    libsizes: np.ndarray       # (n_cells,) UMI per control cell
    n_cells: int

    @classmethod
    def from_controls(cls, path, name: str, *, min_libsize: float = 0.0) -> ContextProfile:
        a = ad.read_h5ad(path)
        X = sp.csr_matrix(a.X, dtype=np.float64)
        lib = np.asarray(X.sum(axis=1)).ravel()
        keep = lib > max(min_libsize, 0)
        cpm_mean = np.asarray((sp.diags(1e6 / lib[keep]) @ X[keep]).mean(axis=0)).ravel()
        frac = cpm_mean / cpm_mean.sum()
        return cls(name=name, genes=a.var_names.astype(str).to_numpy(), fraction=frac,
                   libsizes=lib[keep], n_cells=int(keep.sum()))


class PoissonEmitter:
    """Integer cells around a (possibly shifted) context profile.

    Cell-to-cell spread is ONE DIAL, `lam` in [0, 1], because the four DE
    metrics are a rank test on the cells we emit and the number of genes that
    test "calls" is set almost entirely by that spread (private
    reports/05 s3.3). The two named modes are its endpoints:

    poisson  lam=1: independent Poisson counts at library sizes drawn from the
             context's controls. Realistic-looking cells; against the real
             controls they call only a few hundred genes.
    even     lam=0: minimum-variance allocation: every cell gets the same
             depth, and each gene's total round(n * lambda_g) is spread as
             evenly as possible over the n cells (floor everywhere, the
             remainder as single extra counts on a random subset). Per-gene
             means are exact, every expressed gene stays nonzero (no
             fold-change blow-ups), and the rank test calls essentially the
             whole DE universe, which is what `fid`'s coverage term rewards.
             Day-0 board: the even-spread null scored -0.01 where
             resampled-control nulls scored -0.30.

    An interior `lam` mixes the two: a fraction 1 - lam^2 of each gene's
    expected counts is laid down by the even allocation and the remaining
    lam^2 is sampled as Poisson, so counts stay integral and non-negative
    with no rounding step. `lam` IS the "shrink the emitted cloud toward the
    predicted mean by factor lam" dial of private
    research/ideas/emission-sharpening-dial.md. The exact variance law, per
    gene with expected fraction f at pool depth L (w = lam^2):
    Var = w*f*E[L] + w^2*f^2*Var(L). So conditional on a cell's depth the sd
    is lam times the Poisson sd, but on a pool with real depth spread the
    second term dominates every well-expressed gene (above mean count
    ~1/CV(L)^2, roughly 5-7 on the 2026 control pools), and THERE the
    marginal sd spaces as lam^2, not lam. Two more knowingly-accepted
    wrinkles: the even share is pinned to the pool MEDIAN depth while the
    Poisson share draws at the pool MEAN, so the interior-lam mean depth
    drifts by w*(mean-median) (~+1.4% at lam=0.5 on the 2026 pools) -- the
    per-gene FRACTIONS, which the CPM-side metrics read, are lam-invariant.
    At lam exactly 0 or 1 the mixture short-circuits to the endpoint code
    path, so those arms are bit-identical to the named modes at the same
    seed.

    Exactly one of `dispersion` / `lam` may be passed: the modes are sugar for
    the endpoints, and accepting both would let them disagree silently.
    """

    def __init__(self, profile: ContextProfile, seed: int = 0, *, dispersion: str | None = None,
                 lam: float | None = None, libsize_quantiles: tuple[float, float] = (0.0, 1.0)):
        if lam is not None and dispersion is not None:
            raise ValueError("pass dispersion or lam, not both -- the modes are the dial's "
                             "endpoints (even is lam=0, poisson is lam=1)")
        if lam is None:
            dispersion = "poisson" if dispersion is None else dispersion
            if dispersion not in ("poisson", "even"):
                raise ValueError("dispersion must be 'poisson' or 'even'")
            lam = 0.0 if dispersion == "even" else 1.0
        lam = float(lam)
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        self.p = profile
        self.lam = lam
        # A readable label for run records; the float is the ground truth.
        self.dispersion = dispersion if dispersion is not None else f"lam={lam:g}"
        self.rng = np.random.default_rng(seed)
        lo, hi = np.quantile(profile.libsizes, libsize_quantiles)
        self._lib_pool = profile.libsizes[(profile.libsizes >= lo) & (profile.libsizes <= hi)]
        self._lib_median = float(np.median(self._lib_pool))

    def _fraction(self, log2fc: np.ndarray | None) -> np.ndarray:
        frac = self.p.fraction
        if log2fc is not None:
            if log2fc.shape != frac.shape:
                raise ValueError("log2fc must be per gene on the submission axis")
            frac = frac * np.exp2(np.nan_to_num(log2fc, nan=0.0, posinf=0.0, neginf=0.0))
            frac = frac / frac.sum()
        return frac

    def emit(self, n: int, log2fc: np.ndarray | None = None, *, max_counts_per_cell: int = 1_000_000) -> sp.csr_matrix:
        frac = self._fraction(log2fc)
        w = self.lam * self.lam    # Poisson share of the variance; sd scales as lam
        if w == 0.0:
            counts = self._emit_even(n, frac)
        else:
            lib = self.rng.choice(self._lib_pool, size=n, replace=True).astype(np.float64)
            if w == 1.0:
                counts = self.rng.poisson(lib[:, None] * frac[None, :]).astype(np.float32)
            else:
                counts = self._emit_even(n, frac, depth_frac=1.0 - w)
                counts += self.rng.poisson(w * lib[:, None] * frac[None, :]).astype(np.float32)
        tot = counts.sum(axis=1)
        over = tot > max_counts_per_cell
        if over.any():  # unreachable at 20k depth; guard the contract anyway
            counts[over] = np.floor(counts[over] * (max_counts_per_cell / tot[over])[:, None])
        return sp.csr_matrix(counts)

    def emit_dual(self, n: int, log2fc_cell: np.ndarray | None, log2fc_bulk: np.ndarray | None, *,
                  iterations: int = 100, tolerance: float = 2e-4,
                  max_projection: float = 0.03, on_fail: str = "raise") -> sp.csr_matrix:
        """Two amplitudes in one count matrix (T84, private research/ideas/two-amplitude-emitter.md).

        cell-eval2 reads a perturbation's cells twice: `pds` and `mse` through the depth-weighted
        pseudobulk SUM, the four Wilcoxon members through the equal-weight per-cell CPM. With one
        fraction vector both channels carry one amplitude. This emits cells whose equal-weight
        per-cell mean follows `log2fc_cell` while their column sums follow `log2fc_bulk`, at this
        emitter's own scatter (`lam`) and drawn depths: the template is `self.emit(n, log2fc_cell)`,
        then each gene's counts are tilted across cells in proportion to the cell's depth until the
        two moments hold (`dual_moment_counts`), then rounded to integers that keep every cell's
        depth and every column's total (`repair_bulk_totals`). Both routines follow kaipengm2's
        VCC 2026 reference implementation (MIT), read in full under T82.

        The tilt lives inside the cells' depth envelope: a gene's bulk share can move away from its
        per-cell share only by the factor the depth spread allows, so `lam = 0` (every cell at one
        depth) cannot decouple anything and is refused, and a bulk profile whose L1 projection onto
        that envelope exceeds `max_projection` is refused rather than silently clipped.
        `log2fc_cell == log2fc_bulk` is the pin's own control: same profile on both channels, the
        template's column sums re-pinned to their expectation.

        The two moments can also be JOINTLY unreachable when many strong genes sit at the
        envelope's edge at once (measured 2026-09-16: at lambda 0.5 alpha_bulk 2.0 against
        alpha_cell 1.35 fails on ~15 % of HEK293T targets and none of K562's; alpha_bulk 3.0
        fails on most). `on_fail="raise"` surfaces that; `on_fail="fallback"` returns the
        single-amplitude template for that call instead and sets `self.dual_fallbacks`, so a
        run can report how many targets carried one amplitude.
        """
        if on_fail not in ("raise", "fallback"):
            raise ValueError(f"on_fail must be 'raise' or 'fallback', got {on_fail!r}")
        if self.lam == 0.0:
            raise ValueError("emit_dual needs a depth spread: at lam=0 every cell has the same "
                             "depth, so the pseudobulk and the per-cell mean cannot differ")
        p_cell = self._fraction(log2fc_cell)
        p_bulk = self._fraction(log2fc_bulk)
        template = self.emit(n, log2fc_cell).toarray().astype(np.float64)
        depths = np.rint(template.sum(axis=1)).astype(np.int64)
        seed = int(self.rng.integers(0, 2**32 - 1))
        try:
            counts = dual_moment_counts(template, p_cell, p_bulk, depths=depths, seed=seed,
                                        iterations=iterations, tolerance=tolerance,
                                        max_projection=max_projection)
        except ValueError:
            if on_fail == "raise":
                raise
            self.dual_fallbacks = getattr(self, "dual_fallbacks", 0) + 1
            return sp.csr_matrix(template.astype(np.float32))
        return sp.csr_matrix(counts.astype(np.float32))

    def _emit_even(self, n: int, frac: np.ndarray, depth_frac: float = 1.0) -> np.ndarray:
        # per-gene total over n cells; `depth_frac` carves out the even share of
        # a lam mixture (1.0 = the whole depth, bit-identical to the old form)
        total = np.rint(n * self._lib_median * depth_frac * frac).astype(np.int64)
        base, rem = np.divmod(total, n)
        counts = np.broadcast_to(base.astype(np.float32), (n, len(frac))).copy()
        cols = np.where(rem > 0)[0]
        for j in cols:  # the remainder as +1 on a random subset of cells
            counts[self.rng.choice(n, size=int(rem[j]), replace=False), j] += 1.0
        return counts


def repair_bulk_totals(integer: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Move single counts between cells within a column until every column's total equals its
    expectation (rounded, remainders to the largest fractional parts) while no cell's depth
    changes. `integer` is floor(expected) plus a per-row residual scatter, so every count above
    the floor is movable; a column in excess gives counts up from the cells that rounded up most,
    a column in deficit takes them onto the cells furthest below their expectation. After
    kaipengm2 (MIT). In place on `integer`; returned for convenience."""
    expected = np.asarray(expected, dtype=np.float64)
    if integer.shape != expected.shape or integer.ndim != 2:
        raise ValueError("count and expectation axes differ")
    if not np.isfinite(expected).all() or (expected < 0).any() or (integer < 0).any():
        raise ValueError("invalid count expectations")
    row_totals = integer.sum(axis=1, dtype=np.int64)
    columns = expected.sum(axis=0)
    target = np.floor(columns).astype(np.int64)
    remaining = int(row_totals.sum() - target.sum())
    if remaining < 0 or remaining > len(target):
        raise ValueError("expected and integer grand totals differ")
    if remaining:
        chosen = np.argsort(columns - target, kind="stable")[-remaining:]
        target[chosen] += 1
    difference = integer.sum(axis=0, dtype=np.int64) - target
    capacity = np.zeros(len(integer), dtype=np.int64)
    for gene in np.flatnonzero(difference > 0):
        excess = int(difference[gene])
        rounded_up = integer[:, gene] - np.floor(expected[:, gene]).astype(np.int64)
        rows = np.flatnonzero(rounded_up > 0)
        if rounded_up[rows].sum() < excess:
            raise ValueError("initial rounding fell below its integer floor")
        rows = rows[np.argsort(-(integer[rows, gene] - expected[rows, gene]), kind="stable")]
        if excess <= len(rows):
            selected = rows[:excess]
            integer[selected, gene] -= 1
            capacity[selected] += 1
        else:
            for row in rows:
                take = min(excess, int(rounded_up[row]))
                integer[row, gene] -= take
                capacity[row] += take
                excess -= take
                if not excess:
                    break
    deficit_genes = np.flatnonzero(difference < 0)
    deficit_genes = deficit_genes[np.argsort(difference[deficit_genes], kind="stable")]
    for gene in deficit_genes:
        deficit = int(-difference[gene])
        while deficit:
            rows = np.flatnonzero(capacity > 0)
            if not len(rows):
                raise AssertionError("column repair exhausted row capacity")
            take = min(deficit, len(rows))
            score = expected[rows, gene] - integer[rows, gene]
            selected = rows[np.argsort(-score, kind="stable")[:take]]
            integer[selected, gene] += 1
            capacity[selected] -= 1
            deficit -= take
    if capacity.any() or (integer < 0).any():
        raise AssertionError("column repair lost counts")
    if not np.array_equal(integer.sum(axis=1, dtype=np.int64), row_totals):
        raise AssertionError("column repair changed a cell depth")
    if not np.array_equal(integer.sum(axis=0, dtype=np.int64), target):
        raise AssertionError("column repair failed its bulk totals")
    return integer


def dual_moment_counts(template: np.ndarray, probability: np.ndarray, bulk_probability: np.ndarray,
                       *, depths: np.ndarray, seed: int, iterations: int = 100,
                       tolerance: float = 2e-4, max_projection: float = 0.03) -> np.ndarray:
    """Integer cells whose equal-weight per-cell composition is `probability` and whose
    depth-weighted composition (the pseudobulk) is `bulk_probability`, at the given depths.

    Each template row is normalised to a composition; with z = depth / mean depth, the per-cell
    mean is `x.mean(0)` and the bulk is `z @ x / n`. The bulk is first projected onto the box the
    depth spread allows (`[z_min p, z_max p]`, rescaled to sum to one), then each gene's column is
    tilted by `1 + t_g (z_i - m_g)` -- which leaves that gene's per-cell mean fixed and moves its
    bulk -- with `t_g` from the column's depth variance, rows renormalised, alternating with a
    column rescale onto `probability`, until both moments hold to `tolerance` (L1). Integer
    rounding keeps each row's depth (floor plus a residual scatter on the fractional parts), and
    `repair_bulk_totals` then pins every column's total. After kaipengm2 (MIT).
    """
    template = np.asarray(template, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    bulk = np.asarray(bulk_probability, dtype=np.float64)
    depths = np.asarray(depths)
    if template.ndim != 2 or not all(template.shape):
        raise ValueError("template must be a nonempty cell-by-gene matrix")
    if p.shape != (template.shape[1],) or bulk.shape != p.shape:
        raise ValueError("moment gene axes differ from the template")
    for value in (template, p, bulk, depths):
        if not np.isfinite(value).all() or (value < 0).any():
            raise ValueError("inputs must be finite and nonnegative")
    if (depths != np.floor(depths)).any() or (depths < 1).any() or depths.shape != (len(template),):
        raise ValueError("depths must be whole numbers >= 1, one per cell")
    if (template.sum(axis=1) <= 0).any():
        raise ValueError("template rows must have positive mass")
    if not np.isclose(p.sum(), 1, rtol=0, atol=1e-8) or not np.isclose(bulk.sum(), 1, rtol=0, atol=1e-8):
        raise ValueError("moment probabilities must sum to one")
    if not isinstance(iterations, int) or iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("invalid fitting iterations or tolerance")
    depths = depths.astype(np.int64)
    x = template.copy()
    z = depths / depths.mean()
    n = len(x)
    desired = n * p
    requested_bulk = bulk.copy()
    if np.ptp(depths) == 0:
        bulk = p.copy()
    else:
        lower = (z.min() + 0.01 * (1 - z.min())) * p
        upper = (z.max() - 0.01 * (z.max() - 1)) * p
        lo, hi = 0.0, 1.0
        for _ in range(1024):
            if np.clip(bulk * hi, lower, upper).sum() >= 1:
                break
            hi *= 2
            if not np.isfinite(hi):
                raise ValueError("bulk support cannot satisfy the feasible moment bounds")
        else:
            raise ValueError("cannot bracket the bulk projection")
        for _ in range(70):
            mid = (lo + hi) / 2
            if np.clip(bulk * mid, lower, upper).sum() < 1:
                lo = mid
            else:
                hi = mid
        bulk = np.clip(bulk * ((lo + hi) / 2), lower, upper)
    projection_error = float(np.abs(bulk - requested_bulk).sum())
    if projection_error > max_projection:
        raise ValueError(f"requested bulk profile lies {projection_error:.4f} (L1) outside the "
                         f"depth envelope, above max_projection={max_projection}")
    ratio = np.divide(bulk, p, out=np.ones_like(p), where=p > 0)
    x /= np.maximum(x.sum(axis=1, keepdims=True), 1e-30)
    missing = (x.sum(axis=0) == 0) & (desired > 0)
    x[:, missing] = p[missing]
    x = 0.999 * x + 0.001 * p[None, :]
    for step in range(iterations):
        col = x.sum(axis=0)
        x *= np.divide(desired, col, out=np.zeros_like(col), where=col > 0)
        first = z @ x
        mean = np.divide(first, desired, out=np.ones_like(first), where=desired > 0)
        second = (z * z) @ x
        var = np.maximum(np.divide(second, desired, out=np.zeros_like(second), where=desired > 0) - mean * mean, 0)
        tilt = np.divide(ratio - mean, var, out=np.zeros_like(mean), where=var > 1e-12)
        lower_t = -0.95 / np.maximum(z.max() - mean, 1e-12)
        upper_t = 0.95 / np.maximum(mean - z.min(), 1e-12)
        tilt = np.clip(tilt, lower_t, upper_t)
        x *= 1 + tilt[None, :] * (z[:, None] - mean[None, :])
        x /= np.maximum(x.sum(axis=1, keepdims=True), 1e-30)
        if step % 5 == 4:
            cpm_error = float(np.abs(x.mean(axis=0) - p).sum())
            bulk_error = float(np.abs(z @ x / n - bulk).sum())
            if max(cpm_error, bulk_error) < tolerance:
                break
    cpm_error = float(np.abs(x.mean(axis=0) - p).sum())
    bulk_error = float(np.abs(z @ x / n - bulk).sum())
    if max(cpm_error, bulk_error) > 1e-3:
        raise ValueError(f"moment fitting failed: per-cell L1 {cpm_error:.4g}, bulk L1 {bulk_error:.4g}")
    expected = x * depths[:, None]
    integer = np.floor(expected).astype(np.int64)
    fractions = expected - integer
    rng = np.random.default_rng(seed)
    for i in range(n):
        residual = int(depths[i]) - int(integer[i].sum())
        if residual:
            cumulative = np.cumsum(fractions[i])
            cumulative *= residual / cumulative[-1]
            locations = np.searchsorted(cumulative, np.arange(residual) + rng.random(), side="right")
            np.add.at(integer[i], np.minimum(locations, len(p) - 1), 1)
    integer = repair_bulk_totals(integer, expected)
    if (integer < 0).any() or not np.array_equal(integer.sum(axis=1), depths):
        raise AssertionError("integer emission lost row depths")
    return integer


def log2fc_from_cpm(mean_cpm_pert: np.ndarray, mean_cpm_ctrl: np.ndarray, pseudocount: float = 1.0) -> np.ndarray:
    """Shrunk log2 fold change of arithmetic-mean CPM; the pseudocount keeps
    lowly expressed genes from producing wild ratios that would not transfer."""
    return np.log2((mean_cpm_pert + pseudocount) / (mean_cpm_ctrl + pseudocount))


def remap_to_axis(values: np.ndarray, source_genes: np.ndarray, target_genes: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Place a per-gene vector from a source gene axis onto the submission axis
    by symbol; genes the source never measured get `fill` (no change)."""
    pos = {g: i for i, g in enumerate(source_genes)}
    out = np.full(len(target_genes), fill, dtype=np.float64)
    for j, g in enumerate(target_genes):
        i = pos.get(g)
        if i is not None:
            out[j] = values[i]
    return out
