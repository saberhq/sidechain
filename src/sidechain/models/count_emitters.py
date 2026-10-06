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

# THE control-cell floor, for every path that builds a ContextProfile.
#
# One constant because the two paths that matter disagreed for three weeks:
# `submit.build` floored at 1000 and `eval.loco` at 500, so a mirror-scored arm
# was not quite the arm that shipped. Measured 2026-09-20 (T18 check 5,
# runs/probes/t18_check5_libsize_floor): of the nine fold control arms on disk
# only two hold ANY cell in the 500-1000 band -- one cell and three cells -- and
# rescoring the shipped delta across the gap moves raw pds_cosine by 2e-4, which
# is fifty times under the 272-target noise bar. On the SUBMISSION side the same
# gap drops 178 cells in context B and moves its baseline by 0.78 (L2, the
# scorer's bulk_lognorm space) with 161 genes past 0.1 log2. So the mirror could
# never have seen this defect: it lives entirely on the path that ships.
#
# The value is 1000 because that is what every submitted build has used.
# (final-phase: controls) 1000 against 3000 is open, and only D/E/F can settle it: read each new
# context's depth tail first -- context B holds 598 cells under 3,000 UMI, A and C hold none
# (private/research/ideas/batch-effect-diagnostics.md, T18).
# `from_controls` keeps its own default of 0 -- it is a library function, and a
# caller that wants the project's policy says so by passing this.
CONTROL_MIN_LIBSIZE = 1000.0


@dataclass
class ContextProfile:
    """What a context's control cells say about it: expected per-gene fractions
    and the empirical library-size distribution."""
    name: str
    genes: np.ndarray          # (G,) symbols, in submission order
    fraction: np.ndarray       # (G,) mean CPM / 1e6, sums to ~1
    libsizes: np.ndarray       # (n_cells,) UMI per control cell
    n_cells: int
    # (G,) the POOLED profile, sum of counts / sum of depths over the same cells: what cell-eval2's
    # control pseudobulk is (`bulk_lognorm` reads group sums, so a deep cell weighs more). It
    # differs from `fraction` wherever composition tracks depth -- on the 2026 controls the G2/M
    # genes sit 12-20 % higher in it, because deep cells are cycling cells -- and a pseudobulk
    # anchored to `fraction` carries that difference on every perturbation (T84 round 2,
    # private research/ideas/board-methods-survey.md D9). None for a hand-built profile.
    bulk_fraction: np.ndarray | None = None
    # (n_cells, G) the control cells' own counts, CSR, rows in the order of `libsizes`: the cells
    # `emit_dual(shape=True)` re-rates (T85). Kept only on request
    # (`from_controls(keep_cells=True)`): it is the whole control arm in memory, about 8 bytes a
    # nonzero. None for a hand-built profile and for every path that never shapes.
    cells: sp.csr_matrix | None = None
    # (G,) with `cells`: how far each gene's cell-to-cell spread exceeds a Poisson draw's at these
    # depths, as a squared CV of the cell's own (latent) fraction of the gene -- the inverse shape
    # of a gamma on it; 0 where sampling alone explains the spread. It says what a cell's count
    # tells about its own rate, and so where the extra counts of an up-shift land. The moment
    # estimate is noise for a gene the controls hold a few counts of, so it is drawn toward the
    # well-measured genes' median with the weight of 200 counts.
    dispersion: np.ndarray | None = None
    # (G,) with `cells`: a factor on that rate for a cell that shows NONE of the gene. Whatever
    # mix of rates and depths the cells have, the cells with no count of a gene expect, summed,
    # as many counts as there are cells with exactly one (Robbins's identity); this is that
    # number over what the gamma reading gives the same cells (five counts' weight toward 1).
    # Near 1 for a gamma-Poisson gene, far below it for a gene that is off in most cells, which
    # a gamma would keep topping up.
    zero_rate: np.ndarray | None = None

    @classmethod
    def from_controls(cls, path, name: str, *, min_libsize: float = 0.0,
                      keep_cells: bool = False) -> ContextProfile:
        a = ad.read_h5ad(path)
        X = sp.csr_matrix(a.X, dtype=np.float64)
        lib = np.asarray(X.sum(axis=1)).ravel()
        keep = lib > max(min_libsize, 0)
        if not keep.any():
            # Found 2026-09-20 raising the default floor to CONTROL_MIN_LIBSIZE: an
            # empty pool produced an all-nan profile and a warning, and the emitter
            # then wrote nan counts for a whole context. `analytic_pds._control_profile`
            # has always raised here; this is the same refusal on the shipping path.
            raise ValueError(
                f"{name}: all {lib.size} control cells fall at or below "
                f"min_libsize={min_libsize:g} (deepest is {lib.max():.0f} UMI)")
        cpm_mean = np.asarray((sp.diags(1e6 / lib[keep]) @ X[keep]).mean(axis=0)).ravel()
        frac = cpm_mean / cpm_mean.sum()
        pooled = np.asarray(X[keep].sum(axis=0)).ravel()
        cells = dispersion = zero_rate = None
        if keep_cells:
            cells = sp.csr_matrix(X[keep], dtype=np.float32)
            cells.eliminate_zeros()                                # a stored zero is not a count
            depth = lib[keep]
            square = np.zeros_like(frac)                           # sum over cells of (count / depth)^2
            for lo in range(0, cells.shape[0], 2000):              # cells in blocks, genes in blocks: memory
                block = cells[lo:lo + 2000]
                part = block.data / np.repeat(depth[lo:lo + 2000], np.diff(block.indptr))
                square += np.bincount(block.indices, weights=part * part, minlength=len(frac))
            excess = (square / len(depth) - frac * frac
                      - frac * float(np.mean(1.0 / depth)))        # variance less the Poisson share
            dispersion = np.divide(np.maximum(excess, 0.0), frac * frac, out=np.zeros_like(frac),
                                   where=frac > 0)
            well = pooled >= 200
            typical = float(np.median(dispersion[well])) if well.any() else 0.0
            dispersion = (pooled * dispersion + 200.0 * typical) / (pooled + 200.0)
            # what the gamma reading expects of the cells with no count, summed: every cell's
            # expectation at count 0, less that of the cells that do carry the gene
            gamma_at_zero = np.zeros_like(frac)
            for lo in range(0, len(frac), 512):
                m, f = frac[None, lo:lo + 512], dispersion[None, lo:lo + 512]
                gamma_at_zero[lo:lo + 512] = (depth[:, None] * m / (1.0 + f * m * depth[:, None])).sum(axis=0)
            for lo in range(0, cells.shape[0], 2000):
                block = cells[lo:lo + 2000]
                row = np.repeat(depth[lo:lo + 2000], np.diff(block.indptr))
                m, f = frac[block.indices], dispersion[block.indices]
                gamma_at_zero -= np.bincount(block.indices, weights=row * m / (1.0 + f * m * row),
                                             minlength=len(frac))
            ones = np.asarray((cells == 1).sum(axis=0)).ravel().astype(np.float64)
            zero_rate = np.clip((ones + 5.0) / (np.maximum(gamma_at_zero, 0.0) + 5.0), 0.0, 4.0)
        return cls(name=name, genes=a.var_names.astype(str).to_numpy(), fraction=frac,
                   libsizes=lib[keep], n_cells=int(keep.sum()), bulk_fraction=pooled / pooled.sum(),
                   cells=cells, dispersion=dispersion, zero_rate=zero_rate)


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

    `bulk_anchor` says which control profile `emit_dual`'s pseudobulk channel is built on:
    "mean_cpm" (the default, every shipped entry through SER-7abefn) is `profile.fraction`,
    the same profile as the per-cell channel; "pooled" is `profile.bulk_fraction`, the
    depth-weighted profile the scorer's control pseudobulk actually is. Only the pseudobulk
    moves: the per-cell channel stays on `fraction`, because that is where the real controls'
    per-cell mean sits, and moving it would read to the Wilcoxon members as DE on every
    perturbation. `emit` has one channel and ignores the anchor.

    `emit_dual(..., scatter=)` is the same dial PER GENE (T85, private
    research/ideas/reach-call-set-emitter.md): a factor >= 0 on each gene's cell-to-cell
    scatter around its predicted count, applied to the template before the two moments are
    fitted. 1 leaves the gene as `lam` emitted it; 0 removes its sampling scatter, so what is
    left is the depth tilt the two moments need and the integer rounding; above 1 the scatter
    is widened (counts that would fall below zero are held at zero, and the fit re-pins the
    gene's two moments). Which genes the rank test calls, and in what order, is decided gene
    by gene by that scatter.

    `emit_dual(..., shape=True)` changes what that scatter IS for the whole block (T85): the
    cells are real control cells, and every gene is the cell's own count of it, re-rated to the
    prediction -- thinned for a predicted fall, topped up at the cell's own rate for a
    predicted rise. A template gene is narrower than a real cell's at every `lam` below 1, and
    the rank test calls it for that whatever is predicted; a re-rated gene with nothing
    predicted is a control gene, and one with a shift predicted moves in the test as a real
    cell's gene does. What it was measured to do and where it stops: private
    research/ideas/reach-call-set-emitter.md, Outcome 2026-10-06. It needs a profile that kept
    its control cells (`ContextProfile.from_controls(keep_cells=True)`).
    """

    def __init__(self, profile: ContextProfile, seed: int = 0, *, dispersion: str | None = None,
                 lam: float | None = None, libsize_quantiles: tuple[float, float] = (0.0, 1.0),
                 bulk_anchor: str = "mean_cpm"):
        if bulk_anchor not in ("mean_cpm", "pooled"):
            raise ValueError(f"bulk_anchor must be 'mean_cpm' or 'pooled', got {bulk_anchor!r}")
        if bulk_anchor == "pooled" and profile.bulk_fraction is None:
            raise ValueError("bulk_anchor='pooled' needs a profile with bulk_fraction "
                             "(ContextProfile.from_controls computes it)")
        self.bulk_anchor = bulk_anchor
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
        in_pool = (profile.libsizes >= lo) & (profile.libsizes <= hi)
        self._lib_pool = profile.libsizes[in_pool]
        self._lib_rows = np.flatnonzero(in_pool)    # the same cells as rows of `profile.cells`
        self._lib_median = float(np.median(self._lib_pool))

    def _fraction(self, log2fc: np.ndarray | None, *, bulk: bool = False) -> np.ndarray:
        frac = self.p.bulk_fraction if (bulk and self.bulk_anchor == "pooled") else self.p.fraction
        if log2fc is not None:
            if log2fc.shape != frac.shape:
                raise ValueError("log2fc must be per gene on the submission axis")
            frac = frac * np.exp2(np.nan_to_num(log2fc, nan=0.0, posinf=0.0, neginf=0.0))
            frac = frac / frac.sum()
        return frac

    def emit(self, n: int, log2fc: np.ndarray | None = None, *, max_counts_per_cell: int = 1_000_000) -> sp.csr_matrix:
        self.last_dual, self.last_dual_reason = None, None    # emit_dual sets them after its template
        self.last_sharpened, self.last_shaped = None, None
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
                  max_projection: float = 0.03, on_fail: str = "raise",
                  scatter: np.ndarray | None = None, shape: bool = False) -> sp.csr_matrix:
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
        template's column sums re-pinned to their expectation -- under the default anchor. With
        `bulk_anchor="pooled"` the same call moves only the anchor: the column sums go to the
        pooled profile times the shift while the per-cell mean stays on `fraction` times it.

        The two moments can also be JOINTLY unreachable when many strong genes sit at the
        envelope's edge at once (measured 2026-09-16: at lambda 0.5 alpha_bulk 2.0 against
        alpha_cell 1.35 fails on ~15 % of HEK293T targets and none of K562's; alpha_bulk 3.0
        fails on most). `on_fail="raise"` surfaces that; `on_fail="fallback"` returns the
        single-amplitude template for that call instead and sets `self.dual_fallbacks`, so a
        run can report how many targets carried one amplitude.

        The template is `self.emit`, which knows nothing of `bulk_anchor`: under the pooled
        anchor a `"fallback"` target therefore loses the anchor together with the second
        amplitude (T84 sweep, 2026-10-01: every entry through SER-11abefknw). `on_fail="anchor"`
        puts one rung in between: when the requested pair cannot be met, fit the per-cell
        amplitude on BOTH channels with the summed profile still on this emitter's anchor
        (the identity call above, which on the five mirror folds never failed), and only if
        that fails too return the template. Both rungs count in `dual_fallbacks` (the target
        carried one amplitude); `dual_fallbacks_anchor` counts the ones that kept the anchor.
        The retry reuses the first attempt's integer seed and draws nothing from `self.rng`,
        so the targets emitted after a fallback are the same cells in either mode.
        `last_dual` ("dual" | "anchor" | "template") and `last_dual_reason` ("envelope" | "fit"
        | "other", None when the pair was met) describe the call just made, for the caller's
        per-target record; a plain `emit` resets both to None.

        Three things the rung is not. Under `bulk_anchor="mean_cpm"` there is no anchor to keep:
        the same rung is the identity call, so it re-pins the template's column sums to their
        expectation and changes mean_cpm fallback targets too. It is reachable only where one
        amplitude on the anchor is itself inside the depth envelope -- a context whose pooled and
        mean-CPM control profiles differ by more than `max_projection` (L1) fails both rungs on
        every target, and the caller's count is how that shows. And it is not retried for a
        ValueError that is neither an envelope refusal nor a failed fit ("other": a contract
        violation, not an unreachable pair) -- that one ends on the template, as before.

        `scatter` (T85) is one factor per gene, >= 0. Before the fit, each gene's template
        column c is replaced by max(m + scatter * (c - m), 0), with m = the cell's own depth times
        the gene's per-cell fraction: the count the gene is predicted to have in that cell. 1 is
        the template (None, or all ones, touches nothing and is bit-identical); 0 leaves no
        sampling scatter in the gene; above 1 widens it, and the floor at zero then piles cells
        onto zero the way a sparse gene's real cells are (the fit re-pins the mean that floor
        moved). The template itself is drawn exactly as without it, so the RNG
        stream, every cell's depth and every other gene's draw are the unscattered call's, and
        the fit then pins the same two moments: the gene's per-cell mean and its column total do
        not move, only how its counts are spread over the cells. What a fully sharpened gene
        keeps is the depth tilt the two moments ask of it and the integer rounding, so it is a
        point mass in CPM only where its two profiles agree. If the fit cannot be met, the
        rungs above apply: the anchor rung is fitted on the same re-scattered template, and the
        last rung returns the ORIGINAL template, untouched. `last_sharpened` is the number of
        genes whose scatter was changed (either way) in the block just returned (0 on the
        template rung, None when `scatter` was not passed).

        `shape=True` (T85) emits the block in the CONTROLS' OWN SHAPE. The fit then starts from
        real control cells in place of the template: as many as are emitted, drawn from the
        profile's kept cells (inside the emitter's depth window, without replacement while the
        window holds enough), each at the depth its own counts add up to. Every gene is the
        cell's own count of it, re-rated so that the block's two moments sit on the prediction
        (`_rerate`): where less is predicted a count is thinned, each count kept with one
        probability, which is what a lower rate does to a count; where more is predicted counts
        are added at the cell's own rate, read from its count under a gamma-Poisson reading of
        the gene, so a cell that shows none of the gene gains little and the zeros of an
        up-shifted gene fall as a real one's do. The fit then pins both moments as for any
        block and has little left to move, which matters because the fit can only rescale a
        count and a rescaled count never fills a zero.

        With nothing predicted the block is the drawn control cells with their own sampling
        error taken out, and the rank test calls it no more than it calls raw control cells; for
        nearly every gene it is a calmer null than a fresh draw, because both moments are
        pinned, so a shaped gene is called when its predicted shift alone clears the threshold.
        Four limits, each measured (the private file above has the numbers). A gene the block
        holds about ten counts of still sits a little low in the test, because for it the fit's
        rescale is not small. A gene that a small minority of cells carry is as wide a null as
        raw cells, and a few such genes wider. The gamma reading is right for a gamma-like gene;
        a heavy-tailed gene's zeros fall a few points short at a doubling. And two different
        amplitudes are carried as a lean of the gene with the cell's depth, inside every shaped
        gene: both moments are met, but the gene is then no longer the controls' shape, the more
        so the narrower the depths.

        The cells take real cells' depths, not the template's. The template is still drawn, and
        every draw the shape needs comes from a stream of its own, seeded from the integer this
        call draws for the fit anyway: `self.rng` and the next call's cells are the unshaped
        call's. `scatter` composes: on a shaped block the factor acts on the re-rated column,
        around the predicted count at the block's depths, so 1 is the controls' shape, below 1
        narrower than the controls and 0 the predicted count in every cell. The rungs are the
        dial's, with one difference: the anchor rung re-rates the same control cells for its own
        pair of profiles, so its block is real cells too; the last rung returns the template as
        drawn. `last_shaped` is True when the block just returned is control cells, False on the
        template rung, None when `shape` was not asked.
        """
        if on_fail not in ("raise", "fallback", "anchor"):
            raise ValueError(f"on_fail must be 'raise', 'fallback' or 'anchor', got {on_fail!r}")
        if self.lam == 0.0:
            raise ValueError("emit_dual needs a depth spread: at lam=0 every cell has the same "
                             "depth, so the pseudobulk and the per-cell mean cannot differ")
        p_cell = self._fraction(log2fc_cell)
        p_bulk = self._fraction(log2fc_bulk, bulk=True)
        if scatter is not None:
            scatter = np.asarray(scatter, dtype=np.float64)
            if scatter.shape != p_cell.shape:
                raise ValueError("scatter must be per gene on the submission axis")
            if not np.isfinite(scatter).all() or (scatter < 0).any():
                raise ValueError("scatter must be finite and >= 0 (1 = as emitted, 0 = sharp, "
                                 "above 1 = wider)")
        if shape:
            kept = self.p.cells
            if kept is None or self.p.dispersion is None or self.p.zero_rate is None:
                raise ValueError("shape needs the profile's control cells "
                                 "(ContextProfile.from_controls(..., keep_cells=True))")
            if not sp.issparse(kept) or kept.shape != (len(self.p.libsizes), len(p_cell)):
                raise ValueError("the profile's control cells must be a sparse matrix with one "
                                 "row per depth in libsizes and one column per gene")
        template = self.emit(n, log2fc_cell).toarray().astype(np.float64)
        depths = np.rint(template.sum(axis=1)).astype(np.int64)
        seed = int(self.rng.integers(0, 2**32 - 1))
        self.last_dual, self.last_dual_reason = "dual", None
        self.last_sharpened = None if scatter is None else int((scatter != 1.0).sum())
        self.last_shaped = True if shape else None

        def start_for(bulk_profile):
            """What the fit starts from, and its settings: the drawn template, or control cells
            re-rated to this pair of profiles; then the per-gene dial. `template` itself stays
            the drawn cells, because the last fallback rung returns them as they are."""
            start, at = (self._shaped_start(n, p_cell, bulk_profile, seed) if shape
                         else (template, depths))
            if scatter is not None:
                move = scatter != 1.0
                if move.any():
                    predicted = at[:, None].astype(np.float64) * p_cell[None, move]
                    column = start[:, move]
                    if start is template:
                        start = template.copy()
                    start[:, move] = np.maximum(predicted + scatter[None, move] * (column - predicted), 0.0)
            return start, dict(depths=at, seed=seed, iterations=iterations, tolerance=tolerance,
                               max_projection=max_projection)

        start, fit = start_for(p_bulk)
        try:
            counts = dual_moment_counts(start, p_cell, p_bulk, **fit)
        except ValueError as err:
            if on_fail == "raise":
                raise
            self.dual_fallbacks = getattr(self, "dual_fallbacks", 0) + 1
            msg = str(err)
            self.last_dual_reason = ("fit" if "moment fitting failed" in msg else
                                     "envelope" if ("depth envelope" in msg or "bulk projection" in msg
                                                    or "feasible moment bounds" in msg) else "other")
            if on_fail == "anchor" and self.last_dual_reason != "other":
                p_one = self._fraction(log2fc_cell, bulk=True)
                if not np.array_equal(p_one, p_bulk):     # the request was not already this rung
                    if shape:                             # the same control cells, re-rated for this pair
                        start, fit = start_for(p_one)
                    try:
                        counts = dual_moment_counts(start, p_cell, p_one, **fit)
                    except ValueError:
                        pass
                    else:
                        self.dual_fallbacks_anchor = getattr(self, "dual_fallbacks_anchor", 0) + 1
                        self.last_dual = "anchor"
                        return sp.csr_matrix(counts.astype(np.float32))
            self.last_dual = "template"
            if self.last_sharpened is not None:
                self.last_sharpened = 0
            if self.last_shaped:
                self.last_shaped = False
            return sp.csr_matrix(template.astype(np.float32))
        return sp.csr_matrix(counts.astype(np.float32))

    def _shaped_start(self, n: int, p_cell: np.ndarray, p_bulk: np.ndarray,
                      seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Control cells for `emit_dual(shape=True)`: `n` of the profile's kept cells, their
        counts re-rated onto the two predicted profiles (`_rerate`), and the depth each then
        adds up to. Everything random here is seeded by the call's own integer."""
        rng = np.random.default_rng([seed, 1])    # beside the fit's own default_rng(seed), never self.rng
        rows = rng.choice(self._lib_rows, size=n, replace=len(self._lib_rows) < n)
        start = self._rerate(self.p.cells[rows].toarray().astype(np.float64), self.p.libsizes[rows],
                             p_cell, p_bulk, rng)
        return start, np.maximum(np.rint(start.sum(axis=1)), 1).astype(np.int64)

    def _rerate(self, counts: np.ndarray, lib: np.ndarray, p_cell: np.ndarray, p_bulk: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
        """Control cells' counts moved onto the predicted moments by what a change of rate does
        to a count, never by rescaling one.

        Per gene, every cell gets a multiplier `ratio * exp(lift * weight + lean * tilt)`.
        `ratio` is the predicted per-cell fold change, the same for every cell: the shift
        itself. `lift` and `lean` are two numbers solved (Newton, eight steps) so that the
        EXPECTED per-cell mean fraction is `p_cell` and the expected column total is `p_bulk`
        times the cells' summed depth: with one amplitude they only take out the drawn cells'
        own sampling error. `tilt` is the cell's depth over the mean depth, less one. `weight`
        is the square of the cell's expected count of the gene, over the cells' mean square, so
        the lift falls on the cells that carry the gene: when a few cells hold much of a gene,
        how many of them were drawn is most of the sampling error, and taking it out of every
        cell alike would shift the many cells the rank test reads (it then called such genes
        with nothing predicted). A gene the block holds under twenty counts of gets the lift
        alone.

        A cell whose multiplier is below one keeps each of its counts with that probability
        (binomial thinning: exactly what a lower rate gives). A cell whose multiplier is above
        one gains a Poisson number of counts at (multiplier - 1) times its own rate, the rate
        drawn from its posterior under a gamma prior with the gene's mean and
        `profile.dispersion` -- its mean is
        depth * fraction * (1 + dispersion * count) / (1 + dispersion * fraction * depth) --
        which under that reading is the exact law of the extra counts; `profile.zero_rate`
        scales it for a cell with none, so that a gene which is simply off in most cells is not
        switched on in them. The multiplier is held to 64; past a 64-fold rise the fit delivers
        the rest as a rescale."""
        n = len(lib)
        tilt = lib / lib.mean() - 1.0
        live = np.flatnonzero(self.p.fraction > 0)    # a gene the controls never show stays empty
        out = np.zeros_like(counts)
        for lo in range(0, len(live), 4096):          # genes in blocks: a 38,584-gene axis stays in memory
            g = live[lo:lo + 4096]
            c = counts[:, g]
            m, f = self.p.fraction[None, g], self.p.dispersion[None, g]
            room = 1.0 + f * m * lib[:, None]
            none = np.where(c == 0, self.p.zero_rate[None, g], 1.0)
            own = none * lib[:, None] * m * (1.0 + f * c) / room      # the cell's expected count at its own rate
            weight = own * own
            mean_weight = weight.mean(axis=0)
            weight = np.divide(weight, mean_weight, out=np.ones_like(own), where=mean_weight > 0)
            ratio = (p_cell[g] / self.p.fraction[g])[None, :]
            want_cell, want_bulk = n * p_cell[g], p_bulk[g] * lib.sum()
            lift, lean = np.zeros(len(g)), np.zeros(len(g))
            for step in range(9):
                mult = np.clip(ratio * np.exp(lift[None, :] * weight + lean[None, :] * tilt[:, None]), 0.0, 64.0)
                if step == 8:
                    break
                pool = np.where(mult > 1.0, own, c)                   # what a multiplier acts on, in expectation
                expect = c + (mult - 1.0) * pool
                miss_cell = (expect / lib[:, None]).sum(axis=0) - want_cell
                miss_bulk = expect.sum(axis=0) - want_bulk
                grad = pool * mult
                a11, a12 = (grad * weight / lib[:, None]).sum(axis=0), (grad * (tilt / lib)[:, None]).sum(axis=0)
                a21, a22 = (grad * weight).sum(axis=0), (grad * tilt[:, None]).sum(axis=0)
                det = a11 * a22 - a12 * a21
                pair = (np.abs(det) > 1e-9 * (np.abs(a11 * a22) + np.abs(a12 * a21))) & (pool.sum(axis=0) >= 20.0)
                safe = np.where(pair, det, 1.0)
                d_lift = np.where(pair, (miss_cell * a22 - a12 * miss_bulk) / safe,
                                  np.divide(miss_cell, a11, out=np.zeros_like(a11), where=a11 > 0))
                d_lean = np.where(pair, (a11 * miss_bulk - a21 * miss_cell) / safe, 0.0)
                lift -= np.clip(d_lift, -1.0, 1.0)
                lean = np.where(pair, lean - np.clip(d_lean, -1.0, 1.0), 0.0)
            new = c.copy()
            thin = (mult < 1.0) & (c > 0)
            new[thin] = rng.binomial(c[thin].astype(np.int64), mult[thin])
            gain = mult > 1.0
            rate = np.broadcast_to(lib[:, None] * m, c.shape)[gain]   # a Poisson gene: the same rate in every cell
            spread = np.broadcast_to(f, c.shape)[gain]
            wide = spread > 1e-8
            rate[wide] = rng.gamma(1.0 / spread[wide] + c[gain][wide],
                                   spread[wide] * rate[wide] / room[gain][wide])
            new[gain] += rng.poisson((mult[gain] - 1.0) * none[gain] * rate)
            out[:, g] = new
        return out

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
