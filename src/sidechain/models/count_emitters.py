"""Emit predicted cells as raw integer counts around a context's control profile, shifted by a log2 fold change.

    profile = ContextProfile.from_controls("controls.h5ad", "ctx", min_libsize=CONTROL_MIN_LIBSIZE)
    cells = PoissonEmitter(profile, seed=0, lam=0.5).emit(400, log2fc)    # (cells x genes) CSR counts

Each gene's expected fraction is sampled at library sizes drawn from the context's own controls, as Poisson,
even or mixed counts (`PoissonEmitter`), or with two moments pinned (`emit_dual`). Reads a control h5ad in
`ContextProfile.from_controls`; writes nothing. Notes: private/code-notes/src/sidechain/models/count_emitters.py.md
"""
from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import scipy.sparse as sp

# control cells at or under this many UMI are dropped from every ContextProfile the pipeline builds;
# `from_controls` itself defaults to 0
CONTROL_MIN_LIBSIZE = 1000.0

#: `emit_dual(shape=True)`: no re-rated control cell leaves with more than SHAPE_DEPTH_GAIN_CAP times the
#: depth it came with, nor with more than SHAPE_MAX_COUNTS_PER_CELL counts (the scorer's max_counts_per_cell).
SHAPE_DEPTH_GAIN_CAP = 2.0
SHAPE_MAX_COUNTS_PER_CELL = 1_000_000
SHAPE_SOLVE_STEPS = 64
SHAPE_SOLVE_TOL = 1e-6


@dataclass
class ContextProfile:
    """What a context's control cells say about it: expected per-gene fractions
    and the empirical library-size distribution."""
    name: str
    genes: np.ndarray          # (G,) symbols, in submission order
    fraction: np.ndarray       # (G,) mean CPM / 1e6, sums to ~1
    libsizes: np.ndarray       # (n_cells,) UMI per control cell
    n_cells: int
    # (G,) the pooled profile, sum of counts / sum of depths over the same cells (cell-eval2's control
    # pseudobulk); None for a hand-built profile
    bulk_fraction: np.ndarray | None = None
    # (n_cells, G) the control cells' counts, CSR, rows in the order of `libsizes`, for
    # `emit_dual(shape=True)`; only with `from_controls(keep_cells=True)`
    cells: sp.csr_matrix | None = None
    # (G,) with `cells`: the squared CV of each cell's latent fraction of the gene beyond Poisson sampling
    # (a gamma's inverse shape), drawn toward the median over genes of >= 200 pooled counts at that weight
    dispersion: np.ndarray | None = None
    # (G,) with `cells`: the rate factor for a cell with no count of the gene: the cells with one count over
    # the gamma reading's expected count in the cells with none (Robbins), five counts' weight toward 1
    zero_rate: np.ndarray | None = None

    @classmethod
    def from_controls(cls, path, name: str, *, min_libsize: float = 0.0,
                      keep_cells: bool = False) -> ContextProfile:
        a = ad.read_h5ad(path)
        X = sp.csr_matrix(a.X, dtype=np.float64)
        lib = np.asarray(X.sum(axis=1)).ravel()
        keep = lib > max(min_libsize, 0)
        if not keep.any():
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
    """Integer cells around a (possibly shifted) context profile, at a cell-to-cell spread `lam` in [0, 1].

    lam=1 ("poisson"): Poisson counts at depths drawn from the controls; lam=0 ("even"): every cell at the median
    depth, each gene's total spread as evenly as possible; between, 1 - lam^2 of the counts even, lam^2 Poisson.
    Pass `dispersion` or `lam`, not both. `bulk_anchor` is `emit_dual`'s pseudobulk profile: "mean_cpm" or "pooled".
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
        self.last_shape_held, self.last_shape_unmet = None, None    # set by a shaped block (`_rerate`)
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
        if over.any():
            counts[over] = np.floor(counts[over] * (max_counts_per_cell / tot[over])[:, None])
        return sp.csr_matrix(counts)

    def emit_dual(self, n: int, log2fc_cell: np.ndarray | None, log2fc_bulk: np.ndarray | None, *,
                  iterations: int = 100, tolerance: float = 2e-4,
                  max_projection: float = 0.03, on_fail: str = "raise",
                  scatter: np.ndarray | None = None, shape: bool = False) -> sp.csr_matrix:
        """Cells whose equal-weight per-cell mean follows `log2fc_cell` and whose column sums follow `log2fc_bulk`.

        Starts from `self.emit(n, log2fc_cell)` (with `shape=True`, control cells re-rated by `_rerate`), `scatter`
        scaling each gene's spread around its predicted count; `dual_moment_counts` pins both moments. An unmet pair
        raises, returns the template ("fallback"), or first retries the per-cell amplitude on both channels ("anchor").
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
            """The fit's start (the template, or control cells re-rated to this pair), the per-gene dial applied."""
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
                self.last_shape_held, self.last_shape_unmet = None, None    # no re-rated cell on this rung
            return sp.csr_matrix(template.astype(np.float32))
        return sp.csr_matrix(counts.astype(np.float32))

    def _shaped_start(self, n: int, p_cell: np.ndarray, p_bulk: np.ndarray,
                      seed: int) -> tuple[np.ndarray, np.ndarray]:
        """`n` kept control cells, one per depth slice, re-rated onto the two profiles; and each one's depth."""
        rng = np.random.default_rng([seed, 1])    # beside the fit's own default_rng(seed), never self.rng
        rows = self._depth_spread_rows(n, rng)
        start = self._rerate(self.p.cells[rows].toarray().astype(np.float64), self.p.libsizes[rows],
                             p_cell, p_bulk, rng)
        return start, np.maximum(np.rint(start.sum(axis=1)), 1).astype(np.int64)

    def _depth_spread_rows(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """`n` distinct rows of `profile.cells`, one from each of `n` equal slices of the kept cells by depth;
        drawn with replacement when fewer than `n` are kept."""
        pool = self._lib_rows
        if len(pool) < n:
            return rng.choice(pool, size=n, replace=True)
        order = pool[np.argsort(self.p.libsizes[pool], kind="stable")]
        edge = np.floor(np.linspace(0.0, len(order), n + 1)).astype(np.int64)
        return order[edge[:-1] + np.floor(rng.random(n) * (edge[1:] - edge[:-1])).astype(np.int64)]

    def _rerate(self, counts: np.ndarray, lib: np.ndarray, p_cell: np.ndarray, p_bulk: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
        """Control cells' counts moved onto the predicted moments by thinning or adding counts, never by rescaling.

        Per gene each cell's multiplier is `ratio * exp(lift * weight + lean * tilt)`, clipped to [0, 64], with `lift`
        and `lean` solved by damped Newton for the expected per-cell mean `p_cell` and column total `p_bulk`; below 1
        counts are thinned, above 1 Poisson counts are added at the cell's gamma-posterior rate; caps as SHAPE_*."""
        n = len(lib)
        tilt = lib / lib.mean() - 1.0
        live = np.flatnonzero(self.p.fraction > 0)    # a gene the controls never show stays empty
        out = np.zeros_like(counts)
        unmet = 0
        for lo in range(0, len(live), 4096):          # genes in blocks, for memory
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
            # a miss is read relative to what is wanted; a gene predicted to nothing is met by its ratio alone
            per_cell = np.where(want_cell > 0, want_cell, np.inf)
            per_bulk = np.where(want_bulk > 0, want_bulk, np.inf)
            two = np.where(ratio > 1.0, own, c).sum(axis=0) >= 20.0   # under twenty counts: the lift alone
            at_lift, at_lean = np.zeros(len(g)), np.zeros(len(g))      # each gene's best point so far
            left = np.full(len(g), np.inf)                             # the larger relative miss it leaves
            go_lift, go_lean = np.zeros(len(g)), np.zeros(len(g))      # the Newton step from it
            trust = np.ones(len(g))
            act = np.arange(len(g))                                    # genes still being solved
            for _ in range(SHAPE_SOLVE_STEPS + 1):
                w, ca, oa = weight[:, act], c[:, act], own[:, act]
                lift, lean = at_lift[act] - trust[act] * go_lift[act], at_lean[act] - trust[act] * go_lean[act]
                mult = np.clip(ratio[:, act] * np.exp(lift[None, :] * w + lean[None, :] * tilt[:, None]), 0.0, 64.0)
                pool = np.where(mult > 1.0, oa, ca)                   # what a multiplier acts on, in expectation
                expect = ca + (mult - 1.0) * pool
                miss_cell = (expect / lib[:, None]).sum(axis=0) - want_cell[act]
                miss_bulk = expect.sum(axis=0) - want_bulk[act]
                miss = np.abs(miss_cell) / per_cell[act]
                miss = np.where(two[act], np.maximum(miss, np.abs(miss_bulk) / per_bulk[act]), miss)
                won = miss < left[act]
                grad = pool * mult
                a11, a12 = (grad * w / lib[:, None]).sum(axis=0), (grad * (tilt / lib)[:, None]).sum(axis=0)
                a21, a22 = (grad * w).sum(axis=0), (grad * tilt[:, None]).sum(axis=0)
                det = a11 * a22 - a12 * a21
                pair = two[act] & (np.abs(det) > 1e-9 * (np.abs(a11 * a22) + np.abs(a12 * a21)))
                safe = np.where(pair, det, 1.0)
                d_lift = np.where(pair, (miss_cell * a22 - a12 * miss_bulk) / safe,
                                  np.divide(miss_cell, a11, out=np.zeros_like(a11), where=a11 > 0))
                d_lean = np.where(pair, (a11 * miss_bulk - a21 * miss_cell) / safe, 0.0)
                far = np.maximum(np.abs(d_lift[None, :] * w + d_lean[None, :] * tilt[:, None]).max(axis=0), 1.0)
                k = act[won]
                at_lift[k], at_lean[k], left[k] = lift[won], lean[won], miss[won]
                go_lift[k], go_lean[k] = (d_lift / far)[won], (d_lean / far)[won]
                trust[act] = np.where(won, np.minimum(2.0 * trust[act], 1.0), 0.5 * trust[act])
                act = act[(left[act] > SHAPE_SOLVE_TOL) & (trust[act] > 1e-3)]
                if len(act) == 0:
                    break
            mult = np.clip(ratio * np.exp(at_lift[None, :] * weight + at_lean[None, :] * tilt[:, None]), 0.0, 64.0)
            unmet += int(((left > 0.01) & two).sum())
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
        # no cell leaves with more than SHAPE_DEPTH_GAIN_CAP times the depth it came with, nor over the scorer's limit
        total = out.sum(axis=1)
        ceiling = np.minimum(SHAPE_DEPTH_GAIN_CAP * lib, float(SHAPE_MAX_COUNTS_PER_CELL))
        over = np.flatnonzero(total > ceiling)
        for i in over:
            own_kept = np.minimum(out[i], counts[i]).astype(np.int64)      # the cell's own counts that survived
            added = out[i].astype(np.int64) - own_kept
            room = int(ceiling[i]) - int(own_kept.sum())
            if room >= 0:                             # that many of the added counts, drawn without replacement
                out[i] = own_kept + rng.multivariate_hypergeometric(added, room, method="marginals")
            else:                                     # a control cell itself over the limit: that many of its own
                out[i] = rng.multivariate_hypergeometric(own_kept, int(ceiling[i]), method="marginals")
        self.last_shape_held = int(len(over))
        self.last_shape_unmet = unmet
        return out

    def _emit_even(self, n: int, frac: np.ndarray, depth_frac: float = 1.0) -> np.ndarray:
        # per-gene total over n cells; `depth_frac` is the even share of a lam mixture (1.0 = the whole depth)
        total = np.rint(n * self._lib_median * depth_frac * frac).astype(np.int64)
        base, rem = np.divmod(total, n)
        counts = np.broadcast_to(base.astype(np.float32), (n, len(frac))).copy()
        cols = np.where(rem > 0)[0]
        for j in cols:  # the remainder as +1 on a random subset of cells
            counts[self.rng.choice(n, size=int(rem[j]), replace=False), j] += 1.0
        return counts


def repair_bulk_totals(integer: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Move single counts between cells within each column until its total equals its rounded expectation,
    no cell's depth changing. `integer` is floor(expected) plus a per-row residual scatter; an excess column gives
    up counts from the cells that rounded up most, a short one takes them onto the cells furthest below.
    After kaipengm2 (MIT). In place on `integer`; returned."""
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
    """Integer cells whose per-cell mean composition is `probability` and pseudobulk is `bulk_probability`.

    The bulk is projected onto the depth envelope `[z_min p, z_max p]` (z = depth / mean depth), then each gene's
    column is tilted by `1 + t_g (z_i - m_g)` alternating with a column rescale until both hold to `tolerance`
    (L1); rounding keeps each row's depth and `repair_bulk_totals` pins the column totals. After kaipengm2 (MIT).
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
    """log2((pert + pseudocount) / (ctrl + pseudocount)) of arithmetic-mean CPM."""
    return np.log2((mean_cpm_pert + pseudocount) / (mean_cpm_ctrl + pseudocount))


def remap_to_axis(values: np.ndarray, source_genes: np.ndarray, target_genes: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """A per-gene vector moved from a source gene axis onto the submission axis by symbol; absent genes get `fill`."""
    pos = {g: i for i, g in enumerate(source_genes)}
    out = np.full(len(target_genes), fill, dtype=np.float64)
    for j, g in enumerate(target_genes):
        i = pos.get(g)
        if i is not None:
            out[j] = values[i]
    return out
