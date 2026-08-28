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
