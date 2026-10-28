"""Adaptive shrinkage of fold changes: the model of ashr (Stephens 2017), written in numpy.

    shrunk = adaptive_shrink(fc, var)

Per gene, x_g | b_g ~ N(b_g, s_g^2) and b_g ~ sum_k pi_k N(0, sd_k^2) over a fixed grid of sd_k, no point mass at
zero; the weights pi_k are fitted by SQUAREM-accelerated EM, stopped on a plateau (`TOL_PER_GENE` per gene for
`CALM_CYCLES` cycles) or at `MAX_CYCLES`, and the posterior mean of b_g is returned. Written from Stephens, "False
discovery rates: a new deal", Biostatistics 18(2):275-294, 2017, doi 10.1093/biostatistics/kxw041.
Reads and writes nothing. Notes: private/code-notes/src/sidechain/models/adaptive_shrink.py.md
"""
from __future__ import annotations

import numpy as np

MAX_CYCLES = 5000        # SQUAREM cycles at most
TOL_PER_GENE = 1e-9      # log-likelihood gain per gene per cycle that counts as no gain
CALM_CYCLES = 3          # that many quiet cycles running end the fit
MIN_GENES = 50           # fewer usable genes than this is no basis for a prior: left unshrunk
MAX_COMPONENTS = 200     # a longer prior grid is refused as a broken variance


def prior_grid(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """The prior's standard deviations: s.min()/10, times sqrt(2), up to 2*sqrt(max(x^2 - s^2))."""
    lo = float(s.min()) / 10.0
    with np.errstate(over="ignore"):
        m2 = float(np.max(x * x - s * s))
    hi = 2.0 * np.sqrt(m2) if m2 > 0 else 8.0 * lo
    hi = max(hi, 2.0 * lo)
    if not (lo > 0 and np.isfinite(hi)) or hi / lo > 2.0 ** (MAX_COMPONENTS / 2):
        # a standard error near zero or an estimate near overflow would ask for hundreds of components
        raise ValueError(f"adaptive_shrink: the prior grid from {lo:g} to {hi:g} is not usable "
                         f"(more than {MAX_COMPONENTS} components): check the variances")
    n = int(np.ceil(np.log2(hi / lo) * 2.0)) + 1
    return lo * np.sqrt(2.0) ** np.arange(n)


def likelihood(x: np.ndarray, s: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """(genes x components) N(x_g; 0, s_g^2 + sd_k^2), each row divided by its maximum.

    The row scale cancels in both the weight fit and the posterior, and dividing by it keeps
    every row's largest entry at 1 so no row underflows to all zeros.
    """
    v = s[:, None] ** 2 + sd[None, :] ** 2
    ll = -0.5 * (np.log(v) + (x * x)[:, None] / v)
    ll -= ll.max(axis=1, keepdims=True)
    return np.exp(ll)


def fit_weights(lik: np.ndarray, *, max_cycles: int = MAX_CYCLES,
                tol_per_gene: float = TOL_PER_GENE) -> tuple[np.ndarray, int]:
    """Mixture weights maximising sum_g log(lik @ pi), from a uniform start; (pi, cycles run).

    One SQUAREM cycle is two EM steps and an extrapolated third, kept only when it stays on
    the simplex and does not lower the likelihood, so no cycle is worse than two plain steps.
    """
    n, k = lik.shape

    def step(p):
        nk = p * (lik.T @ (1.0 / (lik @ p)))
        return nk / nk.sum()

    def obj(p):
        return float(np.log(lik @ p).sum())

    pi = np.full(k, 1.0 / k)
    o_prev, calm, it = -np.inf, 0, 0
    for it in range(max_cycles):
        p1 = step(pi)
        p2 = step(p1)
        r = p1 - pi
        v = (p2 - p1) - r
        nv = float(np.sqrt((v * v).sum()))
        cand, o_cand = p2, obj(p2)
        if nv > 0:
            a = min(-float(np.sqrt((r * r).sum())) / nv, -1.0)
            pa = pi - 2.0 * a * r + a * a * v
            if pa.min() >= 0:
                pa = step(pa / pa.sum())
                o_pa = obj(pa)
                if o_pa >= o_cand:
                    cand, o_cand = pa, o_pa
        calm = calm + 1 if (o_cand - o_prev) < tol_per_gene * n else 0
        pi, o_prev = cand, o_cand
        if calm >= CALM_CYCLES:
            break
    return pi, it + 1


def posterior_mean(x: np.ndarray, s: np.ndarray, sd: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """E[b_g | x_g] under the fitted prior: x_g times a weight in (0, 1]."""
    w = likelihood(x, s, sd) * pi[None, :]
    w /= w.sum(axis=1, keepdims=True)
    s2 = (s * s)[:, None]
    g2 = (sd * sd)[None, :]
    return x * (w * (g2 / (g2 + s2))).sum(axis=1)


def adaptive_shrink(fc: np.ndarray, var: np.ndarray, *, stats: dict | None = None,
                    max_cycles: int = MAX_CYCLES) -> np.ndarray:
    """Posterior-mean shrinkage of `fc` given its sampling variance `var`, per gene.

    Genes with a non-finite or non-positive variance, or a non-finite estimate, are returned untouched; with
    fewer than `MIN_GENES` usable genes nothing is shrunk. `stats`, if given, accumulates the fits, their cycles,
    the fits that stopped at `max_cycles` and the vectors too thin to fit.
    """
    fc = np.asarray(fc, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    use = np.isfinite(var) & np.isfinite(fc) & (var > 0)
    if int(use.sum()) < MIN_GENES:
        if stats is not None:
            stats["adaptive_too_few_genes"] = stats.get("adaptive_too_few_genes", 0) + 1
        return fc.copy()
    x, s = fc[use], np.sqrt(var[use])
    sd = prior_grid(x, s)
    pi, cycles = fit_weights(likelihood(x, s, sd), max_cycles=max_cycles)
    if stats is not None:
        stats["adaptive_fits"] = stats.get("adaptive_fits", 0) + 1
        stats["adaptive_cycles"] = stats.get("adaptive_cycles", 0) + cycles
        stats["adaptive_fits_at_cap"] = stats.get("adaptive_fits_at_cap", 0) + int(cycles >= max_cycles)
    out = fc.copy()
    out[use] = posterior_mean(x, s, sd, pi)
    return out
