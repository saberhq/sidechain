"""The ratio of cell-eval2's `mse` correction budget to its claim, read on a context's emitted cells.

    meter = KneeMeter(targets, genes)
    for t in targets: meter.add(t, block_of(t))      # a (cells x genes) count matrix per knockdown
    meter.read()     # {"budget", "claim_uncapped", "ratio_budget_over_claim", "knockdowns", "shares"}

Reads the blocks it is given; writes nothing. Notes: private/code-notes/src/sidechain/submit/knee.py.md
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

TARGET_SUM = 50_000.0   # cell-eval2's bulk_lognorm target sum
JK_CHUNK = 64           # cell-eval2's jackknife block size
SHARE_TOP = 10          # knockdowns listed per read, largest share of the budget first
SHARE_WARN = 0.20       # a written read whose largest share of the budget is at least this prints a warning


def block_stats(block) -> tuple[np.ndarray, float]:
    """One knockdown's cells: their column sums (float64) and cell-eval2's delete-1 jackknife claim."""
    from cell_eval2.moments import jackknife_correction

    X = sp.csr_matrix(block, dtype=np.float64)
    sums = np.asarray(X.sum(axis=0), dtype=np.float64).ravel()
    jk = float(jackknife_correction(X, np.zeros(X.shape[0], dtype=np.intp), 1, TARGET_SUM, chunk=JK_CHUNK)[0])
    return sums, jk


def _scorer_inputs(targets, genes, sums: np.ndarray):
    """cell-eval2's log-normalised summed profiles, own-gene exclusions and per-row weights."""
    from cell_eval2.metrics import delta as ce
    from cell_eval2.prep import bulk_lognorm_means

    labels = [str(t) for t in targets]
    genes = np.asarray(genes).astype(str)
    pm = bulk_lognorm_means(np.asarray(sums, dtype=np.float64), TARGET_SUM)
    ex = ce._exclusion_cols(labels, genes, target_gene_map=None, gate_labels=labels, who="knee",
                            n_genes=len(genes))
    w = ce._row_weights(len(labels), len(genes), ex)
    return pm, ex, w


def budget_and_weights(targets, genes, sums: np.ndarray) -> tuple[float, np.ndarray]:
    """cell-eval2's budget on the knockdowns' summed counts, and its per-row weights (1 / scored genes)."""
    from cell_eval2.metrics import delta as ce

    pm, ex, w = _scorer_inputs(targets, genes, sums)
    return float(ce._across_pert_budget(pm, ex, w)), w


def budget_parts(targets, genes, sums: np.ndarray,
                 chunk: int = 256) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """cell-eval2's budget, its per-row weights, each knockdown's part of it, and the budget each would remove.

    A part is the row's weighted squared distance from each gene's weighted mean over the rows that gene keeps,
    own gene left out; the parts sum to the budget. A drop is how much the budget falls when that row is set to
    those means.
    """
    from cell_eval2.metrics import delta as ce

    pm, ex, w = _scorer_inputs(targets, genes, sums)
    budget = float(ce._across_pert_budget(pm, ex, w))
    w = np.asarray(w, dtype=np.float64)
    dev = pm - pm.mean(axis=0)
    s1 = w @ dev
    wsum = np.full(dev.shape[1], float(w.sum()))
    cnt = np.full(dev.shape[1], float(dev.shape[0]))
    for k, col in ex.items():
        s1[col] -= w[k] * dev[k, col]
        wsum[col] -= w[k]
        cnt[col] -= 1.0
    ok = (cnt >= 2.0) & (wsum > 0.0)
    m = np.zeros_like(s1)
    m[ok] = s1[ok] / wsum[ok]
    parts = np.empty(dev.shape[0], dtype=np.float64)
    drops = np.empty(dev.shape[0], dtype=np.float64)
    inv = 1.0 / wsum[ok]
    for a in range(0, dev.shape[0], chunk):
        d2 = (dev[a:a + chunk] - m)[:, ok] ** 2
        wa = w[a:a + chunk]
        parts[a:a + chunk] = wa * d2.sum(axis=1)
        drops[a:a + chunk] = wa * (d2 * (1.0 + wa[:, None] * inv[None, :])).sum(axis=1)
    for k, col in ex.items():
        if ok[col]:
            d2 = (dev[k, col] - m[col]) ** 2
            parts[k] -= w[k] * d2
            drops[k] -= w[k] * d2 * (1.0 + w[k] / wsum[col])
    return budget, w, parts, drops


def top_shares(targets, budget: float, parts: np.ndarray, drops: np.ndarray, claim: float,
               top: int = SHARE_TOP) -> list[dict]:
    """The `top` knockdowns by share of the budget, each with the ratio read with that row at the panel's means."""
    order = np.argsort(-np.asarray(parts), kind="stable")[:top]
    return [{"knockdown": str(targets[i]),
             "share": float(parts[i] / budget) if budget > 0 else 0.0,
             "ratio_without": float(max(budget - drops[i], 0.0) / claim) if claim > 0 else math.inf}
            for i in order]


def knee_read(targets, genes, sums: np.ndarray, jk: np.ndarray) -> dict:
    """Budget, claim (uncapped), their ratio and the largest shares, from per-knockdown column sums and claims."""
    budget, w, parts, drops = budget_parts(targets, genes, sums)
    claim = float((w * np.asarray(jk, dtype=np.float64)).sum())
    ratio = budget / claim if claim > 0 else math.inf
    return {"knockdowns": len(targets), "budget": budget, "claim_uncapped": claim,
            "ratio_budget_over_claim": ratio, "shares": top_shares(targets, budget, parts, drops, claim)}


def largest_share(read: dict, ceiling: float, warn: float | None = None) -> dict:
    """A read's largest share, the ratio without it, the strength that ratio allows (at most `ceiling`), and
    whether it warns (at `SHARE_WARN` unless `warn` is given)."""
    warn = SHARE_WARN if warn is None else warn
    top = read["shares"][0]
    a, r, r0 = float(read["alpha_bulk"]), float(read["ratio_budget_over_claim"]), top["ratio_without"]
    without = float(ceiling) if not (r0 > 0 and math.isfinite(r)) else min(float(ceiling), a * math.sqrt(r / r0))
    return {**top, "alpha_bulk": a, "alpha_bulk_without": without, "warn": bool(top["share"] >= warn)}


class KneeMeter:
    """Accumulates one context's knockdown blocks; `read()` is `knee_read` on them.

    `jackknife=False` keeps the column sums only (the build's write pass checks them against its probe).
    """

    def __init__(self, targets, genes, *, jackknife: bool = True):
        self.targets = [str(t) for t in targets]
        self.genes = np.asarray(genes).astype(str)
        self._row = {t: i for i, t in enumerate(self.targets)}
        if len(self._row) != len(self.targets):
            raise ValueError("KneeMeter: a target is listed twice")
        self.jackknife = jackknife
        self.sums = np.zeros((len(self.targets), len(self.genes)), dtype=np.float64)
        self.jk = np.zeros(len(self.targets), dtype=np.float64)
        self.seen = np.zeros(len(self.targets), dtype=bool)

    def add(self, target, block) -> None:
        i = self._row[str(target)]
        if self.seen[i]:
            raise ValueError(f"KneeMeter: {target!r} added twice")
        if self.jackknife:
            self.sums[i], self.jk[i] = block_stats(block)
        else:
            X = sp.csr_matrix(block, dtype=np.float64)
            self.sums[i] = np.asarray(X.sum(axis=0), dtype=np.float64).ravel()
        self.seen[i] = True

    def read(self) -> dict:
        if not self.jackknife:
            raise ValueError("KneeMeter: built with jackknife=False, so it holds no claim")
        if not self.seen.all():
            missing = [t for t, s in zip(self.targets, self.seen) if not s]
            raise ValueError(f"KneeMeter: no block for {len(missing)} target(s), e.g. {missing[:3]}")
        return knee_read(self.targets, self.genes, self.sums, self.jk)


AIM_MARGIN = 0.02       # a lowered strength aims at (1 - AIM_MARGIN) * target, so it lands at or under the target


def next_alpha_bulk(alpha_bulk: float, ratio: float, target: float, floor: float,
                    margin: float = AIM_MARGIN) -> float:
    """The next summed-profile strength: alpha_bulk * sqrt((1 - margin) * target / ratio), never below `floor`."""
    if not (0.0 < target and ratio > 0 and math.isfinite(ratio) and 0.0 <= margin < 1.0):
        raise ValueError(f"next_alpha_bulk: target {target!r}, ratio {ratio!r} and margin {margin!r} out of range")
    return max(float(floor), float(alpha_bulk) * math.sqrt((1.0 - margin) * target / ratio))


UNDERSHOOT = 0.10       # a read under (1 - UNDERSHOOT) * target, below the ceiling, is stepped back up


def bracket_alpha_bulk(lo: tuple[float, float], hi: tuple[float, float], target: float,
                       margin: float = AIM_MARGIN) -> float:
    """The strength between two reads, `lo` under the target and `hi` over it, each (strength, ratio)."""
    (a_lo, r_lo), (a_hi, r_hi) = lo, hi
    if not (0.0 < a_lo < a_hi and 0.0 < r_lo <= target < r_hi and math.isfinite(r_hi)):
        raise ValueError(f"bracket_alpha_bulk: {lo!r} and {hi!r} do not straddle the target {target!r}")
    k = math.log(r_hi / r_lo) / math.log(a_hi / a_lo)
    a = a_lo * ((1.0 - margin) * target / r_lo) ** (1.0 / k)
    pad = (a_hi - a_lo) / 20.0
    return min(max(a, a_lo + pad), a_hi - pad)


def knee_next(reads: list[tuple[float, float]], target: float, floor: float, ceiling: float, max_reads: int,
              margin: float = AIM_MARGIN, undershoot: float = UNDERSHOOT) -> float | None:
    """The next summed-profile strength after `reads` ((strength, ratio), the ceiling's first); None to stop."""
    a, r = reads[-1]
    if len(reads) >= max_reads or not math.isfinite(r):
        return None
    if r <= target:
        if r >= (1.0 - undershoot) * target or a >= ceiling:
            return None
        over = [x for x in reads if x[1] > target and x[0] > a and math.isfinite(x[1])]
        return None if not over else bracket_alpha_bulk((a, r), min(over), target, margin)
    if a <= floor:
        return None
    under = [x for x in reads if x[1] <= target and x[0] < a]
    return (bracket_alpha_bulk(max(under), (a, r), target, margin) if under
            else next_alpha_bulk(a, r, target, floor, margin))


def knee_keep(reads: list[tuple[float, float]], target: float) -> int:
    """The read the knee rule writes: the strongest read at or under the target, else the last."""
    met = [i for i, (_a, r) in enumerate(reads) if r <= target]
    return max(met, key=lambda i: reads[i][0]) if met else len(reads) - 1
