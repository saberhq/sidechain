"""The `mse` refund share of a context's emitted cells: cell-eval2's budget over its claim (T84).

cell-eval2 subtracts a sampling correction from `mse`, the CLAIM (the delete-1 jackknife of each knockdown's
emitted cells, weighted per row and summed), and caps it at the BUDGET (the across-knockdown centred sum of
squares of the predicted summed profiles, `bulk_lognorm_means` at target sum 50,000, each row's own gene left
out). `ratio = budget / claim`; below 1 the refund is that share of the claim.

The arithmetic is cell-eval2's own (`metrics.delta._exclusion_cols`, `_row_weights`, `_across_pert_budget`,
`moments.jackknife_correction`, `prep.bulk_lognorm_means`), as
`~/data/sidechain/runs/knobs/t84_stack_20261019/box/entry/knee_built.py` reads it off a built file; here it
is read block by block as the cells are emitted, so nothing is written to disk.

    meter = KneeMeter(targets, genes)
    for t in targets: meter.add(t, block_of(t))      # a (cells x genes) count matrix per knockdown
    meter.read()     # {"budget", "claim_uncapped", "ratio_budget_over_claim", "knockdowns"}

Used by `sidechain.submit.build --alpha-bulk-target` (`next_alpha_bulk` is its step). Task: private/tasks/T84.md.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

TARGET_SUM = 50_000.0   # cell-eval2's bulk_lognorm target sum
JK_CHUNK = 64           # cell-eval2's jackknife block size as knee_built.py calls it (a cost knob, not an answer knob)


def block_stats(block) -> tuple[np.ndarray, float]:
    """One knockdown's cells: their column sums (float64) and cell-eval2's delete-1 jackknife claim."""
    from cell_eval2.moments import jackknife_correction

    X = sp.csr_matrix(block, dtype=np.float64)
    sums = np.asarray(X.sum(axis=0), dtype=np.float64).ravel()
    jk = float(jackknife_correction(X, np.zeros(X.shape[0], dtype=np.intp), 1, TARGET_SUM, chunk=JK_CHUNK)[0])
    return sums, jk


def budget_and_weights(targets, genes, sums: np.ndarray) -> tuple[float, np.ndarray]:
    """cell-eval2's budget on the knockdowns' summed counts, and its per-row weights (1 / scored genes)."""
    from cell_eval2.metrics import delta as ce
    from cell_eval2.prep import bulk_lognorm_means

    labels = [str(t) for t in targets]
    genes = np.asarray(genes).astype(str)
    pm = bulk_lognorm_means(np.asarray(sums, dtype=np.float64), TARGET_SUM)
    ex = ce._exclusion_cols(labels, genes, target_gene_map=None, gate_labels=labels, who="knee",
                            n_genes=len(genes))
    w = ce._row_weights(len(labels), len(genes), ex)
    return float(ce._across_pert_budget(pm, ex, w)), w


def knee_read(targets, genes, sums: np.ndarray, jk: np.ndarray) -> dict:
    """Budget, claim (uncapped) and their ratio from per-knockdown column sums and jackknife claims."""
    budget, w = budget_and_weights(targets, genes, sums)
    claim = float((w * np.asarray(jk, dtype=np.float64)).sum())
    ratio = budget / claim if claim > 0 else math.inf
    return {"knockdowns": len(targets), "budget": budget, "claim_uncapped": claim,
            "ratio_budget_over_claim": ratio}


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
    """The next summed-profile strength: alpha_bulk * sqrt((1 - margin) * target / ratio), never below `floor`.

    The ratio grows about as the square of the strength (A/B/C, 1.35 to 2.00: an exponent of 1.96), so the step
    lands near the aim; with an exponent under 2 it lands a little above it, which the margin absorbs.
    """
    if not (0.0 < target and ratio > 0 and math.isfinite(ratio) and 0.0 <= margin < 1.0):
        raise ValueError(f"next_alpha_bulk: target {target!r}, ratio {ratio!r} and margin {margin!r} out of range")
    return max(float(floor), float(alpha_bulk) * math.sqrt((1.0 - margin) * target / ratio))
