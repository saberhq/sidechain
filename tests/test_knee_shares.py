"""`sidechain.submit.knee`: each knockdown's part of cell-eval2's `mse` budget and the largest-share record.

    .venv/bin/python -m pytest -q tests/test_knee_shares.py

Writes nothing. Notes: private/code-notes/tests/test_knee_shares.py.md
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sidechain.submit import knee

G = 40
GENES = [f"G{i}" for i in range(G)]


def _sums(n=12, seed=0, big=None, scale=1.0):
    """Summed counts for `n` knockdowns, the first `n // 2` named after panel genes; `big` gets a large change."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(5.0, 200.0, size=G)
    targets = GENES[: n // 2] + [f"X{i}" for i in range(n - n // 2)]
    rows = []
    for t in targets:
        sd = 1.5 * scale if t == big else 0.3 * scale
        rows.append(base * np.exp(rng.normal(0, sd, size=G)) * 400)
    return targets, np.asarray(rows)


def _scorer_budget(pm, ex, w):
    from cell_eval2.metrics import delta as ce
    return float(ce._across_pert_budget(pm, ex, w))


def test_the_parts_add_up_to_the_scorers_budget():
    targets, sums = _sums()
    budget, w, parts, drops = knee.budget_parts(targets, GENES, sums)
    assert budget == knee.budget_and_weights(targets, GENES, sums)[0]
    assert (parts >= 0).all() and parts.sum() == pytest.approx(budget, rel=1e-10)
    assert (drops >= parts).all() and (drops <= parts * (1 + 2 / len(targets))).all()


@pytest.mark.parametrize("row", [0, 3, 7, 11])
def test_a_drop_is_the_budget_lost_when_that_row_sits_at_the_panel_mean(row):
    """The scorer's own budget with one row set to each gene's weighted mean over the rows it keeps."""
    targets, sums = _sums()
    pm, ex, w = knee._scorer_inputs(targets, GENES, sums)
    w = np.asarray(w, dtype=np.float64)
    budget, _, _, drops = knee.budget_parts(targets, GENES, sums)
    flat = pm.copy()
    for g in range(G):
        keep = [p for p in range(len(targets)) if ex.get(p) != g]
        flat[row, g] = np.dot(w[keep], pm[keep, g]) / w[keep].sum()
    assert _scorer_budget(flat, ex, w) == pytest.approx(budget - drops[row], rel=1e-9, abs=1e-15)


def test_the_largest_shares_are_listed_in_order_with_the_ratio_without_each():
    targets, sums = _sums(n=30, big="X3")
    budget, w, parts, drops = knee.budget_parts(targets, GENES, sums)
    claim = 0.5 * budget
    top = knee.top_shares(targets, budget, parts, drops, claim)
    assert len(top) == knee.SHARE_TOP and top[0]["knockdown"] == "X3"
    s = [x["share"] for x in top]
    assert s == sorted(s, reverse=True) and math.isclose(s[0], parts.max() / budget)
    for x in top:
        i = targets.index(x["knockdown"])
        assert x["ratio_without"] == pytest.approx((budget - drops[i]) / claim, rel=1e-12)
    assert len(knee.top_shares(targets[:4], budget, parts[:4], drops[:4], claim)) == 4


def test_a_meter_read_carries_the_shares():
    import scipy.sparse as sp
    rng = np.random.default_rng(3)
    targets = GENES[:8]
    meter = knee.KneeMeter(targets, GENES)
    for k, t in enumerate(targets):
        rate = rng.uniform(0.5, 5.0, size=G) * np.exp(rng.normal(0, 1.2 if k == 2 else 0.2, size=G))
        meter.add(t, sp.csr_matrix(rng.poisson(rate[None, :] * 20, size=(15, G)).astype(np.float32)))
    got = meter.read()
    assert got["shares"][0]["knockdown"] == targets[2]
    assert sum(x["share"] for x in got["shares"]) == pytest.approx(1.0, rel=1e-9)     # 8 knockdowns, all listed


def test_largest_share_estimates_the_strength_without_it_and_warns_at_the_level():
    read = {"alpha_bulk": 2.0, "ratio_budget_over_claim": 0.88,
            "shares": [{"knockdown": "K", "share": 0.4, "ratio_without": 0.528}]}
    big = knee.largest_share(read, ceiling=2.6)
    assert big["alpha_bulk_without"] == pytest.approx(2.0 * math.sqrt(0.88 / 0.528)) and big["warn"]
    assert knee.largest_share(read, ceiling=2.2)["alpha_bulk_without"] == 2.2              # never above the ceiling
    small = {**read, "shares": [{"knockdown": "K", "share": 0.05, "ratio_without": 0.836}]}
    assert not knee.largest_share(small, ceiling=2.6)["warn"]
    assert knee.largest_share(small, ceiling=2.6, warn=0.05)["warn"]
    at_ceiling = {"alpha_bulk": 2.6, "ratio_budget_over_claim": 0.5,
                  "shares": [{"knockdown": "K", "share": 0.4, "ratio_without": 0.3}]}
    assert knee.largest_share(at_ceiling, ceiling=2.6)["alpha_bulk_without"] == 2.6
