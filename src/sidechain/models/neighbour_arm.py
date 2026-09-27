"""The neighbour arm (T103): SER's own delta fused with its gene-table neighbours' mean delta.

This is the geometry gate's ``cross``-mode fusion (``scripts/esm2_geometry_gate.py``), moved
into the shipping path so `submit.build` and `eval.loco` can carry it as a knob. The gate
measured it; this module only replays it on pooled deltas. For each predicted target t:

  1. ``m``   the mean pooled delta over the neighbour POOL (a declared list of targets the
             sources measured), each member's own gene zeroed first: the shared response.
  2. ``r_t`` t's pooled delta minus ``m``, t's own gene zeroed: the residual SER carries.
  3. ``n_t`` the mean of ``d_j - m`` over the k pool members j nearest to t by cosine in the
             gene table, t itself excluded: the neighbour arm, which never reads t.
  4. out   ``m + |r_t| * unit(unit(r_t) + w * unit(n_t))``.

Why each step is the shape it is:

- **Residuals.** The gate scores residuals, and `pds` ranks each prediction against every other
  target's truth, so a component every target shares cannot tell them apart. ``m`` is added
  back unchanged, so the members that read levels (`mse`, `nmae`) see SER's.
- **Unit arms and w.** ``unit(r_t) + w * unit(n_t)`` is the gate's ``unit(ser) + w * unit(esm)``
  term for term, so a w read off the gate's sweep means here what it meant there: the cosine
  of ``out - m`` with any truth equals the gate's fused cosine (`tests/test_neighbour_arm.py`
  pins it against the gate's own functions).
- **SER's length.** The fused residual is rescaled to ``|r_t|``: the arm moves SER's direction
  and nothing else, so ``alpha`` and ``alpha_bulk`` keep the meaning they were calibrated with.
- **Own genes zeroed.** A neighbour's pooled delta carries its own knockdown (about -2 log2 at
  its own gene); averaged into t it would predict a silencing of gene j that t's knockdown does
  not cause. The gate zeroes it (``load_delta``); so does this. t's own gene is pinned to the
  knockdown value by the caller afterwards, as for every arm.
- **Never zero-filled.** A target the table lacks keeps SER's delta untouched; a pool member
  the table lacks, or no source covers, is left out of the pool. Both are counted. A zero
  vector would be a fake nearest neighbour of every other zero vector.
- **The pool is declared, never inferred.** The full X-Atlas artifacts cover ~18,300 targets and
  a label subset of them only its labels, so a pool read off "what the sources cover" would make
  one model differ between the box and the Mac. The caller passes the list.

Off is the default everywhere, and off means this module is never imported into the path:
w = 0 is not a fusion at weight zero, it is no fusion (`submit.build`, `eval.loco`).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sidechain.data.gene_aliases import RETIRED_SYMBOLS as ALIAS


def unit(v: np.ndarray) -> np.ndarray:
    """Row-wise (or whole-vector) L2 normalisation with the gate's 1e-12 guard."""
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def load_gene_table(path: str | Path) -> dict:
    """A symbol-keyed gene table (``{symbol: 1-d tensor}``, as every gate table is saved)."""
    import torch

    table = torch.load(Path(path).expanduser(), weights_only=False, map_location="cpu")
    if not isinstance(table, Mapping) or not table:
        raise ValueError(f"{path}: expected a non-empty {{symbol: vector}} dict, got "
                         f"{type(table).__name__}")
    return table


def table_vector(table: Mapping, label: str) -> np.ndarray | None:
    """``label``'s row, through the retired-symbol bridge the gate uses; None if absent.

    A row that is all zeros or carries a non-finite value counts as absent: its cosine with
    everything is 0 or NaN, so its "nearest neighbours" would be whatever the sort left first.
    """
    key = label if label in table else ALIAS.get(label)
    if key is None or key not in table:
        return None
    v = table[key]
    v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
    v = np.asarray(v, dtype=float).ravel()
    if not np.isfinite(v).all() or not v.any():
        return None
    return v


def read_pool(path: str | Path) -> list[str]:
    """Pool labels from a CSV: column ``target_gene`` if present, else the first; de-duplicated."""
    import pandas as pd

    df = pd.read_csv(Path(path).expanduser())
    col = "target_gene" if "target_gene" in df.columns else df.columns[0]
    return list(dict.fromkeys(df[col].astype(str).tolist()))


@dataclass
class NeighbourArm:
    """The fitted-free arm: a pool's residuals, their table rows, and the fusion knobs."""

    k: int
    w: float
    table: Mapping
    pool: list[str]                  # pool labels actually used, in pool-file order
    mean: np.ndarray                 # m, on the caller's axis
    resid: np.ndarray                # (n_pool, n_axis): d_j - m, own gene zeroed
    pool_unit: np.ndarray            # (n_pool, dim): unit table rows
    gene_pos: dict[str, int]         # axis symbol -> column
    stats: dict = field(default_factory=dict)

    def __post_init__(self):
        self._pool_pos = {lab: i for i, lab in enumerate(self.pool)}

    def fuse(self, target: str, delta: np.ndarray) -> np.ndarray:
        """SER's pooled delta for ``target`` -> the fused delta (a new array).

        A target the table cannot resolve, or whose residual is exactly zero, comes back
        unchanged (the same object) and is counted.
        """
        s = self.stats
        e = table_vector(self.table, target)
        if e is None:
            s["targets_unresolved"] = s.get("targets_unresolved", 0) + 1
            return delta
        d = np.array(delta, dtype=float)
        if target in self.gene_pos:
            d[self.gene_pos[target]] = 0.0
        r = d - self.mean
        r_norm = float(np.linalg.norm(r))
        if r_norm == 0.0:
            s["targets_zero_residual"] = s.get("targets_zero_residual", 0) + 1
            return delta
        sim = self.pool_unit @ unit(e)
        own = self._pool_pos.get(target)
        if own is not None:
            sim[own] = -np.inf               # a target is never its own neighbour
        idx = np.argsort(-sim)[: self.k]
        n = self.resid[idx].mean(0)
        ur, un = unit(r), unit(n)
        fused = self.mean + r_norm * unit(ur + self.w * un)
        s["targets_fused"] = s.get("targets_fused", 0) + 1
        s["arm_cosine_sum"] = s.get("arm_cosine_sum", 0.0) + float(ur @ un)
        s["own_in_pool"] = s.get("own_in_pool", 0) + int(own is not None)
        return fused

    def summary(self) -> dict:
        """What a run records: the knobs, the pool, and how the targets fared."""
        s = dict(self.stats)
        n = s.get("targets_fused", 0)
        if n:
            # mean cosine between SER's residual and the neighbour arm: how far they agree
            s["arm_cosine_mean"] = round(s.pop("arm_cosine_sum") / n, 6)
        else:
            s.pop("arm_cosine_sum", None)
        return {"k": self.k, "w": self.w, "pool_used": len(self.pool), **s}


def build_neighbour_arm(pool: Sequence[str], delta_of: Callable[[str], np.ndarray | None],
                        axis: np.ndarray, table: Mapping, *, k: int, w: float) -> NeighbourArm:
    """Pool the residuals once; ``delta_of(label)`` is the caller's pooled delta (None = uncovered).

    ``delta_of`` must be the SAME pooling the predicted targets get (sources, shrinkage,
    floors), or the pool's residuals are in a different space from SER's.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not (np.isfinite(w) and w > 0):
        raise ValueError(f"w must be finite and > 0 (w = 0 is the knob off: do not build the "
                         f"arm), got {w}")
    axis = np.asarray(axis).astype(str)
    gene_pos = {g: i for i, g in enumerate(axis)}
    labels, rows, vecs = [], [], []
    stats = {"pool_requested": 0, "pool_unresolved": 0, "pool_uncovered": 0}
    for lab in dict.fromkeys(str(x) for x in pool):
        stats["pool_requested"] += 1
        e = table_vector(table, lab)
        if e is None:
            stats["pool_unresolved"] += 1
            continue
        d = delta_of(lab)
        if d is None:
            stats["pool_uncovered"] += 1
            continue
        d = np.array(d, dtype=float)
        if lab in gene_pos:
            d[gene_pos[lab]] = 0.0               # the gate's load_delta: own gene zeroed
        labels.append(lab)
        rows.append(d)
        vecs.append(e)
    if len(labels) <= k:
        raise ValueError(f"the neighbour pool has {len(labels)} usable targets, which is not more "
                         f"than k = {k}: every target's neighbours would be the whole pool "
                         f"({stats})")
    dims = {v.shape[0] for v in vecs}
    if len(dims) != 1:
        raise ValueError(f"table rows differ in length: {sorted(dims)}")
    D = np.stack(rows)
    m = D.mean(0)
    return NeighbourArm(k=k, w=float(w), table=table, pool=labels, mean=m,
                        resid=np.ascontiguousarray(D - m), pool_unit=unit(np.stack(vecs)),
                        gene_pos=gene_pos, stats=stats)
