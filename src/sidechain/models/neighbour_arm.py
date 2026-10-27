"""The neighbour arm: a target's pooled delta fused with the mean residual of its gene-table neighbours.

    arm = build_neighbour_arm(pool, delta_of, axis, table, k=k, w=w)    # size, select, cand, picks optional
    fused = arm.fuse(target, delta)

Per target t: m the pool's mean delta, r_t = d_t - m, n_t the mean d_j - m over the k picked pool members
(t excluded; own genes zeroed); out = m + |r_t| * unit(unit(r_t) + w * a_t), with a_t = unit(n_t) under
size="unit" or n_t / s under size="median" (s the pool's median residual length).
Reads a gene table and a pool CSV; writes nothing. Notes: private/code-notes/src/sidechain/models/neighbour_arm.py.md
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sidechain.data.gene_aliases import RETIRED_SYMBOLS as ALIAS

SIZES = ("unit", "median")       # blend shapes
# which pool members are averaged; "constant" averages none (every target's arm is -m)
SELECTS = ("table", "hybrid", "response", "euclid", "constant")


def unit(v: np.ndarray) -> np.ndarray:
    """Row-wise (or whole-vector) L2 normalisation, guarded by 1e-12."""
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def load_gene_table(path: str | Path) -> dict:
    """A symbol-keyed gene table, ``{symbol: 1-d tensor}``."""
    import torch

    table = torch.load(Path(path).expanduser(), weights_only=False, map_location="cpu")
    if not isinstance(table, Mapping) or not table:
        raise ValueError(f"{path}: expected a non-empty {{symbol: vector}} dict, got "
                         f"{type(table).__name__}")
    return table


def table_vector(table: Mapping, label: str) -> np.ndarray | None:
    """``label``'s row, through the retired-symbol bridge; None if absent, all zeros or non-finite."""
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
    size: str = "unit"               # blend shape: "unit" or "median"
    scale: float | None = None       # s: the median |d_j - m| over the pool's residuals
    select: str = "table"            # which k members: one of SELECTS
    cand: int = 100                  # "hybrid": how many table neighbours are re-ranked
    # a declared pick, {target: (pool row indices, weights or None)}; a target it names is
    # averaged over exactly those rows, and a target it does NOT name is refused
    picks: Mapping | None = None

    def __post_init__(self):
        self._pool_pos = {lab: i for i, lab in enumerate(self.pool)}
        self._resid_unit = None      # unit rows of `resid`, built on first use by a response rule
        self._resid_sq = None        # squared row lengths of `resid`, for "euclid"
        if self.select not in SELECTS:
            raise ValueError(f"select must be one of {SELECTS}, got {self.select!r}")
        if self.select == "hybrid" and not self.k <= self.cand < len(self.pool):
            raise ValueError(f"select='hybrid' re-ranks the table's cand nearest and keeps k of "
                             f"them, so it needs k <= cand < the pool's size: got k = {self.k}, "
                             f"cand = {self.cand}, pool = {len(self.pool)}")
        # also checked by `build_neighbour_arm`; repeated here for an arm built by hand
        if self.size not in SIZES:
            raise ValueError(f"size must be one of {SIZES}, got {self.size!r}")
        if self.size == "median" and not (self.scale is not None and self.scale > 0):
            raise ValueError(f"size='median' divides the neighbour mean by the pool's median "
                             f"residual length, so it needs a positive scale, got {self.scale!r}: "
                             f"build the arm with build_neighbour_arm(..., size='median'), which "
                             f"measures it from the pool")
        # the declared pick's smallest and largest member count, for the run record
        self._picks_sizes = (None if not self.picks else
                             (min(len(i) for i, _ in self.picks.values()),
                              max(len(i) for i, _ in self.picks.values())))

    def fuse(self, target: str, delta: np.ndarray) -> np.ndarray:
        """The fused delta for ``target`` (a new array); unresolved or zero-residual comes back as is, counted."""
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
        n, own = self.neighbour_mean(target, e, r)
        un = unit(n)
        ur = unit(r)
        if self.size == "median":
            # the arm keeps its length in units of the pool's median residual length
            arm = n / self.scale
            arm_size = float(np.linalg.norm(arm))
            s["arm_size_sum"] = s.get("arm_size_sum", 0.0) + arm_size
            s["arm_size_gt1"] = s.get("arm_size_gt1", 0) + int(arm_size > 1.0)
        else:
            arm = un
        fused = self.mean + r_norm * unit(ur + self.w * arm)
        s["targets_fused"] = s.get("targets_fused", 0) + 1
        s["arm_cosine_sum"] = s.get("arm_cosine_sum", 0.0) + float(ur @ un)
        s["own_in_pool"] = s.get("own_in_pool", 0) + int(own is not None)
        return fused

    def neighbour_mean(self, target: str, e: np.ndarray,
                       r: np.ndarray | None = None) -> tuple[np.ndarray, int | None]:
        """``n_t``, the mean residual of the picked pool members (t excluded), and t's pool index or None."""
        own = self._pool_pos.get(target)
        s = self.stats
        if self.picks is not None:
            got = self.picks.get(target)
            if got is None:
                # no fallback to `select`: a declared pick must name every target the arm fuses
                raise ValueError(f"the declared pick does not name {target!r}, and it declares "
                                 f"{len(self.picks)} target(s): a declared pick must name every "
                                 f"target the arm fuses -- the pick comes from a rule this arm "
                                 f"cannot compute, so a target it leaves out has no rule at all")
            idx, wts = got
            s["picks_used"] = s.get("picks_used", 0) + 1
            if wts is None:
                return self.resid[idx].mean(0), own
            tot = float(np.sum(wts))
            if not tot > 0.0:                # every declared weight is 0: the flat mean, counted
                s["picks_zero_weight"] = s.get("picks_zero_weight", 0) + 1
                return self.resid[idx].mean(0), own
            return (np.asarray(wts, dtype=float) / tot) @ self.resid[idx], own
        if self.select == "table":
            sim = self.pool_unit @ unit(e)
            if own is not None:
                sim[own] = -np.inf           # a target is never its own neighbour
            idx = np.argsort(-sim)[: self.k]
            return self.resid[idx].mean(0), own
        if self.select == "constant":
            # no member is read and k is unused; the same vector for every target
            return -self.mean, own
        if r is None:
            raise ValueError(f"select={self.select!r} ranks the pool against the target's own "
                             "residual: pass r")
        ur = unit(np.asarray(r, dtype=float))
        if self.select == "hybrid":
            sim = self.pool_unit @ unit(e)
            if own is not None:
                sim[own] = -np.inf
            near = np.argsort(-sim)[: self.cand]      # own sorts last, and cand < the pool's size
            idx = near[np.argsort(-(self._unit_resid()[near] @ ur))[: self.k]]
        elif self.select == "response":
            rs = self._unit_resid() @ ur
            if own is not None:
                rs[own] = -np.inf
            idx = np.argsort(-rs)[: self.k]
        elif self.select == "euclid":     # |d_j - m - r|^2, less the constant |r|^2
            if self._resid_sq is None:
                self._resid_sq = np.einsum("ij,ij->i", self.resid, self.resid)
            d2 = self._resid_sq - 2.0 * (self.resid @ np.asarray(r, dtype=float))
            if own is not None:
                d2[own] = np.inf
            idx = np.argsort(d2)[: self.k]
        else:
            # reachable only if `select` was assigned after construction
            raise ValueError(f"select must be one of {SELECTS}, got {self.select!r}")
        return self.resid[idx].mean(0), own

    def _unit_resid(self) -> np.ndarray:
        """Unit rows of the pool's residuals, built once."""
        if self._resid_unit is None:
            self._resid_unit = unit(self.resid)
        return self._resid_unit

    def neighbour_unit(self, target: str, e: np.ndarray,
                       r: np.ndarray | None = None) -> tuple[np.ndarray, int | None]:
        """``unit(n_t)`` and t's pool index: `neighbour_mean`, normalised."""
        n, own = self.neighbour_mean(target, e, r)
        return unit(n), own

    def summary(self) -> dict:
        """What a run records: the knobs, the pool, and how the targets fared."""
        s = dict(self.stats)
        n = s.get("targets_fused", 0)
        if n:
            # mean cosine between the target's residual and its neighbour arm
            s["arm_cosine_mean"] = round(s.pop("arm_cosine_sum") / n, 6)
        else:
            s.pop("arm_cosine_sum", None)
        tot = s.pop("arm_size_sum", None)
        if self.size == "median" and n:
            # how long the neighbour arm is in pool-median units, and how often (before w)
            # it is longer than the unit residual it is blended with
            s["arm_size_mean"] = round((tot or 0.0) / n, 6)
            s["arm_size_gt1_frac"] = round(s.get("arm_size_gt1", 0) / n, 6)
        # `scale` is rounded for reading; `scale_exact` is the number `fuse` divided by
        return {"k": self.k, "w": self.w, "size": self.size,
                "scale": None if self.scale is None else round(self.scale, 6),
                "scale_exact": self.scale,
                "select": self.select,
                **({"cand": self.cand} if self.select == "hybrid" else {}),
                **({"picks_declared": len(self.picks)} if self.picks is not None else {}),
                # the declared pick's member counts
                **({"picks_members_min": self._picks_sizes[0],
                    "picks_members_max": self._picks_sizes[1]}
                   if self._picks_sizes is not None else {}),
                "pool_used": len(self.pool), **s}


def build_neighbour_arm(pool: Sequence[str], delta_of: Callable[[str], np.ndarray | None],
                        axis: np.ndarray, table: Mapping, *, k: int, w: float,
                        size: str = "unit", select: str = "table", cand: int = 100,
                        picks: Mapping | None = None) -> NeighbourArm:
    """Pool the residuals once; ``delta_of(label)`` is the caller's pooled delta (None = uncovered)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}, got {size!r}")
    if select not in SELECTS:
        raise ValueError(f"select must be one of {SELECTS}, got {select!r}")
    if picks is not None and select != "table":
        raise ValueError(f"a declared pick must name every target the arm fuses, so there is "
                         f"nothing left for select={select!r} to rank: pass one or the other")
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
            d[gene_pos[lab]] = 0.0               # own gene zeroed
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
    resid = np.ascontiguousarray(D - m)
    # row norms in chunks, so no second copy of the residual block is made
    rn = np.empty(len(resid))
    for lo in range(0, len(resid), 64):
        rn[lo:lo + 64] = np.linalg.norm(resid[lo:lo + 64], axis=1)
    scale = float(np.median(rn))
    if size == "median" and not scale > 0.0:          # 0, negative and NaN all land here
        raise ValueError("size='median' divides the neighbour mean by the pool's median "
                         f"residual length, which is {scale!r} here, not a positive number: the "
                         "pool's deltas equal its mean, their lengths underflowed, or a member "
                         f"carries a non-finite value ({stats})")
    return NeighbourArm(k=k, w=float(w), table=table, pool=labels, mean=m,
                        resid=resid, pool_unit=unit(np.stack(vecs)),
                        gene_pos=gene_pos, stats=stats, size=size, scale=scale,
                        select=select, cand=int(cand),
                        picks=None if picks is None else index_picks(picks, labels))


def index_picks(picks: Mapping, pool: Sequence[str]) -> dict:
    """A declared pick, ``{target: {"members", "weights"}}``, as ``{target: (pool row indices, weights)}``."""
    pos = {lab: i for i, lab in enumerate(pool)}
    out = {}
    for target, entry in picks.items():
        members = [str(x) for x in entry["members"]]
        if not members:
            raise ValueError(f"declared pick for {target!r} names no member")
        twice = sorted({x for x in members if members.count(x) > 1})
        if twice:
            raise ValueError(f"declared pick for {target!r} names {len(twice)} member(s) more "
                             f"than once ({twice[:5]}): a repeat is an undeclared weight")
        gone = [x for x in members if x not in pos]
        if gone:
            raise ValueError(f"declared pick for {target!r} names {len(gone)} label(s) that are "
                             f"not in the pool as built ({gone[:5]}): it was made for another "
                             "pool, or for sources that cover other targets")
        if str(target) in members:
            raise ValueError(f"declared pick for {target!r} names the target itself")
        wts = entry.get("weights")
        if wts is not None:
            wts = np.asarray(wts, dtype=float)
            if wts.shape != (len(members),):
                raise ValueError(f"declared pick for {target!r}: {len(members)} members and "
                                 f"{wts.size} weights")
            if not np.isfinite(wts).all() or (wts < 0).any():
                raise ValueError(f"declared pick for {target!r}: weights must be finite and >= 0")
        out[str(target)] = (np.array([pos[x] for x in members], dtype=np.int64), wts)
    return out


@dataclass
class NeighbourMix:
    """Several gene tables on one shared pool: ``out = m + |r| unit(unit(r) + sum_i w_i a_i)``."""

    arms: list[NeighbourArm]
    stats: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return self.arms[0].k

    @property
    def ws(self) -> list[float]:
        return [a.w for a in self.arms]

    @property
    def size(self) -> str:
        return self.arms[0].size

    def fuse(self, target: str, delta: np.ndarray) -> np.ndarray:
        s, a0 = self.stats, self.arms[0]
        es = [table_vector(a.table, target) for a in self.arms]
        if all(e is None for e in es):
            s["targets_unresolved"] = s.get("targets_unresolved", 0) + 1
            return delta
        d = np.array(delta, dtype=float)
        if target in a0.gene_pos:
            d[a0.gene_pos[target]] = 0.0
        r = d - a0.mean
        r_norm = float(np.linalg.norm(r))
        if r_norm == 0.0:
            s["targets_zero_residual"] = s.get("targets_zero_residual", 0) + 1
            return delta
        ur = unit(r)
        acc = ur.copy()
        for i, (a, e) in enumerate(zip(self.arms, es)):
            if e is None:
                s[f"table{i}_unresolved"] = s.get(f"table{i}_unresolved", 0) + 1
                continue
            n, _own = a.neighbour_mean(target, e, r)
            un = unit(n)
            # each arm follows its own size rule; the arms share one pool, so one scale
            if a.size == "median":
                arm = n / a.scale
                sz = float(np.linalg.norm(arm))
                # per table
                s[f"table{i}_size_sum"] = s.get(f"table{i}_size_sum", 0.0) + sz
                s[f"table{i}_size_gt1"] = s.get(f"table{i}_size_gt1", 0) + int(sz > 1.0)
            else:
                arm = un
            acc += a.w * arm
            s[f"table{i}_cosine_sum"] = s.get(f"table{i}_cosine_sum", 0.0) + float(ur @ un)
        s["targets_fused"] = s.get("targets_fused", 0) + 1
        return a0.mean + r_norm * unit(acc)

    def summary(self) -> dict:
        s = dict(self.stats)
        n = s.get("targets_fused", 0)
        for i in range(len(self.arms)):
            c = s.pop(f"table{i}_cosine_sum", None)
            if c is not None and n:
                s[f"table{i}_cosine_mean"] = round(c / n, 6)
            # only under size="median", and only when something was fused
            z = s.pop(f"table{i}_size_sum", None)
            if z is not None and n:
                s[f"table{i}_size_mean"] = round(z / n, 6)
                s[f"table{i}_size_gt1_frac"] = round(s.get(f"table{i}_size_gt1", 0) / n, 6)
        a0 = self.arms[0]
        return {"k": self.k, "w": self.ws, "size": self.size,
                "scale": None if a0.scale is None else round(a0.scale, 6),
                "scale_exact": a0.scale,
                "pool_used": len(a0.pool),
                "pool_requested": a0.stats.get("pool_requested"), **s}


def build_neighbour_arms(pool: Sequence[str], delta_of: Callable[[str], np.ndarray | None],
                         axis: np.ndarray, tables: Sequence[Mapping], *, k: int,
                         ws: Sequence[float], size: str = "unit", select: str = "table",
                         cand: int = 100,
                         picks: Mapping | None = None) -> NeighbourArm | NeighbourMix:
    """One table: `build_neighbour_arm`. Several: a `NeighbourMix` over the pool members every table resolves."""
    if len(tables) != len(ws) or not tables:
        raise ValueError(f"one w per table: got {len(tables)} tables and {len(ws)} weights")
    if len(tables) == 1:
        return build_neighbour_arm(pool, delta_of, axis, tables[0], k=k, w=ws[0], size=size,
                                   select=select, cand=cand, picks=picks)
    if select != "table" or picks is not None:
        raise ValueError("select != 'table' and a declared pick are wired for one table, not for "
                         f"a mix of {len(tables)}")
    labels = list(dict.fromkeys(str(x) for x in pool))
    shared = [lab for lab in labels if all(table_vector(t, lab) is not None for t in tables)]
    memo: dict = {}

    def once(lab):
        if lab not in memo:
            memo[lab] = delta_of(lab)
        return memo[lab]

    arms = [build_neighbour_arm(shared, once, axis, t, k=k, w=w, size=size)
            for t, w in zip(tables, ws)]
    for a in arms[1:]:
        assert a.pool == arms[0].pool and np.array_equal(a.mean, arms[0].mean)
        assert a.scale == arms[0].scale          # one pool, one residual scale
    arms[0].stats["pool_requested"] = len(labels)
    arms[0].stats["pool_unresolved"] = len(labels) - len(shared)
    return NeighbourMix(arms=arms)
