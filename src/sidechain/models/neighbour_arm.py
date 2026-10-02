"""The neighbour arm (T103): SER's own delta fused with its gene-table neighbours' mean delta.

This is the geometry gate's ``cross``-mode fusion (``scripts/esm2_geometry_gate.py``), moved
into the shipping path so `submit.build` and `eval.loco` can carry it as a knob. The gate
measured it; this module only replays it on pooled deltas. For each predicted target t:

  1. ``m``   the mean pooled delta over the neighbour POOL (a declared list of targets the
             sources measured), each member's own gene zeroed first: the shared response.
  2. ``r_t`` t's pooled delta minus ``m``, t's own gene zeroed: the residual SER carries.
  3. ``n_t`` the mean of ``d_j - m`` over the k pool members j nearest to t by cosine in the
             gene table, t itself excluded: the neighbour arm, which never reads t.
  4. out   ``m + |r_t| * unit(unit(r_t) + w * unit(n_t))``   (``size="unit"``, the default)
           or ``m + |r_t| * unit(unit(r_t) + w * n_t / s)``    (``size="median"``), with
           ``s`` the median residual length over the pool, so ``|n_t| / s`` is unitless.

Why each step is the shape it is:

- **Residuals.** The gate scores residuals, and `pds` ranks each prediction against every other
  target's truth, so a component every target shares cannot tell them apart. ``m`` is added
  back unchanged, so the members that read levels (`mse`, `nmae`) see SER's.
- **Unit arms and w.** ``unit(r_t) + w * unit(n_t)`` is the gate's ``unit(ser) + w * unit(esm)``
  term for term, so a w read off the gate's sweep means here what it meant there: the cosine
  of ``out - m`` with any truth equals the gate's fused cosine (`tests/test_neighbour_arm.py`
  pins it against the gate's own functions).
- **Size-aware blend (T103 direction 1 (i), ``size="median"``).** With ``size="unit"`` every
  target's neighbour arm pulls equally hard, however much its neighbourhood agrees. With
  ``size="median"`` the arm keeps its length, divided by one pool-wide number ``s`` (the median
  ``|d_j - m|`` over the pool's residuals, the arm's own scale), so a neighbourhood that
  responds strongly and agrees pulls harder than one that cancels. ``s`` is a constant of the
  pool, not of the target, so no target is rescaled by its own neighbours' size. The default
  stays ``unit``: the gate's blend, bit-identical.
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

- **Which k (T103 round two, ``select``).** ``"table"``, the default, is the gate's pick: the k
  pool members nearest t in the gene table. The other rules also read how the pool members
  RESPONDED in the sources -- their residuals ``d_j - m``, against t's own residual ``r_t`` --
  and never anything of the line being predicted: ``"hybrid"`` takes the table's ``cand``
  nearest and keeps the k of them whose residuals point most like ``r_t`` (cosine);
  ``"response"`` ranks the whole pool by that cosine; ``"euclid"`` by the Euclidean distance
  between residuals, so size counts as well as direction. A rule the arm cannot compute from
  its own residuals (one that needs each member's measurement noise) is handed in as a declared
  pick, ``picks``: for each target the pool members to average and, optionally, their weights.
  A declared pick is built from the same sources as the pool, by the caller, and is recorded by
  its file's hash like the pool is. The default is the gate's pick, bit-identical.

Off is the default everywhere, and off means this module is never imported into the path:
w = 0 is not a fusion at weight zero, it is no fusion (`submit.build`, `eval.loco`).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sidechain.data.gene_aliases import RETIRED_SYMBOLS as ALIAS

SIZES = ("unit", "median")       # the blend shapes: the gate's unit arm, or size-aware
# which k members are averaged: the table's nearest (the gate's), the table's `cand` nearest
# re-ranked by response, or the whole pool ranked by response direction / distance
SELECTS = ("table", "hybrid", "response", "euclid")


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
    size: str = "unit"               # blend shape: "unit" (the gate's) or "median"
    scale: float | None = None       # s: the median |d_j - m| over the pool's residuals
    select: str = "table"            # which k members: one of SELECTS
    cand: int = 100                  # "hybrid": how many table neighbours are re-ranked
    # a declared pick, {target: (pool row indices, weights or None)}; a target it names is
    # averaged over exactly those rows, any other target follows `select`
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
        # `build_neighbour_arm` checks these too; here they also catch an arm built by hand (a
        # screen cross-checking its own fusion against `fuse`), which otherwise divides by None
        # deep inside `fuse` and reads as a TypeError about floats.
        if self.size not in SIZES:
            raise ValueError(f"size must be one of {SIZES}, got {self.size!r}")
        if self.size == "median" and not (self.scale is not None and self.scale > 0):
            raise ValueError(f"size='median' divides the neighbour mean by the pool's median "
                             f"residual length, so it needs a positive scale, got {self.scale!r}: "
                             f"build the arm with build_neighbour_arm(..., size='median'), which "
                             f"measures it from the pool")

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
        """``n_t``, the mean residual of the picked pool members, t excluded; and t's pool index
        (None when t is not in the pool).

        The one place the neighbours are picked: `neighbour_unit` is this, normalised. ``r`` is
        t's own residual, which every rule but ``"table"`` ranks the members against.
        """
        own = self._pool_pos.get(target)
        s = self.stats
        if self.picks is not None:
            got = self.picks.get(target)
            if got is not None:
                idx, wts = got
                s["picks_used"] = s.get("picks_used", 0) + 1
                if wts is None:
                    return self.resid[idx].mean(0), own
                tot = float(np.sum(wts))
                if not tot > 0.0:            # every declared weight is 0: the flat mean, counted
                    s["picks_zero_weight"] = s.get("picks_zero_weight", 0) + 1
                    return self.resid[idx].mean(0), own
                return (np.asarray(wts, dtype=float) / tot) @ self.resid[idx], own
            # a target the declared pick does not name follows the arm's own rule, counted: a
            # caller that meant to declare every target reads this count and refuses the run
            s["picks_missing"] = s.get("picks_missing", 0) + 1
        if self.select == "table":
            sim = self.pool_unit @ unit(e)
            if own is not None:
                sim[own] = -np.inf           # a target is never its own neighbour
            idx = np.argsort(-sim)[: self.k]
            return self.resid[idx].mean(0), own
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
        else:                                         # "euclid": |d_j - m - r|^2, less the constant |r|^2
            if self._resid_sq is None:
                self._resid_sq = np.einsum("ij,ij->i", self.resid, self.resid)
            d2 = self._resid_sq - 2.0 * (self.resid @ np.asarray(r, dtype=float))
            if own is not None:
                d2[own] = np.inf
            idx = np.argsort(d2)[: self.k]
        return self.resid[idx].mean(0), own

    def _unit_resid(self) -> np.ndarray:
        """Unit rows of the pool's residuals, built once: a second (n_pool, n_axis) block, which
        only a response rule pays for."""
        if self._resid_unit is None:
            self._resid_unit = unit(self.resid)
        return self._resid_unit

    def neighbour_unit(self, target: str, e: np.ndarray,
                       r: np.ndarray | None = None) -> tuple[np.ndarray, int | None]:
        """``unit(n_t)`` and t's pool index: `neighbour_mean`, normalised (one pick, one path).

        Diagnostic API: since the size flag, `fuse` takes the unnormalised mean and normalises it
        itself, so nothing in `src/` or `scripts/` calls this -- only tests and a probe that wants
        the gate's unit arm on its own.
        """
        n, own = self.neighbour_mean(target, e, r)
        return unit(n), own

    def summary(self) -> dict:
        """What a run records: the knobs, the pool, and how the targets fared."""
        s = dict(self.stats)
        n = s.get("targets_fused", 0)
        if n:
            # mean cosine between SER's residual and the neighbour arm: how far they agree
            s["arm_cosine_mean"] = round(s.pop("arm_cosine_sum") / n, 6)
        else:
            s.pop("arm_cosine_sum", None)
        tot = s.pop("arm_size_sum", None)
        if self.size == "median" and n:
            # how long the neighbour arm is in pool-median units, and how often (before w)
            # it is longer than the unit residual it is blended with
            s["arm_size_mean"] = round((tot or 0.0) / n, 6)
            s["arm_size_gt1_frac"] = round(s.get("arm_size_gt1", 0) / n, 6)
        # `scale` is rounded for reading; `scale_exact` is the number `fuse` divided by, because
        # rounding to 6 decimals is a relative 4e-9 at a real residual length and a replay from
        # the record alone could not reach the 1e-10 agreement the screen's gate asks for.
        return {"k": self.k, "w": self.w, "size": self.size,
                "scale": None if self.scale is None else round(self.scale, 6),
                "scale_exact": self.scale,
                "select": self.select,
                **({"cand": self.cand} if self.select == "hybrid" else {}),
                **({"picks_declared": len(self.picks)} if self.picks is not None else {}),
                "pool_used": len(self.pool), **s}


def build_neighbour_arm(pool: Sequence[str], delta_of: Callable[[str], np.ndarray | None],
                        axis: np.ndarray, table: Mapping, *, k: int, w: float,
                        size: str = "unit", select: str = "table", cand: int = 100,
                        picks: Mapping | None = None) -> NeighbourArm:
    """Pool the residuals once; ``delta_of(label)`` is the caller's pooled delta (None = uncovered).

    ``delta_of`` must be the SAME pooling the predicted targets get (sources, shrinkage,
    floors), or the pool's residuals are in a different space from SER's.

    ``size`` is the blend shape: ``"unit"`` (the gate's, the default) or ``"median"``. The
    pool's median residual length is measured and recorded either way, so a run says what the
    arm's scale was even when it did not use it.

    ``select`` and ``cand`` are the selection rule (`SELECTS`). ``picks`` is a declared pick in
    its file form, ``{target: {"members": [pool labels], "weights": [...] or None}}``: every
    member must be in the pool as built here (a label that is not means the pick was made for
    another pool or other sources, and is refused by name), and a target may not name itself.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}, got {size!r}")
    if select not in SELECTS:
        raise ValueError(f"select must be one of {SELECTS}, got {select!r}")
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
    resid = np.ascontiguousarray(D - m)
    # chunked: `np.linalg.norm(resid, axis=1)` in one call materialises a second full copy of the
    # residual block (126 MB at 842 members x 18,533 genes), and this runs on every build, size
    # flag or not. Per-row norms do not depend on the chunking, so the number is bit-identical.
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
    """A declared pick, from its file form (labels) to the arm's (pool row indices).

    ``{target: {"members": [...], "weights": [...] | None}}`` ->
    ``{target: (int64 indices, float weights | None)}``. Refused, by name: a member that is not
    in the pool, a target that names itself, an empty member list, weights of another length,
    and a negative or non-finite weight.
    """
    pos = {lab: i for i, lab in enumerate(pool)}
    out = {}
    for target, entry in picks.items():
        members = [str(x) for x in entry["members"]]
        if not members:
            raise ValueError(f"declared pick for {target!r} names no member")
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
    """Several gene tables at once: ``out = m + |r| unit(unit(r) + sum_i w_i a_i)``, with each
    ``a_i`` its arm's own blend shape (``unit(n_i)``, or ``n_i / s`` under ``size="median"``).

    Each table picks its own k nearest neighbours; all of them share one pool (the members
    every table resolves), so they share one ``m`` and one set of residuals, and SER's own
    residual enters once. A target a table cannot resolve simply gets no term from that
    table; one no table resolves keeps SER's delta, counted, as in `NeighbourArm`.
    """

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
                # per table, so a median mix says how hard each table's arm actually pulled
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
            # only under size="median", and only when the table resolved something: a missing
            # reading must never render as "that table's neighbourhoods cancelled"
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
    """One table: exactly `build_neighbour_arm`. Several: a `NeighbourMix` whose arms share the
    pool members every table resolves, pooled once. ``size`` is one rule for every arm.

    A selection rule other than ``"table"`` and a declared pick are wired for ONE table: in a
    mix it is not defined which table's arm they would replace, so both are refused there."""
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
