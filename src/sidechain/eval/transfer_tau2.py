"""tau^2 -- the transfer-error floor between two cell lines, fit from their pseudobulks.

Silence the same gene in two lines and their fold changes will not agree. Part of
that disagreement is counting noise, which each source already reports as a
variance; **tau^2 is what is left after subtracting it** -- the part that is real
biology. For one (target, gene) cell,

    (fc_a - fc_b)^2  ~  var_a + var_b + tau^2

and tau^2 is fit by maximum likelihood over every cell the two lines share.

It is a property of the unordered PAIR, not of either line: symmetric to within
10 %, and one number per pair reproduces every by-receiver mean, so there is no
leftover per-source term (measured 2026-09-01,
``research/ideas/inverse-variance-weight-flooring.md``). Read it as a distance
between two cell lines.

**tau^2 is a diagnostic, not a pooling weight.** Feeding a fitted tau^2 into the
inverse-variance weights is knob ``t``, and it is closed: scrambling tau^2 across
sources at random scored as well as or better than assigning it correctly, on
three folds, including the oracle arm (2026-09-02). This module exists to
measure a distance, never to set a weight.

WHY THIS IS A MODULE AND NOT A SCRIPT
-------------------------------------
It has been hand-rolled twice -- ``runs/mirror/loco_k562gwps_pdex/calibration_20260831/``
and ``runs/t86_tau2_lineage_20260921/`` -- and the second one got it wrong on the
first pass in a way the first could not have caught:

    **A tau^2 is only comparable to another tau^2 fit on the SAME GENE AXIS.**

Let each pair use its own intersection and a pair of X-Atlas lines gets 38,584
genes where a pair involving K562 gets ~8,000. The wide axis is mostly genes
neither line expresses, where both read ~0 and agree trivially, so its tau^2 comes
out **five times too low** (0.00100 against a known 0.0053). Nothing in the data
says this happened; the number just looks small.

So the axis is not a parameter a caller may vary per pair. `fit_pairs` computes
ONE axis across every source in the comparison, stamps each result with that
axis's fingerprint, and `Tau2Fit.assert_comparable` refuses two fits that do not
share it. That is the whole reason this file exists.

EITHER SOURCE TYPE (2026-09-25). A source is a `PseudobulkSums` -- cells we
accumulated, fold change and variance computed here by the production
`_log2fc_with_var` -- or an `LfcTable`, which publishes the contrast already
taken with its own variance (Feng; GWCD4i). Pass `None` as the control for an
LfcTable: it has no control arm, the contrast was divided out upstream. The
shared-axis guard is identical for both, and a fit between a table and a
pseudobulk is exactly the check a new table-type source needs before it is
pooled -- its variance BYPASSES the Poisson floor, so an over-confident one
wins every gene it touches (Feng, c = 9-17x, 2026-08-31).

Scale, for calibration (poisson floor, Gaussian fit, shared axis):
    HCT116 <-> HEK293T   ~0.005    two lines that agree well
    K562 <-> either      ~0.011
    H1 <-> anything      0.020-0.029   far from everything we predict
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from sidechain.data.lfc_table import LfcTable
from sidechain.data.stream_pseudobulk import PseudobulkSums

__all__ = [
    "Tau2Fit", "common_axis", "common_axis_from_paths", "axis_fingerprint",
    "fit_pair", "fit_pairs", "load_for_axis", "ratio_ci",
]

NU = 4.0            # t-distribution degrees of freedom, matching the 2026-08-31 fit
_FAMILIES = ("gauss", "t4")


# --------------------------------------------------------------------- the axis

def axis_fingerprint(genes: Sequence[str], targets: Sequence[str]) -> str:
    """A short digest of the (genes, targets) a fit ran on.

    Two fits are comparable only when this matches. It hashes the SORTED contents,
    not the order, because order is an artifact of how a caller assembled the list
    and changes nothing about what was measured.
    """
    h = hashlib.sha256()
    for part in (sorted(map(str, genes)), sorted(map(str, targets))):
        h.update(str(len(part)).encode())
        h.update(b"\x00".join(x.encode() for x in part))
        h.update(b"\xff")
    return h.hexdigest()[:16]


def _is_table(src) -> bool:
    """True for a source that publishes its own contrast (no control arm)."""
    return isinstance(src, LfcTable)


def _check_controls(sources: Mapping[str, object], controls: Mapping[str, str | None]) -> None:
    """A pseudobulk needs its control label; a table must NOT be given one.

    Both directions are refused rather than tolerated. A pseudobulk without one
    cannot form a contrast at all. A table given one is a caller who believes it
    has a control arm -- it does not, the contrast was taken upstream -- and a
    control label silently ignored is the kind of wrong belief that later picks
    the wrong artifact.
    """
    for name, src in sources.items():
        ctrl = controls.get(name)
        if _is_table(src):
            if ctrl is not None:
                raise ValueError(f"{name} is an LfcTable, which has no control arm; pass "
                                 f"None, not {ctrl!r}")
        elif ctrl is None:
            raise KeyError(f"no control label given for {name!r}")


def _effect(src, ctrl: str | None, target: str, var_floor: str):
    """(log2 fold change, variance) for one target, from either source type."""
    if _is_table(src):
        return src.effect(target)
    from sidechain.submit.build import _log2fc_with_var
    return _log2fc_with_var(src, target, ctrl, var_floor=var_floor)


def common_axis(sources: Mapping[str, PseudobulkSums],
                controls: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """Genes and targets shared by EVERY source given.

    Every control label is excluded from the target list -- a control arm is the
    denominator of both sides of the contrast, never a target of it.
    """
    if len(sources) < 2:
        raise ValueError("need at least two sources to share an axis")
    _check_controls(sources, controls)

    genes: set[str] | None = None
    labels: set[str] | None = None
    for name, pb in sources.items():
        g = {str(x) for x in pb.genes}
        lab = {str(x) for x in pb.labels}
        ctrl = controls.get(name)
        if ctrl is not None and ctrl not in lab:
            raise KeyError(f"{name}: control label {ctrl!r} matches no label "
                           f"(have e.g. {sorted(lab)[:3]})")
        genes = g if genes is None else (genes & g)
        labels = lab if labels is None else (labels & lab)
    targets = sorted(labels - {c for c in controls.values() if c is not None})
    if not genes or not targets:
        raise ValueError(f"empty shared axis: {len(genes or ())} genes, {len(targets)} targets")
    return sorted(genes), targets


def common_axis_from_paths(paths: Mapping[str, "str | Path"],
                           controls: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """`common_axis` for artifacts too large to load -- reads labels and genes only.

    This is the order a real caller needs: decide the axis from metadata, THEN
    `load_for_axis` each source restricted to it. Loading first is what does not
    fit in memory.
    """
    from sidechain.data.stream_pseudobulk import PseudobulkSums as _PB

    genes: set[str] | None = None
    labels: set[str] | None = None
    for name, path in paths.items():
        table = _path_is_table(path)
        ctrl = controls.get(name)
        if table and ctrl is not None:
            raise ValueError(f"{name} is an LfcTable, which has no control arm; pass None")
        if not table and ctrl is None:
            raise KeyError(f"no control label given for {name!r}")
        lab_l, gene_l = _PB.peek(path)       # both formats store labels/genes members
        lab, gene = set(lab_l), set(gene_l)
        if ctrl is not None and ctrl not in lab:
            raise KeyError(f"{name}: control label {ctrl!r} matches no label")
        genes = gene if genes is None else (genes & gene)
        labels = lab if labels is None else (labels & lab)
    targets = sorted(labels - {c for c in controls.values() if c is not None})
    if not genes or not targets:
        raise ValueError(f"empty shared axis: {len(genes or ())} genes, {len(targets)} targets")
    return sorted(genes), targets


def _path_is_table(path) -> bool:
    """An LfcTable npz carries `lfc`/`var`; a PseudobulkSums npz carries `cpm_sum`."""
    import zipfile
    from pathlib import Path as _P
    with zipfile.ZipFile(_P(path).expanduser()) as z:
        names = set(z.namelist())
    if "lfc.npy" in names and "var.npy" in names:
        return True
    if "cpm_sum.npy" in names:
        return False
    raise ValueError(f"{path}: neither an LfcTable nor a PseudobulkSums npz "
                     f"(members {sorted(names)[:6]})")


def load_for_axis(paths: Mapping[str, "str | Path"], controls: Mapping[str, str | None],
                  genes: Sequence[str], targets: Sequence[str]) -> dict:
    """Load every source restricted to one shared axis.

    A pseudobulk comes back with its control arm prepended (it needs it to form a
    contrast); a table comes back with just the targets, columns in `genes` order.
    """
    out = {}
    for name, path in paths.items():
        if _path_is_table(path):
            out[name] = LfcTable.load(path).subset(list(targets), genes)
        else:
            out[name] = PseudobulkSums.load_subset(
                path, [controls[name]] + list(targets), genes)
    return out


# ---------------------------------------------------------------------- the fit

@dataclass(frozen=True)
class Tau2Fit:
    """One pair's transfer-error floor, stamped with the axis it was fit on."""

    pair: tuple[str, str]
    tau2: float                       # the fitted floor, in squared log2 units
    family: str                       # "gauss" or "t4"
    n_targets: int
    n_genes: int
    n_cells: int                      # (target, gene) cells that entered the fit
    axis: str                         # fingerprint -- see assert_comparable
    var_floor: str = "poisson"
    ci95: tuple[float, float] | None = None
    median_var_sum: float = float("nan")
    extra: dict = field(default_factory=dict)

    @property
    def tau(self) -> float:
        """The typical unexplained disagreement, in log2 fold change."""
        return float(np.sqrt(self.tau2))

    def assert_comparable(self, other: Tau2Fit) -> None:
        """Refuse two fits that did not run on the same axis, floor and family.

        This is the guard the module exists for; see the module docstring.
        """
        if self.axis != other.axis:
            raise ValueError(
                f"tau^2 for {self.pair} and {other.pair} were fit on DIFFERENT axes "
                f"({self.n_genes} genes/{self.n_targets} targets vs "
                f"{other.n_genes}/{other.n_targets}); they are not comparable. "
                "Fit every pair you intend to compare through fit_pairs(), which "
                "shares one axis across all of them.")
        if self.var_floor != other.var_floor:
            raise ValueError(f"different var_floor: {self.var_floor} vs {other.var_floor}")
        if self.family != other.family:
            raise ValueError(f"different likelihood family: {self.family} vs {other.family}")

    def ratio_to(self, other: Tau2Fit) -> float:
        """``self.tau2 / other.tau2`` -- how much further apart this pair is."""
        self.assert_comparable(other)
        return float(self.tau2 / other.tau2)


def _nll_gauss(v: np.ndarray, d2: np.ndarray) -> float:
    return float(np.mean(np.log(v) + d2 / v))


def _nll_t4(v: np.ndarray, d2: np.ndarray) -> float:
    s = v * (NU - 2.0) / NU
    return float(np.mean(np.log(s) + (NU + 1.0) * np.log1p(d2 / (NU * s))))


_NLL = {"gauss": _nll_gauss, "t4": _nll_t4}


def _fit_scalar(v: np.ndarray, d2: np.ndarray, family: str,
                lo: float = -6.0, hi: float = 2.0) -> float:
    """Golden-section on log10(tau^2), seeded by a coarse grid.

    Same shape as the 2026-08-31 fit, so numbers stay comparable to that table.
    """
    nll = _NLL[family]
    grid = np.logspace(lo, hi, 33)
    vals = [nll(v + t, d2) for t in grid]
    i = int(np.argmin(vals))
    a, b = np.log10(grid[max(i - 1, 0)]), np.log10(grid[min(i + 1, len(grid) - 1)])
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = b - phi * (b - a), a + phi * (b - a)
    f1, f2 = nll(v + 10 ** x1, d2), nll(v + 10 ** x2, d2)
    for _ in range(60):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - phi * (b - a)
            f1 = nll(v + 10 ** x1, d2)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + phi * (b - a)
            f2 = nll(v + 10 ** x2, d2)
    return float(10 ** ((a + b) / 2.0))


def _per_target_blocks(pb_a, ctrl_a: str | None, pb_b, ctrl_b: str | None,
                       targets: Sequence[str], var_floor: str
                       ) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    """(d2, v) per target, kept blocked so a bootstrap can resample whole targets.

    Uses the production estimator, imported not reimplemented, so a tau^2 is on the
    same ruler as the pooling weights it describes.
    """
    # ONE SLOT PER TARGET, None where the pair has nothing to say about it. Skipping
    # instead of holding a None shifts every later index, so a bootstrap that draws
    # target POSITIONS would read a different target in each pair -- and the shared-index
    # guarantee `fit_pairs` promises would break silently the first time a table-type
    # source abstained on a whole target (GWCD4i's knockdown gate does exactly that).
    d2s: list[np.ndarray | None] = []
    vs: list[np.ndarray | None] = []
    for t in targets:
        ea = _effect(pb_a, ctrl_a, t, var_floor)
        eb = _effect(pb_b, ctrl_b, t, var_floor)
        if ea is None or eb is None:      # a table that does not carry this target
            d2s.append(None)
            vs.append(None)
            continue
        fa, va = ea
        fb, vb = eb
        d2 = (fa - fb) ** 2
        v = va + vb
        ok = np.isfinite(d2) & np.isfinite(v) & (v > 0)
        if not ok.any():                  # an arm that abstained on every gene
            d2s.append(None)
            vs.append(None)
            continue
        d2s.append(np.ascontiguousarray(d2[ok]))
        vs.append(np.ascontiguousarray(v[ok]))
    if all(x is None for x in d2s):
        raise ValueError("no finite (target, gene) cells; check the control labels")
    return d2s, vs


def _present(blocks: list[np.ndarray | None]) -> list[np.ndarray]:
    return [b for b in blocks if b is not None]


# The grid the bootstrap evaluates each target's likelihood on. 801 points over the
# same [1e-6, 1e2] range the exact fit searches, ~2.3 % apart, refined by a parabola
# in log(tau^2) -- far finer than any bootstrap interval this produces.
TAU2_GRID = np.logspace(-6.0, 2.0, 801)


def _profile(d2: np.ndarray, v: np.ndarray, family: str) -> np.ndarray:
    """One target's summed per-cell NLL at every grid tau^2, shape (K,).

    Summing these over a resampled set of targets IS that replicate's NLL curve, so a
    bootstrap needs no refit: precompute once per target, then each replicate is a sum
    and an argmin. That is what turns a 40-minute bootstrap into seconds.
    """
    s = v[None, :] + TAU2_GRID[:, None]                   # (K, n)
    if family == "gauss":
        return (np.log(s) + d2[None, :] / s).sum(axis=1)
    sp = s * (NU - 2.0) / NU
    return (np.log(sp) + (NU + 1.0) * np.log1p(d2[None, :] / (NU * sp))).sum(axis=1)


def _argmin_refined(curve: np.ndarray) -> float:
    """Grid argmin, refined by a parabola through the three points around it."""
    k = int(np.argmin(curve))
    if 0 < k < curve.size - 1:
        x = np.log(TAU2_GRID[k - 1:k + 2])
        y = curve[k - 1:k + 2]
        den = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
        a = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / den
        b = (x[2] ** 2 * (y[0] - y[1]) + x[1] ** 2 * (y[2] - y[0])
             + x[0] ** 2 * (y[1] - y[2])) / den
        if a > 0:
            xv = -b / (2 * a)
            if x[0] <= xv <= x[2]:
                return float(np.exp(xv))
    return float(TAU2_GRID[k])


def fit_pair(pb_a, ctrl_a: str | None, pb_b, ctrl_b: str | None,
             *, genes: Sequence[str], targets: Sequence[str], name_a: str = "a",
             name_b: str = "b", var_floor: str = "poisson", family: str = "gauss",
             ) -> Tau2Fit:
    """Fit one pair on a CALLER-SUPPLIED axis.

    Prefer `fit_pairs` for anything you will compare: it is the one that guarantees
    a shared axis. This is here for a single pair standing alone, and it still
    stamps the fingerprint so a later comparison can be checked.

    ``pb_a``/``pb_b`` must already be restricted to ``genes`` -- build them with
    `PseudobulkSums.load_subset`, which does that without materialising the artifact.
    """
    if family not in _FAMILIES:
        raise ValueError(f"family must be one of {_FAMILIES}, got {family!r}")
    for nm, pb in ((name_a, pb_a), (name_b, pb_b)):
        if [str(x) for x in pb.genes] != [str(g) for g in genes]:
            raise ValueError(f"{nm}: gene axis does not match the one requested; "
                             "restrict it with PseudobulkSums.load_subset first")
    d2s, vs = _per_target_blocks(pb_a, ctrl_a, pb_b, ctrl_b, targets, var_floor)
    d2 = np.concatenate(_present(d2s))
    v = np.concatenate(_present(vs))
    return Tau2Fit(
        pair=(name_a, name_b), tau2=_fit_scalar(v, d2, family), family=family,
        n_targets=len(_present(d2s)), n_genes=len(genes), n_cells=int(d2.size),
        axis=axis_fingerprint(genes, targets), var_floor=var_floor,
        median_var_sum=float(np.median(v)),
    )


def fit_pairs(sources: Mapping[str, PseudobulkSums], controls: Mapping[str, str],
              pairs: Sequence[tuple[str, str]], *, genes: Sequence[str] | None = None,
              targets: Sequence[str] | None = None, var_floor: str = "poisson",
              family: str = "gauss", bootstrap: int = 0, seed: int = 0,
              ) -> list[Tau2Fit]:
    """Fit every pair on ONE shared axis. This is the entry point to use.

    ``genes``/``targets`` default to the intersection over every source in
    ``sources`` -- not over each pair, which is the error described in the module
    docstring. Pass them explicitly only to narrow that intersection further; they
    are still applied identically to every pair.

    ``bootstrap`` resamples TARGETS with replacement, using **one shared index set
    per replicate** across all pairs. That is deliberate: pairs sharing a line share
    its noise, and drawing independently per pair would inflate the interval on any
    ratio between them. The per-pair ``ci95`` and any ratio CI a caller derives are
    therefore both honest.
    """
    if genes is None or targets is None:
        auto_g, auto_t = common_axis(sources, controls)
        genes = auto_g if genes is None else list(genes)
        targets = auto_t if targets is None else list(targets)
    genes, targets = list(genes), list(targets)
    axis = axis_fingerprint(genes, targets)

    _check_controls(sources, controls)
    for a, b in pairs:
        for nm in (a, b):
            if nm not in sources:
                raise KeyError(f"pair ({a}, {b}) names unknown source {nm!r}")
        if a == b:
            raise ValueError(f"pair ({a}, {b}) is a line against itself; tau^2 is 0 by "
                             "construction and the fit is meaningless")

    blocks = {(a, b): _per_target_blocks(sources[a], controls.get(a), sources[b],
                                         controls.get(b), targets, var_floor)
              for a, b in pairs}

    out: list[Tau2Fit] = []
    draws: dict[tuple[str, str], list[float]] = {k: [] for k in blocks}
    if bootstrap:
        # Per target, its NLL curve over the tau^2 grid (zeros where the pair has no
        # cells for it, so a resampled absent target contributes nothing).
        profiles = {}
        for key, (d2s, vs) in blocks.items():
            prof = np.zeros((len(targets), TAU2_GRID.size))
            for j, (d2, v) in enumerate(zip(d2s, vs)):
                if d2 is not None:
                    prof[j] = _profile(d2, v, family)
            profiles[key] = prof
        rng = np.random.default_rng(seed)
        n = len(targets)
        for _ in range(int(bootstrap)):
            # ONE draw of target POSITIONS, shared by every pair -- positions are stable
            # because every pair holds a slot per target (see _per_target_blocks).
            counts = np.bincount(rng.integers(0, n, n), minlength=n)
            for key, prof in profiles.items():
                draws[key].append(_argmin_refined(counts @ prof))

    for (a, b), (d2s, vs) in blocks.items():
        d2 = np.concatenate(_present(d2s))
        v = np.concatenate(_present(vs))
        ci = None
        extra: dict = {}
        if bootstrap:
            ci = (float(np.percentile(draws[(a, b)], 2.5)),
                  float(np.percentile(draws[(a, b)], 97.5)))
            extra = {"bootstrap": int(bootstrap), "seed": int(seed),
                     "draws": [float(x) for x in draws[(a, b)]]}
        out.append(Tau2Fit(
            pair=(a, b), tau2=_fit_scalar(v, d2, family), family=family,
            n_targets=len(_present(d2s)), n_genes=len(genes), n_cells=int(d2.size),
            axis=axis, var_floor=var_floor, ci95=ci,
            median_var_sum=float(np.median(v)), extra=extra,
        ))
    return out


def ratio_ci(num: Tau2Fit, den: Tau2Fit, *, level: float = 0.95) -> dict:
    """Interval on ``num.tau2 / den.tau2`` from their SHARED bootstrap draws.

    Only valid for two fits from one `fit_pairs` call with `bootstrap` on: the same
    seed and the same target positions per replicate, which is what makes the ratio's
    interval narrower than the two intervals would suggest -- the noise the pairs share
    cancels. Refused otherwise, rather than dividing two independent draw sets.
    """
    num.assert_comparable(den)
    a, b = num.extra.get("draws"), den.extra.get("draws")
    if not a or not b or len(a) != len(b):
        raise ValueError("both fits need bootstrap draws of equal length from one fit_pairs call")
    if num.extra.get("seed") != den.extra.get("seed"):
        raise ValueError("draws come from different bootstrap seeds; not paired")
    r = np.asarray(a) / np.asarray(b)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {"point": float(num.tau2 / den.tau2),
            "ci": (float(np.percentile(r, lo)), float(np.percentile(r, hi))),
            "p_above_1": float(np.mean(r > 1.0)), "n": int(r.size)}
