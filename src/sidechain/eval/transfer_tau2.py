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

from sidechain.data.stream_pseudobulk import PseudobulkSums

__all__ = [
    "Tau2Fit", "common_axis", "common_axis_from_paths", "axis_fingerprint",
    "fit_pair", "fit_pairs", "load_for_axis",
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


def common_axis(sources: Mapping[str, PseudobulkSums],
                controls: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """Genes and targets shared by EVERY source given.

    Every control label is excluded from the target list -- a control arm is the
    denominator of both sides of the contrast, never a target of it.
    """
    if len(sources) < 2:
        raise ValueError("need at least two sources to share an axis")
    missing = [k for k in sources if k not in controls]
    if missing:
        raise KeyError(f"no control label given for {missing}")

    genes: set[str] | None = None
    labels: set[str] | None = None
    for name, pb in sources.items():
        g = {str(x) for x in pb.genes}
        lab = {str(x) for x in pb.labels}
        if controls[name] not in lab:
            raise KeyError(f"{name}: control label {controls[name]!r} matches no label "
                           f"(have e.g. {sorted(lab)[:3]})")
        genes = g if genes is None else (genes & g)
        labels = lab if labels is None else (labels & lab)
    targets = sorted(labels - set(controls.values()))
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
        if name not in controls:
            raise KeyError(f"no control label given for {name!r}")
        lab_l, gene_l = _PB.peek(path)
        lab, gene = set(lab_l), set(gene_l)
        if controls[name] not in lab:
            raise KeyError(f"{name}: control label {controls[name]!r} matches no label")
        genes = gene if genes is None else (genes & gene)
        labels = lab if labels is None else (labels & lab)
    targets = sorted(labels - set(controls.values()))
    if not genes or not targets:
        raise ValueError(f"empty shared axis: {len(genes or ())} genes, {len(targets)} targets")
    return sorted(genes), targets


def load_for_axis(paths: Mapping[str, "str | Path"], controls: Mapping[str, str],
                  genes: Sequence[str], targets: Sequence[str]
                  ) -> dict[str, PseudobulkSums]:
    """Load every source restricted to one shared axis, control arm included."""
    return {name: PseudobulkSums.load_subset(path, [controls[name]] + list(targets), genes)
            for name, path in paths.items()}


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


def _per_target_blocks(pb_a: PseudobulkSums, ctrl_a: str,
                       pb_b: PseudobulkSums, ctrl_b: str,
                       targets: Sequence[str], var_floor: str
                       ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(d2, v) per target, kept blocked so a bootstrap can resample whole targets.

    Uses the production estimator, imported not reimplemented, so a tau^2 is on the
    same ruler as the pooling weights it describes.
    """
    from sidechain.submit.build import _log2fc_with_var

    d2s, vs = [], []
    for t in targets:
        fa, va = _log2fc_with_var(pb_a, t, ctrl_a, var_floor=var_floor)
        fb, vb = _log2fc_with_var(pb_b, t, ctrl_b, var_floor=var_floor)
        d2 = (fa - fb) ** 2
        v = va + vb
        ok = np.isfinite(d2) & np.isfinite(v) & (v > 0)
        if not ok.any():
            continue                      # an arm that abstained on every gene
        d2s.append(np.ascontiguousarray(d2[ok]))
        vs.append(np.ascontiguousarray(v[ok]))
    if not d2s:
        raise ValueError("no finite (target, gene) cells; check the control labels")
    return d2s, vs


def fit_pair(pb_a: PseudobulkSums, ctrl_a: str, pb_b: PseudobulkSums, ctrl_b: str,
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
    d2 = np.concatenate(d2s)
    v = np.concatenate(vs)
    return Tau2Fit(
        pair=(name_a, name_b), tau2=_fit_scalar(v, d2, family), family=family,
        n_targets=len(d2s), n_genes=len(genes), n_cells=int(d2.size),
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

    for a, b in pairs:
        for nm in (a, b):
            if nm not in sources:
                raise KeyError(f"pair ({a}, {b}) names unknown source {nm!r}")
        if a == b:
            raise ValueError(f"pair ({a}, {b}) is a line against itself; tau^2 is 0 by "
                             "construction and the fit is meaningless")

    blocks = {(a, b): _per_target_blocks(sources[a], controls[a], sources[b],
                                         controls[b], targets, var_floor)
              for a, b in pairs}

    out: list[Tau2Fit] = []
    draws: dict[tuple[str, str], list[float]] = {k: [] for k in blocks}
    if bootstrap:
        rng = np.random.default_rng(seed)
        n = len(targets)
        for _ in range(int(bootstrap)):
            idx = rng.integers(0, n, n)              # ONE index set, every pair
            for key, (d2s, vs) in blocks.items():
                k = [j for j in idx if j < len(d2s)]
                draws[key].append(_fit_scalar(np.concatenate([vs[j] for j in k]),
                                              np.concatenate([d2s[j] for j in k]),
                                              family))

    for (a, b), (d2s, vs) in blocks.items():
        d2 = np.concatenate(d2s)
        v = np.concatenate(vs)
        ci = None
        if bootstrap:
            ci = (float(np.percentile(draws[(a, b)], 2.5)),
                  float(np.percentile(draws[(a, b)], 97.5)))
        out.append(Tau2Fit(
            pair=(a, b), tau2=_fit_scalar(v, d2, family), family=family,
            n_targets=len(d2s), n_genes=len(genes), n_cells=int(d2.size),
            axis=axis, var_floor=var_floor, ci95=ci,
            median_var_sum=float(np.median(v)),
            extra={"bootstrap": int(bootstrap)} if bootstrap else {},
        ))
    return out
