"""Build a 2026 submission end to end: emitter -> out-of-core h5ad -> .vcc.

    uv run python -m sidechain.submit.build --challenge-config challenges/vcc2026/config.yaml \
        --emitter delta-transfer --h1-cache ~/data/sidechain/cache/vcc2026/h1_pseudobulk.npz \
        --gwps-cache ~/data/sidechain/cache/vcc2026/k562_gwps_targets_pseudobulk.npz \
        --out ~/data/sidechain/vcc2026/submissions/r1_delta_v1

Emitters (cells are integer counts at the target context's depth; --dispersion picks
Poisson or minimum-variance "even" cells, and --emit-lambda replaces both with one dial
between them -- count_emitters.PoissonEmitter):
  control-null     the context's control profile, no shift. A pipeline check; scores ~-0.3
                   because it calls no DE genes (fid charges silence).
  h1-mean-shift    one generic shift for every perturbation: the mean over the 300 H1
                   perturbations of their log2 fold change vs H1 controls. Rung 0b, transferred.
  delta-transfer   per-target log2 fold change pooled (inverse-variance, per gene) over the
                   sources that perturbed that gene -- K562 genome-wide and H1 -- re-anchored on
                   the target context's control profile. Targets no source covers fall back to
                   the H1 mean shift. Rung 1'.

`--source` adds a pseudobulk corpus beyond the two named caches, as `.npz:control_label`
(repeatable) -- the same syntax `sidechain.eval.loco` takes, so an arm scored on the mirror
is submitted with the identical source list. `--shrink-source` adds one whose transferred
log2FCs are shrunk no matter what `--no-shrink` says -- the depth-aware split between deep
essential-scale arms and genome-wide arms (see `pooled_delta`). `--lfc-source` adds a corpus that publishes the
contrast already taken instead of cells (Feng 2026), built by
`python -m sidechain.data.lfc_table`. It pools identically; it just derives its per-gene
variance from an adjusted p-value rather than from CPM spread, and abstains (zero weight) on
genes whose p-value has saturated.

`--limit-perts N` builds a small panel (first N perturbations) and writes a matching
pert_counts CSV so `vcc prep --dry-run --perts <that>` can validate the layout locally.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sidechain.data.lfc_table import LfcTable
from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.models.count_emitters import (
    ContextProfile,
    PoissonEmitter,
    log2fc_from_cpm,
    remap_to_axis,
)
from sidechain.submit.writer import Contract, SubmissionWriter, pack_vcc, verify_h5ad
from sidechain.utils.naming import CLAIMS_RE, check_out_leaf
from sidechain.utils.paths import resolve_config

LN2_SQ = np.log(2) ** 2
TARGET_SELF_LOG2FC = -2.32  # > 80 % knockdown of the target itself; excluded from scoring, kept for realism
GAMMA_MULT_FLOOR = 2.0 ** -16  # where the gamma family predicts <= 0 expression, emit ~none instead


def gamma_transfer(fc: np.ndarray, ctrl_src_cpm: np.ndarray, ctrl_tgt_cpm: np.ndarray,
                   gamma: float, pseudocount: float = 1.0,
                   stats: dict | None = None) -> np.ndarray:
    """Re-express one source's log2FCs under transfer exponent ``gamma``.

    The emitter replays a delta as ``control * 2^log2FC``, i.e. it assumes the FOLD change is
    the context-invariant quantity. This is one endpoint of a one-parameter family
    (private research/ideas/effect-size-from-control-features.md): with
    ``r_g = (ctrl_tgt_g + 1) / (ctrl_src_g + 1)`` in CPM, the effective multiplier is

        m_g = 1 + (2^fc_g - 1) * r_g^(gamma - 1)

    ``gamma = 1`` is exactly today's emitter (callers skip the call entirely, so the default
    path stays bit-identical); ``gamma = 0`` transfers the source's ABSOLUTE CPM change
    instead, added onto the target's controls -- ``(ctrl_src + 1) * (2^fc - 1)`` is exactly
    ``mean_pert - mean_ctrl`` under the emitter's own pseudocounted fold change.

    Where the target expresses far less than the source, an absolute-change transfer can
    predict negative expression (m <= 0). That is outside the family's domain; the honest
    completion is "the gene is emptied", so m is floored at ``GAMMA_MULT_FLOOR`` -- a finite
    log2FC of -16, effectively zero counts after emission -- rather than passed to ``log2``,
    whose nan/-inf the emitter would silently repair to "no change" (``_fraction`` maps
    non-finite shifts to 0.0), turning the strongest predicted silencings into no-ops.

    Genes whose target-side control CPM is unknown (NaN -- the source measures a gene the
    target axis lacks) get r = 1, which is the identity transform; they are dropped at the
    remap onto the target axis anyway, so the value never lands.
    """
    r = np.where(np.isnan(ctrl_tgt_cpm), 1.0,
                 (ctrl_tgt_cpm + pseudocount) / (ctrl_src_cpm + pseudocount))
    mult = 1.0 + np.expm1(fc * np.log(2.0)) * np.power(r, gamma - 1.0)
    clamped = mult < GAMMA_MULT_FLOOR
    if stats is not None:
        stats["gamma_genes_transformed"] = (
            stats.get("gamma_genes_transformed", 0) + int((~np.isnan(ctrl_tgt_cpm)).sum()))
        stats["gamma_mult_clamped"] = stats.get("gamma_mult_clamped", 0) + int(clamped.sum())
    return np.log2(np.maximum(mult, GAMMA_MULT_FLOOR))


def _log2fc_with_var(pb: PseudobulkSums, label: str, control: str, pseudocount: float = 1.0,
                     var_floor: str = "none"):
    """Per-gene log2FC of mean CPM and its delta-method variance, for one source.

    Computed ROW-WISE, not by slicing `pb.mean_cpm()` / `pb.var_cpm()`. Those build the whole
    (labels x genes) matrix and this function needs exactly two of its rows -- which was
    invisible at 301 labels and fatal at 18,294: on a full-corpus X-Atlas artifact each call
    allocated ~5.6 GB per matrix, several times over, for two rows of 38,584 floats. Pooling
    272 targets from two such sources would have asked for hundreds of multi-GB temporaries.

    With `var_floor="none"` the arithmetic is unchanged -- `mean_cpm` and `var_cpm` are
    defined as exactly these per-row expressions, so this returns bit-identical values (held
    by `test_row_wise_log2fc_matches_the_whole_matrix_form`).

    `var_floor="poisson"` floors each arm's per-cell CPM variance at its Poisson sampling
    variance, `(m + pseudocount) * 1e6 / mean_libsize`: observed spread below what counting
    noise alone would produce is undersampling, not certainty. The `+ pseudocount` keeps
    never-expressed genes (observed variance exactly 0 at any n) off the pathological
    max-weight branch. An arm with a single cell has no observed spread at all, so it
    abstains outright (`var = inf`, weight 0) rather than letting the floor pretend one
    cell was a measurement.
    """
    i, c = pb.labels.index(label), pb.labels.index(control)
    ni = max(int(pb.n_cells[i]), 1)
    nc = max(int(pb.n_cells[c]), 1)
    mi = pb.cpm_sum[i] / ni
    mc = pb.cpm_sum[c] / nc
    vi = np.maximum(pb.cpm_sq_sum[i] / ni - mi * mi, 0.0)
    vc = np.maximum(pb.cpm_sq_sum[c] / nc - mc * mc, 0.0)
    if var_floor == "poisson":
        vi = np.maximum(vi, (mi + pseudocount) * 1e6 / (pb.libsize_sum[i] / ni))
        vc = np.maximum(vc, (mc + pseudocount) * 1e6 / (pb.libsize_sum[c] / nc))
    fc = log2fc_from_cpm(mi, mc, pseudocount)
    var = (vi / ni) / (mi + pseudocount) ** 2 + (vc / nc) / (mc + pseudocount) ** 2
    if var_floor == "poisson" and (ni < 2 or nc < 2):
        var = np.full_like(var, np.inf)
    return fc, var / LN2_SQ


def h1_mean_shift(h1: PseudobulkSums, control: str, axis: np.ndarray) -> np.ndarray:
    perts = [lab for lab in h1.labels if lab != control]
    fcs = np.stack([_log2fc_with_var(h1, p, control)[0] for p in perts])
    return remap_to_axis(fcs.mean(axis=0), h1.genes, axis)


def shrink(fc: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Per-gene positive-part shrinkage of log2FCs toward 0: fc * max(0, 1 - var/fc^2).

    A source measures each gene's fold change with its own sampling error; with
    ~170 cells per K562 target, a gene at 5 CPM carries roughly +-0.5 log2 of pure
    noise, which would be transferred as if it were signal and charged by the
    fold-change metric. This is the gene-wise James-Stein rule: a gene whose
    estimate is within one standard error of zero is set to zero, one at three
    standard errors keeps 89 % of its value, one at five keeps 96 %. A single
    global normal prior was tried first and erased the real effects too -- they
    are a few hundred genes among ~8,000 nulls, so any one-variance prior is
    dominated by the nulls.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(fc != 0, 1.0 - var / np.maximum(fc**2, 1e-12), 0.0)
    return fc * np.clip(factor, 0.0, 1.0)


def parse_coverage_tiers(spec: str | None) -> tuple[tuple[float, float], ...] | None:
    """``'3:0.10,10:0.50'`` -> ``((3.0, 0.10), (10.0, 0.50))``; None stays None.

    Each pair is `cut:factor` -- a gene-arm whose `n_eff` is below `cut` has its pooling
    weight multiplied by `factor`. The pairs are read in order and the first match wins,
    so cuts must increase; anything at or above the last cut keeps its full weight, which
    is why the strong tier is never written down.
    """
    if spec is None:
        return None
    tiers = []
    for part in spec.split(","):
        cut, _, factor = part.partition(":")
        if not factor:
            raise SystemExit(f"--coverage-tiers: {part!r} is not cut:factor, e.g. '3:0.10'")
        tiers.append((float(cut), float(factor)))
    cuts = [c for c, _ in tiers]
    if cuts != sorted(cuts) or len(set(cuts)) != len(cuts):
        raise SystemExit(f"--coverage-tiers: cut points must strictly increase, got {cuts}")
    for cut, factor in tiers:
        if cut <= 0:
            raise SystemExit(f"--coverage-tiers: cut {cut} must be positive")
        # A zero factor is refused, not clamped. It would zero the denominator on genes
        # every source calls weak, and the pooled delta comes out 0 -- which the emitter
        # replays as "no change" and `fid` charges as silence. The whole point of a tier
        # is that thin evidence is outvoted where better evidence exists and still used
        # where none does.
        if not 0 < factor <= 1:
            raise SystemExit(f"--coverage-tiers: factor {factor} must be in (0, 1]; a factor "
                             "of 0 silences genes rather than downweighting them -- use a "
                             "small positive factor")
    return tuple(tiers)


def parse_transfer_floor(specs: list[str] | None) -> dict[str, float]:
    """``['h1_pseudobulk=0.0104']`` -> ``{'h1_pseudobulk': 0.0104}``; None/[] -> ``{}``.

    Keyed by the source file's basename stem rather than by position, because the pool's
    order is a command-line accident and a floor attached to the wrong source is silently
    wrong rather than loud. `apply_transfer_floors` refuses a key that matches no source.
    """
    out: dict[str, float] = {}
    for part in specs or []:
        name, _, value = part.partition("=")
        if not value:
            raise SystemExit(f"--transfer-floor: {part!r} is not NAME=TAU2, e.g. "
                             "'h1_pseudobulk=0.0104'")
        try:
            tau2 = float(value)
        except ValueError:
            raise SystemExit(f"--transfer-floor: {value!r} is not a number") from None
        if tau2 < 0:
            raise SystemExit(f"--transfer-floor: tau^2 {tau2} must be >= 0 (it is a variance)")
        if name in out:
            raise SystemExit(f"--transfer-floor: {name!r} given twice")
        out[name] = tau2
    return out


def apply_transfer_floors(sources: list, floors: dict[str, float]) -> list:
    """Attach each ``NAME=TAU2`` to the source whose file stem is NAME.

    Matching is on the stem recorded by `sources_from_specs`. A key matching no source is a
    hard error: the alternative is a run that silently pools uncalibrated weights while its
    `build.json` claims otherwise, which is exactly the class of quiet wrongness the
    exact-match rule in `ingest/checks.py` exists to prevent.
    """
    if not floors:
        return sources
    seen = {}
    for src in sources:
        obj = src[0] if isinstance(src, tuple) else src
        name = getattr(obj, "sidechain_name", None)
        if name is not None:
            seen[name] = src
    missing = sorted(set(floors) - set(seen))
    if missing:
        raise SystemExit(
            f"--transfer-floor names {missing} match no source; have {sorted(seen)}. "
            "The floor is per source and attaching it to the wrong one is silently wrong."
        )
    for name, tau2 in floors.items():
        src = seen[name]
        obj = src[0] if isinstance(src, tuple) else src
        obj.transfer_floor = tau2
    return sources


def coverage_factor(n_eff: np.ndarray, tiers: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Per-gene weight multiplier from the evidence behind each gene.

    `n_eff` is how many cells' worth of evidence sits behind that gene in that arm (see
    `PseudobulkSums.n_eff`). Genes below the first cut get the first factor, and so on;
    genes above the last cut keep their full weight.
    """
    out = np.ones_like(n_eff, dtype=np.float64)
    for cut, factor in reversed(tiers):
        out = np.where(n_eff < cut, factor, out)
    return out


class _PseudobulkDeltaSource:
    """Adapts `(PseudobulkSums, control_label)` to the `effect(target)` shape.

    Exists so `pooled_delta` iterates one kind of thing. Sources that ship
    counts and sources that ship a precomputed contrast then differ only in how
    they answer "what is this target's fold change and how well is it known",
    which is the only question the pooling actually asks.

    `shrink` is this source's own position of the shrinkage knob: None defers
    to `pooled_delta`'s global flag (every historical call), True/False
    overrides it for this source alone. Per source because the right setting
    depends on what the arm measured: a deep essential-scale arm estimates its
    small effects well, so shrinking it only removes noise, while a
    genome-wide arm's small effects are the direction signal shrinkage would
    delete.
    """

    __slots__ = ("control", "pb", "shrink", "var_floor")

    def __init__(self, pb: PseudobulkSums, control: str, shrink: bool | None = None,
                 var_floor: str = "none"):
        # Refused, not coerced: the third tuple slot sits beside var_floor in
        # this signature, and a stray string there ('poisson') would otherwise
        # silently force shrinkage ON -- crash-to-wrong is the bad direction.
        if shrink is not None and not isinstance(shrink, bool):
            raise TypeError(f"shrink must be None, True or False, got {shrink!r} "
                            "-- var_floor is keyword-only in the tuple form")
        self.pb, self.control, self.var_floor = pb, control, var_floor
        self.shrink = shrink

    @property
    def genes(self) -> np.ndarray:
        return self.pb.genes

    @property
    def transfer_floor(self) -> float:
        """This source's tau^2, carried on the underlying `PseudobulkSums`.

        Read through rather than stored, because `as_delta_source` rebuilds this wrapper on
        every `pooled_delta` call while the floor is attached once, to the artifact.
        """
        return float(getattr(self.pb, "transfer_floor", 0.0))

    def control_cpm(self) -> np.ndarray:
        """Mean per-cell CPM of this source's own control arm, on its own gene axis --
        the ``ctrl_src`` the transfer exponent (gamma) compares the target's controls to."""
        c = self.pb.labels.index(self.control)
        return self.pb.cpm_sum[c] / max(int(self.pb.n_cells[c]), 1)

    def effect(self, target: str):
        if target not in self.pb.labels:
            return None
        return _log2fc_with_var(self.pb, target, self.control, var_floor=self.var_floor)

    def n_eff(self, target: str) -> np.ndarray | None:
        """Cells' worth of evidence behind each gene of this contrast, on this source's axis.

        The WEAKER of the two arms, because the estimate is a difference: a gene the
        control barely saw is as poorly known as one the perturbation barely saw.

        Only sources built from cells can answer this. An `LfcTable` publishes a contrast
        with no cells behind it, defines no `n_eff`, and is therefore left at full weight
        by `pooled_delta` -- its abstention already comes from its p-values.
        """
        if target not in self.pb.labels:
            return None
        i, c = self.pb.labels.index(target), self.pb.labels.index(self.control)
        return np.minimum(self.pb.n_eff(i), self.pb.n_eff(c))


def as_delta_source(src, var_floor: str = "none"):
    """Normalise a source into something with `.genes` and `.effect(target)`.

    Accepts the historical `(PseudobulkSums, control_label)` tuple so every
    existing call site and command line keeps working unchanged, and passes an
    `LfcTable` (or anything else implementing the pair) straight through.
    A `(PseudobulkSums, control_label, shrink)` triple additionally pins that
    source's own shrinkage (see `_PseudobulkDeltaSource`); the two-tuple form
    means "follow the global flag", exactly as before.
    `var_floor` only reaches the pseudobulk form: an LfcTable's variance comes
    from a p-value, already carries abstention (`inf`), and has no cell counts
    to floor against.
    """
    if isinstance(src, tuple):
        return _PseudobulkDeltaSource(*src, var_floor=var_floor)
    if hasattr(src, "effect") and hasattr(src, "genes"):
        return src
    raise TypeError(
        f"{type(src).__name__} is not a delta source: expected a "
        "(PseudobulkSums, control_label) tuple or an object with .genes and "
        ".effect(target) -> (fc, var) | None"
    )


def sources_from_specs(source_specs: list[str], shrink_source_specs: list[str]) -> list:
    """Parse `--source` / `--shrink-source` NPZ:CONTROL specs into source tuples.

    The one parser for both entry points: `sidechain.eval.loco` and this
    module's `main` call it instead of splitting the specs themselves, so the
    two command lines cannot drift and a mirror-scored arm really does submit
    verbatim. A missing (or empty) control suffix defaults to 'control';
    `--shrink-source` specs come back as the `(pb, control, True)` triple that
    pins shrinkage on for that source.
    """
    sources = []
    for spec, shrunk in [(s, False) for s in source_specs] + [(s, True) for s in shrink_source_specs]:
        path, _, ctrl = spec.rpartition(":")
        pb = PseudobulkSums.load(path)
        # The stem is how `--transfer-floor NAME=TAU2` finds this source. Recorded here, at
        # the one place a source is built from a path, so the two flags cannot disagree about
        # what a source is called.
        pb.sidechain_name = Path(path).expanduser().stem
        sources.append((pb, ctrl or "control", True) if shrunk else (pb, ctrl or "control"))
    return sources


def control_similarity(ctrl_src: np.ndarray, ctrl_tgt: np.ndarray,
                       *, min_overlap: int = 100) -> float:
    """Cosine between two CONTROL profiles, on log1p CPM, over the genes both measure.

    This is the quantity [[scbasecount-context-matching]] wanted and the thing `reports/05`
    said was impossible ("no line-specific priors are possible -- we cannot even look one
    up"). It turns out not to need scBaseCount at all for the pooling case: we hold 18,400
    control cells for each 2026 context and a control arm inside every pseudobulk source, so
    the similarity between a target context and a source line is directly measurable from
    data already on disk. scBaseCount answers the different question of WHAT a context is.

    log1p first, because a raw-CPM cosine is dominated by the few thousand-CPM genes and
    would rank sources by how ribosomal their libraries are rather than by cell identity.

    Restricted to genes both profiles actually measure -- our pseudobulk sources are
    gene-truncated to 8-10k while a challenge context carries all 18,533, so the union
    would score a source down for genes it never had the chance to report. `min_overlap`
    refuses rather than returning a small number: an empty overlap means an axis mismatch,
    and silently weighting that source to zero is the kind of quiet wrongness that looks
    like a modelling result.
    """
    both = np.isfinite(ctrl_src) & np.isfinite(ctrl_tgt)
    # You cannot demand 100 shared genes from a profile that only has 3. The guard is aimed at
    # an axis MISMATCH -- where the overlap collapses toward zero -- so it scales down to the
    # smaller of the two profiles rather than refusing every small one on principle.
    need = max(1, min(min_overlap,
                      int(np.isfinite(ctrl_src).sum()), int(np.isfinite(ctrl_tgt).sum())))
    if int(both.sum()) < need:
        raise ValueError(
            f"control profiles share only {int(both.sum())} finite genes (need {need}). "
            "That is an axis mismatch, not a dissimilar cell line -- check the source's gene "
            "space before weighting anything by this."
        )
    a = np.log1p(np.maximum(ctrl_src[both], 0.0))
    b = np.log1p(np.maximum(ctrl_tgt[both], 0.0))
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("a control profile is all zeros over the shared genes")
    return float(np.dot(a, b) / (na * nb))


def pooled_delta(target: str, sources: list, axis: np.ndarray,
                 *, shrinkage: bool = True, var_floor: str = "none",
                 gamma: float = 1.0, ctrl_tgt_cpm: np.ndarray | None = None,
                 coverage_tiers: tuple[tuple[float, float], ...] | None = None,
                 similarity_beta: float = 0.0,
                 stats: dict | None = None) -> np.ndarray | None:
    """Inverse-variance pool of the sources that perturbed `target`; None if none did.

    A source may be a `(PseudobulkSums, control_label)` tuple -- counts we
    accumulated, variance from the per-cell CPM spread -- or an `LfcTable`,
    which publishes the contrast already taken and derives its variance from an
    adjusted p-value. Both answer `effect(target)` with `(fc, var)` on their own
    gene axis, and everything below is indifferent to which it got.

    `shrinkage` is the global switch; a source carrying its own `shrink`
    attribute (the tuple triple, or any source object that sets one) overrides
    it for that source alone. That is the depth-aware form: shrink the deep
    essential-scale arms whose sub-noise effects really are noise, leave the
    genome-wide arms -- where those same-size effects are the measured
    direction -- untouched.

    A source may also ABSTAIN per gene by returning `var = inf` there, which
    makes its weight exactly 0. Feng does this on the 98.9 % of rows whose
    p-value has saturated: it has a fold change but no evidence about it, and at
    a median of 25 cells per target "no detectable change" would otherwise drag
    real effects from better-powered sources toward zero. Note `any_src` is set
    on the source COVERING the target, not on it having a usable weight -- a
    target only Feng covers, and only saturated rows at that, returns a
    genuine all-zero delta rather than None, and does not silently fall through
    to the mean-shift fallback.

    `var_floor="poisson"` applies the sampling floor inside each pseudobulk
    source's variance (see `_log2fc_with_var`) and drops the weight clamp to a
    pure numerical guard -- with the floor on, no finite variance can be small
    for a pathological reason, so the clamp no longer has statistical work to do.
    `stats`, if given, accumulates the diagnostics across calls: how many gene
    weights sit at or below the historical 1e-6 clamp (the accidental-certainty
    arm this floor exists to remove), how many (target, source) arms fully
    abstained, and how many covered targets ended with zero total weight.

    `gamma` is the transfer exponent (see `gamma_transfer`): each source's fold
    change is re-expressed against the ratio of the TARGET context's control CPM
    (`ctrl_tgt_cpm`, on `axis` -- required when gamma != 1) to that source's own
    control CPM, per source BEFORE pooling, because the absolute change that
    transfers at gamma = 0 is a per-source quantity. It runs after shrinkage
    (transform the best estimate, not the raw one) and does NOT touch the
    pooling weights -- the weight estimator belongs to the variance floor and a
    gamma term must not quietly reweight the pool. A source that publishes no
    control profile (an LfcTable ships only the contrast) has no ctrl_src, so
    gamma != 1 refuses it rather than guessing. gamma = 1 skips all of this and
    is bit-identical to the historical call.

    `coverage_tiers` multiplies each source's per-gene weight by how much evidence
    stands behind that gene in that arm (`PseudobulkSums.n_eff`). The variance the
    weight comes from knows how many cells the ARM has; it does not know that a gene
    whose counts add up to 100 might be one cell at 100 rather than 100 cells at 1.
    Every factor is strictly positive, so no gene is ever silenced -- thin evidence is
    outvoted where better evidence exists and still used where none does. A source
    with no cells behind it (an LfcTable) is left at full weight. None is the default
    and is bit-identical to the historical call.

    A source may also carry a `transfer_floor` (tau^2), attached by
    `apply_transfer_floors` and read here: a measured constant ADDED to that source's
    variance before it becomes a weight. The floor and the tiers are one variance model
    with two terms, and they are deliberately different shapes -- `var / f(n_eff) +
    tau^2`. The tier is multiplicative because sampling variance really does fall as
    1/n; tau^2 is additive because transfer error does not shrink with cell count (no
    number of HCT116 cells makes HCT116 into K562), so it acts as a CEILING on weight
    (1/tau^2) rather than a rescale. Measured against a held-out fold's truth, never
    tuned -- fitting is the only move available on the challenge contexts, where nothing
    can be scored. tau^2 = 0 is the default and the identity.
    """
    if var_floor not in ("none", "poisson"):
        raise ValueError(f"unknown var_floor {var_floor!r}: expected 'none' or 'poisson'")
    if gamma != 1.0 and ctrl_tgt_cpm is None:
        raise ValueError("gamma != 1 needs ctrl_tgt_cpm: the target context's control CPM "
                         "on the submission axis is the r in gamma_transfer")
    if similarity_beta != 0.0 and ctrl_tgt_cpm is None:
        raise ValueError("similarity_beta != 0 needs ctrl_tgt_cpm: the weighting is a cosine "
                         "between the TARGET context's control profile and each source's own")
    clamp = 1e-6 if var_floor == "none" else 1e-12
    num = np.zeros(len(axis)); den = np.zeros(len(axis)); any_src = False
    for src in (as_delta_source(s, var_floor=var_floor) for s in sources):
        got = src.effect(target)
        if got is None:
            continue
        any_src = True
        fc, var = got
        want = getattr(src, "shrink", None)
        if want is None:
            want = shrinkage
        if want:
            fc = shrink(fc, var)
        if gamma != 1.0:
            get_ctrl = getattr(src, "control_cpm", None)
            if get_ctrl is None:
                raise ValueError(f"{type(src).__name__} publishes no control profile, so the "
                                 "transfer exponent is undefined for it -- pool it at gamma=1 "
                                 "or drop it from a gamma arm")
            tgt_on_src = remap_to_axis(ctrl_tgt_cpm, axis, src.genes, fill=np.nan)
            fc = gamma_transfer(fc, get_ctrl(), tgt_on_src, gamma, stats=stats)
        # `1/inf` is 0, which is the abstention. The `maximum` floor only guards
        # the other end -- a variance so small it would swamp every other
        # source -- and must not be applied to inf, hence the divide as written.
        with np.errstate(divide="ignore"):
            w = 1.0 / np.maximum(var, clamp)
        w = np.where(np.isfinite(var), w, 0.0)
        fc = np.where(np.isfinite(fc), fc, 0.0)
        if similarity_beta != 0.0:
            # A per-SOURCE scalar on the weight, in the same place `coverage_tiers` puts its
            # per-gene one. It says how much this source's cell line looks like the context we
            # are predicting, so a haematopoietic context stops being told about itself by an
            # embryonic stem line at the same volume as by another leukaemia line.
            #
            # beta is the sharpness and beta = 0 is exactly uniform (x**0 == 1), so the default
            # is bit-identical to every historical call -- the same endpoint property the
            # emission dial has. Cosines here run ~0.9-0.99, so beta has to be large to
            # separate them; that is a property of the measure, not a bug, and it is why this
            # is an exponent rather than a multiplier.
            get_ctrl = getattr(src, "control_cpm", None)
            if get_ctrl is None:
                # An LfcTable publishes only the contrast, so it has no control profile and
                # no similarity. Left at full weight and counted, never silently dropped.
                if stats is not None:
                    stats["similarity_sources_unweighted"] = (
                        stats.get("similarity_sources_unweighted", 0) + 1)
            else:
                tgt_on_src = remap_to_axis(ctrl_tgt_cpm, axis, src.genes, fill=np.nan)
                sim = control_similarity(get_ctrl(), tgt_on_src)
                w = w * (sim ** similarity_beta)
                if stats is not None:
                    stats.setdefault("similarity", {})[type(src).__name__ + ":" + str(
                        getattr(src, "name", len(stats.get("similarity", {}))))] = round(sim, 6)
        if coverage_tiers is not None:
            get_neff = getattr(src, "n_eff", None)
            ne = get_neff(target) if get_neff is not None else None
            if ne is not None:
                cf = coverage_factor(ne, coverage_tiers)
                w = w * cf
                if stats is not None:
                    stats["coverage_gene_arms"] = (
                        stats.get("coverage_gene_arms", 0) + int(cf.size))
                    stats["coverage_gene_arms_demoted"] = (
                        stats.get("coverage_gene_arms_demoted", 0) + int((cf < 1.0).sum()))
            elif stats is not None:
                # A source with no cells behind it (LfcTable) keeps full weight. Counted
                # so a run cannot quietly be half-tiered without saying so.
                stats["coverage_sources_unweighted"] = (
                    stats.get("coverage_sources_unweighted", 0) + 1)
        tau2 = float(getattr(src, "transfer_floor", 0.0) or 0.0)
        if tau2:
            # The transfer-error floor: this source's variance gets tau^2 ADDED before it
            # becomes a weight, so no source can claim more certainty than its measured error
            # against a held-out line supports. Written as a transform of the weight rather
            # than of the variance so that it composes with the terms above rather than
            # replacing them: with w = 1/v, w / (1 + tau2*w) is exactly 1/(v + tau2), and the
            # coverage tier has already divided v by its factor -- which is the intended
            # order. The tier scales the SAMPLING variance (that really does fall as 1/n);
            # tau^2 is added after, because transfer error does not shrink with cell count.
            #
            # Additive, not multiplicative, and the difference is the whole point: this caps
            # every weight at 1/tau^2 (w -> 1/tau2 as w -> inf) instead of rescaling the
            # column, so it bites exactly on the genes a source claims to know perfectly and
            # barely touches the ones it already admits are noisy.
            #
            # tau2 = 0 is the identity on every branch, including w = 0 (abstention), so the
            # default path is bit-identical to every historical call.
            w = w / (1.0 + tau2 * w)
            if stats is not None:
                stats["transfer_floor_sources"] = stats.get("transfer_floor_sources", 0) + 1
        if stats is not None:
            finite = np.isfinite(var)
            stats["gene_weights"] = stats.get("gene_weights", 0) + int(finite.sum())
            stats["gene_weights_var_le_1e-6"] = (
                stats.get("gene_weights_var_le_1e-6", 0) + int((finite & (var <= 1e-6)).sum()))
            if not finite.any():
                stats["source_arms_abstained"] = stats.get("source_arms_abstained", 0) + 1
        num += remap_to_axis(fc * w, src.genes, axis, fill=0.0)
        den += remap_to_axis(w, src.genes, axis, fill=0.0)
    if not any_src:
        return None
    out = np.zeros(len(axis))
    nz = den > 0
    if stats is not None and not nz.any():
        stats["targets_zero_weight"] = stats.get("targets_zero_weight", 0) + 1
    out[nz] = num[nz] / den[nz]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--challenge-config", default="challenges/vcc2026/config.yaml")
    ap.add_argument("--emitter", choices=["control-null", "h1-mean-shift", "delta-transfer"], required=True)
    ap.add_argument("--h1-cache")
    ap.add_argument("--gwps-cache")
    ap.add_argument("--source", action="append", default=[], metavar="NPZ:CONTROL",
                    help="additional pseudobulk source for delta-transfer, as .npz:control_label "
                         "(repeatable) -- the syntax sidechain.eval.loco uses, so a mirror-scored "
                         "source list carries over verbatim")
    ap.add_argument("--shrink-source", action="append", default=[], metavar="NPZ:CONTROL",
                    help="like --source, but this source's transferred log2FCs are shrunk "
                         "regardless of --no-shrink (depth-aware shrinkage: shrink the deep "
                         "essential-scale arms, leave genome-wide arms carrying the direction "
                         "signal unshrunk). Same syntax in sidechain.eval.loco.")
    ap.add_argument("--lfc-source", action="append", default=[], metavar="NPZ",
                    help="cached LfcTable .npz -- a source publishing the contrast already "
                         "taken rather than cells (e.g. Feng 2026). Repeatable. Built by "
                         "`python -m sidechain.data.lfc_table`.")
    ap.add_argument("--alpha", type=float, default=1.0, help="scale applied to every transferred log2FC")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="transfer exponent on the target/source control-CPM ratio (see "
                         "gamma_transfer; same knob in sidechain.eval.loco, so a mirror-scored "
                         "arm submits verbatim). 1 = the fold change transfers -- today's "
                         "emitter, bit-identical single-pass shifts. Any other value makes the "
                         "shifts CONTEXT-SPECIFIC: they are pooled once per context inside the "
                         "write loop, against that context's own control profile. delta-transfer "
                         "only. The H1 mean-shift fallback is not gamma-transformed (it is an "
                         "aggregate, not a contrast; fallback is 0 on the current pool).")
    ap.add_argument("--no-shrink", action="store_true", help="disable the per-gene empirical-Bayes shrinkage of transferred log2FCs")
    ap.add_argument("--var-floor", choices=["none", "poisson"], default="none",
                    help="floor each pseudobulk arm's per-gene variance at its Poisson sampling "
                         "variance and abstain on single-cell arms, so observed zero spread stops "
                         "counting as certainty in the pooling weights; 'none' reproduces the "
                         "historical weights bit-for-bit")
    ap.add_argument("--coverage-tiers", metavar="CUT:FACTOR,...",
                    help="weight each source's per-gene vote by how many cells' worth of "
                         "evidence stands behind that gene (n_eff), as cut:factor pairs, "
                         "e.g. '3:0.10,10:0.50' -- below n_eff 3 keep a tenth of the weight, "
                         "3 to 10 keep half, at or above 10 keep it all. Factors must be "
                         "positive: this downweights, it never silences. Same knob in "
                         "sidechain.eval.loco, so a mirror-scored arm submits verbatim.")
    ap.add_argument("--transfer-floor", action="append", default=[], metavar="NAME=TAU2",
                    help="add a measured per-source transfer-error floor tau^2 to that "
                         "source's variance before it becomes a pooling weight, keyed by the "
                         "source file's basename stem (repeatable), e.g. "
                         "'h1_pseudobulk=0.0104'. Caps that source's weight at 1/tau^2 so it "
                         "cannot claim more certainty than its measured error against a "
                         "held-out line supports. Fitted, never tuned. Same knob in "
                         "sidechain.eval.loco, so a mirror-scored arm submits verbatim.")
    ap.add_argument("--limit-perts", type=int, help="build only the first N perturbations (pipeline tests)")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--dispersion", choices=["poisson", "even"], default=None,
                    help="endpoint of the emission dial (default: even); exclusive with "
                         "--emit-lambda (see count_emitters.PoissonEmitter)")
    ap.add_argument("--emit-lambda", type=float, default=None, metavar="LAM",
                    help="emission-sharpening dial in [0, 1]: 0 = even cells, 1 = poisson cells, "
                         "interior values narrow the emitted cloud toward the mean (exact "
                         "variance law: count_emitters.PoissonEmitter). Same knob in "
                         "sidechain.eval.loco, so a mirror-scored arm submits verbatim.")
    ap.add_argument("--out", required=True, help="output stem; writes <out>.h5ad and <out>.vcc")
    ap.add_argument("--no-pack", action="store_true")
    ap.add_argument("--min-libsize", type=float, default=1000.0,
                    help="drop control cells below this depth from the library-size pool")
    args = ap.parse_args(argv)
    cov_tiers = parse_coverage_tiers(args.coverage_tiers)
    if args.gamma != 1.0 and args.emitter != "delta-transfer":
        ap.error("--gamma transforms pooled per-target deltas, so it only applies to "
                 "delta-transfer")
    if args.emit_lambda is not None and args.dispersion is not None:
        ap.error("--dispersion and --emit-lambda are one dial (even is 0, poisson is 1) -- pass one")
    if args.emit_lambda is not None and not 0.0 <= args.emit_lambda <= 1.0:
        # Same check as the emitter's, but before any work: the constructor only
        # runs inside the write loop, an hour into a full build.
        ap.error(f"--emit-lambda must be in [0, 1], got {args.emit_lambda}")
    if args.emit_lambda is None and args.dispersion is None:
        args.dispersion = "even"    # the historical default of this entry point

    stem = Path(args.out).name
    check_out_leaf(stem, context="submit.build", require_slug=True)
    if not CLAIMS_RE.match(stem):
        print(f"note: out stem '{stem}' carries no series tag -- fine for a probe, but a "
              "board submission's stem starts with its lowercased short name (ADR 0005), "
              "e.g. ser-2n_delta4_even_noshrink_v1", flush=True)

    cfg = yaml.safe_load(resolve_config(args.challenge_config).read_text())
    data_dir = Path(cfg["data_dir"]).expanduser()
    genes = pd.read_csv(data_dir / cfg["gene_names_file"]).iloc[:, 0].astype(str).tolist()
    if len(genes) != cfg["n_genes"]:
        raise SystemExit(f"gene_names.csv read as {len(genes)} genes; config says {cfg['n_genes']} -- header handling?")
    perts = pd.read_csv(data_dir / cfg["pert_counts_file"])[cfg["pert_col"]].astype(str).tolist()
    if args.limit_perts:
        perts = perts[: args.limit_perts]
    contexts = [str(c) for c in cfg["phases"][cfg["phase"]]["contexts"]]
    sub = cfg["submission"]
    contract = Contract(
        genes=genes, perturbations=perts, contexts=contexts, cells_per_pert=int(sub["cells_per_pert"]),
        pert_col=cfg["pert_col"], context_col=cfg["context_col"], control_label=cfg["control_label"],
        max_counts_per_cell=int(sub["max_counts_per_cell"]), max_cells=int(sub["max_cells"]),
        max_stored_entries=int(sub["max_stored_entries"]),
    )
    axis = np.asarray(genes)

    # Machine record of what produced this artifact. The stem and the stdout
    # are indistinguishable between, say, a shrunk and an unshrunk build of the
    # same sources; re-running arms to prove which flags produced a file is the
    # incident class this sidecar exists to close.
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".args.json").write_text(json.dumps(vars(args), indent=1, default=str) + "\n")

    # -- per-perturbation log2FC vectors (None = no shift)
    t0 = time.time()
    shifts: dict[str, np.ndarray | None] = {p: None for p in perts}
    fallback = 0
    pool_stats: dict = {}
    if args.emitter in ("h1-mean-shift", "delta-transfer"):
        if not args.h1_cache:
            raise SystemExit("--h1-cache is required for this emitter")
        h1 = PseudobulkSums.load(args.h1_cache)
        # Named like every other source. `--h1-cache` and `--gwps-cache` predate `--source`
        # and load outside `sources_from_specs`, so without this the two oldest arms in the
        # pool are the only ones `--transfer-floor` cannot address -- and H1 is the arm whose
        # variance the calibration measurement found most wrong.
        h1.sidechain_name = Path(args.h1_cache).expanduser().stem
        generic = h1_mean_shift(h1, cfg["control_label"], axis)
        shifts = {p: generic.copy() for p in perts}
    sources = None
    if args.emitter == "delta-transfer":
        if not args.gwps_cache:
            raise SystemExit("--gwps-cache is required for delta-transfer")
        gwps = PseudobulkSums.load(args.gwps_cache)
        gwps.sidechain_name = Path(args.gwps_cache).expanduser().stem
        sources = [(gwps, "control"), (h1, cfg["control_label"])]
        sources += sources_from_specs(args.source, args.shrink_source)
        # Appended, not special-cased: `pooled_delta` normalises both forms, so a
        # source with no cells behind it enters the pool exactly like one that has
        # them and nothing downstream needs to know which it was.
        for path in args.lfc_source:
            tab = LfcTable.load(path)
            tab.sidechain_name = Path(path).expanduser().stem
            sources.append(tab)
        sources = apply_transfer_floors(sources, parse_transfer_floor(args.transfer_floor))
        if args.gamma == 1.0:
            for p in perts:
                d = pooled_delta(p, sources, axis, shrinkage=not args.no_shrink,
                                 var_floor=args.var_floor, coverage_tiers=cov_tiers,
                                 stats=pool_stats)
                if d is None:
                    fallback += 1          # keep the generic shift
                else:
                    shifts[p] = d
    gene_pos = {g: i for i, g in enumerate(genes)}

    def finalize(shift_map):
        # alpha scales the pooled vector; gamma (if any) acted per source inside the pool.
        for p, vec in shift_map.items():
            if vec is not None:
                vec *= args.alpha
                if p in gene_pos:
                    vec[gene_pos[p]] = TARGET_SELF_LOG2FC

    def pool_line(prefix, fb):
        line = f"{prefix}; fallback-to-generic: {fb}"
        if pool_stats.get("gene_weights"):
            frac = pool_stats.get("gene_weights_var_le_1e-6", 0) / pool_stats["gene_weights"]
            line += (f"; var<=1e-6: {frac:.1%} of gene weights"
                     f"; abstained source-arms: {pool_stats.get('source_arms_abstained', 0)}"
                     f"; zero-weight targets: {pool_stats.get('targets_zero_weight', 0)}")
        if pool_stats.get("gamma_genes_transformed"):
            line += (f"; gamma-transformed gene-arms: {pool_stats['gamma_genes_transformed']}"
                     f"; multiplier clamps: {pool_stats.get('gamma_mult_clamped', 0)}")
        if pool_stats.get("coverage_gene_arms"):
            dem = pool_stats.get("coverage_gene_arms_demoted", 0)
            line += (f"; coverage-tiered: {dem / pool_stats['coverage_gene_arms']:.1%} of gene "
                     f"weights demoted")
            if pool_stats.get("coverage_sources_unweighted"):
                line += (f" ({pool_stats['coverage_sources_unweighted']} source-arms have no "
                         "cells and kept full weight)")
        return line

    per_context_shifts = None
    if args.emitter == "delta-transfer" and args.gamma != 1.0:
        # gamma re-expresses each source's fold change against THIS context's control
        # profile, so the shifts stop being shareable across contexts: pool once per
        # context, inside the write loop, on the profile the emitter itself anchors on
        # (the axis-vs-gene_names check has already run by the time this is called).
        def per_context_shifts(prof):
            local = {p: generic.copy() for p in perts}
            fb = 0
            ctrl_cpm = prof.fraction * 1e6
            for p in perts:
                d = pooled_delta(p, sources, axis, shrinkage=not args.no_shrink,
                                 var_floor=args.var_floor, coverage_tiers=cov_tiers,
                                 gamma=args.gamma, ctrl_tgt_cpm=ctrl_cpm,
                                 stats=pool_stats)
                if d is None:
                    fb += 1
                else:
                    local[p] = d
            finalize(local)
            print(pool_line(f"  {prof.name}: gamma={args.gamma:g} shifts ready", fb), flush=True)
            return local

        print(f"gamma={args.gamma:g}: shifts are context-specific and pooled inside the "
              "write loop", flush=True)
    else:
        finalize(shifts)
        print(pool_line(f"shifts ready in {time.time() - t0:.0f}s", fallback), flush=True)

    # -- write
    h5ad = out.with_suffix(".h5ad")
    if args.limit_perts:
        pd.DataFrame({cfg["pert_col"]: perts}).to_csv(out.with_suffix(".pert_counts.csv"), index=False)
    t0 = time.time()
    with SubmissionWriter(h5ad, contract) as w:
        for ci, ctx in enumerate(contexts):
            prof = ContextProfile.from_controls(data_dir / cfg["control_files"][ctx], ctx, min_libsize=args.min_libsize)
            if list(prof.genes) != genes:
                raise SystemExit(f"context {ctx} var_names differ from gene_names.csv")
            ctx_shifts = shifts if per_context_shifts is None else per_context_shifts(prof)
            em = PoissonEmitter(prof, seed=args.seed + ci, dispersion=args.dispersion,
                                lam=args.emit_lambda)
            for k, p in enumerate(perts):
                w.add_block(em.emit(contract.cells_per_pert, ctx_shifts[p]), ctx, p)
                if (k + 1) % 50 == 0:
                    print(f"  {ctx}: {k + 1}/{len(perts)} perturbations  {time.time() - t0:.0f}s", flush=True)
    info = verify_h5ad(h5ad, contract)
    print(json.dumps({"h5ad": str(h5ad), **info, "write_seconds": round(time.time() - t0)}), flush=True)
    if not args.no_pack:
        t0 = time.time()
        vcc = pack_vcc(h5ad, out.with_suffix(".vcc"))
        print(json.dumps({"vcc": str(vcc), "vcc_gb": round(vcc.stat().st_size / 1e9, 2), "pack_seconds": round(time.time() - t0)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
