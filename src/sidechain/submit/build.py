"""Build a 2026 submission: pool per-target log2 fold changes, emit counts, write the h5ad and pack the .vcc.

    uv run python -m sidechain.submit.build --emitter delta-transfer --h1-cache H1.npz --gwps-cache GWPS.npz --out OUT

Emitters: control-null (the context's control profile, no shift); h1-mean-shift (the mean log2FC over the H1
perturbations, for every perturbation); delta-transfer (each target's log2FC pooled over the sources that perturbed
it, re-anchored on the context's control profile; a target no source covers takes the H1 mean shift).
Reads the challenge config, its control files and the source caches; writes <out>.h5ad, <out>.vcc, <out>.args.json
and a sidecar per flag in use. Notes: private/code-notes/src/sidechain/submit/build.py.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sidechain.data.lfc_table import LfcTable
from sidechain.data.loaders import (
    challenge_contexts,
    challenge_control_files,
    challenge_data_dir,
)
from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.models import adaptive_shrink as adaptive_shrink_module
from sidechain.models.adaptive_shrink import adaptive_shrink
from sidechain.models.count_emitters import (
    CONTROL_MIN_LIBSIZE,
    ContextProfile,
    PoissonEmitter,
    log2fc_from_cpm,
    remap_to_axis,
)
from sidechain.submit.alpha_table import load_alpha_table
from sidechain.submit.cell_map import CellMap
from sidechain.submit.ctrl_axes import add_ctrl_axes_args, check_ctrl_axes_args
from sidechain.submit.gene_factor import load_gene_factors, same_factor_groups
from sidechain.submit.knee import AIM_MARGIN as KNEE_AIM_MARGIN
from sidechain.submit.knee import UNDERSHOOT as KNEE_UNDERSHOOT
from sidechain.submit.source_vote import (parse_source_scales, parse_source_weights, pool_kwargs,
                                          read_source_weight_table, source_votes, vote_groups)
from sidechain.submit.knee import TARGET_SUM as KNEE_TARGET_SUM
from sidechain.submit.knee import KneeMeter, knee_keep, knee_next
from sidechain.submit.pie_source import (
    add_prediction_source_args,
    check_prediction_source_args,
    parse_pie_contexts,
    pie_shift_map,
)
from sidechain.submit.variance_model import (
    RULE_VARIANCES,
    VarianceModel,
    add_variance_args,
    apply_dispersion_fits,
    check_variance_args,
    parse_dispersion_fits,
)
from sidechain.submit.writer import Contract, SubmissionWriter, pack_vcc, verify_h5ad
from sidechain.utils.naming import CLAIMS_RE, STEM_RE, check_out_leaf
from sidechain.utils.paths import resolve_config

LN2 = np.log(2)
LN2_SQ = LN2**2
TARGET_SELF_LOG2FC = -2.32  # the target's own gene, at log2(0.2)
GAMMA_MULT_FLOOR = 2.0 ** -16  # where the gamma family predicts <= 0 expression, emit ~none instead


def gamma_transfer(fc: np.ndarray, ctrl_src_cpm: np.ndarray, ctrl_tgt_cpm: np.ndarray,
                   gamma: float, pseudocount: float = 1.0,
                   stats: dict | None = None) -> np.ndarray:
    """Re-express one source's log2FCs under transfer exponent `gamma`.

    m = 1 + (2^fc - 1) * r^(gamma - 1) with r = (ctrl_tgt + 1) / (ctrl_src + 1) in CPM, floored at
    GAMMA_MULT_FLOOR; a gene with no target CPM (NaN) takes r = 1.
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
                     var_floor: str = "none", log_bias_correct: bool = False,
                     variance=None, return_shipped: bool = False, stats: dict | None = None):
    """Per-gene log2FC of mean CPM and its delta-method variance (log2 units), for `label` against `control`.

    `var_floor="poisson"` floors each arm's per-cell variance at its Poisson sampling variance and abstains (inf)
    on an arm under two cells; `variance` swaps the per-cell variances for a model's (needs that floor);
    `log_bias_correct` adds the second-order term; `return_shipped` also returns the unmodelled variance.
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
    if log_bias_correct:
        # + the perturbed arm's downward bias, - the control's; on the shipped per-cell variances whatever
        # `variance` says
        fc = fc + ((vi / ni) / (mi + pseudocount) ** 2
                   - (vc / nc) / (mc + pseudocount) ** 2) / (2.0 * LN2)
    var = (vi / ni) / (mi + pseudocount) ** 2 + (vc / nc) / (mc + pseudocount) ** 2
    var_shipped = var
    if variance is not None:
        if var_floor != "poisson":
            raise ValueError("a variance model needs var_floor='poisson': its Poisson term is that "
                             "floor, and under 'none' the two paths would not measure the same thing")
        si = 1e6 / (pb.libsize_sum[i] / ni)
        sc = 1e6 / (pb.libsize_sum[c] / nc)
        vi_m = variance.cell_variance(mi, si, vi, pseudocount, perturbed=True)
        vc_m = variance.cell_variance(mc, sc, vc, pseudocount, perturbed=False)
        if variance.floor_max:
            fi, fcc = (mi + pseudocount) * si, (mc + pseudocount) * sc
            if stats is not None:
                stats["variance_floor_bound_gene_arms"] = (
                    stats.get("variance_floor_bound_gene_arms", 0) + int(((vi_m < fi) | (vc_m < fcc)).sum()))
            vi_m, vc_m = np.maximum(vi_m, fi), np.maximum(vc_m, fcc)
        var = (vi_m / ni) / (mi + pseudocount) ** 2 + (vc_m / nc) / (mc + pseudocount) ** 2
    if var_floor == "poisson" and (ni < 2 or nc < 2):
        var = np.full_like(var, np.inf)
        var_shipped = var
    if return_shipped:
        return fc, var / LN2_SQ, var_shipped / LN2_SQ
    return fc, var / LN2_SQ


def h1_mean_shift(h1: PseudobulkSums, control: str, axis: np.ndarray) -> np.ndarray:
    perts = [lab for lab in h1.labels if lab != control]
    fcs = np.stack([_log2fc_with_var(h1, p, control)[0] for p in perts])
    return remap_to_axis(fcs.mean(axis=0), h1.genes, axis)


SHRINK_STAGES = ("source", "pooled")
SHRINK_RULES = ("garrote", "adaptive")


def shrink(fc: np.ndarray, var: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Per-gene positive-part shrinkage of log2FCs toward 0: fc * max(0, 1 - k * var/fc^2).

    The non-negative garrote (Breiman 1995): a gene under sqrt(k) standard errors from 0 is set to 0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(fc != 0, 1.0 - k * var / np.maximum(fc**2, 1e-12), 0.0)
    return fc * np.clip(factor, 0.0, 1.0)


def check_shrink_rule(k: float, stage: str, rule: str) -> None:
    """Refuse a shrinkage setting that names no model; shared by `pooled_delta` and the flags."""
    if stage not in SHRINK_STAGES:
        raise ValueError(f"unknown shrink_stage {stage!r}: expected one of {SHRINK_STAGES}")
    if rule not in SHRINK_RULES:
        raise ValueError(f"unknown shrink_rule {rule!r}: expected one of {SHRINK_RULES}")
    if not (math.isfinite(k) and k > 0):
        raise ValueError(f"shrink_k must be finite and > 0, got {k}: 0 would switch the rule "
                         "off, which is what shrinkage=False (--no-shrink) is for")
    if rule == "adaptive":
        if stage != "pooled":
            raise ValueError("shrink_rule='adaptive' is wired after pooling only "
                             "(shrink_stage='pooled'): per source it reads each source's "
                             "standard error as exact, and they are not the same size")
        if k != 1.0:
            raise ValueError("shrink_k is the garrote's threshold and has no meaning under "
                             "shrink_rule='adaptive'")


def parse_coverage_tiers(spec: str | None) -> tuple[tuple[float, float], ...] | None:
    """``'5:0.2,20:0.5'`` -> ``((5.0, 0.2), (20.0, 0.5))``; None stays None.

    Each `cut:factor` multiplies the pooling weight of a gene-arm whose `n_eff` is below `cut`; the first match
    wins, so cuts increase; at or above the last cut the weight is kept.
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
        if not 0 < factor <= 1:
            raise SystemExit(f"--coverage-tiers: factor {factor} must be in (0, 1]; a factor "
                             "of 0 silences genes rather than downweighting them -- use a "
                             "small positive factor")
    return tuple(tiers)


def parse_transfer_floor(specs: list[str] | None) -> dict[str, float]:
    """``['h1_pseudobulk=0.01']`` -> ``{'h1_pseudobulk': 0.01}``; None/[] -> ``{}``.

    Keyed by the source file's basename stem.
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
    """Attach each ``NAME=TAU2`` to the source whose file stem is NAME; a name matching no source is refused."""
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


def source_name(src) -> str | None:
    """The basename stem a source was loaded under (`sources_from_specs`, `--lfc-source`, the two caches)."""
    obj = src[0] if isinstance(src, tuple) else src
    return getattr(obj, "sidechain_name", None)


def parse_source_contexts(specs: list[str] | None) -> dict[str, tuple[str, ...]]:
    """``['NAME=A']`` -> ``{'NAME': ('A',)}``; ``NAME=A,B`` names two; None/[] -> ``{}``.

    Keyed by the source file's basename stem; `route_sources` refuses a name that matches no source.
    """
    out: dict[str, tuple[str, ...]] = {}
    for part in specs or []:
        name, _, value = part.partition("=")
        names = tuple(c.strip() for c in value.split(",") if c.strip())
        if not name or not names:
            raise SystemExit(f"--source-context: {part!r} is not NAME=CTX[,CTX...], e.g. "
                             "'cd4_all_rho01_kd=A'")
        if len(set(names)) != len(names):
            raise SystemExit(f"--source-context: {part!r} names a context twice")
        if name in out:
            raise SystemExit(f"--source-context: {name!r} given twice")
        out[name] = names
    return out


def route_sources(sources: list, routes: dict[str, tuple[str, ...]], contexts: list[str],
                  *, check_contexts: bool = True) -> dict[str, list]:
    """Each context's own pool (`--source-context`): a source named in `routes` votes only in its named contexts,
    every other source in every context; a pool keeps the command line's order and the objects of `sources`.

    Refused: a name matching no source or more than one, a route naming a context outside `contexts` (unless
    `check_contexts` is off), and a context left with no source.
    """
    names = [source_name(s) for s in sources]
    missing = sorted(set(routes) - {n for n in names if n is not None})
    if missing:
        raise SystemExit(f"--source-context names {missing} match no source; have "
                         f"{sorted(n for n in names if n is not None)}")
    twice = sorted(n for n in routes if names.count(n) > 1)
    if twice:
        raise SystemExit(f"--source-context names {twice} match more than one source: two files share "
                         "the stem, so the route cannot say which")
    if check_contexts:
        unknown = sorted({c for cs in routes.values() for c in cs} - set(contexts))
        if unknown:
            raise SystemExit(f"--source-context names context(s) {unknown}, which this build does not "
                             f"write ({', '.join(contexts)})")
    pools = {ctx: [s for s, n in zip(sources, names) if n not in routes or ctx in routes[n]]
             for ctx in contexts}
    empty = [ctx for ctx, pool in pools.items() if not pool]
    if empty:
        raise SystemExit(f"--source-context leaves context(s) {empty} with no source")
    return pools


def context_groups(pools: dict[str, list], contexts: list[str]) -> list[tuple[tuple[str, ...], list]]:
    """The contexts whose pools hold the same sources, grouped in context order, so that each
    distinct pool is pooled once: ``[(('A',), pool_a), (('B', 'C'), pool_bc)]``."""
    groups: list[tuple[tuple[str, ...], list]] = []
    for ctx in contexts:
        pool = pools[ctx]
        for i, (ctxs, have) in enumerate(groups):
            if len(have) == len(pool) and all(a is b for a, b in zip(have, pool)):
                groups[i] = (ctxs + (ctx,), have)
                break
        else:
            groups.append(((ctx,), pool))
    return groups


def split_by_factor(group_maps: list[dict], factor_groups: list[list[str]]) -> list[dict]:
    """Each pooled map split by the contexts' gene factors: a part per set of contexts that read the same factor,
    in context order, the first part keeping the pooled map and every later part a copy.
    """
    key = {c: i for i, g in enumerate(factor_groups) for c in g}
    out = []
    for g in group_maps:
        parts: dict[int, list[str]] = {}
        for c in g["contexts"]:
            parts.setdefault(key[c], []).append(c)
        for j, ctxs in enumerate(parts.values()):
            shifts = g["shifts"] if j == 0 else {p: None if v is None else v.copy() for p, v in g["shifts"].items()}
            out.append({**g, "contexts": ctxs, "shifts": shifts})
    return out


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
    """Adapts `(PseudobulkSums, control_label)` to the `effect(target)` shape `pooled_delta` iterates.

    `shrink`: None follows `pooled_delta`'s global flag; True or False overrides it for this source alone.
    """

    __slots__ = ("control", "log_bias_correct", "pb", "shrink", "var_floor", "variance_model")

    def __init__(self, pb: PseudobulkSums, control: str, shrink: bool | None = None,
                 var_floor: str = "none", log_bias_correct: bool = False,
                 variance_model: VarianceModel | None = None):
        if shrink is not None and not isinstance(shrink, bool):
            raise TypeError(f"shrink must be None, True or False, got {shrink!r} "
                            "-- var_floor is keyword-only in the tuple form")
        self.pb, self.control, self.var_floor = pb, control, var_floor
        self.shrink, self.log_bias_correct = shrink, log_bias_correct
        self.variance_model = (None if variance_model is None or variance_model.is_shipped
                               else variance_model)

    def _variance(self):
        if self.variance_model is None:
            return None
        return self.variance_model.for_source(self.pb, self.control)

    @property
    def genes(self) -> np.ndarray:
        return self.pb.genes

    @property
    def transfer_floor(self) -> float:
        """This source's tau^2, read through from the underlying `PseudobulkSums`."""
        return float(getattr(self.pb, "transfer_floor", 0.0))

    def control_cpm(self) -> np.ndarray:
        """Mean per-cell CPM of this source's own control arm, on its own gene axis --
        the ``ctrl_src`` the transfer exponent (gamma) compares the target's controls to."""
        c = self.pb.labels.index(self.control)
        return self.pb.cpm_sum[c] / max(int(self.pb.n_cells[c]), 1)

    def effect(self, target: str, stats: dict | None = None):
        if target not in self.pb.labels:
            return None
        return _log2fc_with_var(self.pb, target, self.control, var_floor=self.var_floor,
                                log_bias_correct=self.log_bias_correct,
                                variance=self._variance(), stats=stats)

    def effect_pair(self, target: str, stats: dict | None = None):
        """`(fc, var, var_shipped)`: the model's variance and the shipped one, for `--rule-variance`."""
        if target not in self.pb.labels:
            return None
        return _log2fc_with_var(self.pb, target, self.control, var_floor=self.var_floor,
                                log_bias_correct=self.log_bias_correct,
                                variance=self._variance(), return_shipped=True, stats=stats)

    def n_eff(self, target: str) -> np.ndarray | None:
        """Cells' worth of evidence behind each gene of this contrast, on this source's axis: the weaker arm's."""
        if target not in self.pb.labels:
            return None
        i, c = self.pb.labels.index(target), self.pb.labels.index(self.control)
        return np.minimum(self.pb.n_eff(i), self.pb.n_eff(c))


def as_delta_source(src, var_floor: str = "none", log_bias_correct: bool = False,
                    variance_model: VarianceModel | None = None):
    """Normalise a source into something with `.genes` and `.effect(target)`.

    A `(PseudobulkSums, control_label)` tuple, or the triple with a `shrink` slot, is wrapped in
    `_PseudobulkDeltaSource`, the only form `var_floor` reaches; an `LfcTable` or any object with the pair passes.
    """
    if isinstance(src, tuple):
        return _PseudobulkDeltaSource(*src, var_floor=var_floor,
                                      log_bias_correct=log_bias_correct,
                                      variance_model=variance_model)
    if hasattr(src, "effect") and hasattr(src, "genes"):
        return src
    raise TypeError(
        f"{type(src).__name__} is not a delta source: expected a "
        "(PseudobulkSums, control_label) tuple or an object with .genes and "
        ".effect(target) -> (fc, var) | None"
    )


def sources_from_specs(source_specs: list[str], shrink_source_specs: list[str]) -> list:
    """Parse `--source` / `--shrink-source` NPZ:CONTROL specs into source tuples (also `sidechain.eval.loco`'s).

    A missing control suffix reads 'control'; a `--shrink-source` spec comes back as `(pb, control, True)`.
    """
    sources = []
    for spec, shrunk in [(s, False) for s in source_specs] + [(s, True) for s in shrink_source_specs]:
        path, _, ctrl = spec.rpartition(":")
        pb = PseudobulkSums.load(path)
        # the name `--transfer-floor` and `--dispersion-fit` address this source by; the path checks a fit against
        # its file
        pb.sidechain_name = Path(path).expanduser().stem
        pb.sidechain_path = Path(path).expanduser()
        sources.append((pb, ctrl or "control", True) if shrunk else (pb, ctrl or "control"))
    return sources


def control_similarity(ctrl_src: np.ndarray, ctrl_tgt: np.ndarray,
                       *, min_overlap: int = 100) -> float:
    """Cosine between two control profiles, on log1p CPM, over the genes both measure.

    Raises when fewer than `min_overlap` genes are shared (scaled down to the smaller profile) or a profile is 0.
    """
    both = np.isfinite(ctrl_src) & np.isfinite(ctrl_tgt)
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
                 similarity_beta: float = 0.0, log_bias_correct: bool = False,
                 stats: dict | None = None, shrink_k: float = 1.0,
                 shrink_stage: str = "source",
                 shrink_rule: str = "garrote",
                 variance_model: VarianceModel | None = None,
                 rule_variance: str = "model",
                 source_vote=None,
                 source_scale: dict[str, float] | None = None) -> np.ndarray | None:
    """Inverse-variance pool, per gene, of the sources that perturbed `target`, on `axis`; None if none did.

    A source is a `(PseudobulkSums, control_label[, shrink])` tuple or an `LfcTable`; a gene with `var = inf`
    abstains. The keywords move the variance, the shrinkage rule, the weights and the transfer exponent; at their
    defaults it is the plain pool with the garrote per source. `stats` accumulates the run's counts.
    """
    if var_floor not in ("none", "poisson"):
        raise ValueError(f"unknown var_floor {var_floor!r}: expected 'none' or 'poisson'")
    if gamma != 1.0 and ctrl_tgt_cpm is None:
        raise ValueError("gamma != 1 needs ctrl_tgt_cpm: the target context's control CPM "
                         "on the submission axis is the r in gamma_transfer")
    if similarity_beta != 0.0 and ctrl_tgt_cpm is None:
        raise ValueError("similarity_beta != 0 needs ctrl_tgt_cpm: the weighting is a cosine "
                         "between the TARGET context's control profile and each source's own")
    check_shrink_rule(shrink_k, shrink_stage, shrink_rule)
    after = shrinkage and shrink_stage == "pooled"
    if after and (gamma != 1.0 or coverage_tiers is not None or similarity_beta != 0.0):
        raise ValueError("shrink_stage='pooled' divides by the pooled sampling variance "
                         "1/sum(w); gamma != 1, coverage tiers and a similarity weight each "
                         "make sum(w) something else -- run the stage without them")
    if rule_variance not in RULE_VARIANCES:
        raise ValueError(f"unknown rule_variance {rule_variance!r}: expected one of {RULE_VARIANCES}")
    vm = None if variance_model is None or variance_model.is_shipped else variance_model
    if vm is None and rule_variance != "model":
        raise ValueError("rule_variance='shipped' without a variance model: there is no swapped "
                         "variance for the rule to leave")
    if vm is not None and var_floor != "poisson":
        raise ValueError(f"variance_model {vm.spec()!r} needs var_floor='poisson' (see variance_model)")
    # a second denominator from the shipped variances, for a rule held on them while the weights move
    split_rule = vm is not None and rule_variance == "shipped"
    voted = source_vote is not None or bool(source_scale)
    if voted and split_rule:
        raise ValueError("a source vote or strength with rule_variance='shipped' is not wired: the rule's "
                         "second denominator would read plain weights")
    den2 = np.zeros(len(axis)) if voted else None     # sum(m^2 S^2 w): the pooled variance's numerator
    weighted = False                                   # any multiplier off 1 on this target
    clamp = 1e-6 if var_floor == "none" else 1e-12
    num = np.zeros(len(axis)); den = np.zeros(len(axis)); any_src = False
    den_rule = np.zeros(len(axis)) if split_rule else None
    for raw in sources:
        m, sc = 1.0, 1.0
        if voted:
            name = source_name(raw)
            m = source_vote.m(name, target) if source_vote is not None else 1.0
            sc = source_scale.get(name, 1.0) if source_scale and name is not None else 1.0
            if m == 0.0:
                # a zero vote is the source's absence from this pool, exactly as --source-context makes it
                if stats is not None:
                    stats["source_vote_zero_arms"] = stats.get("source_vote_zero_arms", 0) + 1
                continue
        src = as_delta_source(raw, var_floor=var_floor, log_bias_correct=log_bias_correct, variance_model=vm)
        if vm is not None and hasattr(src, "effect_pair"):
            got = src.effect_pair(target, stats=stats)
            if got is None:
                continue
            fc, var, var_shipped = got
            if stats is not None and src._variance() is None:
                # a pseudobulk source the model does not reach (a multiplier naming other sources)
                stats["variance_model_sources_unmodelled"] = (
                    stats.get("variance_model_sources_unmodelled", 0) + 1)
        else:
            got = src.effect(target)
            if got is None:
                continue
            fc, var = got
            var_shipped = var
            if vm is not None and stats is not None:
                # an LfcTable has no cells to model: its p-value variance stands, and is counted
                stats["variance_model_sources_unmodelled"] = (
                    stats.get("variance_model_sources_unmodelled", 0) + 1)
        any_src = True
        var_rule = var_shipped if rule_variance == "shipped" else var
        want = getattr(src, "shrink", None)
        if shrink_stage == "pooled" and want is not None:
            raise ValueError("a per-source shrinkage override (--shrink-source) has no "
                             "meaning when the rule runs after pooling")
        if after:
            if float(getattr(src, "transfer_floor", 0.0) or 0.0):
                raise ValueError("shrink_stage='pooled' with a transfer floor: tau^2 is not "
                                 "sampling variance, so 1/sum(w) would not be the pooled one")
        else:
            if want is None:
                want = shrinkage
            if want:
                fc = shrink(fc, var_rule, shrink_k)
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
            # a per-source scalar on the weight: the cosine of its control profile to the target's, to the power
            # beta (0 = uniform)
            get_ctrl = getattr(src, "control_cpm", None)
            if get_ctrl is None:
                # an LfcTable has no control profile: full weight, counted
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
                # a source with no cells behind it (LfcTable) keeps full weight, counted
                stats["coverage_sources_unweighted"] = (
                    stats.get("coverage_sources_unweighted", 0) + 1)
        tau2 = float(getattr(src, "transfer_floor", 0.0) or 0.0)
        if tau2:
            # tau^2 added to the variance after the coverage tier: w / (1 + tau2*w) = 1/(v + tau2), so w is capped
            # at 1/tau^2
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
        if voted:
            # the strength scales the fold change and leaves the weight at its unscaled value; the vote
            # scales the weight; the pooled variance's numerator takes both squared
            if sc != 1.0:
                fc = fc * sc
            den2 += remap_to_axis(w * (m * m * sc * sc), src.genes, axis, fill=0.0)
            if m != 1.0:
                w = w * m
            weighted = weighted or m != 1.0 or sc != 1.0
        num += remap_to_axis(fc * w, src.genes, axis, fill=0.0)
        den += remap_to_axis(w, src.genes, axis, fill=0.0)
        if den_rule is not None:
            # the shipped variances' own 1/sum(w), with the same abstention pattern: a model
            # variance is finite exactly where the shipped one is
            with np.errstate(divide="ignore"):
                w_rule = 1.0 / np.maximum(var_rule, clamp)
            w_rule = np.where(np.isfinite(var_rule), w_rule, 0.0)
            den_rule += remap_to_axis(w_rule, src.genes, axis, fill=0.0)
    if not any_src:
        return None
    out = np.zeros(len(axis))
    nz = den > 0
    if stats is not None and not nz.any():
        stats["targets_zero_weight"] = stats.get("targets_zero_weight", 0) + 1
    out[nz] = num[nz] / den[nz]
    if after:
        # one rule on the pooled vector, with the variance of an inverse-variance mean
        if den_rule is not None:
            with np.errstate(divide="ignore"):
                pooled_var = np.where(den_rule[nz] > 0, 1.0 / np.maximum(den_rule[nz], 1e-300), np.inf)
        elif weighted:
            # the variance of sum(m w S x) / sum(m w) with w = 1/v: sum(m^2 S^2 w) / (sum m w)^2
            pooled_var = den2[nz] / den[nz] ** 2
        else:
            pooled_var = 1.0 / den[nz]
        out[nz] = (adaptive_shrink(out[nz], pooled_var, stats=stats) if shrink_rule == "adaptive"
                   else shrink(out[nz], pooled_var, shrink_k))
    return out


def variance_kwargs(args) -> dict:
    """The parsed variance flags as `pooled_delta`'s keyword arguments."""
    vm = getattr(args, "_variance_model", None)
    if vm is None:
        vm = VarianceModel.parse(getattr(args, "variance_model", "shipped"))
    return {"variance_model": vm, "rule_variance": getattr(args, "rule_variance", "model")}


def add_source_vote_args(ap: argparse.ArgumentParser, *, twin: str) -> None:
    """The vote weight and the strength per source (`sidechain.submit.source_vote`), shared by `submit.build`
    and `eval.loco`."""
    ap.add_argument("--source-weight", action="append", default=[], metavar="NAME=CTX:W",
                    help="multiply the pooling weight of the source whose file stem is NAME by W >= 0 in context "
                         "CTX (repeatable); W 0 leaves it out of that context's pool. Contexts that vote alike "
                         "share one pool. Multiplies with --source-weight-table. Recorded in "
                         f"<out>.source_weight.json (sidechain.submit.source_vote; same flag on {twin})")
    ap.add_argument("--source-weight-table", type=Path, default=None, metavar="TABLE",
                    help="a parquet or CSV with `source`, `context`, `weight` and optionally `target`: the vote "
                         "per (source, context) or per (source, context, knockdown); an unnamed pair keeps 1. "
                         f"Recorded with its sha256 in <out>.source_weight.json (same flag on {twin})")
    ap.add_argument("--source-scale", action="append", default=[], metavar="NAME=S",
                    help="multiply the fold change of the source whose file stem is NAME by S > 0 before pooling "
                         "(its variance by S^2), its pooling weight kept (repeatable). Recorded in "
                         f"<out>.source_scale.json (same flag on {twin})")


def add_shrink_args(ap: argparse.ArgumentParser, *, twin: str) -> None:
    """The three shrinkage-rule flags, shared by `submit.build` and `eval.loco`."""
    ap.add_argument("--shrink-k", type=float, default=1.0, metavar="K",
                    help="the garrote's threshold (unitless): a transferred log2FC within sqrt(K) standard errors "
                         "of zero is set to zero, a larger one keeps 1 - K*var/fc^2 of itself. Same flag on "
                         f"{twin}")
    ap.add_argument("--shrink-stage", choices=SHRINK_STAGES, default="source",
                    help="where the rule runs: on each source's fold change before pooling (source), or once on "
                         "the pooled vector with the pooled sampling variance (pooled). The pooled stage is "
                         "refused with --shrink-source, --gamma != 1, --coverage-tiers, a similarity weight or a "
                         "transfer floor.")
    ap.add_argument("--shrink-rule", choices=SHRINK_RULES, default="garrote",
                    help="garrote (the default) or adaptive: the posterior mean under a mixture "
                         "prior fitted across each target's genes (sidechain.models."
                         "adaptive_shrink), which shrinks a weak gene without zeroing it. "
                         "adaptive needs --shrink-stage pooled and takes no --shrink-k.")


def check_shrink_args(ap: argparse.ArgumentParser, args) -> None:
    """Refuse an inert or undefined shrinkage setting before any work; both entry points."""
    moved = [flag for flag, on in (("--shrink-k", args.shrink_k != 1.0),
                                   ("--shrink-stage", args.shrink_stage != "source"),
                                   ("--shrink-rule", args.shrink_rule != "garrote")) if on]
    if args.no_shrink and args.shrink_source:
        # the depth-aware arm: the rule runs on the named sources only, at --shrink-k
        moved = [flag for flag in moved if flag != "--shrink-k"]
    if moved and args.no_shrink:
        ap.error(f"{', '.join(moved)} with --no-shrink: the rule is switched off, so the "
                 "setting would do nothing and the arm would be named for a rule it never ran")
    try:
        check_shrink_rule(args.shrink_k, args.shrink_stage, args.shrink_rule)
    except ValueError as err:
        ap.error(str(err))
    if args.shrink_stage == "pooled":
        clash = [flag for flag, on in (
            ("--shrink-source", bool(args.shrink_source)),
            ("--gamma", getattr(args, "gamma", 1.0) != 1.0),
            ("--coverage-tiers", getattr(args, "coverage_tiers", None) is not None),
            ("--similarity-beta", getattr(args, "similarity_beta", 0.0) != 0.0),
            ("--transfer-floor", bool(getattr(args, "transfer_floor", None)))) if on]
        if clash:
            ap.error(f"--shrink-stage pooled with {', '.join(clash)}: the pooled rule divides by "
                     "the pooled sampling variance, which each of these changes")


def shrink_kwargs(args) -> dict:
    """The parsed shrinkage flags as `pooled_delta`'s keyword arguments."""
    return {"shrinkage": not args.no_shrink, "shrink_k": args.shrink_k,
            "shrink_stage": args.shrink_stage, "shrink_rule": args.shrink_rule}


def write_shrink_record(path: Path, args, target_stats: dict, pool_fit_stats: dict | None) -> dict:
    """Write the sidecar a build under a moved shrinkage rule leaves beside the .vcc (`<stem>.shrink.json`).

    The rule; for the adaptive rule also the fit's stopping constants, how its fits ended (targets and neighbour
    pool apart), and the numpy version, BLAS and thread settings.
    """
    import os

    rec = {"shrink_k": args.shrink_k, "shrink_stage": args.shrink_stage, "shrink_rule": args.shrink_rule}
    if args.shrink_rule == "adaptive":
        pick = lambda st: {k: v for k, v in (st or {}).items() if k.startswith("adaptive_")}
        try:
            blas = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {}).get("name")
        except Exception:          # older numpy: no dict form
            blas = None
        rec["adaptive_fit"] = {"max_cycles": adaptive_shrink_module.MAX_CYCLES,
                               "tol_per_gene": adaptive_shrink_module.TOL_PER_GENE,
                               "calm_cycles": adaptive_shrink_module.CALM_CYCLES,
                               "min_genes": adaptive_shrink_module.MIN_GENES,
                               "max_components": adaptive_shrink_module.MAX_COMPONENTS}
        rec["fits_targets"] = pick(target_stats)
        rec["fits_neighbour_pool"] = pick(pool_fit_stats) if pool_fit_stats is not None else None
        rec["numerics"] = {"numpy": np.__version__, "blas": blas,
                           **{k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                              "MKL_NUM_THREADS")}}
    path.write_text(json.dumps(rec, indent=1) + "\n")
    if args.shrink_rule == "adaptive":
        t, q = rec["fits_targets"], rec["fits_neighbour_pool"] or {}
        print(f"adaptive shrinkage: targets {t.get('adaptive_fits', 0)} fits ({t.get('adaptive_fits_at_cap', 0)} at "
              f"the cap, {t.get('adaptive_too_few_genes', 0)} too thin to fit); neighbour pool "
              f"{q.get('adaptive_fits', 0)} fits ({q.get('adaptive_fits_at_cap', 0)} at the cap, "
              f"{q.get('adaptive_too_few_genes', 0)} too thin)", flush=True)
    return rec


def add_neighbour_args(ap: argparse.ArgumentParser, *, twin: str) -> None:
    """The neighbour-arm flags, shared by `submit.build` and `eval.loco`."""
    ap.add_argument("--neighbour-table", type=Path, action="append", default=None, metavar="PT",
                    help="symbol-keyed gene table (.pt, {symbol: vector}) whose cosine neighbourhoods pick each "
                         "target's neighbours (sidechain.models.neighbour_arm). Repeatable, one --neighbour-w per "
                         "table in the same order; two tables blend both neighbour sets. Needs --neighbour-pool. "
                         f"Same flag on {twin}.")
    ap.add_argument("--neighbour-pool", type=Path, default=None, metavar="CSV",
                    help="the targets neighbours are drawn from (column target_gene, else the first). Members no "
                         "source covers or the table lacks are dropped and counted.")
    ap.add_argument("--neighbour-k", type=int, default=25, metavar="K",
                    help="neighbours per target")
    ap.add_argument("--neighbour-w", type=float, action="append", default=None, metavar="W",
                    help="weight of the neighbour arm against the pooled delta's unit residual (unitless): out = "
                         "m + |r| unit(unit(r) + w unit(n)) under --neighbour-size unit, w n/s in place of w "
                         "unit(n) under median. Repeatable, one per --neighbour-table. Unset = off.")
    # the two spellings of models.neighbour_arm.SIZES, named here so that off still never
    # imports that module (tests/test_neighbour_arm.py pins the two lists equal)
    ap.add_argument("--neighbour-size", choices=("unit", "median"), default="unit",
                    help="how long the neighbour arm is in the blend: 'unit' (default), every target's arm the "
                         "same length; 'median', out = m + |r| unit(unit(r) + w n/s) with n the plain mean of the "
                         "k neighbours' residuals and s the median residual length over the pool. Needs "
                         "--neighbour-w.")
    # the spellings of models.neighbour_arm.SELECTS, named here for the same reason as the sizes
    ap.add_argument("--neighbour-select", choices=("table", "hybrid", "response", "euclid",
                                                      "constant"),
                    default="table",
                    help="which k pool members are averaged: 'table' (default), the k nearest in the gene table; "
                         "'hybrid', the table's --neighbour-cand nearest, keeping the k whose residual responses "
                         "in the sources point most like the target's own (cosine); 'response', the whole pool "
                         "ranked by that cosine; 'euclid', by the Euclidean distance between residuals; "
                         "'constant', nobody, every target's arm being minus the pool mean. One table only. Needs "
                         "--neighbour-w.")
    ap.add_argument("--neighbour-cand", type=int, default=100, metavar="N",
                    help="--neighbour-select hybrid: how many of the table's nearest are "
                         "re-ranked by response (default 100)")
    ap.add_argument("--neighbour-picks", type=Path, default=None, metavar="JSON",
                    help="a declared pick: {target: {\"members\": [pool labels], \"weights\": [...] or null}}, built "
                         "from the same sources as the pool. Each target is averaged over exactly those members "
                         "(weighted when weights are given); the file must name every target the build fuses. One "
                         "table only, --neighbour-select table only. Needs --neighbour-w.")


def check_neighbour_args(ap: argparse.ArgumentParser, args) -> None:
    """Refuse a half-set neighbour arm before any work, in both entry points."""
    ws, tables, pool = args.neighbour_w or [], args.neighbour_table or [], args.neighbour_pool
    size = getattr(args, "neighbour_size", "unit")
    select = getattr(args, "neighbour_select", "table")
    cand = int(getattr(args, "neighbour_cand", 100))
    picks = getattr(args, "neighbour_picks", None)
    for w in ws:
        if not math.isfinite(w) or w <= 0:
            ap.error(f"--neighbour-w must be finite and > 0, got {w}: a negative weight steers "
                     "AWAY from the neighbours, which no gate measured, inf or nan fuse every "
                     "target into NaN, and 0 is the knob off -- leave the flag out")
    if not ws:
        half = [f for f, on in (("--neighbour-table", bool(tables)),
                                ("--neighbour-pool", pool is not None),
                                ("--neighbour-size median", size == "median"),
                                (f"--neighbour-select {select}", select != "table"),
                                (f"--neighbour-cand {cand}", cand != 100),
                                ("--neighbour-picks", picks is not None)) if on]
        if half:
            one = len(half) == 1
            ap.error(f"{', '.join(half)} without --neighbour-w "
                     f"{'does' if one else 'do'} nothing; give a weight or "
                     f"drop {'it' if one else 'them'}")
        return
    if not tables or pool is None:
        ap.error("--neighbour-w needs both --neighbour-table and --neighbour-pool")
    if len(tables) != len(ws):
        ap.error(f"one --neighbour-w per --neighbour-table, in the same order: got "
                 f"{len(tables)} tables and {len(ws)} weights")
    for flag, path in ([("--neighbour-table", t) for t in tables] + [("--neighbour-pool", pool)]
                       + ([("--neighbour-picks", picks)] if picks is not None else [])):
        if not Path(path).expanduser().exists():
            ap.error(f"{flag}: no such file {path}")
    if args.neighbour_k < 1:
        ap.error(f"--neighbour-k must be >= 1, got {args.neighbour_k}")
    if len(tables) > 1 and (select != "table" or picks is not None):
        ap.error("--neighbour-select other than table and --neighbour-picks are wired for one "
                 f"--neighbour-table, not for a mix of {len(tables)}: it is not defined which "
                 "table's neighbours they would replace")
    if picks is not None and select != "table":
        ap.error(f"--neighbour-picks names every target the build fuses, so there is nothing "
                 f"left for --neighbour-select {select} to rank: pass one or the other")
    if select == "hybrid" and cand < args.neighbour_k:
        ap.error(f"--neighbour-select hybrid keeps k of the table's --neighbour-cand nearest, so "
                 f"cand must be >= k: got cand = {cand}, k = {args.neighbour_k}")
    # read the declared inputs before any pooling
    if picks is not None:
        parsed = None
        try:
            parsed = json.loads(Path(picks).expanduser().read_text())
        except (OSError, ValueError) as exc:
            ap.error(f"--neighbour-picks: {picks} could not be read as JSON ({exc})")
        if not isinstance(parsed, dict):
            ap.error(f"--neighbour-picks: {picks} must be a mapping of target -> "
                     f'{{"members": [...], "weights": [...] or null}}, got a '
                     f"{type(parsed).__name__}")
        for target, entry in parsed.items():
            if not isinstance(entry, dict):
                ap.error(f"--neighbour-picks: {picks} entry {target!r} must be a mapping "
                         f'carrying "members", got a {type(entry).__name__}')
            if "members" not in entry:
                ap.error(f"--neighbour-picks: {picks} entry {target!r} is a mapping without "
                         f'"members" (it carries {sorted(entry)})')
    if select == "hybrid":
        from sidechain.models.neighbour_arm import read_pool

        # the pool the arm builds can only be SMALLER than its file (members the table lacks or
        # no source covers are dropped), so a cand the file cannot reach never can
        n_declared = len(read_pool(pool))
        if cand >= n_declared:
            ap.error(f"--neighbour-select hybrid re-ranks the table's cand nearest OTHER members, "
                     f"so cand must be < the pool's size: got cand = {cand} and "
                     f"{n_declared} labels in {pool} (the pool as built can only be smaller)")
    if args.gamma != 1.0:
        ap.error("--neighbour-w with --gamma != 1 is not wired: gamma makes the shifts "
                 "context-specific, and the pool's residuals would have to be re-pooled per "
                 "context")


def neighbour_arm_for(args, delta_of, axis: np.ndarray):
    """Build the arm from parsed flags; returns ``(arm, record)`` for the run's sidecar."""
    import hashlib

    from sidechain.models.neighbour_arm import build_neighbour_arms, load_gene_table, read_pool

    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest()

    def as_list(x):
        return list(x) if isinstance(x, (list, tuple)) else [x]

    table_paths = [Path(t).expanduser() for t in as_list(args.neighbour_table)]
    ws = [float(w) for w in as_list(args.neighbour_w)]
    pool_path = Path(args.neighbour_pool).expanduser()
    # getattr: eval.loco's SimpleNamespace and older callers predate the size flag
    size = getattr(args, "neighbour_size", "unit")
    select = getattr(args, "neighbour_select", "table")
    cand = int(getattr(args, "neighbour_cand", 100))
    picks_path = getattr(args, "neighbour_picks", None)
    picks = None
    if picks_path is not None:
        picks_path = Path(picks_path).expanduser()
        picks = json.loads(picks_path.read_text())
    arm = build_neighbour_arms(read_pool(pool_path), delta_of, axis,
                               [load_gene_table(t) for t in table_paths],
                               k=args.neighbour_k, ws=ws, size=size, select=select, cand=cand,
                               picks=picks)
    # one table: scalar fields; a mix: lists
    one = len(table_paths) == 1
    record = {"table": str(table_paths[0]) if one else [str(t) for t in table_paths],
              "table_sha256": (sha256(table_paths[0]) if one
                               else [sha256(t) for t in table_paths]),
              # the flag as parsed; consumers merge arm.summary(), which re-states the size, over it
              "size": size,
              "pool_file": str(pool_path), "pool_sha256": sha256(pool_path)}
    if picks_path is not None:       # a declared pick is an input like the pool: path and hash
        record.update({"picks_file": str(picks_path), "picks_sha256": sha256(picks_path)})
    return arm, record


def fuse_neighbours(args, shifts: dict, covered: list[str], sources: list, axis: np.ndarray,
                    cov_tiers, record_path: Path, fit_stats: dict | None = None,
                    pool_kw: dict | None = None) -> dict:
    """Replace each covered target's pooled delta with its neighbour-fused one, in place, and write the arm's record.

    The pool is pooled as the targets were; a pool member that is also a target reuses its delta. A target no
    source covers keeps the generic fallback shift, unfused.
    """
    t0 = time.time()
    done = set(covered)

    def delta_of(label):
        if label in done:
            return shifts[label]
        return pooled_delta(label, sources, axis, **shrink_kwargs(args), **variance_kwargs(args),
                            log_bias_correct=args.log_bias_correct, var_floor=args.var_floor,
                            coverage_tiers=cov_tiers, stats=fit_stats, **(pool_kw or {}))

    arm, record = neighbour_arm_for(args, delta_of, axis)
    for p in covered:
        shifts[p] = arm.fuse(p, shifts[p])
    record.update(arm.summary())
    record["targets_unfused_fallback"] = len(shifts) - len(covered)
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"neighbour arm: {record.get('targets_fused', 0)} of {len(covered)} targets fused "
          f"(k={record['k']}, w={record['w']}, pool {record['pool_used']} of "
          f"{record['pool_requested']}) in {time.time() - t0:.0f}s", flush=True)
    return record


def scatter_pairs(tab: pd.DataFrame, axis: np.ndarray, perts: list[str], name: str) -> tuple[dict, dict]:
    """The pairs of one per-(perturbation, gene) scatter table that change something, and its counts.

    `tab` carries `target`, `feature` and `scatter` >= 0 (1 = as emitted, 0 = no sampling scatter, above 1 wider).
    Rows off `perts`, off `axis` or on the target's own gene are dropped and counted; a pair listed twice is
    refused. Returns `{target: (gene positions, values)}`; shared with `sidechain.eval.loco`.
    """
    target, feature = tab["target"].astype(str), tab["feature"].astype(str)
    val = tab["scatter"].to_numpy(dtype=np.float64)
    if not np.isfinite(val).all() or (val < 0).any():
        raise SystemExit(f"--scatter-table {name}: scatter must be finite and >= 0")
    if pd.DataFrame({"target": target, "feature": feature}).duplicated().any():
        raise SystemExit(f"--scatter-table {name}: a (target, feature) pair is listed twice")
    pos = {g: i for i, g in enumerate(axis)}
    known = set(perts)
    off_target = ~target.isin(known).to_numpy()
    off_axis = ~feature.isin(pos).to_numpy()
    own = (target == feature).to_numpy()
    keep = ~(off_target | off_axis | own) & (val != 1.0)
    out = {}
    kept = pd.DataFrame({"target": target[keep], "feature": feature[keep], "scatter": val[keep]})
    for t, block in kept.groupby("target", sort=False):
        out[t] = (np.array([pos[g] for g in block["feature"]], dtype=np.int64),
                  block["scatter"].to_numpy(dtype=np.float64))
    sizes = [len(v[0]) for v in out.values()]
    record = {"rows": len(tab), "pairs_applied": int(keep.sum()),
              "targets_with_a_pair": len(out), "pairs_per_target_median": float(np.median(sizes)) if sizes else 0.0,
              "rows_at_scatter_zero": int(((val == 0.0) & keep).sum()),
              "rows_below_one": int(((val < 1.0) & keep).sum()),
              "rows_above_one": int(((val > 1.0) & keep).sum()),
              "rows_dropped": {"target_not_in_this_file": int(off_target.sum()),
                               "gene_not_on_the_axis": int((off_axis & ~off_target).sum()),
                               "the_targets_own_gene": int((own & ~off_target & ~off_axis).sum())}}
    return out, record


def read_scatter_frame(path: Path) -> pd.DataFrame:
    """A scatter table off disk: the three columns present, a context index made a column, a context column under
    another spelling refused, every value finite and not negative.
    """
    tab = pd.read_parquet(path)
    if "context" in tab.index.names:
        tab = tab.reset_index()
    odd = [c for c in tab.columns if str(c).casefold() == "context" and c != "context"]
    if odd:
        raise SystemExit(f"--scatter-table {path.name}: a column named {odd[0]!r} -- the context "
                         "column is spelled 'context'")
    missing = [c for c in ("target", "feature", "scatter") if c not in tab.columns]
    if missing:
        raise SystemExit(f"--scatter-table {path.name}: no column {missing}; it needs target, "
                         "feature and scatter (and may carry context)")
    val = pd.to_numeric(tab["scatter"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(val).all() or (val < 0).any():
        raise SystemExit(f"--scatter-table {path.name}: scatter must be finite and >= 0")
    return tab


def read_context_scatter_table(path: Path, axis: np.ndarray, perts: list[str],
                               contexts: list[str]) -> tuple[dict, dict]:
    """`--scatter-table` for a build: `sidechain.eval.loco`'s table, with an optional `context` column.

    With the column a row acts on its context's block only (a context not built is dropped and counted); without
    it on every context. A table none of whose pairs applies to any context is refused. Returns
    `{context: {target: (gene positions, values)}}` and the record for `<out>.dual.json`.
    """
    path = Path(path).expanduser()
    cols = ["target", "feature", "scatter"]
    tab = read_scatter_frame(path)
    has_context = "context" in tab.columns
    named = tab["context"].astype(str) if has_context else None
    out, per_context = {}, {}
    for ctx in contexts:
        part = tab[named == ctx] if has_context else tab
        out[ctx], per_context[ctx] = scatter_pairs(
            part[cols], axis, perts, f"{path.name} (context {ctx})" if has_context else path.name)
    if not any(out[ctx] for ctx in contexts):
        its = f"; the table names {', '.join(sorted(set(named))[:8])}" if has_context else ""
        raise SystemExit(f"--scatter-table {path.name}: none of its pairs applies to a context of "
                         f"this build ({', '.join(contexts)}{its}) -- its targets, genes or "
                         "contexts are another panel's, --limit-perts cut its targets, or every "
                         "value is 1")
    if has_context:
        for ctx in contexts:
            if not out[ctx]:
                print(f"WARNING --scatter-table {path.name}: no pair of it applies to context "
                      f"{ctx}; its blocks are emitted without the table", flush=True)
    off_context = int((~named.isin(set(contexts))).sum()) if has_context else 0
    record = {"table": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "rows": len(tab), "context_column": has_context,
              "rows_naming_a_context_not_built": off_context, "contexts": per_context}
    return out, record


class BulkAtCeiling(dict):
    """`--alpha-bulk-target`'s summed-profile map `{target: log2FC}` at the `--alpha-bulk` ceiling, with each
    target's vector before any amplitude and its factor, so `at(a)` is the map at `--alpha-bulk a`.
    """

    def __init__(self, gene_pos: dict):
        super().__init__()
        self.gene_pos = gene_pos
        self.base: dict[str, np.ndarray] = {}
        self.factor: dict[str, float] = {}

    def keep(self, p: str, vec: np.ndarray, f_t: float) -> None:
        self.base[p] = vec.copy()
        self.factor[p] = f_t

    def at(self, alpha_bulk: float) -> dict:
        out = {}
        for p in self:
            b = self.base[p] * (alpha_bulk * self.factor[p])     # finalize's arithmetic, at this strength
            if p in self.gene_pos:
                b[self.gene_pos[p]] = TARGET_SELF_LOG2FC
            out[p] = b
        return out


KNEE_MAX_READS = 4      # --alpha-bulk-target: reads per context, the ceiling's included


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--challenge-config", default="challenges/vcc2026/config.yaml")
    ap.add_argument("--emitter", choices=["control-null", "h1-mean-shift", "delta-transfer"], required=True)
    ap.add_argument("--h1-cache")
    ap.add_argument("--gwps-cache")
    ap.add_argument("--gwps-control", default="control", metavar="LABEL",
                    help="the control label inside --gwps-cache (default 'control')")
    ap.add_argument("--source", action="append", default=[], metavar="NPZ:CONTROL",
                    help="additional pseudobulk source for delta-transfer, as .npz:control_label (repeatable; the "
                         "syntax sidechain.eval.loco takes)")
    ap.add_argument("--shrink-source", action="append", default=[], metavar="NPZ:CONTROL",
                    help="like --source, but this source's transferred log2FCs are shrunk regardless of "
                         "--no-shrink. Same syntax in sidechain.eval.loco.")
    ap.add_argument("--lfc-source", action="append", default=[], metavar="NPZ",
                    help="cached LfcTable .npz: a source publishing the contrast already taken rather than cells. "
                         "Repeatable. Built by `python -m sidechain.data.lfc_table`.")
    ap.add_argument("--alpha", type=float, default=1.0, help="scale applied to every transferred log2FC")
    ap.add_argument("--gene-factor", type=Path, default=None, metavar="TABLE",
                    help="a per-gene multiplier on the pooled log2FC vector, applied last before --alpha and the "
                         "knockdown pin, in both channels: a parquet or CSV with `gene` (axis symbols) and "
                         "`factor` (finite, >= 0); a gene the table does not name keeps 1. An optional `context` "
                         "column makes it per line: a row with a context applies only to that context's cells, a "
                         "row with an empty one to every context. Recorded in <out>.gene_factor.json with the "
                         "table's sha256 (sidechain.submit.gene_factor; the same flag on sidechain.eval.loco, "
                         "with --gene-factor-context)")
    ap.add_argument("--cell-lfc-map", type=Path, default=None, metavar="JSON",
                    help="the per-cell channel, after --alpha, through a step map on its |log2FC| (JSON `lower`, "
                         "`size`; sidechain.submit.cell_map), signs kept, the own gene pinned after it; the "
                         "summed profile is untouched, so it needs the two-channel emission. Recorded in "
                         "<out>.dual.json with the file's sha256 (the same flag on sidechain.eval.loco)")
    ap.add_argument("--alpha-table", type=Path, default=None, metavar="TABLE",
                    help="a per-target multiplier on --alpha: a parquet or CSV with `target` (perturbation "
                         "labels) and `factor` (finite, >= 0); a named perturbation's pooled log2FC vector is "
                         "scaled by --alpha times its factor, in both channels, its own gene pinned after; an "
                         "unnamed one keeps --alpha. Recorded in <out>.alpha_table.json with the table's sha256 "
                         "(sidechain.submit.alpha_table; the same flag on sidechain.eval.loco)")
    ap.add_argument("--alpha-bulk", type=float, default=None, metavar="ALPHA_BULK",
                    help="a second amplitude for the pseudobulk channel (unitless): the emitted cells' "
                         "equal-weight per-cell mean follows --alpha, their depth-weighted column sums follow "
                         "this value (count_emitters.PoissonEmitter.emit_dual; same flag on sidechain.eval.loco). "
                         "Needs a depth spread (--emit-lambda > 0 or --dispersion poisson) and the delta-transfer "
                         "emitter; a perturbation whose two moments cannot both be met carries one amplitude and "
                         "is counted. Unset = one amplitude")
    ap.add_argument("--alpha-bulk-target", type=float, default=None, metavar="R",
                    help="the knee rule: --alpha-bulk is the ceiling; each context is emitted at it in memory and "
                         "cell-eval2's budget over claim is read on those cells (sidechain.submit.knee); while it "
                         "is above R the strength is lowered, never below --alpha or --alpha-bulk-floor, and read "
                         "again (at most 4 reads), then the context is written at the strength chosen. R in (0, "
                         "1). Needs --alpha-bulk above --alpha. Recorded in <out>.alpha_bulk.json and .dual.json "
                         "(the same flag on sidechain.eval.loco)")
    ap.add_argument("--alpha-bulk-floor", type=float, default=None, metavar="FLOOR",
                    help="the lowest summed-profile strength --alpha-bulk-target may choose, in place of --alpha; "
                         "it may sit below --alpha. Needs --alpha-bulk-target and --alpha-bulk above it. Recorded "
                         "in <out>.alpha_bulk.json (the same flag on sidechain.eval.loco). Unset = --alpha")
    ap.add_argument("--bulk-anchor", choices=["mean_cpm", "pooled"], default="mean_cpm",
                    help="which control profile the pseudobulk channel is built on: 'mean_cpm' (default), the "
                         "mean of per-cell CPM; 'pooled', sum of counts / sum of depths, the profile cell-eval2's "
                         "control pseudobulk is. The per-cell channel stays on mean_cpm either way; without "
                         "--alpha-bulk the pseudobulk takes --alpha. Needs a depth spread and the delta-transfer "
                         "emitter (count_emitters.PoissonEmitter.bulk_anchor; same flag on sidechain.eval.loco)")
    ap.add_argument("--dual-fallback", choices=["template", "anchor"], default="template",
                    help="what a perturbation whose two moments cannot both be met falls back to. 'template' "
                         "(default): one amplitude on the mean-per-cell-CPM profile, so under --bulk-anchor "
                         "pooled it loses the anchor too. 'anchor': one amplitude with the summed profile kept on "
                         "the anchor, the template only if that fails as well; under --bulk-anchor mean_cpm it "
                         "re-pins the column sums to their expectation instead "
                         "(count_emitters.PoissonEmitter.emit_dual; same flag on sidechain.eval.loco)")
    ap.add_argument("--emit-shape", choices=["template", "controls"], default="template",
                    help="what every block of cells starts from: 'template' (default), the emitter's own drawn "
                         "cells; 'controls', real control cells of the context, each gene's counts re-rated to "
                         "the prediction (count_emitters.PoissonEmitter.emit_dual(shape=True)); a target no "
                         "source covers keeps its generic shift and is re-rated the same way. Needs the "
                         "two-channel emission (--alpha-bulk or --bulk-anchor pooled). Same flag on "
                         "sidechain.eval.loco")
    ap.add_argument("--scatter-table", type=Path, default=None, metavar="PARQUET",
                    help="a per-(context, perturbation, gene) dial on the emitted cell-to-cell scatter: a parquet "
                         "with target, feature, scatter >= 0 and, optionally, context; 1 = as the gene is "
                         "emitted, 0 = no sampling scatter (the gene keeps its per-cell mean and column total), "
                         "above 1 = wider. With a context column a row acts on that context's block only; without "
                         "one, on every context. Under --emit-shape controls it acts on a gene's re-rated counts. "
                         "Needs the two-channel emission (--alpha-bulk or --bulk-anchor pooled)")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="transfer exponent on the target/source control-CPM ratio (gamma_transfer; same flag on "
                         "sidechain.eval.loco): 1 = the fold change transfers, 0 = the absolute CPM change. Any "
                         "other value than 1 pools the shifts once per context, against that context's own "
                         "control profile. delta-transfer only; the H1 mean-shift fallback is not transformed.")
    ap.add_argument("--no-shrink", action="store_true", help="switch the shrinkage of transferred log2FCs off entirely (--shrink-stage and --shrink-rule are refused with it)")
    add_shrink_args(ap, twin="sidechain.eval.loco")
    ap.add_argument("--var-floor", choices=["none", "poisson"], default="none",
                    help="floor each pseudobulk arm's per-gene variance at its Poisson sampling variance and "
                         "abstain on single-cell arms; 'none' leaves the variance as observed")
    ap.add_argument("--coverage-tiers", metavar="CUT:FACTOR,...",
                    help="weight each source's per-gene vote by the cells' worth of evidence behind that gene "
                         "(n_eff), as cut:factor pairs, e.g. '5:0.2,20:0.5': below n_eff 5 a fifth of the weight, "
                         "5 to 20 half, at or above 20 all. Factors in (0, 1]. Same flag on sidechain.eval.loco.")
    ap.add_argument("--transfer-floor", action="append", default=[], metavar="NAME=TAU2",
                    help="add a per-source transfer-error floor tau^2 (log2 units squared) to that source's "
                         "variance before it becomes a pooling weight, keyed by the source file's basename stem "
                         "(repeatable); caps that source's weight at 1/tau^2. Same flag on sidechain.eval.loco.")
    ap.add_argument("--source-context", action="append", default=[], metavar="NAME=CTX[,CTX]",
                    help="the per-context source switch: the source whose file stem is NAME votes only in the "
                         "named contexts, every other source in every context (repeatable, one per source). Each "
                         "distinct pool is pooled once, neighbour arm included; <out>.routing.json records which "
                         "contexts got which sources, and a routed build writes one "
                         "<out>.neighbour.<contexts>.json per pool. Same flag on sidechain.eval.loco, with "
                         "--context")
    add_source_vote_args(ap, twin="sidechain.eval.loco")
    add_neighbour_args(ap, twin="sidechain.eval.loco")
    add_variance_args(ap, twin="sidechain.eval.loco")
    add_ctrl_axes_args(ap, twin="sidechain.eval.loco")
    add_prediction_source_args(ap, twin="sidechain.eval.loco", per_context=True)
    ap.add_argument("--limit-perts", type=int, help="build only the first N perturbations (pipeline tests)")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--dispersion", choices=["poisson", "even"], default=None,
                    help="endpoint of the emission dial (default: even); exclusive with "
                         "--emit-lambda (see count_emitters.PoissonEmitter)")
    ap.add_argument("--emit-lambda", type=float, default=None, metavar="LAM",
                    help="emission-sharpening dial in [0, 1]: 0 = even cells, 1 = poisson cells, interior values "
                         "narrow the emitted cloud toward the mean (count_emitters.PoissonEmitter). Same flag on "
                         "sidechain.eval.loco.")
    ap.add_argument("--out", required=True, help="output stem; writes <out>.h5ad and <out>.vcc")
    ap.add_argument("--no-pack", action="store_true")
    ap.add_argument("--min-libsize", type=float, default=CONTROL_MIN_LIBSIZE,
                    help="drop control cells below this depth from the library-size pool "
                         "(same knob and same default as sidechain.eval.loco)")
    ap.add_argument("--log-bias-correct", action="store_true",
                    help="add back the second-order bias of log2 of a noisy mean (`Var(m)/(2(m+c)^2 ln2)`), per "
                         "arm, before pooling")
    args = ap.parse_args(argv)
    cov_tiers = parse_coverage_tiers(args.coverage_tiers)
    check_neighbour_args(ap, args)
    check_shrink_args(ap, args)
    args._variance_model = check_variance_args(ap, args)     # parsed once, read by variance_kwargs
    variance_moved = not args._variance_model.is_shipped
    if ((args.shrink_k, args.shrink_stage, args.shrink_rule) != (1.0, "source", "garrote")
            and args.emitter != "delta-transfer"):
        ap.error("--shrink-k, --shrink-stage and --shrink-rule act on pooled per-target deltas, "
                 "so they only apply to delta-transfer")
    if variance_moved and args.emitter != "delta-transfer":
        ap.error("--variance-model acts on the pooling weights of per-target deltas, so it only "
                 "applies to delta-transfer")
    if args.neighbour_w and args.emitter != "delta-transfer":
        ap.error("--neighbour-w fuses pooled per-target deltas, so it only applies to "
                 "delta-transfer")
    if args.gamma != 1.0 and args.emitter != "delta-transfer":
        ap.error("--gamma transforms pooled per-target deltas, so it only applies to "
                 "delta-transfer")
    if args.gene_factor is not None and args.emitter != "delta-transfer":
        ap.error("--gene-factor multiplies pooled per-target deltas, so it only applies to "
                 "delta-transfer")
    if args.alpha_table is not None and args.emitter != "delta-transfer":
        ap.error("--alpha-table multiplies pooled per-target deltas, so it only applies to "
                 "delta-transfer")
    if args.emit_lambda is not None and args.dispersion is not None:
        ap.error("--dispersion and --emit-lambda are one dial (even is 0, poisson is 1) -- pass one")
    if args.emit_lambda is not None and not 0.0 <= args.emit_lambda <= 1.0:
        # the emitter's own check, before any work
        ap.error(f"--emit-lambda must be in [0, 1], got {args.emit_lambda}")
    if args.emit_lambda is None and args.dispersion is None:
        args.dispersion = "even"
    if args.alpha_bulk is not None:
        if args.emitter != "delta-transfer":
            ap.error("--alpha-bulk splits the amplitude of pooled per-target deltas, so it only "
                     "applies to delta-transfer")
        if args.dispersion == "even" or args.emit_lambda == 0.0:
            ap.error("--alpha-bulk needs a depth spread (--emit-lambda > 0 or --dispersion "
                     "poisson): at lambda 0 the pseudobulk and the per-cell mean coincide")
    if args.alpha_bulk_target is not None:
        if args.alpha_bulk is None:
            ap.error("--alpha-bulk-target lowers --alpha-bulk per context: pass --alpha-bulk (the ceiling)")
        if not 0.0 < args.alpha_bulk_target < 1.0:
            ap.error(f"--alpha-bulk-target is a budget-over-claim ratio in (0, 1), got {args.alpha_bulk_target}")
        if args.alpha_bulk_floor is None and not args.alpha_bulk > args.alpha:
            ap.error(f"--alpha-bulk-target lowers the summed strength from --alpha-bulk toward --alpha, so "
                     f"--alpha-bulk ({args.alpha_bulk:g}) must be above --alpha ({args.alpha:g})")
    if args.alpha_bulk_floor is not None:
        if args.alpha_bulk_target is None:
            ap.error("--alpha-bulk-floor is the knee rule's lowest strength: pass --alpha-bulk-target")
        if not (math.isfinite(args.alpha_bulk_floor) and args.alpha_bulk_floor > 0):
            ap.error(f"--alpha-bulk-floor must be a positive strength, got {args.alpha_bulk_floor}")
        if not args.alpha_bulk > args.alpha_bulk_floor:
            ap.error(f"--alpha-bulk-target lowers the summed strength from --alpha-bulk toward --alpha-bulk-floor, "
                     f"so --alpha-bulk ({args.alpha_bulk:g}) must be above --alpha-bulk-floor "
                     f"({args.alpha_bulk_floor:g})")
    if args.bulk_anchor != "mean_cpm":
        if args.emitter != "delta-transfer":
            ap.error("--bulk-anchor pooled moves the pseudobulk of pooled per-target deltas, so "
                     "it only applies to delta-transfer")
        if args.dispersion == "even" or args.emit_lambda == 0.0:
            ap.error("--bulk-anchor pooled needs a depth spread (--emit-lambda > 0 or "
                     "--dispersion poisson): at lambda 0 the pseudobulk and the per-cell mean "
                     "coincide")
    # the pseudobulk channel's amplitude, or None for one channel: a pooled anchor is a
    # two-channel emission even at one amplitude (the same rule as eval.loco)
    bulk_alpha = (args.alpha_bulk if args.alpha_bulk is not None
                  else args.alpha if args.bulk_anchor != "mean_cpm" else None)
    if args.dual_fallback != "template" and bulk_alpha is None:
        ap.error("--dual-fallback only acts on a two-channel emission -- pass --alpha-bulk or "
                 "--bulk-anchor pooled")
    if args.emit_shape != "template" and bulk_alpha is None:
        ap.error("--emit-shape controls needs the two-channel emission -- pass --alpha-bulk or "
                 "--bulk-anchor pooled")
    if args.scatter_table is not None and bulk_alpha is None:
        ap.error("--scatter-table acts on the two-channel emission -- pass --alpha-bulk or "
                 "--bulk-anchor pooled")
    if args.cell_lfc_map is not None and bulk_alpha is None:
        ap.error("--cell-lfc-map acts on the per-cell channel alone -- pass --alpha-bulk or "
                 "--bulk-anchor pooled")
    check_ctrl_axes_args(ap, args)
    if args.ctrl_axes and args.emitter != "delta-transfer":
        ap.error("--ctrl-axes acts on pooled per-target deltas, so it only applies to delta-transfer")
    routes = parse_source_contexts(args.source_context)  # checked against the sources once they load
    if routes and args.emitter != "delta-transfer":
        ap.error("--source-context chooses which sources pool per context, so it only applies to delta-transfer")
    voting = bool(args.source_weight or args.source_weight_table is not None or args.source_scale)
    if voting and args.emitter != "delta-transfer":
        ap.error("--source-weight / --source-weight-table / --source-scale act inside the pool, so they only "
                 "apply to delta-transfer")
    check_prediction_source_args(ap, args, per_context=True)
    pie = args.prediction_source == "pie"
    if pie and voting:
        ap.error("--prediction-source pie has no pool for --source-weight / --source-scale to act in")
    if pie and args.emitter != "delta-transfer":
        ap.error("--prediction-source pie stands where the pooled delta stands, so it only applies to delta-transfer")
    if pie and (args.gwps_cache or args.source or args.shrink_source or args.lfc_source or routes
                or args.neighbour_w or args.gamma != 1.0 or args.coverage_tiers or args.transfer_floor
                or args.dispersion_fit or variance_moved or args.log_bias_correct
                or (args.shrink_k, args.shrink_stage, args.shrink_rule) != (1.0, "source", "garrote")):
        ap.error("--prediction-source pie stands where the pooled delta stands: no --gwps-cache, --source, "
                 "--shrink-source, --lfc-source, --source-context, --neighbour-w, --gamma, --coverage-tiers, "
                 "--transfer-floor, --dispersion-fit, --variance-model, shrinkage rule or --log-bias-correct "
                 "(--h1-cache stays: its mean shift is what a target PIE has no row for carries)")

    stem = Path(args.out).name
    check_out_leaf(stem, context="submit.build", require_slug=True)
    rule_moved = (args.shrink_k, args.shrink_stage, args.shrink_rule) != (1.0, "source", "garrote")
    named = STEM_RE.match(stem) if CLAIMS_RE.match(stem) else None
    if named:
        letters = named.group(3)
        if rule_moved and args.no_shrink:
            ap.error(f"'{stem}': the depth-aware arm at a moved --shrink-k has no registered name "
                     "yet (ADR 0005) -- build it under a freeform stem")
        if rule_moved and "s" not in letters:
            ap.error(f"'{stem}': --shrink-k / --shrink-stage / --shrink-rule are off the baseline "
                     "garrote, so the name must carry the letter s (ADR 0005) -- or build under a "
                     "freeform stem")
        if "s" in letters and not rule_moved:
            ap.error(f"'{stem}' carries the letter s, which says the shrinkage rule moved off the "
                     "baseline garrote, but --shrink-k / --shrink-stage / --shrink-rule are at their "
                     "defaults")
        if "s" in letters and "n" in letters:
            ap.error(f"'{stem}' carries both n (shrinkage off) and s (another shrinkage rule): one "
                     "object, two states -- pick one")
        shape_moved = args.emit_shape != "template"
        if shape_moved and "r" not in letters:
            ap.error(f"'{stem}': --emit-shape is off the drawn template, so the name must carry the "
                     "letter r (ADR 0005) -- or build under a freeform stem")
        if "r" in letters and not shape_moved:
            ap.error(f"'{stem}' carries the letter r, which says the emitted cells start from "
                     "something other than the drawn template, but --emit-shape is at its default")
        head_moved = args.scatter_table is not None
        if head_moved and "h" not in letters:
            ap.error(f"'{stem}': --scatter-table makes a chosen set of genes callable, so the name "
                     "must carry the letter h (ADR 0005) -- or build under a freeform stem")
        if "h" in letters and not head_moved:
            ap.error(f"'{stem}' carries the letter h, which says a per-gene table makes a chosen "
                     "set of genes callable, but --scatter-table is not given")
        if args.ctrl_axes and "x" not in letters:
            ap.error(f"'{stem}': --ctrl-axes takes the context's control-cell axes out of the summed "
                     "profile, so the name must carry the letter x (ADR 0005) -- or build under a freeform stem")
        if "x" in letters and not args.ctrl_axes:
            ap.error(f"'{stem}' carries the letter x, which says the control-cell axes come out of the "
                     "summed profile, but --ctrl-axes is 0")
    if CLAIMS_RE.match(stem) and args.alpha_table is not None:
        ap.error(f"'{stem}' is named like a model, and --alpha-table has no registered knob letter yet "
                 "(ADR 0005): register the letter first, or build under a freeform stem")
    if CLAIMS_RE.match(stem) and args.alpha_bulk_target is not None and (named is None or "b" not in named.group(3)):
        ap.error(f"'{stem}' is named like a model, and --alpha-bulk-target is part of letter b's rule "
                 "(ADR 0005), but the name does not carry b: add it, or build under a freeform stem")
    if CLAIMS_RE.match(stem) and args.alpha_bulk_floor is not None:
        ap.error(f"'{stem}' is named like a model, and --alpha-bulk-floor changes letter b's rule with no "
                 "registered name yet (ADR 0005): register it first, or build under a freeform stem")
    if CLAIMS_RE.match(stem) and voting:
        ap.error(f"'{stem}' is named like a model, and --source-weight / --source-weight-table / --source-scale "
                 "have no registered knob letter yet (ADR 0005): register the letter first, or build under a "
                 "freeform stem")
    if CLAIMS_RE.match(stem) and args.cell_lfc_map is not None:
        ap.error(f"'{stem}' is named like a model, and --cell-lfc-map has no registered knob letter yet "
                 "(ADR 0005): register the letter first, or build under a freeform stem")
    if CLAIMS_RE.match(stem) and args.gene_factor is not None:
        ap.error(f"'{stem}' is named like a model, and --gene-factor has no registered knob letter yet "
                 "(ADR 0005): register the letter first, or build under a freeform stem")
    if CLAIMS_RE.match(stem) and pie:
        ap.error(f"'{stem}' is named like a model, and --prediction-source pie has no registered knob letter "
                 "yet (ADR 0005): register the letter first, or build under a freeform stem")
    if CLAIMS_RE.match(stem) and routes:
        ap.error(f"'{stem}' is named like a model, and --source-context has no registered knob letter "
                 "yet (ADR 0005): register the letter first, or build under a freeform stem")
    if CLAIMS_RE.match(stem) and variance_moved:
        ap.error(f"'{stem}' is named like a model, and --variance-model / --rule-variance have no "
                 "registered knob letter yet (ADR 0005): register the letter first, or build under "
                 "a freeform stem")
    if (CLAIMS_RE.match(stem)
            and (args.neighbour_select != "table" or args.neighbour_picks is not None)):
        ap.error(f"'{stem}' is named like a model, and --neighbour-select off its default / "
                 "--neighbour-picks have no registered knob letter yet (ADR 0005): register the "
                 "letter first, or build under a freeform stem")
    if not CLAIMS_RE.match(stem):
        print(f"note: out stem '{stem}' carries no series tag -- fine for a probe, but a "
              "board submission's stem starts with its lowercased short name (ADR 0005), "
              "e.g. ser-2n_delta4_even_noshrink_v1", flush=True)  # public-text: ok, a printed example stem (code)

    cfg = yaml.safe_load(resolve_config(args.challenge_config).read_text())
    data_dir = challenge_data_dir(cfg)
    control_files = challenge_control_files(cfg)          # refuses a context with no file, by name
    genes = pd.read_csv(data_dir / cfg["gene_names_file"]).iloc[:, 0].astype(str).tolist()
    if len(genes) != cfg["n_genes"]:
        raise SystemExit(f"gene_names.csv read as {len(genes)} genes; config says {cfg['n_genes']} -- header handling?")
    perts = pd.read_csv(data_dir / cfg["pert_counts_file"])[cfg["pert_col"]].astype(str).tolist()
    if args.limit_perts:
        perts = perts[: args.limit_perts]
    contexts = challenge_contexts(cfg)
    sub = cfg["submission"]
    contract = Contract(
        genes=genes, perturbations=perts, contexts=contexts, cells_per_pert=int(sub["cells_per_pert"]),
        pert_col=cfg["pert_col"], context_col=cfg["context_col"], control_label=cfg["control_label"],
        max_counts_per_cell=int(sub["max_counts_per_cell"]), max_cells=int(sub["max_cells"]),
        max_stored_entries=int(sub["max_stored_entries"]),
    )
    axis = np.asarray(genes)
    # the per-gene multiplier: read on the axis before any pooling; applied in `finalize`, last before alpha, one
    # vector per context
    gene_factor_of, gene_factor_record = (load_gene_factors(args.gene_factor, axis, contexts)
                                          if args.gene_factor is not None else (None, None))
    # contexts whose factors differ cannot share one pooled map: split below, after the pool
    factor_groups = (same_factor_groups(gene_factor_of, contexts) if gene_factor_of is not None
                     else [list(contexts)])
    factor_split = len(factor_groups) > 1
    # the per-target multiplier on alpha: read against the perturbation list before any pooling; applied in
    # `finalize`, with alpha
    alpha_of, alpha_table_record = (load_alpha_table(args.alpha_table, perts)
                                    if args.alpha_table is not None else (None, None))
    cell_map = CellMap.load(args.cell_lfc_map) if args.cell_lfc_map is not None else None
    if cell_map is not None and cell_map.contexts is not None:
        # one scale per context: contexts whose scales differ cannot share one pooled map either
        factor_groups = [[c for c in grp if cell_map.context_key(c) == k]
                         for grp in factor_groups for k in dict.fromkeys(cell_map.context_key(c) for c in grp)]
        factor_split = len(factor_groups) > 1
    # read before the pooling
    scatter_of, scatter_record = (read_context_scatter_table(args.scatter_table, axis, perts, contexts)
                                  if args.scatter_table is not None else (None, None))

    # the machine record of what produced this artifact
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    recorded = vars(args)
    if not args.ctrl_axes:
        # a flag left off is left out of the record
        recorded = {k: v for k, v in recorded.items() if not k.startswith("ctrl_axes")}
    if not routes:
        recorded = {k: v for k, v in recorded.items() if k != "source_context"}     # likewise
    if not voting:
        recorded = {k: v for k, v in recorded.items()                               # likewise
                    if k not in ("source_weight", "source_weight_table", "source_scale")}
    if not pie:
        recorded = {k: v for k, v in recorded.items()                               # likewise
                    if k not in ("prediction_source", "pie_parquet", "pie_context")}
    knee_target = args.alpha_bulk_target
    if knee_target is None:
        recorded = {k: v for k, v in recorded.items() if k != "alpha_bulk_target"}  # likewise
    if args.alpha_bulk_floor is None:
        recorded = {k: v for k, v in recorded.items() if k != "alpha_bulk_floor"}   # likewise
    if cell_map is None:
        recorded = {k: v for k, v in recorded.items() if k != "cell_lfc_map"}       # likewise
    out.with_suffix(".args.json").write_text(json.dumps(recorded, indent=1, default=str) + "\n")
    if gene_factor_record is not None:
        if factor_split:
            gene_factor_record = {**gene_factor_record, "factor_groups": factor_groups}
        out.with_suffix(".gene_factor.json").write_text(json.dumps(gene_factor_record, indent=1) + "\n")
    if alpha_table_record is not None:
        out.with_suffix(".alpha_table.json").write_text(json.dumps(alpha_table_record, indent=1) + "\n")

    # -- per-perturbation log2FC vectors (None = no shift)
    t0 = time.time()
    shifts: dict[str, np.ndarray | None] = {p: None for p in perts}
    fallback = 0
    pool_stats: dict = {}
    # the neighbour pool's fits under a moved shrinkage rule, counted apart from the targets'
    # (None keeps every default call exactly what it was)
    pool_fit_stats: dict | None = {} if rule_moved and not args.no_shrink and args.neighbour_w else None
    if args.emitter in ("h1-mean-shift", "delta-transfer"):
        if not args.h1_cache:
            raise SystemExit("--h1-cache is required for this emitter")
        h1 = PseudobulkSums.load(args.h1_cache)
        # named like every other source, so `--transfer-floor` can address it
        h1.sidechain_name = Path(args.h1_cache).expanduser().stem
        h1.sidechain_path = Path(args.h1_cache).expanduser()
        generic = h1_mean_shift(h1, cfg["control_label"], axis)
        shifts = {p: generic.copy() for p in perts}
    sources = None
    pools = None                   # {context: its sources} under --source-context
    votes, scales = None, {}       # each context's vote per source, each source's strength
    per_group = bool(routes)       # each context group pools into its own map (routes, or votes that differ)
    group_maps: list[dict] = []    # one per distinct pool (gamma 1): its contexts, sources and shifts
    pool_maps = group_maps         # the distinct pools, as .routing.json records them
    if pie:
        # PIE's map stands where the pooled delta stands, one map per distinct parquet context; a perturbation with
        # no row keeps the generic shift, as a target no source covers
        pie_of = parse_pie_contexts(args.pie_context, contexts)
        pie_records = {}
        for name in dict.fromkeys(pie_of[c] for c in contexts):
            ctxs = [c for c in contexts if pie_of[c] == name]
            m, rec = pie_shift_map(args.pie_parquet, name, axis, perts)
            g_shifts = {p: m.pop(p) if p in m else generic.copy() for p in perts}
            fb = rec["perturbations_missing"]
            fallback += fb
            pie_records[name] = {"contexts": ctxs, **rec}
            group_maps.append({"contexts": ctxs, "sources": [f"pie:{name}"], "shifts": g_shifts,
                               "covered": len(perts) - fb, "fallback": fb, "neighbour_record": None})
        out.with_suffix(".pie.json").write_text(json.dumps(
            {"prediction_source": "pie", "pie_context": pie_of, "maps": pie_records}, indent=1) + "\n")
    elif args.emitter == "delta-transfer":
        if not args.gwps_cache:
            raise SystemExit("--gwps-cache is required for delta-transfer")
        gwps = PseudobulkSums.load(args.gwps_cache)
        gwps.sidechain_name = Path(args.gwps_cache).expanduser().stem
        gwps.sidechain_path = Path(args.gwps_cache).expanduser()
        sources = [(gwps, args.gwps_control), (h1, cfg["control_label"])]
        sources += sources_from_specs(args.source, args.shrink_source)
        # appended: `pooled_delta` normalises both forms
        for path in args.lfc_source:
            tab = LfcTable.load(path)
            tab.sidechain_name = Path(path).expanduser().stem
            sources.append(tab)
        sources = apply_transfer_floors(sources, parse_transfer_floor(args.transfer_floor))
        # the per-source dispersion fits, checked against each source's axis and sha256
        sources = apply_dispersion_fits(sources, parse_dispersion_fits(args.dispersion_fit))
        args._variance_model.check_sources(sources)
        if variance_moved:
            out.with_suffix(".variance.json").write_text(json.dumps(
                {**args._variance_model.record(sources), "rule_variance": args.rule_variance},
                indent=1) + "\n")
        # each context's vote per source, each source's strength
        names = [source_name(s) for s in sources]
        votes, vote_record = source_votes(parse_source_weights(args.source_weight),
                                          read_source_weight_table(args.source_weight_table), contexts, names)
        scales = parse_source_scales(args.source_scale, names)
        if scales:
            out.with_suffix(".source_scale.json").write_text(json.dumps({"source_scale": scales}, indent=1) + "\n")
        # each context's own pool under --source-context, pooled once per distinct pool; unrouted, one group of
        # every context pools into `shifts` itself
        pools = route_sources(sources, routes, contexts) if routes else None
        groups = context_groups(pools, contexts) if routes else [(tuple(contexts), sources)]
        n_pools = len(groups)
        # contexts that share a pool but vote differently are pooled apart
        groups = [(sub, pool) for ctxs, pool in groups for sub in vote_groups(ctxs, votes)]
        per_group = bool(routes) or len(groups) > n_pools
        if vote_record is not None:
            vote_record["pools"] = [list(c) for c, _ in groups]
            out.with_suffix(".source_weight.json").write_text(json.dumps(vote_record, indent=1) + "\n")
        if args.gamma == 1.0:
            for ctxs, pool in groups:
                g_shifts = shifts if not per_group else {p: generic.copy() for p in perts}
                g_record = out.with_suffix(("." + "-".join(ctxs) if per_group else "") + ".neighbour.json")
                g_kw = pool_kwargs(votes, ctxs[0], scales)
                covered = []
                g_fallback = 0
                for p in perts:
                    d = pooled_delta(p, pool, axis, **shrink_kwargs(args), **variance_kwargs(args),
                                     log_bias_correct=args.log_bias_correct,
                                     var_floor=args.var_floor, coverage_tiers=cov_tiers,
                                     stats=pool_stats, **g_kw)
                    if d is None:
                        g_fallback += 1    # keep the generic shift
                    else:
                        g_shifts[p] = d
                        covered.append(p)
                if args.neighbour_w:
                    fuse_neighbours(args, g_shifts, covered, pool, axis, cov_tiers,
                                    g_record, fit_stats=pool_fit_stats, pool_kw=g_kw)
                fallback += g_fallback
                group_maps.append({"contexts": list(ctxs), "sources": [source_name(s) for s in pool],
                                   "shifts": g_shifts, "covered": len(covered), "fallback": g_fallback,
                                   "neighbour_record": g_record.name if args.neighbour_w else None})
            if variance_moved:
                # the counts the model left behind, targets and neighbour pool apart
                rec_path = out.with_suffix(".variance.json")
                rec = json.loads(rec_path.read_text())
                rec["target_counts"] = {k: v for k, v in pool_stats.items() if k.startswith("variance_")}
                rec["neighbour_pool_counts"] = {k: v for k, v in (pool_fit_stats or {}).items()
                                                if k.startswith("variance_")}
                rec_path.write_text(json.dumps(rec, indent=1) + "\n")
    gene_pos = {g: i for i, g in enumerate(genes)}

    def finalize(shift_map, gene_factor, context=None):
        # alpha scales the pooled vector; gamma (if any) acted per source inside the pool.
        # With --alpha-bulk the SAME pooled vector is also scaled by the pseudobulk amplitude
        # (taken before alpha touches it) and pinned identically; the emitter gets both.
        bulk_map = (None if bulk_alpha is None
                    else {} if knee_target is None else BulkAtCeiling(gene_pos))
        cm = None if cell_map is None else cell_map.bind(genes, perts, context=context)
        for p, vec in shift_map.items():
            if vec is not None:
                if gene_factor is not None:
                    vec *= gene_factor         # last before alpha, so both channels carry it
                # the per-target factor rides on alpha in both channels
                f_t = alpha_of[p] if alpha_of is not None else 1.0
                if bulk_map is not None:
                    b = vec * (bulk_alpha * f_t)
                    if p in gene_pos:
                        b[gene_pos[p]] = TARGET_SELF_LOG2FC
                    bulk_map[p] = b
                    if knee_target is not None:
                        bulk_map.keep(p, vec, f_t)     # the vector before any amplitude, for a lower one
                vec *= args.alpha * f_t
                if cm is not None:
                    vec[:] = cm.apply(vec, target=p)       # the per-cell channel only, after alpha
                if p in gene_pos:
                    vec[gene_pos[p]] = TARGET_SELF_LOG2FC
        return bulk_map

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
    bulk_shifts = None
    if args.emitter == "delta-transfer" and args.gamma != 1.0:
        # gamma re-expresses each source's fold change against THIS context's control
        # profile, so the shifts stop being shareable across contexts: pool once per
        # context, inside the write loop, on the profile the emitter itself anchors on
        # (the axis-vs-gene_names check has already run by the time this is called).
        def per_context_shifts(prof):
            local = {p: generic.copy() for p in perts}
            fb = 0
            ctrl_cpm = prof.fraction * 1e6
            ctx_pool = sources if pools is None else pools[prof.name]
            ctx_kw = pool_kwargs(votes, prof.name, scales)
            for p in perts:
                d = pooled_delta(p, ctx_pool, axis, **shrink_kwargs(args), **variance_kwargs(args),
                                 log_bias_correct=args.log_bias_correct,
                                 var_floor=args.var_floor, coverage_tiers=cov_tiers,
                                 gamma=args.gamma, ctrl_tgt_cpm=ctrl_cpm,
                                 stats=pool_stats, **ctx_kw)
                if d is None:
                    fb += 1
                else:
                    local[p] = d
            local_bulk = finalize(local, None if gene_factor_of is None else gene_factor_of[prof.name],
                                  context=prof.name)
            print(pool_line(f"  {prof.name}: gamma={args.gamma:g} shifts ready", fb), flush=True)
            return local, local_bulk

        print(f"gamma={args.gamma:g}: shifts are context-specific and pooled inside the "
              "write loop", flush=True)
    elif per_group or pie or factor_split:
        if factor_split:
            # a per-line gene factor: each pool's contexts split by their factor, a copy of the pooled map for every
            # part after the first (the pool itself is not re-run)
            group_maps = split_by_factor(group_maps, factor_groups)
        for g in group_maps:
            g["bulk"] = finalize(g["shifts"], None if gene_factor_of is None else gene_factor_of[g["contexts"][0]],
                                 context=g["contexts"][0])
            print(f"  contexts {', '.join(g['contexts'])}: {len(g['sources'])} sources "
                  f"({', '.join(map(str, g['sources']))}); fallback-to-generic: {g['fallback']}", flush=True)
        print(pool_line(f"shifts ready in {time.time() - t0:.0f}s for {len(group_maps)} pools", fallback),
              flush=True)
    else:
        bulk_shifts = finalize(shifts, None if gene_factor_of is None else gene_factor_of[contexts[0]],
                               context=contexts[0])
        print(pool_line(f"shifts ready in {time.time() - t0:.0f}s", fallback), flush=True)
    if routes:
        # which contexts pooled which sources; the shifts themselves are in the cells
        out.with_suffix(".routing.json").write_text(json.dumps(
            {"source_context": {n: list(c) for n, c in routes.items()},
             "pools": ([{k: v for k, v in g.items() if k not in ("shifts", "bulk")} for g in pool_maps]
                       if group_maps else
                       [{"contexts": [c], "sources": [source_name(s) for s in pools[c]]} for c in contexts])},
            indent=1) + "\n")
    by_context = ({c: (g["shifts"], g["bulk"]) for g in group_maps for c in g["contexts"]}
                  if per_group or pie or factor_split else None)
    if rule_moved and not args.no_shrink:
        # Written as soon as the shifts exist, before the write loop: every fit is done by now (the
        # adaptive rule is refused with gamma != 1, the one path that pools inside the loop).
        write_shrink_record(out.with_suffix(".shrink.json"), args, pool_stats, pool_fit_stats)

    # -- write
    h5ad = out.with_suffix(".h5ad")
    if args.limit_perts:
        pd.DataFrame({cfg["pert_col"]: perts}).to_csv(out.with_suffix(".pert_counts.csv"), index=False)
    t0 = time.time()
    dual_fallbacks: dict[str, int] = {}
    # which perturbations fell back, per context and per rung, with why the requested pair failed
    fell_back: dict[str, dict[str, list]] = {}
    on_fail = "fallback" if args.dual_fallback == "template" else "anchor"
    # every block as real control cells re-rated to the prediction
    shaped = args.emit_shape == "controls"
    shape_kw = {"shape": True} if shaped else {}
    in_shape: dict[str, dict] = {}
    ctrl_axes_record = ({"k": args.ctrl_axes, "eta": args.ctrl_axes_eta, "basis": args.ctrl_axes_basis,
                         "share": args.ctrl_axes_share, "contexts": {}} if args.ctrl_axes else None)

    def draw(em, ctx, p, ctx_shifts, ctx_bulk):
        """One (context, perturbation) block; whether it went through the two-channel emission, and with
        the scatter table."""
        if ctx_bulk is None or ctx_shifts[p] is None:
            return em.emit(contract.cells_per_pert, ctx_shifts[p]), False, False
        # two amplitudes in one count matrix; a perturbation whose two moments are jointly unreachable carries one
        # amplitude and is counted
        scatter_kw = {}
        if scatter_of is not None and p in scatter_of[ctx]:
            scatter = np.ones(len(axis))
            scatter[scatter_of[ctx][p][0]] = scatter_of[ctx][p][1]
            scatter_kw = {"scatter": scatter}
        block = em.emit_dual(contract.cells_per_pert, ctx_shifts[p], ctx_bulk[p],
                             on_fail=on_fail, **shape_kw, **scatter_kw)
        return block, True, bool(scatter_kw)

    # --alpha-bulk-target: per context, the summed strength the knee rule chose and how it chose it
    knee_record = ({"alpha": args.alpha, "alpha_bulk_ceiling": args.alpha_bulk, "alpha_bulk_target": knee_target,
                    "max_reads": KNEE_MAX_READS, "target_sum": KNEE_TARGET_SUM, "aim_margin": KNEE_AIM_MARGIN,
                    "undershoot": KNEE_UNDERSHOOT,
                    "rule": "alpha_bulk' = alpha_bulk * sqrt((1 - aim_margin) * target / ratio), never below "
                            f"{'alpha' if args.alpha_bulk_floor is None else 'alpha_bulk_floor'}; "
                            "between two reads that straddle the target, the power law through both, solved for "
                            "(1 - aim_margin) * target; a read under (1 - undershoot) * target below the ceiling "
                            "steps back up; the strongest read at or under the target is written; "
                            "ratio = cell-eval2 budget / claim (uncapped) on the context's emitted cells",
                    **({"alpha_bulk_floor": args.alpha_bulk_floor} if args.alpha_bulk_floor is not None else {}),
                    "contexts": {}} if knee_target is not None else None)
    alpha_bulk_of: dict[str, float] = {}

    def choose_alpha_bulk(prof, ci, ctx, ctx_shifts, ctx_bulk, deflate):
        """Emit the context in memory from the ceiling down until its budget over claim is at most the target;
        returns the strength, its record, and the column sums of the cells read at it."""
        ceiling = args.alpha_bulk
        floor = args.alpha if args.alpha_bulk_floor is None else args.alpha_bulk_floor
        a, reads, sums = ceiling, [], []
        while True:
            m = ctx_bulk if a == ceiling else ctx_bulk.at(a)
            if deflate is not None:
                m = deflate(m)
            pem = PoissonEmitter(prof, seed=args.seed + ci, dispersion=args.dispersion,
                                 lam=args.emit_lambda, bulk_anchor=args.bulk_anchor)   # the write pass's seed
            meter = KneeMeter(perts, axis)
            for p in perts:
                meter.add(p, draw(pem, ctx, p, ctx_shifts, m)[0])
            r = {"alpha_bulk": a, **meter.read(), "dual_fallbacks": int(getattr(pem, "dual_fallbacks", 0))}
            reads.append(r)
            sums.append(meter.sums)
            ratio = r["ratio_budget_over_claim"]
            print(f"  {ctx}: knee read {len(reads)}: alpha_bulk={a:.4f} budget {r['budget']:.5f} claim "
                  f"{r['claim_uncapped']:.5f} ratio {ratio:.4f} (target {knee_target:g})  "
                  f"{time.time() - t0:.0f}s", flush=True)
            pairs = [(x["alpha_bulk"], x["ratio_budget_over_claim"]) for x in reads]
            nxt = knee_next(pairs, knee_target, floor, ceiling, KNEE_MAX_READS)
            if nxt is None:
                break
            a = nxt
        kept = knee_keep(pairs, knee_target)        # an undershoot stepped back up may keep an earlier read
        a, ratio = pairs[kept]
        met = ratio <= knee_target
        if not met:
            why = (f"at the floor {'--alpha' if args.alpha_bulk_floor is None else '--alpha-bulk-floor'}" if a <= floor else
                   f"after {KNEE_MAX_READS} reads" if len(reads) >= KNEE_MAX_READS else "the ratio is not finite")
            print(f"  WARNING {ctx}: budget over claim {ratio:.4f} is above the target {knee_target:g} {why}; "
                  f"written at alpha_bulk={a:.4f}", flush=True)
        rec = {"alpha_bulk": a, "met": met, "reads": len(reads), "at_floor": a <= floor,
               "ratio_budget_over_claim": ratio, "tries": reads,
               **({"kept_read": kept + 1} if kept != len(reads) - 1 else {})}
        return a, rec, sums[kept]

    with SubmissionWriter(h5ad, contract) as w:
        for ci, ctx in enumerate(contexts):
            prof = ContextProfile.from_controls(data_dir / control_files[ctx], ctx, min_libsize=args.min_libsize,
                                                **({"keep_cells": True} if shaped else {}))
            if list(prof.genes) != genes:
                raise SystemExit(f"context {ctx} var_names differ from gene_names.csv")
            ctx_shifts, ctx_bulk = (per_context_shifts(prof) if per_context_shifts is not None
                                    else by_context[ctx] if by_context is not None
                                    else (shifts, bulk_shifts))
            em = PoissonEmitter(prof, seed=args.seed + ci, dispersion=args.dispersion,
                                lam=args.emit_lambda, bulk_anchor=args.bulk_anchor)
            if args.ctrl_axes:
                # this context's control-cell axes out of the summed-profile channel only, into a new map (the
                # contexts share `bulk_shifts`); the per-cell shifts are not touched
                from sidechain.submit.ctrl_axes import control_axes, deflate_map
                axes_v, axes_rec = control_axes(data_dir / control_files[ctx], k=args.ctrl_axes,
                                                min_libsize=args.min_libsize, basis=args.ctrl_axes_basis,
                                                cache_dir=args.ctrl_axes_cache, genes=genes)
            knee_probe = None
            if knee_target is not None:
                # this context's summed strength, chosen on its own emitted cells before any is written
                a_ctx, knee_record["contexts"][ctx], knee_probe = choose_alpha_bulk(
                    prof, ci, ctx, ctx_shifts, ctx_bulk,
                    (lambda m, f=prof.bulk_fraction, v=axes_v: deflate_map(m, f, v, args.ctrl_axes_eta, gene_pos,
                                                                         share=args.ctrl_axes_share)[0])
                    if args.ctrl_axes else None)
                if a_ctx != args.alpha_bulk:
                    ctx_bulk = ctx_bulk.at(a_ctx)
                alpha_bulk_of[ctx] = a_ctx
            if args.ctrl_axes:
                ctx_bulk, defl = deflate_map(ctx_bulk, prof.bulk_fraction, axes_v, args.ctrl_axes_eta, gene_pos,
                                             share=args.ctrl_axes_share)
                ctrl_axes_record["contexts"][ctx] = {"axes": axes_rec, "deflation": defl}
                print(f"  {ctx}: ctrl-axes {args.ctrl_axes_basis} K={args.ctrl_axes} eta={args.ctrl_axes_eta:g}: "
                      f"{axes_rec['variance_share_on_axes']:.1%} of the controls' variance on the axes; "
                      f"median {defl['share_on_axes']['median']:.1%} of a target's summed-profile change "
                      f"removed; worst residual {defl['residual_max']:.1e}", flush=True)
                del axes_v
            written = KneeMeter(perts, axis, jackknife=False) if knee_probe is not None else None
            left_on_template: list[str] = []
            held, unmet = 0, []                    # cells held to the emitter's caps; genes its solve left to the fit
            dialled: dict[str, int] = {}           # target -> genes whose scatter its block carries
            for k, p in enumerate(perts):
                block, dual, dialling = draw(em, ctx, p, ctx_shifts, ctx_bulk)
                if dual:
                    if dialling:                   # 0 on the template rung: that block carries none of it
                        dialled[p] = int(getattr(em, "last_sharpened", 0) or 0)
                    if shaped and not getattr(em, "last_shaped", False):
                        left_on_template.append(p)
                    elif shaped and getattr(em, "last_shape_unmet", None) is not None:
                        held += int(em.last_shape_held or 0)
                        unmet.append(int(em.last_shape_unmet))
                    how = getattr(em, "last_dual", "dual")
                    if how in ("anchor", "template"):
                        fell_back.setdefault(ctx, {"anchor": [], "template": []}).setdefault(
                            how, []).append([p, getattr(em, "last_dual_reason", None)])
                if written is not None:
                    written.add(p, block)
                w.add_block(block, ctx, p)
                if (k + 1) % 50 == 0:
                    print(f"  {ctx}: {k + 1}/{len(perts)} perturbations  {time.time() - t0:.0f}s", flush=True)
            if shaped:
                # a block the fit could not carry in the controls' shape falls to the drawn template
                # (the last rung), where its genes are called for their shape again: count and name it
                in_shape[ctx] = {"control_cells_kept": int(prof.n_cells),
                                 "blocks_in_the_controls_shape": len(perts) - len(left_on_template),
                                 "blocks_left_on_the_template": left_on_template,
                                 "cells_held_to_the_caps": held,
                                 "genes_left_unmet_a_block": ({"median": float(np.median(unmet)), "max": max(unmet)}
                                                              if unmet else None)}
                print(f"  {ctx}: emit-shape controls: {len(perts) - len(left_on_template)} of {len(perts)} "
                      f"perturbations emitted as re-rated control cells ({prof.n_cells} kept)"
                      + (f"; {held} cell(s) held to the caps" if held else ""), flush=True)
                if left_on_template:
                    # at one strength there is no middle rung: a failed fit returns the drawn
                    # template, whose genes the scorer calls for their shape again
                    print(f"  WARNING {ctx}: {len(left_on_template)} block(s) fell to the drawn template "
                          f"and are NOT in the controls' shape: {', '.join(left_on_template)}", flush=True)
            if scatter_record is not None:
                # `targets_carrying_it` counts the blocks whose fit took the table's genes; a target
                # that ended on the template rung carries none
                scatter_record["contexts"][ctx] |= {
                    "targets_carrying_it": int(sum(v > 0 for v in dialled.values())),
                    "pairs_carried": int(sum(dialled.values())),
                    "targets_listed_but_not_carrying": sorted(t for t, v in dialled.items() if v == 0)}
                print(f"  {ctx}: scatter table: {sum(v > 0 for v in dialled.values())} of {len(perts)} "
                      f"perturbations carry it ({sum(dialled.values())} of the "
                      f"{scatter_record['contexts'][ctx]['pairs_applied']} pairs that apply to this "
                      "context)", flush=True)
            if ctx_bulk is not None:
                # counted, never refused
                dual_fallbacks[ctx] = int(getattr(em, "dual_fallbacks", 0))
                kept = int(getattr(em, "dual_fallbacks_anchor", 0))
                tail = ("amplitude on the mean_cpm profile" if args.dual_fallback == "template" else
                        f"amplitude, {dual_fallbacks[ctx] - kept} of them from the one-amplitude template")
                print(f"  {ctx}: alpha_bulk={alpha_bulk_of.get(ctx, bulk_alpha):g} anchor={args.bulk_anchor}: "
                      f"{dual_fallbacks[ctx]} of {len(perts)} perturbations carried one {tail}",
                      flush=True)
                if args.dual_fallback == "template" and fell_back.get(ctx):
                    # names for the log only: the default rung's .dual.json keeps its shipped shape
                    print(f"  {ctx}: fell back: "
                          + ", ".join(f"{p} ({why})" for p, why in fell_back[ctx]["template"]), flush=True)
            if written is not None:
                # the cells written are the cells the knee was read on (same seed, same strength)
                same = bool(np.array_equal(written.sums, knee_probe))
                knee_record["contexts"][ctx] |= {"dual_fallbacks": dual_fallbacks[ctx],
                                                 "written_cells_are_the_cells_read": same}
                if not same:
                    raise SystemExit(f"--alpha-bulk-target: context {ctx}'s written cells are not the cells its "
                                     "knee was read on -- the emission is not reproducible at a fixed seed")
    info = verify_h5ad(h5ad, contract)
    if knee_record is not None:
        out.with_suffix(".alpha_bulk.json").write_text(json.dumps(knee_record, indent=1) + "\n")
    if ctrl_axes_record is not None:
        # beside the fallback counts in .dual.json: a deflated target that fell back carries none of it
        ctrl_axes_record["dual_fallbacks"] = dual_fallbacks
        out.with_suffix(".ctrl_axes.json").write_text(json.dumps(ctrl_axes_record, indent=1) + "\n")
    if dual_fallbacks:
        # absent bulk_anchor = mean_cpm
        anchor = {"bulk_anchor": args.bulk_anchor} if args.bulk_anchor != "mean_cpm" else {}
        # the fallback rung and its per-target list are written only off the default (the default's names go to the
        # log above)
        if args.dual_fallback != "template":
            anchor["dual_fallback"] = args.dual_fallback
            if fell_back:
                anchor["dual_fallback_targets"] = fell_back
        if shaped:                                 # written only when the knob is on
            anchor["emit_shape"] = {"shape": args.emit_shape, "contexts": in_shape}
        if scatter_record is not None:             # likewise
            anchor["scatter_table"] = scatter_record
        if cell_map is not None:                   # likewise
            anchor["cell_lfc_map"] = cell_map.record()
        if knee_record is not None:                # likewise
            anchor["alpha_bulk_target"] = knee_target
            anchor["alpha_bulk_by_context"] = alpha_bulk_of
        info = {**info, "alpha_bulk": args.alpha_bulk, **anchor, "dual_fallbacks": dual_fallbacks}
        out.with_suffix(".dual.json").write_text(json.dumps(
            {"alpha": args.alpha, "alpha_bulk": args.alpha_bulk, **anchor,
             "dual_fallbacks": dual_fallbacks}, indent=1) + "\n")
    print(json.dumps({"h5ad": str(h5ad), **info, "write_seconds": round(time.time() - t0)}), flush=True)
    if not args.no_pack:
        t0 = time.time()
        vcc = pack_vcc(h5ad, out.with_suffix(".vcc"))
        print(json.dumps({"vcc": str(vcc), "vcc_gb": round(vcc.stat().st_size / 1e9, 2), "pack_seconds": round(time.time() - t0)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
