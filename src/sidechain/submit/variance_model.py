"""The variance behind the pooling weight and the shrinkage rule's standard error, as one knob (T98).

Every number `pooled_delta` calls "uncertainty" is the delta-method variance of one arm's own
cells (`submit.build._log2fc_with_var`): it sets each source's inverse-variance weight and,
since SER-14aefksw, the standard error the pooled adaptive rule divides by. This module lets a
run swap that variance for another model of the same quantity, with the controls each model
needs as flag values rather than scripts, so an arm and its control are built by the same code
path and recorded by the same string.

    --variance-model SPEC         what the per-cell CPM variance of each pseudobulk row is
    --rule-variance model|shipped whether a shrinkage rule divides by the swapped variance or
                                  stays on the shipped one (the weights follow the swap either way)
    --dispersion-fit NAME=PATH    the saved `data.dispersion.GeneDispersion` of source NAME
                                  (`python -m sidechain.data.dispersion SRC.npz --out FIT.npz`);
                                  needed by every model but `shipped` and `multiplier`

SPEC, by model (the same strings on `sidechain.eval.loco` and `sidechain.submit.build`):

  shipped                        the default; bit-identical to every historical call
  trend[:shuffle=SEED]           the glmGamPoi-style curve only: a row's per-cell variance is
                                 `(m + c) * s + theta_trend(mu_row) * m^2`, the curve read at the
                                 row's own mean count (`T` in the hand-off). The shuffle deals
                                 the curve's values to the wrong expression levels (seeded).
  own[:shuffle=SEED]             the gene's own all-group dispersion, no curve:
                                 `(m + c) * s + theta_ML_g * m^2` (`O`; the limit of infinitely
                                 fine gene categories). The shuffle deals theta_ML to the wrong genes.
  sql[:shuffle=SEED]             `GeneDispersion.variance_cpm` as built, pseudocount kept:
                                 `theta_sql_g * ((m + c) * s + theta_trend_g * m^2)` -- the form
                                 T84's analytic pass called "the gene's own dispersion" (C_gene).
                                 Equal to `own` at the gene's arm-level mean (the quasi-likelihood
                                 identity), different off it. The shuffle deals the per-gene
                                 (theta_sql, theta_trend) pair to the wrong genes.
  category:list=PATH[,shuffle=SEED]
                                 `trend` with one curve per category: genes named in PATH (one
                                 symbol a line) and the rest each get their own sliding-window
                                 median, read at the row's mean count (`C`). The shuffle deals the
                                 category labels to other genes WITHIN each expression bin, so each
                                 bin keeps its in and out counts (a label-only deal would read the
                                 in-group's higher expression as category).
  flat[:NAME=V,...] | flat:theta=V
                                 one dispersion for every gene, `(m + c) s + theta_c m^2`: the
                                 no-curve control the trend is read against. The default theta_c
                                 per source is the mean-count-squared-weighted mean of theta_trend
                                 over the source's expressed genes, so the summed per-cell
                                 overdispersion `sum theta mu^2` equals the trend's (a size-matched
                                 flat); `NAME=V` sets it per source, `theta=V` for every source.
  multiplier:NAME=K1/K2/K3[,NAME=K,...]
                                 the shipped perturbed-arm variance of source NAME scaled by K per
                                 stratum of that source's own control mean CPM -- 1-10, 10-100 and
                                 100 and above; below 1 CPM the factor is always 1 (the floor's
                                 regime). `NAME=K` is the flat form, one factor for every stratum
                                 from 1 CPM up (`M1`, the stratum form's control). The control
                                 row's term is never scaled. A NAME matching no source is refused;
                                 a source with no entry keeps the shipped variance and is counted.

Where `m` is a row's mean CPM, `c` the pseudocount (1.0 everywhere), `s = 1e6 / mean libsize`
(one count in CPM) and `mu_row = m / s` the row's mean raw count. Every dispersion model keeps
`max(model, Poisson floor)`, the `+ c`, the `/ ln2^2` and the `n < 2` abstention of the shipped
path, and it needs `var_floor="poisson"`: the model's Poisson term IS the floor the shipped
recipe already applies, so under `var_floor="none"` the two paths would no longer measure the
same thing. The multiplier scales the floored shipped variance as given: a factor below 1 at
1-10 CPM deflates it below the floor by design (the no-effect sets read z^2 under 1 there).

The fit is attached to the source once (`apply_dispersion_fits`, the way `transfer_floor`
travels) and checked against it: the fit's gene axis must equal the source's, and either its
recorded sha256 is the source file's or the source is a label subset of the fitted artifact (the
same corpus files, every label present with the same cell count; `apply_dispersion_fits`). The
per-source variance function is built once per source
and model and cached on the artifact, because `as_delta_source` rebuilds its wrapper on every
`pooled_delta` call.

Why the controls are flag values: a shuffled arm that is built by a script beside the real one
is a different code path, and the 2026-09-02 close of the per-source calibration turned on a
shuffle beating correct assignment on 3 of 3 folds -- a result only worth believing because the
two arms differed in one number. Pre-registration and reading rules:
`private/research/ideas/gamma-poisson-delta-variance.md`, `gene-category-dispersion-priors.md`,
`replicate-aware-standard-errors.md`; the run: `~/data/sidechain/runs/t98_variance_ab_<date>/`.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sidechain.data.dispersion import GeneDispersion, dispersion_trend, file_sha256, sources_sha256

VARIANCE_MODELS = ("shipped", "trend", "own", "sql", "category", "flat", "multiplier")
RULE_VARIANCES = ("model", "shipped")
FIT_MODELS = ("trend", "own", "sql", "category", "flat")
# the within-expression permutation of a per-gene shuffle: quantile bins of the arm's mean count over the
# expressed genes (the unexpressed genes are a bin of their own), each bin then rescaled by one factor so its
# summed overdispersion `sum theta mu^2` equals the real one. A shuffled arm keeps each expression level's
# dispersions and its summed variance and changes only which gene carries which -- an expression-matched,
# size-matched control; the per-bin factors are recorded. A shuffle over the whole axis hands high-expression
# genes the low end's dispersions (a factor of 10 to 300 on these arms) and inflates the variance for size
# reasons alone; `trend:shuffle` is that kind and is a diagnostic. Ten bins were too coarse (the top decile
# spans two decades of expression: 1.4 to 2.1x the summed variance on three pool sources, T98's critics), hence
# 100 and the rescaling.
SHUFFLE_BINS = 100
# strata of the SOURCE's own control mean CPM, the cheap check's edges (T109): below 1 CPM is the
# Poisson floor's regime and a multiplier never touches it
STRATUM_EDGES = (1.0, 10.0, 100.0)
STRATUM_NAMES = ("lt1", "1to10", "10to100", "ge100")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _bad(spec: str, why: str) -> ValueError:
    return ValueError(f"--variance-model {spec!r}: {why}")


@dataclass(frozen=True)
class VarianceModel:
    """One parsed `--variance-model` value. `shipped` is the identity."""

    kind: str = "shipped"
    shuffle_seed: int | None = None
    category_list: str | None = None
    # `flat`: one dispersion for every gene. None means each source's size-matched value, the
    # mean-count-squared-weighted mean of theta_trend over its expressed genes (read from the fit, recorded);
    # `flat_theta` is one value for every source, `flat_thetas` one per source stem
    flat_theta: float | None = None
    flat_thetas: tuple[tuple[str, float], ...] = ()
    # ((source stem, (k_1to10, k_10to100, k_ge100)), ...) in the order given
    multipliers: tuple[tuple[str, tuple[float, float, float]], ...] = ()

    @classmethod
    def parse(cls, spec: str | None) -> VarianceModel:
        if spec is None or spec == "" or spec == "shipped":
            return cls()
        kind, _, rest = spec.partition(":")
        if kind not in VARIANCE_MODELS:
            raise _bad(spec, f"unknown model {kind!r}; expected one of {VARIANCE_MODELS}")
        if kind == "multiplier":
            if not rest:
                raise _bad(spec, "multiplier needs at least one NAME=K or NAME=K1/K2/K3")
            seen: dict[str, tuple[float, float, float]] = {}
            for part in rest.split(","):
                name, eq, val = part.partition("=")
                if not eq or not name or not val:
                    raise _bad(spec, f"{part!r} is not NAME=K or NAME=K1/K2/K3")
                if not _NAME_RE.match(name):
                    raise _bad(spec, f"{name!r} is not a source stem")
                if name in seen:
                    raise _bad(spec, f"{name!r} given twice")
                ks = val.split("/")
                if len(ks) not in (1, 3):
                    raise _bad(spec, f"{part!r}: give one factor (flat) or three (1-10 / 10-100 / >=100 CPM)")
                try:
                    kf = tuple(float(k) for k in ks)
                except ValueError:
                    raise _bad(spec, f"{val!r} is not a number") from None
                if any(not math.isfinite(k) or k <= 0 for k in kf):
                    raise _bad(spec, f"{val!r}: every factor must be finite and > 0")
                seen[name] = kf if len(kf) == 3 else (kf[0], kf[0], kf[0])
            return cls(kind=kind, multipliers=tuple(seen.items()))
        opts: dict[str, str] = {}
        if rest:
            for part in rest.split(","):
                key, eq, val = part.partition("=")
                if not eq or not val:
                    raise _bad(spec, f"{part!r} is not key=value")
                if key in opts:
                    raise _bad(spec, f"{key!r} given twice")
                opts[key] = val
        if kind == "flat":
            # `theta=V` for every source, or `NAME=V` per source stem; never both, never `shuffle`
            per_source = {k: v for k, v in opts.items() if k != "theta"}
            if "theta" in opts and per_source:
                raise _bad(spec, "flat takes theta=V for every source OR NAME=V per source, not both")
            if "shuffle" in per_source:
                raise _bad(spec, "unknown option(s) ['shuffle'] for flat; a constant has nothing to shuffle")
            thetas = []
            for name, val in per_source.items():
                if not _NAME_RE.match(name):
                    raise _bad(spec, f"{name!r} is not a source stem")
                try:
                    v = float(val)
                except ValueError:
                    raise _bad(spec, f"{val!r} is not a number") from None
                if not math.isfinite(v) or v < 0:
                    raise _bad(spec, f"{name}={val!r}: a dispersion is finite and >= 0")
                thetas.append((name, v))
            theta = None
            if "theta" in opts:
                try:
                    theta = float(opts["theta"])
                except ValueError:
                    raise _bad(spec, f"theta={opts['theta']!r} is not a number") from None
                if not math.isfinite(theta) or theta < 0:
                    raise _bad(spec, f"theta={opts['theta']!r}: a dispersion is finite and >= 0")
            return cls(kind=kind, flat_theta=theta, flat_thetas=tuple(thetas))
        allowed = {"shuffle"} | ({"list"} if kind == "category" else set())
        extra = sorted(set(opts) - allowed)
        if extra:
            raise _bad(spec, f"unknown option(s) {extra} for {kind}; allowed: {sorted(allowed)}")
        seed = None
        if "shuffle" in opts:
            try:
                seed = int(opts["shuffle"])
            except ValueError:
                raise _bad(spec, f"shuffle={opts['shuffle']!r} is not an integer seed") from None
        if kind == "category" and "list" not in opts:
            raise _bad(spec, "category needs list=PATH (one gene symbol a line)")
        return cls(kind=kind, shuffle_seed=seed, category_list=opts.get("list"))

    def spec(self) -> str:
        """The canonical string, so a record and a command line say the same thing."""
        if self.kind == "shipped":
            return "shipped"
        if self.kind == "multiplier":
            parts = []
            for name, ks in self.multipliers:
                parts.append(f"{name}={ks[0]:g}" if ks[0] == ks[1] == ks[2]
                             else f"{name}={ks[0]:g}/{ks[1]:g}/{ks[2]:g}")
            return "multiplier:" + ",".join(parts)
        opts = []
        if self.category_list is not None:
            opts.append(f"list={self.category_list}")
        if self.shuffle_seed is not None:
            opts.append(f"shuffle={self.shuffle_seed}")
        if self.flat_theta is not None:
            opts.append(f"theta={self.flat_theta:g}")
        opts += [f"{n}={v:g}" for n, v in self.flat_thetas]
        return self.kind + (":" + ",".join(opts) if opts else "")

    def flat_theta_for(self, name: str) -> float | None:
        """The flat dispersion given for source `name`, the common one, or None (the size-matched default)."""
        for n, v in self.flat_thetas:
            if n == name:
                return v
        return self.flat_theta

    def identity(self) -> str:
        """The spec plus the category list's sha256: what a cache key and a run record carry.

        Two runs with the same spec string on an edited list are two models; the path alone would not say so.
        """
        if self.category_list is None:
            return self.spec()
        return f"{self.spec()}#list_sha256={file_sha256(self.category_list)}"

    @property
    def is_shipped(self) -> bool:
        return self.kind == "shipped"

    @property
    def needs_fit(self) -> bool:
        return self.kind in FIT_MODELS

    @property
    def floor_max(self) -> bool:
        """Whether `_log2fc_with_var` takes `max(model, Poisson floor)` -- the dispersion models."""
        return self.kind in FIT_MODELS

    def multiplier_for(self, name: str) -> tuple[float, float, float] | None:
        for n, ks in self.multipliers:
            if n == name:
                return ks
        return None

    def for_source(self, pb, control: str) -> SourceVariance | None:
        """The per-cell variance function of one pseudobulk source under this model; None = shipped.

        Cached on the artifact under the model's spec, because the wrapper that calls this is
        rebuilt on every `pooled_delta` call and a category fit is two sliding-window medians
        over the whole gene axis.
        """
        if self.is_shipped:
            return None
        cache = getattr(pb, "_variance_cache", None)
        if cache is None:
            cache = {}
            try:
                pb._variance_cache = cache
            except AttributeError:   # a stub without __dict__; build uncached
                pass
        key = self.spec()
        if key not in cache:
            cache[key] = self._build(pb, control)
        return cache[key]

    def _build(self, pb, control: str) -> SourceVariance | None:
        name = getattr(pb, "sidechain_name", None) or "<unnamed source>"
        if self.kind == "multiplier":
            ks = self.multiplier_for(name) if name else None
            if ks is None:
                return None
            c = pb.labels.index(control)
            m_c = pb.cpm_sum[c] / max(int(pb.n_cells[c]), 1)
            code = np.digitize(m_c, STRATUM_EDGES)          # 0 below 1 CPM ... 3 at 100 and above
            k = np.array([1.0, *ks])[code]
            return SourceVariance(kind="multiplier", k=k, strata=code, source=name,
                                  record={"source": name, "factors": list(ks),
                                          "genes_per_stratum": [int((code == i).sum()) for i in range(4)]})
        gd = getattr(pb, "dispersion_fit", None)
        if gd is None:
            raise ValueError(f"--variance-model {self.spec()} needs the dispersion fit of source "
                             f"{name!r}: pass --dispersion-fit {name}=PATH (made by "
                             "`python -m sidechain.data.dispersion`)")
        if len(gd.genes) != len(pb.genes) or not np.array_equal(np.asarray(gd.genes).astype(str),
                                                                  np.asarray(pb.genes).astype(str)):
            raise ValueError(f"the dispersion fit attached to {name!r} is on another gene axis "
                             f"({len(gd.genes)} genes against the source's {len(pb.genes)})")
        rng = np.random.default_rng(self.shuffle_seed) if self.shuffle_seed is not None else None
        G = len(gd.genes)
        ok = gd.mean_count > 0
        rec = {"source": name, "fit_sha256": getattr(pb, "dispersion_fit_sha256", None),
               "shuffle_seed": self.shuffle_seed, "genes_expressed": int(ok.sum())}
        if self.kind == "trend":
            lx, ty = _curve(gd.mean_count, gd.theta_trend, ok)
            if rng is not None:
                # the curve's values dealt across the whole expression axis: NOT size-preserving (a
                # within-bin permutation of a smooth curve is the curve itself), so this is a diagnostic
                # read beside the arm, never the control that decides -- the energy-matched twin and
                # `flat` are (private/research/ideas/gamma-poisson-delta-variance.md, PREREG § 2)
                ty = ty[rng.permutation(len(ty))]
                rec["shuffle"] = "the curve's values permuted over the expression axis (diagnostic, not size-preserving)"
            return SourceVariance(kind="trend", curve=(lx, ty), source=name, record=rec)
        if self.kind == "flat":
            given = self.flat_theta_for(name)
            mu2 = gd.mean_count[ok] ** 2
            matched = float((gd.theta_trend[ok] * mu2).sum() / mu2.sum())
            theta_c = given if given is not None else matched
            rec["flat_theta"] = theta_c
            rec["flat_theta_size_matched"] = matched
            rec["flat_theta_from"] = ("spec" if given is not None else
                                      "the mean-count-squared-weighted mean of theta_trend over the expressed genes "
                                      "(the trend's summed overdispersion, so a size-matched flat)")
            return SourceVariance(kind="flat", theta=np.full(G, theta_c), source=name, record=rec)
        if self.kind == "own":
            theta = gd.theta_ml.copy()
            if rng is not None:
                theta, scales = shuffle_within_bins(rng, theta, gd.mean_count, ok)
                rec["shuffle"] = (f"theta_ML permuted within {SHUFFLE_BINS} quantile bins of mean count, each bin "
                                  f"rescaled so its summed theta mu^2 equals the real one")
                rec["shuffle_bin_scale_range"] = [float(scales.min()), float(scales.max())]
            return SourceVariance(kind="own", theta=theta, source=name, record=rec)
        if self.kind == "sql":
            sql, trend = gd.theta_sql.copy(), gd.theta_trend.copy()
            if rng is not None:
                p = permutation_within_bins(rng, gd.mean_count, ok)
                sql, trend = sql[p], trend[p]
                # the pair's overdispersion term is theta_sql theta_trend mu^2: rescale the permuted trend per bin
                trend, scales = rescale_within_bins(sql * trend, gd.theta_sql * gd.theta_trend, trend, gd.mean_count, ok)
                rec["shuffle"] = (f"(theta_sql, theta_trend) pairs permuted within {SHUFFLE_BINS} quantile bins of mean "
                                  f"count, theta_trend rescaled per bin so the summed theta_sql theta_trend mu^2 equals the real one")
                rec["shuffle_bin_scale_range"] = [float(scales.min()), float(scales.max())]
            return SourceVariance(kind="sql", theta=trend, theta_sql=sql, source=name, record=rec)
        if self.kind == "category":
            listed = read_gene_list(self.category_list)
            code = np.isin(np.asarray(gd.genes).astype(str), np.asarray(sorted(listed))).astype(np.int8)
            n_in = int(code.sum())
            if n_in == 0 or n_in == G:
                raise ValueError(f"category list {self.category_list}: {n_in} of {G} genes of "
                                 f"{name!r} are in it; a split needs both groups")
            if rng is not None:
                # the labels move within expression bins: each bin keeps its in and out counts, so the
                # shuffled in-group has the real one's expression support (the design of § 6 of T98's
                # pre-registration, at SHUFFLE_BINS bins with no bin dropped; § 6's own gate leaves out
                # bins with fewer than 5 genes in either group. A label-only deal reads composition as
                # category.)
                code = code[permutation_within_bins(rng, gd.mean_count, ok)]
                rec["shuffle"] = f"category labels permuted within {SHUFFLE_BINS} quantile bins of mean count, each bin's in and out counts kept"
            curves = []
            for cat in (0, 1):
                sel = (code == cat) & ok
                tr = dispersion_trend(gd.mean_count[sel], gd.theta_ml[sel])
                curves.append(_curve(gd.mean_count[sel], tr, np.ones(int(sel.sum()), dtype=bool)))
            rec.update({"category_list": self.category_list,
                        "category_list_sha256": file_sha256(self.category_list),
                        "genes_in_list": n_in, "genes_in_list_expressed": int(((code == 1) & ok).sum())})
            return SourceVariance(kind="category", curve=curves[0], curve_in=curves[1], category=code,
                                  source=name, record=rec)
        raise ValueError(f"unknown variance model {self.kind!r}")

    def check_sources(self, sources) -> None:
        """Refuse a multiplier naming no source, and a fit model with a pseudobulk source lacking its fit.

        Called by both entry points once the sources are assembled, the way `apply_transfer_floors`
        checks its names: a factor attached to the wrong source, or a source silently pooled on the
        shipped variance under a dispersion arm, is the quiet wrongness this knob must not allow.
        """
        if self.is_shipped:
            return
        named = {}
        for s in sources:
            pb = s[0] if isinstance(s, tuple) else s
            name = getattr(pb, "sidechain_name", None)
            # a ready delta source (an LfcTable: `.effect` and `.genes`, no cells) is not a pseudobulk
            if name is not None and not (hasattr(pb, "effect") and hasattr(pb, "genes")):
                named[name] = pb
        if self.kind == "multiplier":
            missing = sorted(set(n for n, _ in self.multipliers) - set(named))
            if missing:
                raise SystemExit(f"--variance-model multiplier names {missing}, which match no "
                                 f"pseudobulk source; have {sorted(named)}")
            return
        if self.kind == "flat" and self.flat_thetas:
            # a per-source flat value on a name nobody loaded would silently fall back to the
            # size-matched default for every real source (round-3 critic of T98's pre-registration)
            missing = sorted(set(self.flat_thetas) - set(named))
            if missing:
                raise SystemExit(f"--variance-model flat names {missing}, which match no "
                                 f"pseudobulk source; have {sorted(named)}")
        if self.needs_fit:
            unfit = sorted(n for n, pb in named.items() if getattr(pb, "dispersion_fit", None) is None)
            if unfit:
                raise SystemExit(f"--variance-model {self.spec()}: pseudobulk source(s) {unfit} carry "
                                 "no dispersion fit; pass --dispersion-fit NAME=PATH for each")

    def record(self, sources) -> dict:
        """What a run record keeps: the spec and, per pseudobulk source, the fit or the factors."""
        out = {"spec": self.spec(), "identity": self.identity(), "kind": self.kind,
               "shuffle_seed": self.shuffle_seed, "shuffle_bins": SHUFFLE_BINS if self.shuffle_seed is not None else None,
               "flat_theta": self.flat_theta, "flat_thetas": dict(self.flat_thetas) if self.flat_thetas else None,
               "sources": {}}
        if self.kind == "category":
            out["category_list"] = self.category_list
            out["category_list_sha256"] = file_sha256(self.category_list)
            out["category_list_genes"] = len(read_gene_list(self.category_list))
        for s in sources:
            pb = s[0] if isinstance(s, tuple) else s
            name = getattr(pb, "sidechain_name", None)
            if name is None:
                continue
            entry: dict = {}
            fit_path = getattr(pb, "dispersion_fit_path", None)
            if fit_path is not None:
                entry["dispersion_fit"] = str(fit_path)
                entry["dispersion_fit_sha256"] = getattr(pb, "dispersion_fit_sha256", None)
                entry["dispersion_fit_attached_as"] = getattr(pb, "dispersion_fit_attached_as", None)
            if self.kind == "multiplier":
                ks = self.multiplier_for(name)
                entry["factors"] = list(ks) if ks is not None else None
            # what the built per-source variance recorded (the category split's counts, the flat theta,
            # the shuffle's form), when it has been built
            cache = getattr(pb, "_variance_cache", None) or {}
            sv = cache.get(self.spec())
            if sv is not None and sv.record:
                entry["built"] = {k: v for k, v in sv.record.items() if k != "source"}
            out["sources"][name] = entry
        return out


def _curve(mean_count: np.ndarray, theta: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(log mean count, theta)` over the expressed genes, sorted for `np.interp`."""
    o = np.argsort(mean_count[ok], kind="stable")
    return np.log(mean_count[ok][o]), np.asarray(theta)[ok][o]


def expression_bins(mean_count: np.ndarray, ok: np.ndarray, n_bins: int = SHUFFLE_BINS) -> list[np.ndarray]:
    """The expressed genes cut into `n_bins` quantile bins of mean count (equal counts, by rank), as index arrays."""
    idx = np.flatnonzero(ok)
    if not idx.size:
        return []
    order = idx[np.argsort(mean_count[idx], kind="stable")]
    return [c for c in np.array_split(order, min(n_bins, order.size)) if c.size]


def permutation_within_bins(rng: np.random.Generator, mean_count: np.ndarray, ok: np.ndarray,
                            n_bins: int = SHUFFLE_BINS) -> np.ndarray:
    """A permutation of the gene axis that moves genes only within quantile bins of their mean count.

    The expressed genes (`ok`) are cut into `n_bins` quantile bins of `mean_count` and permuted within
    each; the unexpressed genes are permuted among themselves. The result `p` is used as `values[p]`:
    gene g takes the value of gene p[g], a gene of its own expression level.
    """
    G = len(mean_count)
    p = np.arange(G)
    for chunk in expression_bins(mean_count, ok, n_bins):
        p[chunk] = chunk[rng.permutation(chunk.size)]
    rest = np.flatnonzero(~ok)
    if rest.size:
        p[rest] = rest[rng.permutation(rest.size)]
    return p


def rescale_within_bins(shuffled_term: np.ndarray, real_term: np.ndarray, values: np.ndarray,
                        mean_count: np.ndarray, ok: np.ndarray, n_bins: int = SHUFFLE_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Scale `values` by one factor per expression bin so that `sum(shuffled_term mu^2)` equals
    `sum(real_term mu^2)` in every bin -- the summed overdispersion the permutation alone does not keep
    (within a decile theta still falls with expression). Returns the rescaled values and the factors."""
    out = np.asarray(values, dtype=np.float64).copy()
    mu2 = mean_count ** 2
    scales = []
    for chunk in expression_bins(mean_count, ok, n_bins):
        have = float((shuffled_term[chunk] * mu2[chunk]).sum())
        want = float((real_term[chunk] * mu2[chunk]).sum())
        s = want / have if have > 0 else 1.0
        out[chunk] *= s
        scales.append(s)
    return out, np.asarray(scales if scales else [1.0])


def shuffle_within_bins(rng: np.random.Generator, theta: np.ndarray, mean_count: np.ndarray, ok: np.ndarray,
                        n_bins: int = SHUFFLE_BINS) -> tuple[np.ndarray, np.ndarray]:
    """`theta` dealt to other genes of the same expression bin, each bin rescaled to its real summed theta mu^2."""
    p = permutation_within_bins(rng, mean_count, ok, n_bins)
    shuffled = np.asarray(theta, dtype=np.float64)[p]
    return rescale_within_bins(shuffled, np.asarray(theta, dtype=np.float64), shuffled, mean_count, ok, n_bins)


def read_gene_list(path: str | Path) -> set[str]:
    """A gene list file: one symbol a line, `#` comments and blanks ignored."""
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"category list {p} does not exist")
    out = set()
    for line in p.read_text().splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            out.add(s)
    if not out:
        raise ValueError(f"category list {p} names no gene")
    return out


class SourceVariance:
    """The per-cell CPM variance of one source's rows under one model.

    `cell_variance(m, scale, v_shipped, pseudocount, perturbed)` returns the model's variance for
    a row whose mean CPM is `m`, with `scale` CPM per count and `v_shipped` the floored shipped
    variance of that row; `perturbed` says whether it is the knockdown's row or the control's.
    The caller (`_log2fc_with_var`) applies the floor max for the dispersion models.
    """

    __slots__ = ("category", "curve", "curve_in", "k", "kind", "record", "source", "strata",
                 "theta", "theta_sql")

    def __init__(self, kind: str, *, source: str = "", record: dict | None = None,
                 curve=None, curve_in=None, theta=None, theta_sql=None, k=None, strata=None,
                 category=None):
        self.kind, self.source, self.record = kind, source, record or {}
        self.curve, self.curve_in, self.theta, self.theta_sql = curve, curve_in, theta, theta_sql
        self.k, self.strata, self.category = k, strata, category

    @property
    def floor_max(self) -> bool:
        return self.kind in FIT_MODELS

    def _interp(self, m: np.ndarray, scale: float, curve) -> np.ndarray:
        lx, ty = curve
        with np.errstate(divide="ignore"):
            x = np.log(np.asarray(m, dtype=np.float64) / scale)      # -inf at m = 0 -> the curve's low end
        return np.interp(x, lx, ty)

    def cell_variance(self, m: np.ndarray, scale: float, v_shipped: np.ndarray, pseudocount: float,
                      *, perturbed: bool) -> np.ndarray:
        m = np.asarray(m, dtype=np.float64)
        poisson = (m + pseudocount) * scale
        if self.kind == "trend":
            return poisson + self._interp(m, scale, self.curve) * m * m
        if self.kind in ("own", "flat"):
            return poisson + self.theta * m * m
        if self.kind == "sql":
            return self.theta_sql * (poisson + self.theta * m * m)
        if self.kind == "category":
            th = np.where(self.category == 1, self._interp(m, scale, self.curve_in),
                          self._interp(m, scale, self.curve))
            return poisson + th * m * m
        if self.kind == "multiplier":
            return self.k * v_shipped if perturbed else np.asarray(v_shipped, dtype=np.float64)
        raise ValueError(f"unknown variance model {self.kind!r}")


def parse_dispersion_fits(specs: list[str] | None) -> dict[str, Path]:
    """``['h1_pseudobulk=/path/fit.npz']`` -> ``{'h1_pseudobulk': Path}``; None/[] -> ``{}``."""
    out: dict[str, Path] = {}
    for part in specs or []:
        name, _, value = part.partition("=")
        if not value or not name:
            raise SystemExit(f"--dispersion-fit: {part!r} is not NAME=PATH, e.g. "
                             "'h1_pseudobulk=~/data/sidechain/runs/t98_variance_ab_20261009/fits/h1_pseudobulk.fit.npz'")
        if name in out:
            raise SystemExit(f"--dispersion-fit: {name!r} given twice")
        out[name] = Path(value).expanduser()
    return out


def apply_dispersion_fits(sources: list, fits: dict[str, Path], *, check_sha256: bool = True) -> list:
    """Attach each ``NAME=PATH`` fit to the source whose file stem is NAME, after checking it.

    Two checks, both loud: the fit's gene axis must equal the source's, and the sha256 the fit
    recorded at fitting time must be the source file's (skipped only when the source carries no
    path, which is the in-memory test case). A name matching no source is a hard error, as for
    `--transfer-floor`: a fit on the wrong source is silently wrong.
    """
    if not fits:
        return sources
    seen = {}
    for src in sources:
        obj = src[0] if isinstance(src, tuple) else src
        name = getattr(obj, "sidechain_name", None)
        if name is not None:
            seen[name] = obj
    missing = sorted(set(fits) - set(seen))
    if missing:
        raise SystemExit(f"--dispersion-fit names {missing} match no source; have {sorted(seen)}. "
                         "A fit attached to the wrong source is silently wrong.")
    for name, path in fits.items():
        obj = seen[name]
        if not hasattr(obj, "genes") or not hasattr(obj, "cpm_sum"):
            raise SystemExit(f"--dispersion-fit {name}: that source is not a pseudobulk; a contrast "
                             "table has no cells to model")
        if not path.exists():
            raise SystemExit(f"--dispersion-fit {name}={path}: no such file")
        gd = GeneDispersion.load(path)
        if len(gd.genes) != len(obj.genes) or not np.array_equal(
                np.asarray(gd.genes).astype(str), np.asarray(obj.genes).astype(str)):
            raise SystemExit(f"--dispersion-fit {name}={path}: the fit's gene axis ({len(gd.genes)}) "
                             f"is not the source's ({len(obj.genes)})")
        src_path = getattr(obj, "sidechain_path", None)
        attached_as = "unchecked (in-memory source)"
        if src_path is not None and not gd.fitted_on_sha256:
            raise SystemExit(f"--dispersion-fit {name}={path}: the fit carries no provenance (made from an "
                             "in-memory object); refit it from the file with `python -m sidechain.data.dispersion`")
        if check_sha256 and src_path is not None:
            have = file_sha256(src_path)
            if have == gd.fitted_on_sha256:
                attached_as = "the fitted file itself (sha256 match)"
            else:
                # Not the file itself. A fold pools from a LABEL SUBSET of a full arm
                # (`scripts/subset_pseudobulk_labels.py`, bit-identical for pooling) while the fit is
                # made once on the full arm (the hand-off's rule: a subset's 300 to 800 labels and the
                # arm's 18,000 do not estimate the same dispersion). A subset is recognised by three
                # things together: the full arm's `sources` list, every one of its labels among the
                # fitted labels, and the same cell count for each -- which a construct-level file or
                # another QC over the same corpus files does not have.
                same_corpus = (gd.fitted_on_sources_sha256
                               and gd.fitted_on_sources_sha256 == sources_sha256(getattr(obj, "sources", [])))
                subset = False
                if same_corpus and gd.fitted_labels is not None and gd.fitted_n_cells is not None:
                    pos = {lab: i for i, lab in enumerate(gd.fitted_labels)}
                    rows = [pos.get(str(lab)) for lab in obj.labels]
                    subset = (all(r is not None for r in rows)
                              and np.array_equal(gd.fitted_n_cells[np.asarray(rows)],
                                                 np.asarray(obj.n_cells, dtype=np.int64)))
                if not subset:
                    raise SystemExit(f"--dispersion-fit {name}={path}: fitted on sha256 "
                                     f"{gd.fitted_on_sha256[:12]}..., but the source file is "
                                     f"{have[:12]}... and it is not a label subset of the fitted "
                                     f"artifact (same corpus files, every label present with the same "
                                     f"cell count); refit it")
                attached_as = (f"a label subset ({len(obj.labels)} of the fitted artifact's "
                               f"{len(gd.fitted_labels)} labels, cell counts equal)")
                print(f"--dispersion-fit {name}: {attached_as}", flush=True)
        obj.dispersion_fit = gd
        obj.dispersion_fit_path = path
        obj.dispersion_fit_sha256 = file_sha256(path)
        obj.dispersion_fit_attached_as = attached_as
    return sources


def add_variance_args(ap, *, twin: str) -> None:
    """The three flags, shared by `submit.build` and `eval.loco`."""
    ap.add_argument("--variance-model", default="shipped", metavar="SPEC",
                    help="T98: what the per-cell variance behind each pseudobulk source's pooling "
                         "weight is. shipped (default, bit-identical); trend[:shuffle=SEED] (the "
                         "glmGamPoi-style dispersion curve over expression, read at the row's mean); "
                         "own[:shuffle=SEED] (the gene's own all-group dispersion); sql[:shuffle=SEED] "
                         "(GeneDispersion.variance_cpm as built); category:list=PATH[,shuffle=SEED] "
                         "(one curve per gene category); flat[:NAME=V,...] (one dispersion for every gene, "
                         "the trend's mean-count-squared-weighted mean per source unless NAME=V sets "
                         "it); multiplier:NAME=K1/K2/K3[,NAME=K,...] (the "
                         "shipped perturbed-arm variance of source NAME times K per stratum of its "
                         "control mean CPM, 1-10 / 10-100 / >=100; NAME=K is flat). Dispersion models "
                         "need --dispersion-fit per pseudobulk source and --var-floor poisson "
                         f"(sidechain.submit.variance_model). Same knob in {twin}.")
    ap.add_argument("--rule-variance", choices=RULE_VARIANCES, default="model",
                    help="T98: which variance a shrinkage rule divides by when --variance-model "
                         "moved: the swapped one (model, default) or the shipped one (shipped), the "
                         "pooling weights following the swap either way. Inert with --variance-model "
                         f"shipped and refused there. Same knob in {twin}.")
    ap.add_argument("--dispersion-fit", action="append", default=[], metavar="NAME=PATH",
                    help="T98: the saved per-gene dispersion fit of pseudobulk source NAME (its file "
                         "stem), made once by `python -m sidechain.data.dispersion SRC.npz --out "
                         "FIT.npz` and checked here against the source's gene axis and sha256 "
                         "(repeatable). Needed by trend, own, sql, flat and category; refused otherwise.")


def check_variance_args(ap, args) -> VarianceModel:
    """Refuse an inert or undefined variance setting before any work; both entry points."""
    try:
        vm = VarianceModel.parse(args.variance_model)
    except ValueError as err:
        ap.error(str(err))
    if vm.is_shipped:
        if args.rule_variance != "model":
            ap.error("--rule-variance with --variance-model shipped: there is no swapped variance "
                     "for the rule to leave, so the setting would do nothing")
        if args.dispersion_fit:
            ap.error("--dispersion-fit with --variance-model shipped: the fit would never be read; "
                     "pass a dispersion model (trend, own, sql, category) or drop the fit")
        return vm
    if getattr(args, "var_floor", "none") != "poisson":
        ap.error(f"--variance-model {vm.spec()} needs --var-floor poisson: the model's Poisson term "
                 "is that floor, and under --var-floor none the shipped and the swapped variance "
                 "would not measure the same thing")
    if vm.needs_fit and not args.dispersion_fit:
        ap.error(f"--variance-model {vm.spec()} needs --dispersion-fit NAME=PATH for every "
                 "pseudobulk source")
    if not vm.needs_fit and args.dispersion_fit:
        ap.error(f"--dispersion-fit with --variance-model {vm.spec()}: the fit would never be read")
    if vm.kind == "category":
        try:
            read_gene_list(vm.category_list)
        except ValueError as err:
            ap.error(str(err))
    return vm


def spec_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
