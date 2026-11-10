"""A per-(target, gene) slope of the transferred log2FC on basal expression, applied as a modifier of the pooled delta.

    fit = fit_basal_slopes(perts, sources, axis, var_floor="poisson")
    mod = fit.modifier(target_basal(ctrl_cpm, axis, fit.common), mode="gene")

Reads the sources through `submit.build`; writes nothing. Notes: private/code-notes/src/sidechain/models/basal_slope.py.md
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sidechain.models.count_emitters import remap_to_axis
from sidechain.submit.build import as_delta_source

MODES = ("off", "gene", "global", "shared")
CPM_MASS = 1e6


@dataclass
class BasalSlopeFit:
    """Everything the modifier needs, per (target, gene), on the submission axis."""

    targets: np.ndarray        # [P]
    axis: np.ndarray           # [G]
    gamma_hat: np.ndarray      # [P, G] float32 WLS slope; 0 where unfitted
    se2: np.ndarray            # [P, G] float32 its variance; inf where unfitted
    xbar: np.ndarray           # [P, G] float32 weighted-mean basal of the fitting lines; 0 where unfitted
    n_lines: np.ndarray        # [P, G] int8 lines behind the fit
    tau2: np.ndarray           # [G] per-gene prior variance of the slope
    tau2_global: float         # one prior variance for every gene
    gamma_shared: np.ndarray   # [G] one slope per gene, pooled over targets; 0 where unfitted
    se2_shared: np.ndarray     # [G] its variance; inf where unfitted
    tau2_shared: float         # prior variance of the shared slope, across genes
    common: np.ndarray         # [G] bool: the genes every line shares (the normalisation set)
    line_basal: np.ndarray     # [K, G] each line's rescaled log1p CPM on the axis; nan where absent
    line_names: list[str]

    def shrink(self, mode: str = "gene") -> np.ndarray:
        """``tau2 / (tau2 + se2)``: 1 keeps the raw slope, 0 deletes it. Per (target, gene) for
        ``gene`` and ``global``; per gene (``[G]``) for ``shared``."""
        if mode not in ("gene", "global", "shared"):
            raise ValueError(f"mode must be 'gene', 'global' or 'shared', got {mode!r}")
        if mode == "shared":
            se2 = self.se2_shared.astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where(np.isfinite(se2), self.tau2_shared / (self.tau2_shared + se2), 0.0)
            return np.nan_to_num(f, nan=0.0)
        tau2 = self.tau2[None, :] if mode == "gene" else np.float64(self.tau2_global)
        se2 = self.se2.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(np.isfinite(se2), tau2 / (tau2 + se2), 0.0)
        return np.nan_to_num(f, nan=0.0)

    def modifier(self, x_target: np.ndarray, mode: str = "gene") -> np.ndarray:
        """``[P, G]`` log2FC to ADD to the pooled delta for a line whose rescaled basal is
        ``x_target`` (see `target_basal`). Zero wherever no slope was fitted."""
        x_target = np.asarray(x_target, dtype=np.float64)
        if x_target.shape != (len(self.axis),):
            raise ValueError("x_target must be per gene on the fit's axis")
        if mode == "shared":
            gam = (self.gamma_shared * self.shrink(mode))[None, :]
        else:
            gam = self.gamma_hat.astype(np.float64) * self.shrink(mode)
        out = gam * (x_target[None, :] - self.xbar.astype(np.float64))
        out[self.n_lines < 2] = 0.0
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def stats(self, mode: str = "gene") -> dict:
        fitted = self.n_lines >= 2
        f = self.shrink(mode)
        if mode == "shared":
            f = np.broadcast_to(f[None, :], fitted.shape)
        return {
            "mode": mode,
            "targets": int(len(self.targets)), "genes": int(len(self.axis)),
            "lines": list(self.line_names), "common_genes": int(self.common.sum()),
            "fitted_pairs_frac": float(fitted.mean()),
            "genes_tau2_positive_frac": float((self.tau2 > 0).mean()),
            "tau2_global": float(self.tau2_global),
            "tau2_gene_median": float(np.median(self.tau2)),
            "tau2_shared": float(self.tau2_shared),
            "shared_slope_fitted_genes": int(np.isfinite(self.se2_shared).sum()),
            "shrink_mean_over_fitted": float(f[fitted].mean()) if fitted.any() else 0.0,
            "shrink_gt_half_frac": float((f[fitted] > 0.5).mean()) if fitted.any() else 0.0,
        }


def line_scale(cpm: np.ndarray, genes: np.ndarray, common: set) -> float:
    """The factor that puts ``CPM_MASS`` on this line's shared genes."""
    mask = np.fromiter((g in common for g in genes), dtype=bool, count=len(genes))
    mass = float(np.asarray(cpm, dtype=np.float64)[mask].sum())
    if mass <= 0:
        raise ValueError("a line puts no CPM mass on the shared genes")
    return CPM_MASS / mass


def target_basal(ctrl_tgt_cpm: np.ndarray, axis: np.ndarray, common: np.ndarray) -> np.ndarray:
    """The held-out line's rescaled ``log1p`` basal on the axis, from its control CPM."""
    cpm = np.asarray(ctrl_tgt_cpm, dtype=np.float64)
    if cpm.shape != (len(axis),):
        raise ValueError("ctrl_tgt_cpm must be per gene on the axis")
    mass = float(cpm[np.asarray(common, dtype=bool)].sum())
    if mass <= 0:
        raise ValueError("the target line puts no CPM mass on the shared genes")
    return np.log1p(cpm * (CPM_MASS / mass))


def fit_basal_slopes(targets, sources: list, axis: np.ndarray, *, var_floor: str = "poisson",
                     clamp: float = 1e-12, min_common: int = 100) -> BasalSlopeFit:
    """Fit ``gamma_hat``, ``se2`` and ``xbar`` for every target on ``axis``, then the EB prior.

    ``sources`` are exactly what `pooled_delta` takes; the weights are the pool's own
    ``1 / var`` with the same clamp and the same abstention (``var = inf`` -> weight 0), so the
    fit's intercept at ``xbar`` reproduces the pooled delta on every gene the pool's weights
    also cover. Coverage tiers and transfer floors do NOT enter the fit.
    """
    axis = np.asarray(axis, dtype=str)
    G = len(axis)
    srcs = [as_delta_source(s, var_floor=var_floor) for s in sources]
    pb = [s for s in srcs if getattr(s, "control_cpm", None) is not None]
    if len(pb) < 2:
        raise ValueError("a basal slope needs at least two sources that publish a control "
                         f"profile; got {len(pb)}")
    common = set(axis)
    for s in pb:
        common &= set(np.asarray(s.genes, dtype=str))
    if len(common) < min_common:
        raise ValueError(f"only {len(common)} genes are shared by the axis and every line "
                         f"(min_common={min_common}); the normalisation set is too small to trust")
    common_mask = np.fromiter((g in common for g in axis), dtype=bool, count=G)
    names, X = [], []
    for s in pb:
        cpm = np.asarray(s.control_cpm(), dtype=np.float64)
        sgenes = np.asarray(s.genes, dtype=str)
        x = np.log1p(cpm * line_scale(cpm, sgenes, common))
        X.append(remap_to_axis(x, sgenes, axis, fill=np.nan))
        names.append(str(getattr(getattr(s, "pb", None), "sidechain_name",
                                 getattr(s, "sidechain_name", f"line{len(names)}"))))
    X = np.stack(X)                                   # [K, G]
    K = len(pb)
    targets = np.asarray(targets, dtype=str)
    P = len(targets)
    gamma = np.zeros((P, G), dtype=np.float32)
    se2 = np.full((P, G), np.inf, dtype=np.float32)
    xbar = np.zeros((P, G), dtype=np.float32)
    nl = np.zeros((P, G), dtype=np.int8)
    Y = np.empty((K, G)); W = np.empty((K, G))
    sxx_g = np.zeros(G); sxy_g = np.zeros(G)
    for i, p in enumerate(targets):
        Y.fill(np.nan); W.fill(0.0)
        for k, s in enumerate(pb):
            got = s.effect(str(p))
            if got is None:
                continue
            fc, var = got
            with np.errstate(divide="ignore"):
                w = 1.0 / np.maximum(var, clamp)
            w = np.where(np.isfinite(var), w, 0.0)
            Y[k] = remap_to_axis(np.where(np.isfinite(fc), fc, np.nan), s.genes, axis, fill=np.nan)
            W[k] = remap_to_axis(w, s.genes, axis, fill=0.0)
        valid = np.isfinite(Y) & np.isfinite(X) & (W > 0)
        if not valid.any():
            continue
        Wv = np.where(valid, W, 0.0)
        Xv = np.where(valid, X, 0.0)
        Yv = np.where(valid, Y, 0.0)
        n = valid.sum(axis=0)
        wsum = Wv.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            xb = np.where(wsum > 0, (Wv * Xv).sum(axis=0) / wsum, 0.0)
        dx = np.where(valid, X - xb[None, :], 0.0)
        sxx = (Wv * dx * dx).sum(axis=0)
        sxy = (Wv * dx * Yv).sum(axis=0)
        ok = (n >= 2) & (sxx > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            gamma[i] = np.where(ok, sxy / sxx, 0.0)
            se2[i] = np.where(ok, 1.0 / sxx, np.inf)
        xbar[i] = xb
        nl[i] = n
        sxx_g += np.where(ok, sxx, 0.0)
        sxy_g += np.where(ok, sxy, 0.0)
    fin = np.isfinite(se2)
    excess = np.where(fin, gamma.astype(np.float64) ** 2 - se2.astype(np.float64), 0.0)
    cnt = fin.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau2 = np.where(cnt > 0, excess.sum(axis=0) / np.maximum(cnt, 1), 0.0)
    tau2 = np.maximum(np.nan_to_num(tau2, nan=0.0), 0.0)
    tau2_global = float(max(excess[fin].mean(), 0.0)) if fin.any() else 0.0
    okg = sxx_g > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_shared = np.where(okg, sxy_g / sxx_g, 0.0)
        se2_shared = np.where(okg, 1.0 / sxx_g, np.inf)
    tau2_shared = float(max((gamma_shared[okg] ** 2 - se2_shared[okg]).mean(), 0.0)) if okg.any() else 0.0
    return BasalSlopeFit(targets=targets, axis=axis, gamma_hat=gamma, se2=se2, xbar=xbar,
                         n_lines=nl, tau2=tau2, tau2_global=tau2_global, common=common_mask,
                         line_basal=X, line_names=names, gamma_shared=gamma_shared,
                         se2_shared=se2_shared, tau2_shared=tau2_shared)
