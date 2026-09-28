"""Per-gene cross-line transfer error of the shipped delta, in `nmae`'s own units.

`nmae` -- `de_wilcoxon_lfc_nmae` -- reads, per perturbation, ``mean_g |lfc_pred - lfc_real| /
mean_g |lfc_real|`` over the genes the REAL data calls significant (Wilcoxon, BH per
perturbation, ``p_adj < 0.05``, the target's own gene excluded). It is a per-perturbation
number. This module turns the same two tables around and reads them **per gene**: over every
perturbation whose gate contains gene g, how does our transferred log2FC for g compare in
SIZE to what the held-out line really did? Three summaries, all over the gated (target, gene)
cells of one gene:

    slope_g  = sum_t p * y / sum_t y^2      the amplitude that best rescales our prediction
                                            onto the truth (through the origin; < 1 we
                                            under-shoot, > 1 we over-shoot)
    ratio_g  = sum_t |p| / sum_t |y|        the size ratio, sign-blind
    nmae_g   = sum_t |p - y| / sum_t |y|    the gene's own share of nmae

Born under T102 (session `1430fead`, 2026-09-25), which asks whether a gene's 3'UTR
regulatory load predicts any of these -- and if so, whether a per-gene amplitude in
`submit.build` could be read off it. But the instrument is not tied to that feature: T1's
sibling levers (magnitude for `nmae`) can read any per-gene covariate against the same
tables.

**The two tables, and why they are built the way they are.**

*Truth.* cell-eval2's log2FC is ``log2((mean_t + eps) / (mean_c + eps))`` with ``eps = 1e-9``
on the ARITHMETIC MEAN of the per-cell CPM over a perturbation's cells against the real
controls' cells (``de.mean_calc: arithmetic``, ``control_source: real``), on the genes whose
control-side mean CPM is at least ``filter_gene_min_cpm_cell = 5``. `fold_truth` streams
exactly those means off the fold's real h5ad (one pass, `stream_pseudobulk`), and `lfc_ce2`
applies exactly that formula. Nothing is re-derived from a pseudobulk of the SUM.

*Prediction.* The emitter lays every emitted cell's composition on ``frac * 2^d /
sum(frac * 2^d)`` (`count_emitters.PoissonEmitter._fraction`), with ``d`` the pooled delta
times alpha and the target's own gene pinned, so the per-cell mean CPM the Wilcoxon members
read is ``1e6`` times that fraction, up to emission noise. `pred_lfc` writes it down in closed
form against the SAME real control means the truth uses. At lambda 0 this is exact; at the
shipped lambda 0.5 the per-cell mean is exact in expectation and the noise averages over
400 cells. So the prediction here is what the pipeline INTENDS, not one draw of it -- which
is the right object for a question about a per-gene amplitude.

*Gate.* The real-side Wilcoxon table is what the scorer computes on every run and never
caches, so `gate_from_de_real` reads the ``de_real.parquet`` that ``cell-eval2 run
--write-degenes`` writes -- the metric's own gate, bit for bit. Where that table does not
exist (a fold too large for the Mac), `gate_wald` stands in: a two-sample z on the same
per-cell CPM moments, BH per perturbation at the same threshold. It is an approximation and
every result built on it says so (`gate_kind`).

**What a correlation here can and cannot mean.** A per-gene error is confounded by how well
the gene is measured (control CPM) and by how big its true effects are (`mean_abs_truth`):
low-expression genes are noisier on both sides, and `nmae_g` is a ratio whose denominator
is the true size. `correlate` therefore reports the Spearman correlation raw AND with those
two covariates partialled out (rank residuals), each against a gene-shuffled null -- the
feature permuted across genes with the errors held fixed -- because a feature that is
itself a proxy for expression (3'UTR length tracks it) will correlate with everything.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from sidechain.data.stream_pseudobulk import PseudobulkSums, stream_pseudobulk

__all__ = [
    "FoldTruth", "fold_truth", "lfc_ce2", "pred_lfc", "gate_from_de_real", "gate_wald",
    "per_gene_stats", "correlate", "residualize_ranks", "gate_agreement",
    "nmae_closed_form", "fit_binned_amplitude", "amplitude_on_axis",
]

CE2_EPS = 1e-9                 # cell-eval2 `de.epsilon` under the vcc2026 preset
CE2_MIN_CPM = 5.0              # `filter_gene_min_cpm_cell`: control-side mean CPM floor
CE2_P_ADJ = 0.05               # `de.p_adj_threshold`
CE2_MIN_GATE = 10              # `de_lfc_nmae` min_gate_size: smaller gates score nothing
KNOCKDOWN_LOG2FC = -2.32       # what the emitter pins the target's own gene to


# ------------------------------------------------------------------------- truth --

@dataclass(frozen=True)
class FoldTruth:
    """Per-label arithmetic means of per-cell CPM off a fold's real h5ad."""

    targets: list[str]            # non-control labels, sorted
    genes: np.ndarray             # the fold's gene axis, file order
    mean_cpm: np.ndarray          # [P, G] mean per-cell CPM of each target's cells
    var_cpm: np.ndarray           # [P, G] population variance of the per-cell CPM
    n_cells: np.ndarray           # [P]
    ctrl_mean_cpm: np.ndarray     # [G] the real controls' mean per-cell CPM
    ctrl_var_cpm: np.ndarray      # [G]
    ctrl_n_cells: int
    control: str

    @property
    def universe(self) -> np.ndarray:
        """The DE universe: genes whose control-side mean CPM clears cell-eval2's floor."""
        return self.ctrl_mean_cpm >= CE2_MIN_CPM

    def truth_lfc(self) -> np.ndarray:
        """[P, G] log2FC in cell-eval2's units (see `lfc_ce2`)."""
        return lfc_ce2(self.mean_cpm, self.ctrl_mean_cpm[None, :])


def fold_truth(real_h5ad: str | Path, *, pert_col: str, control: str = "non-targeting",
               cache: str | Path | None = None, progress: bool = False) -> FoldTruth:
    """One streaming pass over the fold's real h5ad; `cache` keeps the sums as an npz.

    The cached object is a `PseudobulkSums` -- the same artifact the pipeline's sources
    are -- so `_log2fc_with_var` and every other reader of that shape can open it too.
    """
    real_h5ad = Path(real_h5ad).expanduser()
    if cache is not None and Path(cache).exists():
        pb = PseudobulkSums.load(cache)
    else:
        pb = stream_pseudobulk(real_h5ad, pert_col, progress=progress)
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            pb.save(cache)
    if control not in pb.labels:
        raise KeyError(f"control {control!r} is not a label of {real_h5ad.name}; "
                       f"have e.g. {pb.labels[:3]}")
    c = pb.labels.index(control)
    rows = [i for i, lab in enumerate(pb.labels) if lab != control]
    n = np.maximum(pb.n_cells, 1).astype(np.float64)
    mean = pb.cpm_sum / n[:, None]
    var = np.maximum(pb.cpm_sq_sum / n[:, None] - mean * mean, 0.0)
    return FoldTruth(
        targets=[pb.labels[i] for i in rows], genes=np.asarray(pb.genes, dtype=str),
        mean_cpm=mean[rows], var_cpm=var[rows], n_cells=pb.n_cells[rows].astype(np.int64),
        ctrl_mean_cpm=mean[c], ctrl_var_cpm=var[c], ctrl_n_cells=int(pb.n_cells[c]),
        control=control,
    )


def lfc_ce2(mean_t: np.ndarray, mean_c: np.ndarray, eps: float = CE2_EPS) -> np.ndarray:
    """cell-eval2's canonical log2FC: ``log2((mean_t + eps) / (mean_c + eps))``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2((np.asarray(mean_t, dtype=np.float64) + eps)
                       / (np.asarray(mean_c, dtype=np.float64) + eps))


# -------------------------------------------------------------------- prediction --

def pred_lfc(deltas: np.ndarray, targets: list[str], genes: np.ndarray, frac: np.ndarray,
             ctrl_mean_cpm: np.ndarray, *, alpha: float = 1.0, covered=None,
             kd_value: float = KNOCKDOWN_LOG2FC) -> np.ndarray:
    """[P, G] log2FC the Wilcoxon members would read off the emitted cells, in closed form.

    ``deltas`` is the pooled delta as `submit.build.pooled_delta` returns it (before alpha);
    ``frac`` the emitter's control profile on the same axis (`ContextProfile.fraction`);
    ``ctrl_mean_cpm`` the real controls' mean per-cell CPM, which is the reference the
    scorer divides by under ``control_source: real``. `covered` masks targets no source
    covered: those are emitted as the bare profile with NO pin (`eval.loco` pins inside
    ``if d is not None``), exactly as `analytic_pds.score_delta` honours it.
    """
    d = np.nan_to_num(np.asarray(deltas, dtype=np.float64) * alpha, nan=0.0, posinf=0.0,
                      neginf=0.0)
    pos = {g: i for i, g in enumerate(genes)}
    cov = np.ones(len(targets), dtype=bool) if covered is None else np.asarray(covered, bool)
    for i, t in enumerate(targets):
        j = pos.get(str(t))
        if j is not None and cov[i]:
            d[i, j] = kd_value
        if not cov[i]:
            d[i, :] = 0.0
    f = frac[None, :] * np.exp2(d)
    f = f / f.sum(axis=1, keepdims=True)
    return lfc_ce2(1e6 * f, ctrl_mean_cpm[None, :])


# ------------------------------------------------------------------------- gates --

def _own_gene_off(gate: np.ndarray, targets: list[str], genes: np.ndarray) -> np.ndarray:
    pos = {g: i for i, g in enumerate(genes)}
    for i, t in enumerate(targets):
        j = pos.get(str(t))
        if j is not None:
            gate[i, j] = False
    return gate


def gate_from_de_real(parquet: str | Path, targets: list[str], genes: np.ndarray,
                      *, p_adj: float = CE2_P_ADJ) -> tuple[np.ndarray, dict]:
    """[P, G] boolean: the real-side Wilcoxon gate as `de_lfc_nmae` builds it.

    Reads ``de_real.parquet`` (``target, feature, log2_fold_change, p_adj``), keeps
    ``p_adj < threshold`` with a finite real log2FC, drops the target's own gene, and
    reports how many perturbations clear ``min_gate_size`` -- the ones nmae scores.
    """
    df = pd.read_parquet(parquet, columns=["target", "feature", "log2_fold_change", "p_adj"])
    df = df[(df["p_adj"] < p_adj) & np.isfinite(df["log2_fold_change"])]
    tpos = {t: i for i, t in enumerate(targets)}
    gpos = {g: i for i, g in enumerate(genes)}
    ti = df["target"].astype(str).map(tpos)
    gi = df["feature"].astype(str).map(gpos)
    ok = ti.notna() & gi.notna()
    gate = np.zeros((len(targets), len(genes)), dtype=bool)
    gate[ti[ok].astype(int).to_numpy(), gi[ok].astype(int).to_numpy()] = True
    gate = _own_gene_off(gate, targets, genes)
    sizes = gate.sum(axis=1)
    info = {"kind": "wilcoxon_de_real", "source": str(parquet), "p_adj": p_adj,
            "rows_significant": int(ok.sum()), "rows_unmatched": int((~ok).sum()),
            "targets_in_table": int(df["target"].nunique()),
            "targets_scored": int((sizes >= CE2_MIN_GATE).sum()),
            "gate_cells": int(gate.sum()), "median_gate": float(np.median(sizes))}
    return gate, info


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg over one vector; NaN stays NaN."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty(m)
    tmp[order] = np.minimum(ranked, 1.0)
    out[ok] = tmp
    return out


def gate_wald(truth: FoldTruth, *, p_adj: float = CE2_P_ADJ) -> tuple[np.ndarray, dict]:
    """[P, G] boolean: a stand-in for the Wilcoxon gate from the same CPM moments.

    Two-sample z on the per-cell CPM means, two-sided normal p, BH per perturbation over
    the DE universe, ``p_adj < threshold``, own gene off. Used only where
    ``de_real.parquet`` could not be produced; every result says which gate it used, and
    `gate_agreement` reports how far it is from the Wilcoxon gate on a fold that has both.

    **The variance is the CONTROLS' variance under the null, on both arms:**
    ``var_c * (1/n_t + 1/n_c)``. A Welch form (each arm's own variance) was the first
    version, and it was wrong in a way that decided the numbers: a gene whose counts are
    exactly zero in every cell of a small perturbation has a target-arm variance of 0, so
    its standard error came from 20,000 control cells alone and the cell was always called
    significant -- 16 % of the gated cells on `loco_hct116`, 37 times their share of the
    universe, each with a truth of about -30 log2 (eps 1e-9), together 86 % of the summed
    |truth|. Caught by the critic pass of 2026-09-26; measured against the real gate on
    `loco_k562gwps` that form had a Jaccard of 0.16. A rank test does not see a zero arm as
    certainty, and neither does this.
    """
    u = truth.universe
    n = np.maximum(truth.n_cells, 1).astype(np.float64)[:, None]
    se2 = truth.ctrl_var_cpm[None, :] * (1.0 / n + 1.0 / max(truth.ctrl_n_cells, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (truth.mean_cpm - truth.ctrl_mean_cpm[None, :]) / np.sqrt(se2)
    p = 2.0 * sps.norm.sf(np.abs(z))
    p[:, ~u] = np.nan
    gate = np.zeros_like(p, dtype=bool)
    for i in range(p.shape[0]):
        q = _bh(p[i])
        gate[i] = np.isfinite(q) & (q < p_adj)
    gate = _own_gene_off(gate, truth.targets, truth.genes)
    sizes = gate.sum(axis=1)
    info = {"kind": "wald_z_approx", "p_adj": p_adj,
            "targets_scored": int((sizes >= CE2_MIN_GATE).sum()),
            "gate_cells": int(gate.sum()), "median_gate": float(np.median(sizes))}
    return gate, info


def gate_agreement(a: np.ndarray, b: np.ndarray) -> dict:
    """Jaccard of two [P, G] gates and each one's cell count -- the stand-in's own check."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    inter, union = int((a & b).sum()), int((a | b).sum())
    return {"cells_a": int(a.sum()), "cells_b": int(b.sum()), "intersection": inter,
            "jaccard": (inter / union) if union else float("nan"),
            "targets_scored_a": int((a.sum(axis=1) >= CE2_MIN_GATE).sum()),
            "targets_scored_b": int((b.sum(axis=1) >= CE2_MIN_GATE).sum())}


# ------------------------------------------------------------------- per gene --

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    if cum[-1] <= 0:
        return float("nan")
    return float(v[np.searchsorted(cum, 0.5 * cum[-1])])


def per_gene_stats(pred: np.ndarray, truth_lfc: np.ndarray, gate: np.ndarray,
                   genes: np.ndarray, *, min_targets: int = 8,
                   min_gate_size: int = CE2_MIN_GATE,
                   truth: FoldTruth | None = None) -> pd.DataFrame:
    """One row per gene with at least `min_targets` gated cells: slope, ratio, l1_scale, ...

    Perturbations whose gate is smaller than `min_gate_size` are dropped first, as nmae
    drops them -- they score nothing, so their cells must not count here either.

    Three size readings, because they answer different questions:
      `slope`     sum p*y / sum y^2 -- the regression of the prediction ON the truth; it is
                  dominated by sign disagreements and by the largest truths, and it is NOT
                  the rescale that puts p on y (that is sum p*y / sum p^2);
      `ratio`     sum |p| / sum |y| -- sign-blind size;
      `l1_scale`  argmin_c sum |c*p - y| -- the factor that would minimise this gene's own
                  contribution to nmae (an L1 loss), which is the |p|-weighted median of
                  y/p over the cells where p != 0. Negative when the signs mostly disagree.
    `truth_se` (when `truth` is given) is the gene's median standard error of the truth's
    log2FC over its gated cells (delta method on the CPM moments), the measurement-noise
    covariate: low-expression genes are noisier on the truth side, and a per-gene error
    read against a feature that tracks expression needs that held fixed.
    """
    scored = gate.sum(axis=1) >= min_gate_size
    g = gate & scored[:, None]
    p = np.where(g, np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    y = np.where(g, truth_lfc, 0.0)
    n = g.sum(axis=0)
    sum_y2 = (y * y).sum(axis=0)
    sum_py = (p * y).sum(axis=0)
    sum_abs_y = np.abs(y).sum(axis=0)
    sum_abs_p = np.abs(p).sum(axis=0)
    sum_abs_err = np.abs(p - y).sum(axis=0)
    agree = ((np.sign(p) == np.sign(y)) & g).sum(axis=0)
    l1 = np.full(len(genes), np.nan)
    for j in np.flatnonzero(n >= min_targets):
        cells = g[:, j] & (p[:, j] != 0)
        if cells.sum() >= 2:
            l1[j] = _weighted_median(y[cells, j] / p[cells, j], np.abs(p[cells, j]))
    se = np.full(len(genes), np.nan)
    if truth is not None:
        nt = np.maximum(truth.n_cells, 1).astype(np.float64)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            se_cell = np.sqrt(truth.var_cpm / nt / np.maximum(truth.mean_cpm, CE2_EPS) ** 2
                              + (truth.ctrl_var_cpm / max(truth.ctrl_n_cells, 1)
                                 / np.maximum(truth.ctrl_mean_cpm, CE2_EPS) ** 2)[None, :]) / np.log(2)
        for j in np.flatnonzero(n >= min_targets):
            se[j] = float(np.median(se_cell[g[:, j], j]))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame({
            "gene": genes,
            "n_targets": n,
            "slope": sum_py / sum_y2,
            "ratio": sum_abs_p / sum_abs_y,
            "l1_scale": l1,
            "nmae_g": sum_abs_err / sum_abs_y,
            "mean_abs_truth": sum_abs_y / np.maximum(n, 1),
            "mean_abs_pred": sum_abs_p / np.maximum(n, 1),
            "sign_agree": agree / np.maximum(n, 1),
            "truth_se": se,
        })
    return out[out["n_targets"] >= min_targets].reset_index(drop=True)


# ---------------------------------------------------------------- correlation --

def residualize_ranks(x: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Rank-transform `x`, regress it on the rank-transformed covariates, return residuals."""
    rx = sps.rankdata(x)
    C = np.column_stack([np.ones(len(rx))] + [sps.rankdata(c) for c in np.atleast_2d(covariates)])
    beta, *_ = np.linalg.lstsq(C, rx, rcond=None)
    return rx - C @ beta


def correlate(stat: np.ndarray, feature: np.ndarray, covariates: np.ndarray | None = None,
              *, n_perm: int = 2000, seed: int = 0) -> dict:
    """Spearman rho of `feature` with `stat`, raw and partial, each against a gene shuffle.

    `covariates` is [k, n]. The partial correlation is the Pearson correlation of the two
    rank residuals (Spearman's partial). The null shuffles the FEATURE across genes,
    `n_perm` times, with the statistic and the covariates fixed -- so the null keeps
    every property of the error vector and of the confounders, and asks only whether
    this particular assignment of feature values to genes is special. `p_perm` is
    two-sided: the share of shuffles with |rho| at least the observed.
    """
    s = np.asarray(stat, dtype=np.float64)
    f = np.asarray(feature, dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(f)
    if covariates is not None:
        cov = np.atleast_2d(np.asarray(covariates, dtype=np.float64))
        ok &= np.isfinite(cov).all(axis=0)
        cov = cov[:, ok]
    s, f = s[ok], f[ok]
    n = int(ok.sum())
    if n < 20:
        return {"n": n, "rho": float("nan"), "p_perm": float("nan"),
                "rho_partial": float("nan"), "p_perm_partial": float("nan")}
    rng = np.random.default_rng(seed)
    rho = float(sps.spearmanr(f, s).correlation)
    rs = sps.rankdata(s)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = np.corrcoef(sps.rankdata(rng.permutation(f)), rs)[0, 1]
    out = {"n": n, "rho": rho,
           "p_perm": float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1)),
           "null_sd": float(null.std())}
    if covariates is not None:
        r_s = residualize_ranks(s, cov)
        r_f = residualize_ranks(f, cov)
        rho_p = float(np.corrcoef(r_f, r_s)[0, 1])
        # Kennedy's form: permute the feature's RESIDUAL, with the statistic's residual
        # fixed -- the exchangeable quantity under the partial null, and no refit per draw.
        null_p = np.empty(n_perm)
        for k in range(n_perm):
            null_p[k] = np.corrcoef(rng.permutation(r_f), r_s)[0, 1]
        out |= {"rho_partial": rho_p,
                "p_perm_partial": float((np.sum(np.abs(null_p) >= abs(rho_p)) + 1) / (n_perm + 1)),
                "null_sd_partial": float(null_p.std()),
                "n_covariates": int(cov.shape[0])}
    return out


def write_json(path: str | Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


# ------------------------------------------------------- a per-gene amplitude --

def nmae_closed_form(pred: np.ndarray, truth_lfc: np.ndarray, gate: np.ndarray,
                     *, min_gate_size: int = CE2_MIN_GATE) -> tuple[float, np.ndarray]:
    """Raw `de_lfc_nmae` of a closed-form prediction: per scored perturbation, then the mean.

    The same arithmetic as cell-eval2's member on the same gate -- what differs from a real
    scoring run is only that `pred` is the emitter's intended per-cell mean, not one draw
    of it. Returns ``(mean, per_perturbation)``; perturbations whose gate is smaller than
    `min_gate_size` score nothing, as in the metric.
    """
    scored = np.flatnonzero(gate.sum(axis=1) >= min_gate_size)
    p = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.empty(len(scored))
    for k, i in enumerate(scored):
        g = gate[i]
        vals[k] = np.abs(p[i, g] - truth_lfc[i, g]).mean() / np.abs(truth_lfc[i, g]).mean()
    return (float(vals.mean()) if vals.size else float("nan")), vals


def fit_binned_amplitude(feature: np.ndarray, slope: np.ndarray, *, n_bins: int = 5,
                         floor: float = 0.05) -> dict:
    """A per-gene amplitude from a feature: quantile bins, one factor per bin.

    Within each bin the factor is ``1 / median(slope)`` -- the rescale that would put the
    bin's typical prediction on the truth -- divided by the geometric mean of the factors,
    so the term REDISTRIBUTES amplitude between genes and leaves the global level to alpha.
    Bins are quantiles of the feature over the genes given (ties broken by rank), so a
    feature that is mostly zeros still splits. Returns the bin edges (on the feature's
    values), the factors, and the per-bin medians it was read from. A bin whose median is at
    or below `floor` REFUSES the fit (ValueError) rather than returning a flattened factor.
    """
    f = np.asarray(feature, dtype=np.float64)
    sl = np.asarray(slope, dtype=np.float64)
    ok = np.isfinite(f) & np.isfinite(sl)
    f, sl = f[ok], sl[ok]
    if f.size < 10 * n_bins:
        raise ValueError(f"{f.size} genes is too few for {n_bins} bins")
    # Edges on DISTINCT values, and a gene sits in the bin of every gene with the same value:
    # a third of genes carry zero conserved sites, and rank-splitting ties would fit a bin on
    # a mix of zero-site and one-site genes while `amplitude_on_axis` sent every zero to bin
    # 0 (critic pass, 2026-09-26). Bin b holds the genes with more than b edges below them.
    qs = np.quantile(f, np.arange(1, n_bins) / n_bins, method="lower")
    edges = sorted({float(q) for q in qs if q < f.max()})
    which = _bin_of(f, edges)
    n_used = len(edges) + 1
    medians = np.array([float(np.median(sl[which == b])) for b in range(n_used)])
    if (medians <= floor).any():
        raise ValueError(
            f"a bin's median {medians.round(4).tolist()} is at or below the floor {floor}: "
            "a factor of 1/floor would be a guess, so this feature is not fit rather than "
            "flattened (the first run returned all-ones factors this way and read as a null)")
    factors = 1.0 / medians
    factors = factors / np.exp(np.log(factors).mean())
    return {"edges": edges, "factors": factors.tolist(), "bin_median_slope": medians.tolist(),
            "bin_n": [int((which == b).sum()) for b in range(n_used)], "n_bins": n_used}


def _bin_of(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Bin index = number of edges strictly below the value (ties share a bin)."""
    return np.searchsorted(np.asarray(edges, dtype=np.float64), values, side="left")


def amplitude_on_axis(feature_on_axis: np.ndarray, fit: dict, *, default: float = 1.0) -> np.ndarray:
    """[G] per-gene factor from a fit: `np.searchsorted` on the edges; no feature -> `default`."""
    f = np.asarray(feature_on_axis, dtype=np.float64)
    out = np.full(f.shape, default, dtype=np.float64)
    ok = np.isfinite(f)
    b = _bin_of(f[ok], fit["edges"])
    out[ok] = np.asarray(fit["factors"], dtype=np.float64)[b]
    return out



# ------------------------------------------- the universal-per-gene reading (T102, step 1) --

def pair_consistency(a: pd.DataFrame, b: pd.DataFrame, stat: str, *,
                     covariates: tuple[str, ...] = ("log_ctrl_cpm", "log_truth_se"),
                     n_perm: int = 2000, seed: int = 0) -> dict:
    """Is a gene's per-gene error the same gene's error on another fold?

    Spearman of `stat` between two folds' per-gene tables (`per_gene_stats` output joined
    to the fold's covariates) on their shared genes, raw and partial on BOTH folds'
    covariates (expression and measurement noise are correlated between lines and would
    manufacture a consistency of their own), each against a gene shuffle (`correlate`).
    Two folds of the SAME held-out line share their truth and read near the ceiling; two
    folds of different lines read the share of the error that belongs to the gene rather
    than to the receiving line -- the quantity a sequence table could hope to predict.
    """
    m = a.merge(b, on="gene", suffixes=("_a", "_b"))
    if not len(m):
        return {"n": 0, "rho": float("nan"), "p_perm": float("nan"),
                "rho_partial": float("nan"), "p_perm_partial": float("nan")}
    cov = [m[f"{c}_{s}"].to_numpy(dtype=np.float64) for c in covariates for s in ("a", "b")]
    r = correlate(m[f"{stat}_b"].to_numpy(dtype=np.float64), m[f"{stat}_a"].to_numpy(dtype=np.float64),
                  covariates=np.vstack(cov) if cov else None, n_perm=n_perm, seed=seed)
    r["shared_genes"] = int(len(m))
    return r


def decompose_across_lines(tables: dict[str, pd.DataFrame], stat: str, *,
                           covariates: tuple[str, ...] = ("log_ctrl_cpm", "log_truth_se")
                           ) -> pd.DataFrame:
    """Genes scored on EVERY fold given: the across-line mean of `stat` (the universal part),
    each fold's deviation from it, and the spread of the deviations.

    Returns one row per shared gene: ``gene, mean_<stat>, dev_<fold>_<stat> per fold,
    sd_<stat>, absdev_<stat>`` (mean |deviation|), plus the across-fold mean of each
    covariate (``mean_<cov>``) for partialling. The universal part is what a
    line-invariant table can predict; the spread is what only a line-specific one could.
    """
    if len(tables) < 2:
        raise ValueError("need at least two folds")
    names = list(tables)
    m = None
    for f in names:
        t = tables[f][["gene", stat, *covariates]].rename(
            columns={stat: f"{stat}__{f}", **{c: f"{c}__{f}" for c in covariates}})
        m = t if m is None else m.merge(t, on="gene")
    vals = np.column_stack([m[f"{stat}__{f}"].to_numpy(dtype=np.float64) for f in names])
    ok = np.isfinite(vals).all(axis=1)
    m, vals = m[ok].reset_index(drop=True), vals[ok]
    out = pd.DataFrame({"gene": m["gene"]})
    mean = vals.mean(axis=1)
    out[f"mean_{stat}"] = mean
    for j, f in enumerate(names):
        out[f"dev_{f}_{stat}"] = vals[:, j] - mean
    out[f"sd_{stat}"] = vals.std(axis=1, ddof=1)
    out[f"absdev_{stat}"] = np.abs(vals - mean[:, None]).mean(axis=1)
    for c in covariates:
        out[f"mean_{c}"] = np.column_stack(
            [m[f"{c}__{f}"].to_numpy(dtype=np.float64) for f in names]).mean(axis=1)
    return out


def knockdown_lfc(pb, label: str, control: str, *, pseudocount: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """One knockdown's log2FC and delta-method variance from a `PseudobulkSums`, the
    pipeline's own arithmetic (`submit.build._log2fc_with_var`, Poisson floor)."""
    from sidechain.submit.build import _log2fc_with_var
    return _log2fc_with_var(pb, label, control, pseudocount=pseudocount, var_floor="poisson")


def derepressed(lfc: np.ndarray, var: np.ndarray, expressed: np.ndarray, *,
                min_lfc: float = 0.25, min_z: float = 2.0) -> np.ndarray:
    """Genes UP after a knockdown, with evidence: expressed, log2FC above `min_lfc`, and
    log2FC / sqrt(var) above `min_z`. After DICER1 / DROSHA / DGCR8 this is the line's
    miRNA-repressed set as the cells reported it (the NAR 2016 note's use (ii))."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = lfc / np.sqrt(var)
    return expressed & np.isfinite(z) & (lfc > min_lfc) & (z > min_z)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = int((a | b).sum())
    return (int((a & b).sum()) / u) if u else float("nan")
