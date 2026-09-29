#!/usr/bin/env python
"""T102, closing step 1 -- how much of the pipeline's per-gene error is tied to the knockdown AND
shared across lines, once the two folds' prediction pools share nothing?

    .venv/bin/python scripts/t102_close.py                          # all three line pairs
    .venv/bin/python scripts/t102_close.py --pair hct116_hek293t --n-null 20

Step 1's review (workflow `wf_bfdb1e89-4e1`, 2026-09-28) found that a gene shuffle cannot tell a
gene property from a pipeline property: every fold's recorded pool holds the other fold's held-out
line, and H1 sits in every pool, so two folds' per-gene errors agree partly because their
PREDICTIONS agree. Its one clean test -- HCT116 x HEK293T, each fold predicting from ONE
essential-gene panel that holds neither held-out line, against the same pools with the knockdown
labels deranged -- put the knockdown-tied shared part at about 0.06 on `nmae_g`, on the stand-in
gate. This script is that test on the metric's own gate, for all three line pairs, with the
checks the judge listed beside it:

(a) **disjoint pools.** Per line pair, the targets both folds hold and every candidate source
    covers. Each fold predicts from one single-line source that holds NEITHER held-out line
    (HepG2, Jurkat, RPE1 everywhere; plus the K562 essential panel for HCT116 x HEK293T, the
    HEK293T panel for HCT116 x K562, the HCT116 panel for HEK293T x K562), and every ordered pair
    of different sources is read: Spearman of each per-gene statistic between the two folds,
    partial on a pre-registered covariate set, against K deranged-label draws of the SAME two
    pools (the target -> delta assignment permuted with no fixed point, in each fold, the own-gene
    pin kept on the right gene). **excess = rho(real) - mean rho(deranged)** is the part of the
    agreement that needs the prediction to be for the right knockdown; the deranged agreement is
    which genes are hard for ANY prediction from that pool. Same-source pairs are read beside
    them as the coupled reference.
(b) **recorded pools.** The same, with each fold's recorded pool (the shipped recipe's sources):
    the pool-coupled null the judge asked for the size statistics.
(c) **two ceilings.** (c1) truth side: each held-out line's cells split in two by a seeded
    draw within every label (control included); the same prediction's per-gene statistics against
    each half's truth -- the prediction referenced to that half's own control, so the control
    cancels as it does on the full data -- on the same full-data gate, correlated with the same
    partial, Spearman-Brown to full size. It ignores pool-side noise, so it is a loose upper bound.
    (c2) pool side: the SAME line predicted from two different single-line pools, real against
    deranged exactly as in (a) -- the knockdown-tied agreement a gene carries when only the pool
    changes. `cross_over_same_line` = the cross-line excess over the geometric mean of the two
    lines' same-line excesses: the share of the knockdown-tied per-gene error that survives a
    change of line.

Every real rho carries a gene bootstrap, and the median excess over the disjoint pairings a
bootstrap that resamples the same genes for every pairing (they share tables, so min / max over
pairings is not a range).

Statistics: `nmae_g`, `sign_agree`, `l1_scale`, `slope`, `log_ratio` (per_gene_transfer).
Covariates, pre-registered this time: control expression, the truth's measurement noise, the
truth's own size and how many targets score the gene (`COV_PRE`); the prediction's size is a
named sensitivity (`COV_SENS`), since it may mediate rather than confound; and `COV_DIV` adds the
gene's across-line truth divergence (the S8 discriminator: a gene whose true response differs
between the two lines is missed by any shared predictor). `div` is a sensitivity, not the clean
reading: the divergence is built from the same two truths the statistics are, so it is a common
effect of both. A statistic is never partialled on its own parts (`COMPONENTS`).

Truth is streamed off each fold's real h5ad in two halves (the full truth is their sum, and is
checked cell by cell against the log2FC cell-eval2 wrote into `de_real.parquet`); the gate is
`de_real.parquet`'s own (`gate_from_de_real`). Everything lands under
``runs/probes/t102_utr_load/close/``: ``close_<pair>.json``, per-gene parquets, ``run.log``.
Nothing here is a knob.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats as sps

from sidechain.data.stream_pseudobulk import PseudobulkSums, _obs_labels, stream_pseudobulk_file
from sidechain.eval.analytic_pds import LEGACY_MIN_LIBSIZE, delta_from_parts, pool_parts, prep_fold
from sidechain.eval.per_gene_transfer import (
    FoldTruth,
    gate_from_de_real,
    pair_consistency,
    per_gene_stats,
    pred_lfc,
    residualize_ranks,
    write_json,
)
from sidechain.submit.build import sources_from_specs

DATA = Path("~/data/sidechain").expanduser()
CACHE = DATA / "cache/vcc2026"
PROBE = DATA / "runs/probes/t102_utr_load"
MIRRORS = DATA / "runs/mirror"
OUT = PROBE / "close"

ALPHA = 1.35
MIN_TARGETS = 8
SPLIT_SEED = 20260928
SEP = "\t"
FOLDS = {  # fold dir -> (real h5ad stem, pert_col, control, held-out line)
    "loco_hct116": ("loco_hct116", "perturbation", "non-targeting", "HCT116"),
    "loco_hek293t": ("loco_hek293t", "perturbation", "non-targeting", "HEK293T"),
    "loco_k562gwps_union": ("loco_k562gwps_union", "gene", "non-targeting", "K562"),
}
SOURCES = {  # single-line sources: path, control label, the line it holds
    "HepG2": (CACHE / "hepg2_all_pseudobulk.npz", "control", "HepG2"),
    "Jurkat": (CACHE / "jurkat_all_pseudobulk.npz", "control", "Jurkat"),
    "RPE1": (CACHE / "rpe1_all_pseudobulk.npz", "control", "RPE1"),
    "K562ess": (CACHE / "k562_essential_all_pseudobulk.npz", "control", "K562"),
    "HEK": (CACHE / "foldsub/hek293t_full_union849.npz", "Non-Targeting", "HEK293T"),
    "HCT": (CACHE / "foldsub/hct116_full_union849.npz", "Non-Targeting", "HCT116"),
}
CORE = ["HepG2", "Jurkat", "RPE1"]          # the three sources every pair can use
PAIRS = {
    "hct116_hek293t": ("loco_hct116", "loco_hek293t", CORE + ["K562ess"]),
    "hct116_k562union": ("loco_hct116", "loco_k562gwps_union", CORE + ["HEK"]),
    "hek293t_k562union": ("loco_hek293t", "loco_k562gwps_union", CORE + ["HCT"]),
}
STATS = ["nmae_g", "sign_agree", "l1_scale", "slope", "log_ratio"]
COV_PRE = ("log_ctrl_cpm", "log_truth_se", "log_mean_abs_truth", "log_n_targets")
COV_SENS = COV_PRE + ("log_mean_abs_pred",)
COV_DIV = COV_PRE + ("truth_div",)
COVSETS = {"pre": COV_PRE, "sens": COV_SENS, "div": COV_DIV}
#: A statistic is never partialled on its own algebraic parts (pre-launch review, 2026-09-28):
#: log_ratio IS log_mean_abs_pred - log_mean_abs_truth, so holding both reads it against itself.
COMPONENTS = {"log_ratio": ("log_mean_abs_pred", "log_mean_abs_truth")}
TRUTH_CLIP = 10.0      # |log2FC| cap inside truth_div only: an all-zero arm reads -30 under eps 1e-9
N_BOOT = 200           # gene bootstrap behind each real rho and behind the median excess
MIN_CELLS_PER_ARM = 20
N_PAIRED_NULL = 10     # deranged draws re-read on every paired bootstrap resample


def cov_for(stat: str, cov: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(c for c in cov if c not in COMPONENTS.get(stat, ()))


def _log(msg: str, fh) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def _labels_genes(path: Path) -> tuple[list[str], list[str]]:
    return PseudobulkSums.peek(path)


# ------------------------------------------------------------------ truth, in halves --

def _split_assignment(labels: np.ndarray, seed: int) -> np.ndarray:
    """'A' / 'B' per cell: within every label a seeded random half (floor n/2) goes to A."""
    _, inv = np.unique(labels, return_inverse=True)
    rng = np.random.default_rng(seed)
    u = rng.random(len(labels))
    order = np.lexsort((u, inv))
    counts = np.bincount(inv)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.empty(len(labels), dtype=np.int64)
    rank[order] = np.arange(len(labels)) - np.repeat(starts, counts)
    return np.where(rank < counts[inv] // 2, "A", "B")


def _half(pb: PseudobulkSums, tag: str) -> PseudobulkSums:
    rows = [i for i, lab in enumerate(pb.labels) if lab.endswith(SEP + tag)]
    return PseudobulkSums(labels=[pb.labels[i].rsplit(SEP, 1)[0] for i in rows], genes=pb.genes,
                          count_sum=pb.count_sum[rows], cpm_sum=pb.cpm_sum[rows],
                          cpm_sq_sum=pb.cpm_sq_sum[rows], n_cells=pb.n_cells[rows],
                          libsize_sum=pb.libsize_sum[rows], sources=pb.sources)


def split_truth(fold: str, log) -> dict[str, PseudobulkSums]:
    """One streaming pass over the fold's real h5ad into two halves; cached as two npz."""
    stem, pert_col, _, _ = FOLDS[fold]
    cache = {h: OUT / "split" / f"{fold}_{h}.npz" for h in "AB"}
    if all(p.exists() for p in cache.values()):
        return {h: PseudobulkSums.load(p) for h, p in cache.items()}
    real = CACHE / f"{stem}_real.h5ad"
    t0 = time.time()
    with h5py.File(real, "r") as f:
        labels = _obs_labels(f, pert_col)
        half = _split_assignment(labels, SPLIT_SEED)
        tagged = np.char.add(np.char.add(labels.astype(str), SEP), half)
        pb = stream_pseudobulk_file(f, pert_col, labels_all=tagged, source=str(real))
    out = {h: _half(pb, h) for h in "AB"}
    for h, p in cache.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        out[h].save(p)
    log(f"   {fold}: streamed {len(labels):,} cells into halves in {time.time() - t0:.0f}s "
        f"(A {int(out['A'].n_cells.sum()):,}, B {int(out['B'].n_cells.sum()):,})")
    return out


def to_truth(pb: PseudobulkSums, control: str, keep: list[str] | None = None) -> FoldTruth:
    c = pb.labels.index(control)
    rows = [i for i, lab in enumerate(pb.labels) if lab != control and (keep is None or lab in keep)]
    n = np.maximum(pb.n_cells, 1).astype(np.float64)
    mean = pb.cpm_sum / n[:, None]
    var = np.maximum(pb.cpm_sq_sum / n[:, None] - mean * mean, 0.0)
    return FoldTruth(targets=[pb.labels[i] for i in rows], genes=np.asarray(pb.genes, dtype=str),
                     mean_cpm=mean[rows], var_cpm=var[rows], n_cells=pb.n_cells[rows].astype(np.int64),
                     ctrl_mean_cpm=mean[c], ctrl_var_cpm=var[c], ctrl_n_cells=int(pb.n_cells[c]),
                     control=control)


def _only(pb: PseudobulkSums, labels: list[str]) -> PseudobulkSums:
    rows = [pb.labels.index(lab) for lab in labels]
    return PseudobulkSums(labels=list(labels), genes=pb.genes, count_sum=pb.count_sum[rows],
                          cpm_sum=pb.cpm_sum[rows], cpm_sq_sum=pb.cpm_sq_sum[rows], n_cells=pb.n_cells[rows],
                          libsize_sum=pb.libsize_sum[rows], sources=pb.sources)


def summed(a: PseudobulkSums, b: PseudobulkSums) -> PseudobulkSums:
    """The full sums, a + b. A label one half lacks (a two-cell arm that lost its only cell with
    a library to one side) keeps the other half's sums -- the full truth has every cell."""
    if not np.array_equal(a.genes, b.genes):
        raise ValueError("the two halves disagree on genes")
    labels = sorted(set(a.labels) | set(b.labels))
    G = len(a.genes)
    out = PseudobulkSums(labels=labels, genes=a.genes, count_sum=np.zeros((len(labels), G)),
                         cpm_sum=np.zeros((len(labels), G)), cpm_sq_sum=np.zeros((len(labels), G)),
                         n_cells=np.zeros(len(labels), dtype=np.int64), libsize_sum=np.zeros(len(labels)),
                         sources=a.sources)
    for src in (a, b):
        pos = np.array([labels.index(lab) for lab in src.labels])
        out.count_sum[pos] += src.count_sum
        out.cpm_sum[pos] += src.cpm_sum
        out.cpm_sq_sum[pos] += src.cpm_sq_sum
        out.n_cells[pos] += src.n_cells
        out.libsize_sum[pos] += src.libsize_sum
    return out


# ------------------------------------------------------------------------- one fold --

class Fold:
    """Truth (full and halves), the exact gate and the emitter profile of one held-out line."""

    def __init__(self, fold: str, log, *, keep_halves: bool = True):
        stem, pert_col, control, line = FOLDS[fold]
        self.name, self.line, self.control = fold, line, control
        halves = split_truth(fold, log)
        full = summed(halves["A"], halves["B"])
        self.full = to_truth(full, control)
        # each half on the FULL label list, so row i is target i everywhere; a label a half lacks
        # reads NaN there (and drops out of that half's per-gene statistics)
        self.halves, self.half_missing = {}, {}
        if keep_halves:
            for h, pb in halves.items():
                missing = [lab for lab in full.labels if lab not in set(pb.labels)]
                self.half_missing[h] = missing
                t = to_truth(summed(pb, _only(full, missing)) if missing else pb, control)
                if missing:          # summed() filled them from the full sums; blank them instead
                    rows = [t.targets.index(lab) for lab in missing if lab != control]
                    t.mean_cpm[rows] = np.nan
                    t.var_cpm[rows] = np.nan
                self.halves[h] = t
        self.axis = self.full.genes
        fc = prep_fold(CACHE / f"{stem}_real.h5ad", pert_col=pert_col, min_libsize=LEGACY_MIN_LIBSIZE,
                       cache=MIRRORS / fold / "analytic_fold_cache.npz")
        if list(fc.genes) != list(self.axis):
            raise SystemExit(f"{fold}: fold cache genes differ from the truth's axis")
        self.frac = fc.frac
        de_real = PROBE / "de_real" / fold / "run" / "de_real.parquet"
        gate, info = gate_from_de_real(de_real, self.full.targets, self.axis)
        before = int(gate.sum())
        gate &= self.full.universe[None, :]
        info["gate_cells_after_universe"] = int(gate.sum())
        info["universe_dropped"] = before - int(gate.sum())
        info["targets_scored_after_universe"] = int((gate.sum(axis=1) >= 10).sum())
        self.gate, self.gate_info = gate, info
        self.y = self.full.truth_lfc()
        self.y_half = {h: t.truth_lfc() for h, t in self.halves.items()}
        self.tpos = {t: i for i, t in enumerate(self.full.targets)}
        self.gpos = {g: i for i, g in enumerate(self.axis)}
        self.gate_info["truth_check"] = self._check_against_ce2(de_real)
        self.gate_info["labels_missing_from_a_half"] = {h: m for h, m in self.half_missing.items() if m}
        log(f"   {fold}: {len(self.full.targets)} targets, gate {info['gate_cells']:,} cells, "
            f"{info['targets_scored']} scored; truth vs cell-eval2 lfc on gated cells: "
            f"median |diff| {self.gate_info['truth_check']['median_abs_diff']:.2e}, "
            f"max {self.gate_info['truth_check']['max_abs_diff']:.2e}; labels a half lacks: "
            f"{ {h: len(m) for h, m in self.half_missing.items()} }")
        if self.gate_info["truth_check"]["median_abs_diff"] > 1e-6:
            raise SystemExit(f"{fold}: our truth disagrees with cell-eval2's own log2FC "
                             f"({self.gate_info['truth_check']}); refusing to read anything off it")

    def _check_against_ce2(self, de_real: Path) -> dict:
        """Our full truth against the log2FC cell-eval2 itself wrote, on the gated cells."""
        df = pd.read_parquet(de_real, columns=["target", "feature", "log2_fold_change", "p_adj"])
        df = df[(df["p_adj"] < 0.05) & np.isfinite(df["log2_fold_change"])]
        ti = df["target"].astype(str).map(self.tpos)
        gi = df["feature"].astype(str).map(self.gpos)
        ok = ti.notna() & gi.notna()
        ti, gi = ti[ok].astype(int).to_numpy(), gi[ok].astype(int).to_numpy()
        ours = self.y[ti, gi]
        diff = np.abs(ours - df["log2_fold_change"].to_numpy()[ok.to_numpy()])
        fin = np.isfinite(diff)
        return {"cells": int(fin.sum()), "median_abs_diff": float(np.median(diff[fin])),
                "q99_abs_diff": float(np.quantile(diff[fin], 0.99)), "max_abs_diff": float(diff[fin].max()),
                "share_over_1e-3": float((diff[fin] > 1e-3).mean())}

    def rows(self, targets: list[str]) -> np.ndarray:
        return np.array([self.tpos[t] for t in targets])


# ------------------------------------------------------------------------- pools --

def _load_source(name: str, targets: list[str]) -> tuple[PseudobulkSums, str]:
    path, ctrl, _ = SOURCES[name]
    labels, genes = _labels_genes(path)
    s = PseudobulkSums.load_subset(path, [ctrl, *[t for t in targets if t in set(labels)]], genes)
    s.sidechain_name = path.stem
    return s, ctrl


def _recorded_specs(fold: str) -> list[str]:
    spec = importlib.util.spec_from_file_location("t102_utr_load", Path(__file__).with_name("t102_utr_load.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    specs, _ = mod.recorded_sources(fold)
    return specs


_POOL_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}   # the same targets recur across pairs


def pool_delta(targets: list[str], sources, axis) -> tuple[np.ndarray, np.ndarray]:
    num, den = pool_parts(targets, sources, axis, var_floor="poisson", verify=3)
    return delta_from_parts(num, den), (den > 0).any(axis=1)


def derangement(n: int, rng) -> np.ndarray:
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


# ------------------------------------------------------------------- per-gene tables --

def gene_table(pred: np.ndarray, y: np.ndarray, gate: np.ndarray, fold: Fold, truth: FoldTruth,
               rows: np.ndarray, genes_ok: np.ndarray) -> pd.DataFrame:
    """per_gene_stats on the pair's targets (`rows`) and the pair's gene set, plus covariates.
    A cell whose truth is not finite (a label one half lacks) leaves the gate, and so does every arm
    under `MIN_CELLS_PER_ARM` cells on the FULL data (a zero-mean arm of 11 cells reads -36 log2 under
    eps 1e-9 and would be most of a gene's nmae_g; the same rule as `t102_universal.py`)."""
    big = fold.full.n_cells[rows] >= MIN_CELLS_PER_ARM
    g = gate[rows] & genes_ok[None, :] & np.isfinite(y[rows]) & big[:, None]
    sub = FoldTruth(targets=[truth.targets[i] for i in rows], genes=truth.genes,
                    mean_cpm=truth.mean_cpm[rows], var_cpm=truth.var_cpm[rows],
                    n_cells=truth.n_cells[rows], ctrl_mean_cpm=truth.ctrl_mean_cpm,
                    ctrl_var_cpm=truth.ctrl_var_cpm, ctrl_n_cells=truth.ctrl_n_cells, control=truth.control)
    st = per_gene_stats(pred, y[rows], g, fold.axis, min_targets=MIN_TARGETS, truth=sub)
    idx = np.array([fold.gpos[x] for x in st["gene"]], dtype=np.int64)
    st["log_ctrl_cpm"] = np.log10(np.clip(truth.ctrl_mean_cpm[idx], 1e-3, None))
    st["log_truth_se"] = np.log10(st["truth_se"].astype(float).clip(lower=1e-6))
    st["log_mean_abs_truth"] = np.log10(st["mean_abs_truth"].astype(float).clip(lower=1e-6))
    st["log_mean_abs_pred"] = np.log10(st["mean_abs_pred"].astype(float).clip(lower=1e-6))
    st["log_n_targets"] = np.log10(st["n_targets"].astype(float).clip(lower=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        st["log_ratio"] = np.log10(st["ratio"].astype(float))
    st.loc[~np.isfinite(st["log_ratio"]), "log_ratio"] = np.nan
    return st


def truth_divergence(f1: Fold, f2: Fold, targets: list[str], genes: list[str]) -> pd.Series:
    """Per gene: sum |y1 - y2| / sum (|y1| + |y2|) over the targets, on cells either fold gates."""
    r1, r2 = f1.rows(targets), f2.rows(targets)
    g1 = np.array([f1.gpos[g] for g in genes]); g2 = np.array([f2.gpos[g] for g in genes])
    y1 = np.clip(f1.y[np.ix_(r1, g1)], -TRUTH_CLIP, TRUTH_CLIP)
    y2 = np.clip(f2.y[np.ix_(r2, g2)], -TRUTH_CLIP, TRUTH_CLIP)
    cells = (f1.gate[np.ix_(r1, g1)] | f2.gate[np.ix_(r2, g2)]) & np.isfinite(y1) & np.isfinite(y2)
    num = np.where(cells, np.abs(y1 - y2), 0.0).sum(axis=0)
    den = np.where(cells, np.abs(y1) + np.abs(y2), 0.0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series(np.where(den > 0, num / den, np.nan), index=genes)


def _merged(a: pd.DataFrame, b: pd.DataFrame, stat: str, cov: tuple[str, ...]) -> pd.DataFrame:
    """The two folds' tables on shared genes, finite on the statistic and every covariate."""
    cols = [stat, *cov]
    m = a[["gene", *cols]].merge(b[["gene", *cols]], on="gene", suffixes=("_a", "_b"))
    keep = np.isfinite(m.drop(columns="gene").to_numpy(dtype=np.float64)).all(axis=1)
    return m[keep].set_index("gene")


def _prho(m: pd.DataFrame, stat: str, cov: tuple[str, ...]) -> float:
    """Spearman's partial of the statistic between the two folds, on both folds' covariates --
    `pair_consistency`'s rho_partial, without its permutations (for bootstrap and null draws)."""
    if len(m) < 20:
        return float("nan")
    C = np.vstack([m[f"{c}_{s}"].to_numpy(dtype=np.float64) for c in cov for s in ("a", "b")]) if cov else None
    x, y = m[f"{stat}_a"].to_numpy(dtype=np.float64), m[f"{stat}_b"].to_numpy(dtype=np.float64)
    if C is None:
        return float(sps.spearmanr(x, y).correlation)
    return float(np.corrcoef(residualize_ranks(x, C), residualize_ranks(y, C))[0, 1])


def rho(a: pd.DataFrame, b: pd.DataFrame, stat: str, cov: tuple[str, ...], n_perm: int) -> dict:
    r = pair_consistency(a, b, stat, covariates=cov, n_perm=n_perm, seed=0)
    return {"n": r.get("shared_genes", 0), "n_used": r.get("n", 0), "rho": r["rho"],
            "rho_partial": r.get("rho_partial", float("nan")),
            "p_perm_partial": r.get("p_perm_partial", float("nan")),
            "null_sd_partial": r.get("null_sd_partial", float("nan"))}


def _deranged_delta(d: np.ndarray, perm: np.ndarray, targets: list[str], gpos: dict[str, int]) -> np.ndarray:
    """Row i carries donor perm[i]'s delta WITHOUT the donor's own knockdown: the real prediction
    never has a strong knockdown at a gene other than its own target, so neither does the null
    (pre-launch review, 2026-09-28). `pred_lfc` pins row i's own gene afterwards."""
    dd = d[perm].copy()
    for i, j in enumerate(perm):
        k = gpos.get(str(targets[j]))
        if k is not None:
            dd[i, k] = 0.0
    return dd


# ------------------------------------------------------------------------- one pair --

def run_pair(key: str, folds: dict[str, Fold], n_null: int, n_perm: int, n_boot: int, log) -> dict:
    fa, fb, pools = PAIRS[key]
    A, B = folds[fa], folds[fb]
    # targets both folds hold and every single-line source of this pair covers
    targets = set(A.full.targets) & set(B.full.targets)
    for s in pools:
        targets &= set(_labels_genes(SOURCES[s][0])[0])
    targets = sorted(targets)
    # genes: on both folds' axes and every source's axis (a gene no source measures is 0 in some pools)
    genes = set(A.axis) & set(B.axis)
    for s in pools:
        genes &= set(_labels_genes(SOURCES[s][0])[1])
    genes = sorted(genes)
    log(f"== {key}: {fa} x {fb}; pools {pools}; {len(targets)} targets, {len(genes)} genes")
    out = {"pair": key, "folds": [fa, fb], "lines": [A.line, B.line], "pools": pools,
           "targets": len(targets), "genes": len(genes), "n_null": n_null, "n_perm": n_perm, "n_boot": n_boot,
           "covariate_sets": {k: list(v) for k, v in COVSETS.items()},
           "covariate_exclusions": {k: list(v) for k, v in COMPONENTS.items()}, "gate": {}, "coverage": {}}
    div = truth_divergence(A, B, targets, genes)
    tabs: dict[tuple, pd.DataFrame] = {}        # (fold, pool, draw) -> table; draw -1 = real
    halves: dict[tuple, pd.DataFrame] = {}      # (fold, pool, draw, half) -> table
    for F in (A, B):
        rows = F.rows(targets)
        genes_ok = np.isin(F.axis, genes)
        g = F.gate[rows] & genes_ok[None, :]
        out["gate"][F.name] = {"cells": int(g.sum()), "targets_scored": int((g.sum(axis=1) >= 10).sum()),
                               "genes_with_8_targets": int((g[(g.sum(axis=1) >= 10)].sum(axis=0) >= MIN_TARGETS).sum())}
        log(f"   {F.name}: the pair's gate holds {out['gate'][F.name]['cells']:,} cells, "
            f"{out['gate'][F.name]['targets_scored']} targets scored, "
            f"{out['gate'][F.name]['genes_with_8_targets']} genes with >= {MIN_TARGETS} targets")
        variants = {p: [p] for p in pools}
        variants["recorded"] = None
        for v, names in variants.items():
            t0 = time.time()
            ck = (F.name, v, tuple(targets))
            if ck in _POOL_CACHE:
                d, cov = _POOL_CACHE[ck]
            else:
                srcs = sources_from_specs(_recorded_specs(F.name), []) if names is None else \
                    [_load_source(n, targets) for n in names]
                d, cov = pool_delta(targets, srcs, F.axis)
                del srcs
                _POOL_CACHE[ck] = (d, cov)
            out["coverage"][f"{F.name}|{v}"] = int(cov.sum())
            seed = int(hashlib.sha256(f"{key}|{F.name}|{v}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            draws = [None] + [derangement(len(targets), rng) for _ in range(n_null)]
            for k, perm in enumerate(draws, start=-1):
                dd, cc = (d, cov) if perm is None else (_deranged_delta(d, perm, targets, F.gpos), cov[perm])
                P = pred_lfc(dd, targets, F.axis, F.frac, F.full.ctrl_mean_cpm, alpha=ALPHA, covered=cc)
                t = gene_table(P, F.y, F.gate, F, F.full, rows, genes_ok)
                t["truth_div"] = t["gene"].map(div).astype(float)
                tabs[(F.name, v, k)] = t
                if k <= 0 and F.halves:   # the truth-side ceiling, for the real prediction and one null draw
                    for h in "AB":
                        # the prediction against the HALF's own control, so the reference cancels
                        # in the error exactly as it does on the full data
                        Ph = pred_lfc(dd, targets, F.axis, F.frac, F.halves[h].ctrl_mean_cpm, alpha=ALPHA, covered=cc)
                        th = gene_table(Ph, F.y_half[h], F.gate, F, F.halves[h], rows, genes_ok)
                        th["truth_div"] = th["gene"].map(div).astype(float)
                        halves[(F.name, v, k, h)] = th
            log(f"   {F.name} <- {v}: covered {int(cov.sum())}/{len(targets)}, genes scored "
                f"{len(tabs[(F.name, v, -1)])}, median nmae_g {tabs[(F.name, v, -1)]['nmae_g'].median():.3f} "
                f"(deranged {tabs[(F.name, v, 0)]['nmae_g'].median():.3f}), {time.time() - t0:.0f}s")
    # the per-gene tables of the real predictions, and of every deranged draw under draws/, so the
    # null half of every excess can be re-derived from files (results review, 2026-09-29)
    (OUT / "draws").mkdir(exist_ok=True)
    for (fname, v, k), t in tabs.items():
        if k == -1:
            t.to_parquet(OUT / f"pg_{key}_{fname}_{v}.parquet", index=False)
        else:
            t.to_parquet(OUT / "draws" / f"pg_{key}_{fname}_{v}_d{k:02d}.parquet", index=False)

    # (c1) the truth-side ceiling: the same prediction against two halves of the line's cells
    ceiling = {}
    for F in (A, B):
        if not F.halves:
            continue
        for v in [*pools, "recorded"]:
            for k in (-1, 0):
                for stat in STATS:
                    for cs, cov in COVSETS.items():
                        r = rho(halves[(F.name, v, k, "A")], halves[(F.name, v, k, "B")], stat, cov_for(stat, cov), n_perm=1)
                        rh = r["rho_partial"]
                        sb = 2 * rh / (1 + rh) if np.isfinite(rh) and rh > -1 else float("nan")
                        ceiling[f"{F.name}|{v}|{'real' if k == -1 else 'null'}|{stat}|{cs}"] = {
                            "rho_half": rh, "spearman_brown": sb, "n": r["n"]}
    out["ceiling"] = ceiling
    out["ceiling_note"] = ("truth-side only: the same prediction and gate against two halves of the line's "
                           "cells, Spearman-Brown to full size; it ignores pool-side noise, so it bounds "
                           "the cross-line rho from above loosely -- read same_line beside it")

    def ceil(fname, v, kind, stat, cs):
        c = ceiling.get(f"{fname}|{v}|{kind}|{stat}|{cs}", {}).get("spearman_brown", float("nan"))
        return c if np.isfinite(c) and c > 0 else float("nan")

    def read_rows(Fa: Fold, Fb: Fold, pairings, scope: str, n_perm_real: int):
        brng = np.random.default_rng(7)
        found = []
        for a, b, kind in pairings:
            for stat in STATS:
                for cs, cov0 in COVSETS.items():
                    cov = cov_for(stat, cov0)
                    real = rho(tabs[(Fa.name, a, -1)], tabs[(Fb.name, b, -1)], stat, cov, n_perm=n_perm_real)
                    null = np.array([_prho(_merged(tabs[(Fa.name, a, k)], tabs[(Fb.name, b, k)], stat, cov), stat, cov)
                                     for k in range(n_null)])
                    fin = null[np.isfinite(null)]
                    nm = float(fin.mean()) if fin.size else float("nan")
                    m = _merged(tabs[(Fa.name, a, -1)], tabs[(Fb.name, b, -1)], stat, cov)
                    boots = np.array([_prho(m.iloc[brng.integers(0, len(m), len(m))], stat, cov) for _ in range(n_boot)]) \
                        if len(m) >= 20 else np.array([])
                    boots = boots[np.isfinite(boots)]
                    rr = real["rho_partial"]
                    c_real = np.sqrt(ceil(Fa.name, a, "real", stat, cs) * ceil(Fb.name, b, "real", stat, cs))
                    found.append({
                        "scope": scope, "fold_a": Fa.name, "fold_b": Fb.name,
                        "pool_a": a, "pool_b": b, "kind": kind, "core": bool(a in CORE and b in CORE),
                        "stat": stat, "covset": cs, "covariates": list(cov), "n": real["n"], "n_used": real["n_used"],
                        "rho_raw": real["rho"], "rho_deranged_draws": null.tolist(),
                        "rho_real": rr, "p_gene_shuffle": real["p_perm_partial"],
                        # the real side's gene bootstrap only; the paired interval on the excess is
                        # in the summary (covset pre), where it decides a reading
                        "rho_real_boot_q025": float(np.quantile(boots, 0.025)) if boots.size else float("nan"),
                        "rho_real_boot_q975": float(np.quantile(boots, 0.975)) if boots.size else float("nan"),
                        "rho_deranged_mean": nm, "n_null_finite": int(fin.size),
                        "rho_deranged_sd": float(fin.std(ddof=1)) if fin.size > 1 else float("nan"),
                        "rho_deranged_q975": float(np.quantile(fin, 0.975)) if fin.size > 1 else float("nan"),
                        "excess": rr - nm,
                        "excess_boot_q025": float(np.quantile(boots, 0.025)) - nm if boots.size else float("nan"),
                        "excess_boot_q975": float(np.quantile(boots, 0.975)) - nm if boots.size else float("nan"),
                        "p_deranged": float((np.sum(fin >= rr) + 1) / (fin.size + 1)) if np.isfinite(rr) else float("nan"),
                        "ceiling_truth_side": c_real,
                        "rho_real_disattenuated_truth_side": rr / c_real if np.isfinite(c_real) else float("nan"),
                    })
            log(f"   read {scope} {a} x {b} ({kind})")
        return found

    # (a) + (b): cross-line, every pool pairing; (c2) same line, two different single-line pools --
    # how much of the knockdown-tied per-gene error a gene carries when only the POOL changes, the
    # ceiling the cross-line excess is read against
    cross = [(a, b, "disjoint") for a, b in itertools.permutations(pools, 2)]
    cross += [(a, a, "shared") for a in pools]
    cross += [("recorded", "recorded", "recorded")]
    rows_out = read_rows(A, B, cross, "cross_line", n_perm)
    same = [(a, b, "same_line") for a, b in itertools.combinations(pools, 2)]
    for F in (A, B):
        rows_out += read_rows(F, F, same, f"same_line|{F.name}", 200)
    df = pd.DataFrame(rows_out)
    df.to_parquet(OUT / f"rows_{key}.parquet", index=False)
    out["rows"] = rows_out

    # the median excess over the disjoint pairings, with a gene bootstrap that resamples the SAME
    # genes for every pairing in a draw (the pairings share tables, so they are not independent)
    brng = np.random.default_rng(11)
    summary = {}
    for stat in STATS:
        for cs, cov0 in COVSETS.items():
            cov = cov_for(stat, cov0)
            s = df[(df["stat"] == stat) & (df["covset"] == cs)]
            xs = s[s["scope"] == "cross_line"]
            dis = xs[xs["kind"] == "disjoint"]
            core = dis[dis["core"]]
            rec = xs[xs["kind"] == "recorded"].iloc[0]
            merged = {(r.pool_a, r.pool_b): (_merged(tabs[(A.name, r.pool_a, -1)], tabs[(B.name, r.pool_b, -1)], stat, cov),
                                             r.rho_deranged_mean) for r in dis.itertuples()}
            universe = np.array(sorted(set().union(*[set(m.index) for m, _ in merged.values()])))
            med, med_paired = [], []
            # PAIRED for the pre-registered set (results review, 2026-09-29): the real and the
            # deranged rho are read on the same resampled genes, so the interval is the one of the
            # difference; N_PAIRED_NULL of the draws per resample keep it to minutes
            nulls = {pk: [_merged(tabs[(A.name, pk[0], k)], tabs[(B.name, pk[1], k)], stat, cov)
                          for k in range(min(n_null, N_PAIRED_NULL))] for pk in merged} if cs == "pre" else {}
            for _ in range(n_boot):
                pick = universe[brng.integers(0, len(universe), len(universe))]
                ex = [_prho(m.reindex(pick).dropna(), stat, cov) - nm for m, nm in merged.values()]
                ex = [e for e in ex if np.isfinite(e)]
                if ex:
                    med.append(float(np.median(ex)))
                if nulls:
                    exp_ = []
                    for pk, (m, _) in merged.items():
                        rb = _prho(m.reindex(pick).dropna(), stat, cov)
                        nb = [_prho(n_.reindex(pick).dropna(), stat, cov) for n_ in nulls[pk]]
                        nb = [x for x in nb if np.isfinite(x)]
                        if np.isfinite(rb) and nb:
                            exp_.append(rb - float(np.mean(nb)))
                    if exp_:
                        med_paired.append(float(np.median(exp_)))
            same_ex = {F.name: float(s[s["scope"] == f"same_line|{F.name}"]["excess"].median()) for F in (A, B)}
            geo = np.sqrt(same_ex[A.name] * same_ex[B.name]) if all(v > 0 for v in same_ex.values()) else float("nan")
            summary[f"{stat}|{cs}"] = {
                "covariates": list(cov),
                "disjoint_excess_median": float(dis["excess"].median()),
                "disjoint_excess_median_boot_q025": float(np.quantile(med, 0.025)) if med else float("nan"),
                "disjoint_excess_median_boot_q975": float(np.quantile(med, 0.975)) if med else float("nan"),
                "disjoint_excess_median_paired_q025": float(np.quantile(med_paired, 0.025)) if med_paired else float("nan"),
                "disjoint_excess_median_paired_q975": float(np.quantile(med_paired, 0.975)) if med_paired else float("nan"),
                "disjoint_excess_median_paired_centre": float(np.median(med_paired)) if med_paired else float("nan"),
                "disjoint_excess_min": float(dis["excess"].min()), "disjoint_excess_max": float(dis["excess"].max()),
                "disjoint_excess_core_median": float(core["excess"].median()),
                "disjoint_real_median": float(dis["rho_real"].median()),
                "disjoint_deranged_median": float(dis["rho_deranged_mean"].median()),
                "disjoint_deranged_draw_sd_median": float(dis["rho_deranged_sd"].median()),
                "disjoint_pairs_beyond_deranged_q975": int((dis["rho_real"] > dis["rho_deranged_q975"]).sum()),
                "disjoint_pairs": int(len(dis)),
                "shared_source_excess_median": float(xs[xs["kind"] == "shared"]["excess"].median()),
                "recorded_real": float(rec["rho_real"]), "recorded_deranged": float(rec["rho_deranged_mean"]),
                "recorded_excess": float(rec["excess"]),
                "same_line_excess_median": same_ex,
                "cross_over_same_line": (float(dis["excess"].median()) / geo) if np.isfinite(geo) and geo > 0 else float("nan"),
                "ceiling_truth_side_median": float(np.nanmedian(dis["ceiling_truth_side"])),
                "disjoint_real_disattenuated_truth_side_median": float(dis["rho_real_disattenuated_truth_side"].median()),
            }
            if cs == "pre":
                v = summary[f"{stat}|{cs}"]
                log(f"   {key} {stat:10s} disjoint: real {v['disjoint_real_median']:+.3f} deranged "
                    f"{v['disjoint_deranged_median']:+.3f} excess {v['disjoint_excess_median']:+.3f} "
                    f"(boot [{v['disjoint_excess_median_boot_q025']:+.3f}, {v['disjoint_excess_median_boot_q975']:+.3f}], "
                    f"paired [{v['disjoint_excess_median_paired_q025']:+.3f}, {v['disjoint_excess_median_paired_q975']:+.3f}]) "
                    f"core {v['disjoint_excess_core_median']:+.3f}; {v['disjoint_pairs_beyond_deranged_q975']}/"
                    f"{v['disjoint_pairs']} beyond q97.5; same-line excess "
                    f"{', '.join(f'{x:+.3f}' for x in same_ex.values())}; cross/same {v['cross_over_same_line']:.2f}; "
                    f"recorded excess {v['recorded_excess']:+.3f}; truth-side ceiling {v['ceiling_truth_side_median']:.2f}")
    out["summary"] = summary
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs="+", choices=list(PAIRS), default=list(PAIRS))
    ap.add_argument("--n-null", type=int, default=20, help="deranged-label draws per (fold, pool)")
    ap.add_argument("--n-perm", type=int, default=1000, help="gene shuffles behind each real rho's p")
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help="gene bootstrap draws")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logf = (OUT / "run.log").open("a")
    log = lambda m: _log(m, logf)                                   # noqa: E731
    log(f"--- t102_close.py pairs={args.pair} n_null={args.n_null} n_perm={args.n_perm}")
    need = sorted({f for p in args.pair for f in PAIRS[p][:2]})
    folds = {}
    for f in need:
        folds[f] = Fold(f, log)
    write_json(OUT / "gates.json", {f: F.gate_info for f, F in folds.items()})
    for key in args.pair:
        t0 = time.time()
        res = run_pair(key, folds, args.n_null, args.n_perm, args.n_boot, log)
        res["seconds"] = round(time.time() - t0)
        write_json(OUT / f"close_{key}.json", res)
        log(f"   wrote close_{key}.json ({res['seconds']}s)")
    log("done")
    (OUT / "CLOSE_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
