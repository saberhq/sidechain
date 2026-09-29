#!/usr/bin/env python
"""T102, step 1 -- is a gene's cross-line transfer error a property of the GENE (universal,
the same whichever line receives it) or of the receiving LINE? And which 3'UTR features
predict each part?

    .venv/bin/python scripts/t102_universal.py                      # all three parts
    .venv/bin/python scripts/t102_universal.py --part pairs decompose
    .venv/bin/python scripts/t102_universal.py --part corpus --n-reference 200

Saber's framing (hand-off 2026-09-28): a 3'UTR is the same in every cell line, so the layer
should speak for something universal per gene, not for a weighting scheme. This script asks
the data three ways, all from tables the earlier T102 sessions left under
``runs/probes/t102_utr_load/`` and from the essential-gene pseudobulks:

(a) **pairs** -- `pair_consistency` of each per-gene statistic between every pair of folds'
    ``per_gene_<fold>.parquet`` tables, on their shared genes, raw and partial on both
    folds' expression and measurement noise, against a gene shuffle. Two folds that hold
    out the SAME line share their truth and read the ceiling; two folds of DIFFERENT lines
    read the share of the error that belongs to the gene. Kill, pre-registered: if every
    cross-line pair's `nmae_g` and `sign_agree` consistency sits inside its shuffle, nothing
    per-gene is universal at this resolution and the thread ends there.
(b) **decompose** -- on the three full-panel folds (HCT116, HEK293T, K562 held out), each
    gene's across-line MEAN error (the universal part) and its across-line SPREAD (the
    line part), each correlated with the 3'UTR features -- the seven load features of the
    first read and, when ``derived/targetscan-vert_80/utr_seq_features.parquet`` exists,
    the composition, motif and cooperation counts from Saber's NAR 2016 paper -- partial
    on the mean expression and noise, against a gene shuffle.
(c) **corpus** -- the NAR 2016 reading on our own knockdowns: the essential-gene panels of
    HepG2, Jurkat, RPE1 and K562 carry DICER1, DROSHA, DGCR8, XPO5, TNRC6A (the miRNA
    pathway) and PUM1, HNRNPC, HNRNPL, PTBP1, CNOT1 (RBPs and decay). For each, the
    Spearman of its log2FC between every pair of lines, against the same statistic over a
    reference draw of shared knockdowns; the de-repressed set after DICER1 / DROSHA /
    DGCR8 per line and its overlap across lines (the paper's Supplementary Figure S1
    question -- is the miRNA-repressed set universal? -- measured on cells); and whether
    the TargetScan load predicts de-repression in each line and in the across-line mean.

Everything lands under ``runs/probes/t102_utr_load/universal/`` (``pairs.json``,
``decompose.json`` with one parquet per statistic, ``corpus.json``, ``run.log``). Nothing
here scores a submission and nothing here is a knob: it decides what a knob could be.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.eval.per_gene_transfer import (
    CE2_MIN_CPM,
    correlate,
    decompose_across_lines,
    derepressed,
    icc1,
    jaccard,
    knockdown_lfc,
    pair_consistency,
    residualize_ranks,
    share_in_mean,
    write_json,
)

PROBE = Path("~/data/sidechain/runs/probes/t102_utr_load").expanduser()
OUT = PROBE / "universal"
CACHE = Path("~/data/sidechain/cache/vcc2026").expanduser()
DERIVED = Path("~/data/sidechain/derived/targetscan-vert_80").expanduser()

HELD_OUT_LINE = {"loco_k562gwps": "K562", "loco_k562gwps_union": "K562",
                 "loco_k562gwps_union_ch272": "K562", "loco_hek293t": "HEK293T",
                 "loco_hek293t_ch272": "HEK293T", "loco_hct116": "HCT116"}
FULL_FOLDS = ["loco_hct116", "loco_hek293t", "loco_k562gwps_union"]
STATS = ["nmae_g", "sign_agree", "l1_scale", "log_ratio", "slope", "mean_abs_truth"]
LOAD_FEATURES = ["log_utr_len", "n_cons_sites", "sites_per_kb", "context_score", "n_families",
                 "n_cons_8mer", "n_noncons_sites"]
SEQ_FEATURES = ["au_content", "dn_UU", "dn_AA", "dn_AU", "dn_GC", "m_pum", "m_are_auuua",
                "m_polyu", "m_msi", "m_are_heptamer", "n_sites_rbp_within200", "n_sites_pum_within200",
                "n_sites_rbp_overlap", "frac_sites_rbp_overlap", "n_sites_last15pct"]
COVARIATES = ("log_ctrl_cpm", "log_truth_se")
#: The sensitivity set (review of 2026-09-28): the size of the truth and of OUR prediction, and
#: how many targets score the gene. Every fold's pool holds the other fold's held-out line and
#: H1 sits in every pool, so a cross-line agreement of the error can be an agreement of the
#: predictions; mean_abs_pred may be a mediator rather than a confounder, so it is reported as a
#: named sensitivity beside the pre-registered set, never swapped in silently.
COVARIATES_SENS = ("log_ctrl_cpm", "log_truth_se", "log_mean_abs_truth", "log_mean_abs_pred", "log_n_targets")
MIN_CELLS_PER_ARM = 20        # a knockdown arm below this is reported but not read (5-cell arms drove two medians)

PANELS = {"HepG2": "hepg2_all_pseudobulk.npz", "Jurkat": "jurkat_all_pseudobulk.npz",
          "RPE1": "rpe1_all_pseudobulk.npz", "K562": "k562_essential_all_pseudobulk.npz"}
PANEL_CONTROL = "control"
MIRNA_PATHWAY = ["DICER1", "DROSHA", "DGCR8", "XPO5", "TNRC6A"]
RBP_DECAY = ["PUM1", "HNRNPC", "HNRNPL", "PTBP1", "CNOT1"]
DEREPRESSION_KDS = ["DICER1", "DROSHA", "DGCR8"]


def _log(msg: str, fh) -> None:
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def load_per_gene() -> dict[str, pd.DataFrame]:
    out = {}
    for f in HELD_OUT_LINE:
        p = PROBE / f"per_gene_{f}.parquet"
        if p.exists():
            t = pd.read_parquet(p)
            t["log_mean_abs_pred"] = np.log10(t["mean_abs_pred"].astype(float).clip(lower=1e-6))
            t["log_n_targets"] = np.log10(t["n_targets"].astype(float).clip(lower=1))
            if "log_mean_abs_truth" not in t.columns:
                t["log_mean_abs_truth"] = np.log10(t["mean_abs_truth"].astype(float).clip(lower=1e-6))
            out[f] = t
    return out


def with_seq_features(t: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Refresh the load features and join the sequence features from the CURRENT widened table
    (`utr_features.parquet`) on the representative transcript, so a rebuilt table (the 3'UTR
    length fix of 2026-09-28) reaches the reading without recomputing the per-gene statistics.
    Falls back to `utr_seq_features.parquet` when the widened table is absent."""
    if "transcript_id" not in t.columns:
        return t, []
    p = DERIVED / "utr_features.parquet"
    if p.exists():
        ft = pd.read_parquet(p)
        ft["log_utr_len"] = np.log10(ft["utr_len"].astype(float).clip(lower=1))
        load_cols = [c for c in LOAD_FEATURES if c in ft.columns] + (["n_noncons_sites_all"] if "n_noncons_sites_all" in ft.columns else [])
        seq_cols = [c for c in SEQ_FEATURES if c in ft.columns]
        base = t.drop(columns=[c for c in load_cols + seq_cols if c in t.columns])
        out = base.merge(ft[["transcript_id", *load_cols, *seq_cols]].drop_duplicates("transcript_id"),
                         on="transcript_id", how="left")
        return out, seq_cols + (["n_noncons_sites_all"] if "n_noncons_sites_all" in load_cols else [])
    p = DERIVED / "utr_seq_features.parquet"
    if not p.exists():
        return t, []
    seq = pd.read_parquet(p)
    cols = [c for c in SEQ_FEATURES if c in seq.columns]
    return t.merge(seq[["transcript_id", *cols]], on="transcript_id", how="left"), cols


# ------------------------------------------------------------------------ (a) pairs --

def run_pairs(tables: dict[str, pd.DataFrame], n_perm: int, log) -> dict:
    rows = []
    for a, b in itertools.combinations(sorted(tables), 2):
        same = HELD_OUT_LINE[a] == HELD_OUT_LINE[b]
        for stat in STATS:
            r = pair_consistency(tables[a], tables[b], stat, covariates=COVARIATES, n_perm=n_perm, seed=0)
            rs = pair_consistency(tables[a], tables[b], stat, covariates=COVARIATES_SENS, n_perm=n_perm, seed=0)
            rows.append({"fold_a": a, "fold_b": b, "line_a": HELD_OUT_LINE[a], "line_b": HELD_OUT_LINE[b],
                         "same_line": same, "stat": stat, **r,
                         "rho_partial_sens": rs.get("rho_partial", float("nan")),
                         "p_perm_partial_sens": rs.get("p_perm_partial", float("nan"))})
            log(f"   {a:26s} x {b:26s} {'same ' if same else 'cross'} {stat:14s} n {r.get('shared_genes', 0):5d} "
                f"rho {r['rho']:+.3f} (p {r['p_perm']:.3f})  |cov {r.get('rho_partial', float('nan')):+.3f} "
                f"(p {r.get('p_perm_partial', float('nan')):.3f})  |sens {rs.get('rho_partial', float('nan')):+.3f}")
    df = pd.DataFrame(rows)
    # the pre-registered kill: every cross-line pair inside its shuffle on nmae_g and sign_agree
    cross = df[~df["same_line"] & df["stat"].isin(["nmae_g", "sign_agree"]) & (df["shared_genes"] >= 200)]
    kill = bool(len(cross)) and bool((cross["p_perm_partial"] > 0.05).all())
    summary = {"rows": rows, "n_perm": n_perm, "covariates": list(COVARIATES), "covariates_sens": list(COVARIATES_SENS),
               "kill_rule": "every cross-line pair with >= 200 shared genes has p_perm_partial > 0.05 "
                            "on nmae_g and sign_agree",
               "kill_fires": kill,
               "cross_line_partial_rho": {
                   stat: {f"{r.fold_a}|{r.fold_b}": r.rho_partial for r in cross.itertuples() if r.stat == stat}
                   for stat in ("nmae_g", "sign_agree")}}
    return summary


# -------------------------------------------------------------------- (b) decompose --

def run_decompose(tables: dict[str, pd.DataFrame], n_perm: int, log) -> dict:
    have = [f for f in FULL_FOLDS if f in tables]
    if len(have) < 2:
        return {"skipped": f"need two of {FULL_FOLDS}, have {have}"}
    seq_cols: list[str] = []
    joined = {}
    for f in have:
        t, seq_cols = with_seq_features(tables[f])
        joined[f] = t
    features = LOAD_FEATURES + seq_cols
    # the features are per gene and identical across folds (one UTR table): take them from
    # the first fold's table by gene
    feat = joined[have[0]][["gene", *[c for c in features if c in joined[have[0]].columns]]].drop_duplicates("gene")
    out = {"folds": have, "n_perm": n_perm, "features": features, "stats": {}}
    for stat in STATS:
        d = decompose_across_lines(joined, stat, covariates=COVARIATES)
        d = d.merge(feat, on="gene", how="left")
        d.to_parquet(OUT / f"decompose_{stat}.parquet", index=False)
        cov = np.vstack([d[f"mean_{c}"].to_numpy(dtype=np.float64) for c in COVARIATES])
        cov_s = np.vstack([d[f"mean_{c}"].to_numpy(dtype=np.float64) for c in COVARIATES_SENS])
        # how much of the gene's error belongs to the gene: the naive share of variance in the
        # across-line mean floors near 1/k with no gene effect, so it is read next to its
        # within-fold gene-shuffle floor; ICC(1) on fold-centred values is the statistic with a zero
        vals = np.column_stack([d[f"mean_{stat}"] + d[f"dev_{f}_{stat}"] for f in have])
        share = share_in_mean(vals)
        rng = np.random.default_rng(0)
        floor = np.array([share_in_mean(np.column_stack([rng.permutation(vals[:, j]) for j in range(vals.shape[1])]))
                          for _ in range(200)])
        ranks = np.column_stack([pd.Series(vals[:, j]).rank().to_numpy() for j in range(vals.shape[1])])
        table = []
        for target in (f"mean_{stat}", f"absdev_{stat}"):
            for c in features:
                if c not in d.columns:
                    continue
                r = correlate(d[target].to_numpy(dtype=np.float64), d[c].to_numpy(dtype=np.float64),
                              covariates=cov, n_perm=n_perm, seed=0)
                rs = correlate(d[target].to_numpy(dtype=np.float64), d[c].to_numpy(dtype=np.float64),
                               covariates=cov_s, n_perm=n_perm, seed=0)
                table.append({"target": target, "feature": c, **r,
                              "rho_partial_sens": rs.get("rho_partial", float("nan")),
                              "p_perm_partial_sens": rs.get("p_perm_partial", float("nan"))})
                log(f"   {stat:14s} {target:22s} x {c:24s} n {r['n']:5d} rho {r['rho']:+.3f} (p {r['p_perm']:.3f}) "
                    f"|cov {r.get('rho_partial', float('nan')):+.3f} (p {r.get('p_perm_partial', float('nan')):.3f}) "
                    f"|sens {rs.get('rho_partial', float('nan')):+.3f} (p {rs.get('p_perm_partial', float('nan')):.3f})")
        out["stats"][stat] = {"genes": int(len(d)),
                              "share_in_mean": share,
                              "share_in_mean_shuffle_floor": {"mean": float(floor.mean()), "q975": float(np.quantile(floor, 0.975))},
                              "icc1": icc1(vals), "icc1_ranks": icc1(ranks),
                              "median_absdev": float(d[f"absdev_{stat}"].median()),
                              "median_mean": float(d[f"mean_{stat}"].median()),
                              "correlations": table}
        log(f"   {stat}: {len(d)} genes on all {len(have)} folds; share in the across-line mean {share:.2f} "
            f"(shuffle floor {floor.mean():.2f}); ICC(1) {out['stats'][stat]['icc1']:.2f}, on ranks {out['stats'][stat]['icc1_ranks']:.2f}")
    return out


# ----------------------------------------------------------------------- (c) corpus --

def _bridge() -> dict[str, str]:
    """symbol -> Ensembl id: the JSON the Mac wrote, else `scripts/t102_utr_load.symbol_to_ensg`."""
    p = PROBE / "bridge" / "symbol_to_ensg.json"
    if p.exists():
        return json.loads(p.read_text())
    spec = importlib.util.spec_from_file_location("t102_utr_load", Path(__file__).with_name("t102_utr_load.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.symbol_to_ensg()


def load_panel(name: str, labels: list[str]) -> PseudobulkSums | None:
    path = CACHE / PANELS[name]
    if not path.exists():
        return None
    all_labels, genes = PseudobulkSums.peek(path) if hasattr(PseudobulkSums, "peek") else (None, None)
    if all_labels is None:
        z = np.load(path, allow_pickle=True)
        all_labels = [str(x) for x in z["labels"]]
        genes = [str(x) for x in z["genes"]]
    want = [lab for lab in labels if lab in all_labels]
    if PANEL_CONTROL not in all_labels:
        raise SystemExit(f"{path.name}: no {PANEL_CONTROL!r} label")
    return PseudobulkSums.load_subset(path, [PANEL_CONTROL, *want], list(genes))


def _partial_rho(y: np.ndarray, x: np.ndarray, cov: np.ndarray) -> float:
    """Spearman's partial of y and x on `cov` ([k, n]), no permutation: for reference draws."""
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(cov).all(axis=0)
    if ok.sum() < 20:
        return float("nan")
    return float(np.corrcoef(residualize_ranks(x[ok], cov[:, ok]), residualize_ranks(y[ok], cov[:, ok]))[0, 1])


def _percentile(value: float, ref) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    return float((ref < value).mean()) if ref.size and np.isfinite(value) else float("nan")


def run_corpus(n_reference: int, n_perm: int, log) -> dict:
    rng = np.random.default_rng(0)
    # the shared knockdown vocabulary, for the reference draw
    labels_by_panel = {}
    for name in PANELS:
        path = CACHE / PANELS[name]
        if not path.exists():
            log(f"   missing {path.name}: {name} skipped")
            continue
        z = np.load(path, allow_pickle=True)
        labels_by_panel[name] = set(str(x) for x in z["labels"]) - {PANEL_CONTROL}
    if len(labels_by_panel) < 2:
        return {"skipped": "fewer than two panels on disk"}
    shared = sorted(set.intersection(*labels_by_panel.values()))
    # a named knockdown counts when two or more panels carry it (DROSHA, DGCR8, TNRC6A and PUM1
    # are in HepG2, Jurkat and RPE1 but not in the K562 essential panel); the reference draw
    # stays on the knockdowns every panel shares
    named = [k for k in MIRNA_PATHWAY + RBP_DECAY
             if sum(k in labs for labs in labels_by_panel.values()) >= 2]
    pool = [k for k in shared if k not in named]
    reference = sorted(rng.choice(pool, size=min(n_reference, len(pool)), replace=False).tolist())
    kds = named + reference
    log(f"   panels {list(labels_by_panel)}; shared knockdowns {len(shared)}; named present {named}; "
        f"reference draw {len(reference)}")
    panels = {name: load_panel(name, kds) for name in labels_by_panel}
    # one gene axis: the intersection of the panels' axes
    axis = sorted(set.intersection(*[set(pb.genes) for pb in panels.values()]))
    pos = {name: {g: i for i, g in enumerate(pb.genes)} for name, pb in panels.items()}
    idx = {name: np.array([pos[name][g] for g in axis]) for name in panels}
    log(f"   shared gene axis {len(axis)}")
    ctrl_cpm = {name: (pb.cpm_sum[pb.labels.index(PANEL_CONTROL)] / max(int(pb.n_cells[pb.labels.index(PANEL_CONTROL)]), 1))[idx[name]]
                for name, pb in panels.items()}
    expressed = {name: ctrl_cpm[name] >= CE2_MIN_CPM for name in panels}
    lfc, var, ncells, nde = {}, {}, {}, {}
    for name, pb in panels.items():
        for kd in kds:
            if kd in pb.labels:
                f, v = knockdown_lfc(pb, kd, PANEL_CONTROL)
                lfc[(name, kd)] = f[idx[name]]
                var[(name, kd)] = v[idx[name]]
                ncells[(name, kd)] = int(pb.n_cells[pb.labels.index(kd)])
                # the knockdown's strength: responsive genes (|z| > 3 and |lfc| > 0.25 on the
                # expressed set); cross-line agreement follows it (review of 2026-09-28)
                with np.errstate(divide="ignore", invalid="ignore"):
                    zz = np.abs(f[idx[name]]) / np.sqrt(v[idx[name]])
                nde[(name, kd)] = int((expressed[name] & np.isfinite(zz) & (zz > 3) & (np.abs(f[idx[name]]) > 0.25)).sum())
    names = list(panels)
    thin = {k for k, n in ncells.items() if n < MIN_CELLS_PER_ARM}
    log(f"   arms below {MIN_CELLS_PER_ARM} cells (reported, not read): "
        f"{sorted(f'{n}|{k} ({ncells[(n, k)]})' for (n, k) in thin if k in named)}")
    # (i) the same knockdown across lines: Spearman on genes expressed in both, ranked against
    # the reference knockdowns raw and after a straight line on log strength is taken out
    agree = []
    for kd in kds:
        for a, b in itertools.combinations(names, 2):
            if (a, kd) not in lfc or (b, kd) not in lfc:
                continue
            ok = expressed[a] & expressed[b] & np.isfinite(lfc[(a, kd)]) & np.isfinite(lfc[(b, kd)])
            if ok.sum() < 200:
                continue
            rho = float(pd.Series(lfc[(a, kd)][ok]).corr(pd.Series(lfc[(b, kd)][ok]), method="spearman"))
            group = ("mirna_pathway" if kd in MIRNA_PATHWAY else "rbp_decay" if kd in RBP_DECAY else "reference")
            agree.append({"knockdown": kd, "group": group, "line_a": a, "line_b": b, "genes": int(ok.sum()), "rho": rho,
                          "cells_a": ncells[(a, kd)], "cells_b": ncells[(b, kd)], "nde_a": nde[(a, kd)], "nde_b": nde[(b, kd)],
                          "thin": bool((a, kd) in thin or (b, kd) in thin),
                          "log_strength": float(np.log10(max(np.sqrt(nde[(a, kd)] * nde[(b, kd)]), 1.0)))})
    ag = pd.DataFrame(agree)
    refrows = ag[(ag["group"] == "reference") & ~ag["thin"]]
    strength_rho = float(refrows["rho"].corr(refrows["log_strength"], method="spearman")) if len(refrows) > 10 else float("nan")
    coef = np.polyfit(refrows["log_strength"], refrows["rho"], 1) if len(refrows) > 10 else np.array([0.0, 0.0])
    ag["rho_resid"] = ag["rho"] - np.polyval(coef, ag["log_strength"])
    ref = refrows["rho"].to_numpy()
    ref_resid = ag.loc[(ag["group"] == "reference") & ~ag["thin"], "rho_resid"].to_numpy()
    by_group = {}
    for g, sub in ag[~ag["thin"]].groupby("group"):
        by_group[g] = {"n": int(len(sub)), "median_rho": float(sub["rho"].median()),
                       "q25": float(sub["rho"].quantile(0.25)), "q75": float(sub["rho"].quantile(0.75)),
                       "median_rho_resid": float(sub["rho_resid"].median()),
                       "median_log_strength": float(sub["log_strength"].median())}
    per_kd = {}
    for kd in named:
        allp = ag[ag["knockdown"] == kd]
        sub = allp[~allp["thin"]]
        if not len(allp):
            continue
        med = float(sub["rho"].median()) if len(sub) else float("nan")
        med_resid = float(sub["rho_resid"].median()) if len(sub) else float("nan")
        per_kd[kd] = {"median_rho": med, "pairs_read": int(len(sub)), "pairs_thin": int(allp["thin"].sum()),
                      "pairs": allp[["line_a", "line_b", "genes", "rho", "cells_a", "cells_b", "nde_a", "nde_b", "thin"]].to_dict("records"),
                      "percentile_in_reference": _percentile(med, ref),
                      "median_rho_resid": med_resid, "percentile_strength_matched": _percentile(med_resid, ref_resid)}
        log(f"   {kd:8s} cross-line rho median {med:+.3f} over {len(sub)} pairs ({int(allp['thin'].sum())} thin dropped); "
            f"reference pct {per_kd[kd]['percentile_in_reference']:.2f}, strength-matched pct {per_kd[kd]['percentile_strength_matched']:.2f}")
    log(f"   reference: rho vs log strength Spearman {strength_rho:+.3f}, slope {coef[0]:+.3f} per log10 responsive genes")
    for g, s in by_group.items():
        log(f"   group {g:14s} pairs {s['n']:4d} median rho {s['median_rho']:+.3f} [{s['q25']:+.3f}, {s['q75']:+.3f}] "
            f"strength-matched residual {s['median_rho_resid']:+.3f}")
    # (ii) the de-repressed set after DICER1 / DROSHA / DGCR8, per line (centred on the expressed
    # median), its overlap across lines, and the same overlap for every reference knockdown in
    # the same two lines: a uniform draw is the wrong null, any knockdown's up-set repeats
    def upset(name, kd):
        return derepressed(lfc[(name, kd)], var[(name, kd)], expressed[name])
    derep = {(name, kd): upset(name, kd) for kd in DEREPRESSION_KDS for name in names if (name, kd) in lfc}
    overlap = []
    for a, b in itertools.combinations(names, 2):
        both = expressed[a] & expressed[b]
        n = int(both.sum())
        ref_enr = []
        for kd in reference:
            if (a, kd) in lfc and (b, kd) in lfc and (a, kd) not in thin and (b, kd) not in thin:
                ua, ub = upset(a, kd) & both, upset(b, kd) & both
                exp = int(ua.sum()) * int(ub.sum()) / n if n else 0.0
                if exp > 0:
                    ref_enr.append(int((ua & ub).sum()) / exp)
        ref_enr = np.array(ref_enr)
        for kd in DEREPRESSION_KDS:
            if (a, kd) not in derep or (b, kd) not in derep:
                continue
            da, db = derep[(a, kd)] & both, derep[(b, kd)] & both
            ka, kb = int(da.sum()), int(db.sum())
            expected = (ka * kb / n) if n else float("nan")
            observed = int((da & db).sum())
            enr = (observed / expected) if expected else float("nan")
            overlap.append({"knockdown": kd, "line_a": a, "line_b": b, "expressed_both": n,
                            "set_a": ka, "set_b": kb, "overlap": observed, "expected_random": expected,
                            "jaccard": jaccard(da, db), "enrichment": enr,
                            "thin": bool((a, kd) in thin or (b, kd) in thin),
                            "reference_enrichment": {"n": int(ref_enr.size),
                                                     "median": float(np.median(ref_enr)) if ref_enr.size else float("nan"),
                                                     "q25": float(np.quantile(ref_enr, 0.25)) if ref_enr.size else float("nan"),
                                                     "q75": float(np.quantile(ref_enr, 0.75)) if ref_enr.size else float("nan"),
                                                     "q90": float(np.quantile(ref_enr, 0.90)) if ref_enr.size else float("nan")},
                            "percentile_in_reference": _percentile(enr, ref_enr)})
            log(f"   {kd:7s} {a:6s} x {b:6s}: sets {ka} / {kb} of {n}, overlap {observed} (x{enr:.2f} random; reference "
                f"median x{overlap[-1]['reference_enrichment']['median']:.2f}, pct {overlap[-1]['percentile_in_reference']:.2f})"
                f"{'  THIN' if overlap[-1]['thin'] else ''}")
    # (iii) the TargetScan load against de-repression, per line and in the across-line mean
    bridge = _bridge()
    load = pd.read_parquet(DERIVED / "utr_load.parquet")
    by_id = load.drop_duplicates("gene_id").set_index("gene_id")
    by_sym = load.drop_duplicates("symbol").set_index("symbol")
    rows = []
    for g in axis:                                  # Ensembl id through the bridge, then the symbol
        e = bridge.get(g)
        rows.append(by_id.loc[e] if e in by_id.index else (by_sym.loc[g] if g in by_sym.index else None))
    has = np.array([r is not None for r in rows])
    feat = {c: np.array([float(r[c]) if r is not None else np.nan for r in rows]) for c in ("n_cons_sites", "context_score", "utr_len")}
    feat["log_utr_len"] = np.log10(np.clip(feat["utr_len"], 1, None))
    log(f"   UTR table joined for {int(has.sum())} of {len(axis)} axis genes")
    # every reference knockdown's own partial rho with the site count, per line and for the
    # across-line mean over the lines a named knockdown has, so a named knockdown is ranked
    # against knockdowns of the same line(s) and the mean's rise is ranked against the same
    # averaging of the references
    cov_line = {name: np.vstack([np.log10(np.clip(ctrl_cpm[name], 1e-3, None))]) for name in names}
    ref_site = {name: np.array([_partial_rho(np.where(expressed[name], lfc[(name, kd)], np.nan), feat["n_cons_sites"], cov_line[name])
                                for kd in reference if (name, kd) in lfc and (name, kd) not in thin]) for name in names}
    load_vs = []
    for kd in DEREPRESSION_KDS:
        stack, lines_used = [], []
        for name in names:
            if (name, kd) not in lfc:
                continue
            y = np.where(expressed[name], lfc[(name, kd)], np.nan)
            cov = cov_line[name]
            cov_len = np.vstack([cov, feat["log_utr_len"]])
            for c in ("n_cons_sites", "context_score", "log_utr_len"):
                r = correlate(y, feat[c], covariates=cov, n_perm=n_perm, seed=0)
                row = {"knockdown": kd, "line": name, "feature": c, "cells": ncells[(name, kd)], "thin": bool((name, kd) in thin), **r}
                if c == "n_cons_sites":
                    row["rho_partial_expr_len"] = _partial_rho(y, feat[c], cov_len)
                    row["percentile_in_reference_line"] = _percentile(r.get("rho_partial", float("nan")), ref_site[name])
                    row["reference_line_median"] = float(np.nanmedian(ref_site[name])) if ref_site[name].size else float("nan")
                load_vs.append(row)
                log(f"   {kd:7s} {name:6s} lfc x {c:14s} n {r['n']:5d} rho {r['rho']:+.3f} (p {r['p_perm']:.3f}) "
                    f"|expr {r.get('rho_partial', float('nan')):+.3f} (p {r.get('p_perm_partial', float('nan')):.3f})"
                    + (f" |expr+len {row['rho_partial_expr_len']:+.3f}; reference-line pct {row['percentile_in_reference_line']:.2f}" if c == "n_cons_sites" else "")
                    + ("  THIN" if row["thin"] else ""))
            # the paper's Figure 4 shape: sites versus no sites
            s = feat["n_cons_sites"]
            ok = np.isfinite(y) & np.isfinite(s)
            load_vs.append({"knockdown": kd, "line": name, "feature": "median_lfc_sites_vs_none",
                            "with_sites": float(np.median(y[ok & (s > 0)])), "no_sites": float(np.median(y[ok & (s == 0)])),
                            "n_with": int((ok & (s > 0)).sum()), "n_none": int((ok & (s == 0)).sum())})
            stack.append(y)
            lines_used.append(name)
        if len(stack) >= 2:
            m = np.vstack(stack)
            mean_lfc = np.where(np.isfinite(m).all(axis=0), m.mean(axis=0), np.nan)
            cov = np.vstack([np.log10(np.clip(np.mean([ctrl_cpm[n] for n in lines_used], axis=0), 1e-3, None))])
            # the same averaging over the same lines for every reference knockdown they all carry
            ref_mean = []
            for rk in reference:
                if all((n, rk) in lfc and (n, rk) not in thin for n in lines_used):
                    rm = np.vstack([np.where(expressed[n], lfc[(n, rk)], np.nan) for n in lines_used])
                    ref_mean.append(_partial_rho(np.where(np.isfinite(rm).all(axis=0), rm.mean(axis=0), np.nan),
                                                 feat["n_cons_sites"], cov))
            ref_mean = np.array(ref_mean)
            for c in ("n_cons_sites", "context_score", "log_utr_len"):
                r = correlate(mean_lfc, feat[c], covariates=cov, n_perm=n_perm, seed=0)
                row = {"knockdown": kd, "line": "MEAN_ACROSS_LINES", "lines": lines_used, "feature": c, **r}
                if c == "n_cons_sites":
                    row["rho_partial_expr_len"] = _partial_rho(mean_lfc, feat[c], np.vstack([cov, feat["log_utr_len"]]))
                    row["percentile_in_reference_mean"] = _percentile(r.get("rho_partial", float("nan")), ref_mean)
                    row["reference_mean_median"] = float(np.nanmedian(ref_mean)) if ref_mean.size else float("nan")
                    row["reference_mean_n"] = int(ref_mean.size)
                load_vs.append(row)
                log(f"   {kd:7s} MEAN   lfc x {c:14s} n {r['n']:5d} rho {r['rho']:+.3f} (p {r['p_perm']:.3f}) "
                    f"|expr {r.get('rho_partial', float('nan')):+.3f} (p {r.get('p_perm_partial', float('nan')):.3f})"
                    + (f" |expr+len {row['rho_partial_expr_len']:+.3f}; reference-mean pct {row['percentile_in_reference_mean']:.2f} "
                       f"(median {row['reference_mean_median']:+.3f}, n {row['reference_mean_n']})" if c == "n_cons_sites" else ""))
    return {"panels": names, "control": PANEL_CONTROL, "shared_knockdowns": len(shared),
            "named_present": named, "reference": reference, "axis_genes": len(axis),
            "axis_note": "the genes every panel carries; the control CPM >= 5 rule keeps all but a few of them",
            "expressed_per_line": {n: int(e.sum()) for n, e in expressed.items()},
            "min_cells_per_arm": MIN_CELLS_PER_ARM,
            "thin_arms": sorted(f"{n}|{k} ({ncells[(n, k)]})" for (n, k) in thin if k in named),
            "strength": {"definition": "responsive genes: expressed, |z| > 3 and |lfc| > 0.25; a pair's strength is "
                                       "log10 of the geometric mean of its two arms' counts",
                         "reference_spearman_rho_vs_strength": strength_rho,
                         "reference_fit_slope_intercept": [float(coef[0]), float(coef[1])]},
            "agreement_by_group": by_group, "agreement_named": per_kd, "agreement_rows": agree,
            "derepression": {"rule": "expressed & centred lfc > 0.25 & centred lfc / sqrt(var) > 2 "
                                     "(centred on the expressed median; Poisson-floored delta-method var)",
                             "sets": {f"{n}|{kd}": int(v.sum()) for (n, kd), v in derep.items()},
                             "overlap": overlap},
            "load_vs_derepression": load_vs, "utr_joined": int(has.sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", nargs="+", choices=["pairs", "decompose", "corpus"],
                    default=["pairs", "decompose", "corpus"])
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-reference", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logf = (OUT / "run.log").open("a")
    log = lambda m: _log(m, logf)                                   # noqa: E731
    log(f"--- t102_universal.py {time.strftime('%Y-%m-%dT%H:%M:%S')} parts={args.part} n_perm={args.n_perm}")
    tables = load_per_gene()
    log(f"per-gene tables: {', '.join(f'{k} ({len(v)})' for k, v in tables.items())}")
    if "pairs" in args.part:
        t0 = time.time()
        log("== (a) pairs")
        res = run_pairs(tables, args.n_perm, log)
        res["seconds"] = round(time.time() - t0)
        write_json(OUT / "pairs.json", res)
        log(f"   kill fires: {res['kill_fires']}  ({res['seconds']}s)")
    if "decompose" in args.part:
        t0 = time.time()
        log("== (b) decompose")
        res = run_decompose(tables, args.n_perm, log)
        res["seconds"] = round(time.time() - t0)
        write_json(OUT / "decompose.json", res)
    if "corpus" in args.part:
        t0 = time.time()
        log("== (c) corpus")
        res = run_corpus(args.n_reference, args.n_perm, log)
        res["seconds"] = round(time.time() - t0)
        write_json(OUT / "corpus.json", res)
    log("done")


if __name__ == "__main__":
    sys.exit(main())
