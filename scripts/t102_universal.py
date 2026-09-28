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
    jaccard,
    knockdown_lfc,
    pair_consistency,
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
                "m_polyu", "m_msi", "n_sites_rbp_within200", "n_sites_pum_within200",
                "n_sites_rbp_overlap", "frac_sites_rbp_overlap", "n_sites_last15pct"]
COVARIATES = ("log_ctrl_cpm", "log_truth_se")

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
            out[f] = pd.read_parquet(p)
    return out


def with_seq_features(t: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Join the sequence-feature table on the representative transcript, when it exists."""
    p = DERIVED / "utr_seq_features.parquet"
    if not p.exists() or "transcript_id" not in t.columns:
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
            rows.append({"fold_a": a, "fold_b": b, "line_a": HELD_OUT_LINE[a], "line_b": HELD_OUT_LINE[b],
                         "same_line": same, "stat": stat, **r})
            log(f"   {a:26s} x {b:26s} {'same ' if same else 'cross'} {stat:14s} n {r.get('shared_genes', 0):5d} "
                f"rho {r['rho']:+.3f} (p {r['p_perm']:.3f})  |cov {r.get('rho_partial', float('nan')):+.3f} "
                f"(p {r.get('p_perm_partial', float('nan')):.3f})")
    df = pd.DataFrame(rows)
    # the pre-registered kill: every cross-line pair inside its shuffle on nmae_g and sign_agree
    cross = df[~df["same_line"] & df["stat"].isin(["nmae_g", "sign_agree"]) & (df["shared_genes"] >= 200)]
    kill = bool(len(cross)) and bool((cross["p_perm_partial"] > 0.05).all())
    summary = {"rows": rows, "n_perm": n_perm, "covariates": list(COVARIATES),
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
        # how much of the gene's error is universal: the share of variance in the mean
        vals = np.column_stack([d[f"mean_{stat}"] + d[f"dev_{f}_{stat}"] for f in have])
        var_total = float(np.var(vals, ddof=1))
        var_mean = float(np.var(d[f"mean_{stat}"], ddof=1))
        table = []
        for target in (f"mean_{stat}", f"absdev_{stat}"):
            for c in features:
                if c not in d.columns:
                    continue
                r = correlate(d[target].to_numpy(dtype=np.float64), d[c].to_numpy(dtype=np.float64),
                              covariates=cov, n_perm=n_perm, seed=0)
                table.append({"target": target, "feature": c, **r})
                log(f"   {stat:14s} {target:22s} x {c:24s} n {r['n']:5d} rho {r['rho']:+.3f} (p {r['p_perm']:.3f}) "
                    f"|cov {r.get('rho_partial', float('nan')):+.3f} (p {r.get('p_perm_partial', float('nan')):.3f})")
        out["stats"][stat] = {"genes": int(len(d)), "var_total": var_total, "var_of_mean": var_mean,
                              "share_universal": (var_mean / var_total) if var_total > 0 else float("nan"),
                              "median_absdev": float(d[f"absdev_{stat}"].median()),
                              "median_mean": float(d[f"mean_{stat}"].median()),
                              "correlations": table}
        log(f"   {stat}: {len(d)} genes on all {len(have)} folds; variance share in the across-line mean "
            f"{out['stats'][stat]['share_universal']:.2f}")
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
    lfc, var = {}, {}
    for name, pb in panels.items():
        for kd in kds:
            if kd in pb.labels:
                f, v = knockdown_lfc(pb, kd, PANEL_CONTROL)
                lfc[(name, kd)] = f[idx[name]]
                var[(name, kd)] = v[idx[name]]
    # (i) the same knockdown across lines: Spearman on genes expressed in both
    names = list(panels)
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
            agree.append({"knockdown": kd, "group": group, "line_a": a, "line_b": b, "genes": int(ok.sum()), "rho": rho})
    ag = pd.DataFrame(agree)
    by_group = {}
    for g, sub in ag.groupby("group"):
        by_group[g] = {"n": int(len(sub)), "median_rho": float(sub["rho"].median()),
                       "q25": float(sub["rho"].quantile(0.25)), "q75": float(sub["rho"].quantile(0.75))}
    ref = ag[ag["group"] == "reference"]["rho"].to_numpy()
    per_kd = {}
    for kd in named:
        sub = ag[ag["knockdown"] == kd]
        if not len(sub):
            continue
        med = float(sub["rho"].median())
        per_kd[kd] = {"median_rho": med, "pairs": sub[["line_a", "line_b", "genes", "rho"]].to_dict("records"),
                      "percentile_in_reference": float((ref < med).mean()) if ref.size else float("nan")}
        log(f"   {kd:8s} cross-line rho median {med:+.3f} (reference percentile {per_kd[kd]['percentile_in_reference']:.2f})")
    for g, s in by_group.items():
        log(f"   group {g:14s} pairs {s['n']:4d} median rho {s['median_rho']:+.3f} [{s['q25']:+.3f}, {s['q75']:+.3f}]")
    # (ii) the de-repressed set after DICER1 / DROSHA / DGCR8, per line, and its overlap
    derep = {}
    for kd in DEREPRESSION_KDS:
        for name in names:
            if (name, kd) in lfc:
                derep[(name, kd)] = derepressed(lfc[(name, kd)], var[(name, kd)], expressed[name])
    overlap = []
    for kd in DEREPRESSION_KDS:
        for a, b in itertools.combinations(names, 2):
            if (a, kd) in derep and (b, kd) in derep:
                da, db = derep[(a, kd)], derep[(b, kd)]
                both = expressed[a] & expressed[b]
                n = int(both.sum())
                ka, kb = int((da & both).sum()), int((db & both).sum())
                expected = (ka * kb / n) if n else float("nan")            # random overlap of two sets of these sizes
                observed = int((da & db & both).sum())
                overlap.append({"knockdown": kd, "line_a": a, "line_b": b, "expressed_both": n,
                                "set_a": ka, "set_b": kb, "overlap": observed,
                                "expected_random": expected, "jaccard": jaccard(da & both, db & both),
                                "enrichment": (observed / expected) if expected else float("nan")})
                log(f"   {kd:7s} {a:6s} x {b:6s}: sets {ka} / {kb} of {n} expressed, overlap {observed} "
                    f"(random {expected:.1f}, x{(observed / expected) if expected else float('nan'):.1f}), jaccard {overlap[-1]['jaccard']:.3f}")
    # (iii) the TargetScan load against de-repression, per line and in the across-line mean
    bridge = _bridge()
    load = pd.read_parquet(DERIVED / "utr_load.parquet").drop_duplicates("gene_id").set_index("gene_id")
    ensg = [bridge.get(g) for g in axis]
    has = np.array([e in load.index for e in ensg])
    feat = {c: np.array([load.at[e, c] if h else np.nan for e, h in zip(ensg, has)], dtype=float)
            for c in ("n_cons_sites", "context_score", "utr_len")}
    feat["log_utr_len"] = np.log10(np.clip(feat["utr_len"], 1, None))
    log(f"   UTR table joined for {int(has.sum())} of {len(axis)} axis genes")
    load_vs = []
    for kd in DEREPRESSION_KDS:
        stack = []
        for name in names:
            if (name, kd) not in lfc:
                continue
            y = np.where(expressed[name], lfc[(name, kd)], np.nan)
            cov = np.vstack([np.log10(np.clip(ctrl_cpm[name], 1e-3, None))])
            for c in ("n_cons_sites", "context_score", "log_utr_len"):
                r = correlate(y, feat[c], covariates=cov, n_perm=n_perm, seed=0)
                load_vs.append({"knockdown": kd, "line": name, "feature": c, **r})
                log(f"   {kd:7s} {name:6s} lfc x {c:14s} n {r['n']:5d} rho {r['rho']:+.3f} (p {r['p_perm']:.3f}) "
                    f"|expr {r.get('rho_partial', float('nan')):+.3f} (p {r.get('p_perm_partial', float('nan')):.3f})")
            # the paper's Figure 4 shape: sites versus no sites
            s = feat["n_cons_sites"]
            ok = np.isfinite(y) & np.isfinite(s)
            load_vs.append({"knockdown": kd, "line": name, "feature": "median_lfc_sites_vs_none",
                            "with_sites": float(np.median(y[ok & (s > 0)])), "no_sites": float(np.median(y[ok & (s == 0)])),
                            "n_with": int((ok & (s > 0)).sum()), "n_none": int((ok & (s == 0)).sum())})
            stack.append(y)
        if len(stack) >= 2:
            m = np.vstack(stack)
            mean_lfc = np.where(np.isfinite(m).all(axis=0), m.mean(axis=0), np.nan)
            cov = np.vstack([np.log10(np.clip(np.mean([ctrl_cpm[n] for n in names if (n, kd) in lfc], axis=0), 1e-3, None))])
            for c in ("n_cons_sites", "context_score", "log_utr_len"):
                r = correlate(mean_lfc, feat[c], covariates=cov, n_perm=n_perm, seed=0)
                load_vs.append({"knockdown": kd, "line": "MEAN_ACROSS_LINES", "feature": c, **r})
                log(f"   {kd:7s} MEAN   lfc x {c:14s} n {r['n']:5d} rho {r['rho']:+.3f} (p {r['p_perm']:.3f}) "
                    f"|expr {r.get('rho_partial', float('nan')):+.3f} (p {r.get('p_perm_partial', float('nan')):.3f})")
    return {"panels": names, "control": PANEL_CONTROL, "shared_knockdowns": len(shared),
            "named_present": named, "reference": reference, "axis_genes": len(axis),
            "expressed_per_line": {n: int(e.sum()) for n, e in expressed.items()},
            "agreement_by_group": by_group, "agreement_named": per_kd, "agreement_rows": agree,
            "derepression": {"rule": "expressed & lfc > 0.25 & lfc/sqrt(var) > 2 (Poisson-floored delta-method var)",
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
