#!/usr/bin/env python
"""T102 -- what a UNIVERSAL per-gene table could still speak for: the gene-consistent part of the
TRUTH, not of our error.

    .venv/bin/python scripts/t102_universal_truth.py
    .venv/bin/python scripts/t102_universal_truth.py --n-perm 2000

Step 1 (idea file, Outcome 2026-09-28) found the true response size to be the most gene-consistent
quantity we measure: a gene that responds strongly in one held-out line responds strongly in the
others (ICC(1) 0.46, 0.70 on ranks, over 1,922 genes on the three full folds), while the error's
amplitude part carries almost nothing (ICC 0.02). A table that is the same in every line can only
speak for something that is the same in every line, so this reads the survey's two MEASURED
universal per-gene tables -- beside the 3'UTR features -- against that gene-consistent truth:

- **the consensus mRNA half-life** (Agarwal & Kelley 2022, Table S2, `half-life (PC1)` over 54
  human samples; the authors found no detectable cell-type signal), `MRNAStabilitySource.halflife_table`;
- **codon optimality** (Wu et al. 2019's codon stability coefficients, which agree across their
  lines at Spearman 0.91-0.95, averaged over each gene's ORF on TargetScan's representative
  transcript), `MRNAStabilitySource.codon_table`.

Three per-gene truth statistics per fold (HCT116, HEK293T, K562 held out; exact Wilcoxon gate):
  `log_mean_abs_truth`  mean |log2FC| over the gene's gated cells (the ICC 0.46 quantity; genes
                        with >= 8 gated targets);
  `de_freq`             the share of the fold's scored knockdowns whose gate holds the gene (how
                        often ANY knockdown moves it; every expressed gene);
  `log_mean_abs_all`    mean |log2FC| over every scored knockdown, gate-free (|lfc| capped at 10).
Each has its prediction-side twin from the shipped recipe's pooled delta (closed form, alpha 1.35):
`log_mean_abs_pred` on the same cells, and the gate-free `log_mean_abs_pred_all`.

Read three ways: (1) ICC(1) of each statistic across the three lines -- how gene-consistent it is;
(2) Spearman of each table with the across-line mean and with each line alone, partial on control
expression and the truth's noise (and the number of gated targets for the gated statistic),
against a gene shuffle -- what the table predicts about the universal truth; (3) the same with the
PREDICTION's twin also held -- what the table predicts that the pooled delta does not already
carry. A table that only speaks through (2) and goes silent in (3) has nothing to add to this
pipeline. Beside them, how well the pooled prediction's own per-gene size tracks the truth's, and
two line-specific contrasts: Table S2's K562 and HEK293 columns against PC1, and Wu's per-line
decay rates against the matched fold (293T endogenous -> HEK293T, K562 SLAM-seq -> K562).

Lands in ``runs/probes/t102_utr_load/universal_truth/``. Nothing here is a knob.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sidechain.eval.per_gene_transfer import (CE2_MIN_GATE, correlate, icc1, per_gene_stats, pred_lfc,
                                              residualize_ranks, write_json)
from sidechain.submit.build import sources_from_specs

HERE = Path(__file__).parent


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


close = _sibling("t102_close")
DATA = close.DATA
PROBE = close.PROBE
OUT = PROBE / "universal_truth"
ALPHA = close.ALPHA
FOLDS = list(close.FOLDS)
MIN_TARGETS = 8
LFC_CAP = 10.0
HALFLIFE = DATA / "derived/halflife-agarwal-kelley-2022/halflife.parquet"
CODON = DATA / "derived/codon-optimality-wu2019/codon_score.parquet"
DECAY = DATA / "derived/codon-optimality-wu2019/decay_rates.parquet"
UTR = DATA / "derived/targetscan-vert_80/utr_features.parquet"
TRUTH_STATS = {  # statistic -> (its prediction twin, gated?)
    "log_mean_abs_truth": ("log_mean_abs_pred", True),
    "de_freq": ("log_mean_abs_pred_all", False),
    "log_mean_abs_all": ("log_mean_abs_pred_all", False),
}
UTR_FEATURES = ["log_utr_len", "n_cons_sites", "au_content", "dn_UU"]
MATCHED_DECAY = {"loco_hek293t": "decay_rate_293T_endogenous", "loco_k562gwps_union": "decay_rate_k562_SLAM_seq"}


def _log(msg: str, fh) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def fold_table(fold: str, log) -> pd.DataFrame:
    """One row per expressed gene of the fold: the three truth statistics and their twins."""
    t0 = time.time()
    F = close.Fold(fold, log, keep_halves=False)
    targets = list(F.full.targets)
    srcs = sources_from_specs(close._recorded_specs(fold), [])
    delta, covered = close.pool_delta(targets, srcs, F.axis)
    del srcs
    P = pred_lfc(delta, targets, F.axis, F.frac, F.full.ctrl_mean_cpm, alpha=ALPHA, covered=covered)
    st = per_gene_stats(P, F.y, F.gate, F.axis, min_targets=MIN_TARGETS, truth=F.full)
    st = st.set_index("gene")
    scored = F.gate.sum(axis=1) >= CE2_MIN_GATE
    u = F.full.universe
    own = np.zeros_like(F.gate)
    for i, t in enumerate(targets):
        if t in F.gpos:
            own[i, F.gpos[t]] = True
    keep = scored[:, None] & ~own
    y = np.clip(np.where(np.isfinite(F.y), F.y, 0.0), -LFC_CAP, LFC_CAP)
    p = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    n_scored = keep.sum(axis=0)
    out = pd.DataFrame({"gene": F.axis})
    out["log_ctrl_cpm"] = np.log10(np.clip(F.full.ctrl_mean_cpm, 1e-3, None))
    out["de_freq"] = (F.gate & keep).sum(axis=0) / np.maximum(n_scored, 1)
    out["log_mean_abs_all"] = np.log10(np.clip(np.where(keep, np.abs(y), 0.0).sum(axis=0) / np.maximum(n_scored, 1), 1e-6, None))
    out["log_mean_abs_pred_all"] = np.log10(np.clip(np.where(keep, np.abs(p), 0.0).sum(axis=0) / np.maximum(n_scored, 1), 1e-6, None))
    out = out[u].copy()
    g = out["gene"]
    out["log_mean_abs_truth"] = np.log10(st["mean_abs_truth"].reindex(g).astype(float).clip(lower=1e-6)).to_numpy()
    out["log_mean_abs_pred"] = np.log10(st["mean_abs_pred"].reindex(g).astype(float).clip(lower=1e-6)).to_numpy()
    out["log_truth_se"] = np.log10(st["truth_se"].reindex(g).astype(float).clip(lower=1e-6)).to_numpy()
    out["log_n_targets"] = np.log10(st["n_targets"].reindex(g).astype(float).clip(lower=1)).to_numpy()
    out["fold"] = fold
    log(f"   {fold}: {len(out)} expressed genes, {int(np.isfinite(out['log_mean_abs_truth']).sum())} with >= "
        f"{MIN_TARGETS} gated targets, {int(scored.sum())} scored knockdowns ({time.time() - t0:.0f}s)")
    return out


def features(genes: pd.Series) -> tuple[pd.DataFrame, dict]:
    """The universal tables on our symbols: Ensembl id through the bridge, then symbol."""
    bridge = json.loads((PROBE / "bridge" / "symbol_to_ensg.json").read_text())
    ens = genes.map(bridge)
    out = pd.DataFrame({"gene": genes, "gene_id": ens})
    info = {}

    def join(path: Path, cols: list[str], tag: str):
        if not path.exists():
            info[tag] = f"missing {path}"
            return
        t = pd.read_parquet(path)
        have = [c for c in cols if c in t.columns]
        by_id = t.drop_duplicates("gene_id").set_index("gene_id")[have]
        got = by_id.reindex(out["gene_id"]).reset_index(drop=True)
        if "symbol" in t.columns:          # fall back to the symbol where the bridge found no id
            by_sym = t.drop_duplicates("symbol").set_index("symbol")[have]
            alt = by_sym.reindex(out["gene"]).reset_index(drop=True)
            got = got.fillna(alt)
        for c in have:
            out[c] = got[c].to_numpy(dtype=float)
        info[tag] = {"rows": int(len(t)), "joined": {c: int(np.isfinite(out[c]).sum()) for c in have}}

    join(HALFLIFE, ["halflife_pc1"], "halflife")
    join(CODON, ["csc_endo_mean", "orf_len", "gc3"], "codon")
    if "orf_len" in out:
        out["log_orf_len"] = np.log10(out["orf_len"].clip(lower=1))
    utr = pd.read_parquet(UTR)
    utr["log_utr_len"] = np.log10(utr["utr_len"].astype(float).clip(lower=1))
    have = [c for c in UTR_FEATURES if c in utr.columns]
    by_id = utr.drop_duplicates("gene_id").set_index("gene_id")[have]
    by_sym = utr.drop_duplicates("symbol").set_index("symbol")[have]
    got = by_id.reindex(out["gene_id"]).reset_index(drop=True).fillna(by_sym.reindex(out["gene"]).reset_index(drop=True))
    for c in have:
        out[c] = got[c].to_numpy(dtype=float)
    info["utr"] = {c: int(np.isfinite(out[c]).sum()) for c in have}
    return out, info


def read(y: np.ndarray, x: np.ndarray, cov: np.ndarray, cov_sens: np.ndarray, n_perm: int) -> dict:
    r = correlate(y, x, covariates=cov, n_perm=n_perm, seed=0)
    s = correlate(y, x, covariates=cov_sens, n_perm=n_perm, seed=0)
    return {"n": r["n"], "rho": r["rho"], "p_perm": r["p_perm"],
            "rho_partial": r.get("rho_partial", float("nan")), "p_perm_partial": r.get("p_perm_partial", float("nan")),
            "rho_partial_beyond_pred": s.get("rho_partial", float("nan")),
            "p_perm_partial_beyond_pred": s.get("p_perm_partial", float("nan"))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logf = (OUT / "run.log").open("a")
    log = lambda m: _log(m, logf)                                   # noqa: E731
    log(f"--- t102_universal_truth.py n_perm={args.n_perm}")
    per = {f: fold_table(f, log) for f in FOLDS}
    for f, t in per.items():
        t.to_parquet(OUT / f"per_gene_{f}.parquet", index=False)
    genes = sorted(set.intersection(*[set(t["gene"]) for t in per.values()]))
    wide = pd.DataFrame({"gene": genes})
    cols = ["log_ctrl_cpm", "log_truth_se", "log_n_targets", *TRUTH_STATS, "log_mean_abs_pred", "log_mean_abs_pred_all"]
    for f, t in per.items():
        t = t.set_index("gene").reindex(genes)
        for c in cols:
            wide[f"{c}__{f}"] = t[c].to_numpy(dtype=float)
    for c in cols:
        wide[f"mean_{c}"] = np.column_stack([wide[f"{c}__{f}"] for f in FOLDS]).mean(axis=1)
    feat, feat_info = features(wide["gene"])
    wide = wide.merge(feat, on="gene", how="left")
    wide.to_parquet(OUT / "wide.parquet", index=False)
    feature_cols = [c for c in ("halflife_pc1", "csc_endo_mean", "log_orf_len", "gc3", *UTR_FEATURES) if c in wide.columns]
    res = {"folds": FOLDS, "genes_on_all_folds": len(genes), "features": feature_cols, "feature_join": feat_info,
           "n_perm": args.n_perm, "stats": {}}
    for stat, (twin, gated) in TRUTH_STATS.items():
        vals = np.column_stack([wide[f"{stat}__{f}"] for f in FOLDS])
        ok = np.isfinite(vals).all(axis=1)
        ranks = np.column_stack([pd.Series(vals[ok, j]).rank().to_numpy() for j in range(vals.shape[1])])
        pv = np.column_stack([wide[f"{twin}__{f}"] for f in FOLDS])
        entry = {"genes": int(ok.sum()), "icc1": icc1(vals[ok]), "icc1_ranks": icc1(ranks),
                 "icc1_pred_twin": icc1(pv[ok]),
                 "truth_vs_pred_twin_rho": float(pd.Series(wide.loc[ok, f"mean_{stat}"]).corr(
                     pd.Series(wide.loc[ok, f"mean_{twin}"]), method="spearman")),
                 "across_lines": {}, "per_line": {}}
        base = ["log_ctrl_cpm"] + (["log_truth_se", "log_n_targets"] if gated else [])
        # the scale the ICC is on, said out loud (results review, 2026-09-29): step 1 quoted 0.46
        # for the RAW mean |log2FC|; these statistics are logs; and how much of the gene part is
        # left once each fold's own expression, noise and scoring targets are held
        entry["icc1_scale"] = "log10" if stat.startswith("log_") else "raw"
        if stat.startswith("log_"):
            entry["icc1_raw_scale"] = icc1(np.power(10.0, vals[ok]))
        resid = []
        for j, f in enumerate(FOLDS):
            C = np.vstack([wide[f"{c}__{f}"].to_numpy(dtype=float)[ok] for c in base])
            good = np.isfinite(C).all(axis=0)
            r_ = np.full(int(ok.sum()), np.nan)
            r_[good] = residualize_ranks(vals[ok][good, j], C[:, good])
            resid.append(r_)
        entry["icc1_ranks_after_covariates"] = icc1(np.column_stack(resid))
        for scope in ["mean", *FOLDS]:
            ycol = f"mean_{stat}" if scope == "mean" else f"{stat}__{scope}"
            ccols = [f"mean_{c}" if scope == "mean" else f"{c}__{scope}" for c in base]
            tcol = f"mean_{twin}" if scope == "mean" else f"{twin}__{scope}"
            y = wide[ycol].to_numpy(dtype=float)
            cov = np.vstack([wide[c].to_numpy(dtype=float) for c in ccols])
            cov_s = np.vstack([cov, wide[tcol].to_numpy(dtype=float)])
            block = {}
            for c in feature_cols:
                block[c] = read(y, wide[c].to_numpy(dtype=float), cov, cov_s, args.n_perm if scope == "mean" else 500)
                if scope == "mean":
                    b = block[c]
                    log(f"   {stat:18s} mean x {c:14s} n {b['n']:5d} rho {b['rho']:+.3f} |cov {b['rho_partial']:+.3f} "
                        f"(p {b['p_perm_partial']:.4f}) |+pred {b['rho_partial_beyond_pred']:+.3f} (p {b['p_perm_partial_beyond_pred']:.4f})")
            (entry["across_lines"] if scope == "mean" else entry["per_line"]).update(
                block if scope == "mean" else {scope: block})
        log(f"   {stat}: {entry['genes']} genes on all folds, ICC(1) {entry['icc1']:.2f} (ranks {entry['icc1_ranks']:.2f}); "
            f"prediction twin ICC {entry['icc1_pred_twin']:.2f}; truth vs prediction size rho {entry['truth_vs_pred_twin_rho']:+.3f}")
        res["stats"][stat] = entry
    # line-specific contrasts
    contrasts = {}
    hl = pd.read_parquet(HALFLIFE) if HALFLIFE.exists() else None
    if hl is not None:
        line_cols = [c for c in hl.columns if c.startswith("halflife_") and c.endswith("_mean")]
        contrasts["table_s2_lines_vs_pc1"] = {
            c: float(hl["halflife_pc1"].corr(hl[c], method="spearman")) for c in line_cols}
    if DECAY.exists():
        dec = pd.read_parquet(DECAY)
        bridge = json.loads((PROBE / "bridge" / "symbol_to_ensg.json").read_text())
        for fold, sheet in MATCHED_DECAY.items():
            col = sheet if sheet in dec.columns else None
            if col is None:
                continue
            t = per[fold].copy()
            t["gene_id"] = t["gene"].map(bridge)
            t = t.merge(dec[["gene_id", col]].drop_duplicates("gene_id"), on="gene_id", how="left")
            # the consensus straight from its own table (every gene the fold has, not only genes
            # on all three folds), and every reading on the genes BOTH tables cover (results review,
            # 2026-09-29: side-by-side on two gene sets cannot support "adds nothing")
            if HALFLIFE.exists():
                h = pd.read_parquet(HALFLIFE)[["gene_id", "halflife_pc1"]].drop_duplicates("gene_id")
                t = t.merge(h, on="gene_id", how="left")
            both = np.isfinite(t[col].to_numpy(dtype=float)) & np.isfinite(t.get("halflife_pc1", pd.Series(np.nan, index=t.index)).to_numpy(dtype=float))
            t = t[both].reset_index(drop=True)
            blk = {"genes_with_both": int(len(t)),
                   "line_table_vs_pc1_spearman": float(t[col].corr(t["halflife_pc1"], method="spearman")) if len(t) else float("nan"),
                   "sign_note": "the 293T endogenous and K562 SLAM-seq columns rise with half-life (Spearman ~ +0.77 / +0.79 "
                                "with the consensus): they are stability-signed, whatever the sheet calls them"}
            for stat, (twin, gated) in TRUTH_STATS.items():
                base = ["log_ctrl_cpm"] + (["log_truth_se", "log_n_targets"] if gated else [])
                cov = np.vstack([t[c].to_numpy(dtype=float) for c in base])
                cov_s = np.vstack([cov, t[twin].to_numpy(dtype=float)])
                y = t[stat].to_numpy(dtype=float)
                line = t[col].to_numpy(dtype=float)
                pc1 = t["halflife_pc1"].to_numpy(dtype=float)
                blk[stat] = {"line_table": read(y, line, cov, cov_s, 2000),
                             "consensus": read(y, pc1, cov, cov_s, 2000),
                             # nested, both ways: what each table says once the other is held
                             "line_table_given_consensus": read(y, line, np.vstack([cov, pc1]), np.vstack([cov_s, pc1]), 2000),
                             "consensus_given_line_table": read(y, pc1, np.vstack([cov, line]), np.vstack([cov_s, line]), 2000)}
            contrasts[f"{fold}|{col}"] = blk
    res["line_specific"] = contrasts
    write_json(OUT / "universal_truth.json", res)
    (OUT / "UTRUTH_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
