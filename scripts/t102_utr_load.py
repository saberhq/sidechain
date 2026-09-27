#!/usr/bin/env python
"""T102 -- does a gene's 3'UTR regulatory load predict how badly its knockdown response
transfers across cell lines? Read on the LOCO fold rotation, in `nmae`'s own units.

    .venv/bin/python scripts/t102_utr_load.py --fold loco_k562gwps          # one fold
    .venv/bin/python scripts/t102_utr_load.py                                # every fold
    .venv/bin/python scripts/t102_utr_load.py --gate wald                    # the fallback gate

Per fold, from the shipped recipe (SER-7abefn: the `afn_e050_ab150_ac135` arm's own recorded
sources, no shrink, Poisson variance floor, alpha 1.35 on the per-cell channel):

  1. the pooled delta per target (`analytic_pds.pool_parts`, verified against
     `submit.build.pooled_delta`), on the fold's own gene axis;
  2. the log2FC the Wilcoxon members would read off the emitted cells, in closed form
     (`per_gene_transfer.pred_lfc`), against the real controls' mean CPM;
  3. the truth: the held-out line's own log2FC in cell-eval2's units, off one streaming
     pass over the fold's real h5ad (`per_gene_transfer.fold_truth`);
  4. the gate: cell-eval2's real-side Wilcoxon table where `real_de_gate.py` has produced
     it (`de_real/<fold>/run/de_real.parquet`), else the Wald stand-in, and every output
     says which;
  5. per gene, over its gated cells: `slope`, `ratio`, `nmae_g` (`per_gene_stats`);
  6. the 3'UTR features from `MiRNATargetSource.utr_load_table()` joined by Ensembl id
     through the corpora's own gene tables (`symbol_to_ensg`: our axes carry 2024 symbols,
     TargetScan 2018 ones), then by symbol, then through `gene_aliases.RETIRED_SYMBOLS`; and
     for each feature x statistic: Spearman rho raw, partial on log control CPM and log
     mean |truth|, each against a 2,000-shuffle gene-permutation null.

Writes `runs/probes/t102_utr_load/per_gene_<fold>.parquet` (one row per gene: the stats,
the features, the covariates) and `result.json` (per fold: the gate used, the coverage, the
correlation table, a quintile table of median slope by load within expression tertiles).
Nothing here scores a submission; if a feature tracks, the per-gene term is a separate step.
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

from sidechain.data.gene_aliases import RETIRED_SYMBOLS
from sidechain.eval.analytic_pds import LEGACY_MIN_LIBSIZE, delta_from_parts, pool_parts, prep_fold
from sidechain.eval.per_gene_transfer import (
    amplitude_on_axis,
    correlate,
    fit_binned_amplitude,
    fold_truth,
    gate_agreement,
    gate_from_de_real,
    gate_wald,
    nmae_closed_form,
    per_gene_stats,
    pred_lfc,
    write_json,
)
from sidechain.priors.posttx_mirna import MiRNATargetSource, spec_from_registry
from sidechain.submit.build import sources_from_specs

MIRRORS = Path("~/data/sidechain/runs/mirror").expanduser()
CACHE = Path("~/data/sidechain/cache/vcc2026").expanduser()
PROBE = Path("~/data/sidechain/runs/probes/t102_utr_load").expanduser()
ARM = "afn_e050_ab150_ac135"          # the shipped recipe, scored on every fold below
ALPHA = 1.35                           # the per-cell amplitude the Wilcoxon members read
FOLDS = {                              # fold dir -> (real h5ad stem, pert_col)
    "loco_k562gwps": ("loco_k562gwps", "gene"),
    "loco_hek293t_ch272": ("loco_hek293t_ch272", "perturbation"),
    "loco_k562gwps_union_ch272": ("loco_k562gwps_union_ch272", "gene"),
    "loco_k562gwps_union": ("loco_k562gwps_union", "gene"),
    "loco_hct116": ("loco_hct116", "perturbation"),
    "loco_hek293t": ("loco_hek293t", "perturbation"),
}
# The recipe's arm lives under the pdex mirror for the K562 challenge-panel fold.
ARM_DIR = {"loco_k562gwps": "loco_k562gwps_pdex"}
FEATURES = ["log_utr_len", "n_cons_sites", "sites_per_kb", "context_score", "n_families",
            "n_cons_8mer", "n_noncons_sites"]
STATS = ["slope", "log_ratio", "l1_scale", "nmae_g"]
MIN_TARGETS = 8


def _replay_localise():
    """`localise` from scripts/analytic_pds_replay.py -- one source of truth for box paths."""
    spec = importlib.util.spec_from_file_location(
        "analytic_pds_replay", Path(__file__).with_name("analytic_pds_replay.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.localise


def recorded_sources(fold: str) -> tuple[list[str], dict]:
    arm = MIRRORS / ARM_DIR.get(fold, fold) / ARM
    s = json.loads((arm / "summary.json").read_text())
    b = s["build"]
    if b.get("shrinkage") is not False or b.get("var_floor") != "poisson" or b.get("alpha") != ALPHA:
        raise SystemExit(f"{arm}: not the shipped recipe (shrinkage {b.get('shrinkage')}, "
                         f"var_floor {b.get('var_floor')}, alpha {b.get('alpha')})")
    localise = _replay_localise()
    specs = []
    for spec in s["sources"]["pseudobulk"]:
        got, note = localise(spec, fold)
        if got is None:
            raise SystemExit(f"{fold}: {note}")
        specs.append(got)
    return specs, {"arm": str(arm), "nmae_recorded": s["members"]["de_wilcoxon_lfc_nmae"],
                   "overall_recorded": s["overall"], "sources": specs}


def pooled_for(fold: str, targets: list[str], axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """[P, G] pooled delta and the coverage mask, cached per fold."""
    specs, info = recorded_sources(fold)
    # keyed on what changes the pooling, not on the fold name alone: a different source
    # list, floor or alpha must not read a stale delta back
    import hashlib
    key = hashlib.sha256(("|".join(specs) + "|poisson|noshrink").encode()).hexdigest()[:10]
    cache = PROBE / "pooled" / f"{fold}_{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["targets"]) == list(targets) and list(z["genes"]) == list(axis):
            return z["deltas"], z["covered"], info | {"pooled_from_cache": True}
    sources = sources_from_specs(specs, [])
    num, den = pool_parts(targets, sources, axis, var_floor="poisson", verify=15)
    deltas = delta_from_parts(num, den)
    covered = (den > 0).any(axis=1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, deltas=deltas, covered=covered,
                        targets=np.asarray(targets, dtype=object), genes=np.asarray(axis, dtype=object))
    return deltas, covered, info | {"pooled_from_cache": False}


_BRIDGE: dict[str, str] | None = None


def symbol_to_ensg() -> dict[str, str]:
    """Symbol -> Ensembl gene id, from the corpora's own gene tables, so a 2024 symbol on our
    axes meets TargetScan's 2018 symbol through the id both carry.

    Sources, first match wins: the X-Atlas gene map (the streamed corpus's own
    `metadata/gene_metadata.parquet`, inside its recorded gate, sha256-verified into
    `bridge/`), the Replogle K562 gwps h5ad's `var.ensembl_id`, the 2025 challenge h5ad's
    `var.gene_id`, and the Ensembl REST fallback the geometry gate built on 2026-09-14.
    Without this, 708 of the K562 axis's 8,248 symbols missed the UTR table -- INTS11 was
    CPSF3L, ELOA was TCEB3 -- and the 48-pair retired-symbol table bridges none of them.
    """
    global _BRIDGE
    if _BRIDGE is not None:
        return _BRIDGE
    import h5py
    from anndata.io import read_elem
    bridge: dict[str, str] = {}
    xa = PROBE / "bridge" / "xatlas_gene_metadata.parquet"
    if xa.exists():
        g = pd.read_parquet(xa)
        for sym, eid in zip(g["gene_name"].astype(str), g["ensembl_id"].astype(str)):
            bridge.setdefault(sym, eid)
    rep = Path("~/data/sidechain/external/zenodo-13350497/ReplogleWeissman2022_K562_gwps.h5ad").expanduser()
    if rep.exists():
        with h5py.File(rep) as f:
            v = read_elem(f["var"])
        for sym, eid in zip(v.index.astype(str), v["ensembl_id"].astype(str)):
            bridge.setdefault(sym, eid)
    y25 = Path("~/data/sidechain/vcc2025/adata_Training.h5ad").expanduser()
    if y25.exists():
        with h5py.File(y25) as f:
            v = read_elem(f["var"])
        for sym, eid in zip(v.index.astype(str), v["gene_id"].astype(str)):
            bridge.setdefault(sym, eid)
    fb = Path("~/data/sidechain/runs/geometry_gate/transcriptformer_20260914/ensembl_fallback.json").expanduser()
    if fb.exists():
        for sym, eid in json.loads(fb.read_text()).items():
            bridge.setdefault(str(sym), str(eid))
    _BRIDGE = {k: v.split(".")[0] for k, v in bridge.items() if v.startswith("ENSG")}
    return _BRIDGE


def join_features(stats: pd.DataFrame, utr: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join the UTR table by Ensembl id first (through `symbol_to_ensg`), then by symbol,
    then through the retired-symbol table in either direction."""
    by_id = utr.drop_duplicates("gene_id").set_index("gene_id")
    by_sym = utr.set_index("symbol")
    bridge = symbol_to_ensg()
    cur2old = {v: k for k, v in RETIRED_SYMBOLS.items()}
    rows, how = [], []
    for g in stats["gene"]:
        eid = bridge.get(g)
        if eid in by_id.index:
            rows.append(by_id.loc[eid]); how.append("ensembl_id")
        elif g in by_sym.index:
            rows.append(by_sym.loc[g]); how.append("symbol")
        elif RETIRED_SYMBOLS.get(g) in by_sym.index:
            rows.append(by_sym.loc[RETIRED_SYMBOLS[g]]); how.append("axis_old_to_ts_current")
        elif cur2old.get(g) in by_sym.index:
            rows.append(by_sym.loc[cur2old[g]]); how.append("axis_current_to_ts_old")
        else:
            rows.append(None); how.append("unmatched")
    keep = [i for i, r in enumerate(rows) if r is not None]
    feat = pd.DataFrame([rows[i].drop(labels=[c for c in ("symbol", "gene_id") if c in rows[i].index])
                         for i in keep])
    feat.index = keep
    out = stats.iloc[keep].reset_index(drop=True).copy()
    feat = feat.reset_index(drop=True)
    out = pd.concat([out, feat], axis=1)
    out["match"] = [how[i] for i in keep]
    out["log_utr_len"] = np.log10(out["utr_len"].astype(float).clip(lower=1))
    for c in FEATURES:
        if c in out.columns:
            out[c] = out[c].astype(float)
    counts = pd.Series(how).value_counts().to_dict()
    unmatched = [g for g, h in zip(stats["gene"], how) if h == "unmatched"]
    return out, {"genes_with_stats": int(len(stats)), "matched": int(len(out)), **counts,
                 "unmatched_sample": unmatched[:40]}


def quintile_table(df: pd.DataFrame, feature: str, stat: str = "slope") -> list[dict]:
    """Median `stat` by feature quintile, within control-CPM tertiles: the picture behind rho."""
    d = df[np.isfinite(df[feature]) & np.isfinite(df[stat])].copy()
    if len(d) < 50:
        return []
    d["expr_bin"] = pd.qcut(d["log_ctrl_cpm"], 3, labels=["low", "mid", "high"])
    d["feat_bin"] = pd.qcut(d[feature].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rows = []
    for (e, f), g in d.groupby(["expr_bin", "feat_bin"], observed=True):
        rows.append({"expr_tertile": str(e), "feature_quintile": int(f), "n": int(len(g)),
                     "median_stat": float(g[stat].median()),
                     "median_feature": float(g[feature].median())})
    return rows


def load_fold(fold: str, gate_mode: str, log) -> dict:
    """Everything one fold needs, from the caches: truth, pooled delta, profile, gate."""
    t0 = time.time()
    stem, pert_col = FOLDS[fold]
    real = CACHE / f"{stem}_real.h5ad"
    fc = prep_fold(real, pert_col=pert_col, min_libsize=LEGACY_MIN_LIBSIZE,
                   cache=MIRRORS / fold / "analytic_fold_cache.npz")
    truth = fold_truth(real, pert_col=pert_col, cache=PROBE / "truth" / f"{fold}_truth_pseudobulk.npz")
    targets = list(truth.targets)
    axis = truth.genes
    if list(fc.genes) != list(axis):
        raise SystemExit(f"{fold}: fold cache genes differ from the truth's axis")
    deltas, covered, pool_info = pooled_for(fold, targets, axis)
    log(f"   pooled: {int(covered.sum())}/{len(targets)} targets covered "
        f"({'cache' if pool_info['pooled_from_cache'] else 'fresh'}), {time.time() - t0:.0f}s")
    de_real = PROBE / "de_real" / fold / "run" / "de_real.parquet"
    wald, wald_info = gate_wald(truth)
    wald &= truth.universe[None, :]
    if gate_mode == "wilcoxon" or (gate_mode == "auto" and de_real.exists()):
        if not de_real.exists():
            raise SystemExit(f"{fold}: no {de_real}; run real_de_gate.py first or pass --gate wald")
        gate, gate_info = gate_from_de_real(de_real, targets, axis)
        gate &= truth.universe[None, :]
        # the stand-in's own check, on every fold that has the real gate
        gate_info["wald_agreement"] = gate_agreement(gate, wald)
        log(f"   wald stand-in vs wilcoxon: jaccard {gate_info['wald_agreement']['jaccard']:.3f} "
            f"({gate_info['wald_agreement']['cells_b']:,} vs {gate_info['wald_agreement']['cells_a']:,} cells)")
    else:
        gate, gate_info = wald, wald_info
    y = truth.truth_lfc()
    # cells whose target arm is all zeros read -30 log2 under eps 1e-9; counted, and per-gene
    # readings can be checked against them (none on the Wilcoxon gate of loco_k562gwps)
    gate_info["gate_cells_target_mean_zero"] = int((gate & (truth.mean_cpm == 0)).sum())
    gate_info["gate_cells_abs_truth_gt_10"] = int((gate & (np.abs(y) > 10)).sum())
    log(f"   gate: {gate_info['kind']}, {gate_info['gate_cells']:,} cells, "
        f"{gate_info['targets_scored']} targets scored, median gate {gate_info['median_gate']:.0f}, "
        f"target-mean-zero cells {gate_info['gate_cells_target_mean_zero']}")
    return {"fold": fold, "real": real, "pert_col": pert_col, "fc": fc, "truth": truth,
            "targets": targets, "axis": axis, "deltas": deltas, "covered": covered,
            "pool_info": pool_info, "gate": gate, "gate_info": gate_info}


def run_fold(fold: str, gate_mode: str, n_perm: int, utr: pd.DataFrame, log) -> dict:
    t0 = time.time()
    log(f"== {fold}")
    L = load_fold(fold, gate_mode, log)
    real, pert_col, fc, truth = L["real"], L["pert_col"], L["fc"], L["truth"]
    targets, axis, deltas, covered = L["targets"], L["axis"], L["deltas"], L["covered"]
    pool_info, gate, gate_info = L["pool_info"], L["gate"], L["gate_info"]
    pred = pred_lfc(deltas, targets, axis, fc.frac, truth.ctrl_mean_cpm, alpha=ALPHA, covered=covered)
    y = truth.truth_lfc()
    universe = truth.universe

    # the fold-level nmae of the closed-form prediction, as a sanity anchor against the
    # recorded arm (the emitted arm carries noise and the bulk channel; this is the intent)
    scored = gate.sum(axis=1) >= 10
    per_t = []
    for i in np.flatnonzero(scored):
        g = gate[i]
        per_t.append(np.abs(pred[i, g] - y[i, g]).mean() / np.abs(y[i, g]).mean())
    nmae_closed_form = float(np.mean(per_t)) if per_t else float("nan")
    log(f"   raw nmae (closed-form pred, this gate): {nmae_closed_form:.4f} over {len(per_t)} targets; "
        f"recorded arm member {pool_info['nmae_recorded']:.4f} (scaled)")

    stats = per_gene_stats(pred, y, gate, axis, min_targets=MIN_TARGETS, truth=truth)
    pos = {g: i for i, g in enumerate(axis)}
    idx = np.array([pos[g] for g in stats["gene"]])
    stats["log_ctrl_cpm"] = np.log10(truth.ctrl_mean_cpm[idx])
    stats["log_mean_abs_truth"] = np.log10(stats["mean_abs_truth"].clip(lower=1e-6))
    stats["log_truth_se"] = np.log10(stats["truth_se"].clip(lower=1e-6))
    stats["log_ratio"] = np.log2(stats["ratio"].clip(lower=1e-6))
    df, match = join_features(stats, utr)
    log(f"   genes with >= {MIN_TARGETS} gated targets: {len(stats):,}; with a UTR row: {len(df):,} "
        f"({match})")
    df.to_parquet(PROBE / f"per_gene_{fold}.parquet", index=False)

    # Two partials. `expr`: the held-out line's control CPM alone. `expr+se`: plus the
    # truth's own measurement noise (median delta-method SE of its log2FC over the gene's
    # gated cells) -- what a feature that tracks expression could be standing in for.
    # `mean_abs_truth` is NOT partialled: it is a descendant of the outcome and sits in
    # nmae_g's denominator, so conditioning on it can make or erase a link (critic pass).
    cov_expr = np.vstack([df["log_ctrl_cpm"].to_numpy()])
    cov_both = np.vstack([df["log_ctrl_cpm"].to_numpy(), df["log_truth_se"].to_numpy()])
    table = []
    for feat in FEATURES:
        for stat in STATS:
            r1 = correlate(df[stat].to_numpy(), df[feat].to_numpy(), covariates=cov_expr,
                           n_perm=n_perm, seed=0)
            r2 = correlate(df[stat].to_numpy(), df[feat].to_numpy(), covariates=cov_both,
                           n_perm=n_perm, seed=0)
            row = {"feature": feat, "stat": stat, "n": r1["n"], "rho": r1["rho"], "p_perm": r1["p_perm"],
                   "rho_partial_expr": r1["rho_partial"], "p_perm_partial_expr": r1["p_perm_partial"],
                   "rho_partial_expr_se": r2["rho_partial"], "p_perm_partial_expr_se": r2["p_perm_partial"]}
            table.append(row)
            log(f"   {feat:16s} x {stat:9s}: rho {r1['rho']:+.3f} (p {r1['p_perm']:.3f})  "
                f"|expr {r1['rho_partial']:+.3f} (p {r1['p_perm_partial']:.3f})  "
                f"|expr+se {r2['rho_partial']:+.3f} (p {r2['p_perm_partial']:.3f})  n {r1['n']}")
    # the confounders themselves, so the reader sees what was partialled and why
    confound = {c: correlate(df["slope"].to_numpy(), df[c].to_numpy(), n_perm=200, seed=0)
                for c in ("log_ctrl_cpm", "log_truth_se", "log_mean_abs_truth")}
    feat_vs_expr = {f: float(pd.Series(df[f]).corr(df["log_ctrl_cpm"], method="spearman"))
                    for f in FEATURES}
    summary = {
        "fold": fold, "real": str(real), "pert_col": pert_col, "arm": pool_info["arm"],
        "sources": pool_info["sources"], "alpha": ALPHA,
        "targets": len(targets), "targets_covered": int(covered.sum()),
        "genes_axis": int(len(axis)), "genes_universe": int(universe.sum()),
        "gate": gate_info, "nmae_closed_form_raw": nmae_closed_form,
        "nmae_recorded_scaled": pool_info["nmae_recorded"],
        "genes_with_stats": int(len(stats)), "match": match,
        "slope_median": float(df["slope"].median()), "slope_iqr": [float(df["slope"].quantile(q)) for q in (0.25, 0.75)],
        "ratio_median": float(df["ratio"].median()), "l1_scale_median": float(df["l1_scale"].median()),
        "nmae_g_median": float(df["nmae_g"].median()), "sign_agree_median": float(df["sign_agree"].median()),
        "confounders_vs_slope": confound, "feature_vs_log_ctrl_cpm_spearman": feat_vs_expr,
        "correlations": table,
        "quintiles_slope_by_n_cons_sites": quintile_table(df, "n_cons_sites", "slope"),
        "quintiles_slope_by_log_utr_len": quintile_table(df, "log_utr_len", "slope"),
        "quintiles_nmae_by_n_cons_sites": quintile_table(df, "n_cons_sites", "nmae_g"),
        "seconds": round(time.time() - t0),
    }
    return summary


HELD_OUT_LINE = {"loco_k562gwps": "K562", "loco_k562gwps_union": "K562",
                 "loco_k562gwps_union_ch272": "K562", "loco_hek293t": "HEK293T",
                 "loco_hek293t_ch272": "HEK293T", "loco_hct116": "HCT116"}
FIT_FOLDS = ["loco_hct116", "loco_hek293t", "loco_k562gwps_union"]     # the full-panel folds


def run_amplitude(fit_folds: list[str], eval_folds: list[str], features: list[str],
                  gate_mode: str, utr: pd.DataFrame, n_shuffle: int, log,
                  fit_stat: str = "l1_scale", floor: float = 0.01) -> dict:
    """What a per-gene amplitude read off a 3'UTR feature would buy on `nmae`, cross-line.

    For every (fit fold, eval fold) pair whose held-out LINES differ: fit quintile factors on
    the fit fold's per-gene size statistic (`fit_binned_amplitude`), map them onto the eval
    fold's axis, scale the pooled delta gene-wise BEFORE alpha and the emitter, and read the
    raw closed-form nmae on the eval fold's gate against the unscaled baseline. Beside it:
    the same factors shuffled across genes (`n_shuffle` draws -- does the assignment matter,
    or only the spread?) and the in-sample oracle (factors fitted on the eval fold itself),
    which bounds what any quintile term of that feature could reach. Every number is a raw
    nmae, lower is better, on the closed-form prediction; nothing here is a scored run.

    `fit_stat` is `ratio` by default -- the sign-blind size ratio sum|p| / sum|y| -- not
    `slope`. The least-squares slope through the origin is dominated by sign disagreements
    and reads 0.01-0.02 at the median on the full-panel folds, so every quintile fell under
    the floor and the factors came out all ones (first run, 2026-09-26). An amplitude is a
    rescale of size; the ratio is the size.
    """
    out = {"pairs": [], "features": features, "fit_folds": fit_folds, "eval_folds": eval_folds,
           "fit_stat": fit_stat, "floor": floor}
    loaded: dict[str, dict] = {}
    per_gene: dict[str, pd.DataFrame] = {}
    for f in sorted(set(fit_folds) | set(eval_folds)):
        pq = PROBE / f"per_gene_{f}.parquet"
        if not pq.exists():
            log(f"   no {pq.name}: run the fold first"); continue
        per_gene[f] = pd.read_parquet(pq)
    rng = np.random.default_rng(0)
    for ev in eval_folds:
        if ev not in per_gene:
            continue
        log(f"== amplitude on {ev}")
        L = loaded.get(ev) or load_fold(ev, gate_mode, log)
        loaded[ev] = L
        fc, truth, targets, axis = L["fc"], L["truth"], L["targets"], L["axis"]
        deltas, covered, gate = L["deltas"], L["covered"], L["gate"]
        y = truth.truth_lfc()
        base_pred = pred_lfc(deltas, targets, axis, fc.frac, truth.ctrl_mean_cpm, alpha=ALPHA, covered=covered)
        base, base_per_t = nmae_closed_form(base_pred, y, gate)
        # the feature on this fold's axis, through the SAME join the correlations use
        # (Ensembl id, then symbol, then the retired-symbol table)
        joined, _ = join_features(pd.DataFrame({"gene": axis}), utr)
        pos = {g: i for i, g in enumerate(axis)}
        feat_axis = {}
        for feat in features:
            vals = np.full(len(axis), np.nan)
            vals[[pos[g] for g in joined["gene"]]] = joined[feat].to_numpy()
            feat_axis[feat] = vals
        for fit in fit_folds:
            if fit not in per_gene or HELD_OUT_LINE[fit] == HELD_OUT_LINE[ev]:
                continue
            for feat in features:
                df = per_gene[fit]
                try:
                    fitted = fit_binned_amplitude(df[feat].to_numpy(), df[fit_stat].to_numpy(), n_bins=5,
                                                  floor=floor)
                except ValueError as exc:
                    out["pairs"].append({"fit_fold": fit, "eval_fold": ev, "feature": feat,
                                         "refused": str(exc)})
                    log(f"   fit {fit:26s} feat {feat:14s}: REFUSED ({exc})")
                    continue
                a = amplitude_on_axis(feat_axis[feat], fitted)
                pred = pred_lfc(deltas * a[None, :], targets, axis, fc.frac, truth.ctrl_mean_cpm,
                                alpha=ALPHA, covered=covered)
                with_term, term_per_t = nmae_closed_form(pred, y, gate)
                # the paired per-target difference is the honest bar on a delta: the
                # targets are the sampling unit, and a knob shifts every target together
                diff = term_per_t - base_per_t
                bar = float(2.0 * diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
                shuf = []
                for _ in range(n_shuffle):
                    a_s = a.copy()
                    has = np.isfinite(feat_axis[feat])
                    a_s[has] = rng.permutation(a[has])
                    p_s = pred_lfc(deltas * a_s[None, :], targets, axis, fc.frac, truth.ctrl_mean_cpm,
                                   alpha=ALPHA, covered=covered)
                    shuf.append(nmae_closed_form(p_s, y, gate)[0])
                dfe = per_gene[ev]
                try:
                    oracle_fit = fit_binned_amplitude(dfe[feat].to_numpy(), dfe[fit_stat].to_numpy(),
                                                      n_bins=5, floor=floor)
                    a_o = amplitude_on_axis(feat_axis[feat], oracle_fit)
                    p_o = pred_lfc(deltas * a_o[None, :], targets, axis, fc.frac, truth.ctrl_mean_cpm,
                                   alpha=ALPHA, covered=covered)
                    oracle, _ = nmae_closed_form(p_o, y, gate)
                except ValueError:
                    oracle = float("nan")
                row = {"fit_fold": fit, "eval_fold": ev, "feature": feat,
                       "gate": L["gate_info"]["kind"], "targets_scored": L["gate_info"]["targets_scored"],
                       "nmae_base": base, "nmae_with_term": with_term, "delta": with_term - base,
                       "delta_2se_paired": bar, "targets_paired": int(len(diff)),
                       "shuffle_mean": float(np.mean(shuf)), "shuffle_sd": float(np.std(shuf)),
                       "delta_vs_shuffle": with_term - float(np.mean(shuf)),
                       "nmae_oracle_in_sample": oracle, "delta_oracle": oracle - base,
                       "factors": fitted["factors"], "bin_median_stat": fitted["bin_median_slope"],
                       "fit_stat": fit_stat,
                       "genes_with_factor": int(np.isfinite(feat_axis[feat]).sum())}
                out["pairs"].append(row)
                log(f"   fit {fit:26s} feat {feat:14s}: base {base:.4f} -> {with_term:.4f} "
                    f"(d {with_term - base:+.4f} 2se {bar:.4f}; shuffle {np.mean(shuf):.4f} +/- {np.std(shuf):.4f}; "
                    f"oracle {oracle:.4f}, d {oracle - base:+.4f}) factors "
                    f"{[round(x, 2) for x in fitted['factors']]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", action="append", choices=list(FOLDS), help="restrict to these folds")
    ap.add_argument("--gate", choices=["auto", "wilcoxon", "wald"], default="auto")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--amplitude", action="store_true",
                    help="skip the correlations; run the cross-line per-gene amplitude test on "
                         "the folds already analysed (needs their per_gene parquets)")
    ap.add_argument("--amp-features", nargs="+", default=["n_cons_sites", "log_utr_len"])
    ap.add_argument("--n-shuffle", type=int, default=20)
    ap.add_argument("--amp-stat", choices=["l1_scale", "ratio", "slope"], default="l1_scale",
                    help="the per-gene size statistic the bin factors are read from")
    ap.add_argument("--fit-folds", nargs="+", choices=list(FOLDS), default=FIT_FOLDS,
                    help="folds whose per-gene tables the factors are fitted on (pairs whose "
                         "held-out line equals the eval fold's are skipped)")
    ap.add_argument("--amp-tag", default="", help="suffix for the amplitude output file")
    args = ap.parse_args()
    PROBE.mkdir(parents=True, exist_ok=True)
    logf = (PROBE / "run.log").open("a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    log(f"--- t102_utr_load.py {time.strftime('%Y-%m-%dT%H:%M:%S')} gate={args.gate} n_perm={args.n_perm}")
    src = MiRNATargetSource(spec_from_registry("targetscan"), {})
    utr = src.utr_load_table()
    log(f"UTR table: {len(utr):,} genes ({src.derived / 'utr_load.parquet'})")
    if args.amplitude:
        res = run_amplitude(args.fit_folds, args.fold or list(FOLDS), args.amp_features, args.gate,
                            utr, args.n_shuffle, log, fit_stat=args.amp_stat)
        write_json(PROBE / f"amplitude_{args.gate}_{args.amp_stat}{args.amp_tag}.json", res)
        log(f"wrote {PROBE / 'amplitude.json'}")
        return
    # One JSON per fold, then the assembled view: two runs on different folds can overlap
    # without one's write dropping the other's entry.
    for fold in (args.fold or list(FOLDS)):
        try:
            summary = run_fold(fold, args.gate, args.n_perm, utr, log)
        except SystemExit as exc:
            log(f"   SKIPPED {fold}: {exc}")
            summary = {"fold": fold, "skipped": str(exc)}
        write_json(PROBE / f"result_{fold}.json", summary)
    result = {"folds": {f: json.loads((PROBE / f"result_{f}.json").read_text())
                        for f in FOLDS if (PROBE / f"result_{f}.json").exists()}}
    write_json(PROBE / "result.json", result)
    log(f"wrote {PROBE / 'result.json'}")


if __name__ == "__main__":
    sys.exit(main())
