#!/usr/bin/env python
"""T102 -- a direction prior for the challenge's own miRNA-pathway targets (AGO1, TARBP2), SIZED
before anything is built.

    .venv/bin/python scripts/t102_mirna_targets.py
    .venv/bin/python scripts/t102_mirna_targets.py --n-shuffle 100 --fold loco_hct116

Step 1's one mechanism that held across lines: TargetScan's conserved site count tracks the
log2FC after a miRNA-pathway knockdown (DICER1, DROSHA, DGCR8) beyond 98-100 % of reference
knockdowns in the same line, in every line where the knockdown bites. Two of the challenge's 300
targets are miRNA machinery -- AGO1 (an Argonaute) and TARBP2 (TRBP, Dicer's partner) -- and all
three full loco folds hold both. Four more are decay or ARE-binding factors (DCP2, PAN2, KHSRP,
RC3H2) and are read beside them with the same instrument, as a reference, not as candidates.

The question, per fold (HCT116, HEK293T, K562 held out), per target:

1. **How much of the TRUE response is site-driven de-repression?** Spearman of the held-out
   line's own log2FC (cell-eval2's units, the DE universe, own gene out) with the site count,
   partial on control expression and 3'UTR length; ranked against every other knockdown of the
   same fold, overall and among the 50 nearest in strength (gate size, the metric's own DE count).
2. **How much of it does the pooled delta already carry?** The same partial for the shipped
   recipe's pooled prediction (the recorded pool, alpha 1.35, in closed form), and for the
   residual truth - prediction.
3. **What would a term buy?** delta' = delta + beta * s, with s the site count's rank-normal score
   centred on the expressed genes (so the term redistributes, never shifts), beta on a grid in
   units of the pooled delta's own sd, BEFORE alpha. Read on raw `pds_cosine` (cell-eval2's own
   `discrimination_score`, the target's row against the fold's full retrieval pool, the
   `--dispersion even` analytic path) and raw nmae on the target's exact gate (closed form; read
   only where the gate holds >= 100 genes). Four readings: the MECHANISTIC oracle (beta >= 0 only:
   more conserved sites -> predicted up, the de-repression step 1 found) and the free oracle (any
   sign), both hindsight ceilings on the held-out truth itself; each against the same oracle on a
   gene-shuffled s (what ANY vector of that shape buys in-sample -- the control that decides); and
   the TRANSFER, a multiplier learned from the OTHER two folds (their median, and one pooled fit on
   their mean gain curve) applied here -- what a pipeline could actually ship.

The panel-level ceiling is the per-target gain divided by 300 (two targets of 300). Everything
lands under ``runs/probes/t102_utr_load/mirna_targets/``. Nothing here is a knob.
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
from scipy import stats as sps

from sidechain.eval.analytic_pds import emitted_sums, pds_cosine  # noqa: F401  (pds_cosine: the settings)
from sidechain.eval.analytic_pds import BULK_TARGET_SUM, CONTROL, KNOCKDOWN_LOG2FC, PERT_COL
from sidechain.eval.analytic_pds import LEGACY_MIN_LIBSIZE, prep_fold
from sidechain.eval.per_gene_transfer import CE2_MIN_GATE, pred_lfc, residualize_ranks, write_json
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
OUT = PROBE / "mirna_targets"
DERIVED = DATA / "derived/targetscan-vert_80"
ALPHA = close.ALPHA
FOLDS = list(close.FOLDS)
NAMED = ["AGO1", "TARBP2"]
REFERENCE_NAMED = ["DCP2", "PAN2", "KHSRP", "RC3H2"]
FEATURES = ["n_cons_sites", "n_cons_8mer", "context_score", "log_utr_len", "m_are_auuua", "au_content"]
MIN_CELLS = 20
N_NEIGHBOURS = 50
#: the term's size in units of the pooled delta's own sd over the eligible genes (pre-launch review:
#: a fixed log2 grid stepped past the whole delta at its first step); 0 is on the grid, so a gain is >= 0
MULTS = np.array(sorted({s * m for m in (0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5) for s in (1, -1)}))
MIN_GATE_NMAE = 100    # below this the nmae leg is a fit on a handful of genes, so it is not read


def _log(msg: str, fh) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def utr_on_axis(axis: np.ndarray) -> pd.DataFrame:
    """The widened UTR table on the fold axis: Ensembl id through the bridge first, then symbol."""
    bridge = json.loads((PROBE / "bridge" / "symbol_to_ensg.json").read_text())
    ft = pd.read_parquet(DERIVED / "utr_features.parquet")
    ft["log_utr_len"] = np.log10(ft["utr_len"].astype(float).clip(lower=1))
    by_id = ft.drop_duplicates("gene_id").set_index("gene_id")
    by_sym = ft.drop_duplicates("symbol").set_index("symbol")
    cols = [c for c in FEATURES if c in ft.columns]
    out = np.full((len(axis), len(cols)), np.nan)
    for i, g in enumerate(axis):
        e = bridge.get(str(g))
        row = by_id.loc[e] if e in by_id.index else (by_sym.loc[g] if g in by_sym.index else None)
        if row is not None:
            out[i] = row[cols].to_numpy(dtype=float)
    return pd.DataFrame(out, columns=cols)


def partial(y: np.ndarray, x: np.ndarray, cov: np.ndarray, ok: np.ndarray) -> float:
    ok = ok & np.isfinite(y) & np.isfinite(x) & np.isfinite(cov).all(axis=0)
    if ok.sum() < 50:
        return float("nan")
    return float(np.corrcoef(residualize_ranks(x[ok], cov[:, ok]), residualize_ranks(y[ok], cov[:, ok]))[0, 1])


def rank_normal(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Rank-normal score of x on `ok`, mean 0 / sd 1 there, 0 elsewhere."""
    s = np.zeros(len(x))
    v = x[ok]
    r = sps.rankdata(v) / (len(v) + 1)
    z = sps.norm.ppf(r)
    s[ok] = (z - z.mean()) / z.std()
    return s


class FoldPds:
    """The pieces the analytic pds of ONE target row needs, fixed per fold."""

    def __init__(self, fold: str, targets: list[str]):
        stem, pert_col, _, _ = close.FOLDS[fold]
        fc = prep_fold(close.CACHE / f"{stem}_real.h5ad", pert_col=pert_col, min_libsize=LEGACY_MIN_LIBSIZE,
                       cache=close.MIRRORS / fold / "analytic_fold_cache.npz")
        self.fc = fc
        self.cells = dict(zip(targets, fc.cells_for(targets)))
        self.gpos = {g: i for i, g in enumerate(fc.genes)}

    def pds(self, target: str, delta_row: np.ndarray) -> float:
        from cell_eval2.metrics.discrimination import discrimination_score
        from cell_eval2.prep import bulk_lognorm_means
        d = np.asarray(delta_row, dtype=np.float64) * ALPHA
        j = self.gpos.get(target)
        if j is not None:
            d = d.copy()
            d[j] = KNOCKDOWN_LOG2FC
        sums = emitted_sums(d[None, :], self.fc.frac, self.fc.lib_median, [self.cells[target]])
        pred_means = bulk_lognorm_means(np.asarray(sums, dtype=np.float64), BULK_TARGET_SUM)
        out = discrimination_score(
            pred_bulk=(np.asarray([target], dtype=str), pred_means),
            real_bulk=(np.asarray(self.fc.perts, dtype=str), self.fc.real_means),
            pert_col=PERT_COL, control=CONTROL, distance="cosine", rank_denominator="n-1",
            tie_policy="midrank", exclude_target_gene=True, exclusion_scope="panel",
            control_source="real", genes=np.asarray(self.fc.genes, dtype=str))
        return float(out[target])


def nmae_row(p: np.ndarray, y: np.ndarray, gate: np.ndarray) -> float:
    if gate.sum() < CE2_MIN_GATE:
        return float("nan")
    return float(np.abs(p[gate] - y[gate]).mean() / np.abs(y[gate]).mean())


def run_fold(fold: str, n_shuffle: int, log) -> dict:
    t0 = time.time()
    F = close.Fold(fold, log, keep_halves=False)
    targets = list(F.full.targets)
    srcs = sources_from_specs(close._recorded_specs(fold), [])
    delta, covered = close.pool_delta(targets, srcs, F.axis)
    del srcs
    P = pred_lfc(delta, targets, F.axis, F.frac, F.full.ctrl_mean_cpm, alpha=ALPHA, covered=covered)
    utr = utr_on_axis(F.axis)
    expressed = F.full.universe & np.isfinite(utr["n_cons_sites"].to_numpy())
    logc = np.log10(np.clip(F.full.ctrl_mean_cpm, 1e-3, None))
    cov_e = logc[None, :]
    cov_el = np.vstack([logc, utr["log_utr_len"].to_numpy()])
    sites = utr["n_cons_sites"].to_numpy()
    pds_tool = FoldPds(fold, targets)
    log(f"   {fold}: pooled {int(covered.sum())}/{len(targets)} covered; {int(expressed.sum())} expressed genes "
        f"with a UTR row ({time.time() - t0:.0f}s)")

    gate_size = F.gate.sum(axis=1)
    ncells = F.full.n_cells
    rows = []
    for i, t in enumerate(targets):
        if ncells[i] < MIN_CELLS:
            continue
        own = np.zeros(len(F.axis), dtype=bool)
        if t in F.gpos:
            own[F.gpos[t]] = True
        ok = expressed & ~own
        y, p = F.y[i], P[i]
        rows.append({"target": t, "cells": int(ncells[i]), "gate": int(gate_size[i]), "covered": bool(covered[i]),
                     "truth_sites": partial(y, sites, cov_e, ok), "truth_sites_len": partial(y, sites, cov_el, ok),
                     "pred_sites_len": partial(p, sites, cov_el, ok) if covered[i] else float("nan"),
                     "resid_sites_len": partial(y - p, sites, cov_el, ok) if covered[i] else float("nan")})
    ref = pd.DataFrame(rows)
    ref.to_parquet(OUT / f"reference_{fold}.parquet", index=False)

    def pct(t, col):
        row = ref[ref["target"] == t]
        if not len(row) or not np.isfinite(row[col].iloc[0]):
            return float("nan"), float("nan")
        v = float(row[col].iloc[0])
        others = ref[(ref["target"] != t) & ~ref["target"].isin(NAMED + REFERENCE_NAMED)]
        others = others[np.isfinite(others[col])]
        allp = float((others[col] < v).mean())
        lg = np.log10(np.maximum(others["gate"].to_numpy(), 1))
        near = others.iloc[np.argsort(np.abs(lg - np.log10(max(int(row["gate"].iloc[0]), 1))))[:N_NEIGHBOURS]]
        return allp, float((near[col] < v).mean())

    out = {"fold": fold, "targets": len(targets), "reference_knockdowns": int(len(ref)),
           "reference_median": {c: float(ref[c].median()) for c in ("truth_sites", "truth_sites_len",
                                                                      "pred_sites_len", "resid_sites_len")},
           "per_target": {}}
    rng = np.random.default_rng(0)
    for t in NAMED + REFERENCE_NAMED:
        if t not in F.tpos:
            continue
        i = F.tpos[t]
        own = np.zeros(len(F.axis), dtype=bool)
        if t in F.gpos:
            own[F.gpos[t]] = True
        ok = expressed & ~own
        y, p = F.y[i], P[i]
        rec = {"cells": int(ncells[i]), "gate": int(gate_size[i]), "scored_by_nmae": bool(gate_size[i] >= CE2_MIN_GATE),
               "covered": bool(covered[i])}
        for f in FEATURES:
            x = utr[f].to_numpy() if f in utr else None
            if x is None:
                continue
            rec[f"truth_{f}"] = partial(y, x, cov_e if f == "log_utr_len" else cov_el, ok)
            rec[f"pred_{f}"] = partial(p, x, cov_e if f == "log_utr_len" else cov_el, ok)
            rec[f"resid_{f}"] = partial(y - p, x, cov_e if f == "log_utr_len" else cov_el, ok)
        for col in ("truth_sites_len", "pred_sites_len", "resid_sites_len"):
            rec[f"pct_{col}"], rec[f"pct_{col}_strength_matched"] = pct(t, col)
        # the Figure 4 shape, centred on the expressed median: genes with sites vs without
        yc = y - np.nanmedian(y[ok])
        pc = p - np.nanmedian(p[ok])
        rec["truth_median_with_sites"] = float(np.nanmedian(yc[ok & (sites > 0)]))
        rec["truth_median_no_sites"] = float(np.nanmedian(yc[ok & (sites == 0)]))
        rec["pred_median_with_sites"] = float(np.nanmedian(pc[ok & (sites > 0)]))
        rec["pred_median_no_sites"] = float(np.nanmedian(pc[ok & (sites == 0)]))
        if t in NAMED:
            if not covered[i]:
                raise SystemExit(f"{fold}: {t} is not covered by the recorded pool; the pin conventions would differ")
            s = rank_normal(np.log1p(np.nan_to_num(sites, nan=0.0)), ok)
            g = F.gate[i]
            sd = float(np.std(delta[i][ok]))
            betas = MULTS * sd                 # the grid in units of the pooled delta's own spread
            base_pds = pds_tool.pds(t, delta[i])
            base_nmae = nmae_row(p, y, g)
            nmae_ok = int(g.sum()) >= MIN_GATE_NMAE

            def curve(sv):
                pd_, nm_ = [], []
                for b in betas:
                    d2 = delta[i] + b * sv
                    pd_.append(pds_tool.pds(t, d2))
                    if nmae_ok:
                        p2 = pred_lfc(d2[None, :], [t], F.axis, F.frac, F.full.ctrl_mean_cpm, alpha=ALPHA)[0]
                        nm_.append(nmae_row(p2, y, g))
                    else:
                        nm_.append(np.nan)
                return np.array(pd_), np.array(nm_)

            def best(values, maximise, positive_only):
                """argbest over the grid, ties to the smallest |beta|; positive_only = the
                mechanistic sign (more conserved sites -> predicted UP, de-repression)."""
                v = np.where(MULTS >= 0, values, np.nan) if positive_only else values.copy()
                if not np.isfinite(v).any():
                    return None
                target = np.nanmax(v) if maximise else np.nanmin(v)
                cand = np.flatnonzero(np.isclose(v, target, rtol=0, atol=1e-12))
                return int(cand[np.argmin(np.abs(MULTS[cand]))])

            pds_c, nmae_c = curve(s)
            rec |= {"base_pds": base_pds, "base_nmae": base_nmae, "delta_sd": sd,
                    "nmae_gate": int(g.sum()), "nmae_read": nmae_ok,
                    "curve": {"mult": MULTS.tolist(), "beta": betas.tolist(), "pds": pds_c.tolist(), "nmae": nmae_c.tolist()}}
            for tag, pos in (("mechanistic", True), ("free", False)):
                jb = best(pds_c, True, pos)
                rec[f"oracle_pds_{tag}"] = {"mult": float(MULTS[jb]), "beta": float(betas[jb]), "pds": float(pds_c[jb]),
                                            "gain": float(pds_c[jb] - base_pds)}
                jn = best(nmae_c, False, pos) if nmae_ok else None
                if jn is not None:
                    rec[f"oracle_nmae_{tag}"] = {"mult": float(MULTS[jn]), "beta": float(betas[jn]),
                                                 "nmae": float(nmae_c[jn]), "gain": float(base_nmae - nmae_c[jn])}
            shuf = {k: [] for k in ("pds_mechanistic", "pds_free", "nmae_mechanistic", "nmae_free")}
            for _ in range(n_shuffle):
                sv = s.copy()
                sv[ok] = rng.permutation(s[ok])
                a, b = curve(sv)
                for tag, pos in (("mechanistic", True), ("free", False)):
                    jb = best(a, True, pos)
                    shuf[f"pds_{tag}"].append(float(a[jb] - base_pds))
                    jn = best(b, False, pos) if nmae_ok else None
                    if jn is not None:
                        shuf[f"nmae_{tag}"].append(float(base_nmae - b[jn]))
            for k, vals in shuf.items():
                real_key = f"oracle_{k}"
                if vals and real_key in rec:
                    arr = np.array(vals)
                    rec[f"shuffle_{k}"] = {"n": int(arr.size), "median": float(np.median(arr)),
                                           "q95": float(np.quantile(arr, 0.95)),
                                           "share_at_or_above_real": float(np.mean(arr >= rec[real_key]["gain"]))}
            rec["_s"] = s            # kept for the transfer pass, dropped before writing
            rec["_delta"] = delta[i].copy()
            rec["_curves"] = {"pds": pds_c, "nmae": nmae_c}
        out["per_target"][t] = rec
        msg = (f"   {fold} {t:7s} cells {rec['cells']:4d} gate {rec['gate']:5d}: truth x sites|expr,len "
               f"{rec.get('truth_n_cons_sites', float('nan')):+.3f} (pct {rec['pct_truth_sites_len']:.2f}, matched "
               f"{rec['pct_truth_sites_len_strength_matched']:.2f}); pred {rec.get('pred_n_cons_sites', float('nan')):+.3f}; "
               f"resid {rec.get('resid_n_cons_sites', float('nan')):+.3f}")
        if t in NAMED:
            om, sm = rec["oracle_pds_mechanistic"], rec.get("shuffle_pds_mechanistic", {})
            of, sf = rec["oracle_pds_free"], rec.get("shuffle_pds_free", {})
            msg += (f"; pds base {rec['base_pds']:.3f}, mechanistic oracle +{om['gain']:.4f} at {om['mult']:+.2f} sd "
                    f"(shuffled median +{sm.get('median', float('nan')):.4f}, share >= real {sm.get('share_at_or_above_real', float('nan')):.2f}); "
                    f"free oracle +{of['gain']:.4f} at {of['mult']:+.2f} sd (shuffled median +{sf.get('median', float('nan')):.4f}); "
                    f"nmae gate {rec['nmae_gate']}{'' if rec['nmae_read'] else ' (too small, not read)'}")
        log(msg)
    out["_fold"] = F
    out["_pds"] = pds_tool
    return out


def transfer(results: dict[str, dict], log) -> dict:
    """What a pipeline could learn from the OTHER lines: per fold and target, the donors' oracle
    multipliers (median, sign agreement) and one pooled multiplier fit on the donors' mean gain curve,
    each applied to this fold in units of its own delta's spread."""
    tr = {}
    for fold, res in results.items():
        F, tool = res["_fold"], res["_pds"]
        for t in NAMED:
            rec = res["per_target"].get(t)
            if rec is None or "curve" not in rec:
                continue
            donors = [results[o]["per_target"][t] for o in results if o != fold and "curve" in results[o]["per_target"].get(t, {})]
            if not donors:
                continue
            i = F.tpos[t]
            y, g = F.y[i], F.gate[i]
            got = {"donors": len(donors)}
            for metric in ("pds", "nmae"):
                for tag in ("mechanistic", "free"):
                    key = f"oracle_{metric}_{tag}"
                    mults = [d[key]["mult"] for d in donors if key in d]
                    if not mults:
                        continue
                    curves = [np.asarray(d["_curves"][metric]) - (d["base_pds"] if metric == "pds" else 0.0) for d in donors if key in d]
                    gains = np.nanmean(np.vstack(curves), axis=0)
                    if metric == "nmae":
                        gains = -gains
                    allowed = MULTS >= 0 if tag == "mechanistic" else np.ones_like(MULTS, bool)
                    gains = np.where(allowed, gains, np.nan)
                    pooled = float(MULTS[int(np.nanargmax(gains))]) if np.isfinite(gains).any() else float("nan")
                    for how, m in (("median", float(np.median(mults))), ("pooled", pooled)):
                        if not np.isfinite(m):
                            continue
                        d2 = rec["_delta"] + m * rec["delta_sd"] * rec["_s"]
                        if metric == "pds":
                            gain = tool.pds(t, d2) - rec["base_pds"]
                        elif rec["nmae_read"]:
                            p2 = pred_lfc(d2[None, :], [t], F.axis, F.frac, F.full.ctrl_mean_cpm, alpha=ALPHA)[0]
                            gain = rec["base_nmae"] - nmae_row(p2, y, g)
                        else:
                            continue
                        got[f"{metric}_{tag}_{how}"] = {"mult": m, "gain": float(gain), "donor_mults": mults,
                                                         "donors_same_sign": bool(len({np.sign(x) for x in mults}) <= 1)}
            tr[f"{fold}|{t}"] = got
            pm = got.get("pds_mechanistic_pooled", {})
            pf = got.get("pds_free_pooled", {})
            log(f"   transfer {fold} {t}: pds mechanistic pooled {pm.get('mult', float('nan')):+.2f} sd -> {pm.get('gain', float('nan')):+.4f}; "
                f"free pooled {pf.get('mult', float('nan')):+.2f} sd -> {pf.get('gain', float('nan')):+.4f} "
                f"(donor mults {pf.get('donor_mults')}, same sign {pf.get('donors_same_sign')})")
    return tr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", nargs="+", choices=FOLDS, default=FOLDS)
    ap.add_argument("--n-shuffle", type=int, default=100)
    args = ap.parse_args()
    if args.n_shuffle < 1:
        ap.error("--n-shuffle must be at least 1: the shuffled feature is the control")
    OUT.mkdir(parents=True, exist_ok=True)
    close.OUT.mkdir(parents=True, exist_ok=True)
    logf = (OUT / "run.log").open("a")
    log = lambda m: _log(m, logf)                                   # noqa: E731
    log(f"--- t102_mirna_targets.py folds={args.fold} n_shuffle={args.n_shuffle}")
    results = {f: run_fold(f, args.n_shuffle, log) for f in args.fold}
    tr = transfer(results, log)
    clean = {}
    for f, res in results.items():
        clean[f] = {k: v for k, v in res.items() if not k.startswith("_")}
        for t, rec in clean[f]["per_target"].items():
            clean[f]["per_target"][t] = {k: v for k, v in rec.items() if not k.startswith("_")}
    summary = {"folds": clean, "transfer": tr, "mults": MULTS.tolist(), "alpha": ALPHA,
               "min_gate_nmae": MIN_GATE_NMAE,
               "note": "gains are per target, raw; the panel mean moves by gain / n_targets (300 on the challenge)"}
    write_json(OUT / "mirna_targets.json", summary)
    (OUT / "MIRNA_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
