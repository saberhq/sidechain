#!/usr/bin/env python
"""T102 -- the per-gene TRUST screen: shrink (or boost) each gene's transferred delta by its
3'UTR regulatory load, and read raw `pds` and raw `nmae` on the fold rotation. No cells emitted.

    .venv/bin/python scripts/t102_gene_trust_screen.py                    # every fold
    .venv/bin/python scripts/t102_gene_trust_screen.py --fold loco_hct116

The question, in one line: T102's reading says genes with a heavy 3'UTR load are the ones whose
knockdown response transfers WORST across lines (higher per-gene error, sign flipping by line).
The amplitude test showed no size correction survives a change of line. What it never asked is
whether taking those genes DOWN -- multiplying their transferred log2FC by a factor below one, so
they contribute less to the predicted vector -- improves the DIRECTION the cosine reads (`pds`)
or the L1 error (`nmae`). Shrinking a wrong-signed gene toward zero helps both; shrinking a
right-signed one hurts. Which wins is what this measures.

The knob family, per gene g with load quantile q_g in [0, 1] (rank of `n_cons_sites` or
`log_utr_len` among the genes the table covers; genes it does not cover keep 1):

    shrink:  s_g = 1 - beta * q_g        heavy-load genes taken down, beta in (0, 1]
    boost:   s_g = 1 + beta * q_g        the reverse, because K562 and HCT116 wanted more size

applied to the pooled delta BEFORE alpha and the knockdown pin, exactly where a `--gene-trust`
knob in `submit.build` would sit. Both signs are tried because the size direction flipped by line.

Read-outs, all raw, all on the analytic path (even emission, alpha 1.35):
  * `pds_cosine` through cell-eval2's own kernel (`analytic_pds.score_delta`), per target, so
    the paired 2*se bar over targets is the noise bar (T59's estimator, per contrast);
  * `nmae` in closed form on the metric's own gate (`per_gene_transfer.nmae_closed_form`), the
    Wilcoxon gate where it exists, the corrected stand-in where it does not, and the result
    says which.
Controls: the load shuffled across genes (`n_shuffle` draws) at every setting -- the SAME spread
of factors assigned at random -- and an in-sample per-gene oracle for nmae (each gene's own L1
factor from the held-out truth, clipped to [0, 3]) as the ceiling on ANY per-gene multiplier.

Writes `runs/probes/t102_utr_load/trust/<fold>.json` and `trust/summary.json`. A screen, not a
score: whatever it selects still needs a real `sidechain.eval.loco` run before the number is
quoted (analytic_pds docstring, boundary 3).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cell_eval2.metrics.discrimination import discrimination_score
from cell_eval2.prep import bulk_lognorm_means

from sidechain.eval.analytic_pds import BULK_TARGET_SUM, CONTROL, PERT_COL, emitted_sums
from sidechain.eval.per_gene_transfer import nmae_closed_form, pred_lfc, write_json

PROBE = Path("~/data/sidechain/runs/probes/t102_utr_load").expanduser()
ALPHA = 1.35
KD = -2.32
BETAS = (0.25, 0.5, 0.75, 1.0)
FEATURES = ("n_cons_sites", "log_utr_len")


def _driver():
    spec = importlib.util.spec_from_file_location(
        "t102_utr_load", Path(__file__).with_name("t102_utr_load.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pds_per_target(deltas: np.ndarray, targets: list[str], fc, covered) -> np.ndarray:
    """Raw pds_cosine per target for a [P, G] delta (alpha applied here, pin honoured)."""
    d = np.asarray(deltas, dtype=np.float64) * ALPHA
    pos = {g: i for i, g in enumerate(fc.genes)}
    for i, t in enumerate(targets):
        j = pos.get(str(t))
        if j is not None and covered[i]:
            d[i, j] = KD
    sums = emitted_sums(d, fc.frac, fc.lib_median, fc.cells_for(targets))
    pred_means = bulk_lognorm_means(sums, BULK_TARGET_SUM)
    out = discrimination_score(
        pred_bulk=(np.asarray(targets, dtype=str), pred_means),
        real_bulk=(np.asarray(fc.perts, dtype=str), fc.real_means),
        pert_col=PERT_COL, control=CONTROL, distance="cosine", rank_denominator="n-1",
        tie_policy="midrank", exclude_target_gene=True, exclusion_scope="panel",
        control_source="real", genes=np.asarray(fc.genes, dtype=str))
    if isinstance(out, dict):
        return np.array([out[str(t)] for t in targets], dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def bar(diff: np.ndarray) -> float:
    return float(2.0 * diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")


def run_fold(fold: str, mod, utr: pd.DataFrame, n_shuffle: int, log) -> dict:
    t0 = time.time()
    log(f"== {fold}")
    L = mod.load_fold(fold, "auto", log)
    fc, truth, targets, axis = L["fc"], L["truth"], L["targets"], L["axis"]
    deltas, covered, gate = L["deltas"], L["covered"], L["gate"]
    y = truth.truth_lfc()
    # the load on this fold's axis, through the same join the correlations used
    joined, match = mod.join_features(pd.DataFrame({"gene": axis}), utr)
    pos = {g: i for i, g in enumerate(axis)}
    have = np.zeros(len(axis), dtype=bool)
    have[[pos[g] for g in joined["gene"]]] = True
    quant = {}
    for feat in FEATURES:
        v = np.full(len(axis), np.nan)
        v[[pos[g] for g in joined["gene"]]] = joined[feat].to_numpy()
        q = np.full(len(axis), np.nan)
        r = pd.Series(v[have]).rank(method="average").to_numpy()
        q[have] = (r - 1) / max(len(r) - 1, 1)
        quant[feat] = q
    log(f"   load on the axis: {int(have.sum()):,} of {len(axis):,} genes ({match['matched']} matched)")

    def evaluate(factor: np.ndarray):
        d = deltas * factor[None, :]
        p = pds_per_target(d, targets, fc, covered)
        pred = pred_lfc(d, targets, axis, fc.frac, truth.ctrl_mean_cpm, alpha=ALPHA, covered=covered)
        n_mean, n_per = nmae_closed_form(pred, y, gate)
        return p, n_per

    base_p, base_n = evaluate(np.ones(len(axis)))
    log(f"   base: raw pds {base_p.mean():.4f} over {len(base_p)} targets; raw nmae {base_n.mean():.4f} "
        f"over {len(base_n)} targets ({L['gate_info']['kind']})")
    rng = np.random.default_rng(0)
    rows = []
    for feat in FEATURES:
        q = quant[feat]
        for sign, name in ((-1.0, "shrink"), (1.0, "boost")):
            for beta in BETAS:
                s = np.where(np.isfinite(q), 1.0 + sign * beta * np.nan_to_num(q), 1.0)
                p, n = evaluate(s)
                dp, dn = p - base_p, n - base_n
                shuf_p, shuf_n = [], []
                for _ in range(n_shuffle):
                    qs = q.copy()
                    qs[have] = rng.permutation(q[have])
                    ss = np.where(np.isfinite(qs), 1.0 + sign * beta * np.nan_to_num(qs), 1.0)
                    ps, ns = evaluate(ss)
                    shuf_p.append(ps.mean() - base_p.mean()); shuf_n.append(ns.mean() - base_n.mean())
                row = {"feature": feat, "family": name, "beta": beta,
                       "d_pds": float(dp.mean()), "d_pds_2se": bar(dp),
                       "d_nmae": float(dn.mean()), "d_nmae_2se": bar(dn),
                       "shuffle_d_pds_mean": float(np.mean(shuf_p)), "shuffle_d_pds_sd": float(np.std(shuf_p)),
                       "shuffle_d_nmae_mean": float(np.mean(shuf_n)), "shuffle_d_nmae_sd": float(np.std(shuf_n))}
                rows.append(row)
                log(f"   {feat:12s} {name:6s} beta {beta:.2f}: d pds {dp.mean():+.4f} (2se {bar(dp):.4f}; "
                    f"shuffle {np.mean(shuf_p):+.4f}+/-{np.std(shuf_p):.4f})  d nmae {dn.mean():+.4f} "
                    f"(2se {bar(dn):.4f}; shuffle {np.mean(shuf_n):+.4f}+/-{np.std(shuf_n):.4f})")
    # the in-sample oracle: each gene's own L1 factor from the held-out truth, clipped to [0, 3]
    per_gene = pd.read_parquet(PROBE / f"per_gene_{fold}.parquet")
    o = np.ones(len(axis))
    idx = [pos[g] for g in per_gene["gene"] if g in pos]
    o[idx] = np.clip(np.nan_to_num(per_gene["l1_scale"].to_numpy()[[i for i, g in enumerate(per_gene["gene"]) if g in pos]], nan=1.0), 0.0, 3.0)
    po, no = evaluate(o)
    oracle = {"d_pds": float((po - base_p).mean()), "d_pds_2se": bar(po - base_p),
              "d_nmae": float((no - base_n).mean()), "d_nmae_2se": bar(no - base_n),
              "genes_with_factor": int(len(idx))}
    # and the oracle's SIGN-ONLY cousin: zero every gene the truth says we get wrong more often than not
    z = np.ones(len(axis))
    sa = per_gene["sign_agree"].to_numpy()
    z[idx] = np.where(sa[[i for i, g in enumerate(per_gene["gene"]) if g in pos]] < 0.5, 0.0, 1.0)
    pz, nz = evaluate(z)
    oracle_zero = {"d_pds": float((pz - base_p).mean()), "d_nmae": float((nz - base_n).mean()),
                   "genes_zeroed": int((z[idx] == 0).sum())}
    log(f"   oracle (per-gene L1 factor, in-sample): d pds {oracle['d_pds']:+.4f}, d nmae {oracle['d_nmae']:+.4f}; "
        f"zero-the-wrong-sign genes: d pds {oracle_zero['d_pds']:+.4f}, d nmae {oracle_zero['d_nmae']:+.4f} "
        f"({oracle_zero['genes_zeroed']} genes)")
    return {"fold": fold, "gate": L["gate_info"]["kind"], "targets_pds": int(len(base_p)),
            "targets_nmae": int(len(base_n)), "base_pds": float(base_p.mean()), "base_nmae": float(base_n.mean()),
            "genes_with_load": int(have.sum()), "rows": rows, "oracle_l1": oracle, "oracle_zero_wrong_sign": oracle_zero,
            "seconds": round(time.time() - t0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", action="append")
    ap.add_argument("--n-shuffle", type=int, default=5)
    args = ap.parse_args()
    # cell-eval2 repeats its target-gene-exclusion notice on every call of the kernel; the
    # analytic path already documents it (43 of 272 K562 labels name no measured gene)
    logging.getLogger("cell_eval2").setLevel(logging.ERROR)
    mod = _driver()
    out_dir = PROBE / "trust"; out_dir.mkdir(exist_ok=True)
    logf = (out_dir / "run.log").open("a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    log(f"--- t102_gene_trust_screen.py {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    src = mod.MiRNATargetSource(mod.spec_from_registry("targetscan"), {})
    utr = src.utr_load_table()
    for fold in (args.fold or list(mod.FOLDS)):
        try:
            res = run_fold(fold, mod, utr, args.n_shuffle, log)
        except SystemExit as exc:
            log(f"   SKIPPED {fold}: {exc}"); res = {"fold": fold, "skipped": str(exc)}
        write_json(out_dir / f"{fold}.json", res)
    summary = {f: json.loads((out_dir / f"{f}.json").read_text()) for f in mod.FOLDS if (out_dir / f"{f}.json").exists()}
    write_json(out_dir / "summary.json", summary)
    log(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    sys.exit(main())
