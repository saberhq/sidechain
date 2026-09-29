#!/usr/bin/env python
"""T102, closing step 1 (c) -- how reproducible is a line's de-repressed set WITHIN the line, before
anyone calls the cross-line overlap small?

    .venv/bin/python scripts/t102_derepression_split.py

Step 1's corpus reading (`scripts/t102_universal.py --part corpus`) found that the genes going up
after DICER1, DROSHA or DGCR8 overlap between lines at 1.3 to 3.1 times a uniform draw -- about
what any reference knockdown's up-set does -- and that TargetScan's conserved site count tracks the
log2FC in every line where the knockdown bites. The review asked for the missing ceiling: split
each line's cells in two (a seeded half within every label, control included) and read

1. the WITHIN-line overlap of the de-repressed set, first half against second half (enrichment over a
   uniform draw and Jaccard), for the three miRNA-pathway knockdowns and the same 200 reference
   knockdowns `corpus.json` drew;
2. the CROSS-line overlap on the same footing, a line's first half against another line's first half (and second against second), so
   the ratio cross / within says how much of the within-line reproducibility survives a change of
   line (1 = as shared as the measurement allows);
3. the site-count partial rho (control expression and 3'UTR length held) in each half, for the
   miRNA-pathway knockdowns and the references -- is the one surviving mechanism reproducible
   inside a line at half the cells.

Inputs: the four essential-gene panels' cell-level h5ads (Zenodo 13350497, `scperturb_multicontext`
in `configs/datasets.yaml`, `perturbation` / `control`), `universal/corpus.json` for the reference
draw, the TargetScan per-gene table. Arithmetic: `per_gene_transfer.knockdown_lfc` (the pipeline's
own log2FC and Poisson-floored variance) and `derepressed` (centred, lfc > 0.25 and z > 2), exactly
as the corpus reading. Lands in ``runs/probes/t102_utr_load/derepression_split/``.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from sidechain.data.stream_pseudobulk import _obs_labels, stream_pseudobulk_file
from sidechain.eval.per_gene_transfer import CE2_MIN_CPM, derepressed, jaccard, knockdown_lfc, residualize_ranks, write_json

HERE = Path(__file__).parent


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


close = _sibling("t102_close")
DATA = close.DATA
PROBE = close.PROBE
OUT = PROBE / "derepression_split"
EXT = DATA / "external/zenodo-13350497"
PANELS = {"HepG2": "NadigOConner2024_hepg2.h5ad", "Jurkat": "NadigOConner2024_jurkat.h5ad",
          "RPE1": "ReplogleWeissman2022_rpe1.h5ad", "K562": "ReplogleWeissman2022_K562_essential.h5ad"}
PERT_COL, CONTROL = "perturbation", "control"
KDS = ["DICER1", "DROSHA", "DGCR8"]
MIN_CELLS_HALF = 10          # half of the corpus reading's 20-cell floor per arm


def _log(msg: str, fh) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def stream_halves(name: str, keep: set[str], log):
    cache = {h: OUT / "split" / f"{name}_{h}.npz" for h in "AB"}
    from sidechain.data.stream_pseudobulk import PseudobulkSums
    if all(p.exists() for p in cache.values()):
        return {h: PseudobulkSums.load(p) for h, p in cache.items()}
    t0 = time.time()
    path = EXT / PANELS[name]
    with h5py.File(path, "r") as f:
        labels = _obs_labels(f, PERT_COL)
        half = close._split_assignment(labels, close.SPLIT_SEED)
        tagged = np.char.add(np.char.add(labels.astype(str), close.SEP), half)
        want = {f"{k}{close.SEP}{h}" for k in keep for h in "AB"}
        pb = stream_pseudobulk_file(f, PERT_COL, keep=want, labels_all=tagged, source=str(path))
    out = {h: close._half(pb, h) for h in "AB"}
    for h, p in cache.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        out[h].save(p)
    log(f"   {name}: {len(labels):,} cells streamed into halves ({time.time() - t0:.0f}s)")
    return out


def utr_sites(axis: list[str]) -> tuple[np.ndarray, np.ndarray]:
    bridge = json.loads((PROBE / "bridge" / "symbol_to_ensg.json").read_text())
    load = pd.read_parquet(DATA / "derived/targetscan-vert_80/utr_load.parquet")
    by_id = load.drop_duplicates("gene_id").set_index("gene_id")
    by_sym = load.drop_duplicates("symbol").set_index("symbol")
    sites, length = np.full(len(axis), np.nan), np.full(len(axis), np.nan)
    for i, g in enumerate(axis):
        e = bridge.get(g)
        row = by_id.loc[e] if e in by_id.index else (by_sym.loc[g] if g in by_sym.index else None)
        if row is not None:
            sites[i], length[i] = float(row["n_cons_sites"]), float(row["utr_len"])
    return sites, np.log10(np.clip(length, 1, None))


def partial(y, x, cov) -> float:
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(cov).all(axis=0)
    if ok.sum() < 50:
        return float("nan")
    return float(np.corrcoef(residualize_ranks(x[ok], cov[:, ok]), residualize_ranks(y[ok], cov[:, ok]))[0, 1])


def enrichment(a: np.ndarray, b: np.ndarray, universe: np.ndarray) -> float:
    a, b = a & universe, b & universe
    n = int(universe.sum())
    exp = int(a.sum()) * int(b.sum()) / n if n else 0.0
    return float((a & b).sum() / exp) if exp > 0 else float("nan")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    logf = (OUT / "run.log").open("a")
    log = lambda m: _log(m, logf)                                   # noqa: E731
    log("--- t102_derepression_split.py")
    corpus = json.loads((PROBE / "universal" / "corpus.json").read_text())
    reference = list(corpus["reference"])
    keep = {CONTROL, *KDS, *reference}
    halves = {n: stream_halves(n, keep, log) for n in PANELS}
    axis = sorted(set.intersection(*[set(h["A"].genes) for h in halves.values()]))
    pos = {n: {g: i for i, g in enumerate(h["A"].genes)} for n, h in halves.items()}
    idx = {n: np.array([pos[n][g] for g in axis]) for n in halves}
    sites, log_len = utr_sites(axis)
    log(f"   shared axis {len(axis)} genes; UTR rows for {int(np.isfinite(sites).sum())}")
    lfc, var, ncells, expressed = {}, {}, {}, {}
    for n, hh in halves.items():
        for h, pb in hh.items():
            c = pb.labels.index(CONTROL)
            cpm = (pb.cpm_sum[c] / max(int(pb.n_cells[c]), 1))[idx[n]]
            expressed[(n, h)] = cpm >= CE2_MIN_CPM
            for kd in [*KDS, *reference]:
                if kd in pb.labels:
                    f_, v_ = knockdown_lfc(pb, kd, CONTROL)
                    lfc[(n, h, kd)], var[(n, h, kd)] = f_[idx[n]], v_[idx[n]]
                    ncells[(n, h, kd)] = int(pb.n_cells[pb.labels.index(kd)])
    names = list(halves)
    both = {n: expressed[(n, "A")] & expressed[(n, "B")] for n in names}

    def dset(n, h, kd):
        return derepressed(lfc[(n, h, kd)], var[(n, h, kd)], expressed[(n, h)])

    def thin(n, kd):
        return min(ncells.get((n, "A", kd), 0), ncells.get((n, "B", kd), 0)) < MIN_CELLS_HALF

    # (1) within-line, first half vs second half
    within = {}
    for n in names:
        ref_enr = [enrichment(dset(n, "A", k), dset(n, "B", k), both[n]) for k in reference
                   if (n, "A", k) in lfc and not thin(n, k)]
        ref_enr = np.array([e for e in ref_enr if np.isfinite(e)])
        for kd in KDS:
            if (n, "A", kd) not in lfc:
                continue
            a, b = dset(n, "A", kd), dset(n, "B", kd)
            within[f"{n}|{kd}"] = {"cells_half": [ncells[(n, "A", kd)], ncells[(n, "B", kd)]], "thin": thin(n, kd),
                                   "set_a": int((a & both[n]).sum()), "set_b": int((b & both[n]).sum()),
                                   "enrichment": enrichment(a, b, both[n]), "jaccard": jaccard(a & both[n], b & both[n]),
                                   "reference_within_median": float(np.median(ref_enr)) if ref_enr.size else float("nan"),
                                   "reference_within_iqr": [float(np.quantile(ref_enr, q)) for q in (0.25, 0.75)] if ref_enr.size else []}
            w = within[f"{n}|{kd}"]
            log(f"   within {n:6s} {kd:7s} halves {w['cells_half']} sets {w['set_a']}/{w['set_b']} "
                f"x{w['enrichment']:.2f} (J {w['jaccard']:.3f}); reference within x{w['reference_within_median']:.2f}"
                f"{'  THIN' if w['thin'] else ''}")
    # (2) cross-line on the same footing: first half of one line vs first half of the other (and second vs second)
    cross = {}
    for a, b in itertools.combinations(names, 2):
        u = both[a] & both[b]
        ref_c, ref_ratio = [], []
        for k in reference:
            if (a, "A", k) in lfc and (b, "A", k) in lfc and not thin(a, k) and not thin(b, k):
                ec = np.nanmean([enrichment(dset(a, h, k), dset(b, h, k), u) for h in "AB"])
                ref_c.append(ec)
                # the same ratio for every reference knockdown, from its own two within-line
                # enrichments (results review: a ratio of medians is not a median of ratios)
                wa_k = enrichment(dset(a, "A", k), dset(a, "B", k), both[a])
                wb_k = enrichment(dset(b, "A", k), dset(b, "B", k), both[b])
                if np.isfinite(ec) and np.isfinite(wa_k) and np.isfinite(wb_k) and wa_k > 1 and wb_k > 1:
                    ref_ratio.append((ec - 1) / (np.sqrt(wa_k * wb_k) - 1))
        ref_c = np.array([e for e in ref_c if np.isfinite(e)])
        ref_ratio = np.array(ref_ratio)
        for kd in KDS:
            if (a, "A", kd) not in lfc or (b, "A", kd) not in lfc:
                continue
            e = float(np.nanmean([enrichment(dset(a, h, kd), dset(b, h, kd), u) for h in "AB"]))
            wa = within.get(f"{a}|{kd}", {}).get("enrichment", float("nan"))
            wb = within.get(f"{b}|{kd}", {}).get("enrichment", float("nan"))
            # read only when BOTH lines reproduce their own set (enrichment above 1): HepG2's
            # DICER1 sits at 0.78 and would otherwise shrink the denominator (results review)
            ok_w = np.isfinite(wa) and np.isfinite(wb) and wa > 1 and wb > 1
            ceil = float(np.sqrt(wa * wb)) if ok_w else float("nan")
            cross[f"{a}|{b}|{kd}"] = {"enrichment_half_vs_half": e, "within_geomean": ceil,
                                      "cross_over_within": (e - 1) / (ceil - 1) if np.isfinite(ceil) and ceil > 1 else float("nan"),
                                      "reference_cross_median": float(np.median(ref_c)) if ref_c.size else float("nan"),
                                      "reference_cross_over_within_median": float(np.median(ref_ratio)) if ref_ratio.size else float("nan"),
                                      "reference_cross_over_within_n": int(ref_ratio.size),
                                      "thin": thin(a, kd) or thin(b, kd)}
            c = cross[f"{a}|{b}|{kd}"]
            log(f"   cross  {a:6s} x {b:6s} {kd:7s} x{e:.2f} half-vs-half; within geomean x{ceil:.2f}; "
                f"(cross-1)/(within-1) {c['cross_over_within']:.2f}; reference cross x{c['reference_cross_median']:.2f}, "
                f"reference (cross-1)/(within-1) median {c['reference_cross_over_within_median']:.2f} (n {c['reference_cross_over_within_n']})"
                f"{'  THIN' if c['thin'] else ''}")
    # (3) the site-count partial in each half, against the same line's references in the same half
    sites_rho = {}
    for n in names:
        for h in "AB":
            logc = np.log10(np.clip((halves[n][h].cpm_sum[halves[n][h].labels.index(CONTROL)]
                                     / max(int(halves[n][h].n_cells[halves[n][h].labels.index(CONTROL)]), 1))[idx[n]], 1e-3, None))
            cov = np.vstack([logc, log_len])
            refs = np.array([partial(np.where(expressed[(n, h)], lfc[(n, h, k)], np.nan), sites, cov)
                             for k in reference if (n, h, k) in lfc and not thin(n, k)])
            refs = refs[np.isfinite(refs)]
            for kd in KDS:
                if (n, h, kd) not in lfc:
                    continue
                r = partial(np.where(expressed[(n, h)], lfc[(n, h, kd)], np.nan), sites, cov)
                sites_rho.setdefault(f"{n}|{kd}", {})[h] = {
                    "rho_partial_expr_len": r, "percentile_in_reference": float((refs < r).mean()) if refs.size else float("nan"),
                    "reference_median": float(np.median(refs)) if refs.size else float("nan"), "thin": thin(n, kd)}
        for kd in KDS:
            s = sites_rho.get(f"{n}|{kd}")
            if s:
                log(f"   sites  {n:6s} {kd:7s} first half {s['A']['rho_partial_expr_len']:+.3f} (pct {s['A']['percentile_in_reference']:.2f}), "
                    f"second half {s['B']['rho_partial_expr_len']:+.3f} (pct {s['B']['percentile_in_reference']:.2f})"
                    f"{'  THIN' if s['A']['thin'] else ''}")
    write_json(OUT / "derepression_split.json", {
        "panels": names, "axis_genes": len(axis), "reference": reference, "min_cells_half": MIN_CELLS_HALF,
        "split_seed": close.SPLIT_SEED, "within": within, "cross": cross, "sites": sites_rho,
        "rules": {"derepressed": "expressed in the half & centred lfc > 0.25 & z > 2 (per_gene_transfer.derepressed)",
                  "enrichment": "overlap / (|a| |b| / n) on genes expressed in both halves (and both lines for cross)",
                  "cross_over_within": "(cross - 1) / (geomean within - 1): 1 = as shared as within-line reproducibility allows"}})
    (OUT / "DEREP_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
