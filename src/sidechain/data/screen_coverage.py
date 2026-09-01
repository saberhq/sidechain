"""Screen-coverage QC for one pooled source: is it dense enough to be worth pooling?

    uv run python -m sidechain.data.screen_coverage \
        ~/data/sidechain/cache/vcc2026/k562_gwps_targets_pseudobulk.npz:control \
        ~/data/sidechain/derived/xatlas-orion/hct116_panel.npz:Non-Targeting \
        [--json out.json] [--tiers 3,10]

Answers the two coverage questions a CRISPR screen is judged on before any model sees
it, for every source we pool from, on artifacts we already hold:

  ARMS   cells per perturbation -- "which targets are too thin to trust". Straight from
         `n_cells`, the same number the `.qc.npz` sidecars report.
  GENES  cells' worth of evidence per (perturbation, gene) -- "which of this arm's genes
         are one lucky cell". `PseudobulkSums.n_eff`, computable because the sums we keep
         already carry it; the exact per-gene cell count does not survive streaming.

Plus mean UMIs per cell (log10) per arm, which is the third metric of the intake set.

The gene tier is the one with no other source: `count_sum` says a gene got 100 counts and
cannot say whether that was 100 cells at 1 or one cell at 100. See
`PseudobulkSums.n_eff` for what the number is and what it is not.

This reports; it does not filter. What the pooling does with the tiers is
`submit.build.coverage_factor` (private research/ideas/coverage-tiered-pooling-weights.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sidechain.data.stream_pseudobulk import PseudobulkSums

# Cells per arm. The screen-QC convention Saber brought over: report the fraction under
# 5 / 10 / 25, and 50 because X-Atlas's tail is fat enough that 25 misses it.
ARM_CUTS = (5, 10, 25, 50)


def arm_coverage(pb: PseudobulkSums, control: str | None = None) -> dict:
    """Cells per perturbation, and the thin tail. Controls excluded -- a pooled control
    arm is tens of thousands of cells and would move every summary statistic."""
    keep = np.array([lab != control for lab in pb.labels])
    n = pb.n_cells[keep]
    lib = pb.libsize_sum[keep] / np.maximum(pb.n_cells[keep], 1)
    return {
        "arms": int(n.size),
        "cells_min": int(n.min()) if n.size else 0,
        "cells_median": float(np.median(n)) if n.size else 0.0,
        "cells_max": int(n.max()) if n.size else 0,
        "under": {str(c): int((n < c).sum()) for c in ARM_CUTS},
        "under_pct": {str(c): float((n < c).mean() * 100) for c in ARM_CUTS} if n.size else {},
        "log10_umis_per_cell_median": float(np.log10(np.median(lib[lib > 0]))) if (lib > 0).any() else 0.0,
    }


def gene_coverage(pb: PseudobulkSums, control: str, tiers: tuple[float, float],
                  max_arms: int | None = None) -> dict:
    """How many (perturbation, gene) votes fall in each evidence tier, and what share of
    the inverse-variance weight they carry.

    Accumulated into counters one arm at a time -- a full-corpus X-Atlas artifact has
    ~700 million gene-arms and the values are never all held at once.

    The tier is taken on the WEAKER of the two arms behind the contrast (perturbed and
    control), because the pooled estimate is a difference: a gene the control barely
    saw is as poorly known as one the perturbation barely saw.
    """
    if control not in pb.labels:
        raise SystemExit(f"control {control!r} not in this source's labels "
                         f"(first few: {pb.labels[:4]})")
    weak, normal = tiers
    ci = pb.labels.index(control)
    ne_c = pb.n_eff(ci)
    perts = [i for i, lab in enumerate(pb.labels) if lab != control and pb.n_cells[i] > 0]
    if max_arms:
        perts = perts[:max_arms]

    counts = np.zeros(3, dtype=np.int64)
    weight = np.zeros(3)
    absfc = [[], [], []]
    for i in perts:
        ne = np.minimum(pb.n_eff(i), ne_c)
        # The same weight the pool would hand this arm, with the sampling floor on --
        # anything else would misreport the share (the floor already demotes thin genes).
        w = _poisson_floored_weight(pb, i, ci)
        band = np.digitize(ne, [weak, normal])          # 0 weak, 1 normal, 2 strong
        for b in (0, 1, 2):
            m = band == b
            counts[b] += int(m.sum())
            weight[b] += float(w[m].sum())
            if m.any():
                absfc[b].append(float(np.median(np.abs(_log2fc(pb, i, ci)[m]))))
    total_w = weight.sum() or 1.0
    total_c = counts.sum() or 1
    names = (f"weak (n_eff<{weak:g})", f"normal ({weak:g}-{normal:g})", f"strong (>={normal:g})")
    return {
        "arms_measured": len(perts),
        "genes": int(pb.genes.size),
        "tiers": [
            {"name": names[b], "gene_arms": int(counts[b]),
             "gene_arms_pct": float(counts[b] / total_c * 100),
             "weight_pct": float(weight[b] / total_w * 100),
             "median_abs_log2fc": float(np.median(absfc[b])) if absfc[b] else 0.0}
            for b in (0, 1, 2)
        ],
    }


def _log2fc(pb: PseudobulkSums, i: int, c: int, pseudocount: float = 1.0) -> np.ndarray:
    mi = pb.cpm_sum[i] / max(int(pb.n_cells[i]), 1)
    mc = pb.cpm_sum[c] / max(int(pb.n_cells[c]), 1)
    return np.log2((mi + pseudocount) / (mc + pseudocount))


def _poisson_floored_weight(pb: PseudobulkSums, i: int, c: int,
                            pseudocount: float = 1.0) -> np.ndarray:
    """`1 / var` as `submit.build.pooled_delta` computes it with `--var-floor poisson`.

    Duplicated rather than imported to keep this module free of the submission path;
    `test_screen_coverage.py` pins it against `_log2fc_with_var` so the two cannot drift.
    """
    ln2_sq = np.log(2) ** 2
    ni, nc = max(int(pb.n_cells[i]), 1), max(int(pb.n_cells[c]), 1)
    mi, mc = pb.cpm_sum[i] / ni, pb.cpm_sum[c] / nc
    vi = np.maximum(pb.cpm_sq_sum[i] / ni - mi * mi, 0.0)
    vc = np.maximum(pb.cpm_sq_sum[c] / nc - mc * mc, 0.0)
    vi = np.maximum(vi, (mi + pseudocount) * 1e6 / (pb.libsize_sum[i] / ni))
    vc = np.maximum(vc, (mc + pseudocount) * 1e6 / (pb.libsize_sum[c] / nc))
    var = ((vi / ni) / (mi + pseudocount) ** 2 + (vc / nc) / (mc + pseudocount) ** 2) / ln2_sq
    if ni < 2 or nc < 2:
        return np.zeros_like(var)
    return 1.0 / np.maximum(var, 1e-12)


def report(path: Path, control: str, tiers: tuple[float, float],
           max_arms: int | None = None) -> dict:
    pb = PseudobulkSums.load(path)
    return {
        "source": str(path),
        "control": control,
        "arms": arm_coverage(pb, control),
        "genes": gene_coverage(pb, control, tiers, max_arms=max_arms),
    }


def render(r: dict) -> str:
    a, g = r["arms"], r["genes"]
    out = [f"{Path(r['source']).name}  (control {r['control']!r})",
           (f"  ARMS   {a['arms']:,} perturbations · cells min {a['cells_min']} "
            f"median {a['cells_median']:.0f} max {a['cells_max']:,} · "
            f"log10 UMIs/cell {a['log10_umis_per_cell_median']:.2f}")]
    thin = "  ".join(f"<{c}: {a['under'][str(c)]:,} ({a['under_pct'].get(str(c), 0):.2f}%)"
                     for c in ARM_CUTS)
    out.append(f"         thin tail   {thin}")
    out.append(f"  GENES  {g['arms_measured']:,} arms x {g['genes']:,} genes")
    for t in g["tiers"]:
        out.append(f"         {t['name']:<20} {t['gene_arms_pct']:5.1f}% of votes  "
                   f"{t['weight_pct']:5.2f}% of weight  "
                   f"median |log2FC| {t['median_abs_log2fc']:.3f}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", metavar="NPZ:CONTROL",
                    help="pseudobulk .npz and its own control label -- the same syntax "
                         "sidechain.eval.loco and sidechain.submit.build take")
    ap.add_argument("--tiers", default="3,10", metavar="WEAK,NORMAL",
                    help="the two n_eff cut points (default 3,10): below WEAK is weak "
                         "evidence, WEAK..NORMAL is normal, at or above NORMAL is strong")
    ap.add_argument("--max-arms", type=int,
                    help="measure only the first N perturbations (a quick look at a "
                         "full-corpus artifact, which has ~700M gene-arms)")
    ap.add_argument("--json", type=Path, help="also write the report as JSON")
    args = ap.parse_args(argv)

    cuts = tuple(float(x) for x in args.tiers.split(","))
    if len(cuts) != 2 or not 0 < cuts[0] < cuts[1]:
        raise SystemExit(f"--tiers wants two increasing positive cut points, got {args.tiers!r}")

    reports = []
    for spec in args.sources:
        path, _, ctrl = spec.rpartition(":")
        r = report(Path(path).expanduser(), ctrl or "control", cuts, max_arms=args.max_arms)
        reports.append(r)
        print(render(r), flush=True)
        print(flush=True)
    if args.json:
        args.json.expanduser().write_text(json.dumps(reports, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
