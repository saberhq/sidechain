"""Emit site/data/coverage.json — what each held corpus covers of the 2026 target panel.

    uv run python -m sidechain.data.coverage [--check]

Reads the pseudobulk caches under ~/data/sidechain/cache/vcc2026/ (built by
``sidechain.data.stream_pseudobulk``) and the challenge's 300-target panel
(``pert_counts.csv``), and writes the JSON the site's "What the models draw on"
section renders at build time. Run it offline after an ingest changes a cache;
commit the JSON — CI renders, it never computes (ADR 0006). Numbers are measured
from the caches, never typed in: a dashboard that drifts from the pipeline is
worse than no dashboard.

Only facts that are public anyway belong here: which public corpora are held, the
cell line, whether a corpus feeds the model or the local evaluation, and per-target
cell depth — all properties of published datasets or of code that is itself public.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path.home() / "data" / "sidechain"
CACHE = DATA / "cache" / "vcc2026"
PANEL = DATA / "vcc2026" / "pert_counts.csv"
OUT = Path(__file__).resolve().parents[3] / "site" / "data" / "coverage.json"

# What each cache is, in public terms: the cell line and the screen scale, no more.
# `role`: "prior" feeds the model's per-gene effects; "eval" is a local mirror panel
# used to score methods before submitting (sidechain.eval.mirror2026, public); "bench"
# is held and measured but not in the pool (all three declared in configs/datasets.yaml).
# `cache:` is relative to cache/vcc2026/, `path:` to ~/data/sidechain/ — same tree,
# two entry points, because streamed corpora land their aggregates under derived/.
# `loader: lfc` reads a published per-gene contrast table (labels × lfc, no cells).
SOURCES = [
    {"cache": "h1_pseudobulk.npz", "name": "VCC 2025 training data", "line": "H1 hESC", "role": "prior",
     "note": "the 2025 challenge corpus; its 300 perturbations are a different panel than 2026's"},
    {"cache": "k562_gwps_targets_pseudobulk.npz", "name": "Genome-wide CRISPRi screen", "line": "K562", "role": "prior",
     "note": "streamed for the 2026 panel targets only — the cache holds just those columns"},
    {"path": "derived/xatlas-orion/hct116_full.npz", "name": "Genome-wide dual-guide CRISPRi (X-Atlas/Orion)",
     "line": "HCT116", "role": "prior",
     "note": "streamed once from 126 GB of parquet, never stored — only this aggregate lands (both lines)"},
    {"path": "derived/xatlas-orion/hek293t_full.npz", "name": "Genome-wide dual-guide CRISPRi (X-Atlas/Orion)",
     "line": "HEK293T", "role": "prior"},
    {"path": "cache/vcc2026/feng_genomewide_lfc.npz", "name": "Genome-wide CRISPRi, published log2FCs",
     "line": "iPSC pool", "role": "bench", "loader": "lfc",
     "note": "a published per-gene contrast table, not cells; scored on the local mirror, did not earn a slot in the pool"},
    {"path": "derived/lamin-pertdata/sunshine23_all_pseudobulk.npz", "name": "CRISPRi host-factor screen",
     "line": "Calu-3", "role": "bench"},
    {"path": "derived/lamin-perturbench/mcfaline23_none_pseudobulk.npz", "name": "CRISPRi screen (vehicle arm)",
     "line": "GBM pool", "role": "bench"},
    {"cache": "k562_essential_all_pseudobulk.npz", "name": "Essential-gene panel", "line": "K562", "role": "eval"},
    {"cache": "rpe1_all_pseudobulk.npz", "name": "Essential-gene panel", "line": "RPE1", "role": "eval"},
    {"cache": "jurkat_all_pseudobulk.npz", "name": "Essential-gene panel", "line": "Jurkat", "role": "eval"},
    {"cache": "hepg2_all_pseudobulk.npz", "name": "Essential-gene panel", "line": "HepG2", "role": "eval"},
]

CONTROL_MARKERS = ("non-targeting", "non_targeting", "control")


def is_control(label: str) -> bool:
    return any(m in label.lower() for m in CONTROL_MARKERS)


def measure(panel: set[str]) -> dict:
    sources, prior_union = [], set()
    for spec in SOURCES:
        path = CACHE / spec["cache"] if "cache" in spec else DATA / spec["path"]
        if not path.exists():
            print(f"skipping {path.name} (not built)", file=sys.stderr)
            continue
        d = np.load(path, allow_pickle=True)
        labels = [str(x) for x in d["labels"]]
        if spec.get("loader") == "lfc":
            # a published contrast table: labels are targets, and there are no cells
            targets = labels
            cells = {}
        else:
            cells = {lab: int(n) for lab, n in zip(labels, d["n_cells"])}
            targets = [lab for lab in labels if not is_control(lab)]
        covered = sorted(set(targets) & panel)
        depth = sorted(cells[t] for t in covered) if cells else []
        if spec["role"] == "prior":
            prior_union |= set(covered)
        sources.append({
            "name": spec["name"], "line": spec["line"], "role": spec["role"],
            "targets_held": len(targets),
            "panel_covered": len(covered),
            "cells_targets": int(sum(cells[t] for t in targets)) if cells else None,
            "cells_control": int(sum(n for lab, n in cells.items() if is_control(lab))) if cells else None,
            "cells_per_covered_target": (
                {"min": depth[0], "median": int(np.median(depth)), "max": depth[-1]} if depth else None),
            "genes": len(d["genes"]),
            **({"note": spec["note"]} if spec.get("note") else {}),
        })
    return {
        "_about": ("Generated by `uv run python -m sidechain.data.coverage` from the pseudobulk caches "
                   "under ~/data/sidechain/cache/vcc2026/ — do not edit by hand. Re-run after an ingest "
                   "changes a cache; the site renders this at build time and never computes."),
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "panel_size": len(panel),
        "prior_covered": len(prior_union),
        "uncovered": len(panel) - len(prior_union),
        "sources": sources,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if site/data/coverage.json is out of date")
    args = ap.parse_args()

    panel = set(PANEL.read_text().splitlines()[1:])  # 2026 file has a `target_gene` header
    if len(panel) != 300:
        sys.exit(f"panel read {len(panel)} targets, expected 300 — check {PANEL}")
    payload = measure(panel)

    if args.check:
        if not OUT.exists():
            print("site/data/coverage.json missing")
            return 1
        old = json.loads(OUT.read_text())
        new = {k: v for k, v in payload.items() if k != "generated_utc"}
        old.pop("generated_utc", None)
        same = old == new
        print("coverage.json " + ("current" if same else "OUT OF DATE — re-run without --check"))
        return 0 if same else 1

    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    for s in payload["sources"]:
        d = s["cells_per_covered_target"]
        depth = f"cells/target min {d['min']} · med {d['median']}" if d else "—"
        print(f"{s['line']:7s} {s['name']:28s} {s['role']:5s} panel {s['panel_covered']:3d}/300  {depth}")
    print(f"prior union {payload['prior_covered']}/300, uncovered {payload['uncovered']} -> {OUT.relative_to(OUT.parents[2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
