#!/usr/bin/env python
"""T102 -- fetch the whole TargetScan Human 8.0 release through the ADR 0003 gate and build
the four derived tables, printing each table's size and time and writing a summary.

    .venv/bin/python scripts/t102_build_targetscan.py                 # fetch what is missing, build what is missing
    .venv/bin/python scripts/t102_build_targetscan.py --refresh       # accept a changed file list (PROVENANCE.json rewritten)
    .venv/bin/python scripts/t102_build_targetscan.py --rebuild --no-all

The tables, in build order (each reads the ones before it):
  utr_load.parquet          one row per gene, the per-gene load (2026-09-25)
  sites.parquet             one row per site of a conserved family, with its 3'UTR position
  edges_default.parquet     one row per (family, gene), the edge form (default predictions)
  edges_all.parquet         the same over every family (skipped with --no-all; 2.4 GB of text)
  utr_seq_features.parquet  one row per representative transcript: composition, motifs, cooperation counts
  utr_features.parquet      utr_load joined with utr_seq_features

Runs on the box (Saber's rule for T102), a few GB of RAM: every all-species table is read in
chunks and filtered to human. The build summary lands beside the tables as
``build_summary.json``; the per-table lineage is in ``LINEAGE.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from sidechain.priors.posttx_mirna import MiRNATargetSource, spec_from_registry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="accept a changed block / upstream state")
    ap.add_argument("--rebuild", action="store_true", help="ignore cached parquets")
    ap.add_argument("--no-all", action="store_true", help="skip edges_all (the 2.4 GB all-predictions file)")
    ap.add_argument("--skip-fetch", action="store_true")
    args = ap.parse_args()
    src = MiRNATargetSource(spec_from_registry("targetscan"), {})
    summary = {"steps": []}

    def step(name, fn):
        t0 = time.time()
        out = fn()
        n = None if out is None else (len(out) if hasattr(out, "__len__") else out)
        rec = {"step": name, "rows": n, "seconds": round(time.time() - t0)}
        summary["steps"].append(rec)
        print(f"  {name:22s} rows {n!s:>10s}  {rec['seconds']:6d}s", flush=True)
        return out

    if not args.skip_fetch:
        print("fetch through the gate", flush=True)
        step("fetch", lambda: str(src.fetch(refresh=args.refresh, progress=True)))
    print("tables", flush=True)
    step("utr_load", lambda: src.utr_load_table(rebuild=args.rebuild))
    sites = step("sites", lambda: src.site_table(rebuild=args.rebuild))
    step("edges_default", lambda: src.edge_table(scope="default", rebuild=args.rebuild))
    if not args.no_all:
        step("edges_all", lambda: src.edge_table(scope="all", rebuild=args.rebuild))
    step("utr_seq_features", lambda: src.utr_sequence_table(rebuild=args.rebuild, sites=sites))
    step("utr_features", lambda: src.utr_features_table(rebuild=args.rebuild))
    lineage = json.loads((src.derived / "LINEAGE.json").read_text())
    summary["lineage_entries"] = sorted(lineage["entries"])
    seq = lineage["entries"].get("targetscan-vert_80/utr_seq_features", {})
    summary["site_coordinate_check"] = seq.get("site_coordinate_check")
    (src.derived / "build_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"site coordinate check: {summary['site_coordinate_check']}")
    print(f"wrote {src.derived / 'build_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
