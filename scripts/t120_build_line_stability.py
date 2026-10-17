"""T120 -- the four measured per-line stability tables through the gate and onto disk (phase 3): Jurkat's
resting decay rate (Blackinton & Keene 2016), an HCT116 actinomycin D rate (Pasterczyk 2024), K562
TimeLapse-seq half-lives (Schofield 2018) and HEK293/MCF7 4sU half-lives (Schueler 2014); prints the counts.

    .venv/bin/python scripts/t120_build_line_stability.py                 # fetch (idempotent), build what is missing
    .venv/bin/python scripts/t120_build_line_stability.py --rebuild       # rebuild the four tables
    .venv/bin/python scripts/t120_build_line_stability.py --only k562_halflife_schofield2018

Reads, under ~/data/sidechain/: the files the four blocks of configs/data_sources.yaml name (landed under
external/geo-GSE71382, external/geo-GSE234877, external/springernature-nmeth-4582, external/geo-GSE49831),
vcc2026/gene_names.csv, runs/probes/t102_utr_load/bridge/symbol_to_ensg.json and
external/hgnc-custom-uniprot-20261005/hgnc_custom_uniprot.tsv. Writes derived/<source>/stability.parquet and a
LINEAGE.json entry per source. Under a minute on the Mac; then rebuild the view with scripts/t120_build_ptr_table.py.

The parsers and the fit: src/sidechain/priors/posttx_line_stability.py; the results: private
research/ideas/ptr-feature-table.md § Outcome; the task: private tasks/T120.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from sidechain.priors.posttx_line_stability import BLOCKS, line_sources
from sidechain.priors.posttx_mirna import DATA_ROOT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--only", action="append", choices=BLOCKS, default=None, help="one block; repeatable")
    ap.add_argument("--root", default=str(DATA_ROOT))
    args = ap.parse_args(argv)
    t0 = time.time()
    for src in line_sources(args.root):
        if args.only and src.name not in args.only:
            continue
        dest = src.fetch(progress=True)
        table = src.stability_table(rebuild=args.rebuild)
        entry = json.loads((src.derived / "LINEAGE.json").read_text())["entries"][src.lineage_key]
        counts = {k: v for k, v in entry["counts"].items() if not isinstance(v, (dict, list))}
        print(f"{src.name}: files at {dest}; {len(table)} genes, {len(entry['columns'])} columns  "
              f"({time.time() - t0:.0f}s)")
        print("  " + json.dumps(counts))
        if "symbol_resolution" in entry["counts"]:
            print("  symbols: " + json.dumps(entry["counts"]["symbol_resolution"]))
        if "lines" in entry["counts"]:
            print("  lines: " + json.dumps(entry["counts"]["lines"]))
        print(f"  wrote {src.derived}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
