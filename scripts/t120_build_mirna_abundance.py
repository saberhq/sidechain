"""T120 -- the three measured per-line miRNA-abundance sources through the gate and onto disk (phase 3): the CCLE
2019 NanoString panel, ENCODE microRNA-seq and miRNATissueAtlas 2025, each as a per-line family-abundance table and
a per-gene TargetScan site load weighted by the line's repertoire; prints the counts.

    .venv/bin/python scripts/t120_build_mirna_abundance.py                 # fetch (idempotent), build what is missing
    .venv/bin/python scripts/t120_build_mirna_abundance.py --rebuild       # rebuild the tables
    .venv/bin/python scripts/t120_build_mirna_abundance.py --only ccle_mirna_2019

Reads, under ~/data/sidechain/: the files the three blocks of configs/data_sources.yaml name (external/broad-ccle-
mirna-2019, external/encode-mirna-seq, external/mirnatissueatlas-2025), external/targetscan-vert_80/miR_Family_Info.txt.zip,
derived/targetscan-vert_80/edges_default.parquet and external/gencode-v47/gencode.v47.basic.annotation.gtf.gz. Writes
derived/<source>/{mirna_family_abundance,mirna_load}.parquet and LINEAGE.json entries. A few minutes on the Mac (the
atlas h5ad is 1 GB); then rebuild the view with scripts/t120_build_ptr_table.py.

The parsers, the name-to-family rule and the load: src/sidechain/priors/posttx_mirna_abundance.py; the results:
private research/ideas/ptr-feature-table.md § Outcome; the task: private tasks/T120.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from sidechain.priors.posttx_mirna import DATA_ROOT
from sidechain.priors.posttx_mirna_abundance import BLOCKS, abundance_sources


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--only", action="append", choices=BLOCKS, default=None, help="one block; repeatable")
    ap.add_argument("--root", default=str(DATA_ROOT))
    args = ap.parse_args(argv)
    t0 = time.time()
    for src in abundance_sources(args.root):
        if args.only and src.name not in args.only:
            continue
        dest = src.fetch(progress=True)
        table = src.gene_table(rebuild=args.rebuild)
        lineage = json.loads((src.derived / "LINEAGE.json").read_text())["entries"]
        ab = lineage[f"{src.derived.name}/mirna_family_abundance"]["counts"]
        ld = lineage[src.lineage_key]["counts"]
        print(f"{src.name}: files at {dest}; {len(table)} genes, {len(table.columns) - 2} columns  ({time.time() - t0:.0f}s)")
        print("  names -> families: " + json.dumps(ab["name_mapping"]) + f"; families measured {ab['families_measured']}; "
              f"lines {ab['lines']}")
        print("  share of signal mapped: " + json.dumps({k: round(v, 3) for k, v in ab["share_of_signal_mapped"].items()}))
        print("  per line: " + json.dumps({l: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()}
                                           for l, c in ld["per_line"].items()}))
        print("  line~line family Spearman: " + json.dumps({k: round(v, 3) for k, v in ld["line_family_spearman"].items()}))
        print(f"  wrote {src.derived}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
