"""T120 -- build the wide per-gene post-transcriptional (PTR) feature table and its column
dictionary as a view over the built per-source tables, and print the coverage counts.

    .venv/bin/python scripts/t120_build_ptr_table.py

Reads, under ~/data/sidechain/: vcc2026/gene_names.csv (the axis), runs/probes/t102_utr_load/bridge/
symbol_to_ensg.json (the symbol -> Ensembl bridge), and the per-source parquets and LINEAGE.json
files named in sidechain.priors.ptr_table.SOURCES (derived/targetscan-vert_80, derived/halflife-
agarwal-kelley-2022, derived/codon-optimality-wu2019, derived/polyasite-2.0). Writes
derived/ptr-feature-table/ptr_features.parquet, ptr_columns.parquet, ptr_columns.json and a
LINEAGE.json entry. Minutes on the Mac.

The join and the dictionary: src/sidechain/priors/ptr_table.py; the plan: private
plans/2026-10-15-t102-ptr-feature-table.md; the results: private research/ideas/ptr-feature-table.md § Outcome.
"""
from __future__ import annotations

import sys

from sidechain.priors.ptr_table import main

if __name__ == "__main__":
    sys.exit(main())
