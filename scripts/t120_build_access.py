"""T120 -- RNAplfold accessibility of TargetScan's representative 3'UTRs, per gene and around the conserved
miRNA sites and consensus RBP motifs; prints the counts.

    .venv/bin/python scripts/t120_build_access.py --workers 8

Reads, under ~/data/sidechain/: external/targetscan-vert_80/{UTR_Sequences,Gene_info}.txt.zip and
derived/targetscan-vert_80/sites.parquet. Writes derived/rnaplfold-access/utr_accessibility.parquet and a
LINEAGE.json entry. About 46 CPU-minutes (37.5 Mnt at 0.074 s per kb, measured 2026-10-16); ten minutes with
eight workers.

The folding parameters and the columns: src/sidechain/priors/posttx_access.py; the results: private
research/ideas/ptr-feature-table.md § Outcome; the task: private tasks/T120.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from sidechain.priors.posttx_access import DERIVED, LINEAGE_KEY, write_accessibility
from sidechain.priors.posttx_mirna import DATA_ROOT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--root", default=str(DATA_ROOT))
    args = ap.parse_args(argv)
    t0 = time.time()
    out = write_accessibility(Path(args.root), workers=args.workers, progress=True)
    c = json.loads((out / "LINEAGE.json").read_text())["entries"][LINEAGE_KEY]["counts"]
    print(f"transcripts {c['transcripts']}, bases folded {c['bases_folded']}, conserved sites {c['conserved_sites_on_representatives']} "
          f"on {c['transcripts_with_a_site']} transcripts; median per-nt accessibility {c['median_acc_mean_u1']:.3f}, "
          f"median site excess {c['median_site_excess']:+.3f}  ({time.time() - t0:.0f}s)")
    print(f"wrote {out} ({DERIVED})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
