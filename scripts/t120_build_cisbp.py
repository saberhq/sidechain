"""T120 -- CisBP-RNA 2.0 through the gate, its motif table, and the per-gene RBP site counts over
TargetScan's representative 3'UTRs; prints the counts.

    .venv/bin/python scripts/t120_build_cisbp.py              # fetch (idempotent), build what is missing
    .venv/bin/python scripts/t120_build_cisbp.py --rebuild    # rebuild both tables
    .venv/bin/python scripts/t120_build_cisbp.py --rel 0.90   # a stricter twin beside the default (suffix _r090)

Reads, under ~/data/sidechain/: external/cisbp-rna-2.0/ (the archive the gate lands, block cisbp_rna_2 in
configs/data_sources.yaml) and external/targetscan-vert_80/{UTR_Sequences.txt,Gene_info.txt}.zip. Writes
derived/cisbp-rna-2.0/{rbp_motifs,pfms,rbp_site_counts}.parquet and LINEAGE.json entries. About fifteen
minutes on the Mac (two scans of 586 matrices over 37 Mnt, the second on shuffled sequences).

The scan and the columns: src/sidechain/priors/posttx_cisbp.py; the results: private
research/ideas/ptr-feature-table.md § Outcome; the task: private tasks/T120.md.
"""
from __future__ import annotations

import argparse
import sys
import time

from sidechain.priors.posttx_cisbp import CISBP_BLOCK, CisBPRNASource
from sidechain.priors.posttx_mirna import DATA_ROOT, spec_from_registry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--rel", type=float, default=None, help="relative-score threshold; a non-default one writes "
                    "rbp_site_counts_r<NNN>.parquet beside the default table")
    ap.add_argument("--root", default=str(DATA_ROOT))
    args = ap.parse_args(argv)
    src = CisBPRNASource(spec_from_registry(CISBP_BLOCK), {}, root=args.root)
    t0 = time.time()
    dest = src.fetch(progress=True)
    print(f"archive at {dest}")
    motifs = src.motif_table(rebuild=args.rebuild)
    print(f"motif rows {len(motifs)}, RBPs {motifs['rbp_id'].nunique()}, "
          f"direct {motifs.loc[motifs['status'] == 'D', 'rbp_id'].nunique()}, "
          f"inferred-only {motifs.loc[motifs['status'] == 'I', 'rbp_id'].nunique()}, "
          f"matrices {motifs['motif_id'].nunique()}  ({time.time() - t0:.0f}s)")
    kw = {} if args.rel is None else {"rel_score_min": args.rel}
    genes = src.gene_table(rebuild=args.rebuild, progress=True, **kw)
    rb = [c for c in genes.columns if c.startswith("rb_") and c not in
          ("rb_sites_direct", "rb_n_rbps_direct", "rb_sites_inferred", "rb_sites_direct_null")]
    print(f"gene rows {len(genes)}, RBP columns {len(rb)}, median direct sites "
          f"{genes['rb_sites_direct'].median():.0f}, median direct RBPs with a site "
          f"{genes['rb_n_rbps_direct'].median():.0f}  ({time.time() - t0:.0f}s)")
    print(f"wrote {src.derived}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
