"""T102 / T120 -- bring PolyASite 2.0 (streamed) and GENCODE v47 (downloaded) through the ADR 0003
gate and build the per-gene polyadenylation layout, printing each step's size and time.

    .venv/bin/python scripts/t102_build_polyasite.py              # fetch what is missing, build what is missing
    .venv/bin/python scripts/t102_build_polyasite.py --rebuild    # ignore the cached parquets (streams the atlas again)
    .venv/bin/python scripts/t102_build_polyasite.py --refresh    # accept a changed block or upstream state

Reads: the `polyasite_2`, `gencode_v47_basic` and `targetscan` blocks of configs/data_sources.yaml;
TargetScan's Gene_info.txt.zip under external/targetscan-vert_80/. Writes under ~/data/sidechain/:
external/gencode-v47/ (the GTF, PROVENANCE.json), external/polyasite-2.0/PROVENANCE.json (nothing
else lands), derived/gencode-v47/transcript_structure.parquet, derived/polyasite-2.0/pas_gene_features.parquet,
each with a LINEAGE.json entry, plus derived/polyasite-2.0/build_summary.json. Runs on the Mac in minutes.

The columns and the window rules: src/sidechain/priors/posttx_polya.py; the plan: private
plans/2026-10-15-t102-polyasite.md; the results: private research/ideas/ptr-feature-table.md § Outcome.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from sidechain.priors.posttx_mirna import spec_from_registry
from sidechain.priors.posttx_polya import (
    GENCODE_BLOCK,
    POLYASITE_BLOCK,
    PolyASiteSource,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="accept a changed block / upstream state")
    ap.add_argument("--rebuild", action="store_true", help="ignore cached parquets")
    ap.add_argument("--skip-fetch", action="store_true")
    args = ap.parse_args()
    atlas = PolyASiteSource(spec_from_registry(POLYASITE_BLOCK), {})
    gencode = PolyASiteSource(spec_from_registry(GENCODE_BLOCK), {})
    summary = {"steps": []}

    def step(name, fn):
        t0 = time.time()
        out = fn()
        n = None if out is None else (len(out) if hasattr(out, "__len__") and not isinstance(out, str) else out)
        rec = {"step": name, "rows": n if not isinstance(n, str) else None, "seconds": round(time.time() - t0)}
        summary["steps"].append(rec)
        print(f"  {name:24s} {n!s:>60s}  {rec['seconds']:6d}s", flush=True)
        return out

    if not args.skip_fetch:
        print("gate", flush=True)
        step("fetch gencode", lambda: str(gencode.fetch(refresh=args.refresh, progress=True)))
        step("gate polyasite", lambda: str(atlas.fetch(refresh=args.refresh)))
    print("tables", flush=True)
    step("transcript_structure", lambda: gencode.transcript_table(rebuild=args.rebuild))
    step("pas_gene_features", lambda: atlas.gene_table(rebuild=args.rebuild, progress=True))
    lineage = json.loads((atlas.derived / "LINEAGE.json").read_text())
    entry = lineage["entries"]["polyasite-2.0/pas_gene_features"]
    summary["counts"] = entry["counts"]
    summary["atlas_sha256"] = entry["inputs"]["atlas.clusters.2.0.GRCh38.96.tsv.gz"]["sha256"]
    (atlas.derived / "build_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(entry["counts"], indent=1))
    print(f"wrote {atlas.derived / 'build_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
