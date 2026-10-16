"""T120 -- the per-context expressed-factor family from each context's control cells: the expressed RBP and
miRNA-machinery set, the per-gene motif sites split by expressed and silent factors, the DepMap check; prints
the counts per context (final-phase: controls).

    .venv/bin/python scripts/t120_build_control_factors.py                        # every vcc2026/context_*.h5ad
    .venv/bin/python scripts/t120_build_control_factors.py --h5ad ~/data/sidechain/vcc2026/context_D.h5ad --h5ad ...
    .venv/bin/python scripts/t120_build_control_factors.py --no-depmap            # skip the DepMap rank check

Reads, under ~/data/sidechain/: vcc2026/context_<X>.h5ad, derived/cisbp-rna-2.0/{rbp_motifs,rbp_site_counts}.parquet,
external/figshare-27993248/{Model.csv,OmicsExpressionProteinCodingGenesTPMLogp1.csv}; and configs/manifests/
encode_eclip.json. Writes derived/control-factor-sets/{factor_expression,ctx_site_counts}.parquet, depmap_check.json
and a LINEAGE.json entry. A minute on the Mac; then rebuild the view with scripts/t120_build_ptr_table.py.

The rules and the columns: src/sidechain/priors/ctx_factors.py; the brief: private
handoffs/2026-10-16-t120-control-cell-factor-set.md; the results: private research/ideas/ptr-feature-table.md § Outcome.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sidechain.priors.ctx_factors import DERIVED, write_control_factors
from sidechain.priors.posttx_mirna import DATA_ROOT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad", action="append", default=None, help="a context_<X>.h5ad; repeatable; default: all")
    ap.add_argument("--no-depmap", action="store_true")
    ap.add_argument("--root", default=str(DATA_ROOT))
    args = ap.parse_args(argv)
    out = write_control_factors(Path(args.root), h5ads=args.h5ad, depmap=not args.no_depmap)
    lineage = json.loads((out / "LINEAGE.json").read_text())["entries"]["control-factor-sets/ctx_site_counts"]
    for ctx, c in lineage["contexts"].items():
        print(f"context {ctx}: {c['cells']} cells; {c['expressed']} of {c['factors_on_axis']} factors expressed; "
              f"direct RBPs expressed {c['direct_rbps_expressed']}, silent {c['direct_rbps_silent']}, off-axis "
              f"{c['direct_rbps_off_axis']}; miRNA machinery {', '.join(c['mirna_machinery_expressed'])}; "
              f"median expressed share of a gene's sites {c['median_frac_sites_expressed']:.2f}; variable RBPs on "
              f"here {len(c['variable_rbps_expressed'])} of {len(lineage['variable_direct_rbps'])}")
    check = lineage["depmap_check"]
    if "contexts" in check:
        for ctx, e in check["contexts"].items():
            named = f"{e['named_line']} ranks {e['named_rank']} of {check['models']} (rho {e['named_rho']})" \
                if e.get("named_line") else "no name yet"
            print(f"  DepMap check {ctx}: {named}; top: " +
                  ", ".join(f"{t['line']} {t['rho']:.2f}" for t in e["top"]))
    else:
        print(f"  DepMap check: {check}")
    print(f"wrote {out} ({DERIVED})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
