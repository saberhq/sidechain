"""Dump the hosted instance's registries to local CSV — the exit plan, kept warm.

    uv run python scripts/lamin_export.py                 # -> ~/data/sidechain/lamin_export/
    uv run python scripts/lamin_export.py --out /somewhere/else

ADR 0007 §7. `scripts/lamin_register.py` already calls this after every upload, so the
catalogue is never behind the bucket; run it by hand when you want it refreshed without
registering anything — before the Pro-trial decision, say.

Why it exists at all: lamindb stores objects under **virtual keys**, so the S3 object for
`derived/xatlas-orion/hct116_full.npz` is `.lamindb/xgKU70jAkvF6LUi60000.npz`, and the
mapping lives only in the instance's Postgres. A bytes-only extraction returns correct
bytes with meaningless names. `manifest.csv` is what turns them back into the tree.

lamindb ships `lamin io exportdb` for this and it **does not work on our instance**: its
Postgres path uses `COPY … TO STDOUT`, which bypasses the wrapper that injects the
row-level-security JWT, so it dies with `JWT is not set` (reproduced 2026-08-28). The ORM
path used here works fine. When upstream fixes that, this can shrink to a call.

CSV on purpose. This is the format that outlives the tools that wrote it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sidechain.utils.lamin import export_registries, instance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: <data root>/lamin_export)")
    args = ap.parse_args(argv)

    import lamindb as ln

    if inst := instance():
        ln.connect(inst)

    result = export_registries(args.out)
    for name, n in result["counts"].items():
        print(f"{name}.csv  {n} rows")
    print(f"manifest.csv  {result['n_manifest']} rows")
    print(f"\nexit plan refreshed in {result['out']}  "
          f"({result['total_bytes'] / 1e9:.2f} GB catalogued)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
