"""Dump the hosted lamindb instance's registries to local CSV, with the uid -> key manifest.

    uv run python scripts/lamin_export.py                 # -> ~/data/sidechain/lamin_export/
    uv run python scripts/lamin_export.py --out /somewhere/else

`manifest.csv` maps each uid-named storage object back to its key, the path under the data root.
Reads the connected instance; writes the CSVs, RESTORE.md and export_meta.json (`utils.lamin.export_registries`).
Notes: private/code-notes/scripts/lamin_export.py.md
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
