"""Register local artifacts into the hosted lamindb instance, keyed by their
path under ~/data/sidechain so the instance mirrors the local tree.

    uv run python scripts/lamin_register.py \
        ~/data/sidechain/derived/lamin-pertdata/sunshine23_all_pseudobulk.npz \
        ~/data/sidechain/derived/lamin-pertdata/sunshine23_all_pseudobulk.lineage.json \
        --description "..."

Uploads go to the instance's default storage (the Lamin-managed us-west-2
bucket), which is the whole point: a Brev box later runs
`ln.Artifact.get(key=...).cache()` and pulls from S3 in-region instead of the
Mac uplink shipping the same file for the third time that week.

The instance comes from SIDECHAIN_LAMIN_INSTANCE (default saberhq/sidechain).
`ln.connect()` is process-local (verified 2026-08-27: a fresh process still
sees none/none), so running this never changes machine state for the other
sessions in this checkout.

Registration is tracked (`ln.track()`), so the instance records which script
version produced the upload. Unlike `utils/logging.log_run` this is NOT
non-fatal -- registering an artifact is the task here, not a side effect, so
a failure should fail loudly.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_INSTANCE = "saberhq/sidechain"
DATA_ROOT = Path.home() / "data" / "sidechain"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--description", help="one line, applied to every artifact given")
    ap.add_argument("--key-prefix", default=None,
                    help="override the derived key prefix; default keys each file by its "
                         "path relative to ~/data/sidechain")
    args = ap.parse_args(argv)

    import lamindb as ln

    ln.connect(os.environ.get("SIDECHAIN_LAMIN_INSTANCE") or DEFAULT_INSTANCE)
    ln.track()
    for path in args.paths:
        p = path.expanduser().resolve()
        # A directory registers as ONE folder artifact (lamindb handles the
        # tree) -- the mirror bundles are the case: seven small files that are
        # only meaningful together.
        if not p.exists():
            raise SystemExit(f"{p} does not exist")
        if args.key_prefix:
            key = f"{args.key_prefix.rstrip('/')}/{p.name}"
        else:
            try:
                key = str(p.relative_to(DATA_ROOT))
            except ValueError:
                raise SystemExit(
                    f"{p} is outside {DATA_ROOT}; pass --key-prefix to key it explicitly"
                ) from None
        art = ln.Artifact(str(p), key=key, description=args.description).save()
        print(f"registered {key}  uid={art.uid}  size={art.size}  hash={art.hash}")
    ln.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
