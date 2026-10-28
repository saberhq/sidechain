"""Pull registered artifacts out of the hosted lamindb instance into the local tree, each verified by its hash.

    uv run python scripts/lamin_pull.py derived/<source>/<file>.npz
    uv run python scripts/lamin_pull.py --prefix runs/mirror/<fold>/
    uv run python scripts/lamin_pull.py --prefix derived/<source>/ --dry-run

The inverse of `scripts/lamin_register.py`: key `derived/x.npz` lands at `~/data/sidechain/derived/x.npz`, written
with `download_to`, re-hashed with lamindb's hasher and compared to the registry's `hash`; a mismatch exits non-zero.
Reads the connected instance; writes the pulled files under the data root. Notes: private/code-notes/scripts/lamin_pull.py.md
"""
from __future__ import annotations

import argparse

from sidechain.utils.lamin import hash_local, instance, local_path, prepare_dest


def _human(n: int | None) -> str:
    size = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:,.1f} TB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("keys", nargs="*", help="artifact keys, i.e. paths under ~/data/sidechain")
    ap.add_argument("--prefix", help="pull every latest artifact whose key starts with this")
    ap.add_argument("--dry-run", action="store_true", help="list what would be pulled, and its size")
    ap.add_argument("--force", action="store_true",
                    help="re-place even if the local file already matches the registry hash")
    args = ap.parse_args(argv)

    if not args.keys and not args.prefix:
        ap.error("give at least one key, or --prefix")

    import lamindb as ln

    if inst := instance():
        ln.connect(inst)

    artifacts = []
    for key in args.keys:
        try:
            local_path(key)  # reject a traversing or absolute key before touching the hub
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        art = ln.Artifact.filter(key=key, is_latest=True).one_or_none()
        if art is None:
            raise SystemExit(f"no artifact with key {key!r} in {inst} (registered yet?)")
        artifacts.append(art)
    if args.prefix:
        found = list(ln.Artifact.filter(key__startswith=args.prefix, is_latest=True))
        if not found:
            raise SystemExit(f"no artifacts under prefix {args.prefix!r} in {inst}")
        artifacts.extend(found)

    total = sum(a.size or 0 for a in artifacts)
    print(f"{len(artifacts)} artifact(s), {_human(total)}")
    try:
        dests = {art.key: local_path(art.key) for art in artifacts}
    except ValueError as exc:  # a key from the hub is remote data; never write outside the tree
        raise SystemExit(str(exc)) from None

    if args.dry_run:
        for art in artifacts:
            print(f"  {_human(art.size):>12}  {art.key} -> {dests[art.key]}")
        return 0

    for art in artifacts:
        dest = dests[art.key]
        if dest.exists() and not args.force and hash_local(dest) == art.hash:
            print(f"current  {art.key}  ({_human(art.size)})")
            continue
        art.path.download_to(prepare_dest(dest))
        got = hash_local(dest)
        if got != art.hash:
            raise SystemExit(
                f"HASH MISMATCH for {art.key}: registry {art.hash}, pulled {got}. "
                f"Left at {dest} for inspection; do not use it."
            )
        print(f"pulled   {art.key}  ({_human(art.size)}, hash ok)  -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
