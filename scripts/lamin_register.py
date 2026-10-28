"""Register local artifacts into the hosted lamindb instance, keyed by their path under ~/data/sidechain.

    uv run python scripts/lamin_register.py ~/data/sidechain/derived/<source>/<file>.npz --description "..."
    uv run python scripts/lamin_register.py --from-file paths.txt    # many paths: argv stays under CLI_ARGS_MAX

A directory registers as one folder artifact; a changed directory over a key that holds one is refused without
`--allow-folder-overwrite` (folder versions share one storage key). `SIDECHAIN_LAMIN_INSTANCE` picks the instance.
Reads the paths given; writes the artifacts and a tracked run to the instance, then refreshes the registry export.
Notes: private/code-notes/scripts/lamin_register.py.md
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from sidechain.utils.lamin import artifact_key, data_root, export_registries, instance

# lamindb writes " ".join(sys.argv[1:]) into Run.cli_args, a varchar(1024).
CLI_ARGS_MAX = 1024


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--from-file", type=Path,
                    help="read paths from this file, one per line (# comments and blanks "
                         "ignored) -- keeps argv under lamindb's 1024-char cli_args cap")
    ap.add_argument("--description", help="one line, applied to every artifact given")
    ap.add_argument("--key-prefix", default=None,
                    help="override the derived key prefix; default keys each file by its "
                         "path relative to ~/data/sidechain")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the keys that would be written, upload nothing")
    ap.add_argument("--no-export", action="store_true",
                    help="skip refreshing the local exit-plan catalogue afterwards")
    ap.add_argument("--allow-folder-overwrite", action="store_true",
                    help="re-register a DIRECTORY under a key that already holds one "
                         "with different contents; this destroys the stored copy of the "
                         "existing version")
    args = ap.parse_args(argv)

    paths = list(args.paths)
    if args.from_file:
        paths += [Path(line.strip()) for line in args.from_file.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
    if not paths:
        ap.error("give at least one path, or --from-file")

    # Preflight: an over-long argv would fail later inside ln.track().
    cli_args = " ".join(argv if argv is not None else sys.argv[1:])
    if len(cli_args) > CLI_ARGS_MAX:
        raise SystemExit(
            f"command line is {len(cli_args)} chars; lamindb stores it in "
            f"Run.cli_args, capped at {CLI_ARGS_MAX}, and overflows with an opaque "
            f"Django DataError. Put the paths in a file and use --from-file."
        )

    plan: list[tuple[Path, str]] = []
    for path in paths:
        p = path.expanduser().resolve()
        # A directory registers as one folder artifact.
        if not p.exists():
            raise SystemExit(f"{p} does not exist")
        if args.key_prefix:
            plan.append((p, f"{args.key_prefix.rstrip('/')}/{p.name}"))
        else:
            try:
                plan.append((p, artifact_key(p)))
            except ValueError:
                raise SystemExit(
                    f"{p} is outside {data_root()}; pass --key-prefix to key it explicitly"
                ) from None

    if args.dry_run:
        for p, key in plan:
            print(f"would register {key}  ({p})")
        return 0

    import lamindb as ln

    if inst := instance():
        ln.connect(inst)

    # A folder artifact's versions share one storage key (`uid[:16]`): saving a changed
    # directory over one replaces the previous version's bytes. Files version side by side.
    for p, key in plan:
        if not p.is_dir():
            continue
        prior = ln.Artifact.filter(key=key, is_latest=True).one_or_none()
        if prior is None or args.allow_folder_overwrite:
            continue
        from sidechain.utils.lamin import hash_local

        if hash_local(p) != prior.hash:
            raise SystemExit(
                f"{key} is already registered as a FOLDER artifact with different "
                f"contents (stored hash {prior.hash}, local {hash_local(p)}). Saving "
                f"over it would destroy the stored copy of that version -- folder "
                f"versions share one storage key. Register the new one under its own "
                f"key, or pass --allow-folder-overwrite if losing the old copy is "
                f"what you mean."
            )

    ln.track()
    for p, key in plan:
        art = ln.Artifact(str(p), key=key, description=args.description).save()
        print(f"registered {key}  uid={art.uid}  size={art.size}  hash={art.hash}")
    ln.finish()

    # Refresh the registry export after every registration; a failure here only warns.
    if not args.no_export:
        try:
            result = export_registries()
            print(f"exit plan refreshed: {result['n_manifest']} artifacts catalogued in "
                  f"{result['out']}")
        except Exception as exc:  # noqa: BLE001 - see above
            warnings.warn(
                f"registration succeeded but the exit-plan export did not "
                f"({exc!r}); re-run scripts/lamin_export.py", RuntimeWarning, stacklevel=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
