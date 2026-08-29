"""Register local artifacts into the hosted lamindb instance, keyed by their
path under ~/data/sidechain so the instance mirrors the local tree.

    uv run python scripts/lamin_register.py \
        ~/data/sidechain/derived/lamin-pertdata/sunshine23_all_pseudobulk.npz \
        ~/data/sidechain/derived/lamin-pertdata/sunshine23_all_pseudobulk.lineage.json \
        --description "..."
    uv run python scripts/lamin_register.py --from-file paths.txt   # many paths: see CLI_ARGS_MAX

Uploads go to the instance's default storage (the Lamin-managed us-west-2
bucket), which is the whole point: a Brev box later runs
`scripts/lamin_pull.py` and pulls from S3 in-region instead of the Mac uplink
shipping the same file for the third time that week. That script is the exact
reciprocal of this one -- key -> path where this is path -> key -- and both take
the mapping from `sidechain.utils.lamin`, so the two directions cannot drift.

The instance comes from SIDECHAIN_LAMIN_INSTANCE (default saberhq/sidechain;
empty means "don't connect", the same rule `log_run` and `lamin_pull` follow).
`ln.connect()` is process-local (verified 2026-08-27: a fresh process still
sees none/none), so running this never changes machine state for the other
sessions in this checkout.

Registration is tracked (`ln.track()`), so the instance records which script
version produced the upload. Unlike `utils/logging.log_run` this is NOT
non-fatal -- registering an artifact is the task here, not a side effect, so
a failure should fail loudly.

**Directories are folder artifacts, and folder artifacts overwrite their own history.**
Every version of one shares a single storage key (`uid[:16]`), so re-registering a
changed directory under the same key replaces the previous version's bytes in S3 and
makes that version permanently unreadable. Files do not behave this way -- they version
side by side. This script refuses the destructive case unless you ask for it; there is
a preflight below, not a warning after the fact.

`--from-file` exists because of a lamindb limit found the hard way on 2026-08-28:
`ln.track()` stores the whole argv in `Run.cli_args`, a `varchar(1024)`, with no
truncation (`lamindb/core/_context.py:814`, `lamindb/models/run.py:339`). Thirteen
absolute paths on one command line overflow it and the registration dies inside
Django with `DataError: value too long for type character varying(1024)` -- a message
that names no field and no file. The preflight below turns that into a sentence, and
`--from-file` is the way to register a backlog without tripping it.
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
                         "with different contents. This DESTROYS the stored copy of the "
                         "existing version -- read the note in the module docstring first")
    args = ap.parse_args(argv)

    paths = list(args.paths)
    if args.from_file:
        paths += [Path(line.strip()) for line in args.from_file.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
    if not paths:
        ap.error("give at least one path, or --from-file")

    # Preflight, not a post-mortem: ln.track() would otherwise fail inside Django
    # AFTER connecting, with an error naming neither the field nor the fix.
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
        # A directory registers as ONE folder artifact (lamindb handles the
        # tree) -- the mirror bundles are the case: seven small files that are
        # only meaningful together.
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

    # Folder artifacts overwrite their own history: every version shares one storage
    # key (`uid[:16]`, lamindb/core/storage/paths.py:39-43), so re-registering a
    # directory whose contents changed REPLACES the stored bytes of the previous
    # version, and that version's `.cache()` raises from then on. For the mirror
    # bundles -- the 32 KB this whole ADR exists to protect -- that would silently
    # destroy the backup while printing "registered". Files are safe: they version
    # side by side.
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

    # Refresh the exit-plan catalogue here, not on a calendar. The bucket stores objects
    # under virtual keys, so `manifest.csv` is the only thing that maps a uid-named blob
    # back to `derived/…`; a catalogue that is one registration behind is a catalogue
    # that cannot restore what you just uploaded. Doing it on every write makes staleness
    # structurally impossible, which a schedule a human keeps does not.
    #
    # Non-fatal, unlike everything else here: the upload already succeeded, and a failed
    # catalogue refresh is fixed by re-running `scripts/lamin_export.py`. Failing the
    # command now would say the registration failed, which would be a lie.
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
