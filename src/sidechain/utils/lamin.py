"""The Mac↔box data plane: keys, local paths, and the one rule that ties them.

ADR 0007. The hosted instance is a **cloud mirror of the local tree**, not a second
naming scheme: an artifact's key *is* its path under `~/data/sidechain/`. Everything
here exists to keep that identity exact in both directions —
`scripts/lamin_register.py` maps path → key, `scripts/lamin_pull.py` maps key → path,
and `utils.logging.log_run` uses the same rule so a run's own outputs land where the
tree says they live.

That single rule is also the exit plan, with one caveat worth stating here because it
is invisible from the bucket: lamindb stores objects under *virtual* keys, so the S3
side is `.lamindb/<uid>.npz`, not our path. The key→uid mapping lives only in the hub's
Postgres, which is why `scripts/lamin_export.py` dumps it on a schedule. With that dump
in hand the bucket is a literal copy of the tree and leaving costs a sync and a rename;
without it, an extraction is a pile of opaque blobs. Nothing downstream reads a lamin
key — the code reads local paths — so the data plane can be swapped for plain S3, GCS,
or `brev copy` without touching a model.

`hash_local` deliberately reuses lamindb's own hasher rather than plain md5, because a
pull that "verified" against a different digest than the registry stores would be a
check that always passes. Be honest about what it proves: over 50 MB lamindb switches
to `sha1-fl` (first and last chunk only), so the middle of a 6.9 GB npz is never
hashed. It catches a truncated or wrong-artifact download, not silent bit rot.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_INSTANCE = "saberhq/sidechain"
_DEFAULT_DATA_ROOT = Path.home() / "data" / "sidechain"


def data_root() -> Path:
    """`~/data/sidechain`, or `SIDECHAIN_DATA_ROOT`.

    Read per call rather than bound at import: tests point it at a tmp_path, and a
    box's home is not the Mac's.
    """
    return Path(os.environ.get("SIDECHAIN_DATA_ROOT") or _DEFAULT_DATA_ROOT).expanduser()


def instance() -> str:
    """The instance to connect to. Empty string means "don't connect" (see `log_run`)."""
    return os.environ.get("SIDECHAIN_LAMIN_INSTANCE", DEFAULT_INSTANCE)


def artifact_key(path: str | Path) -> str:
    """The instance key for a local path: its path relative to the data root.

    Raises ValueError for anything outside the tree — a file with no place in the
    mirror has no key, and inventing one (`runs/<basename>`, say) is how five
    unrelated runs became five *versions* of one `runs/summary.json` on 2026-08-28.
    """
    p = Path(path).expanduser().resolve()
    root = data_root().resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        raise ValueError(
            f"{p} is outside {root}: it has no key in the mirror. Move it under the "
            f"data root, or pass an explicit --key-prefix."
        ) from None


def local_path(key: str) -> Path:
    """The local path a key maps back to — the inverse of `artifact_key`.

    A key read back from the hub is remote data, so `../../.ssh/id_rsa` is treated as
    hostile input rather than a typo: anything resolving outside the data root raises.
    """
    if not key or key.startswith("/") or Path(key).is_absolute():
        raise ValueError(f"key {key!r} must be a relative path under the data root")
    root = data_root().resolve()
    dest = (root / key).resolve()
    if dest != root and root not in dest.parents:
        raise ValueError(f"key {key!r} resolves to {dest}, outside {root}")
    return dest


def hash_local(path: Path) -> str:
    """lamindb's own hash of a local file or directory, for comparing to `Artifact.hash`.

    Files under 50 MB hash as md5; larger ones as `sha1-fl` (first+last chunk), and a
    directory as `md5-d` over its members — so a plain md5 would disagree with the
    registry on every large artifact we have. Delegating to `lamindb_setup` keeps the
    two definitions from drifting apart.
    """
    from lamindb_setup.core.hashing import hash_dir, hash_file

    if path.is_dir():
        return hash_dir(path)[1]
    return hash_file(path)[1]


def prepare_dest(dest: Path) -> Path:
    """Make `dest` ready to receive a downloaded artifact, and return it.

    Clearing first matters for folder artifacts: `download_to` is a recursive *merge*,
    so a re-pull into a directory that still holds a member the new version dropped
    leaves a tree that is neither version — and `hash_dir` hashes every member it
    finds, so the mismatch would surface as an unexplained failure rather than as the
    stale file it is.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_dir():
        shutil.rmtree(dest)
    elif dest.exists():
        dest.unlink()
    return dest
