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
from datetime import UTC, datetime
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


RESTORE = """# Restoring `{instance}` without lamindb

Exported {when} from `{instance}`, storage root `{root}`.
{n_artifacts} artifacts, {total_gb:.2f} GB, {n_runs} runs, {n_transforms} transforms.

The bytes in that bucket are named by uid, not by path. `manifest.csv` is the mapping.

1. **Get credentials.** `s3://lamin-us-west-2` is *Lamin's* bucket, not ours — your own
   AWS credentials get `NoCredentialsError`. The hub vends short-lived federated STS
   credentials for our prefix, and that dependency (the hub answering, the account in
   good standing) is the real lock-in surface, not the filenames:

   ```python
   from lamindb_setup.core._hub_core import access_aws
   from lamindb_setup.core._settings import settings
   print(access_aws("{root}", access_token=settings.user.access_token))
   ```

   Export the three values as `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
   `AWS_SESSION_TOKEN`. They are ordinary AWS credentials and expire in the hour.

2. **Sync the bytes.** `aws s3 sync {root}/ ./blobs/`

3. **Rename them into the tree.** For each row of `manifest.csv` with a non-empty `key`,
   copy `blobs/<storage_object>` to `<data root>/<key>` (a directory for `n_files > 0`),
   then check `hash` with `lamindb_setup.core.hashing.hash_file` / `hash_dir` (md5 up to
   50 MB, `sha1-fl` above, `md5-d` for folders).

   Rows to expect and skip: `key` empty — lamindb's own run-log `.txt` artifacts, a few
   KB, no local counterpart; and `is_latest` false — superseded versions, which for a
   *folder* artifact are not even readable, since folder versions share one storage key.

Step 3 is usually unnecessary: the artifacts we register are keyed by their path under
`~/data/sidechain/`, and the Mac is the authoritative copy. This exists for the case
where it is not — a box registered something and was deleted before the Mac pulled it.

**Not exported:** the bytes themselves (that is step 2), and `external/`, which is not
in the instance at all — its recovery story is the upstream host plus `PROVENANCE.json`.

**Also worth trying, before hand-rolling any of this:** `lamin io snapshot` builds a
SQLite clone of the instance and reconnects as the root DB user, so it may succeed where
`lamin io exportdb` fails. Untested on this instance as of the export date.
"""


def export_registries(out: Path | None = None) -> dict:
    """Dump every registry to CSV plus a uid→key manifest. ADR 0007 §7.

    Assumes a connected instance. Returns a summary dict; writes `artifacts.csv`,
    `runs.csv`, `transforms.csv`, `storages.csv`, `manifest.csv`, `RESTORE.md` and
    `export_meta.json` under `out` (default `<data root>/lamin_export`).

    Lives here rather than in the script because `lamin_register.py` calls it after every
    upload: the catalogue is only useful if it is never behind the bucket, and a schedule
    a human keeps is exactly the thing that silently stops being kept.

    Built by ITERATING each queryset, not via `to_dataframe()`. Two reasons, both measured
    2026-08-28 against this instance: `to_dataframe()` truncates to 20 rows unless you pass
    `limit=None`, and even then it returned 43 of 55 artifacts -- it drops keyless records
    (lamindb's own source-code snapshots). An export that silently omits rows is worse than
    no export, because it looks complete.
    """
    import lamindb as ln
    import pandas as pd

    out = (out or data_root() / "lamin_export").expanduser()
    out.mkdir(parents=True, exist_ok=True)

    counts = {}
    for name, model in {"artifacts": ln.Artifact, "runs": ln.Run,
                        "transforms": ln.Transform, "storages": ln.Storage}.items():
        rows = [{k: v for k, v in rec.__dict__.items() if not k.startswith("_")}
                for rec in model.filter()]
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
        counts[name] = len(rows)

    manifest = pd.DataFrame([
        {"uid": a.uid, "key": a.key, "hash": a.hash, "size": a.size, "n_files": a.n_files,
         "is_latest": a.is_latest, "version": a.version_tag, "description": a.description,
         # The whole point of the manifest: uid-named object -> our path.
         "storage_object": str(a.path)}
        for a in ln.Artifact.filter()
    ])
    manifest.to_csv(out / "manifest.csv", index=False)

    storage = ln.Storage.filter().first()
    total = int(manifest["size"].fillna(0).sum()) if len(manifest) else 0
    (out / "RESTORE.md").write_text(RESTORE.format(
        instance=instance(), when=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        root=storage.root if storage else "?", n_artifacts=counts.get("artifacts", 0),
        total_gb=total / 1e9, n_runs=counts.get("runs", 0),
        n_transforms=counts.get("transforms", 0)))

    import json

    (out / "export_meta.json").write_text(json.dumps({
        "instance": instance(), "exported_at": datetime.now(UTC).isoformat(),
        "storage_root": storage.root if storage else None,
        "lamindb_version": ln.__version__, "counts": counts, "total_bytes": total,
    }, indent=1) + "\n")
    return {"out": out, "counts": counts, "total_bytes": total, "n_manifest": len(manifest)}
