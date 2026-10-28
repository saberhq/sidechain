"""Keys and local paths for the lamindb mirror of `~/data/sidechain/`: an artifact's key is its path under the data root.

    key = artifact_key(path); path = local_path(key)     # inverse maps; anything outside the root raises
    hash_local(path) == artifact.hash                    # lamindb's own hasher: md5, sha1-fl over 50 MB, md5-d

Used by `scripts/lamin_register.py` (path -> key), `scripts/lamin_pull.py` (key -> path) and `utils.logging.log_run`.
`export_registries` writes the registries as CSV plus the uid -> key manifest under `<data root>/lamin_export`.
Notes: private/code-notes/src/sidechain/utils/lamin.py.md
"""
from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

# Saved files already sit at their key: this skips lamindb's save() copy into its cache.
# `1` trips only that check; lamindb's other two uses test for the string `true`.
os.environ.setdefault("LAMINDB_MULTI_INSTANCE", "1")

DEFAULT_INSTANCE = "saberhq/sidechain"
_DEFAULT_DATA_ROOT = Path.home() / "data" / "sidechain"


def data_root() -> Path:
    """`~/data/sidechain`, or `SIDECHAIN_DATA_ROOT`, read per call."""
    return Path(os.environ.get("SIDECHAIN_DATA_ROOT") or _DEFAULT_DATA_ROOT).expanduser()


def instance() -> str:
    """The instance to connect to. Empty string means "don't connect" (see `log_run`)."""
    return os.environ.get("SIDECHAIN_LAMIN_INSTANCE", DEFAULT_INSTANCE)


def artifact_key(path: str | Path) -> str:
    """The instance key for a local path: its path relative to the data root; ValueError outside it."""
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
    """The local path a key maps back to, the inverse of `artifact_key`; ValueError outside the data root."""
    if not key or key.startswith("/") or Path(key).is_absolute():
        raise ValueError(f"key {key!r} must be a relative path under the data root")
    root = data_root().resolve()
    dest = (root / key).resolve()
    if dest != root and root not in dest.parents:
        raise ValueError(f"key {key!r} resolves to {dest}, outside {root}")
    return dest


def hash_local(path: Path) -> str:
    """lamindb's own hash of a local file or directory, for comparing to `Artifact.hash`."""
    from lamindb_setup.core.hashing import hash_dir, hash_file

    if path.is_dir():
        return hash_dir(path)[1]
    return hash_file(path)[1]


def prepare_dest(dest: Path) -> Path:
    """Remove whatever is at `dest` and create its parent, so a download never merges into old files; returns it."""
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

**A second route, tested 2026-09-25:** `lamin io snapshot --no-upload --no-track`, run from a
scratch directory, reconnects as the root DB user and copies every registry record into a local
SQLite clone (`./<name>-clone/`), checking each table's count. `lamin io exportdb` does not work
on this instance (`JWT is not set`).
"""


def export_registries(out: Path | None = None) -> dict:
    """Dump every registry to CSV plus a uid -> key manifest, iterating each queryset in full; returns a summary.

    Assumes a connected instance. Writes `artifacts.csv`, `runs.csv`, `transforms.csv`, `storages.csv`,
    `manifest.csv`, `RESTORE.md` and `export_meta.json` under `out` (default `<data root>/lamin_export`).
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
         # uid-named object -> our path.
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
