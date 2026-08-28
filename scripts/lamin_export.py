"""Dump the hosted instance's registries to local CSV — the exit plan, kept warm.

    uv run python scripts/lamin_export.py                 # -> ~/data/sidechain/lamin_export/
    uv run python scripts/lamin_export.py --out /somewhere/else

ADR 0007 §7. The bucket holds our bytes, but **lamindb stores them under virtual keys**:
a single-file artifact is `.lamindb/<uid><suffix>`, and the mapping from that uid back to
`derived/lamin-pertdata/sunshine23_all_pseudobulk.npz` lives only in the instance's
Postgres. (Folder artifacts are less opaque — they land under `.lamindb/<uid[:16]>/` with
their member filenames intact — but the folder's own identity is still just a uid.) So a
bytes-only extraction returns correctly-hashed, correctly-sized, mostly unidentifiable
objects. This script is the other half: run it, and `manifest.csv` beside the bytes turns
them back into the tree. Run it *before* you need it; a subscription that has lapsed may
take the Postgres with it, and nothing published commits to a grace period.

It also captures what is genuinely hub-only rather than a copy of something local: the
Run and Transform records — which code, which config, which metrics, for every scored
run. The bytes are all duplicates of `~/data/sidechain/`; the lineage is not.

lamindb ships `lamin io exportdb` for this, and it **does not work on our instance**:
its Postgres path uses `COPY … TO STDOUT`, which bypasses the wrapper that injects the
row-level-security JWT, so it dies with `JWT is not set` (verified 2026-08-28). The ORM
path used here works fine. When upstream fixes that, this script can shrink to a call.

CSV on purpose. This is the format that outlives the tools that wrote it.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from sidechain.utils.lamin import data_root, instance

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: <data root>/lamin_export)")
    ap.add_argument("--no-paths", action="store_true",
                    help="skip resolving each artifact's S3 object path (one query per "
                         "artifact; the manifest is much less useful without it)")
    args = ap.parse_args(argv)

    out = (args.out or data_root() / "lamin_export").expanduser()
    out.mkdir(parents=True, exist_ok=True)

    import lamindb as ln

    inst = instance()
    if inst:
        ln.connect(inst)

    import pandas as pd

    registries = {"artifacts": ln.Artifact, "runs": ln.Run,
                  "transforms": ln.Transform, "storages": ln.Storage}
    counts = {}
    for name, model in registries.items():
        # Built by ITERATING, not via `to_dataframe()`. Two reasons, both measured
        # 2026-08-28 against this instance: `to_dataframe()` truncates to 20 rows
        # unless you pass `limit=None`, and even then it returned 43 of 55 artifacts --
        # it drops keyless records (lamindb's own source-code snapshots). An export
        # that silently omits rows is worse than no export, because it looks complete.
        rows = [{k: v for k, v in rec.__dict__.items() if not k.startswith("_")}
                for rec in model.filter()]
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
        counts[name] = len(rows)
        print(f"{name}.csv  {len(rows)} rows")

    rows = []
    for art in ln.Artifact.filter():
        row = {"uid": art.uid, "key": art.key, "hash": art.hash, "size": art.size,
               "n_files": art.n_files, "is_latest": art.is_latest,
               "version": art.version_tag, "description": art.description}
        if not args.no_paths:
            # The whole point of the manifest: uid-named object -> our path.
            row["storage_object"] = str(art.path)
        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    print(f"manifest.csv  {len(manifest)} rows")

    storage_root = ln.Storage.filter().first()
    total = int(manifest["size"].fillna(0).sum()) if len(manifest) else 0
    (out / "RESTORE.md").write_text(RESTORE.format(
        instance=inst, when=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        root=storage_root.root if storage_root else "?",
        n_artifacts=counts.get("artifacts", 0), total_gb=total / 1e9,
        n_runs=counts.get("runs", 0), n_transforms=counts.get("transforms", 0)))
    (out / "export_meta.json").write_text(json.dumps({
        "instance": inst, "exported_at": datetime.now(UTC).isoformat(),
        "storage_root": storage_root.root if storage_root else None,
        "lamindb_version": ln.__version__, "counts": counts, "total_bytes": total,
    }, indent=1) + "\n")
    print(f"\nexit plan refreshed in {out}  ({total / 1e9:.2f} GB catalogued)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
