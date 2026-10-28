"""Run one dataset's ingest from `configs/datasets.yaml`.

    uv run python -m sidechain.ingest.fetch --dataset scperturb_multicontext
    uv run python -m sidechain.ingest.fetch --dataset ... --check   # verify only

Probes the host, runs the gate, and writes `<root>/<dest>/PROVENANCE.json` before any file lands; then prints
the `curl` commands for missing files, or with --check verifies sizes and checksums. Downloads nothing itself.
Notes: private/code-notes/src/sidechain/ingest/fetch.py.md
"""
from __future__ import annotations

import argparse
import json
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import yaml

from sidechain.ingest.provenance import (
    DOWNLOAD,
    EVIDENCE_PREFIXES,
    STREAM,
    UNSTATED,
    GateError,
    diff_against_recorded,
    gate,
    probe_figshare,
    probe_https,
    probe_huggingface,
    probe_lamin,
    probe_s3,
    probe_zenodo,
    read_provenance,
    write_provenance,
)
from sidechain.utils.paths import resolve_config

DEFAULT_CONFIG = "configs/datasets.yaml"
DEFAULT_ROOT = Path.home() / "data" / "sidechain"

PROBES = {
    "zenodo": probe_zenodo,
    "huggingface": probe_huggingface,
    "figshare": probe_figshare,
    "lamin": probe_lamin,
    "s3": probe_s3,
    "https": probe_https,
}


def load_datasets(path: str | Path = DEFAULT_CONFIG) -> dict[str, dict]:
    cfg = yaml.safe_load(resolve_config(path).read_text())
    return {d["name"]: d for d in cfg.get("datasets", [])}


def select_dataset(name: str, config: str | Path = DEFAULT_CONFIG) -> dict:
    datasets = load_datasets(config)
    if name not in datasets:
        raise GateError(f"no dataset {name!r} in {config}. Known: {', '.join(sorted(datasets))}")
    block = datasets[name]
    if not block.get("enabled", True):
        raise GateError(f"dataset {name!r} is disabled in {config}; enable it deliberately")
    return block


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    return digest(path, "md5", chunk=chunk)


def digest(path: Path, algo: str, *, chunk: int = 1 << 20) -> str:
    """Hash `path` with `algo`, the algorithm the host published."""
    try:
        h = hashlib.new(algo)
    except ValueError as exc:
        raise GateError(
            f"{path.name}: host published a {algo!r} checksum, which this Python's "
            "hashlib cannot compute. Verify it by hand rather than skipping it."
        ) from exc
    with path.open("rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


def block_route(block: dict) -> str:
    """The block's declared `route`; absent means download."""
    return str(block.get("route", DOWNLOAD))


def block_files(block: dict) -> list[dict]:
    """The block's file list: `files:` inline, or `files_manifest:`, a checked-in JSON list of file dicts."""
    manifest = block.get("files_manifest")
    inline = block.get("files")
    if manifest and inline:
        raise GateError(f"dataset {block.get('name')!r} declares both `files` and `files_manifest`; "
                        "a block has one listing")
    if not manifest:
        return list(inline or [])
    path = resolve_config(manifest)
    files = json.loads(path.read_text())
    if not isinstance(files, list) or not files or not all(isinstance(f, dict) and f.get("name") for f in files):
        raise GateError(f"dataset {block.get('name')!r}: {path} must be a non-empty JSON list of "
                        "file dicts, each with a `name`")
    return files


def run_gate(block: dict, root: Path, *, refresh: bool = False,
             config: str | Path = DEFAULT_CONFIG) -> tuple:
    """probe -> gate -> record provenance. Returns (record, selected, dest).

    PROVENANCE.json is written on the first run and compared against after; `refresh=True` rewrites it.
    `config` names the registry the block came from, for the notes.
    """
    host = block["host"]
    if host not in PROBES:
        raise GateError(f"no probe for host {host!r}; known: {', '.join(sorted(PROBES))}")

    route = block_route(block)
    files = block_files(block)
    wanted = [f["name"] for f in files]
    dest = root / block["dest"]
    # https cannot list itself: it is handed the names, and `entries` only when an entry carries a url,
    # a digest or a request
    extra = ({"entries": files}
             if any(f.get("url") or f.get("md5") or f.get("sha256") or f.get("request") for f in files)
             else {})
    record = (PROBES[host](block["record"], names=wanted, **extra) if host == "https"
              else PROBES[host](block["record"]))

    space_dest = dest
    if route == STREAM:
        if not block.get("derived"):
            raise GateError(
                f"dataset {block['name']!r} declares route: stream but no `derived:`. "
                "A stream produces an aggregate and that aggregate needs a declared home "
                "under derived/ -- external/ is for someone else's immutable bytes and is "
                "the first thing deleted under disk pressure."
            )
        space_dest = root / block["derived"]

    # `license_override_source` lets the block's `license:` stand in for a host that states none
    override_source = block.get("license_override_source")
    if block.get("license") == UNSTATED and not override_source:
        raise GateError(
            f"dataset {block['name']!r} declares license: {UNSTATED} but no "
            "`license_override_source`. That value is a record of a search -- which sources "
            "were checked, on what date, and what sanctions the use instead. Without it, it "
            "is a word that skips the licence check."
        )
    license_override = (block["license"], str(override_source)) if override_source else None

    allow_missing = bool(block.get("allow_missing_checksum", False))
    selected = gate(record, budget_gb=float(block["budget_gb"]), select=wanted,
                    dest=dest, route=route, space_dest=space_dest,
                    license_override=license_override,
                    allow_missing_checksum=allow_missing,
                    citation=block.get("citation"))

    declared = block.get("license")
    if license_override:
        # PROVENANCE.json carries the effective terms; its notes keep what the host stated
        record = replace(record, license=declared)
    elif declared and declared != record.license:
        raise GateError(
            f"license changed: config says {declared!r}, the {host} API now says "
            f"{record.license!r}. Terms were re-checked and do not match what was "
            "agreed -- resolve before fetching."
        )

    notes = {"dataset": block["name"], "config": str(config),
             "specs": {f["name"]: f.get("spec", {}) for f in files}}
    if block.get("files_manifest"):
        notes["files_manifest"] = str(block["files_manifest"])
    generated = {f["name"]: f["request"] for f in files if isinstance(f.get("request"), dict)}
    if generated:
        # how a host-built archive was requested, so the request can be repeated
        notes["generated"] = generated
    if allow_missing:
        notes["allow_missing_checksum"] = True
    if block.get("citation"):
        notes["citation"] = str(block["citation"])
    if license_override:
        notes["license_override"] = {
            "host_stated": "unknown",
            "applied": declared,
            "source": str(override_source),
        }
    if route == STREAM:
        notes["derived"] = block.get("derived")
        notes["output_budget_gb"] = float(block["budget_gb"])
    recorded = read_provenance(dest)

    if recorded is None:
        write_provenance(dest, record, selected, notes=notes, route=route)
    elif refresh:
        write_provenance(dest, record, selected, notes=notes, route=route)
        print("  PROVENANCE.json refreshed on request")
    else:
        diffs = diff_against_recorded(recorded, record, selected, route=route)
        if diffs:
            listed = "\n    ".join(diffs)
            raise GateError(
                f"upstream no longer matches the recorded provenance at {dest}:\n"
                f"    {listed}\n"
                "  The bytes on disk were fetched under the recorded terms. Re-run with "
                "--refresh to accept the new state deliberately, having decided the change "
                "is acceptable."
            )
    return record, selected, dest


def verify(selected, dest: Path) -> int:
    """Check every selected file's size and checksum. Returns a process exit code."""
    bad = 0
    for f in selected:
        path = dest / f.name
        if not path.exists():
            print(f"  MISSING  {f.name}")
            bad += 1
            continue
        if path.stat().st_size != f.size_bytes:
            print(f"  SHORT    {f.name} ({path.stat().st_size} != {f.size_bytes})")
            bad += 1
            continue
        if not f.checksum:
            print(f"  no-sum   {f.name} (host published none)")
            continue
        # the first colon: a Last-Modified stamp carries colons of its own
        algo, _, want = f.checksum.partition(":")
        if algo in EVIDENCE_PREFIXES:
            # identity evidence (an ETag, a Last-Modified stamp), not a digest: nothing to recompute
            print(f"  no-sum   {f.name} (host offers {algo} evidence, not a digest)")
            continue
        got = digest(path, algo or "md5")
        print(f"  {'ok      ' if got == want else 'MISMATCH'} {f.name}")
        bad += got != want
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="name from configs/datasets.yaml")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--check", action="store_true", help="verify existing files, fetch nothing")
    ap.add_argument(
        "--refresh", action="store_true",
        help="accept an upstream change and rewrite PROVENANCE.json (otherwise a difference "
             "against the recorded provenance is a refusal)",
    )
    args = ap.parse_args(argv)

    try:
        block = select_dataset(args.dataset, args.config)
        record, selected, dest = run_gate(block, args.root, refresh=args.refresh)
    except GateError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    total = sum(f.size_bytes for f in selected) / 1e9
    route = block_route(block)
    print(f"{record.host}:{record.record_id}  v{record.version}  {record.license}")
    if route == STREAM:
        print(f"  {len(selected)} files, {total:.2f} GB to READ over the network "
              f"(nothing lands; budget {block['budget_gb']} GB bounds the OUTPUT)")
    else:
        print(f"  {len(selected)} files, {total:.2f} GB (budget {block['budget_gb']} GB)")
    print(f"  -> {dest}/PROVENANCE.json matches the recorded fetch\n")

    if route == STREAM:
        # a stream lands no files: nothing to plan or verify
        print("route: stream -- the gate is satisfied and nothing is downloaded here.")
        print(f"  derived aggregates -> {args.root / block['derived']}")
        print("  the streaming reader consumes this PROVENANCE.json and writes LINEAGE.json")
        print("  beside the aggregate (sidechain.data.stream_parquet_pseudobulk, step b).")
        return 0

    if args.check:
        return verify(selected, dest)

    missing = [f for f in selected if not (dest / f.name).exists()]
    if not missing:
        print("all files present; verifying\n")
        return verify(selected, dest)

    print(f"{len(missing)} file(s) to fetch. Run:\n")
    print(f"  cd {dest}")
    for f in missing:
        # --create-dirs: a name may carry directories; -C -: a re-run resumes from the bytes already there
        print(f"  curl -sSL -C - --create-dirs -o {f.name} '{f.url}'")
    print(f"\nthen: uv run python -m sidechain.ingest.fetch --dataset {args.dataset} --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
