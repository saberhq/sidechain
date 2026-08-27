"""Run one dataset's ingest from `configs/datasets.yaml`.

    uv run python -m sidechain.ingest.fetch --dataset scperturb_multicontext
    uv run python -m sidechain.ingest.fetch --dataset ... --check   # verify only

Order of operations is the whole design (`research/decisions/0003`, private):

    probe host API  ->  gate  ->  write PROVENANCE.json  ->  fetch  ->  verify

The gate runs before any byte moves, and PROVENANCE.json is written before the
first file lands, so it is the thing the download is checked against rather than
a description of whatever happened to arrive.

This module deliberately does not implement a downloader. `curl` already
resumes, honours ranged requests and handles flaky links better than anything
worth writing here, so --plan emits the fetch commands and --check verifies the
result. What cannot be delegated -- deciding whether the fetch is allowed at all,
and proving the bytes are the right ones -- is what lives here.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import yaml

from sidechain.ingest.provenance import (
    DOWNLOAD,
    STREAM,
    GateError,
    diff_against_recorded,
    gate,
    probe_figshare,
    probe_huggingface,
    probe_lamin,
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
    """Hash `path` with the algorithm the HOST published, not a fixed one.

    Zenodo states `md5:...`; Hugging Face stores LFS objects under their sha256
    and states `sha256:...`. Hardcoding md5 -- which this did while zenodo was
    the only host -- would hash an HF file with the wrong algorithm, compare the
    result to a digest it could never equal, and report MISMATCH on bytes that
    are perfectly fine. A verifier that cries wolf gets ignored, which costs
    more than having no verifier.
    """
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
    """The declared route for `block`, defaulting to download.

    Absent means download: every block written before streaming existed is a
    download, and that is the reading that keeps the old blocks true.
    """
    return str(block.get("route", DOWNLOAD))


def run_gate(block: dict, root: Path, *, refresh: bool = False) -> tuple:
    """probe -> gate -> record provenance. Returns (record, selected, dest).

    PROVENANCE.json is written ONCE, on the first fetch, and thereafter only
    compared against. That is the point of it: it is evidence of what we
    actually took, and the thing later downloads are checked against. An earlier
    version rewrote it on every invocation, which quietly turned it into a record
    of the last time the command ran -- so an upstream change to the license or
    the file list would have been absorbed silently instead of raising.

    `refresh=True` accepts the upstream state and rewrites, for when a change is
    real and intended. It has to be asked for.
    """
    host = block["host"]
    if host not in PROBES:
        raise GateError(f"no probe for host {host!r}; known: {', '.join(sorted(PROBES))}")

    record = PROBES[host](block["record"])
    route = block_route(block)
    wanted = [f["name"] for f in block["files"]]
    dest = root / block["dest"]

    # A stream MUST say where its aggregate goes. Without `derived:` the free-
    # space check would silently measure the wrong tree and the output would
    # have nowhere declared to live -- and the checked-in config passing a
    # contract test says nothing about the next block someone adds.
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

    # `license_override_source` authorises the block's `license:` to stand in
    # for a host that CANNOT state one (lamin has no license field on Artifact).
    # The value names where the terms were actually verified; gate() refuses an
    # override that contradicts a host that DID state a license, and refuses
    # one with no source. Recorded into PROVENANCE.json below either way.
    override_source = block.get("license_override_source")
    license_override = (block["license"], str(override_source)) if override_source else None

    selected = gate(record, budget_gb=float(block["budget_gb"]), select=wanted,
                    dest=dest, route=route, space_dest=space_dest,
                    license_override=license_override)

    declared = block.get("license")
    if license_override:
        # PROVENANCE.json must carry the EFFECTIVE terms (so license_flags are
        # computed from something real), while the notes keep the host's own
        # silence visible beside where the terms actually came from.
        record = replace(record, license=declared)
    elif declared and declared != record.license:
        raise GateError(
            f"license changed: config says {declared!r}, the {host} API now says "
            f"{record.license!r}. Terms were re-checked and do not match what was "
            "agreed -- resolve before fetching."
        )

    notes = {"dataset": block["name"], "config": str(DEFAULT_CONFIG),
             "specs": {f["name"]: f.get("spec", {}) for f in block["files"]}}
    if license_override:
        notes["license_override"] = {
            "host_stated": "unknown",
            "applied": declared,
            "source": str(override_source),
        }
    if route == STREAM:
        # Where the aggregate goes, and under what terms. Recorded here because
        # the stream itself writes LINEAGE.json beside the aggregate and needs
        # to point back at exactly this file.
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
        algo, _, want = f.checksum.rpartition(":")
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
        # No curl plan and no checksum pass: both would describe files that are
        # never going to exist, and `verify` would report all of them MISSING.
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
        print(f"  curl -sSL -C - -o {f.name} '{f.url}'")
    print(f"\nthen: uv run python -m sidechain.ingest.fetch --dataset {args.dataset} --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
