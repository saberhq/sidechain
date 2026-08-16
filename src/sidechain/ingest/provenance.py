"""The metadata-first gate: ask the host what it has, before fetching any of it.

Every external corpus enters through here first. The probe calls the host's own
API, records sizes, checksums and license into a `HostRecord`, and the gate
decides whether a fetch may proceed at all. Nothing here downloads data.

The rule this enforces: a dataset we cannot use is discovered by an API call,
not by a 200 GB download. That is not hypothetical caution -- three real
findings during the 2026-08-15 recon, all from metadata alone:

  * X-Atlas/Orion's Figshare record is 559.7 GB, so the download route was never
    viable and we would have found out the hard way;
  * that record's `computed_md5` is empty for one file, an integrity gap the
    bytes themselves cannot reveal;
  * Arc's own `State-Replogle-Filtered` declares no license at all -- the
    `unknown` branch, hit on the third dataset we inspected.

Stdlib only (urllib, not requests): the gate must run before any environment is
set up, including on a fresh box whose only job is to fetch data.
"""
from __future__ import annotations

import json
import shutil
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # contract imports anndata; the gate must run without it
    from sidechain.ingest.contract import Provenance

USER_AGENT = "sidechain-ingest/0.1 (+https://github.com/saberhq/sidechain)"

# What we are willing to proceed on without a human in the loop. The values are
# what a `Provenance.license` will carry downstream, so they are SPDX where SPDX
# has an identifier.
#
# NonCommercial is fine for a non-commercial competition entry. ShareAlike is
# also fine to *ingest* -- it constrains redistribution of derived artifacts,
# which is a later and separate decision.
LICENSE_POLICY: dict[str, str] = {
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc-by-nc-4.0": "CC-BY-NC-4.0",
    "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0",
    "cc0-1.0": "CC0-1.0",
    "mit": "MIT",
    "apache-2.0": "Apache-2.0",
}

# Licenses whose terms reach past ingest. Recorded on the artifact so the
# implication survives the session that discovered it.
REDISTRIBUTION_ENCUMBERED = {"CC-BY-SA-4.0", "CC-BY-NC-SA-4.0"}
NONCOMMERCIAL = {"CC-BY-NC-4.0", "CC-BY-NC-SA-4.0"}


class GateError(RuntimeError):
    """A fetch was refused. The message says which rule and what to do."""


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size_bytes: int
    checksum: str | None   # as the host states it, e.g. "md5:1a2b..."; None means absent
    url: str

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9


@dataclass(frozen=True)
class HostRecord:
    """What the host says it has. Verbatim, before any of it is fetched."""

    host: str            # "zenodo" | "figshare" | "huggingface"
    record_id: str
    api_url: str
    title: str
    license: str         # SPDX where known, else stated terms, else "unknown"
    retrieved: str       # ISO date
    version: str | None = None
    doi: str | None = None
    concept_doi: str | None = None
    files: tuple[RemoteFile, ...] = field(default_factory=tuple)

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)

    def select(self, names: list[str]) -> tuple[RemoteFile, ...]:
        """Pick files by name, raising on any that the record does not have.

        Strict for the same reason `loaders.gene_index` is strict: silently
        skipping a name we asked for yields a partial corpus that looks whole.
        """
        by_name = {f.name: f for f in self.files}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise GateError(
                f"{self.host}:{self.record_id} has no file(s): {', '.join(missing)}. "
                f"Available: {', '.join(sorted(by_name))}"
            )
        return tuple(by_name[n] for n in names)


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def probe_zenodo(record_id: str | int) -> HostRecord:
    """Zenodo record -> HostRecord.

    Note the version/concept distinction: the record id pins one version, while
    `conceptdoi` resolves to "latest" and would change under us. We record both
    and fetch by the pinned id.
    """
    api = f"https://zenodo.org/api/records/{record_id}"
    raw = _get_json(api)
    meta = raw.get("metadata", {})
    stated = (meta.get("license") or {}).get("id", "") or ""
    return HostRecord(
        host="zenodo",
        record_id=str(record_id),
        api_url=api,
        title=meta.get("title", ""),
        license=LICENSE_POLICY.get(stated.lower(), stated or "unknown"),
        retrieved=datetime.now(UTC).date().isoformat(),
        version=meta.get("version"),
        doi=raw.get("doi"),
        concept_doi=raw.get("conceptdoi"),
        files=tuple(
            RemoteFile(
                name=f["key"],
                size_bytes=f["size"],
                checksum=f.get("checksum"),
                url=f.get("links", {}).get("self", ""),
            )
            for f in sorted(raw.get("files", []), key=lambda f: f["key"])
        ),
    )


def gate(
    record: HostRecord,
    *,
    budget_gb: float,
    select: list[str] | None = None,
    allow_missing_checksum: bool = False,
    dest: Path | None = None,
    headroom_gb: float = 10.0,
) -> tuple[RemoteFile, ...]:
    """Decide whether this fetch may proceed. Raises `GateError` if not.

    Every refusal here is cheap; every one it prevents is not.

    `budget_gb` is the ceiling declared for THIS dataset -- it stops a record
    that grew (a new version, extra files) from quietly pulling more than was
    agreed. `dest` additionally checks real free space, because a per-dataset
    ceiling says nothing about whether the disk can take it: several datasets
    each inside their own budget can still fill a volume between them.
    """
    files = record.select(select) if select else record.files
    if not files:
        raise GateError(f"{record.host}:{record.record_id} lists no files")

    if record.license == "unknown" or not record.license:
        raise GateError(
            f"{record.host}:{record.record_id} declares no license. This is a STOP, not a "
            "warning: unstated terms are more restrictive than a permissive tag, not less. "
            "Resolve with the depositor before fetching."
        )

    size_gb = sum(f.size_bytes for f in files) / 1e9
    if size_gb > budget_gb:
        raise GateError(
            f"selection is {size_gb:.2f} GB, over the {budget_gb:.2f} GB budget declared for "
            f"{record.host}:{record.record_id}. Raise the budget deliberately or narrow the "
            "selection; do not let a fetch decide how much disk it takes."
        )

    unchecksummed = [f.name for f in files if not f.checksum]
    if unchecksummed and not allow_missing_checksum:
        raise GateError(
            f"no checksum published for: {', '.join(unchecksummed)}. Pass "
            "allow_missing_checksum=True to proceed -- it is recorded in PROVENANCE.json so the "
            "exception stays visible later."
        )

    if dest is not None:
        needed = sum(f.size_bytes for f in files if not (dest / f.name).exists())
        anchor = next((p for p in [dest, *dest.parents] if p.exists()), Path("/"))
        free = shutil.disk_usage(anchor).free
        if free - needed < headroom_gb * 1e9:
            raise GateError(
                f"not enough disk: need {needed / 1e9:.2f} GB, {free / 1e9:.1f} GB free, "
                f"and {headroom_gb:.0f} GB must stay free. A per-dataset budget does not "
                "know about the other datasets; this does."
            )
    return files


def to_provenance(record: HostRecord) -> Provenance:
    """HostRecord (what the host said) -> Provenance (what the dataset carries).

    The bridge exists so the license reaching a HarmonizedDataset is the one the
    API actually returned, rather than a value retyped from memory or a README
    weeks later. Without it the two representations drift, and `Provenance.license`
    stops being evidence.

    Imported lazily: contract imports anndata, and the gate is meant to run on a
    fetch-only box with no scientific stack installed.
    """
    from sidechain.ingest import contract

    return contract.Provenance(
        source=record.host,
        accession=record.record_id,
        license=record.license,
        retrieved=record.retrieved,
    )


def write_provenance(
    dest: Path,
    record: HostRecord,
    selected: tuple[RemoteFile, ...],
    *,
    notes: dict | None = None,
) -> Path:
    """Write PROVENANCE.json into `dest` BEFORE any data file lands there.

    The ordering is the point. A provenance file written afterwards documents
    what we happened to get; written first, it is the thing the fetch is checked
    against.
    """
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "record": {**asdict(record), "files": [asdict(f) for f in record.files]},
        "selected": [asdict(f) for f in selected],
        "selected_bytes": sum(f.size_bytes for f in selected),
        "license_flags": {
            "noncommercial": record.license in NONCOMMERCIAL,
            "redistribution_encumbered": record.license in REDISTRIBUTION_ENCUMBERED,
        },
        "notes": notes or {},
    }
    path = dest / "PROVENANCE.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def read_provenance(dest: Path) -> dict | None:
    """The provenance already recorded at `dest`, or None if this is a first fetch."""
    path = dest / "PROVENANCE.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def diff_against_recorded(recorded: dict, record: HostRecord,
                          selected: tuple[RemoteFile, ...]) -> list[str]:
    """What changed upstream since the recorded fetch. Empty list means nothing.

    Compares only the fields that would make the bytes different or the terms
    different. `retrieved` is deliberately excluded -- it changes on every probe
    by definition, and treating it as a difference would make every run look
    like a change.
    """
    was = recorded.get("record", {})
    diffs: list[str] = []

    for field_name, now in (("license", record.license), ("version", record.version)):
        before = was.get(field_name)
        if before != now:
            diffs.append(f"{field_name}: recorded {before!r}, host now says {now!r}")

    before_files = {f["name"]: f for f in recorded.get("selected", [])}
    now_files = {f.name: f for f in selected}

    for name in sorted(set(before_files) - set(now_files)):
        diffs.append(f"{name}: was in the recorded selection, no longer selected")
    for name in sorted(set(now_files) - set(before_files)):
        diffs.append(f"{name}: newly selected, not in the recorded provenance")

    for name in sorted(set(before_files) & set(now_files)):
        b, n = before_files[name], now_files[name]
        if b.get("size_bytes") != n.size_bytes:
            diffs.append(f"{name}: size {b.get('size_bytes')} -> {n.size_bytes}")
        if b.get("checksum") != n.checksum:
            diffs.append(f"{name}: checksum {b.get('checksum')} -> {n.checksum}")
    return diffs
