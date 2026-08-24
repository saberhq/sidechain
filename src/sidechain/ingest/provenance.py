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

import fnmatch
import json
import shutil
import urllib.parse
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

# The accepted set, derived from the policy above so the two cannot drift.
# A license that is STATED but not on this list is refused: ADR 0003 says the
# gate stops on "terms we have not accepted for that use", and only the
# literal "unknown" was ever being caught. HF makes that gap live -- its
# namespace includes `other`, `unknown` and bespoke slugs, and an unrecognised
# NonCommercial or NoDerivatives license would otherwise sail through AND be
# recorded with both license_flags false, i.e. as less encumbered than it is.
ACCEPTED_LICENSES = frozenset(LICENSE_POLICY.values())

# Licenses whose terms reach past ingest. Recorded on the artifact so the
# implication survives the session that discovered it.
REDISTRIBUTION_ENCUMBERED = {"CC-BY-SA-4.0", "CC-BY-NC-SA-4.0"}
NONCOMMERCIAL = {"CC-BY-NC-4.0", "CC-BY-NC-SA-4.0"}

# How a dataset is consumed. `download` puts every selected byte on disk;
# `stream` reads it over the network once and keeps only an aggregate.
#
# The distinction reaches the gate because it changes what `budget_gb` MEANS.
# Downloaded, the budget bounds the selection: refuse if the files exceed it.
# Streamed, the corpus never lands, so bounding the selection would be
# meaningless -- X-Atlas/Orion is 126 GB of parquet that yields ~10 GB of
# pseudobulk -- and the budget bounds the OUTPUT instead. Neither reading is
# wrong; leaving it implicit is, which is why it is a declared field.
DOWNLOAD = "download"
STREAM = "stream"
ROUTES = (DOWNLOAD, STREAM)


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
        """Pick files by exact name or glob, raising on anything that matches nothing.

        Strict for the same reason `loaders.gene_index` is strict: silently
        skipping a name we asked for yields a partial corpus that looks whole.

        A pattern containing `*`, `?` or `[` is expanded against the record and
        must match at least one file; anything else is an exact name and must be
        present. Globs exist because X-Atlas/Orion is 332 per-batch parquet
        files -- listing them individually would be a config block that goes
        stale the moment Xaira adds a batch, and the failure mode of a stale
        list is the silent partial corpus this method exists to prevent.

        A zero-match glob is an error, NOT an empty selection. `data/HCT116_*`
        against a repo that renamed its folder should stop the run, not
        quietly stream nothing and report success.

        Note `fnmatch` semantics: `*` spans `/`, so `data/*.parquet` also
        matches `data/sub/x.parquet`. Fine for the flat layouts we ingest, and
        stated here rather than discovered.
        """
        by_name = {f.name: f for f in self.files}
        picked: list[str] = []
        empty: list[str] = []

        for name in names:
            if any(ch in name for ch in "*?["):
                hits = sorted(fnmatch.filter(by_name, name))
                if not hits:
                    empty.append(name)
                picked.extend(hits)
            elif name in by_name:
                picked.append(name)
            else:
                empty.append(name)

        if empty:
            available = sorted(by_name)
            shown = ", ".join(available[:8])
            more = f" (+{len(available) - 8} more)" if len(available) > 8 else ""
            raise GateError(
                f"{self.host}:{self.record_id} has no file(s) matching: "
                f"{', '.join(empty)}. Available: {shown}{more}"
            )

        seen: set[str] = set()
        ordered = [n for n in picked if not (n in seen or seen.add(n))]
        return tuple(by_name[n] for n in ordered)


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _post_json(url: str, payload: dict) -> list | dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
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


HF_API = "https://huggingface.co/api/datasets"
HF_RESOLVE = "https://huggingface.co/datasets"

# paths-info accepts a list of paths per POST. 100 keeps the request small
# enough to be reliable and still sizes a 337-file repo in four calls.
HF_PATHS_INFO_BATCH = 100


def _hf_paths_info(repo: str, revision: str, paths: list[str]) -> dict[str, dict]:
    """Sizes and checksums for `paths`, batched. Keyed by path.

    Split out because it is the half of the HF probe that can fail partially:
    the repo endpoint answers in one call, this one in several, and a batch
    that silently returned nothing would leave files at size 0.
    """
    url = f"{HF_API}/{repo}/paths-info/{revision}"
    out: dict[str, dict] = {}
    for i in range(0, len(paths), HF_PATHS_INFO_BATCH):
        batch = paths[i : i + HF_PATHS_INFO_BATCH]
        for entry in _post_json(url, {"paths": batch, "expand": True}) or []:
            if entry.get("type") == "file" and "path" in entry:
                out[entry["path"]] = entry
    return out


def probe_huggingface(record_id: str) -> HostRecord:
    """Hugging Face dataset repo -> HostRecord.

    Two calls, because HF splits the answer in two and the second half is the
    half the gate needs:

      * `/api/datasets/<owner>/<name>` lists filenames, the license and the
        current commit sha -- but NO sizes. Its `siblings` entries carry
        `rfilename` and nothing else.
      * `/api/datasets/<owner>/<name>/paths-info/<sha>` carries `size` and, for
        anything stored in LFS, a sha256 in `lfs.oid`.

    A probe that stopped after the first call would build a HostRecord whose
    every size was 0, and `gate` would then wave through any budget at all --
    the exact failure the gate exists to prevent, wearing the gate's authority.
    So sizes are required here: a file the second call does not resolve raises
    rather than defaulting to zero.

    **The commit sha is the pin.** HF has no immutable per-version record id the
    way Zenodo does; `main` moves when the depositor pushes. So the sha is read
    once, recorded as `version`, and used for both `paths-info` and every file
    URL -- which makes the record re-checkable, makes a download or a stream
    read exactly the bytes that were probed, and makes `diff_against_recorded`
    fire if the repo is republished under us.
    """
    repo = str(record_id).strip("/")
    api = f"{HF_API}/{repo}"
    raw = _get_json(api)

    sha = raw.get("sha") or ""
    if not sha:
        raise GateError(
            f"huggingface:{repo} returned no commit sha. Without it nothing can be "
            "pinned: `main` moves under us and the record stops being re-checkable."
        )

    # cardData.license is the declaration; the `license:` tag is the same fact
    # restated and is the fallback. A repo with NEITHER is the Arc case, and
    # falls through to "unknown", which `gate` treats as a stop.
    stated = (raw.get("cardData") or {}).get("license") or ""
    if isinstance(stated, list):
        stated = stated[0] if stated else ""
    if not stated:
        tags = [t for t in raw.get("tags", []) if str(t).startswith("license:")]
        stated = tags[0].split(":", 1)[1] if tags else ""

    names = sorted(
        s["rfilename"] for s in raw.get("siblings", []) or [] if s.get("rfilename")
    )
    info = _hf_paths_info(repo, sha, names)
    unresolved = [n for n in names if n not in info]
    if unresolved:
        raise GateError(
            f"huggingface:{repo} listed {len(unresolved)} file(s) that paths-info did not "
            f"resolve: {', '.join(unresolved[:5])}. Sizes drive the budget check, so a "
            "file of unknown size is not a file we may proceed on."
        )

    files = []
    for name in names:
        entry = info[name]
        lfs = entry.get("lfs") or {}
        oid = lfs.get("oid")
        files.append(
            RemoteFile(
                name=name,
                size_bytes=int(entry.get("size", 0)),
                # sha256, not md5: LFS stores the object under its sha256. Small
                # non-LFS files (README, .gitattributes) have no lfs block and so
                # no published checksum -- `gate` stops on those unless overridden.
                checksum=f"sha256:{oid}" if oid else None,
                url=f"{HF_RESOLVE}/{repo}/resolve/{sha}/{urllib.parse.quote(name)}",
            )
        )

    return HostRecord(
        host="huggingface",
        record_id=repo,
        api_url=api,
        title=raw.get("id", repo),
        license=LICENSE_POLICY.get(str(stated).lower(), stated or "unknown"),
        retrieved=datetime.now(UTC).date().isoformat(),
        version=sha,
        doi=None,
        concept_doi=None,
        files=tuple(files),
    )


FIGSHARE_API = "https://api.figshare.com/v2/articles"

# Figshare states a license by human NAME ("MIT", "CC BY 4.0"), never as an
# SPDX id the way Zenodo does. Lowercasing and swapping spaces for dashes turns
# most of its spellings into a LICENSE_POLICY key ("CC BY 4.0" -> "cc-by-4.0");
# these are the ones that rule does not reach. Kept local to this probe rather
# than added to LICENSE_POLICY, because they are one host's spelling of a
# license the policy already knows -- not a new license we have accepted.
FIGSHARE_LICENSE_ALIASES = {
    "cc0": "cc0-1.0",
    "cc-0": "cc0-1.0",
    "public-domain-dedication-(cc0)": "cc0-1.0",
    "mit-license": "mit",
    "apache-2.0-license": "apache-2.0",
}


def _figshare_license(raw: dict) -> str:
    """The record's license, normalised to the SPDX string the gate compares.

    Passing Figshare's `name` through raw would hand `gate` the string
    "CC BY 4.0", which is not "CC-BY-4.0" and is therefore not on
    ACCEPTED_LICENSES -- refusing a perfectly usable CC-BY dataset. Normalising
    here rather than loosening the gate keeps the refusal meaningful.
    """
    stated = ((raw.get("license") or {}).get("name") or "").strip()
    if not stated:
        return "unknown"
    key = stated.lower().replace(" ", "-")
    key = FIGSHARE_LICENSE_ALIASES.get(key, key)
    return LICENSE_POLICY.get(key, stated)


def probe_figshare(record_id: str | int) -> HostRecord:
    """Figshare article -> HostRecord. `record_id` is `<id>.v<n>`, or a bare id.

    **The `.v<n>` suffix is the pin, and it matters more here than on Zenodo.**
    A Zenodo record id already names one immutable version; a bare Figshare
    article id resolves to *latest* and moves when the depositor publishes
    again -- the same trap as Zenodo's concept DOI, wearing an article id's
    clothes. So `26819743.v1` reads the versioned endpoint, and a bare id is
    accepted but resolves latest; `version` is recorded either way, so
    `diff_against_recorded` still catches a bump on the next run. Prefer the
    pin: catching a change afterwards is worse than not having one.

    **Empty `computed_md5` is an ABSENT checksum, not a checksum.** Figshare
    returns `""` rather than omitting the field when it never verified the
    uploader's hash -- exactly the X-Atlas/Orion HCT116 finding of ADR 0003 §1,
    still unresolved upstream. Passing `""` through would make `verify` compare
    a real digest against the empty string and report MISMATCH on good bytes,
    forever. Returning None instead routes it to the gate's missing-checksum
    refusal, where the exception has to be asked for and gets recorded.

    Deliberately NOT falling back to `supplied_md5`: that is the uploader's own
    claim about their own bytes, which is the thing `computed_md5` exists to
    corroborate. Substituting one for the other would erase the distinction the
    ADR was written around.
    """
    ref = str(record_id).strip().strip("/")
    article, _, ver = ref.partition(".v")
    api = f"{FIGSHARE_API}/{article}/versions/{ver}" if ver else f"{FIGSHARE_API}/{article}"
    raw = _get_json(api)

    doi = raw.get("doi") or None
    # Figshare publishes no concept DOI field; the versionless form of the
    # article DOI is that identifier, and it is the one that follows "latest".
    # Recorded so a reader can see which DOI is pinned and which drifts.
    concept = doi.rsplit(".v", 1)[0] if doi and ".v" in doi else None

    files = []
    for f in sorted(raw.get("files", []) or [], key=lambda f: f.get("name", "")):
        computed = (f.get("computed_md5") or "").strip()
        files.append(
            RemoteFile(
                name=f.get("name", ""),
                size_bytes=int(f.get("size") or 0),
                checksum=f"md5:{computed}" if computed else None,
                url=f.get("download_url", ""),
            )
        )

    return HostRecord(
        host="figshare",
        record_id=ref,
        api_url=api,
        title=raw.get("title", ""),
        license=_figshare_license(raw),
        retrieved=datetime.now(UTC).date().isoformat(),
        version=str(raw.get("version")) if raw.get("version") is not None else None,
        doi=doi,
        concept_doi=concept,
        files=tuple(files),
    )


def gate(
    record: HostRecord,
    *,
    budget_gb: float,
    select: list[str] | None = None,
    allow_missing_checksum: bool = False,
    dest: Path | None = None,
    headroom_gb: float = 10.0,
    route: str = DOWNLOAD,
    space_dest: Path | None = None,
) -> tuple[RemoteFile, ...]:
    """Decide whether this fetch may proceed. Raises `GateError` if not.

    Every refusal here is cheap; every one it prevents is not.

    `budget_gb` is the ceiling declared for THIS dataset -- it stops a record
    that grew (a new version, extra files) from quietly pulling more than was
    agreed. `dest` additionally checks real free space, because a per-dataset
    ceiling says nothing about whether the disk can take it: several datasets
    each inside their own budget can still fill a volume between them.

    `route` changes what the budget is measured AGAINST, and nothing else:

      * `download` -- the budget bounds the selection. Refuse if the files
        exceed it, and check the disk against their combined size.
      * `stream` -- the corpus never lands, so the selection's size is not a
        disk claim and is not checked against the budget. The budget bounds the
        AGGREGATE we write instead, so that is what the disk is checked
        against. The selection's real size is still recorded, so a 126 GB
        stream never looks like a 12 GB one.

    An unrecognised route is a refusal rather than a fallback: `route: streamed`
    silently treated as `download` would try to put 126 GB of X-Atlas parquet
    on a laptop, which is precisely the accident this module was built after.

    `space_dest` is where the bytes ACTUALLY land, when that is not `dest`. On a
    stream they are two different trees -- `dest` takes PROVENANCE.json under
    external/, the aggregate goes to derived/ -- and free space has to be
    measured where the writing happens. They sit on one volume today, so this
    is latent rather than broken; it is fixed here because "same volume" is an
    assumption nothing was checking, and an external disk under derived/ is an
    entirely ordinary thing to do.
    """
    if route not in ROUTES:
        raise GateError(
            f"unknown route {route!r} for {record.host}:{record.record_id}; "
            f"known: {', '.join(ROUTES)}. Not defaulting to {DOWNLOAD!r} -- a typo here "
            "would put a streamed corpus on the disk in full."
        )

    files = record.select(select) if select else record.files
    if not files:
        raise GateError(f"{record.host}:{record.record_id} lists no files")

    if record.license == "unknown" or not record.license:
        raise GateError(
            f"{record.host}:{record.record_id} declares no license. This is a STOP, not a "
            "warning: unstated terms are more restrictive than a permissive tag, not less. "
            "Resolve with the depositor before fetching."
        )

    if record.license not in ACCEPTED_LICENSES:
        raise GateError(
            f"{record.host}:{record.record_id} states {record.license!r}, which is not on the "
            f"accepted list ({', '.join(sorted(ACCEPTED_LICENSES))}). Stated-but-unrecognised "
            "is not the same as permissive: read the terms, then add it to LICENSE_POLICY "
            "deliberately. Letting it through would also record it with both license_flags "
            "false -- as LESS encumbered than it actually is."
        )

    size_gb = sum(f.size_bytes for f in files) / 1e9
    if route == DOWNLOAD and size_gb > budget_gb:
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

    check_at = space_dest if space_dest is not None else dest
    if check_at is not None:
        dest = check_at
        if route == STREAM:
            # Nothing from the corpus lands, so the claim on the disk is the
            # aggregate we are about to write -- which is what budget_gb bounds
            # on this route. Checking the 126 GB selection here would refuse
            # every stream on a laptop; checking nothing would let an unbounded
            # aggregate fill the volume.
            needed = int(budget_gb * 1e9)
        else:
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
    route: str = DOWNLOAD,
) -> Path:
    """Write PROVENANCE.json into `dest` BEFORE any data file lands there.

    The ordering is the point. A provenance file written afterwards documents
    what we happened to get; written first, it is the thing the fetch is checked
    against. On a streamed dataset "before any data file lands" means before the
    first range request: `dest` ends up holding this file and nothing else, which
    is the intended shape (ADR 0003 section 1), not a failed download.

    `route` is recorded because `selected_bytes` means two different things
    without it -- bytes on disk when downloaded, bytes read over the network
    when streamed. A reader who finds 126 GB selected and a 0.9 GB directory
    should be able to tell "streamed" from "the download died".
    """
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "route": route,
        "record": {**asdict(record), "files": [asdict(f) for f in record.files]},
        "selected": [asdict(f) for f in selected],
        "selected_bytes": sum(f.size_bytes for f in selected),
        "lands_on_disk": route == DOWNLOAD,
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
                          selected: tuple[RemoteFile, ...],
                          *, route: str | None = None) -> list[str]:
    """What changed upstream since the recorded fetch. Empty list means nothing.

    Compares only the fields that would make the bytes different or the terms
    different. `retrieved` is deliberately excluded -- it changes on every probe
    by definition, and treating it as a difference would make every run look
    like a change.
    """
    was = recorded.get("record", {})
    diffs: list[str] = []

    # A route flip changes what budget_gb bounds AND whether the bytes are ever
    # verified, so absorbing one silently would let a config edit turn off the
    # checksum pass on an already-recorded dataset. Only compared when the
    # caller states a route, so a schema-1 record (written before routes
    # existed) does not read as a change on every run.
    if route is not None and "route" in recorded and recorded["route"] != route:
        diffs.append(f"route: recorded {recorded['route']!r}, config now says {route!r}")

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
