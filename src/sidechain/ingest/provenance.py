"""The metadata-first gate: probe a host's API for sizes, checksums and licence, and decide whether a fetch may proceed.

    record = probe_zenodo("1234567")
    files = gate(record, budget_gb=5.0, select=["data.h5ad"], dest=dest)
    write_provenance(dest, record, files)

Reads host APIs (stdlib only); writes PROVENANCE.json. Downloads nothing.
Notes: private/code-notes/src/sidechain/ingest/provenance.py.md
"""
from __future__ import annotations

import fnmatch
import json
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # contract imports anndata; the gate must run without it
    from sidechain.ingest.contract import Provenance

USER_AGENT = "sidechain-ingest/0.1 (+https://github.com/saberhq/sidechain)"

# Stated licence (lowercased) -> the value a `Provenance.license` carries; SPDX where SPDX has an id.
LICENSE_POLICY: dict[str, str] = {
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc-by-nc-4.0": "CC-BY-NC-4.0",
    "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0",
    "cc0-1.0": "CC0-1.0",
    "mit": "MIT",
    "apache-2.0": "Apache-2.0",
    # Not SPDX from here on: stated terms, reached only through `license_override_source`.
    # A publisher's terms: free for research use, with a citation.
    "free-for-research-with-citation": "Free-for-research-with-citation",
    # NCBI's policy for its databases (GEO): no restrictions on use or distribution.
    "ncbi-no-restrictions": "NCBI-no-restrictions",
    # The GENCODE project's terms: open access, no restriction named.
    "gencode-open-access": "GENCODE-open-access",
    # The ENCODE Consortium's data-use policy: unrestricted use, citation requested.
    "encode-unrestricted": "ENCODE-unrestricted",
    # A publisher's or data portal's terms: non-commercial use only, no right to redistribute.
    "publisher-terms-noncommercial": "Publisher-terms-noncommercial",
}

# The accepted set; a stated licence not on it is refused.
ACCEPTED_LICENSES = frozenset(LICENSE_POLICY.values())

# Terms stated nowhere: reachable only through `license_override`, needs a `citation` on `route: download`,
# and records as encumbered on both flags below.
UNSTATED = "UNSTATED"

# Licences whose terms reach past ingest, recorded as PROVENANCE.json's `license_flags`.
REDISTRIBUTION_ENCUMBERED = {"CC-BY-SA-4.0", "CC-BY-NC-SA-4.0", "Publisher-terms-noncommercial", UNSTATED}
NONCOMMERCIAL = {"CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "Publisher-terms-noncommercial", UNSTATED}

# `download` puts every selected byte on disk, and `budget_gb` bounds the selection; `stream` reads it
# once, keeps only an aggregate, and `budget_gb` bounds that aggregate.
DOWNLOAD = "download"
STREAM = "stream"
ROUTES = (DOWNLOAD, STREAM)

# Checksum-slot prefixes that are identity evidence, not digests: `probe_s3`'s multipart ETag and
# `probe_https`'s Last-Modified stamp. The gate counts such a file as unchecksummed.
EVIDENCE_PREFIXES = ("s3-etag", "http-last-modified")


def is_digest(checksum: str | None) -> bool:
    """True when the checksum slot holds something `fetch.verify` can recompute."""
    return bool(checksum) and checksum.partition(":")[0] not in EVIDENCE_PREFIXES


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

    host: str            # "zenodo" | "figshare" | "huggingface" | "lamin" | "s3" | "https"
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
        """Pick files by exact name or glob (`fnmatch`: `*` spans `/`); a name or glob that matches nothing raises."""
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
    """Zenodo record -> HostRecord; the record id pins one version, and `conceptdoi` (latest) is recorded beside it."""
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

# paths per paths-info POST
HF_PATHS_INFO_BATCH = 100


def _hf_paths_info(repo: str, revision: str, paths: list[str]) -> dict[str, dict]:
    """Sizes and checksums for `paths` from HF's paths-info, batched; keyed by path."""
    url = f"{HF_API}/{repo}/paths-info/{revision}"
    out: dict[str, dict] = {}
    for i in range(0, len(paths), HF_PATHS_INFO_BATCH):
        batch = paths[i : i + HF_PATHS_INFO_BATCH]
        for entry in _post_json(url, {"paths": batch, "expand": True}) or []:
            if entry.get("type") == "file" and "path" in entry:
                out[entry["path"]] = entry
    return out


def probe_huggingface(record_id: str) -> HostRecord:
    """Hugging Face dataset repo -> HostRecord, pinned to the repo's commit sha (`version`).

    The repo endpoint gives names, licence and sha; `paths-info` at that sha gives sizes and LFS sha256s.
    A listed file that paths-info does not resolve raises.
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

    # cardData.license first, then a `license:` tag; neither gives "unknown"
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
                # LFS objects carry a sha256; a non-LFS file has no published checksum
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

# Figshare licence names that lowercasing and dashing do not turn into a LICENSE_POLICY key
FIGSHARE_LICENSE_ALIASES = {
    "cc0": "cc0-1.0",
    "cc-0": "cc0-1.0",
    "public-domain-dedication-(cc0)": "cc0-1.0",
    "mit-license": "mit",
    "apache-2.0-license": "apache-2.0",
}


def _figshare_license(raw: dict) -> str:
    """The record's licence name, normalised to the SPDX string the gate compares."""
    stated = ((raw.get("license") or {}).get("name") or "").strip()
    if not stated:
        return "unknown"
    key = stated.lower().replace(" ", "-")
    key = FIGSHARE_LICENSE_ALIASES.get(key, key)
    return LICENSE_POLICY.get(key, stated)


def probe_figshare(record_id: str | int) -> HostRecord:
    """Figshare article -> HostRecord. `record_id` is `<id>.v<n>` (pinned) or a bare id (latest).

    An empty `computed_md5` is recorded as no checksum; `supplied_md5` is not read.
    """
    ref = str(record_id).strip().strip("/")
    article, _, ver = ref.partition(".v")
    api = f"{FIGSHARE_API}/{article}/versions/{ver}" if ver else f"{FIGSHARE_API}/{article}"
    raw = _get_json(api)

    doi = raw.get("doi") or None
    # the versionless article DOI, which follows latest
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


def probe_lamin(record_id: str) -> HostRecord:
    """LaminDB instance -> HostRecord. `record_id` is `<owner>/<name>`; lists keyed latest artifacts, moves no bytes.

    Licence is always "unknown" (an Artifact carries no terms), so a block passes only with `license_override`.
    Hashes are recorded under `lamin:` as identity evidence.
    """
    import lamindb as ln

    db = ln.DB(record_id)
    rows = list(
        db.Artifact.filter(is_latest=True, key__isnull=False)
        .values("key", "size", "hash")
    )
    files = tuple(
        RemoteFile(
            name=r["key"],
            size_bytes=int(r["size"] or 0),
            checksum=f"lamin:{r['hash']}" if r.get("hash") else None,
            url="",
        )
        for r in sorted(rows, key=lambda r: r["key"])
    )
    return HostRecord(
        host="lamin",
        record_id=record_id,
        api_url=f"https://lamin.ai/{record_id}",
        title=record_id,
        license="unknown",
        retrieved=datetime.now(UTC).date().isoformat(),
        version=None,
        files=files,
    )


S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _s3_etag_checksum(etag: str | None) -> str | None:
    """An S3 ETag as a checksum: single-part is `md5:`, multipart (`-<n>`) is `s3-etag:` evidence; None stays None."""
    if not etag:
        return None
    tag = etag.strip().strip('"')
    if not tag:
        return None
    if "-" in tag:
        return f"s3-etag:{tag}"
    if len(tag) == 32 and all(c in "0123456789abcdefABCDEF" for c in tag):
        return f"md5:{tag.lower()}"
    return f"s3-etag:{tag}"


def probe_s3(record_id: str) -> HostRecord:
    """A public S3 bucket -> HostRecord. `record_id` is `<bucket>` or `<bucket>/<prefix>`; an anonymous ListObjectsV2.

    Licence is always "unknown" and `version` None: a block needs `license_override_source`, and upstream drift
    shows in each file's size and ETag.
    """
    bucket, _, prefix = record_id.partition("/")
    if not bucket:
        raise GateError(f"s3 record {record_id!r} names no bucket")

    files: list[RemoteFile] = []
    token: str | None = None
    for _ in range(1000):                      # pagination guard; 1000 x 1000 keys
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        url = f"https://{bucket}.s3.amazonaws.com/?{urllib.parse.urlencode(query)}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            root = ET.fromstring(resp.read())

        for node in root.findall("s3:Contents", S3_NS):
            key = node.findtext("s3:Key", default="", namespaces=S3_NS)
            if not key or key.endswith("/"):   # a folder marker is not a file
                continue
            size = int(node.findtext("s3:Size", default="0", namespaces=S3_NS) or 0)
            etag = node.findtext("s3:ETag", namespaces=S3_NS)
            files.append(RemoteFile(
                name=key[len(prefix):] if prefix and key.startswith(prefix) else key,
                size_bytes=size,
                checksum=_s3_etag_checksum(etag),
                url=f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}",
            ))

        if (root.findtext("s3:IsTruncated", default="false", namespaces=S3_NS) or
                "false").strip().lower() != "true":
            break
        token = root.findtext("s3:NextContinuationToken", namespaces=S3_NS)
        if not token:
            break
    else:
        raise GateError(f"s3://{record_id}: listing did not terminate in 1000 pages")

    if not files:
        raise GateError(f"s3://{record_id}: listing returned no objects -- wrong bucket "
                        "or prefix, or the bucket is not public")

    return HostRecord(
        host="s3",
        record_id=record_id,
        api_url=f"https://{bucket}.s3.amazonaws.com/?list-type=2&prefix="
                f"{urllib.parse.quote(prefix)}",
        title=record_id,
        license="unknown",                     # by construction; see the docstring
        retrieved=datetime.now(UTC).date().isoformat(),
        version=None,
        files=tuple(sorted(files, key=lambda f: f.name)),
    )


GENERATED_PAGE_MAX_BYTES = 1 << 20     # a form's answer is a page, never the archive itself


def generated_archive_url(base: str, name: str, request: dict) -> str:
    """POST a block entry's ``request`` form and return the generated archive's https URL.

    ``request`` is ``{post: <https endpoint>, form: {field: value | [values]}, link: <regex>}``; the first match
    of ``link`` in the answer page, resolved against ``post``, is the URL.
    """
    post = str(request.get("post", ""))
    form = request.get("form")
    link = request.get("link")
    if not post.startswith("https://") or not isinstance(form, dict) or not link:
        raise GateError(f"https:{base}: {name!r} carries a request that needs an https `post` "
                        f"endpoint, a `form` mapping and a `link` regex; got {request!r}")
    body = urllib.parse.urlencode(form, doseq=True).encode()
    req = urllib.request.Request(post, data=body, method="POST",
                                 headers={"User-Agent": USER_AGENT,
                                          "Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            ctype = str(resp.headers.get("Content-Type", ""))
            page = resp.read(GENERATED_PAGE_MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise GateError(f"https:{base}: POST for {name!r} returned {exc.code}") from exc
    if len(page) > GENERATED_PAGE_MAX_BYTES or ("html" not in ctype and "text" not in ctype):
        raise GateError(f"https:{base}: the POST for {name!r} answered {ctype!r}, "
                        f"{len(page)} bytes, not the page that names the generated archive")
    m = re.search(str(link), page.decode("utf-8", errors="replace"))
    if not m:
        raise GateError(f"https:{base}: the POST for {name!r} answered a page with no link "
                        f"matching {link!r}; the form or the page changed")
    url = urllib.parse.urljoin(post, m.group(0))
    if not url.startswith("https://"):
        raise GateError(f"https:{base}: {name!r}'s generated link {url!r} is not https://")
    return url


def probe_https(base_url: str, names: list[str] | None = None, *,
                entries: list[dict] | None = None) -> HostRecord:
    """A plain HTTPS directory -> HostRecord: each of `names` is HEADed under `base_url`; licence is "unknown".

    The checksum slot holds an entry's stated `md5`/`sha256`, else `http-last-modified:` evidence. An entry in
    `entries` may carry its own https `url`, or a `request` (`generated_archive_url`), whose date is not kept.
    """
    base = base_url.rstrip("/")
    if not base.startswith("https://"):
        raise GateError(f"https record {base_url!r} must be an https:// URL")
    if not names:
        raise GateError(f"https:{base}: a plain web server cannot list its files, so the "
                        "block must name them (`files:`); nothing to probe")
    by_name = {e["name"]: e for e in (entries or []) if isinstance(e, dict) and e.get("name")}
    files: list[RemoteFile] = []
    for name in names:
        if any(ch in name for ch in "*?["):
            raise GateError(f"https:{base}: {name!r} is a glob, and a web server cannot "
                            "expand one -- name each file")
        entry = by_name.get(name, {})
        request = entry.get("request")
        if request:
            url = generated_archive_url(base, name, request)
        else:
            url = str(entry.get("url") or f"{base}/{urllib.parse.quote(name)}")
        if not url.startswith("https://"):
            raise GateError(f"https:{base}: {name!r} names the url {url!r}, which is not https://")
        stated = next((f"{algo}:{entry[algo]}" for algo in ("sha256", "md5") if entry.get(algo)), None)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                headers = resp.headers
                status = resp.status
        except urllib.error.HTTPError as exc:
            raise GateError(f"https:{base}: HEAD {name} returned {exc.code}; the file is not "
                            "there under that name") from exc
        length = headers.get("Content-Length")
        if status != 200 or length is None:
            raise GateError(f"https:{base}: HEAD {name} gave status {status} and "
                            f"Content-Length {length!r}; without a size the budget cannot "
                            "be checked before the bytes move")
        modified = headers.get("Last-Modified")
        if request:
            modified = None            # the archive was built for this request; its date is now
        files.append(RemoteFile(
            name=name,
            size_bytes=int(length),
            checksum=stated or (f"http-last-modified:{modified.strip()}" if modified else None),
            url=url,
        ))
    return HostRecord(
        host="https",
        record_id=base,
        api_url=base,
        title=base,
        license="unknown",                     # by construction; see the docstring
        retrieved=datetime.now(UTC).date().isoformat(),
        version=None,
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
    license_override: tuple[str, str] | None = None,
    citation: str | None = None,
) -> tuple[RemoteFile, ...]:
    """Decide whether this fetch may proceed; returns the selected files or raises `GateError`.

    Checks the licence (`license_override` fills a host's "unknown", never contradicts it), the budget (the
    selection on `download`, the aggregate on `stream`), checksums, and free space at `space_dest`, else `dest`.
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

    effective_license = record.license
    if license_override:
        override, source = license_override
        if not source or not str(source).strip():
            raise GateError(
                f"license_override {override!r} for {record.host}:{record.record_id} names no "
                "source. The override IS the verification record; without where the terms "
                "were actually read, it is just an assertion."
            )
        if record.license and record.license != "unknown":
            raise GateError(
                f"{record.host}:{record.record_id} states {record.license!r} but the config "
                f"overrides it to {override!r}. An override fills an absence; it does not "
                "outvote the host. Resolve the conflict at the source."
            )
        effective_license = override

    if effective_license == "unknown" or not effective_license:
        raise GateError(
            f"{record.host}:{record.record_id} declares no license. This is a STOP, not a "
            "warning: unstated terms are more restrictive than a permissive tag, not less. "
            "Resolve with the depositor before fetching."
        )

    if effective_license == UNSTATED:
        if not license_override:
            raise GateError(
                f"{record.host}:{record.record_id} arrived at {UNSTATED!r} without a "
                "license_override. That value is a RECORD OF A SEARCH, not a license a host "
                "can state: it may only be applied by a config block that also names where "
                "the terms were looked for and what sanctions the use instead."
            )
        if route != STREAM and not (citation and str(citation).strip()):
            raise GateError(
                f"{record.host}:{record.record_id} declares {UNSTATED!r} on route={route!r} "
                "with no citation. Unstated terms may be downloaded only by a block that names "
                f"the citation we give the source wherever it is used; otherwise read it on {STREAM!r}."
            )
    elif effective_license not in ACCEPTED_LICENSES:
        raise GateError(
            f"{record.host}:{record.record_id} states {effective_license!r}, which is not on the "
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

    unchecksummed = [f.name for f in files if not is_digest(f.checksum)]
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
            # a stream lands only its aggregate, which `budget_gb` bounds
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
    """HostRecord -> Provenance, carrying the licence the host returned; imports `contract` lazily."""
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
    """Write PROVENANCE.json into `dest` before any data file lands; `route` says what `selected_bytes` counts."""
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 3,
        "route": route,
        # the selection only; the host's catalogue is kept as a count
        "record": {**{k: v for k, v in asdict(record).items() if k != "files"},
                   "host_file_count": len(record.files)},
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
    """What changed upstream since the recorded fetch: route, licence, version, files, sizes, checksums."""
    was = recorded.get("record", {})
    diffs: list[str] = []

    # compared only when the caller states a route and the record carries one
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
