"""Contract tests for ``probe_https`` -- the sixth host, a plain web server (T102, TargetScan).

A web server has no record API, so three things the other probes take from the host come
from the block instead, and each is a refusal we want to keep:

* the block's ``files`` list IS the listing -- a probe with no names has nothing to HEAD;
* a name the server answers 404 to is refused HERE, before any download;
* ``Content-Length`` is required, because without it the budget cannot be checked before
  the bytes move -- which is the whole order ADR 0003 exists for.

And one thing it records rather than invents: ``Last-Modified`` goes in the checksum slot
under ``http-last-modified:`` -- evidence a later probe can diff, never a digest a check
could hash a file into (same shape as ``probe_s3``'s multipart ETag). ``fetch.verify``
therefore has to treat such a value as "no digest" instead of raising inside a check.
"""
from __future__ import annotations

import email.utils
import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from sidechain.ingest import fetch as fetch_mod
from sidechain.ingest.provenance import GateError, HostRecord, RemoteFile, probe_https

BASE = "https://example.org/vert_80/vert_80_data_download"


class _Resp:
    def __init__(self, status=200, headers=None):
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _urlopen_factory(sizes: dict[str, int], modified: str | None = "Sat, 22 Mar 2025 22:56:54 GMT",
                     drop_length: bool = False):
    def _urlopen(req, timeout=0):
        assert req.get_method() == "HEAD", "the probe must not GET a file"
        name = req.full_url.rsplit("/", 1)[-1]
        if name not in sizes:
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO())
        headers = {} if drop_length else {"Content-Length": str(sizes[name])}
        if modified:
            headers["Last-Modified"] = modified
        return _Resp(200, headers)
    return _urlopen


def test_probe_records_size_and_last_modified_as_evidence():
    with patch("urllib.request.urlopen", _urlopen_factory({"a.zip": 10, "b.zip": 20})):
        rec = probe_https(BASE, ["b.zip", "a.zip"])
    assert rec.host == "https" and rec.license == "unknown" and rec.version is None
    assert [f.name for f in rec.files] == ["b.zip", "a.zip"]          # the block's order
    assert rec.files[0].size_bytes == 20
    assert rec.files[0].checksum == "http-last-modified:Sat, 22 Mar 2025 22:56:54 GMT"
    assert rec.files[0].url == f"{BASE}/b.zip"


def test_probe_refuses_no_names_glob_404_and_missing_length():
    with pytest.raises(GateError, match="cannot list"):
        probe_https(BASE, [])
    with pytest.raises(GateError, match="glob"):
        probe_https(BASE, ["*.zip"])
    with pytest.raises(GateError, match="https://"):
        probe_https("http://example.org/x", ["a.zip"])
    with patch("urllib.request.urlopen", _urlopen_factory({"a.zip": 10})):
        with pytest.raises(GateError, match="404"):
            probe_https(BASE, ["missing.zip"])
    with patch("urllib.request.urlopen", _urlopen_factory({"a.zip": 10}, drop_length=True)):
        with pytest.raises(GateError, match="Content-Length"):
            probe_https(BASE, ["a.zip"])


def test_no_last_modified_means_no_checksum_at_all():
    with patch("urllib.request.urlopen", _urlopen_factory({"a.zip": 10}, modified=None)):
        rec = probe_https(BASE, ["a.zip"])
    assert rec.files[0].checksum is None


def _block(tmp_path, allow_missing=True):
    return {
        "name": "targetscan_test", "host": "https", "record": BASE,
        "dest": "external/x", "budget_gb": 0.001,
        "license": "Free-for-research-with-citation",
        "license_override_source": "the publisher's FAQ, read on a date",
        "allow_missing_checksum": allow_missing,
        "files": [{"name": "a.zip"}],
    }


def test_run_gate_admits_only_with_the_written_exception(tmp_path, monkeypatch):
    monkeypatch.setattr("sidechain.ingest.provenance.shutil.disk_usage",
                        lambda p: type("U", (), {"free": 500 * 10**9})())
    with patch("urllib.request.urlopen", _urlopen_factory({"a.zip": 10})):
        record, selected, dest = fetch_mod.run_gate(_block(tmp_path), tmp_path,
                                                    config="configs/data_sources.yaml")
    prov = json.loads((dest / "PROVENANCE.json").read_text())
    assert prov["notes"]["allow_missing_checksum"] is True
    assert prov["notes"]["config"] == "configs/data_sources.yaml"
    assert prov["record"]["license"] == "Free-for-research-with-citation"
    assert prov["license_flags"] == {"noncommercial": False, "redistribution_encumbered": False}
    # the same block WITHOUT the written exception is refused: the host publishes no digest,
    # and a Last-Modified stamp in the checksum slot does not count as one (it did, once)
    with patch("urllib.request.urlopen", _urlopen_factory({"a.zip": 10})):
        with pytest.raises(GateError, match="no checksum"):
            fetch_mod.run_gate(_block(tmp_path / "two", allow_missing=False), tmp_path / "two")
    with patch("urllib.request.urlopen", _urlopen_factory({"a.zip": 10}, modified=None)):
        with pytest.raises(GateError, match="no checksum"):
            fetch_mod.run_gate(_block(tmp_path / "three", allow_missing=False), tmp_path / "three")


def test_verify_treats_evidence_as_no_digest_not_an_error(tmp_path, capsys):
    (tmp_path / "a.zip").write_bytes(b"0123456789")
    files = (RemoteFile("a.zip", 10, "http-last-modified:Sat, 22 Mar 2025 22:56:54 GMT", "u"),
             RemoteFile("missing.zip", 5, "s3-etag:abc-2", "u"))
    rc = fetch_mod.verify(files, tmp_path)
    out = capsys.readouterr().out
    assert "no-sum   a.zip (host offers http-last-modified evidence" in out
    assert "MISSING  missing.zip" in out
    assert rc == 1                                   # the missing file, not the evidence, fails it


def test_evidence_is_not_a_digest_for_the_gate():
    from sidechain.ingest.provenance import gate, is_digest
    assert is_digest("md5:abc") and is_digest("sha256:abc")
    assert not is_digest("s3-etag:abc-2") and not is_digest("http-last-modified:Sat, 1 Jan 2025 00:00:00 GMT")
    assert not is_digest(None) and not is_digest("")
    rec = HostRecord(host="s3", record_id="b", api_url="u", title="b", license="CC-BY-4.0",
                     retrieved="2026-09-26", files=(RemoteFile("x", 1, "s3-etag:abc-2", "u"),))
    with pytest.raises(GateError, match="no checksum"):
        gate(rec, budget_gb=1.0)
    assert gate(rec, budget_gb=1.0, allow_missing_checksum=True)


def test_last_modified_parses_as_a_date():
    # The value is kept verbatim; this pins that a verbatim RFC 1123 stamp is what lands.
    stamp = "Sat, 22 Mar 2025 22:56:54 GMT"
    assert email.utils.parsedate_to_datetime(stamp).year == 2025
    rec = HostRecord(host="https", record_id=BASE, api_url=BASE, title=BASE, license="unknown",
                     retrieved="2026-09-25",
                     files=(RemoteFile("a", 1, f"http-last-modified:{stamp}", "u"),))
    assert rec.files[0].checksum.split(":", 1)[1] == stamp
