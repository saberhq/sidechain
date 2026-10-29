"""`probe_https`, a plain web server as a host, and `fetch.verify` on its evidence.

    .venv/bin/python -m pytest -q tests/test_ingest_probe_https.py

Writes temporary files only. Notes: private/code-notes/tests/test_ingest_probe_https.py.md
"""
from __future__ import annotations

import email.utils
import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from sidechain.ingest import fetch as fetch_mod
from sidechain.ingest.provenance import GateError, HostRecord, RemoteFile, is_digest, probe_https

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
    # without the written exception the block is refused: a Last-Modified stamp is not a digest
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


# ---------------------------------------------------------------- per-file url and digest --
# A block's entry may give its file its own url and md5.

def test_an_entry_gives_its_file_its_own_url_and_a_stated_digest():
    entries = [{"name": "ENCFF000AAA.tsv", "url": "https://host.org/files/ENCFF000AAA/@@download/ENCFF000AAA.tsv",
                "md5": "9a481eeee53a776b416614332a54caac"},
               {"name": "b.zip"}]
    seen = []

    def _urlopen(req, timeout=0):
        assert req.get_method() == "HEAD"
        seen.append(req.full_url)
        return _Resp(200, {"Content-Length": "7", "Last-Modified": "Sat, 22 Mar 2025 22:56:54 GMT"})

    with patch("urllib.request.urlopen", _urlopen):
        rec = probe_https(BASE, ["ENCFF000AAA.tsv", "b.zip"], entries=entries)
    a, b = rec.files
    assert a.url == entries[0]["url"] and seen[0] == entries[0]["url"]
    assert a.checksum == "md5:9a481eeee53a776b416614332a54caac" and is_digest(a.checksum)
    # an entry without a url or digest reads as before: base/<name>, Last-Modified as evidence
    assert b.url == f"{BASE}/b.zip" and b.checksum.startswith("http-last-modified:")
    assert not is_digest(b.checksum)


def test_an_entry_url_must_be_https():
    with patch("urllib.request.urlopen", _urlopen_factory({"a.zip": 10})):
        with pytest.raises(GateError, match="not https"):
            probe_https(BASE, ["a.zip"], entries=[{"name": "a.zip", "url": "http://host.org/a.zip"}])


def test_a_stated_digest_admits_the_block_without_the_written_exception(tmp_path, monkeypatch):
    monkeypatch.setattr("sidechain.ingest.provenance.shutil.disk_usage",
                        lambda p: type("U", (), {"free": 500 * 10**9})())
    block = _block(tmp_path, allow_missing=False)
    block["files"] = [{"name": "a.zip", "url": "https://host.org/files/X/@@download/a.zip",
                       "md5": "d41d8cd98f00b204e9800998ecf8427e"}]
    with patch("urllib.request.urlopen", _urlopen_factory({"a.zip": 10})):
        record, selected, dest = fetch_mod.run_gate(block, tmp_path, config="configs/data_sources.yaml")
    prov = json.loads((dest / "PROVENANCE.json").read_text())
    assert "allow_missing_checksum" not in prov["notes"]
    assert prov["selected"][0]["checksum"] == "md5:d41d8cd98f00b204e9800998ecf8427e"
    assert prov["selected"][0]["url"] == "https://host.org/files/X/@@download/a.zip"


def test_a_block_may_list_its_files_in_a_checked_in_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("sidechain.ingest.provenance.shutil.disk_usage",
                        lambda p: type("U", (), {"free": 500 * 10**9})())
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([{"name": "a.zip", "url": "https://host.org/files/X/@@download/a.zip",
                                     "md5": "d41d8cd98f00b204e9800998ecf8427e"}]))
    block = _block(tmp_path, allow_missing=False)
    del block["files"]
    block["files_manifest"] = str(manifest)
    assert fetch_mod.block_files(block)[0]["name"] == "a.zip"
    with patch("urllib.request.urlopen", _urlopen_factory({"a.zip": 10})):
        record, selected, dest = fetch_mod.run_gate(block, tmp_path, config="configs/data_sources.yaml")
    prov = json.loads((dest / "PROVENANCE.json").read_text())
    assert prov["notes"]["files_manifest"] == str(manifest)
    assert [f["name"] for f in prov["selected"]] == ["a.zip"]
    # one listing per block, and a manifest that is not a list of named files is refused
    both = dict(block, files=[{"name": "a.zip"}])
    with pytest.raises(GateError, match="one listing"):
        fetch_mod.block_files(both)
    manifest.write_text(json.dumps({"name": "a.zip"}))
    with pytest.raises(GateError, match="JSON list"):
        fetch_mod.block_files(block)


# ------------------------------------------------- a host-built archive --

GEN_REQUEST = {"post": "https://example.org/bulk_archive.php",
               "form": {"selSpec": "Homo_sapiens", "Spec[]": ["RBP_Information", "PFMs"],
                        "submit": "Download Species Archive!"},
               "link": r'tmp/[^"<> ]+\.zip'}


class _Page(_Resp):
    def __init__(self, body: bytes, ctype="text/html; charset=UTF-8"):
        super().__init__(200, {"Content-Type": ctype})
        self._body = body

    def read(self, n=-1):
        return self._body if n < 0 else self._body[:n]


def _generating_host(page: bytes, size=248765, modified="Fri, 16 Oct 2026 22:24:54 GMT", seen=None):
    def _urlopen(req, timeout=0):
        if req.get_method() == "POST":
            if seen is not None:
                seen.append(req.data)
            assert req.full_url == GEN_REQUEST["post"]
            return _Page(page)
        assert req.get_method() == "HEAD", "the probe must not GET a file"
        if seen is not None:
            seen.append(req.full_url)
        if not req.full_url.endswith(".zip"):
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO())
        return _Resp(200, {"Content-Length": str(size), "Last-Modified": modified})
    return _urlopen


def test_a_request_entry_posts_the_form_heads_the_generated_link_and_records_no_date():
    page = b'<html>Please click the link <a href="tmp/Homo_sapiens_2026_10_16_10:23_pm.zip">Download</a></html>'
    seen: list = []
    entries = [{"name": "human.zip", "request": GEN_REQUEST}]
    with patch("urllib.request.urlopen", _generating_host(page, seen=seen)):
        rec = probe_https("https://example.org", ["human.zip"], entries=entries)
    f = rec.files[0]
    assert f.name == "human.zip" and f.size_bytes == 248765
    assert f.url == "https://example.org/tmp/Homo_sapiens_2026_10_16_10:23_pm.zip"
    assert f.checksum is None                    # no evidence for a generated archive
    body = seen[0].decode()
    assert "selSpec=Homo_sapiens" in body and body.count("Spec%5B%5D=") == 2 and "submit=" in body
    assert seen[1] == f.url                      # the HEAD went to the generated link


def test_a_request_entry_refuses_a_page_without_the_link_a_non_page_and_a_bad_request():
    entries = [{"name": "human.zip", "request": GEN_REQUEST}]
    with patch("urllib.request.urlopen", _generating_host(b"<html>Sorry, try again</html>")):
        with pytest.raises(GateError, match="no link"):
            probe_https("https://example.org", ["human.zip"], entries=entries)

    def _zip_answer(req, timeout=0):
        return _Page(b"PK\x03\x04...", ctype="application/zip")
    with patch("urllib.request.urlopen", _zip_answer):
        with pytest.raises(GateError, match="not the page"):
            probe_https("https://example.org", ["human.zip"], entries=entries)
    bad = [{"name": "human.zip", "request": {"post": "http://example.org/x", "form": {}, "link": "a"}}]
    with pytest.raises(GateError, match="https `post`"):
        probe_https("https://example.org", ["human.zip"], entries=bad)


def test_run_gate_records_the_request_in_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr("sidechain.ingest.provenance.shutil.disk_usage",
                        lambda p: type("U", (), {"free": 500 * 10**9})())
    block = {**_block(tmp_path), "license": "UNSTATED", "citation": "Someone 2026",
             "license_override_source": "searched, nothing stated",
             "files": [{"name": "human.zip", "request": GEN_REQUEST}]}
    page = b'<a href="tmp/Homo_sapiens_x.zip">Download</a>'
    with patch("urllib.request.urlopen", _generating_host(page)):
        record, selected, dest = fetch_mod.run_gate(block, tmp_path, config="configs/data_sources.yaml")
    prov = json.loads((dest / "PROVENANCE.json").read_text())
    assert prov["notes"]["generated"] == {"human.zip": GEN_REQUEST}
    assert prov["selected"][0]["checksum"] is None and prov["selected"][0]["size_bytes"] == 248765
    assert prov["license_flags"] == {"noncommercial": True, "redistribution_encumbered": True}
    # the same request a second time: the generated link differs, the size does not -> no drift
    with patch("urllib.request.urlopen", _generating_host(b'<a href="tmp/Homo_sapiens_y.zip">D</a>')):
        fetch_mod.run_gate(block, tmp_path, config="configs/data_sources.yaml")
