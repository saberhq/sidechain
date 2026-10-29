"""`probe_s3`, a public S3 bucket as a host: ETags, listing, pagination, refusals.

    .venv/bin/python -m pytest -q tests/test_ingest_probe_s3.py

Writes temporary files only. Notes: private/code-notes/tests/test_ingest_probe_s3.py.md
"""
from __future__ import annotations

import urllib.error
from unittest.mock import patch

import pytest

from sidechain.ingest.provenance import (
    GateError,
    HostRecord,
    _s3_etag_checksum,
    probe_s3,
)

NS = 'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'


def _listing(contents: str, *, truncated: bool = False, token: str | None = None) -> bytes:
    tok = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f"<ListBucketResult {NS}>"
        f"<Name>b</Name>"
        f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>"
        f"{tok}{contents}"
        f"</ListBucketResult>"
    ).encode()


def _obj(key: str, size: int, etag: str | None = None) -> str:
    tag = f"<ETag>&quot;{etag}&quot;</ETag>" if etag else ""
    return (f"<Contents><Key>{key}</Key><Size>{size}</Size>{tag}"
            f"<LastModified>2026-05-28T00:00:00.000Z</LastModified>"
            f"<StorageClass>STANDARD</StorageClass></Contents>")


class _Resp:
    def __init__(self, payload: bytes):
        self._p = payload

    def read(self):
        return self._p

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _urlopen_returning(*pages: bytes):
    seq = list(pages)
    calls: list[str] = []

    def fake(req, timeout=None):
        calls.append(req.full_url)
        return _Resp(seq.pop(0))

    fake.calls = calls
    return fake


# ------------------------------------------------------------------ the checksum

def test_a_multipart_etag_is_never_labelled_md5():
    """A multipart ETag is recorded as `s3-etag:`, never `md5:`."""
    out = _s3_etag_checksum('"c9ff52fcc6d6ce8a387a76dc757a5b97-2002"')
    assert out == "s3-etag:c9ff52fcc6d6ce8a387a76dc757a5b97-2002"
    assert not out.startswith("md5:")


def test_a_single_part_etag_is_a_real_md5():
    assert _s3_etag_checksum('"0123456789ABCDEF0123456789abcdef"') == \
        "md5:0123456789abcdef0123456789abcdef"


@pytest.mark.parametrize("etag", [None, "", '""', "   "])
def test_a_missing_etag_stays_none(etag):
    """A missing or blank ETag is None."""
    assert _s3_etag_checksum(etag) is None


def test_a_non_hex_etag_is_not_promoted_to_md5():
    assert _s3_etag_checksum('"zzzz"') == "s3-etag:zzzz"


# ------------------------------------------------------------------ the listing

def test_probe_reads_keys_sizes_and_checksums():
    page = _listing(
        _obj("marson2025_data/GWCD4i.DE_stats.h5ad", 16786240107,
             "c9ff52fcc6d6ce8a387a76dc757a5b97-2002")
        + _obj("marson2025_data/suppl_tables/small.csv", 969086,
               "0123456789abcdef0123456789abcdef"))
    with patch("urllib.request.urlopen", _urlopen_returning(page)):
        rec = probe_s3("genome-scale-tcell-perturb-seq/marson2025_data/")

    assert isinstance(rec, HostRecord)
    assert rec.host == "s3"
    assert rec.version is None            # a bucket has no record version to pin
    assert rec.total_bytes == 16786240107 + 969086
    by = {f.name: f for f in rec.files}
    assert set(by) == {"GWCD4i.DE_stats.h5ad", "suppl_tables/small.csv"}
    assert by["GWCD4i.DE_stats.h5ad"].checksum.startswith("s3-etag:")
    assert by["suppl_tables/small.csv"].checksum.startswith("md5:")
    assert by["GWCD4i.DE_stats.h5ad"].url == (
        "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/"
        "marson2025_data/GWCD4i.DE_stats.h5ad")


def test_the_prefix_is_stripped_so_config_names_stay_short():
    """File names are relative to the probed prefix."""
    page = _listing(_obj("marson2025_data/x.h5ad", 10, "a" * 32))
    with patch("urllib.request.urlopen", _urlopen_returning(page)):
        rec = probe_s3("bkt/marson2025_data/")
    assert rec.files[0].name == "x.h5ad"


def test_no_prefix_keeps_the_whole_key():
    page = _listing(_obj("a/b/c.h5ad", 10, "a" * 32))
    with patch("urllib.request.urlopen", _urlopen_returning(page)):
        rec = probe_s3("bkt")
    assert rec.files[0].name == "a/b/c.h5ad"


def test_pagination_is_followed_to_the_end():
    """A truncated listing is followed through its continuation token."""
    p1 = _listing(_obj("p/a.h5ad", 1, "a" * 32), truncated=True, token="TOKEN-2")
    p2 = _listing(_obj("p/b.h5ad", 2, "b" * 32))
    fake = _urlopen_returning(p1, p2)
    with patch("urllib.request.urlopen", fake):
        rec = probe_s3("bkt/p/")
    assert [f.name for f in rec.files] == ["a.h5ad", "b.h5ad"]
    assert len(fake.calls) == 2
    assert "continuation-token=TOKEN-2" in fake.calls[1]


def test_truncated_without_a_token_stops_rather_than_looping():
    page = _listing(_obj("p/a.h5ad", 1, "a" * 32), truncated=True)   # no token
    with patch("urllib.request.urlopen", _urlopen_returning(page)):
        rec = probe_s3("bkt/p/")
    assert len(rec.files) == 1


def test_folder_markers_are_not_files():
    page = _listing(_obj("p/", 0) + _obj("p/real.h5ad", 5, "a" * 32))
    with patch("urllib.request.urlopen", _urlopen_returning(page)):
        rec = probe_s3("bkt/p/")
    assert [f.name for f in rec.files] == ["real.h5ad"]


def test_files_come_back_sorted_so_provenance_is_stable():
    page = _listing(_obj("p/z.h5ad", 1, "a" * 32) + _obj("p/a.h5ad", 1, "b" * 32))
    with patch("urllib.request.urlopen", _urlopen_returning(page)):
        rec = probe_s3("bkt/p/")
    assert [f.name for f in rec.files] == ["a.h5ad", "z.h5ad"]


# ------------------------------------------------------------------ the refusals

def test_a_bucket_never_states_a_licence():
    """An S3 record's licence is `unknown`."""
    page = _listing(_obj("p/a.h5ad", 1, "a" * 32))
    with patch("urllib.request.urlopen", _urlopen_returning(page)):
        rec = probe_s3("bkt/p/")
    assert rec.license == "unknown"


def test_an_empty_listing_is_an_error_not_an_empty_corpus():
    with patch("urllib.request.urlopen", _urlopen_returning(_listing(""))):
        with pytest.raises(GateError, match="no objects"):
            probe_s3("bkt/nope/")


def test_a_record_with_no_bucket_is_refused():
    with pytest.raises(GateError, match="names no bucket"):
        probe_s3("/just-a-prefix")


def test_select_still_globs_over_an_s3_record():
    """`select` globs, and refuses an absent name, on an S3 record."""
    page = _listing(_obj("p/suppl_tables/a.csv", 1, "a" * 32)
                    + _obj("p/suppl_tables/b.csv", 1, "b" * 32)
                    + _obj("p/big.h5ad", 9, "c" * 32))
    with patch("urllib.request.urlopen", _urlopen_returning(page)):
        rec = probe_s3("bkt/p/")
    assert len(rec.select(["suppl_tables/*.csv"])) == 2
    assert len(rec.select(["big.h5ad"])) == 1
    with pytest.raises(GateError, match="no file"):
        rec.select(["does-not-exist.h5ad"])


def test_probe_s3_is_registered_as_a_host():
    from sidechain.ingest.fetch import PROBES
    assert PROBES["s3"] is probe_s3


# ------------------------------------------------------------------ live (opt-in)

@pytest.mark.skipif(
    not __import__("os").environ.get("SIDECHAIN_NETWORK_TESTS"),
    reason="set SIDECHAIN_NETWORK_TESTS=1 to hit the live GWCD4i bucket")
def test_live_gwcd4i_bucket_still_lists():
    """The live bucket still lists the DE file at its recorded size; opt-in."""
    try:
        rec = probe_s3("genome-scale-tcell-perturb-seq/marson2025_data/")
    except urllib.error.URLError as exc:            # offline is not a failure
        pytest.skip(f"network unavailable: {exc}")
    by = {f.name: f for f in rec.files}
    assert "GWCD4i.DE_stats.h5ad" in by
    assert by["GWCD4i.DE_stats.h5ad"].size_bytes == 16786240107
    assert by["GWCD4i.DE_stats.h5ad"].checksum.startswith("s3-etag:")
    assert rec.license == "unknown"
