"""Tests for the metadata-first gate.

Every assertion here is a refusal we want to keep. The gate's whole value is
that it says no *before* a download starts, so a gate that silently says yes is
worse than no gate at all -- it would carry the authority of having checked.

The three cases that motivated it are all real: a 559.7 GB record that would
have filled the disk, a file whose host-side checksum was never computed, and
a published repo declaring no license at all.
"""
import json

import pytest

from sidechain.ingest.provenance import (
    GateError,
    HostRecord,
    RemoteFile,
    gate,
    write_provenance,
)


def _record(license="CC-BY-4.0", files=None):
    if files is None:
        files = [
            RemoteFile("small.h5ad", 1_000_000_000, "md5:aaa", "https://x/small"),
            RemoteFile("big.h5ad", 9_000_000_000, "md5:bbb", "https://x/big"),
        ]
    return HostRecord(
        host="zenodo",
        record_id="13350497",
        api_url="https://zenodo.org/api/records/13350497",
        title="test record",
        license=license,
        retrieved="2026-08-16",
        version="1.4",
        files=tuple(files),
    )


# ------------------------------------------------------------------ select --


def test_select_is_strict_about_unknown_names():
    """Skipping a requested file yields a partial corpus that looks whole.

    Same failure shape as a non-strict gene_index: no error, wrong answer.
    """
    with pytest.raises(GateError, match="no file"):
        _record().select(["small.h5ad", "not_there.h5ad"])


def test_select_preserves_requested_order():
    got = _record().select(["big.h5ad", "small.h5ad"])
    assert [f.name for f in got] == ["big.h5ad", "small.h5ad"]


# -------------------------------------------------------------------- gate --


def test_unknown_license_is_a_stop():
    """Arc's State-Replogle-Filtered declares none. Unstated is more
    restrictive than permissive, not less."""
    with pytest.raises(GateError, match="no license"):
        gate(_record(license="unknown"), budget_gb=100)


def test_empty_license_string_is_also_a_stop():
    with pytest.raises(GateError, match="no license"):
        gate(_record(license=""), budget_gb=100)


def test_over_budget_is_a_stop():
    """The X-Atlas case: 559.7 GB discovered from metadata, not from a disk
    that filled up mid-transfer."""
    with pytest.raises(GateError, match="over the"):
        gate(_record(), budget_gb=5.0)


def test_budget_is_measured_on_the_selection_not_the_record():
    # 10 GB record, but we only asked for the 1 GB file.
    got = gate(_record(), budget_gb=2.0, select=["small.h5ad"])
    assert [f.name for f in got] == ["small.h5ad"]


def test_missing_checksum_is_a_stop_by_default():
    files = [RemoteFile("nomd5.h5ad", 1_000, None, "https://x/n")]
    with pytest.raises(GateError, match="no checksum"):
        gate(_record(files=files), budget_gb=100)


def test_missing_checksum_can_be_overridden_deliberately():
    files = [RemoteFile("nomd5.h5ad", 1_000, None, "https://x/n")]
    got = gate(_record(files=files), budget_gb=100, allow_missing_checksum=True)
    assert len(got) == 1


def test_record_with_no_files_is_a_stop():
    with pytest.raises(GateError, match="no files"):
        gate(_record(files=[]), budget_gb=100)


def test_gate_passes_and_returns_the_selection():
    rec = _record()
    got = gate(rec, budget_gb=15.0)
    assert len(got) == 2
    assert sum(f.size_bytes for f in got) == rec.total_bytes


# ------------------------------------------------------------- provenance --


def test_provenance_is_written_before_data_exists(tmp_path):
    """The ordering is the point: written first, it is the thing the fetch is
    checked against; written after, it merely describes what we happened to get.
    """
    dest = tmp_path / "zenodo-13350497"
    rec = _record()
    sel = gate(rec, budget_gb=15.0)
    path = write_provenance(dest, rec, sel)

    assert path.exists()
    assert not list(dest.glob("*.h5ad"))  # no data yet, by construction

    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    assert payload["record"]["license"] == "CC-BY-4.0"
    assert payload["record"]["version"] == "1.4"
    assert payload["selected_bytes"] == rec.total_bytes
    assert len(payload["selected"]) == 2


def test_provenance_flags_license_terms_that_outlive_ingest(tmp_path):
    """ShareAlike constrains redistribution of derived artifacts -- a decision
    that surfaces long after the download, so it is recorded, not remembered."""
    rec = _record(license="CC-BY-NC-SA-4.0")
    payload = json.loads(
        write_provenance(tmp_path / "d", rec, rec.files).read_text()
    )
    assert payload["license_flags"]["noncommercial"] is True
    assert payload["license_flags"]["redistribution_encumbered"] is True


def test_permissive_license_carries_no_flags(tmp_path):
    rec = _record(license="CC-BY-4.0")
    payload = json.loads(
        write_provenance(tmp_path / "d", rec, rec.files).read_text()
    )
    assert payload["license_flags"]["noncommercial"] is False
    assert payload["license_flags"]["redistribution_encumbered"] is False
