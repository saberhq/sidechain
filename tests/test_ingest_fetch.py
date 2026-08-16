"""Tests for the dataset registry and the ingest entry point.

These are contract tests over `configs/datasets.yaml` as much as over the code:
a spec block that forgets to declare a control label is a bug that would only
surface much later, as a delta computed against an empty control pool.
"""
import pytest
import yaml

from sidechain.ingest.fetch import load_datasets, run_gate, select_dataset
from sidechain.ingest.provenance import GateError, HostRecord, RemoteFile, to_provenance

REQUIRED_SPEC_KEYS = {"context", "pert_col", "control_label", "modality", "role"}


# ------------------------------------------------------- the registry file --


def test_every_dataset_block_declares_its_gate_inputs():
    for name, block in load_datasets().items():
        for key in ("host", "record", "budget_gb", "dest", "files"):
            assert key in block, f"{name} is missing {key!r}"
        assert block["files"], f"{name} lists no files"


def test_every_file_declares_a_full_spec():
    """Declared, not derived. HepG2's obs says `Hep-G2` while Arc's splits say
    `hepg2`, so nothing here may be left to a normalization rule."""
    for name, block in load_datasets().items():
        for entry in block["files"]:
            spec = entry.get("spec", {})
            missing = REQUIRED_SPEC_KEYS - set(spec)
            assert not missing, f"{name}/{entry['name']} missing spec keys: {sorted(missing)}"


def test_record_is_pinned_not_a_concept_doi():
    """Zenodo concept DOIs resolve to 'latest' and would change under us."""
    for name, block in load_datasets().items():
        if block["host"] == "zenodo":
            assert "7041848" not in str(block["record"]), (
                f"{name} pins the scPerturb CONCEPT doi, which follows 'latest'"
            )


def test_cas13_dataset_is_excluded_from_training():
    """Cas13 degrades mRNA; CRISPRi represses transcription. Mixing them into
    one holdout confounds modality with context."""
    for block in load_datasets().values():
        for entry in block["files"]:
            spec = entry.get("spec", {})
            if spec.get("modality") == "cas13":
                assert spec["role"] == "excluded", (
                    f"{entry['name']} is cas13 but role={spec['role']!r}"
                )


def test_unknown_dataset_is_refused():
    with pytest.raises(GateError, match="no dataset"):
        select_dataset("does_not_exist")


def test_disabled_dataset_is_refused(tmp_path):
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(yaml.safe_dump({"datasets": [{"name": "shelved", "enabled": False}]}))
    with pytest.raises(GateError, match="disabled"):
        select_dataset("shelved", cfg)


# ------------------------------------------------------------- the bridge --


def _record(license="CC-BY-4.0"):
    return HostRecord(
        host="zenodo", record_id="13350497",
        api_url="https://zenodo.org/api/records/13350497",
        title="t", license=license, retrieved="2026-08-16", version="1.4",
        files=(RemoteFile("a.h5ad", 10, "md5:aa", "https://x/a"),),
    )


def test_to_provenance_carries_the_licence_the_api_returned():
    """Without this bridge the license on a HarmonizedDataset is retyped from
    memory, and stops being evidence."""
    prov = to_provenance(_record("CC-BY-NC-SA-4.0"))
    assert prov.license == "CC-BY-NC-SA-4.0"
    assert prov.source == "zenodo"
    assert prov.accession == "13350497"
    assert prov.retrieved == "2026-08-16"


def test_license_drift_between_config_and_host_is_refused(tmp_path, monkeypatch):
    """If the host's stated terms stop matching what we agreed to, that is a
    stop -- not a warning logged into a file nobody reads."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "zenodo", lambda _: _record("CC-BY-NC-4.0"))
    block = {
        "name": "d", "host": "zenodo", "record": "1", "budget_gb": 1.0,
        "dest": "external/d", "license": "CC-BY-4.0",
        "files": [{"name": "a.h5ad", "spec": {}}],
    }
    with pytest.raises(GateError, match="license changed"):
        run_gate(block, tmp_path)


def test_gate_writes_provenance_before_any_data_exists(tmp_path, monkeypatch):
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "zenodo", lambda _: _record())
    block = {
        "name": "d", "host": "zenodo", "record": "1", "budget_gb": 1.0,
        "dest": "external/d", "license": "CC-BY-4.0",
        "files": [{"name": "a.h5ad", "spec": {"context": "k562"}}],
    }
    _, selected, dest = run_gate(block, tmp_path)
    assert (dest / "PROVENANCE.json").exists()
    assert not list(dest.glob("*.h5ad"))
    assert len(selected) == 1


def test_unknown_host_is_refused(tmp_path):
    block = {"name": "d", "host": "ftp-somewhere", "record": "1", "budget_gb": 1.0,
             "dest": "d", "files": [{"name": "a"}]}
    with pytest.raises(GateError, match="no probe for host"):
        run_gate(block, tmp_path)
