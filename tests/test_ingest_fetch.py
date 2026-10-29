"""The dataset registry `configs/datasets.yaml` and the ingest gate `sidechain.ingest.fetch`.

    .venv/bin/python -m pytest -q tests/test_ingest_fetch.py

Writes temporary files only. Notes: private/code-notes/tests/test_ingest_fetch.py.md
"""
import json
import shutil

import pytest
import yaml

from sidechain.ingest.fetch import load_datasets, run_gate, select_dataset
from sidechain.ingest.provenance import (
    GateError,
    HostRecord,
    RemoteFile,
    gate,
    to_provenance,
)

# Every test here sees a volume with this much free space.
FAKE_FREE_BYTES = 500 * 10**9


@pytest.fixture(autouse=True)
def _roomy_disk(monkeypatch):
    usage = shutil._ntuple_diskusage(
        total=2 * FAKE_FREE_BYTES, used=FAKE_FREE_BYTES, free=FAKE_FREE_BYTES
    )
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: usage)


# What an expression file's spec must declare.
REQUIRED_SPEC_KEYS = {"context", "pert_col", "control_label", "modality", "role"}

# The spec keys each file kind must declare.
REQUIRED_SPEC_KEYS_BY_KIND = {
    "expression": REQUIRED_SPEC_KEYS,
    "gene_map": {"role"},
    "lfc_table": {"target_col", "gene_id_col", "effect_col", "pvalue_col", "modality", "role"},
    "cell_metadata": {"role"},
    "sample_metadata": {"role"},
    "observational": {"context", "context_col", "sample_from", "sample_id_col",
                      "assay", "content_filter", "role"},
}

# Keys a kind may not declare at all: absent, not null.
FORBIDDEN_SPEC_KEYS_BY_KIND = {
    "observational": {"pert_col", "control_label"},
}

# Kinds that must name their context.
CONTEXTUAL_KINDS = {"expression", "lfc_table", "observational"}

# The roles a file may declare.
KNOWN_ROLES = {"train", "analysis", "reference", "excluded", "prior"}


# ------------------------------------------------------- the registry file --


def test_every_dataset_block_declares_its_gate_inputs():
    for name, block in load_datasets().items():
        for key in ("host", "record", "budget_gb", "dest", "files"):
            assert key in block, f"{name} is missing {key!r}"
        assert block["files"], f"{name} lists no files"


def test_every_file_declares_a_full_spec():
    """Every file declares the spec keys its kind requires."""
    for name, block in load_datasets().items():
        for entry in block["files"]:
            kind = entry.get("kind", "expression")
            assert kind in REQUIRED_SPEC_KEYS_BY_KIND, (
                f"{name}/{entry['name']} declares unknown kind {kind!r}"
            )
            spec = entry.get("spec", {})
            missing = REQUIRED_SPEC_KEYS_BY_KIND[kind] - set(spec)
            assert not missing, f"{name}/{entry['name']} missing spec keys: {sorted(missing)}"


def test_every_contextual_file_names_its_context_exactly_once():
    """A contextual file sets exactly one of `context` and `context_col`."""
    for name, block in load_datasets().items():
        for entry in block["files"]:
            if entry.get("kind", "expression") not in CONTEXTUAL_KINDS:
                continue
            spec = entry.get("spec", {})
            named = spec.get("context") is not None
            by_col = spec.get("context_col") is not None
            assert named != by_col, (
                f"{name}/{entry['name']} must set exactly one of context / context_col; "
                f"got context={spec.get('context')!r}, context_col={spec.get('context_col')!r}"
            )


def test_no_file_declares_a_spec_key_its_kind_cannot_answer():
    """No file declares a spec key its kind forbids."""
    for name, block in load_datasets().items():
        for entry in block["files"]:
            kind = entry.get("kind", "expression")
            forbidden = FORBIDDEN_SPEC_KEYS_BY_KIND.get(kind, set())
            present = forbidden & set(entry.get("spec", {}))
            assert not present, (
                f"{name}/{entry['name']} is kind {kind!r} and declares {sorted(present)}. "
                "Remove the keys rather than setting them null."
            )


def test_every_file_declares_a_known_role():
    for name, block in load_datasets().items():
        for entry in block["files"]:
            role = entry.get("spec", {}).get("role")
            assert role in KNOWN_ROLES, (
                f"{name}/{entry['name']} declares role {role!r}; known: {sorted(KNOWN_ROLES)}"
            )


def test_every_observational_file_declares_a_real_content_filter():
    """An observational file declares a non-empty `content_filter`."""
    for name, block in load_datasets().items():
        for entry in block["files"]:
            if entry.get("kind") != "observational":
                continue
            cf = entry.get("spec", {}).get("content_filter")
            assert isinstance(cf, dict) and cf, (
                f"{name}/{entry['name']} declares content_filter={cf!r}. It must be a "
                "non-empty mapping."
            )


def test_a_sidecar_join_names_the_key_on_both_sides():
    """A file that joins a sample sidecar names `sample_id_col` and `obs_sample_col`."""
    for name, block in load_datasets().items():
        for entry in block["files"]:
            spec = entry.get("spec", {})
            if not spec.get("sample_from"):
                continue
            for key in ("sample_id_col", "obs_sample_col"):
                assert spec.get(key), (
                    f"{name}/{entry['name']} joins a sample sidecar but declares no {key!r}"
                )


def test_a_declared_sidecar_is_a_file_the_gate_actually_selects():
    """An `obs_from` or `sample_from` sidecar is listed in its block's files."""
    for name, block in load_datasets().items():
        listed = {f["name"] for f in block["files"]}
        for entry in block["files"]:
            spec = entry.get("spec", {})
            for key in ("obs_from", "sample_from"):
                sidecar = spec.get(key)
                if not sidecar:
                    continue
                assert sidecar in listed, (
                    f"{name}/{entry['name']} declares {key}={sidecar!r}, which the block "
                    f"does not list. Add it as a file so it is inside the gate's selection."
                )


def test_figshare_records_pin_a_version():
    """A Figshare record carries a `.v<n>` version."""
    for name, block in load_datasets().items():
        if block["host"] == "figshare":
            assert ".v" in str(block["record"]), (
                f"{name} names Figshare article {block['record']!r} without a .v<n> "
                "version; that follows 'latest' and would change under us"
            )


def test_streamed_blocks_declare_where_the_aggregate_goes():
    """A streamed block declares `derived:` under derived/, apart from `dest` under external/."""
    for name, block in load_datasets().items():
        if block.get("route") != "stream":
            continue
        assert "derived" in block, f"{name} streams but declares no `derived:`"
        assert block["dest"] != block["derived"], (
            f"{name} points dest and derived at the same tree"
        )
        assert block["dest"].startswith("external/"), f"{name} dest is not under external/"
        assert block["derived"].startswith("derived/"), f"{name} derived is not under derived/"


def test_no_block_declares_an_unknown_route():
    """Every block declares a known route."""
    from sidechain.ingest.provenance import ROUTES

    for name, block in load_datasets().items():
        route = block.get("route", "download")
        assert route in ROUTES, f"{name} declares route={route!r}; known: {ROUTES}"


def test_the_real_xatlas_block_would_be_refused_as_a_download():
    """The `xatlas_orion` block streams, with a budget below the corpus size."""
    block = load_datasets()["xatlas_orion"]
    assert block["route"] == "stream"
    assert float(block["budget_gb"]) < 126, "budget should be the OUTPUT ceiling, not the corpus"


def test_the_real_scbasecount_block_is_observational_and_streamed():
    """The `scbasecount_human_gene` block: observational, UNSTATED with a source, streamed, and
    refused by the gate as a download.
    """
    from sidechain.ingest.provenance import UNSTATED

    block = load_datasets()["scbasecount_human_gene"]
    assert block["route"] == "stream"
    assert block["license"] == UNSTATED
    assert block["license_override_source"].strip(), "UNSTATED without a source is an assertion"

    counts = next(f for f in block["files"] if f.get("kind") == "observational")
    assert "pert_col" not in counts["spec"] and "control_label" not in counts["spec"]
    assert counts["spec"]["context_col"] == "cell_line", "the counts are split by cell line"
    assert float(block["budget_gb"]) < 2378, "budget is the output ceiling, not the corpus size"

    corpus = HostRecord(
        host="huggingface", record_id=block["record"], api_url="u", title="t",
        license="CC-BY-NC-SA-4.0", retrieved="2026-08-23", version="deadbeef",
        files=(RemoteFile("data/HCT116_Batch1.parquet", 126_260_000_000,
                          "sha256:aa", "https://x/a"),),
    )
    # as a download: refused on size
    with pytest.raises(GateError, match="over the"):
        gate(corpus, budget_gb=float(block["budget_gb"]), route="download")
    # as a stream: allowed
    got = gate(corpus, budget_gb=float(block["budget_gb"]), route="stream")
    assert len(got) == 1


def test_record_is_pinned_not_a_concept_doi():
    """No Zenodo block names the scPerturb concept DOI."""
    for name, block in load_datasets().items():
        if block["host"] == "zenodo":
            assert "7041848" not in str(block["record"]), (
                f"{name} pins the scPerturb CONCEPT doi, which follows 'latest'"
            )


def test_cas13_dataset_is_excluded_from_training():
    """A cas13 file's role is `excluded`."""
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
    """`to_provenance` carries the host's licence, source, accession and retrieval date."""
    prov = to_provenance(_record("CC-BY-NC-SA-4.0"))
    assert prov.license == "CC-BY-NC-SA-4.0"
    assert prov.source == "zenodo"
    assert prov.accession == "13350497"
    assert prov.retrieved == "2026-08-16"


def test_license_drift_between_config_and_host_is_refused(tmp_path, monkeypatch):
    """A host licence that differs from the block's is refused."""
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


# ------------------------------------- provenance is evidence, not a mirror --


def _block(tmp_path, license="CC-BY-4.0"):
    return {
        "name": "d", "host": "zenodo", "record": "1", "budget_gb": 1.0,
        "dest": "external/d", "license": license,
        "files": [{"name": "a.h5ad", "spec": {"context": "k562"}}],
    }


def test_provenance_is_written_once_and_not_rewritten(tmp_path, monkeypatch):
    """A second run with nothing changed upstream leaves PROVENANCE.json as it was."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "zenodo", lambda _: _record())
    block = _block(tmp_path)

    _, _, dest = fetch.run_gate(block, tmp_path)
    path = dest / "PROVENANCE.json"
    first = path.read_text()

    fetch.run_gate(block, tmp_path)  # second run, nothing changed upstream
    assert path.read_text() == first


def test_upstream_checksum_change_is_refused(tmp_path, monkeypatch):
    """A changed upstream checksum for the same file is refused."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "zenodo", lambda _: _record())
    fetch.run_gate(_block(tmp_path), tmp_path)

    changed = HostRecord(
        host="zenodo", record_id="13350497", api_url="u", title="t",
        license="CC-BY-4.0", retrieved="2026-09-01", version="1.4",
        files=(RemoteFile("a.h5ad", 10, "md5:DIFFERENT", "https://x/a"),),
    )
    monkeypatch.setitem(fetch.PROBES, "zenodo", lambda _: changed)
    with pytest.raises(GateError, match="no longer matches the recorded provenance"):
        fetch.run_gate(_block(tmp_path), tmp_path)


def test_upstream_version_change_is_refused(tmp_path, monkeypatch):
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "zenodo", lambda _: _record())
    fetch.run_gate(_block(tmp_path), tmp_path)

    v2 = HostRecord(
        host="zenodo", record_id="13350497", api_url="u", title="t",
        license="CC-BY-4.0", retrieved="2026-09-01", version="2.0",
        files=(RemoteFile("a.h5ad", 10, "md5:aa", "https://x/a"),),
    )
    monkeypatch.setitem(fetch.PROBES, "zenodo", lambda _: v2)
    with pytest.raises(GateError, match="version"):
        fetch.run_gate(_block(tmp_path), tmp_path)


def test_refresh_accepts_an_upstream_change_deliberately(tmp_path, monkeypatch):
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "zenodo", lambda _: _record())
    fetch.run_gate(_block(tmp_path), tmp_path)

    v2 = HostRecord(
        host="zenodo", record_id="13350497", api_url="u", title="t",
        license="CC-BY-4.0", retrieved="2026-09-01", version="2.0",
        files=(RemoteFile("a.h5ad", 99, "md5:bb", "https://x/a"),),
    )
    monkeypatch.setitem(fetch.PROBES, "zenodo", lambda _: v2)
    _, selected, dest = fetch.run_gate(_block(tmp_path), tmp_path, refresh=True)
    assert selected[0].size_bytes == 99
    assert json.loads((dest / "PROVENANCE.json").read_text())["record"]["version"] == "2.0"


def test_retrieved_date_alone_is_not_a_difference(tmp_path, monkeypatch):
    """A new retrieval date alone is not a difference."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "zenodo", lambda _: _record())
    fetch.run_gate(_block(tmp_path), tmp_path)

    later = HostRecord(
        host="zenodo", record_id="13350497", api_url="u", title="t",
        license="CC-BY-4.0", retrieved="2027-01-01", version="1.4",
        files=(RemoteFile("a.h5ad", 10, "md5:aa", "https://x/a"),),
    )
    monkeypatch.setitem(fetch.PROBES, "zenodo", lambda _: later)
    fetch.run_gate(_block(tmp_path), tmp_path)  # must not raise


def test_gate_refuses_when_the_disk_cannot_take_it(tmp_path):
    """The gate refuses a selection larger than the free disk."""
    huge = HostRecord(
        host="zenodo", record_id="1", api_url="u", title="t", license="CC-BY-4.0",
        retrieved="2026-08-16", version="1",
        files=(RemoteFile("big.h5ad", 10**15, "md5:aa", "https://x/b"),),
    )
    with pytest.raises(GateError, match="not enough disk"):
        gate(huge, budget_gb=10**9, dest=tmp_path / "nowhere")


# ------------------------------------------------------- route, end to end --


def _hf_record(files=None, license="CC-BY-NC-SA-4.0"):
    if files is None:
        files = [RemoteFile("data/A_Batch1.parquet", 50_000_000_000, "sha256:aa", "https://x/a")]
    return HostRecord(
        host="huggingface", record_id="Owner/Repo", api_url="u", title="t",
        license=license, retrieved="2026-08-23", version="c0ffee",
        files=tuple(files),
    )


def test_streamed_block_writes_provenance_and_records_the_route(tmp_path, monkeypatch):
    """A streamed block's PROVENANCE.json records the route, the bytes read and the output budget;
    no data lands.
    """
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "huggingface", lambda _: _hf_record())
    block = {
        "name": "x", "host": "huggingface", "record": "Owner/Repo", "budget_gb": 12.0,
        "dest": "external/hf-Owner-Repo", "derived": "derived/x", "route": "stream",
        "license": "CC-BY-NC-SA-4.0",
        "files": [{"name": "data/A_*.parquet", "spec": {"context": "a"}}],
    }
    _, selected, dest = fetch.run_gate(block, tmp_path)
    payload = json.loads((dest / "PROVENANCE.json").read_text())

    assert payload["route"] == "stream"
    assert payload["lands_on_disk"] is False
    assert payload["selected_bytes"] == 50_000_000_000     # what we READ
    assert payload["notes"]["output_budget_gb"] == 12.0     # what we may WRITE
    assert payload["notes"]["derived"] == "derived/x"
    assert not list(dest.glob("*.parquet"))
    assert len(selected) == 1


def test_a_download_block_still_records_route_download(tmp_path, monkeypatch):
    """A block with no `route:` records `route: download`."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "zenodo", lambda _: _record())
    _, _, dest = fetch.run_gate(_block(tmp_path), tmp_path)
    payload = json.loads((dest / "PROVENANCE.json").read_text())
    assert payload["route"] == "download"
    assert payload["lands_on_disk"] is True


def test_a_typod_route_is_refused_not_downloaded(tmp_path, monkeypatch):
    """A misspelt route is refused."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "huggingface", lambda _: _hf_record())
    block = {
        "name": "x", "host": "huggingface", "record": "Owner/Repo", "budget_gb": 12.0,
        "dest": "external/x", "route": "streamed",   # <- typo
        "license": "CC-BY-NC-SA-4.0",
        "files": [{"name": "data/A_*.parquet", "spec": {}}],
    }
    with pytest.raises(GateError, match="unknown route"):
        fetch.run_gate(block, tmp_path)


def test_provenance_records_the_selection_not_the_hosts_whole_catalogue(tmp_path, monkeypatch):
    """PROVENANCE.json lists the selected files and only a count of the host's files."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "huggingface", lambda _: _hf_record(files=[
        RemoteFile(f"data/A_Batch{i}.parquet", 1_000_000, "sha256:aa", "u") for i in range(5)
    ] + [RemoteFile("unrelated/other.parquet", 9, "sha256:bb", "u")]))
    block = {
        "name": "x", "host": "huggingface", "record": "Owner/Repo", "budget_gb": 12.0,
        "dest": "external/x", "derived": "derived/x", "route": "stream",
        "license": "CC-BY-NC-SA-4.0",
        "files": [{"name": "data/A_*.parquet", "spec": {}}],
    }
    _, _, dest = fetch.run_gate(block, tmp_path)
    payload = json.loads((dest / "PROVENANCE.json").read_text())

    assert "files" not in payload["record"], "the host's catalogue is not evidence about our fetch"
    assert payload["record"]["host_file_count"] == 6
    assert len(payload["selected"]) == 5
    # still the fields drift detection actually compares
    assert payload["record"]["license"] == "CC-BY-NC-SA-4.0"
    assert payload["record"]["version"] == "c0ffee"


# ------------------------------------------------- the unstated licence --


def _lamin_record(files=None):
    """A lamin record, whose host states no licence."""
    if files is None:
        files = [RemoteFile("k/a.h5ad", 2_000_000_000, "lamin:aa", "")]
    return HostRecord(
        host="lamin", record_id="owner/inst", api_url="u", title="owner/inst",
        license="unknown", retrieved="2026-08-29", files=tuple(files),
    )


def _unstated_block(**over):
    block = {
        "name": "u", "host": "lamin", "record": "owner/inst", "budget_gb": 2.5,
        "dest": "external/lamin-owner-inst", "derived": "derived/u", "route": "stream",
        "license": "UNSTATED",
        "license_override_source": "checked five publisher-controlled sources, none states terms",
        "files": [{"name": "k/a.h5ad", "kind": "observational", "spec": {}}],
    }
    block.update(over)
    return block


def test_unstated_terms_pass_the_gate_as_a_stream(tmp_path, monkeypatch):
    """UNSTATED with an override source passes the gate as a stream and is recorded as the override."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "lamin", lambda _: _lamin_record())
    _, _, dest = fetch.run_gate(_unstated_block(), tmp_path)
    payload = json.loads((dest / "PROVENANCE.json").read_text())
    assert payload["route"] == "stream"
    assert payload["notes"]["license_override"]["applied"] == "UNSTATED"
    assert payload["notes"]["license_override"]["host_stated"] == "unknown"


def test_unstated_terms_are_recorded_as_maximally_encumbered(tmp_path, monkeypatch):
    """UNSTATED is recorded noncommercial and redistribution-encumbered."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "lamin", lambda _: _lamin_record())
    _, _, dest = fetch.run_gate(_unstated_block(), tmp_path)
    flags = json.loads((dest / "PROVENANCE.json").read_text())["license_flags"]
    assert flags["redistribution_encumbered"] is True
    assert flags["noncommercial"] is True


def test_unstated_terms_are_refused_on_a_download_without_a_citation(tmp_path, monkeypatch):
    """An UNSTATED download block with no citation, or a blank one, is refused."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "lamin", lambda _: _lamin_record())
    block = _unstated_block(route="download", budget_gb=100.0)
    block.pop("derived")
    with pytest.raises(GateError, match="with no citation"):
        fetch.run_gate(block, tmp_path)
    with pytest.raises(GateError, match="with no citation"):
        fetch.run_gate({**block, "citation": "  "}, tmp_path)


def test_unstated_terms_download_with_a_citation_and_stay_maximally_encumbered(tmp_path, monkeypatch):
    """An UNSTATED download with a citation passes, records the citation and keeps both flags."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "lamin", lambda _: _lamin_record())
    block = _unstated_block(route="download", budget_gb=100.0, citation="Author et al. 2020, NAR 48:D174")
    block.pop("derived")
    _, _, dest = fetch.run_gate(block, tmp_path)
    payload = json.loads((dest / "PROVENANCE.json").read_text())
    assert payload["route"] == "download"
    assert payload["notes"]["citation"] == "Author et al. 2020, NAR 48:D174"
    assert payload["notes"]["license_override"]["applied"] == "UNSTATED"
    assert payload["license_flags"]["noncommercial"] is True
    assert payload["license_flags"]["redistribution_encumbered"] is True


def test_unstated_without_an_override_source_is_refused(tmp_path, monkeypatch):
    """UNSTATED without `license_override_source` is refused."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "lamin", lambda _: _lamin_record())
    block = _unstated_block()
    block.pop("license_override_source")
    with pytest.raises(GateError, match="record of a search"):
        fetch.run_gate(block, tmp_path)


def test_the_gate_itself_refuses_unstated_arriving_without_an_override(tmp_path):
    """`gate()` refuses an UNSTATED licence that arrives without an override."""
    from sidechain.ingest.provenance import UNSTATED

    record = HostRecord(
        host="lamin", record_id="owner/inst", api_url="u", title="t",
        license=UNSTATED, retrieved="2026-08-29",
        files=(RemoteFile("k/a.h5ad", 10, "lamin:aa", ""),))
    with pytest.raises(GateError, match="without a license_override"):
        gate(record, budget_gb=1.0, route="stream", dest=tmp_path)


def test_unstated_cannot_paper_over_a_host_that_did_state_terms(tmp_path, monkeypatch):
    """UNSTATED is refused against a host that states a licence."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "lamin", lambda _: HostRecord(
        host="lamin", record_id="owner/inst", api_url="u", title="t",
        license="CC-BY-NC-SA-4.0", retrieved="2026-08-29",
        files=(RemoteFile("k/a.h5ad", 10, "lamin:aa", ""),)))
    with pytest.raises(GateError, match="does not outvote the host"):
        fetch.run_gate(_unstated_block(), tmp_path)


def test_a_stated_but_unrecognised_licence_is_still_refused(tmp_path, monkeypatch):
    """A stated licence off the accepted list (`other`) is refused."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "huggingface", lambda _: _hf_record(license="other"))
    block = {
        "name": "x", "host": "huggingface", "record": "Owner/Repo", "budget_gb": 12.0,
        "dest": "external/x", "derived": "derived/x", "route": "stream",
        "license": "other", "files": [{"name": "data/A_*.parquet", "spec": {}}],
    }
    with pytest.raises(GateError, match="not on the accepted list"):
        fetch.run_gate(block, tmp_path)


# ------------------------------------------------ checksums, per algorithm --


def test_verify_uses_the_algorithm_the_host_published(tmp_path):
    """`verify` hashes with the algorithm the checksum names."""
    import hashlib

    from sidechain.ingest import fetch

    payload = b"some parquet bytes"
    (tmp_path / "f.parquet").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    md5 = hashlib.md5(payload).hexdigest()

    good = (RemoteFile("f.parquet", len(payload), f"sha256:{sha}", "u"),)
    assert fetch.verify(good, tmp_path) == 0

    # the same file with the md5 digest labelled sha256 must NOT pass
    mislabelled = (RemoteFile("f.parquet", len(payload), f"sha256:{md5}", "u"),)
    assert fetch.verify(mislabelled, tmp_path) == 1


def test_verify_rejects_an_algorithm_it_cannot_compute(tmp_path):
    from sidechain.ingest import fetch

    (tmp_path / "f.bin").write_bytes(b"x")
    bad = (RemoteFile("f.bin", 1, "quantumhash:abc", "u"),)
    with pytest.raises(GateError, match="hashlib cannot compute"):
        fetch.verify(bad, tmp_path)


def test_a_stream_block_without_derived_is_refused_at_runtime(tmp_path, monkeypatch):
    """`run_gate` refuses a stream block with no `derived:`."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "huggingface", lambda _: _hf_record())
    block = {
        "name": "x", "host": "huggingface", "record": "Owner/Repo", "budget_gb": 12.0,
        "dest": "external/x", "route": "stream", "license": "CC-BY-NC-SA-4.0",
        "files": [{"name": "data/A_*.parquet", "spec": {}}],
    }
    with pytest.raises(GateError, match="no `derived:`"):
        fetch.run_gate(block, tmp_path)
