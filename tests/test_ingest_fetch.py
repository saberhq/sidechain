"""Tests for the dataset registry and the ingest entry point.

These are contract tests over `configs/datasets.yaml` as much as over the code:
a spec block that forgets to declare a control label is a bug that would only
surface much later, as a delta computed against an empty control pool.
"""
import json

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

# What an EXPRESSION file's spec must declare. A `gene_map` file is a lookup
# table, not perturbation data: asking it for a control label is meaningless,
# so it declares `role` alone. Kept as two explicit sets rather than one
# relaxed set -- the point of this test is that nothing is left to a default.
REQUIRED_SPEC_KEYS = {"context", "pert_col", "control_label", "modality", "role"}

# An `lfc_table` is a PRECOMPUTED contrast: the control arm was divided out
# upstream, so it has no counts, no control label and nothing to pseudobulk.
# Requiring `control_label` of it would force a block to name a control that
# does not exist in the file -- which is exactly the fiction this test exists
# to prevent, arriving through the test itself. It declares the columns it
# really has instead. `cell_metadata` is the obs sidecar for a counts file that
# ships its labels in a separate table; like `gene_map` it is a lookup, not
# perturbation data, so `role` alone.
REQUIRED_SPEC_KEYS_BY_KIND = {
    "expression": REQUIRED_SPEC_KEYS,
    "gene_map": {"role"},
    "lfc_table": {"target_col", "gene_id_col", "effect_col", "pvalue_col", "modality", "role"},
    "cell_metadata": {"role"},
}

# Kinds that describe perturbation data and therefore have to say which context
# they are. A lookup table does not.
CONTEXTUAL_KINDS = {"expression", "lfc_table"}


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
            kind = entry.get("kind", "expression")
            assert kind in REQUIRED_SPEC_KEYS_BY_KIND, (
                f"{name}/{entry['name']} declares unknown kind {kind!r}"
            )
            spec = entry.get("spec", {})
            missing = REQUIRED_SPEC_KEYS_BY_KIND[kind] - set(spec)
            assert not missing, f"{name}/{entry['name']} missing spec keys: {sorted(missing)}"


def test_every_contextual_file_names_its_context_exactly_once():
    """`context` (whole file) or `context_col` (varies per row) -- one, never both.

    Feng's targeted table is the case that forced this: 19 cell lines live in
    one file under `Cell_Line`, so a per-file `context:` would have to invent a
    name for a mixture. The opposite error is worse and quieter -- declaring
    both lets a reader take the per-file name and silently pool 19 lines into
    one context, which is precisely the mistake report 07 §3.1 caught the
    genome-wide arm being described with.
    """
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


def test_figshare_records_pin_a_version():
    """A bare Figshare article id resolves to *latest* and moves when the
    depositor republishes -- the same trap as a Zenodo concept DOI, wearing an
    article id's clothes. `.v<n>` is the pin."""
    for name, block in load_datasets().items():
        if block["host"] == "figshare":
            assert ".v" in str(block["record"]), (
                f"{name} names Figshare article {block['record']!r} without a .v<n> "
                "version; that follows 'latest' and would change under us"
            )


def test_streamed_blocks_declare_where_the_aggregate_goes():
    """`dest` and `derived` are different trees with opposite recovery
    properties (ADR 0003 §1): external/ is re-downloadable and safe to delete,
    derived/ is hours of network to rebuild. A streamed block that pointed both
    at one directory would make `rm -rf external/<x>` destroy the expensive
    half."""
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
    """A typo'd route must never fall back to `download`: on X-Atlas that is
    the difference between a stream and 126 GB on a laptop."""
    from sidechain.ingest.provenance import ROUTES

    for name, block in load_datasets().items():
        route = block.get("route", "download")
        assert route in ROUTES, f"{name} declares route={route!r}; known: {ROUTES}"


def test_the_real_xatlas_block_would_be_refused_as_a_download():
    """The registry's own numbers, not a fixture: X-Atlas declares 12 GB and is
    126 GB on the wire. That combination is only legal because it streams, and
    this asserts the gate actually depends on the route rather than the block
    merely claiming to."""
    block = load_datasets()["xatlas_orion"]
    assert block["route"] == "stream"
    assert float(block["budget_gb"]) < 126, "budget should be the OUTPUT ceiling, not the corpus"

    corpus = HostRecord(
        host="huggingface", record_id=block["record"], api_url="u", title="t",
        license="CC-BY-NC-SA-4.0", retrieved="2026-08-23", version="deadbeef",
        files=(RemoteFile("data/HCT116_Batch1.parquet", 126_260_000_000,
                          "sha256:aa", "https://x/a"),),
    )
    # as a download: refused on size
    with pytest.raises(GateError, match="over the"):
        gate(corpus, budget_gb=float(block["budget_gb"]), route="download")
    # as a stream: allowed, because those bytes never land
    got = gate(corpus, budget_gb=float(block["budget_gb"]), route="stream")
    assert len(got) == 1


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


# ------------------------------------- provenance is evidence, not a mirror --


def _block(tmp_path, license="CC-BY-4.0"):
    return {
        "name": "d", "host": "zenodo", "record": "1", "budget_gb": 1.0,
        "dest": "external/d", "license": license,
        "files": [{"name": "a.h5ad", "spec": {"context": "k562"}}],
    }


def test_provenance_is_written_once_and_not_rewritten(tmp_path, monkeypatch):
    """The original bug: every invocation rewrote PROVENANCE.json, turning
    evidence of what we fetched into a log of when we last ran the command."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "zenodo", lambda _: _record())
    block = _block(tmp_path)

    _, _, dest = fetch.run_gate(block, tmp_path)
    path = dest / "PROVENANCE.json"
    first = path.read_text()

    fetch.run_gate(block, tmp_path)  # second run, nothing changed upstream
    assert path.read_text() == first


def test_upstream_checksum_change_is_refused(tmp_path, monkeypatch):
    """Same filename, different bytes upstream. Silently absorbing this is how
    the provenance record stops meaning anything."""
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
    """It changes on every probe by definition; treating it as a change would
    make every run look like upstream moved."""
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
    """A per-dataset budget knows nothing about the other datasets sharing the
    volume; this check does."""
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
    """A streamed dataset still passes through the gate before the first range
    request. `dest` ends up holding PROVENANCE.json and nothing else -- which is
    the intended shape, so the file has to say `route: stream` or a later reader
    reads it as a download that died."""
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
    """Absent `route:` means download -- every block written before streaming
    existed stays true."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "zenodo", lambda _: _record())
    _, _, dest = fetch.run_gate(_block(tmp_path), tmp_path)
    payload = json.loads((dest / "PROVENANCE.json").read_text())
    assert payload["route"] == "download"
    assert payload["lands_on_disk"] is True


def test_a_typod_route_is_refused_not_downloaded(tmp_path, monkeypatch):
    """The expensive accident this guards: `route: streamed` silently read as
    `download` would try to put 126 GB of parquet on the Mac."""
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


# ------------------------------------------------ checksums, per algorithm --


def test_verify_uses_the_algorithm_the_host_published(tmp_path):
    """Zenodo says md5, HF says sha256. Hashing an HF file with md5 would
    report MISMATCH on bytes that are perfectly fine, and a verifier that cries
    wolf gets ignored."""
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
    """The contract test only sees the checked-in config; this is the next
    block someone adds."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "huggingface", lambda _: _hf_record())
    block = {
        "name": "x", "host": "huggingface", "record": "Owner/Repo", "budget_gb": 12.0,
        "dest": "external/x", "route": "stream", "license": "CC-BY-NC-SA-4.0",
        "files": [{"name": "data/A_*.parquet", "spec": {}}],
    }
    with pytest.raises(GateError, match="no `derived:`"):
        fetch.run_gate(block, tmp_path)
