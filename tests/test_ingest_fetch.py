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
# An `observational` file is per-cell counts with NO perturbation label
# anywhere. It is the mirror image of `lfc_table`: that one has the contrast
# already taken, this one has no contrast to take. It declares the facts about
# the MEASUREMENT (`assay`) where expression declares the facts about the
# perturbation (`modality`), and it declares a `content_filter`, which no other
# kind needs -- labelled data catches a bad row downstream (a control arm that
# matches nothing, a delta that comes out zero) and unlabelled data has no such
# check, so the QC rule has to be at the front door or it is nowhere.
# `sample_metadata` is `cell_metadata` one grain coarser: one row per
# EXPERIMENT. Same required keys, kept apart because the name is what stops a
# reader joining a 35,259-row manifest on a cell barcode.
REQUIRED_SPEC_KEYS_BY_KIND = {
    "expression": REQUIRED_SPEC_KEYS,
    "gene_map": {"role"},
    "lfc_table": {"target_col", "gene_id_col", "effect_col", "pvalue_col", "modality", "role"},
    "cell_metadata": {"role"},
    "sample_metadata": {"role"},
    "observational": {"context", "context_col", "sample_from", "sample_id_col",
                      "assay", "content_filter", "role"},
}

# Keys a kind may not declare AT ALL -- absent, not null. `null` is this
# registry's idiom for "the other branch of a mutually exclusive pair"
# (context/context_col, gene_id_col), and reusing it for "this question is
# meaningless" would let `pert_col: null` read as a column we have not named yet
# rather than a link that does not exist in the bytes. scBaseCount is the case:
# thousands of its experiments ARE Perturb-seq, but SRA submits the guide
# library as a separate feature-barcoding run, so perturbation status is known
# per experiment and never per cell.
FORBIDDEN_SPEC_KEYS_BY_KIND = {
    "observational": {"pert_col", "control_label"},
}

# Kinds that describe perturbation data and therefore have to say which context
# they are. A lookup table does not. `observational` is here because a
# co-expression prior pooled across cell types measures COMPOSITION rather than
# regulation -- the within-line restriction is the whole claim, so the context
# declaration is not optional.
CONTEXTUAL_KINDS = {"expression", "lfc_table", "observational"}

# What a file is FOR. Enumerated so a typo cannot invent a role that silently
# matches no consumer: `role: trian` would just never be trained on.
# `prior` means a corpus we build priors FROM -- it feeds data_sources.yaml and
# never reaches `pooled_delta`, which is the distinction that keeps an
# unlabelled corpus out of the delta path by declaration rather than by luck.
KNOWN_ROLES = {"train", "analysis", "reference", "excluded", "prior"}


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


def test_no_file_declares_a_spec_key_its_kind_cannot_answer():
    """The complement of the test above, and the one that keeps `observational`
    honest. Requiring keys stops a block leaving a question unanswered; this
    stops a block answering a question that does not exist. A `pert_col` on a
    corpus with no per-cell guide link is not an unused field -- it is an
    invitation for a reader to compute a delta out of noise."""
    for name, block in load_datasets().items():
        for entry in block["files"]:
            kind = entry.get("kind", "expression")
            forbidden = FORBIDDEN_SPEC_KEYS_BY_KIND.get(kind, set())
            present = forbidden & set(entry.get("spec", {}))
            assert not present, (
                f"{name}/{entry['name']} is kind {kind!r} and declares {sorted(present)}. "
                "Remove the keys rather than setting them null -- null means 'the other "
                "branch', not 'meaningless'."
            )


def test_every_file_declares_a_known_role():
    for name, block in load_datasets().items():
        for entry in block["files"]:
            role = entry.get("spec", {}).get("role")
            assert role in KNOWN_ROLES, (
                f"{name}/{entry['name']} declares role {role!r}; known: {sorted(KNOWN_ROLES)}"
            )


def test_every_observational_file_declares_a_real_content_filter():
    """An empty filter is not a filter, and on an unlabelled corpus nothing
    downstream would notice. scBaseCount is why the rule exists: 20.8 % of its
    human cells -- 32-55 % within the four lines we actually want -- are
    guide-capture sub-libraries quantified against the gene reference, tens of
    thousands of "cells" at a median of 1 UMI. Streamed unfiltered, the prior is
    mostly zeros and reports success."""
    for name, block in load_datasets().items():
        for entry in block["files"]:
            if entry.get("kind") != "observational":
                continue
            cf = entry.get("spec", {}).get("content_filter")
            assert isinstance(cf, dict) and cf, (
                f"{name}/{entry['name']} declares content_filter={cf!r}. It must be a "
                "non-empty mapping: an unlabelled corpus has no control arm to fail "
                "against, so this is the only place bad rows get stopped."
            )


def test_a_sidecar_join_names_the_key_on_both_sides():
    """Feng's three spellings of one barcode (`MP-ST-...`, `ST-...`, `MP.ST...`)
    cost the registry a documented warning about silent misjoins. scBaseCount
    has the same shape one grain up: the manifest says `srx_accession`, the
    matrix obs says `SRX_accession`. Declaring only one side means the reader
    picks the other by guessing."""
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
    """`obs_from` / `sample_from` name the ONLY place a corpus's labels live --
    the cell line, the guide call, the batch. A sidecar outside the block's
    `files` would be fetched off the record, hashed by nobody and absent from
    PROVENANCE.json, so the labels would have no provenance while the counts
    did."""
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


def test_the_real_scbasecount_block_is_observational_and_cannot_be_downloaded():
    """The registry's own block, not a fixture. Three properties travel
    together and each is load-bearing: the corpus has no per-cell perturbation
    label so its counts file is `observational`; its terms are stated nowhere so
    the licence is the UNSTATED sentinel; and UNSTATED is only legal on a
    stream, because reading what any public reader may read is not the same as
    holding a copy of it."""
    from sidechain.ingest.provenance import UNSTATED

    block = load_datasets()["scbasecount_human_gene"]
    assert block["route"] == "stream"
    assert block["license"] == UNSTATED
    assert block["license_override_source"].strip(), "UNSTATED without a source is an assertion"

    counts = next(f for f in block["files"] if f.get("kind") == "observational")
    assert "pert_col" not in counts["spec"] and "control_label" not in counts["spec"]
    assert counts["spec"]["context_col"] == "cell_line", "the within-line restriction IS the claim"
    assert float(block["budget_gb"]) < 2378, "budget is the OUTPUT ceiling, not the 2.4 TB read"

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


# ------------------------------------------------- the unstated licence --


def _lamin_record(files=None):
    """A host that CANNOT state terms. No field on lamin's Artifact model
    carries a licence, so `unknown` here is a measured property of the host, not
    a gap in the probe."""
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


def test_unstated_terms_pass_the_gate_only_as_a_stream(tmp_path, monkeypatch):
    """Naming the absence is allowed. The corpus is read over the network
    exactly as any public reader may read it, and only our own aggregate
    lands."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "lamin", lambda _: _lamin_record())
    _, _, dest = fetch.run_gate(_unstated_block(), tmp_path)
    payload = json.loads((dest / "PROVENANCE.json").read_text())
    assert payload["route"] == "stream"
    assert payload["notes"]["license_override"]["applied"] == "UNSTATED"
    assert payload["notes"]["license_override"]["host_stated"] == "unknown"


def test_unstated_terms_are_recorded_as_maximally_encumbered(tmp_path, monkeypatch):
    """The flags are the whole reason UNSTATED is safe to allow. "We do not
    know" has to resolve to "assume the strictest thing", or it is not
    conservative at all -- and these two booleans are what force a fresh
    decision before anything derived from this corpus is ever published."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "lamin", lambda _: _lamin_record())
    _, _, dest = fetch.run_gate(_unstated_block(), tmp_path)
    flags = json.loads((dest / "PROVENANCE.json").read_text())["license_flags"]
    assert flags["redistribution_encumbered"] is True
    assert flags["noncommercial"] is True


def test_unstated_terms_are_refused_on_a_download(tmp_path, monkeypatch):
    """The distinction the whole sentinel rests on: we may READ a corpus whose
    terms are stated nowhere; we may not HOLD a copy of one."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "lamin", lambda _: _lamin_record())
    block = _unstated_block(route="download", budget_gb=100.0)
    block.pop("derived")
    with pytest.raises(GateError, match="only tolerable on"):
        fetch.run_gate(block, tmp_path)


def test_unstated_without_an_override_source_is_refused(tmp_path, monkeypatch):
    """Without a source the sentinel is unreachable: it is a record of a search,
    and with no record of the search it is just a word that skips the licence
    check."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "lamin", lambda _: _lamin_record())
    block = _unstated_block()
    block.pop("license_override_source")
    with pytest.raises(GateError, match="record of a search"):
        fetch.run_gate(block, tmp_path)


def test_the_gate_itself_refuses_unstated_arriving_without_an_override(tmp_path):
    """The backstop under the check above, for a caller that reaches `gate()`
    directly. UNSTATED is not a licence a host can state, so it may only ever
    arrive as an override -- otherwise a probe returning that literal string
    would skip the accepted-licence check entirely."""
    from sidechain.ingest.provenance import UNSTATED

    record = HostRecord(
        host="lamin", record_id="owner/inst", api_url="u", title="t",
        license=UNSTATED, retrieved="2026-08-29",
        files=(RemoteFile("k/a.h5ad", 10, "lamin:aa", ""),))
    with pytest.raises(GateError, match="without a license_override"):
        gate(record, budget_gb=1.0, route="stream", dest=tmp_path)


def test_unstated_cannot_paper_over_a_host_that_did_state_terms(tmp_path, monkeypatch):
    """The pre-existing rule, asserted for the new value: an override fills an
    absence, it does not outvote a host. Declaring UNSTATED against a host that
    published a licence would be laundering a known licence into an unknown
    one -- the opposite direction from the CC0 trap, and just as wrong."""
    from sidechain.ingest import fetch

    monkeypatch.setitem(fetch.PROBES, "lamin", lambda _: HostRecord(
        host="lamin", record_id="owner/inst", api_url="u", title="t",
        license="CC-BY-NC-SA-4.0", retrieved="2026-08-29",
        files=(RemoteFile("k/a.h5ad", 10, "lamin:aa", ""),)))
    with pytest.raises(GateError, match="does not outvote the host"):
        fetch.run_gate(_unstated_block(), tmp_path)


def test_a_stated_but_unrecognised_licence_is_still_refused(tmp_path, monkeypatch):
    """The sentinel must not have widened the gate. `other` is a real Hugging
    Face licence slug and it still stops the run: UNSTATED is for terms that do
    not exist, not for terms we have not read yet."""
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
