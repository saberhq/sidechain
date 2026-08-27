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
    probe_huggingface,
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


def test_license_override_fills_an_absence():
    """A host with no license field (lamin) passes only via an override that
    names where the terms were actually verified."""
    got = gate(_record(license="unknown"), budget_gb=100,
               license_override=("CC-BY-4.0", "zenodo record 13350497 v1.4, verified 2026-08-16"))
    assert [f.name for f in got] == ["small.h5ad", "big.h5ad"]


def test_license_override_never_outvotes_a_stated_license():
    with pytest.raises(GateError, match="does not outvote"):
        gate(_record(license="CC-BY-NC-SA-4.0"), budget_gb=100,
             license_override=("CC-BY-4.0", "some source"))


def test_license_override_without_a_source_is_an_assertion_not_a_record():
    with pytest.raises(GateError, match="names no source"):
        gate(_record(license="unknown"), budget_gb=100,
             license_override=("CC-BY-4.0", "  "))


def test_license_override_still_faces_the_accepted_list():
    with pytest.raises(GateError, match="not on the"):
        gate(_record(license="unknown"), budget_gb=100,
             license_override=("Proprietary-EULA", "somewhere"))


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
    assert payload["schema_version"] == 2
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


# ------------------------------------------------------------ glob select --


def _globbable():
    """A record shaped like X-Atlas: many per-batch files under data/."""
    files = [
        RemoteFile(f"data/HCT116_Batch{i}.parquet", 400_000_000, f"sha256:h{i}", f"u{i}")
        for i in range(1, 4)
    ] + [
        RemoteFile("data/HEK293T_Batch1.parquet", 500_000_000, "sha256:k1", "uk1"),
        RemoteFile("metadata/gene_metadata.parquet", 881_915, "sha256:g", "ug"),
        RemoteFile("README.md", 4_000, None, "ur"),
    ]
    return HostRecord(
        host="huggingface", record_id="Xaira-Therapeutics/X-Atlas-Orion", api_url="u",
        title="t", license="CC-BY-NC-SA-4.0", retrieved="2026-08-23",
        version="53a5bc98", files=tuple(files),
    )


def test_glob_expands_to_every_matching_file():
    """332 per-batch files cannot be listed by hand without the list going
    stale the moment Xaira adds a batch."""
    got = _globbable().select(["data/HCT116_*.parquet"])
    assert [f.name for f in got] == [
        "data/HCT116_Batch1.parquet",
        "data/HCT116_Batch2.parquet",
        "data/HCT116_Batch3.parquet",
    ]


def test_glob_matching_nothing_is_a_stop_not_an_empty_selection():
    """The whole point of strict selection. A renamed upstream folder must
    stop the run, not stream zero cells and report success."""
    with pytest.raises(GateError, match="no file"):
        _globbable().select(["data/HELA_*.parquet"])


def test_one_dead_glob_among_live_ones_still_stops():
    """A partial corpus that looks whole is the failure being prevented, so a
    selection is all-or-nothing."""
    with pytest.raises(GateError, match="HELA"):
        _globbable().select(["data/HCT116_*.parquet", "data/HELA_*.parquet"])


def test_globs_and_exact_names_mix_and_keep_request_order():
    got = _globbable().select(["data/HEK293T_*.parquet", "metadata/gene_metadata.parquet"])
    assert [f.name for f in got] == [
        "data/HEK293T_Batch1.parquet",
        "metadata/gene_metadata.parquet",
    ]


def test_overlapping_globs_do_not_double_count():
    """Two patterns matching the same file must select it once: the gate sums
    sizes, and a double-counted 46 GB line would misreport the corpus."""
    got = _globbable().select(["data/HCT116_*.parquet", "data/*.parquet"])
    assert len(got) == len({f.name for f in got})
    assert len(got) == 4


def test_exact_name_behaviour_is_unchanged_by_glob_support():
    """A name with no metacharacters must still be an exact match, so every
    existing block keeps its meaning."""
    rec = _globbable()
    assert [f.name for f in rec.select(["README.md"])] == ["README.md"]
    with pytest.raises(GateError, match="no file"):
        rec.select(["READ"])       # prefix, not a glob -> still absent


# ------------------------------------------------------- the HF probe --


class _FakeHF:
    """The two HF endpoints, with the shape the real API returned 2026-08-23."""

    def __init__(self, *, sha="53a5bc98", card=None, tags=(), sizes=None, omit=()):
        self.sha, self.card, self.tags = sha, card, list(tags)
        self.sizes = sizes or {"data/A.parquet": 400_000_000, "README.md": 4_000}
        self.omit = set(omit)

    def get(self, url):
        return {
            "id": "Owner/Repo", "sha": self.sha,
            "cardData": self.card if self.card is not None else {},
            "tags": self.tags,
            "siblings": [{"rfilename": n} for n in self.sizes],
        }

    def post(self, url, payload):
        out = []
        for path in payload["paths"]:
            if path in self.omit:
                continue
            entry = {"type": "file", "path": path, "size": self.sizes[path]}
            if path.endswith(".parquet"):        # LFS-backed -> sha256 published
                entry["lfs"] = {"oid": "a" * 64, "size": self.sizes[path]}
            out.append(entry)
        return out


def _install(monkeypatch, fake):
    from sidechain.ingest import provenance as prov

    monkeypatch.setattr(prov, "_get_json", fake.get)
    monkeypatch.setattr(prov, "_post_json", fake.post)


def test_probe_pins_the_commit_sha_as_the_version(monkeypatch):
    """HF has no immutable record id -- `main` moves when the depositor pushes.
    The sha is the only thing that makes the record re-checkable, and is what
    makes diff_against_recorded fire on a republish."""
    _install(monkeypatch, _FakeHF(card={"license": "cc-by-nc-sa-4.0"}))
    rec = probe_huggingface("Owner/Repo")
    assert rec.version == "53a5bc98"
    assert rec.host == "huggingface"
    assert rec.record_id == "Owner/Repo"


def test_probe_pins_file_urls_to_that_sha(monkeypatch):
    """So a stream reads exactly the bytes that were probed, not whatever
    `main` points at by the time the run starts."""
    _install(monkeypatch, _FakeHF(card={"license": "cc-by-nc-sa-4.0"}))
    rec = probe_huggingface("Owner/Repo")
    assert all("/resolve/53a5bc98/" in f.url for f in rec.files)


def test_probe_records_sha256_from_the_lfs_pointer(monkeypatch):
    """LFS stores an object under its sha256, so that is the checksum HF can
    actually publish. Recorded with its algorithm so verify hashes correctly."""
    _install(monkeypatch, _FakeHF(card={"license": "cc-by-nc-sa-4.0"}))
    rec = probe_huggingface("Owner/Repo")
    by = {f.name: f for f in rec.files}
    assert by["data/A.parquet"].checksum == "sha256:" + "a" * 64
    assert by["README.md"].checksum is None      # not LFS -> nothing published


def test_probe_reads_sizes_and_does_not_leave_them_zero(monkeypatch):
    """The repo endpoint carries no sizes at all. A probe that skipped
    paths-info would build a record of zero-byte files, and the budget check
    would then wave through anything at all -- wearing the gate's authority."""
    _install(monkeypatch, _FakeHF(card={"license": "cc-by-nc-sa-4.0"}))
    rec = probe_huggingface("Owner/Repo")
    assert rec.total_bytes == 400_004_000
    assert all(f.size_bytes > 0 for f in rec.files)


def test_probe_refuses_when_a_file_has_no_resolvable_size(monkeypatch):
    _install(monkeypatch, _FakeHF(card={"license": "cc-by-nc-sa-4.0"},
                                  omit=["data/A.parquet"]))
    with pytest.raises(GateError, match="paths-info did not resolve"):
        probe_huggingface("Owner/Repo")


def test_probe_falls_back_to_the_license_tag(monkeypatch):
    """A `license:` tag is a declaration too. The fallback exists so the
    `unknown` stop fires only for a repo that declares NOTHING."""
    _install(monkeypatch, _FakeHF(card={}, tags=["license:cc-by-nc-sa-4.0"]))
    assert probe_huggingface("Owner/Repo").license == "CC-BY-NC-SA-4.0"


def test_probe_reports_unknown_when_nothing_is_declared(monkeypatch):
    """Arc's State-Replogle-Filtered is the live example, and `gate` turns this
    into a stop."""
    _install(monkeypatch, _FakeHF(card={}, tags=[]))
    rec = probe_huggingface("Owner/Repo")
    assert rec.license == "unknown"
    with pytest.raises(GateError, match="no license"):
        gate(rec, budget_gb=100)


def test_probe_handles_a_list_of_license_tags(monkeypatch):
    _install(monkeypatch, _FakeHF(card={"license": ["cc-by-nc-sa-4.0"]}))
    assert probe_huggingface("Owner/Repo").license == "CC-BY-NC-SA-4.0"


def test_probe_refuses_a_repo_with_no_sha(monkeypatch):
    _install(monkeypatch, _FakeHF(sha="", card={"license": "mit"}))
    with pytest.raises(GateError, match="no commit sha"):
        probe_huggingface("Owner/Repo")


# ------------------------------------------------------------ route rules --


def test_unknown_route_is_refused_rather_than_defaulted():
    with pytest.raises(GateError, match="unknown route"):
        gate(_record(), budget_gb=100, route="streamed")


def test_stream_route_does_not_measure_the_budget_against_the_corpus():
    """126 GB read, 12 GB written. Bounding the selection would refuse every
    stream; bounding nothing would let the aggregate fill the volume."""
    got = gate(_record(), budget_gb=0.5, route="stream")
    assert len(got) == 2


def test_stream_route_still_checks_the_disk_against_the_output_budget(tmp_path):
    with pytest.raises(GateError, match="not enough disk"):
        gate(_record(), budget_gb=10**9, route="stream", dest=tmp_path / "nowhere")


def test_stream_route_still_enforces_licence_and_checksums():
    """Streaming relaxes the SIZE reading of the budget and nothing else."""
    with pytest.raises(GateError, match="no license"):
        gate(_record(license="unknown"), budget_gb=1, route="stream")
    nosum = [RemoteFile("a.parquet", 10, None, "u")]
    with pytest.raises(GateError, match="no checksum"):
        gate(_record(files=nosum), budget_gb=1, route="stream")


# ------------------------------------- licence: stated is not the same as accepted --


def test_stated_but_unrecognised_licence_is_a_stop():
    """The gap the HF host exposes. HF's namespace includes `other`, `unknown`
    and bespoke slugs; before this, anything non-empty that was not literally
    "unknown" passed -- AND was recorded with both license_flags false, i.e. as
    less encumbered than it is. ADR 0003 says the gate stops on terms we have
    not accepted, and this is that stop."""
    with pytest.raises(GateError, match="not on the accepted list"):
        gate(_record(license="other"), budget_gb=100)


def test_an_unrecognised_noncommercial_licence_cannot_slip_through_as_unencumbered():
    """The concrete harm: a CC-BY-NC-ND variant we do not model would be
    recorded noncommercial=False, redistribution_encumbered=False."""
    with pytest.raises(GateError, match="not on the accepted list"):
        gate(_record(license="CC-BY-NC-ND-4.0"), budget_gb=100)


def test_every_accepted_licence_still_passes():
    """The stop must not become a wall: everything on the policy list works."""
    from sidechain.ingest.provenance import ACCEPTED_LICENSES

    for lic in ACCEPTED_LICENSES:
        assert len(gate(_record(license=lic), budget_gb=100)) == 2


def test_the_licence_stop_applies_on_a_stream_too():
    with pytest.raises(GateError, match="not on the accepted list"):
        gate(_record(license="other"), budget_gb=1, route="stream")


# ------------------------------ free space is measured where the bytes land --


def test_stream_checks_free_space_where_the_aggregate_is_written(tmp_path):
    """dest and space_dest are different trees on a stream: PROVENANCE.json
    goes to external/, the 12 GB aggregate to derived/. Measuring external/'s
    volume assumes they share one, and nothing was checking that."""
    seen = {}
    import sidechain.ingest.provenance as prov

    real = prov.shutil.disk_usage

    class _Spy:
        @staticmethod
        def disk_usage(path):
            seen["path"] = path
            return real("/")

    prov.shutil, saved = _Spy, prov.shutil
    try:
        external = tmp_path / "external" / "hf-x"
        derived = tmp_path / "derived" / "x"
        derived.mkdir(parents=True)
        gate(_record(), budget_gb=0.001, route="stream", dest=external,
             space_dest=derived, headroom_gb=0)
    finally:
        prov.shutil = saved
    assert seen["path"] == derived, "free space was measured on the wrong tree"


def test_download_route_still_measures_dest(tmp_path):
    seen = {}
    import sidechain.ingest.provenance as prov

    real = prov.shutil.disk_usage

    class _Spy:
        @staticmethod
        def disk_usage(path):
            seen["path"] = path
            return real("/")

    prov.shutil, saved = _Spy, prov.shutil
    try:
        dest = tmp_path / "external" / "zenodo-1"
        dest.mkdir(parents=True)
        gate(_record(), budget_gb=100, dest=dest, headroom_gb=0)
    finally:
        prov.shutil = saved
    assert seen["path"] == dest


# ---------------------------------------------- a route flip is a real change --


def test_a_route_flip_is_reported_as_a_difference():
    """Flipping route changes what budget_gb bounds AND whether the bytes are
    ever verified. Absorbing that silently would let a config edit switch off
    the checksum pass on an already-recorded dataset."""
    recorded = {"route": "stream", "record": {"license": "CC-BY-4.0", "version": "1.4"},
                "selected": [{"name": "a", "size_bytes": 1, "checksum": "md5:x"}]}
    rec = HostRecord(host="zenodo", record_id="1", api_url="u", title="t",
                     license="CC-BY-4.0", retrieved="2026-08-23", version="1.4",
                     files=(RemoteFile("a", 1, "md5:x", "u"),))
    from sidechain.ingest.provenance import diff_against_recorded

    diffs = diff_against_recorded(recorded, rec, rec.files, route="download")
    assert any("route" in d for d in diffs)
    assert not diff_against_recorded(recorded, rec, rec.files, route="stream")


def test_a_schema_1_record_does_not_read_as_a_route_change():
    """Records written before routes existed carry no `route` key. Treating
    that absence as a difference would make every scPerturb run raise."""
    recorded = {"record": {"license": "CC-BY-4.0", "version": "1.4"},
                "selected": [{"name": "a", "size_bytes": 1, "checksum": "md5:x"}]}
    rec = HostRecord(host="zenodo", record_id="1", api_url="u", title="t",
                     license="CC-BY-4.0", retrieved="2026-08-23", version="1.4",
                     files=(RemoteFile("a", 1, "md5:x", "u"),))
    from sidechain.ingest.provenance import diff_against_recorded

    assert not diff_against_recorded(recorded, rec, rec.files, route="download")


# ------------------------------------------------- the Figshare probe --


class _FakeFigshare:
    """One Figshare article, with the shape the real API returned 2026-08-23.

    Figshare differs from Zenodo in the two ways that matter to the gate: it
    states a license by human NAME rather than SPDX id, and it publishes both
    a `computed_md5` (its own verification) and a `supplied_md5` (the
    uploader's claim), either of which can be the empty string.
    """

    def __init__(self, *, license_name="MIT", files=None, version=1):
        self.license_name, self.version = license_name, version
        self.files = files if files is not None else [
            {"name": "GenomeWideScreen_LFC_byGene.tsv.gz", "size": 794_610_300,
             "computed_md5": "03cb8a3b4328de50d45746a9d784bbc5",
             "supplied_md5": "03cb8a3b4328de50d45746a9d784bbc5",
             "download_url": "https://ndownloader.figshare.com/files/49291654"},
        ]
        self.urls: list[str] = []

    def get(self, url):
        self.urls.append(url)
        return {
            "id": 26819743,
            "title": "A genome-scale single cell CRISPRi map",
            "version": self.version,
            "doi": f"10.6084/m9.figshare.26819743.v{self.version}",
            "license": {"value": 3, "name": self.license_name,
                        "url": "https://opensource.org/licenses/MIT"},
            "files": self.files,
        }


def _install_get(monkeypatch, fake):
    from sidechain.ingest import provenance as prov

    monkeypatch.setattr(prov, "_get_json", fake.get)


def test_figshare_version_suffix_reads_the_pinned_endpoint(monkeypatch):
    """`.v1` is the pin. A bare article id resolves to *latest* and moves when
    the depositor republishes -- the same trap as a Zenodo concept DOI."""
    from sidechain.ingest.provenance import probe_figshare

    fake = _FakeFigshare()
    _install_get(monkeypatch, fake)

    rec = probe_figshare("26819743.v1")
    assert fake.urls == ["https://api.figshare.com/v2/articles/26819743/versions/1"]
    assert rec.host == "figshare"
    assert rec.record_id == "26819743.v1"
    assert rec.version == "1"

    fake.urls.clear()
    probe_figshare("26819743")
    assert fake.urls == ["https://api.figshare.com/v2/articles/26819743"]


def test_figshare_records_the_versionless_doi_as_the_concept_doi(monkeypatch):
    """Figshare publishes no concept-DOI field, but the versionless form of the
    article DOI is that identifier -- the one that follows 'latest'. Recording
    both makes it visible which is pinned and which drifts."""
    from sidechain.ingest.provenance import probe_figshare

    _install_get(monkeypatch, _FakeFigshare())
    rec = probe_figshare("26819743.v1")
    assert rec.doi == "10.6084/m9.figshare.26819743.v1"
    assert rec.concept_doi == "10.6084/m9.figshare.26819743"


def test_figshare_licence_name_is_normalised_to_the_spdx_the_gate_compares(monkeypatch):
    """Figshare says "CC BY 4.0", the gate's accepted list says "CC-BY-4.0".
    Passing the name through raw would refuse a perfectly usable dataset --
    a false refusal, which erodes the gate as surely as a false pass."""
    from sidechain.ingest.provenance import ACCEPTED_LICENSES, probe_figshare

    for stated, expected in [("MIT", "MIT"), ("CC BY 4.0", "CC-BY-4.0"),
                             ("CC BY-NC-SA 4.0", "CC-BY-NC-SA-4.0"), ("CC0", "CC0-1.0")]:
        _install_get(monkeypatch, _FakeFigshare(license_name=stated))
        got = probe_figshare("26819743.v1").license
        assert got == expected, f"{stated!r} -> {got!r}"
        assert got in ACCEPTED_LICENSES

    # and a licence Figshare states that we have NOT accepted still reaches the
    # gate as itself, so the gate refuses it rather than silently accepting.
    _install_get(monkeypatch, _FakeFigshare(license_name="CC BY-ND 4.0"))
    rec = probe_figshare("26819743.v1")
    assert rec.license == "CC BY-ND 4.0"
    assert rec.license not in ACCEPTED_LICENSES
    with pytest.raises(GateError, match="not on the accepted list"):
        gate(rec, budget_gb=100.0)


def test_figshare_missing_licence_is_unknown_and_therefore_a_stop(monkeypatch):
    from sidechain.ingest.provenance import probe_figshare

    _install_get(monkeypatch, _FakeFigshare(license_name=""))
    rec = probe_figshare("26819743.v1")
    assert rec.license == "unknown"
    with pytest.raises(GateError, match="declares no license"):
        gate(rec, budget_gb=100.0)


def test_empty_computed_md5_is_an_absent_checksum_not_an_empty_one(monkeypatch):
    """The live X-Atlas/Orion defect, generalised: Figshare returns "" rather
    than omitting the field when it never verified the uploader's hash.

    Carried through as a checksum, `verify` would compare a real digest against
    "" and report MISMATCH on good bytes forever -- a verifier that cries wolf
    gets ignored. As None it routes to the gate's refusal, where the exception
    has to be asked for and is recorded in PROVENANCE.json.
    """
    from sidechain.ingest.provenance import probe_figshare

    _install_get(monkeypatch, _FakeFigshare(files=[
        {"name": "unverified.h5ad", "size": 100, "computed_md5": "",
         "supplied_md5": "deadbeef", "download_url": "https://x/u"},
    ]))
    rec = probe_figshare("26819743.v1")
    assert rec.files[0].checksum is None

    with pytest.raises(GateError, match="no checksum published"):
        gate(rec, budget_gb=100.0)
    assert gate(rec, budget_gb=100.0, allow_missing_checksum=True)[0].name == "unverified.h5ad"


def test_supplied_md5_is_never_substituted_for_a_missing_computed_md5(monkeypatch):
    """`supplied_md5` is the uploader's claim about their own bytes, which is
    the very thing `computed_md5` exists to corroborate. Falling back to it
    would erase the distinction ADR 0003 was written around and turn an
    unverified file into a verified-looking one."""
    from sidechain.ingest.provenance import probe_figshare

    _install_get(monkeypatch, _FakeFigshare(files=[
        {"name": "a.gz", "size": 10, "computed_md5": "", "supplied_md5": "c" * 32,
         "download_url": "https://x/a"},
    ]))
    assert probe_figshare("26819743.v1").files[0].checksum is None


def test_figshare_files_carry_size_checksum_and_url(monkeypatch):
    from sidechain.ingest.provenance import probe_figshare

    _install_get(monkeypatch, _FakeFigshare())
    f = probe_figshare("26819743.v1").files[0]
    assert f.name == "GenomeWideScreen_LFC_byGene.tsv.gz"
    assert f.size_bytes == 794_610_300
    assert f.checksum == "md5:03cb8a3b4328de50d45746a9d784bbc5"
    assert f.url == "https://ndownloader.figshare.com/files/49291654"
