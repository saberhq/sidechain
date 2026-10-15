"""Contract tests for `PolyASiteSource` (T102, T120) on a synthetic GENCODE GTF, TargetScan
Gene_info and PolyASite atlas, the atlas served over a fake network so the stream path is
the real one.

The build is pinned on the properties that decide the numbers:

* the representative transcript follows the ordered rules -- TargetScan's (by stem, and only
  when the GENE stem matches), then MANE_Select, then Ensembl_canonical, then the longest CDS --
  and a v19-to-v47 version bump is flagged, not silently absorbed;
* the window is strand-aware: on the minus strand the last exon is the LOWEST one, "downstream"
  is lower coordinates, and a cluster on the other strand never joins;
* the window starts after the stop codon when the stop lies in the last exon, and the 3'UTR
  length to a site counts the exonic bases between a stop in an earlier exon and the acceptor;
* the support filter needs both protocols AND a share of the window's TPM, and the unfiltered
  counts are written beside the filtered ones;
* a single-exon gene has no acceptor distance, a non-coding gene no 3'UTR length, a gene with
  no cluster zero sites (a measurement, not NaN);
* per-line usage sums only the atlas's UNPERTURBED libraries of a named line;
* the atlas never lands: `dest` holds PROVENANCE.json alone and the LINEAGE entry carries the
  sha256 of the bytes read; the GENCODE md5 is checked after landing;
* `build()` is a node_feature aligned to the gene space with a `has_value` mask.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from sidechain.ingest.provenance import RemoteFile
from sidechain.priors.posttx_polya import (
    GENCODE_MD5,
    GENE_COLUMNS,
    PolyASiteSource,
    atlas_chrom,
    classify_library,
    parse_gtf_attributes,
)

# ------------------------------------------------------------------- the GTF --


def _attrs(gene: str, gv: int, tx: str | None = None, tv: int = 1, gtype="protein_coding",
           ttype="protein_coding", name="G", tags=()) -> str:
    out = f'gene_id "{gene}.{gv}"; '
    if tx:
        out += f'transcript_id "{tx}.{tv}"; '
    out += f'gene_type "{gtype}"; gene_name "{name}"; '
    if tx:
        out += f'transcript_type "{ttype}"; '
    out += "level 2; " + " ".join(f'tag "{t}";' for t in ("basic", *tags))
    return out


def _rows(chrom, strand, gene, gv, tx, tv, exons, cds=(), stop=None, **kw) -> list[str]:
    a = _attrs(gene, gv, tx, tv, **kw)
    out = [f"{chrom}\tHAVANA\ttranscript\t{min(s for s, _ in exons)}\t{max(e for _, e in exons)}\t.\t{strand}\t.\t{a}"]
    for s, e in exons:
        out.append(f"{chrom}\tHAVANA\texon\t{s}\t{e}\t.\t{strand}\t.\t{a}")
    for s, e in cds:
        out.append(f"{chrom}\tHAVANA\tCDS\t{s}\t{e}\t.\t{strand}\t0\t{a}")
    if stop:
        out.append(f"{chrom}\tHAVANA\tstop_codon\t{stop[0]}\t{stop[1]}\t.\t{strand}\t0\t{a}")
    return out


GTF_LINES = [
    "##description: synthetic", "##provider: test",
    # G1, chr1 +: two transcripts sharing the acceptor at 500; the stop codon in the last exon
    f"chr1\tHAVANA\tgene\t100\t900\t.\t+\t.\t{_attrs('ENSG1', 7, name='G1')}",
    *_rows("chr1", "+", "ENSG1", 7, "ENST1", 5, [(100, 200), (300, 400), (500, 700)],
           cds=[(150, 200), (300, 400), (500, 550)], stop=(548, 550), name="G1", tags=("MANE_Select",)),
    *_rows("chr1", "+", "ENSG1", 7, "ENST2", 1, [(100, 200), (500, 900)],
           cds=[(150, 200), (500, 550)], stop=(548, 550), name="G1", tags=("Ensembl_canonical",)),
    # G2, chr2 -: the stop codon in the MIDDLE exon, 50 exonic bases before the last exon
    f"chr2\tHAVANA\tgene\t1000\t3300\t.\t-\t.\t{_attrs('ENSG2', 3, name='G2')}",
    *_rows("chr2", "-", "ENSG2", 3, "ENST3", 2, [(1000, 1200), (2000, 2100), (3000, 3300)],
           cds=[(2050, 2100), (3000, 3300)], stop=(2050, 2052), name="G2", tags=("MANE_Select",)),
    # G3, chr1 +: a single-exon lncRNA
    f"chr1\tHAVANA\tgene\t5000\t5500\t.\t+\t.\t{_attrs('ENSG3', 1, gtype='lncRNA', name='G3')}",
    *_rows("chr1", "+", "ENSG3", 1, "ENST4", 1, [(5000, 5500)], gtype="lncRNA", ttype="lncRNA",
           name="G3", tags=("Ensembl_canonical",)),
    # G4, chr3 +: no cluster anywhere near it
    f"chr3\tHAVANA\tgene\t100\t2000\t.\t+\t.\t{_attrs('ENSG4', 1, name='G4')}",
    *_rows("chr3", "+", "ENSG4", 1, "ENST5", 1, [(100, 300), (1000, 2000)], cds=[(150, 300), (1000, 1100)],
           stop=(1098, 1100), name="G4", tags=("MANE_Select",)),
    # G5, chr3 +: no tags, not in TargetScan -- the longest CDS wins
    f"chr3\tHAVANA\tgene\t10000\t12000\t.\t+\t.\t{_attrs('ENSG5', 1, name='G5')}",
    *_rows("chr3", "+", "ENSG5", 1, "ENST6", 1, [(10000, 10100), (11000, 12000)], cds=[(10050, 10100), (11000, 11050)],
           stop=(11048, 11050), name="G5"),
    *_rows("chr3", "+", "ENSG5", 1, "ENST7", 1, [(10000, 10100), (11000, 12000)], cds=[(10020, 10100), (11000, 11200)],
           stop=(11198, 11200), name="G5"),
]

GENE_INFO = (
    "Transcript ID\tGene ID\tGene symbol\tGene description\tSpecies ID\t3P-seq tags\tRepresentative transcript?\n"
    "ENST1.3\tENSG1.7\tG1\tdesc\t9606\t74\t1\n"          # representative; v19 version 3, v47 has 5
    "ENST2.1\tENSG1.7\tG1\tdesc\t9606\t2\t0\n"           # not representative
    "ENST3.2\tENSG99.1\tG2\tdesc\t9606\t5\t1\n"          # gene stem does NOT match v47 -> refused
    "ENST9.1\tENSG9.1\tGONE\tdesc\t9606\t5\t1\n"         # not in v47 at all
    "ENSMUST1.1\tENSMUSG1.1\tg1\tdesc\t10090\t1\t1\n"    # mouse
)

# ------------------------------------------------------------------ the atlas --

LIBS = ["GSM1|A-seq|HeLa cells|HeLa rep 1|none",
        "GSM2|A-seq|HeLa cells|HeLa si|si-CstF64/64tau",        # a knockdown: counts for no line
        "GSM3|3P-Seq|HEK293|HEK ctrl|siRNA Ctrl",
        "GSM4|DRS|K562|K562 kd|siRNA aCP",                      # a knockdown
        "GSM5|DRS|K562|K562 ctrl|siRNA control",
        "GSM6|PAS-Seq|testis|t|none"]
HEADER = "\t".join(["chrom", "chromStart", "chromEnd", "name", "score", "strand", "rep", "frac_samples",
                    "nr_prots", "annotation", "gene_name", "gene_id", "repSite_signals", *LIBS])


def _cluster(chrom, rep, strand, tpm, nprot, ann, signals="NA", gene="ENSG1", libs=(0, 0, 0, 0, 0, 0)):
    return "\t".join([chrom, str(rep - 1), str(rep), f"{chrom}:{rep}:{strand}", str(tpm), strand, str(rep),
                      "0.5", str(nprot), ann, "G", gene, signals, *map(str, libs)])


ATLAS_ROWS = [
    # G1's window on chr1 + runs 551..1900
    _cluster("1", 560, "+", 10.0, 3, "TE", "AATAAA@-20@540", libs=(4, 9, 2, 1, 3, 0)),   # proximal, passes
    _cluster("1", 650, "+", 0.3, 1, "TE", libs=(1, 1, 1, 1, 1, 1)),                         # one protocol: fails
    _cluster("1", 880, "+", 5.0, 2, "TE", "ATTAAA@-25@855", libs=(6, 1, 2, 1, 1, 0)),       # distal, passes
    _cluster("1", 1500, "+", 0.2, 2, "DS"),                                                 # under 5 % of 15.5: fails
    _cluster("1", 350, "+", 4.0, 2, "IN"),                                                  # intronic, inside the gene span
    _cluster("1", 2500, "+", 9.0, 3, "DS"),                                                 # past the window end
    _cluster("1", 560, "-", 100.0, 3, "TE"),                                                # wrong strand
    # G2's window on chr2 - runs, in genomic terms, 0..1200
    _cluster("2", 1150, "-", 8.0, 2, "TE", "AATAAA@-18@1168", gene="ENSG2", libs=(2, 0, 4, 0, 0, 0)),
    _cluster("2", 800, "-", 2.0, 2, "DS", gene="ENSG2", libs=(2, 0, 4, 0, 0, 0)),
    _cluster("2", 1150, "+", 100.0, 3, "TE", gene="ENSG2"),                                  # wrong strand
    # G3's window on chr1 + runs 5000..6500
    _cluster("1", 5400, "+", 3.0, 2, "TE", gene="ENSG3"),
    # mitochondrial spelling, nowhere near a gene
    _cluster("MT", 100, "+", 1.0, 2, "IG", gene="NA"),
]
ATLAS_TEXT = HEADER + "\n" + "\n".join(ATLAS_ROWS) + "\n"


# ----------------------------------------------------------------- the wiring --


class _Resp:
    def __init__(self, payload: bytes, start: int, status: int, stall_after: int | None):
        self._data, self.status, self._stall, self._sent = payload[start:], status, stall_after, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n: int) -> bytes:
        if self._stall is not None and self._sent >= self._stall:
            raise TimeoutError("timed out")
        n = min(n, 1000) if self._stall is None else min(n, 1000, self._stall - self._sent)
        chunk, self._data = self._data[:n], self._data[n:]
        self._sent += len(chunk)
        return chunk


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A registry with the three blocks, the GTF and Gene_info on disk, the atlas on a fake host."""
    import urllib.request

    import sidechain.ingest.fetch as fetch_mod

    root = tmp_path
    gencode = root / "external" / "gencode"
    gencode.mkdir(parents=True)
    with gzip.open(gencode / "gencode.v47.basic.annotation.gtf.gz", "wt") as fh:
        fh.write("\n".join(GTF_LINES) + "\n")
    (gencode / "PROVENANCE.json").write_text(json.dumps({"selected": [
        {"name": "gencode.v47.basic.annotation.gtf.gz", "size_bytes": 1, "checksum": "http-last-modified:g"}],
        "license_flags": {"noncommercial": False, "redistribution_encumbered": False}}))
    ts = root / "external" / "ts"
    ts.mkdir(parents=True)
    with zipfile.ZipFile(ts / "Gene_info.txt.zip", "w") as z:
        z.writestr("Gene_info.txt", GENE_INFO)
    (ts / "PROVENANCE.json").write_text(json.dumps({"selected": [
        {"name": "Gene_info.txt.zip", "size_bytes": 1, "checksum": "http-last-modified:t"}]}))
    atlas_dest = root / "external" / "polyasite"
    atlas_dest.mkdir(parents=True)
    (atlas_dest / "PROVENANCE.json").write_text(json.dumps({
        "route": "stream", "selected": [], "license_flags": {"noncommercial": True, "redistribution_encumbered": True}}))

    payload = gzip.compress(ATLAS_TEXT.encode())
    selected = [RemoteFile(name="atlas.clusters.2.0.GRCh38.96.tsv.gz", size_bytes=len(payload),
                           checksum="http-last-modified:a", url="https://host.example/atlas.tsv.gz")]
    monkeypatch.setattr(fetch_mod, "run_gate", lambda spec, root, **kw: (None, selected, atlas_dest))
    ranges: list[str | None] = []
    stalls: list[int | None] = [None]

    def urlopen(req, timeout=None):
        rng = req.get_header("Range")
        ranges.append(rng)
        start = int(rng.removeprefix("bytes=").rstrip("-")) if rng else 0
        stall = stalls[len(ranges) - 1] if len(ranges) <= len(stalls) else None
        return _Resp(payload, start, 206 if rng else 200, stall)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    registry = root / "data_sources.yaml"
    registry.write_text(yaml.safe_dump({"sources": [
        {"name": "polyasite_2", "kind": "node_feature", "layer": "posttx", "relation": "polyadenylation_layout",
         "host": "https", "route": "stream", "dest": "external/polyasite", "derived": "derived/polyasite",
         "license": "UNSTATED", "files": [{"name": "atlas.clusters.2.0.GRCh38.96.tsv.gz"}]},
        {"name": "gencode_v47_basic", "kind": "node_feature", "layer": "cis", "relation": "transcript_structure",
         "host": "https", "dest": "external/gencode", "derived": "derived/gencode",
         "license": "GENCODE-open-access", "files": [{"name": "gencode.v47.basic.annotation.gtf.gz"}]},
        {"name": "targetscan", "kind": "edge", "layer": "posttx", "relation": "mirna_target",
         "dest": "external/ts", "derived": "derived/ts", "license": "Free-for-research-with-citation",
         "entity_src": "mirna_family"},
    ]}))
    from sidechain.priors.posttx_mirna import spec_from_registry
    src = PolyASiteSource(spec_from_registry("polyasite_2", registry), {}, root=root, registry=registry)
    return src, payload, ranges, stalls, atlas_dest


# ---------------------------------------------------------------- the helpers --


def test_gtf_attributes_keep_first_value_and_collect_tags():
    attrs, tags = parse_gtf_attributes('gene_id "ENSG1.2"; transcript_id "ENST1.3"; tag "basic"; tag "MANE_Select"; level 2;')
    assert attrs["gene_id"] == "ENSG1.2" and attrs["transcript_id"] == "ENST1.3"
    assert tags == {"basic", "MANE_Select"}


def test_atlas_chromosome_spelling():
    assert atlas_chrom("10") == "chr10" and atlas_chrom("MT") == "chrM" and atlas_chrom("chrX") == "chrX"


@pytest.mark.parametrize("header, line", [
    ("GSM1|A-seq|HeLa cells|t|none", "hela"),
    ("GSM1|A-seq|HeLa|t|", "hela"),
    ("GSM1|A-seq|HeLa cells|t|si-Ctrl", "hela"),
    ("GSM1|A-seq|HeLa cells|t|si-CstF64/64tau", None),
    ("GSM1|3P|HEK293|t|siRNA Ctrl", "hek293"),
    ("GSM1|3P|293T|t|N/A", "hek293"),
    ("GSM1|3P|HEK293|t|siRNA CF Im68", None),
    ("GSM1|DRS|K562|t|siRNA control", "k562"),
    ("GSM1|DRS|K562|t|siRNA aCP", None),
    ("GSM1|PAS|testis|t|none", None),
    ("GSM1|PAS|MCF10A|t|expression of HRAS (G12V)", None),
])
def test_library_classification(header, line):
    assert classify_library(header)[0] == line


# ------------------------------------------------------------ the transcripts --


def test_transcript_table_last_exon_stop_and_utr_on_both_strands(wired):
    src, *_ = wired
    tx = src.transcript_table().set_index("transcript_id")
    t1 = tx.loc["ENST1"]
    assert (t1["acceptor"], t1["stop_3p"], t1["stop_in_last_exon"]) == (500, 550, True)
    assert t1["exonic_stop_to_acceptor"] == 0 and t1["utr3_len"] == 150 and t1["transcript_version"] == 5
    assert t1["mane_select"] and not t1["ensembl_canonical"] and t1["n_exons"] == 3
    t3 = tx.loc["ENST3"]
    # minus strand: the last exon is the lowest, the acceptor its HIGHER end, the stop in the middle exon
    assert (t3["last_exon_start"], t3["last_exon_end"], t3["acceptor"]) == (1000, 1200, 1200)
    assert t3["stop_3p"] == 2050 and not t3["stop_in_last_exon"]
    assert t3["exonic_stop_to_acceptor"] == 50 and t3["utr3_len"] == 251
    t4 = tx.loc["ENST4"]
    assert np.isnan(t4["acceptor"]) and np.isnan(t4["stop_3p"]) and t4["cds_len"] == 0 and np.isnan(t4["utr3_len"])
    assert (t4["gene_start"], t4["gene_end"]) == (5000, 5500)


def test_representative_rules_in_order(wired):
    src, *_ = wired
    reps = src.representatives(src.transcript_table()).set_index("gene_id")
    assert reps.loc["ENSG1", "transcript_id"] == "ENST1" and reps.loc["ENSG1", "pa_tx_source"] == "targetscan"
    assert bool(reps.loc["ENSG1", "pa_tx_version_changed"])        # v19 .3 against v47 .5
    assert reps.loc["ENSG2", "pa_tx_source"] == "mane_select"       # TargetScan's row names another gene
    assert reps.loc["ENSG3", "pa_tx_source"] == "ensembl_canonical"
    assert reps.loc["ENSG5", "transcript_id"] == "ENST7" and reps.loc["ENSG5", "pa_tx_source"] == "longest_cds"


# ------------------------------------------------------------------ the genes --


def test_gene_table_window_filter_distances_and_usage(wired):
    src, payload, ranges, _stalls, atlas_dest = wired
    g = src.gene_table().set_index("gene_id")
    assert list(g.columns[: len(GENE_COLUMNS) - 1]) == list(GENE_COLUMNS)[1:]
    r = g.loc["ENSG1"]
    assert r["pa_n_sites_all"] == 4 and r["pa_n_sites"] == 2 and r["pa_has_apa"]
    assert r["pa_window_tpm_sum"] == pytest.approx(15.5)
    assert r["pa_acc_to_prox_nt"] == 60 and r["pa_acc_to_prox_nt_all"] == 60
    assert r["pa_log10_acc_to_prox"] == pytest.approx(np.log10(61))
    assert (r["pa_stop_to_prox_nt"], r["pa_stop_to_dist_nt"], r["pa_span_nt"]) == (10, 330, 320)
    assert r["pa_prox_usage"] == pytest.approx(10 / 15)
    assert r["pa_prox_has_canonical_pas"] and r["pa_prox_nr_prots"] == 3
    assert r["pa_n_intronic"] == 1 and r["pa_window_te_ds_share"] == 1.0
    assert r["pa_te_len"] == 201 and r["pa_window_len"] == 1900 - 551 + 1
    # per-line usage over the unperturbed libraries only: HeLa = GSM1, HEK293 = GSM3, K562 = GSM5
    assert r["pa_prox_usage_hela"] == pytest.approx(4 / (4 + 6))
    assert r["pa_prox_usage_hek293"] == pytest.approx(2 / (2 + 2))
    assert r["pa_prox_usage_k562"] == pytest.approx(3 / (3 + 1))
    assert r["pa_window_tpm_hela"] == pytest.approx(4 + 1 + 6 + 0)

    r2 = g.loc["ENSG2"]
    assert r2["pa_n_sites"] == 2 and r2["pa_acc_to_prox_nt"] == 50
    assert (r2["pa_stop_to_prox_nt"], r2["pa_stop_to_dist_nt"], r2["pa_span_nt"]) == (101, 451, 350)
    assert r2["pa_prox_usage"] == pytest.approx(0.8) and r2["pa_prox_has_canonical_pas"]

    r3 = g.loc["ENSG3"]
    assert r3["pa_single_exon"] and r3["pa_n_sites"] == 1 and np.isnan(r3["pa_acc_to_prox_nt"])
    assert np.isnan(r3["pa_stop_to_prox_nt"]) and not r3["pa_coding"]

    r4 = g.loc["ENSG4"]
    assert r4["pa_n_sites"] == 0 and r4["pa_n_sites_all"] == 0 and not r4["pa_has_apa"]
    assert np.isnan(r4["pa_acc_to_prox_nt"]) and r4["pa_window_tpm_sum"] == 0

    # nothing of the atlas landed; the lineage carries the sha256 of the bytes READ
    assert sorted(p.name for p in atlas_dest.iterdir()) == ["PROVENANCE.json"]
    lineage = json.loads((src.derived / "LINEAGE.json").read_text())
    entry = lineage["entries"]["polyasite-2.0/pas_gene_features"]
    assert entry["inputs"]["atlas.clusters.2.0.GRCh38.96.tsv.gz"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["inputs"]["atlas.clusters.2.0.GRCh38.96.tsv.gz"]["streamed"] is True
    assert entry["licence"] == "UNSTATED" and entry["license_flags"] == {"noncommercial": True, "redistribution_encumbered": True}
    assert entry["counts"]["clusters"] == len(ATLAS_ROWS) and entry["counts"]["genes_out"] == 5
    assert entry["counts"]["line_libraries"] == {"hela": 1, "hek293": 1, "k562": 1}
    assert entry["line_columns"]["k562"] == ["GSM5|DRS|K562|K562 ctrl|siRNA control"]
    assert entry["filter"] == {"min_protocols": 2, "min_tpm_frac_of_window": 0.05, "downstream_nt": 1000}
    # the cache: a second call reads the parquet, not the network
    n = len(ranges)
    src.gene_table()
    assert len(ranges) == n


def test_gene_table_survives_a_stalled_stream(wired):
    src, _payload, ranges, stalls, _ = wired
    stalls[:] = [7, 0, 40, None]
    g = src.gene_table().set_index("gene_id")
    assert g.loc["ENSG1", "pa_n_sites"] == 2
    assert ranges == [None, "bytes=7-", "bytes=7-", "bytes=47-"]


def test_build_is_a_node_feature_with_a_mask(wired):
    src, *_ = wired
    src.gene_index = {"ENSG1": 0, "ENSG2": 1, "ENSG3": 2, "ENSGX": 3, "ENSG4": 4}
    art = src.build_checked()
    assert art.kind == "node_feature" and art.features.shape == (5, 3)
    assert art.meta["feature_names"] == ["pa_log10_acc_to_prox", "pa_n_sites", "pa_prox_usage"]
    assert art.features[0, 0] == pytest.approx(np.log10(61)) and art.features[0, 1] == 2
    assert art.meta["has_value"][2].tolist() == [False, True, True]       # single-exon: no acceptor distance
    assert art.meta["has_value"][3].tolist() == [False, False, False]     # off the atlas
    assert art.meta["has_value"][4].tolist() == [False, True, False] and art.features[4, 1] == 0
    assert art.meta["licence"] == "UNSTATED"


def test_fetch_checks_the_gencode_md5_after_landing(wired, monkeypatch):
    import sidechain.priors.posttx_polya as mod

    src, *_ = wired
    gencode = src._for("gencode_v47_basic")
    monkeypatch.setattr(mod, "fetch_gated_files", lambda spec, root, **kw: gencode.dest)
    with pytest.raises(RuntimeError, match="md5"):
        gencode.fetch()
    assert (gencode.dest / "gencode.v47.basic.annotation.gtf.gz").exists()      # kept for a look
    monkeypatch.setattr(mod, "GENCODE_MD5", mod._md5_of(gencode.dest / "gencode.v47.basic.annotation.gtf.gz"))
    assert gencode.fetch() == gencode.dest
    assert GENCODE_MD5 == "823c2d44a2268d2c79a11986ffa29028"


def test_fetch_of_the_stream_block_writes_nothing_but_provenance(wired):
    src, _payload, ranges, _stalls, atlas_dest = wired
    assert src.fetch() == atlas_dest
    assert sorted(p.name for p in atlas_dest.iterdir()) == ["PROVENANCE.json"] and ranges == []


def test_registry_blocks_are_shelved_and_declare_their_gate(tmp_path):
    from sidechain.priors.posttx_mirna import spec_from_registry

    pa = spec_from_registry("polyasite_2")
    assert pa["kind"] == "node_feature" and pa["layer"] == "posttx" and pa["enabled"] is False
    assert pa["host"] == "https" and pa["route"] == "stream" and pa["license"] == "UNSTATED"
    assert pa["license_override_source"] and pa["allow_missing_checksum"] is True
    assert pa["dest"].startswith("external/") and pa["derived"].startswith("derived/")
    assert [f["name"] for f in pa["files"]] == ["atlas.clusters.2.0.GRCh38.96.tsv.gz"]
    gc = spec_from_registry("gencode_v47_basic")
    assert gc["kind"] == "node_feature" and gc["layer"] == "cis" and gc["enabled"] is False
    assert gc["host"] == "https" and "route" not in gc and gc["license"] == "GENCODE-open-access"
    assert [f["name"] for f in gc["files"]] == ["gencode.v47.basic.annotation.gtf.gz"]
    assert GENCODE_MD5 in gc["notes"]
    from sidechain.ingest.provenance import (
        ACCEPTED_LICENSES,
        NONCOMMERCIAL,
        REDISTRIBUTION_ENCUMBERED,
    )
    assert "GENCODE-open-access" in ACCEPTED_LICENSES
    assert "GENCODE-open-access" not in NONCOMMERCIAL | REDISTRIBUTION_ENCUMBERED


def test_registry_resolves_the_loader():
    from sidechain.data.registry import LOADERS
    assert LOADERS["PolyASiteSource"] is PolyASiteSource


def test_wired_fixture_root_is_isolated(wired, tmp_path):
    src, *_ = wired
    assert Path(src.root) == tmp_path
