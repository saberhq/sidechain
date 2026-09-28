"""Contract tests for `MiRNATargetSource` (T102) on a synthetic TargetScan release.

The parsers are pinned on the properties that decide the numbers:

* only human (9606) rows and only REPRESENTATIVE transcripts make a gene row;
* the 3'UTR length is the sum of the transcript's GFF exon widths, read from the publisher's
  score column (which equals ``end - start``: the coordinates are half-open despite the
  extension) -- an off-by-one per exon is the failure this guards;
* a gene with no row in the site table has ZERO sites and a zero context score, not NaN;
* two representative transcripts sharing a symbol collapse to one row (the longer UTR) and
  the collapse is counted in LINEAGE.json;
* the Summary_Counts files key a family by its SEED, the site files by its NAME, and the
  edge table maps one onto the other through miR_Family_Info;
* the site table flags default predictions, keeps nonconserved sites of conserved families,
  and joins the context-score file whose coordinates sit one below (both ends);
* the sequence table verifies the site coordinate convention against the seed complement,
  then counts the NAR 2016 co-occurrence (within 200 nt) and competition (overlap) events;
* `build()` is a bipartite artifact: row 0 indexes the families, row 1 the gene space, one
  mask for both, and `build_checked` accepts it.
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from sidechain.priors.posttx_mirna import (
    DINUCLEOTIDES,
    EDGE_COLUMNS,
    SITE_COLUMNS,
    UTR_LOAD_COLUMNS,
    MiRNATargetSource,
    revcomp_rna,
)

GENE_INFO = (
    "Transcript ID\tGene ID\tGene symbol\tGene description\tSpecies ID\t3P-seq tags\tRepresentative transcript?\n"
    "ENST1.3\tENSG1.7\tA1BG\tdesc\t9606\t74\t1\n"
    "ENST2.2\tENSG2.10\tA1CF\tdesc\t9606\t80\t1\n"
    "ENST3.1\tENSG2.10\tA1CF\tdesc\t9606\t2\t0\n"          # not representative: ignored
    "ENST4.1\tENSG4.1\tNOSITES\tdesc\t9606\t5\t1\n"        # in the GFF, absent from the site table
    "ENST5.1\tENSG5.1\tDUP\tdesc\t9606\t5\t1\n"            # two representative transcripts, one symbol
    "ENST6.1\tENSG6.1\tDUP\tdesc\t9606\t5\t1\n"
    "ENST7.1\tENSG7.1\tNOUTR\tdesc\t9606\t5\t1\n"          # representative but no GFF row: dropped
    "ENSMUST1.1\tENSMUSG1.1\tA1bg\tdesc\t10090\t1\t1\n"    # mouse: ignored
)

GFF = (
    "browser pack wgEncodeGencodeBasicV19\n"
    'track name="Reference 3-prime UTRs" description="x" visibility=2\n'
    "chr1\tTS7\tUTR\t100\t200\t100\t+\t.\tENST1.3\n"       # score == end - start
    "chr1\tTS7\tUTR\t300\t350\t50\t+\t.\tENST1.3\n"        # second exon of the same UTR
    "chr2\tTS7\tUTR\t10\t1010\t1000\t-\t.\tENST2.2\n"
    "chr3\tTS7\tUTR\t10\t510\t500\t-\t.\tENST4.1\n"
    "chr4\tTS7\tUTR\t10\t110\t100\t-\t.\tENST5.1\n"
    "chr4\tTS7\tUTR\t10\t2010\t2000\t-\t.\tENST6.1\n"
)

COUNTS_HEADER = ("Transcript ID\tGene Symbol\tmiRNA family\tSpecies ID\tTotal num conserved sites\t"
                 "Number of conserved 8mer sites\tNumber of conserved 7mer-m8 sites\t"
                 "Number of conserved 7mer-1a sites\tTotal num nonconserved sites\t"
                 "Number of nonconserved 8mer sites\tNumber of nonconserved 7mer-m8 sites\t"
                 "Number of nonconserved 7mer-1a sites\tNumber of 6mer sites\tRepresentative miRNA\t"
                 "Total context++ score\tCumulative weighted context++ score\tAggregate PCT\t"
                 "Predicted occupancy - low miRNA\tPredicted occupancy - high miRNA\t"
                 "Predicted occupancy - transfected miRNA\n")
# the `miRNA family` column is the SEED+m8, as in the real files (GCAGCAU), not the name
COUNTS = COUNTS_HEADER + (
    "ENST1.3\tA1BG\tGGAAUGU\t9606\t3\t1\t2\t0\t1\t0\t0\t1\t0\thsa-miR-1-3p\t-0.6\t-0.5\t0.9\tNULL\tNULL\tNULL\n"
    "ENST1.3\tA1BG\tUAAGGCA\t9606\t2\t2\t0\t0\t0\t0\t0\t0\t0\thsa-miR-2-5p\t-0.3\t-0.2\t0.4\tNULL\tNULL\tNULL\n"
    "ENST1.3\tA1BG\tGGAAUGU\t10090\t3\t0\t0\t3\t1\t0\t0\t1\t0\tmmu-miR-1a-3p\t-0.6\t-0.6\t1.0\tNULL\tNULL\tNULL\n"   # mouse row
    "ENST2.2\tA1CF\tGGAAUGU\t9606\t0\t0\t0\t0\t4\t1\t1\t2\t0\thsa-miR-1-3p\t-0.1\t-0.05\t0.0\tNULL\tNULL\tNULL\n"  # nonconserved only
    "ENST6.1\tDUP\tGGAAUGU\t9606\t7\t7\t0\t0\t0\t0\t0\t0\t0\thsa-miR-1-3p\t-1.0\t-0.9\t0.99\tNULL\tNULL\tNULL\n"
)
# the all-predictions file: the default rows plus a nonconserved family (no human family row)
COUNTS_ALL = COUNTS + (
    "ENST1.3\tA1BG\tCCCCCCC\t9606\t0\t0\t0\t0\t2\t0\t1\t1\t0\thsa-miR-9999\t-0.05\t-0.02\tNULL\tNULL\tNULL\tNULL\n"
    "ENST3.1\tA1CF\tGGAAUGU\t9606\t1\t1\t0\t0\t0\t0\t0\t0\t0\thsa-miR-1-3p\t-0.4\t-0.4\t0.5\tNULL\tNULL\tNULL\n"  # not representative
)

FAMILY = (
    "miR family\tSeed+m8\tSpecies ID\tMiRBase ID\tMature sequence\tFamily Conservation?\tMiRBase Accession\n"
    "miR-1\tGGAAUGU\t9606\thsa-miR-1-3p\tUGGAAUGUAAAGAAGUAUGUAU\t2\tMIMAT0000416\n"
    "miR-2\tUAAGGCA\t9606\thsa-miR-2-5p\tAUAAGGCAUUUUUUUUUUUUUU\t2\tMIMAT0000002\n"
    "miR-1\tGGAAUGU\t10090\tmmu-miR-1a-3p\tUGGAAUGUAAAGAAGUAUGUAU\t2\tMIMAT0000123\n"
)

SITES_HEADER = ("miR Family\tGene ID\tGene Symbol\tTranscript ID\tSpecies ID\tUTR start\tUTR end\t"
                "MSA start\tMSA end\tSeed match\tPCT\n")
# the default predictions: conserved sites of conserved families, with positions (1-based)
PREDICTED = SITES_HEADER + (
    "miR-1\tENSG1.7\tA1BG\tENST1.3\t9606\t21\t28\t30\t37\t8mer\t0.85\n"
    "miR-2\tENSG1.7\tA1BG\tENST1.3\t9606\t100\t106\t140\t146\t7mer-m8\t0.50\n"
    "miR-1\tENSG6.1\tDUP\tENST6.1\t9606\t10\t17\t10\t17\t8mer\t0.90\n"
    "miR-1\tENSMUSG1.1\tA1bg\tENSMUST1.1\t10090\t21\t28\t30\t37\t8mer\t0.85\n"      # mouse row
)
# the family file: the same plus a nonconserved site (PCT NULL) and a duplicate row
CONSERVED_FAMILY = PREDICTED + (
    "miR-1\tENSG1.7\tA1BG\tENST1.3\t9606\t130\t136\t180\t186\t7mer-a1\tNULL\n"
    "miR-1\tENSG1.7\tA1BG\tENST1.3\t9606\t21\t28\t30\t37\t8mer\t0.85\n"              # duplicate
)

CONTEXT = (
    "Gene ID\tGene Symbol\tTranscript ID\tGene Tax ID\tmiRNA\tSite Type\tUTR_start\tUTR end\t"
    "context++ score\tcontext++ score percentile\tweighted context++ score\t"
    "weighted context++ score percentile\tPredicted relative KD\n"
    # coordinates one BELOW the family files' (both ends), as in the real release
    "ENSG1.7\tA1BG\tENST1.3\t9606\thsa-miR-1-3p\t3\t20\t27\t-0.50\t98\t-0.45\t97\tNULL\n"
    "ENSG1.7\tA1BG\tENST1.3\t9606\thsa-miR-2-5p\t2\t99\t105\t-0.25\t90\t-0.20\t88\tNULL\n"
    "ENSG1.7\tA1BG\tENST1.3\t10090\tmmu-miR-1a-3p\t3\t20\t27\t-0.50\t98\t-0.99\t97\tNULL\n"  # mouse row
)

# ENST1's 150-nt UTR, laid out so every count below is known by hand (1-based positions):
#   21-28  ACAUUCCA  the miR-1 8mer site (revcomp of seed GGAAUGU, then A)
#   60-67  UGUAAAUA  a PUM motif, 32 nt after site 1 and 33 nt before site 2 (within 200 of both)
#  100-106 UGCCUUA   the miR-2 7mer-m8 site (revcomp of UAAGGCA)
#  106-110 AUUUA     an ARE sharing site 2's last nucleotide (overlap = competition)
#  130-136 CAUUCCA   the nonconserved 7mer-a1 site (not counted: not a default prediction)
UTR1 = ("C" * 20 + "ACAUUCCA" + "C" * 31 + "UGUAAAUA" + "C" * 32 + "UGCCUUA" + "UUUA"
        + "C" * 19 + "CAUUCCA" + "C" * 14)
assert len(UTR1) == 150
UTR_SEQ = (
    "Refseq ID\tGene ID\tGene Symbol\tSpecies ID\tUTR sequence\n"
    f"ENST1.3\tENSG1.7\tA1BG\t9606\t{UTR1[:40]}----{UTR1[40:]}\n"                 # gapped, as in the alignment
    "ENST1.3\tENSG1.7\tA1BG\t10090\tCCCC----CCCC\n"                                 # mouse row
    "ENST2.2\tENSG2.10\tA1CF\t9606\tACGUACGUAC\n"
    "ENST3.1\tENSG2.10\tA1CF\t9606\tACGU\n"                                         # not representative
    "ENST6.1\tENSG6.1\tDUP\t9606\t" + "G" * 9 + "ACAUUCCA" + "G" * 3 + "\n"          # site at 10-17
)


def _zip(path, member, text):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(member, text)


@pytest.fixture
def source(tmp_path):
    root = tmp_path
    dest = root / "external" / "ts"
    dest.mkdir(parents=True)
    files = {
        "Gene_info.txt.zip": ("Gene_info.txt", GENE_INFO),
        "TSHuman_7_hg19_3UTRs.gff.zip": ("TSHuman_7_hg19_3UTRs.gff", GFF),
        "Summary_Counts.default_predictions.txt.zip": ("Summary_Counts.default_predictions.txt", COUNTS),
        "Summary_Counts.all_predictions.txt.zip": ("Summary_Counts.all_predictions.txt", COUNTS_ALL),
        "miR_Family_Info.txt.zip": ("miR_Family_Info.txt", FAMILY),
        "Predicted_Targets_Info.default_predictions.txt.zip": ("Predicted_Targets_Info.default_predictions.txt", PREDICTED),
        "Conserved_Family_Info.txt.zip": ("Conserved_Family_Info.txt", CONSERVED_FAMILY),
        "Conserved_Site_Context_Scores.txt.zip": ("Conserved_Site_Context_Scores.txt", CONTEXT),
        "UTR_Sequences.txt.zip": ("UTR_Sequences.txt", UTR_SEQ),
    }
    for name, (member, text) in files.items():
        _zip(dest / name, member, text)
    (dest / "PROVENANCE.json").write_text(json.dumps({"selected": [
        {"name": n, "size_bytes": (dest / n).stat().st_size, "checksum": "http-last-modified:x"}
        for n in files]}))
    spec = {"name": "targetscan", "kind": "edge", "layer": "posttx", "relation": "mirna_target",
            "dest": "external/ts", "derived": "derived/ts", "license": "Free-for-research-with-citation",
            "entity_src": "mirna_family"}
    return MiRNATargetSource(spec, {}, root=root)


# ------------------------------------------------------------ the per-gene load --

def test_table_rows_lengths_counts_and_zeros(source):
    t = source.utr_load_table(rebuild=True).set_index("symbol")
    assert list(t.reset_index().columns) == list(UTR_LOAD_COLUMNS)
    assert set(t.index) == {"A1BG", "A1CF", "NOSITES", "DUP"}        # NOUTR dropped, mouse ignored
    assert t.loc["A1BG", "utr_len"] == 150                             # 100 + 50, the score column
    assert t.loc["A1BG", "n_cons_sites"] == 5 and t.loc["A1BG", "n_cons_8mer"] == 3
    assert t.loc["A1BG", "n_cons_7mer"] == 2 and t.loc["A1BG", "n_families"] == 2
    assert t.loc["A1BG", "context_score"] == pytest.approx(-0.7)      # -0.5 + -0.2, mouse row excluded
    assert t.loc["A1BG", "aggregate_pct_max"] == pytest.approx(0.9)
    assert t.loc["A1BG", "sites_per_kb"] == pytest.approx(5 / 0.150)
    assert t.loc["A1CF", "n_cons_sites"] == 0 and t.loc["A1CF", "n_noncons_sites"] == 4
    assert t.loc["NOSITES", "n_cons_sites"] == 0 and t.loc["NOSITES", "context_score"] == 0.0
    assert t.loc["NOSITES", "utr_len"] == 500
    # DUP: two representative transcripts; the longer UTR's row (ENST6, 2000 nt, 7 sites) wins
    assert t.loc["DUP", "transcript_id"] == "ENST6" and t.loc["DUP", "utr_len"] == 2000
    assert t.loc["DUP", "n_cons_sites"] == 7


def test_lineage_counts_and_cache(source):
    source.utr_load_table(rebuild=True)
    lin = json.loads((source.derived / "LINEAGE.json").read_text())
    c = lin["entries"]["targetscan-vert_80/utr_load"]["counts"]
    assert c["representative_transcripts"] == 6 and c["genes_out"] == 4
    assert c["dropped_no_utr_in_gff"] == 1 and c["symbols_deduplicated"] == 1
    assert c["gff_score_equals_width_on_rows"] == 6
    assert c["summary_counts_rows"] == 5 and c["summary_counts_human_rows"] == 4   # all species vs human
    # the cache is read back without rebuilding
    again = source.utr_load_table()
    assert len(again) == 4


def test_lineage_entries_merge_rather_than_overwrite(source):
    source.utr_load_table(rebuild=True)
    source.edge_table(rebuild=True)
    lin = json.loads((source.derived / "LINEAGE.json").read_text())
    assert {"targetscan-vert_80/utr_load", "targetscan-vert_80/edges_default"} <= set(lin["entries"])


# ---------------------------------------------------------------- the edge form --

def test_edge_table_maps_seeds_to_family_names_and_keeps_representative_transcripts(source):
    e = source.edge_table(rebuild=True)
    assert list(e.columns) == list(EDGE_COLUMNS)
    pairs = set(zip(e["family"], e["gene_id"]))
    assert pairs == {("miR-1", "ENSG1"), ("miR-2", "ENSG1"), ("miR-1", "ENSG2"), ("miR-1", "ENSG6")}
    row = e.set_index(["family", "gene_id"]).loc[("miR-1", "ENSG1")]
    assert row["seed_m8"] == "GGAAUGU" and row["n_cons_sites"] == 3 and row["n_cons_7mer_m8"] == 2
    assert row["context_score_weighted"] == pytest.approx(-0.5) and row["aggregate_pct"] == pytest.approx(0.9)
    assert bool(e["family_conserved"].all())
    lin = json.loads((source.derived / "LINEAGE.json").read_text())
    c = lin["entries"]["targetscan-vert_80/edges_default"]["counts"]
    assert c["rows_all_species"] == 5 and c["rows_human"] == 4 and c["edges_out"] == 4
    assert c["seeds_without_a_human_family_row"] == 0 and c["families"] == 2


def test_edge_table_all_scope_adds_nonconserved_families_and_names_unmapped_seeds(source):
    e = source.edge_table(scope="all", rebuild=True)
    assert len(e) == 5                                                  # ENST3 (not representative) dropped
    extra = e[e["seed_m8"] == "CCCCCCC"].iloc[0]
    assert extra["family"] == "CCCCCCC" and not extra["family_conserved"]
    assert extra["n_noncons_sites"] == 2 and np.isnan(extra["aggregate_pct"])
    assert e.loc[e["seed_m8"] == "GGAAUGU", "family_conserved"].all()
    with pytest.raises(ValueError, match="scope"):
        source.edge_table(scope="everything")


def test_build_is_a_bipartite_artifact_the_contract_accepts(source):
    src = MiRNATargetSource(source.spec, {"ENSG1": 0, "ENSG2": 1, "ENSG6": 2, "ENSG9": 3}, root=source.root)
    art = src.build_checked()
    assert art.kind == "edge" and art.src_names == ["miR-1", "miR-2"]
    assert art.edge_index.shape == (2, 4) and art.edge_index.dtype == np.int64
    assert art.edge_index[0].max() == 1 and art.edge_index[1].max() == 2
    assert art.edge_attr.shape == (4, 3)
    assert art.meta["edge_attr_names"][1] == "context_score_weighted"
    # the pair (miR-2, ENSG1): family index 1, gene index 0, weighted context -0.2
    k = [i for i in range(4) if art.edge_index[0, i] == 1 and art.edge_index[1, i] == 0]
    assert len(k) == 1 and art.edge_attr[k[0], 1] == pytest.approx(-0.2)


def test_build_drops_a_pair_whose_gene_is_off_the_axis_with_its_attr(source):
    src = MiRNATargetSource(source.spec, {"ENSG1": 0, "ENSG2": 1}, root=source.root)   # no ENSG6
    art = src.build_checked()
    assert art.edge_index.shape == (2, 3) and art.edge_attr.shape == (3, 3)
    assert art.meta["candidate_edges"] == 4 and art.meta["kept_edges"] == 3


def test_build_checked_rejects_a_bipartite_row_out_of_range():
    from sidechain.priors.base import PriorArtifact, PriorSource

    class Bad(PriorSource):
        def fetch(self):
            pass

        def build(self):
            return PriorArtifact(kind="edge", relation="r", layer="posttx",
                                 edge_index=np.array([[0, 5], [0, 1]], dtype=np.int64),
                                 src_names=["f0", "f1"])

    with pytest.raises(ValueError, match="row 0 out of range"):
        Bad({"name": "bad", "kind": "edge", "layer": "posttx", "relation": "r"}, {"a": 0, "b": 1}).build_checked()


# -------------------------------------------------------------------- the sites --

def test_site_table_flags_default_predictions_dedupes_and_joins_context_one_below(source):
    s = source.site_table(rebuild=True)
    assert list(s.columns) == list(SITE_COLUMNS)
    a1bg = s[s["transcript_id"] == "ENST1"].sort_values("utr_start")
    assert list(a1bg["utr_start"]) == [21, 100, 130]                   # the duplicate row collapsed
    assert list(a1bg["conserved_site"]) == [True, True, False]
    assert list(a1bg["site_type"]) == ["8mer", "7mer-m8", "7mer-a1"]
    assert a1bg["pct"].tolist()[:2] == pytest.approx([0.85, 0.50]) and np.isnan(a1bg["pct"].iloc[2])
    # context++ joined at (start - 1, end - 1); the nonconserved site has no row
    assert a1bg["context_pp"].tolist()[:2] == pytest.approx([-0.45, -0.20])
    assert np.isnan(a1bg["context_pp"].iloc[2])
    assert bool(s["representative"].all())
    assert (s["transcript_id"] == "ENSMUST1").sum() == 0                # mouse rows filtered
    lin = json.loads((source.derived / "LINEAGE.json").read_text())
    c = lin["entries"]["targetscan-vert_80/sites"]["counts"]
    assert c["sites_out"] == 4 and c["sites_conserved"] == 3 and c["duplicates_dropped"] == 1
    assert c["context_join_rate_on_conserved_sites"] == pytest.approx(2 / 3)   # DUP's site has no row


def test_site_coordinate_check_finds_the_one_based_convention(source):
    s = source.site_table(rebuild=True)
    seqs = {"ENST1": UTR1, "ENST6": "G" * 9 + "ACAUUCCA" + "G" * 3}
    conv = source.check_site_coordinates(s, seqs)
    assert conv["n"] == 3 and conv["one_based"] == 1.0 and conv["zero_based"] == 0.0
    assert conv["convention"] == "one_based"
    # shift every site by one and the zero-based reading wins instead
    shifted = s.copy()
    shifted["utr_start"] = shifted["utr_start"] - 1
    shifted["utr_end"] = shifted["utr_end"] - 1
    conv2 = source.check_site_coordinates(shifted, seqs)
    assert conv2["convention"] == "zero_based" and conv2["zero_based"] == 1.0


# ---------------------------------------------------------- sequence features --

def test_sequence_features_composition_and_motifs():
    f = MiRNATargetSource.sequence_features(UTR1)
    assert f["seq_len"] == 150
    assert sum(f[f"dn_{d}"] for d in DINUCLEOTIDES) == pytest.approx(1.0)
    assert f["frac_A"] + f["frac_C"] + f["frac_G"] + f["frac_U"] == pytest.approx(1.0)
    assert f["au_content"] == pytest.approx(f["frac_A"] + f["frac_U"])
    assert f["m_pum"] == 1 and f["m_are_auuua"] == 1 and f["m_are_nonamer"] == 0
    assert f["m_polyu"] == 0 and f["m_msi"] == 0 and f["m_qki"] == 0 and f["m_ca_repeat"] == 0
    empty = MiRNATargetSource.sequence_features("")
    assert empty["seq_len"] == 0 and np.isnan(empty["au_content"]) and empty["m_pum"] == 0


def test_cooperation_counts_by_hand():
    sites = [(21, 28), (100, 106)]
    motifs = MiRNATargetSource.motif_intervals(UTR1)
    assert motifs["pum"] == [(60, 67)] and motifs["are_auuua"] == [(106, 110)]
    c = MiRNATargetSource.cooperation_counts(sites, motifs, 150)
    assert c["n_cons_sites_seq"] == 2
    assert c["n_sites_rbp_within200"] == 2 and c["n_sites_pum_within200"] == 2
    assert c["n_sites_are_auuua_within200"] == 2
    assert c["n_sites_rbp_overlap"] == 1 and c["frac_sites_rbp_overlap"] == pytest.approx(0.5)
    assert c["n_sites_polyu_within200"] == 0 and c["n_sites_msi_within200"] == 0
    assert c["n_sites_first15pct"] == 1 and c["n_sites_last15pct"] == 0 and c["n_sites_last500nt"] == 2
    # a window too short to reach the PUM motif from site 1 (gap 32) leaves only site 2 (gap 33)
    c2 = MiRNATargetSource.cooperation_counts(sites, {"pum": motifs["pum"]}, 150, window=32)
    assert c2["n_sites_pum_within200"] == 1
    none = MiRNATargetSource.cooperation_counts([], motifs, 150)
    assert none["n_cons_sites_seq"] == 0 and np.isnan(none["frac_sites_rbp_overlap"])


def test_utr_sequence_table_streams_human_representative_rows_and_records_the_convention(source):
    t = source.utr_sequence_table(rebuild=True).set_index("transcript_id")
    assert set(t.index) == {"ENST1", "ENST2", "ENST6"}                 # ENST3 not representative, mouse out
    assert t.loc["ENST1", "seq_len"] == 150 and t.loc["ENST2", "seq_len"] == 10   # gaps stripped
    assert t.loc["ENST1", "n_sites_rbp_overlap"] == 1 and t.loc["ENST1", "n_sites_pum_within200"] == 2
    assert t.loc["ENST6", "n_cons_sites_seq"] == 1 and t.loc["ENST2", "n_cons_sites_seq"] == 0
    lin = json.loads((source.derived / "LINEAGE.json").read_text())
    e = lin["entries"]["targetscan-vert_80/utr_seq_features"]
    assert e["site_coordinate_check"]["convention"] == "one_based" and e["coordinate_offset_applied"] == 0
    assert e["counts"]["utr_rows_human"] == 4 and e["counts"]["representative_transcripts_with_sequence"] == 3
    assert "pum" in e["motifs"]


def test_utr_features_table_joins_load_and_sequence(source):
    t = source.utr_features_table(rebuild=True).set_index("symbol")
    assert t.loc["A1BG", "utr_len"] == 150 and t.loc["A1BG", "seq_len"] == 150
    assert t.loc["A1BG", "n_sites_rbp_overlap"] == 1
    assert np.isnan(t.loc["NOSITES", "seq_len"])                        # no sequence row: left join keeps the gene


def test_revcomp_rna():
    assert revcomp_rna("GGAAUGU") == "ACAUUCC"
