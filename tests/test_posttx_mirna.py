"""Contract tests for `MiRNATargetSource.utr_load_table` (T102) on synthetic TargetScan files.

The three parsers are pinned on the properties that decide the numbers:

* only human (9606) rows and only REPRESENTATIVE transcripts make a gene row;
* the 3'UTR length is the sum of the transcript's GFF exon widths, read from the publisher's
  score column (which equals ``end - start``: the coordinates are half-open despite the
  extension) -- an off-by-one per exon is the failure this guards;
* a gene with no row in the site table has ZERO sites and a zero context score, not NaN;
* two representative transcripts sharing a symbol collapse to one row (the longer UTR) and
  the collapse is counted in LINEAGE.json;
* the edge form (`build`) is unbuilt and says so.
"""
from __future__ import annotations

import json
import zipfile

import pytest

from sidechain.priors.posttx_mirna import UTR_LOAD_COLUMNS, MiRNATargetSource

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
COUNTS = COUNTS_HEADER + (
    "ENST1.3\tA1BG\tUGGCACU\t9606\t3\t1\t2\t0\t1\t0\t0\t1\t0\thsa-miR-1\t-0.6\t-0.5\t0.9\tNULL\tNULL\tNULL\n"
    "ENST1.3\tA1BG\tGUAAACA\t9606\t2\t2\t0\t0\t0\t0\t0\t0\t0\thsa-miR-2\t-0.3\t-0.2\t0.4\tNULL\tNULL\tNULL\n"
    "ENST1.3\tA1BG\tUGGCACU\t10090\t3\t0\t0\t3\t1\t0\t0\t1\t0\tmmu-miR-1\t-0.6\t-0.6\t1.0\tNULL\tNULL\tNULL\n"   # mouse row
    "ENST2.2\tA1CF\tUGGCACU\t9606\t0\t0\t0\t0\t4\t1\t1\t2\t0\thsa-miR-1\t-0.1\t-0.05\t0.0\tNULL\tNULL\tNULL\n"  # nonconserved only
    "ENST6.1\tDUP\tUGGCACU\t9606\t7\t7\t0\t0\t0\t0\t0\t0\t0\thsa-miR-1\t-1.0\t-0.9\t0.99\tNULL\tNULL\tNULL\n"
)


def _zip(path, member, text):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(member, text)


@pytest.fixture
def source(tmp_path):
    root = tmp_path
    dest = root / "external" / "ts"
    dest.mkdir(parents=True)
    _zip(dest / "Gene_info.txt.zip", "Gene_info.txt", GENE_INFO)
    _zip(dest / "TSHuman_7_hg19_3UTRs.gff.zip", "TSHuman_7_hg19_3UTRs.gff", GFF)
    _zip(dest / "Summary_Counts.default_predictions.txt.zip",
         "Summary_Counts.default_predictions.txt", COUNTS)
    (dest / "PROVENANCE.json").write_text(json.dumps({"selected": [
        {"name": n, "size_bytes": (dest / n).stat().st_size, "checksum": "http-last-modified:x"}
        for n in ("Gene_info.txt.zip", "TSHuman_7_hg19_3UTRs.gff.zip",
                  "Summary_Counts.default_predictions.txt.zip")]}))
    spec = {"name": "targetscan", "kind": "edge", "layer": "posttx", "relation": "mirna_target",
            "dest": "external/ts", "derived": "derived/ts", "license": "Free-for-research-with-citation"}
    return MiRNATargetSource(spec, {}, root=root)


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
    assert c["summary_counts_human_rows"] == 4
    # the cache is read back without rebuilding
    again = source.utr_load_table()
    assert len(again) == 4


def test_edge_form_is_unbuilt_and_says_where_the_table_is(source):
    with pytest.raises(NotImplementedError, match="utr_load_table"):
        source.build()
