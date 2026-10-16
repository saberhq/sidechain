"""Contract tests for `CisBPRNASource` (T120, phase 2) on a synthetic CisBP-RNA archive and a
synthetic TargetScan 3'UTR file.

Pinned:

* a matrix file is parsed on its ``Pos A C G U`` header and renormalised; an empty one is
  skipped and counted, a motif the table lists with no file is counted apart;
* the archive's two DBID columns are told apart (the RBP's Ensembl id first);
* a planted motif copy is a hit at the relative-score threshold and a random window is not;
  overlapping copies count once per start position; a window never crosses a sequence
  boundary and a base outside ACGU fails it; the sequences are read ungapped, T as U,
  representative transcripts only, human only;
* an RBP's count is the union of its matrices' hits, an inferred RBP sharing a measured
  motif gets that motif's count, the direct and inferred totals are kept apart;
* the per-sequence shuffle keeps each sequence's composition and the null rate is written
  per matrix beside the real one;
* `build()` is a node_feature of the direct RBPs aligned to the gene space with a mask.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sidechain.priors.posttx_cisbp import (
    ARCHIVE_FILE,
    PFM_DIR,
    RBP_TABLE,
    CisBPRNASource,
    background_of,
    encode,
    information_content,
    log_odds,
    parse_pfm,
    relative_threshold,
    scan_hits,
    shuffle_within,
)

HEADER = ("RBP_ID\tFamily_ID\tRSource_ID\tMotif_ID\tMSource_ID\tDBID\tRBP_Name\tRBP_Species\tRBP_Status\t"
          "Family_Name\tRBDs\tRBD_Count\tCutoff\tDBID\tMotif_Type\tMSource_Identifier\tMSource_Type\t"
          "MSource_Author\tMSource_Year\tPMID\tMSource_Version\tRBPSource_Name\tRBPSource_URL\t"
          "RBPSource_Year\tRBPSource_Month\tRBPSource_Day")


def _row(rbp_id, motif, ensg, name, status, mtype="RNAcompete", src="RNA_Zoo"):
    return "\t".join([rbp_id, "F1", "TS1", motif, "MS1", ensg, name, "Homo_sapiens", status, "RRM", "RRM", "1",
                      "0.70", f"{motif}_src", mtype, src, mtype, "Ray", "2013", "23846655", "NULL", "Ensembl",
                      "http://www.ensembl.org/", "2018", "Dec", "8"])


RBP_ROWS = "\n".join([
    HEADER,
    _row("T1_2.00", "M00001_2.00", "ENSGR1", "RBP1", "D"),
    _row("T1_2.00", "M00003_2.00", "ENSGR1", "RBP1", "D", mtype="HTR-SELEX", src="Jolma2020_RBPs"),
    _row("T2_2.00", "M00001_2.00", "ENSGR2", "RBP2", "I", mtype="JPLE", src="RNA_Zoo_2"),
    _row("T3_2.00", "M00002_2.00", "ENSGR3", "RBP3", "D", mtype="RBPDB", src="RBPDB"),
    _row("T4_2.00", ".", "ENSGR4", "RBP4", "N", mtype=".", src="."),
    _row("T5_2.00", "M00009_2.00", "ENSGR5", "RBP5", "D"),          # listed, no matrix file
]) + "\n"


def _pfm(seq: str, p: float = 0.97) -> str:
    lines = ["Pos\tA\tC\tG\tU"]
    for i, b in enumerate(seq, 1):
        row = {k: (1 - p) / 3 for k in "ACGU"}
        row[b] = p
        lines.append("\t".join([str(i)] + [f"{row[k]:.6f}" for k in "ACGU"]))
    return "\n".join(lines) + "\n"


M1 = "UGUAAAUA"
M3 = "GGGGG"
GENE_INFO = (
    "Transcript ID\tGene ID\tGene symbol\tGene description\tSpecies ID\t3P-seq tags\tRepresentative transcript?\n"
    "ENST1.3\tENSG1.7\tG1\tdesc\t9606\t74\t1\n"
    "ENST2.1\tENSG2.1\tG2\tdesc\t9606\t2\t1\n"
    "ENST3.2\tENSG3.1\tG3\tdesc\t9606\t5\t1\n"
    "ENST4.1\tENSG4.1\tG4\tdesc\t9606\t5\t1\n"
    "ENST5.1\tENSG1.7\tG1\tdesc\t9606\t1\t0\n"          # not representative
    "ENSMUST1.1\tENSMUSG1.1\tg1\tdesc\t10090\t1\t1\n"    # mouse
)
BG = "ACGUCAGUAGCUGACU"
UTR_SEQ = (
    "Refseq ID\tGene ID\tGene Symbol\tSpecies ID\tUTR sequence\n"
    # two adjacent copies of M1 (written with gaps and T's) and one G run flanked by non-G
    f"ENST1.3\tENSG1.7\tG1\t9606\t{BG}TGTA-AATA{M1}{BG}C{M3}C{BG}\n"
    # ends with the first half of M1 ...
    f"ENST2.1\tENSG2.1\tG2\t9606\t{BG}UGUA\n"
    # ... and the next sequence starts with the second half: never a hit across the boundary
    f"ENST3.2\tENSG3.1\tG3\t9606\tAAUA{BG}\n"
    # a base outside ACGU inside the motif
    f"ENST4.1\tENSG4.1\tG4\t9606\t{BG}UGUANAUA{BG}\n"
    f"ENST5.1\tENSG1.7\tG1\t9606\t{M1}{M1}{M1}\n"                 # not representative: never read
    f"ENSMUST1.1\tENSMUSG1.1\tg1\t10090\t{M1}{M1}\n"             # mouse: never read
)


@pytest.fixture
def wired(tmp_path):
    root = tmp_path
    dest = root / "external" / "cisbp"
    dest.mkdir(parents=True)
    with zipfile.ZipFile(dest / ARCHIVE_FILE, "w") as z:
        z.writestr(RBP_TABLE, RBP_ROWS)
        z.writestr(f"{PFM_DIR}/M00001_2.00.txt", _pfm(M1))
        z.writestr(f"{PFM_DIR}/M00003_2.00.txt", _pfm(M3))
        z.writestr(f"{PFM_DIR}/M00002_2.00.txt", "")
    (dest / "PROVENANCE.json").write_text(json.dumps({
        "route": "download", "selected": [{"name": ARCHIVE_FILE, "size_bytes": 1, "checksum": None}],
        "license_flags": {"noncommercial": True, "redistribution_encumbered": True}}))
    ts = root / "external" / "ts"
    ts.mkdir(parents=True)
    with zipfile.ZipFile(ts / "Gene_info.txt.zip", "w") as z:
        z.writestr("Gene_info.txt", GENE_INFO)
    with zipfile.ZipFile(ts / "UTR_Sequences.txt.zip", "w") as z:
        z.writestr("UTR_Sequences.txt", UTR_SEQ)
    registry = root / "data_sources.yaml"
    registry.write_text(yaml.safe_dump({"sources": [
        {"name": "cisbp_rna_2", "kind": "node_feature", "layer": "posttx", "relation": "rbp_motif_sites",
         "host": "https", "dest": "external/cisbp", "derived": "derived/cisbp", "license": "UNSTATED",
         "files": [{"name": ARCHIVE_FILE}]},
        {"name": "targetscan", "kind": "edge", "layer": "posttx", "relation": "mirna_target",
         "dest": "external/ts", "derived": "derived/ts", "license": "Free-for-research-with-citation",
         "entity_src": "mirna_family"},
    ]}))
    from sidechain.priors.posttx_mirna import spec_from_registry
    return CisBPRNASource(spec_from_registry("cisbp_rna_2", registry), {}, root=root, registry=registry)


# ---------------------------------------------------------------- the helpers --


def test_parse_pfm_checks_the_header_and_renormalises():
    pfm = parse_pfm("Pos\tA\tC\tG\tU\n1\t2\t2\t2\t2\n2\t0\t0\t0\t1\n")
    assert pfm.shape == (2, 4) and np.allclose(pfm[0], 0.25) and pfm[1, 3] == 1.0
    assert abs(information_content(pfm) - 2.0) < 1e-9
    with pytest.raises(ValueError, match="empty"):
        parse_pfm("\n")
    with pytest.raises(ValueError, match="header"):
        parse_pfm("A\tC\tG\tU\n0.25\t0.25\t0.25\t0.25\n")


def test_planted_window_hits_and_a_random_one_does_not():
    pfm = parse_pfm(_pfm(M1))
    bg = np.full(4, 0.25)
    w = log_odds(pfm, bg)
    thr = relative_threshold(w)
    code, _, remaining = encode([M1, "ACGUACGU"])
    hits = scan_hits(code, remaining, w, thr)
    assert hits.tolist() == [0]
    # one substitution in the middle still passes 0.85 of the range; three do not
    code, _, remaining = encode(["UGUACAUA", "UGCCCAUA"])
    assert scan_hits(code, remaining, w, thr).tolist() == [0]


def test_a_window_never_crosses_a_sequence_boundary_and_n_fails_it():
    w = log_odds(parse_pfm(_pfm(M1)), np.full(4, 0.25))
    thr = relative_threshold(w)
    code, seq_index, remaining = encode(["CCUGUA", "AAUACC"])
    assert scan_hits(code, remaining, w, thr).size == 0
    code, _, remaining = encode(["UGUANAUA"])
    assert scan_hits(code, remaining, w, thr).size == 0
    code, _, remaining = encode(["UGUAAAUAUGUAAAUA"])          # two adjacent copies: two starts
    assert scan_hits(code, remaining, w, thr).tolist() == [0, 8]


def test_shuffle_keeps_each_sequence_composition():
    seqs = ["AAAACCGU", "GGGGUU"]
    out = shuffle_within(seqs, seed=1)
    assert [sorted(s) for s in out] == [sorted(s) for s in seqs]
    code, _, _ = encode(seqs)
    assert np.allclose(background_of(code), [4 / 14, 2 / 14, 5 / 14, 3 / 14])


# --------------------------------------------------------------- the tables --


def test_motif_table_keeps_usable_matrices_and_counts_the_rest(wired):
    motifs = wired.motif_table()
    assert sorted(zip(motifs["rbp"], motifs["motif_id"])) == [
        ("RBP1", "M00001_2.00"), ("RBP1", "M00003_2.00"), ("RBP2", "M00001_2.00")]
    assert motifs.set_index("rbp")["rbp_ensembl_id"].to_dict() == {"RBP1": "ENSGR1", "RBP2": "ENSGR2"}
    assert motifs.set_index("motif_id")["motif_source_id"].to_dict()["M00001_2.00"] == "M00001_2.00_src"
    assert motifs.set_index("motif_id")["pfm_len"].to_dict() == {"M00001_2.00": 8, "M00003_2.00": 5}
    pfms = wired.pfms()
    assert pfms["M00001_2.00"].shape == (8, 4) and np.allclose(pfms["M00001_2.00"].sum(axis=1), 1.0)
    lineage = json.loads((wired.derived / "LINEAGE.json").read_text())["entries"]["cisbp-rna-2.0/rbp_motifs"]
    assert lineage["counts"]["usable_matrices"] == 2 and lineage["counts"]["skipped_matrices"] == 1
    assert lineage["skipped"] == {"M00002_2.00": "empty file"}
    assert lineage["motifs_listed_without_a_file"] == ["M00009_2.00"]
    assert lineage["counts"]["rbp_status_in_archive"] == {"D": 3, "I": 1, "N": 1}
    assert len(lineage["inputs"][ARCHIVE_FILE]["content_sha256"]) == 64
    assert lineage["license_flags"] == {"noncommercial": True, "redistribution_encumbered": True}


def test_gene_table_counts_the_union_per_rbp_on_representative_human_utrs(wired):
    genes = wired.gene_table().set_index("transcript_id")
    assert list(genes.index) == ["ENST1", "ENST2", "ENST3", "ENST4"]      # no mouse, no non-representative
    assert genes.loc["ENST1", "rb_RBP1"] == 3 and genes.loc["ENST1", "rb_RBP2"] == 2
    assert genes.loc["ENST2", "rb_RBP1"] == 0 and genes.loc["ENST3", "rb_RBP1"] == 0
    assert genes.loc["ENST4", "rb_RBP1"] == 0
    assert genes.loc["ENST1", "rb_sites_direct"] == 3 and genes.loc["ENST1", "rb_sites_inferred"] == 2
    assert genes.loc["ENST1", "rb_n_rbps_direct"] == 1 and genes.loc["ENST2", "rb_n_rbps_direct"] == 0
    assert genes.loc["ENST1", "seq_len"] == len(BG) * 3 + 16 + 7 and genes.loc["ENST1", "gene_id"] == "ENSG1"
    assert "rb_RBP3" not in genes.columns and "rb_RBP4" not in genes.columns
    assert "rb_sites_direct_null" in genes.columns and (genes["rb_sites_direct_null"] >= 0).all()
    # a stricter twin lands beside the default and leaves it alone
    twin = wired.gene_table(rebuild=True, rel_score_min=0.95)
    assert (wired.derived / "rbp_site_counts_r095.parquet").exists() and (wired.derived / "rbp_motifs_r095.parquet").exists()
    assert twin.set_index("transcript_id").loc["ENST1", "rb_RBP1"] <= 3
    assert "cisbp-rna-2.0/rbp_site_counts_r095" in json.loads((wired.derived / "LINEAGE.json").read_text())["entries"]
    motifs = wired.motif_table()
    assert {"threshold", "hits_per_kb", "null_hits_per_kb"} <= set(motifs.columns)
    assert (motifs["hits_per_kb"] >= 0).all() and np.isfinite(motifs["null_hits_per_kb"]).all()
    lineage = json.loads((wired.derived / "LINEAGE.json").read_text())["entries"]["cisbp-rna-2.0/rbp_site_counts"]
    assert lineage["counts"]["transcripts"] == 4 and lineage["counts"]["rbp_columns"] == 2
    assert lineage["scan"]["rel_score_min"] == 0.85 and len(lineage["scan"]["background_acgu"]) == 4
    assert "rb_RBP1" in lineage["columns"] and "(D; Ensembl ENSGR1; 2 matrices" in lineage["columns"]["rb_RBP1"]
    assert "(I; Ensembl ENSGR2; 1 matrices" in lineage["columns"]["rb_RBP2"]
    assert "UTR_Sequences.txt.zip" in lineage["inputs"]["targetscan"]["files"]


def test_build_is_a_direct_rbp_node_feature_with_a_mask(wired):
    wired.gene_index = {"ENSG1": 0, "ENSG2": 1, "ENSGX": 2}
    art = wired.build_checked()
    assert art.kind == "node_feature" and art.meta["feature_names"] == ["rb_RBP1"]
    assert art.features.shape == (3, 1) and art.features[:, 0].tolist() == [3.0, 0.0, 0.0]
    assert art.meta["has_value"][:, 0].tolist() == [True, True, False]
    assert art.meta["genes_with_any_value"] == 2 and art.meta["zero_is_a_real_value"] is True
