"""Contract tests for `MiRNAAbundanceSource` (T120, phase 3) on synthetic copies of the three sources.

Pinned:

* the name-to-family rule: an exact miRBase name maps; a name without an arm maps when one arm exists
  or when exactly one of two arms belongs to a conserved family, and is left unmatched (counted) when
  both or neither are; a combined probe maps only when its parts agree; a GENCODE miRNA gene name is
  translated to the mature stem (MIRLET7A1 -> hsa-let-7a, MIR9-1 -> hsa-miR-9);
* the CCLE panel: our five lines are read by their panel column, a family's abundance is the sum of its
  mapped probes, an unmapped probe adds nothing;
* ENCODE: each file's counts become CPM over its miRNA genes, a line's replicate files are averaged, the
  line is read off the manifest note;
* the atlas: only cell-line samples of the named labels are kept, the per-line value is the median over
  samples, the label is translated to our spelling;
* the load: sites x log2(1 + abundance) summed over a gene's families, the expressed-site count uses the
  line's median family abundance as the floor, the top family is the max, a family with no abundance
  row adds nothing, the cross-line mean and spread are written;
* every table leaves a LINEAGE entry, `build()` aligns the universal mean behind a mask, and the view
  scopes each per-line column to its line.
"""
from __future__ import annotations

import gzip
import io
import json
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

from sidechain.priors import ptr_table
from sidechain.priors.posttx_mirna_abundance import (
    ATLAS_BLOCK,
    BLOCKS,
    CCLE_BLOCK,
    ENCODE_BLOCK,
    MiRNAAbundanceSource,
    family_of,
    gene_loads,
    map_names,
    mature_stem,
    read_family_table,
)

FAMILY_ROWS = [
    # miR family, Seed+m8, Species ID, MiRBase ID, Mature sequence, Family Conservation?, MiRBase Accession
    ("let-7-5p/98-5p", "GAGGUAG", "9606", "hsa-let-7a-5p", "UGAGG", "2", "MIMAT1"),
    ("let-7-3p", "UAUACAA", "9606", "hsa-let-7a-3p", "CUAUA", "-1", "MIMAT2"),
    ("miR-17-5p/20-5p", "AAAGUGC", "9606", "hsa-miR-17-5p", "CAAAG", "2", "MIMAT3"),
    ("miR-17-3p", "CUGCAGU", "9606", "hsa-miR-17-3p", "ACUGC", "-1", "MIMAT4"),
    ("miR-9-5p", "CUUUGGU", "9606", "hsa-miR-9-5p", "UCUUU", "2", "MIMAT5"),
    ("miR-9-3p", "UAAAGCU", "9606", "hsa-miR-9-3p", "AUAAA", "1", "MIMAT6"),          # both arms conserved -> ambiguous
    ("miR-21-5p", "AGCUUAU", "9606", "hsa-miR-21-5p", "UAGCU", "2", "MIMAT7"),       # one arm only
    ("miR-17-5p/20-5p", "AAAGUGC", "9606", "hsa-miR-20a-5p", "UAAAG", "2", "MIMAT8"),   # same family as miR-17
    ("miR-99-5p", "ACCCGUA", "10090", "mmu-miR-99a-5p", "AACCC", "2", "MIMAT9"),     # mouse, ignored
]


def _family_zip(path: Path) -> None:
    cols = ["miR family", "Seed+m8", "Species ID", "MiRBase ID", "Mature sequence", "Family Conservation?", "MiRBase Accession"]
    text = "\t".join(cols) + "\n" + "\n".join("\t".join(r) for r in FAMILY_ROWS) + "\n"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("miR_Family_Info.txt", text)


def _provenance(dest: Path, names: list[str], licence: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "PROVENANCE.json").write_text(json.dumps({
        "route": "download", "record": {"license": licence},
        "license_flags": {"noncommercial": False, "redistribution_encumbered": False},
        "selected": [{"name": n, "size_bytes": (dest / n).stat().st_size, "checksum": "x"} for n in names]}))


@pytest.fixture
def root(tmp_path):
    (tmp_path / "external/targetscan-vert_80").mkdir(parents=True)
    _family_zip(tmp_path / "external/targetscan-vert_80/miR_Family_Info.txt.zip")
    (tmp_path / "derived/targetscan-vert_80").mkdir(parents=True)
    # gene A: 2 let-7 sites + 1 miR-17 site; gene B: 1 miR-21 site; gene C: 3 miR-9 sites (family unmeasured everywhere)
    pd.DataFrame({"gene_id": ["ENSG1.2", "ENSG1.2", "ENSG2", "ENSG3"], "symbol": ["A", "A", "B", "C"],
                  "family": ["let-7-5p/98-5p", "miR-17-5p/20-5p", "miR-21-5p", "miR-9-5p"],
                  "n_cons_sites": [2, 1, 1, 3]}).to_parquet(tmp_path / "derived/targetscan-vert_80/edges_default.parquet", index=False)
    # GENCODE: miRNA genes
    (tmp_path / "external/gencode-v47").mkdir(parents=True)
    with gzip.open(tmp_path / "external/gencode-v47/gencode.v47.basic.annotation.gtf.gz", "wt") as fh:
        fh.write("##description: test\n")
        for gid, name, gtype in (("ENSGM1.1", "MIRLET7A1", "miRNA"), ("ENSGM2.1", "MIR17", "miRNA"), ("ENSGM3.1", "MIR21", "miRNA"),
                                 ("ENSGM4.1", "MIR9-1", "miRNA"), ("ENSGX.1", "TSPAN6", "protein_coding"), ("ENSGM5.1", "MIR548AC", "miRNA")):
            fh.write(f'chr1\tHAVANA\tgene\t1\t100\t.\t+\t.\tgene_id "{gid}"; gene_type "{gtype}"; gene_name "{name}";\n')

    # CCLE gct: probes let-7a (no arm; 5p conserved, 3p not -> conserved arm), miR-17 (same), miR-9 (both conserved -> ambiguous),
    # miR-21 (one arm), a combined probe that agrees, a retired name
    ccle = tmp_path / "external/ca"
    ccle.mkdir(parents=True)
    lines = ["K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE", "HCT116_LARGE_INTESTINE", "HEPG2_LIVER",
             "JURKAT_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE", "CAL33_UPPER_AERODIGESTIVE_TRACT", "OTHER_LUNG"]
    probes = [("nmiR1", "hsa-let-7a", [100, 200, 300, 400, 500, 1]), ("nmiR2", "hsa-miR-17", [10, 20, 30, 40, 50, 1]),
              ("nmiR3", "hsa-miR-9", [5, 5, 5, 5, 5, 1]), ("nmiR4", "hsa-miR-21", [1000, 1000, 1000, 1000, 1000, 1]),
              ("nmiR5", "hsa-miR-17+hsa-miR-20a", [7, 7, 7, 7, 7, 1]), ("nmiR6", "hsa-miR-1201", [3, 3, 3, 3, 3, 1])]
    with (ccle / "CCLE_miRNA_20181103.gct").open("w") as fh:
        fh.write("#1.2\n6\t6\n")
        fh.write("\t".join(["Name", "Description", *lines]) + "\n")
        for pid, name, vals in probes:
            fh.write("\t".join([pid, name, *map(str, vals)]) + "\n")
    _provenance(ccle, ["CCLE_miRNA_20181103.gct"], "Publisher-terms-noncommercial")

    # ENCODE: two K562 files (replicates), one H1 file; STAR shape
    enc = tmp_path / "external/em"
    enc.mkdir(parents=True)
    manifest = []
    def _star(name, counts):
        with (enc / name).open("w") as fh:
            fh.write("N_unmapped\t1\t1\t1\nN_multimapping\t1\t1\t1\nN_noFeature\t1\t1\t1\nN_ambiguous\t0\t0\t0\n")
            for gid, c in counts.items():
                fh.write(f"{gid}\t{c}\t{c}\t0\n")
    _star("F1.tsv", {"ENSGM1.1": 500, "ENSGM2.1": 300, "ENSGM3.1": 200, "ENSGM4.1": 0, "ENSGM5.1": 0})
    _star("F2.tsv", {"ENSGM1.1": 1000, "ENSGM2.1": 600, "ENSGM3.1": 400, "ENSGM4.1": 0, "ENSGM5.1": 0})
    _star("F3.tsv", {"ENSGM1.1": 100, "ENSGM2.1": 100, "ENSGM3.1": 800, "ENSGM4.1": 0, "ENSGM5.1": 0})
    for n, note in (("F1.tsv", "microRNA-seq ENCSR1 K562 rep [1] ENCODE4 GRCh38 V29"), ("F2.tsv", "microRNA-seq ENCSR1 K562 rep [2] ENCODE4 GRCh38 V29"),
                    ("F3.tsv", "microRNA-seq ENCSR2 H1 rep [1] ENCODE4 GRCh38 V29")):
        manifest.append({"name": n, "url": f"https://x/{n}", "bytes": (enc / n).stat().st_size, "md5": "0" * 32, "note": note})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    _provenance(enc, ["F1.tsv", "F2.tsv", "F3.tsv"], "ENCODE-unrestricted")

    # atlas h5ad: HeLa (3 cell-line samples, 1 EV), K562 (2), a tissue
    atl = tmp_path / "external/ma"
    atl.mkdir(parents=True)
    var = ["hsa-let-7a-5p", "hsa-miR-17-5p", "hsa-miR-21-5p", "hsa-miR-9-5p", "hsa-miR-9-3p", "hsa-miR-unknown-5p"]
    X = np.array([[10, 1, 100, 0, 0, 5], [30, 3, 100, 0, 0, 5], [20, 2, 100, 0, 0, 5],       # HeLa cell_line x3 -> medians 20, 2, 100
                  [999, 999, 999, 999, 999, 999],                                          # HeLa EV, dropped
                  [1, 50, 10, 0, 0, 0], [3, 70, 10, 0, 0, 0],                               # K562 x2 -> medians 2, 60, 10
                  [5, 5, 5, 5, 5, 5]], dtype=float)                                        # a tissue sample
    obs = pd.DataFrame({"Biotype": ["cell_line"] * 3 + ["EV"] + ["cell_line"] * 2 + ["tissue"],
                        "Tissue": ["HeLa"] * 4 + ["K562"] * 2 + ["liver"], "Organ_system": ["x"] * 7,
                        "Project": ["SRA"] * 7, "Sex": ["f"] * 7, "Species": ["human"] * 7},
                       index=[f"S{i}" for i in range(7)])
    ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var)).write_h5ad(atl / "all_human_miRNA_rpmm_harmonized_meta_data.h5ad")
    _provenance(atl, ["all_human_miRNA_rpmm_harmonized_meta_data.h5ad"], "UNSTATED")

    registry = tmp_path / "data_sources.yaml"
    registry.write_text(yaml.safe_dump({"sources": [
        {"name": CCLE_BLOCK, "kind": "node_feature", "layer": "posttx", "relation": "mirna_abundance_line",
         "dest": "external/ca", "derived": "derived/ca", "license": "Publisher-terms-noncommercial", "files": [{"name": "CCLE_miRNA_20181103.gct"}]},
        {"name": ENCODE_BLOCK, "kind": "node_feature", "layer": "posttx", "relation": "mirna_abundance_line",
         "dest": "external/em", "derived": "derived/em", "license": "ENCODE-unrestricted", "files_manifest": str(tmp_path / "manifest.json")},
        {"name": ATLAS_BLOCK, "kind": "node_feature", "layer": "posttx", "relation": "mirna_abundance_line",
         "dest": "external/ma", "derived": "derived/ma", "license": "UNSTATED", "files": [{"name": "all_human_miRNA_rpmm_harmonized_meta_data.h5ad"}]},
    ]}))
    return tmp_path, registry


def _src(root, block, gene_index=None):
    tmp, registry = root
    spec = next(s for s in yaml.safe_load(registry.read_text())["sources"] if s["name"] == block)
    return MiRNAAbundanceSource(spec, gene_index or {}, root=tmp, registry=registry)


def _lineage(src, key):
    return json.loads((src.derived / "LINEAGE.json").read_text())["entries"][key]


# ----------------------------------------------------------------- names --

def test_family_rule_exact_one_arm_conserved_arm_ambiguous_combined(root):
    tmp, _ = root
    fam = read_family_table(tmp)
    assert len(fam) == 8                                        # the mouse row is gone
    assert family_of("hsa-let-7a-5p", fam) == ("let-7-5p/98-5p", "exact")
    assert family_of("hsa-let-7a", fam) == ("let-7-5p/98-5p", "conserved_arm")
    assert family_of("hsa-miR-21", fam) == ("miR-21-5p", "one_arm")
    assert family_of("hsa-miR-9", fam) == (None, "ambiguous_arms")
    assert family_of("hsa-miR-17+hsa-miR-20a", fam) == ("miR-17-5p/20-5p", "combined")
    assert family_of("hsa-miR-17+hsa-miR-21", fam) == (None, "combined_disagree")
    assert family_of("hsa-miR-1201", fam) == (None, "none")
    fams, how = map_names(pd.Series(["hsa-let-7a", "hsa-miR-9", "hsa-miR-1201"]), fam)
    assert fams.tolist() == ["let-7-5p/98-5p", None, None] and how == {"conserved_arm": 1, "ambiguous_arms": 1, "none": 1}


def test_mature_stem_from_gencode_gene_names():
    assert mature_stem("MIRLET7A1") == "hsa-let-7a"
    assert mature_stem("MIR17") == "hsa-miR-17"
    assert mature_stem("MIR9-1") == "hsa-miR-9"
    assert mature_stem("MIR103A1") == "hsa-miR-103a"
    assert mature_stem("MIR548AC") is None and mature_stem("TSPAN6") is None


# ----------------------------------------------------------------- sources --

def test_ccle_five_lines_family_sums_and_unmapped_probes(root):
    src = _src(root, CCLE_BLOCK)
    t = src.abundance_table().set_index(["line", "family"])["abundance"]
    assert sorted({l for l, _ in t.index}) == ["cal33", "hct116", "hepg2", "jurkat", "k562"]
    assert t[("k562", "let-7-5p/98-5p")] == 100 and t[("cal33", "let-7-5p/98-5p")] == 500
    assert t[("k562", "miR-17-5p/20-5p")] == 10 + 7                # the combined probe joins the family
    assert t[("k562", "miR-21-5p")] == 1000
    assert ("k562", "miR-9-5p") not in t.index                      # ambiguous arms: unmatched
    e = _lineage(src, "ca/mirna_family_abundance")
    assert e["counts"]["name_mapping"]["ambiguous_arms"] == 5 and e["counts"]["name_mapping"]["none"] == 5
    assert e["counts"]["panel_lines"] == 6 and e["licence"] == "Publisher-terms-noncommercial"


def test_encode_cpm_per_file_replicates_averaged_line_from_note(root):
    src = _src(root, ENCODE_BLOCK)
    t = src.abundance_table().set_index(["line", "family"])["abundance"]
    # F1: let-7 500/1000 reads -> 5e5 CPM; F2: 1000/2000 -> 5e5; mean 5e5. miR-17: 3e5; miR-21: 2e5
    assert t[("k562", "let-7-5p/98-5p")] == pytest.approx(5e5)
    assert t[("k562", "miR-17-5p/20-5p")] == pytest.approx(3e5)
    assert t[("h1", "miR-21-5p")] == pytest.approx(8e5)
    assert ("k562", "miR-9-5p") not in t.index
    e = _lineage(src, "em/mirna_family_abundance")
    assert e["counts"]["replicate_spearman_log_cpm_median"]["k562"] == pytest.approx(1.0)
    assert e["counts"]["gene_names_without_mature_stem"] == 1       # MIR548AC


def test_atlas_keeps_cell_line_samples_of_named_labels_and_takes_the_median(root):
    src = _src(root, ATLAS_BLOCK)
    t = src.abundance_table().set_index(["line", "family"])["abundance"]
    assert sorted({l for l, _ in t.index}) == ["hela", "k562"]
    assert t[("hela", "let-7-5p/98-5p")] == 20 and t[("hela", "miR-17-5p/20-5p")] == 2 and t[("hela", "miR-21-5p")] == 100
    assert t[("k562", "miR-17-5p/20-5p")] == 60
    e = _lineage(src, "ma/mirna_family_abundance")
    assert e["counts"]["samples_per_line"] == {"hela": 3, "k562": 2}
    assert set(e["counts"]["lines_without_samples"]) == {"hek293", "hct116", "hepg2", "jurkat"}


# ------------------------------------------------------------------- loads --

def test_gene_loads_by_hand():
    abund = pd.DataFrame({"line": ["x", "x", "y", "y"], "family": ["let-7-5p/98-5p", "miR-17-5p/20-5p"] * 2,
                          "abundance": [3.0, 1.0, 1.0, 7.0]})
    edges = pd.DataFrame({"gene_id": ["ENSG1", "ENSG1", "ENSG2", "ENSG3"], "symbol": ["A", "A", "B", "C"],
                          "family": ["let-7-5p/98-5p", "miR-17-5p/20-5p", "miR-21-5p", "miR-9-5p"], "n_cons_sites": [2, 1, 1, 3]})
    t, columns, counts = gene_loads(abund, edges)
    t = t.set_index("gene_id")
    assert t.loc["ENSG1", "mirna_load_log_x"] == pytest.approx(2 * np.log2(4) + 1 * np.log2(2))
    assert t.loc["ENSG1", "mirna_load_log_y"] == pytest.approx(2 * np.log2(2) + 1 * np.log2(8))
    # floor in x is the median of {3, 1} = 2: let-7 expressed (2 sites), miR-17 not
    assert t.loc["ENSG1", "mirna_sites_expressed_x"] == 2 and t.loc["ENSG1", "mirna_sites_expressed_y"] == 1
    assert t.loc["ENSG1", "mirna_top_family_log_x"] == pytest.approx(np.log2(4))
    assert np.isnan(t.loc["ENSG2", "mirna_load_log_x"]) and np.isnan(t.loc["ENSG3", "mirna_load_log_y"])   # unmeasured families
    assert t.loc["ENSG1", "mirna_load_log_mean"] == pytest.approx((t.loc["ENSG1", "mirna_load_log_x"] + t.loc["ENSG1", "mirna_load_log_y"]) / 2)
    assert set(counts["per_line"]) == {"x", "y"} and "x~y" in counts["line_family_spearman"]
    assert "mirna_load_log_x" in columns and "mirna_load_log_sd" in columns


def test_gene_table_lineage_and_build(root):
    src = _src(root, CCLE_BLOCK, gene_index={"ENSG1": 0, "ENSG2": 1, "ENSG3": 2, "ENSG9": 3})
    t = src.gene_table().set_index("gene_id")
    assert list(t.index) == ["ENSG1", "ENSG2", "ENSG3"]          # versions stripped, every gene with a site
    assert t.loc["ENSG1", "mirna_load_log_k562"] == pytest.approx(2 * np.log2(101) + np.log2(18))
    assert np.isnan(t.loc["ENSG3", "mirna_load_log_k562"])        # miR-9 unmatched in the panel
    e = _lineage(src, "ca/mirna_load")
    assert e["counts"]["genes"] == 3 and "mirna_load_log_jurkat" in e["columns"]
    art = src.build()
    assert art.kind == "node_feature" and art.features.shape == (4, 1)
    assert art.meta["has_value"][:, 0].tolist() == [True, True, False, False]


def test_loader_refuses_a_block_it_does_not_serve(root):
    tmp, registry = root
    with pytest.raises(ValueError, match="not a block this loader serves"):
        MiRNAAbundanceSource({"name": "targetscan", "kind": "node_feature", "dest": "x", "derived": "y"}, {}, root=tmp, registry=registry)


# ---------------------------------------------------------------- the view --

def test_the_view_scopes_per_line_loads_and_keeps_the_mean_universal():
    by_key = {s.key: s for s in ptr_table.SOURCES}
    assert {by_key[k].block for k in ("ca", "em", "ma")} == set(BLOCKS)
    assert ptr_table.column_scope(by_key["ca"], "mirna_load_log_cal33") == "line:cal33"
    assert ptr_table.column_scope(by_key["em"], "mirna_sites_expressed_h1") == "line:h1"
    assert ptr_table.column_scope(by_key["ma"], "mirna_top_family_log_hela") == "line:hela"
    assert ptr_table.column_scope(by_key["ma"], "mirna_load_log_hek293") == "line:hek293"
    for k in ("ca", "em", "ma"):
        assert ptr_table.column_scope(by_key[k], "mirna_load_log_mean") == "universal"
        assert ptr_table.column_scope(by_key[k], "mirna_load_log_sd") == "universal"
    assert len({s.key for s in ptr_table.SOURCES}) == len(ptr_table.SOURCES)
