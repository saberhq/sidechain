"""Contract tests for the RNAplfold accessibility family (T120, phase 2) on synthetic 3'UTRs.

Pinned: a stem reads near 0 and a loop near 1 per nucleotide; a site's span probability is the u-nt
probability ending at the site's end (1-based inclusive), so a site in a loop reads open and one in a
stem reads closed; the flank excludes the site; a transcript with no site gets NaN site columns and a
zero count; a consensus motif's accessibility is the mean over its match positions and NaN with no match;
the table is written with its LINEAGE entry, parameters and the TargetScan licence.
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd
import pytest
import yaml

from sidechain.priors.posttx_access import (
    COLUMNS,
    accessibility_row,
    build_accessibility,
    unpaired_probabilities,
    write_accessibility,
)

HAIRPIN = "GGGGGGGGAAAAAAAACCCCCCCC"            # stem 1-8 and 17-24, loop 9-16
TAIL = "AUAUAUAUAUAUAUAUAUAUAUAUAUAUAU"           # alternating, little structure of its own
GENE_INFO = (
    "Transcript ID\tGene ID\tGene symbol\tGene description\tSpecies ID\t3P-seq tags\tRepresentative transcript?\n"
    "ENST1.1\tENSG1.1\tG1\tdesc\t9606\t1\t1\n"
    "ENST2.1\tENSG2.1\tG2\tdesc\t9606\t1\t1\n"
)


def test_stem_closed_loop_open_and_the_window_probability():
    up = unpaired_probabilities(HAIRPIN + TAIL)
    assert up.shape == (len(HAIRPIN + TAIL) + 1, 9)
    assert up[3, 1] < 0.1 and up[12, 1] > 0.9 and up[20, 1] < 0.1          # stem, loop, stem
    assert up[16, 8] > 0.8                                                  # the whole 8-nt loop ending at 16
    assert up[8, 8] < 0.1                                                   # the 8-nt stem ending at 8


def test_row_reads_sites_flanks_and_motifs():
    seq = HAIRPIN + TAIL + "AUUUAUUUA" + TAIL                                 # an ARE at 55..63
    loop_site, stem_site = (9, 16), (1, 8)
    row = accessibility_row(seq, [loop_site, stem_site])
    assert row["seq_len"] == len(seq) and row["acc_mirna_n_sites"] == 2
    assert row["acc_mirna_site_max"] > 0.8 and 0.3 < row["acc_mirna_site_mean"] < 0.6
    assert row["acc_mirna_site_excess"] == pytest.approx(row["acc_mirna_site_mean"] - row["acc_mean_u8"])
    # the loop site's flank is the two stems (closed) plus the tail (open); the mean sits between
    only_loop = accessibility_row(seq, [loop_site])
    assert 0.2 < only_loop["acc_mirna_flank_mean"] < 0.9
    assert np.isfinite(row["acc_motif_are_auuua_mean"]) and row["acc_motif_are_auuua_mean"] > 0.5
    assert np.isnan(row["acc_motif_pum_mean"]) and np.isfinite(row["acc_motif_rbp_mean"])
    empty = accessibility_row(TAIL, [])
    assert empty["acc_mirna_n_sites"] == 0 and np.isnan(empty["acc_mirna_site_mean"]) and np.isnan(empty["acc_mirna_flank_mean"])
    # a site past the end is ignored, not an error
    assert accessibility_row(TAIL, [(100, 107)])["acc_mirna_n_sites"] == 0


@pytest.fixture
def root(tmp_path):
    ts = tmp_path / "external" / "ts"
    ts.mkdir(parents=True)
    with zipfile.ZipFile(ts / "Gene_info.txt.zip", "w") as z:
        z.writestr("Gene_info.txt", GENE_INFO)
    (ts / "PROVENANCE.json").write_text(json.dumps({"selected": [], "license_flags": {"noncommercial": False, "redistribution_encumbered": False}}))
    registry = tmp_path / "data_sources.yaml"
    registry.write_text(yaml.safe_dump({"sources": [
        {"name": "targetscan", "kind": "edge", "layer": "posttx", "relation": "mirna_target", "dest": "external/ts",
         "derived": "derived/ts", "license": "Free-for-research-with-citation", "entity_src": "mirna_family"}]}))
    return tmp_path, registry


def test_build_and_write_land_the_table_and_lineage(root):
    tmp, registry = root
    seqs = pd.DataFrame({"transcript_id": ["ENST2", "ENST1"], "gene_id": ["ENSG2", "ENSG1"], "symbol": ["G2", "G1"],
                         "seq": [TAIL, HAIRPIN + TAIL]})
    sites = {"ENST1": [(9, 16)]}
    table, entry = build_accessibility(tmp, registry=registry, sequences=seqs, sites=sites)
    assert list(table.columns) == list(COLUMNS) and table["transcript_id"].tolist() == ["ENST1", "ENST2"]
    assert table.loc[0, "acc_mirna_n_sites"] == 1 and table.loc[1, "acc_mirna_n_sites"] == 0
    assert entry["counts"]["transcripts"] == 2 and entry["counts"]["conserved_sites_on_representatives"] == 1
    assert entry["params"]["window"] == 80 and entry["licence"] == "Free-for-research-with-citation"
    out = write_accessibility(tmp, registry=registry, sequences=seqs, sites=sites)
    lineage = json.loads((out / "LINEAGE.json").read_text())["entries"]["rnaplfold-access/utr_accessibility"]
    assert lineage["sha256"] and lineage["license_flags"] == {"noncommercial": False, "redistribution_encumbered": False}
    assert len(pd.read_parquet(out / "utr_accessibility.parquet")) == 2
