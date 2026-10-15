"""Contract tests for the wide PTR feature table (T120) on synthetic per-source tables.

Pinned: the join goes by Ensembl id first and by the source's symbol second; a gene a source
lacks is NaN, never 0; every column's scope follows the source's rules (universal, or the
measured line under one spelling); a column from an UNSTATED or NonCommercial source keeps
its flags in the dictionary; `genes_covered` counts the axis genes with a value; the wide
column names are prefixed once; and the LINEAGE entry records each input's sha256 and how
many genes each source matched by id, by symbol and not at all.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from sidechain.priors.ptr_table import (
    SourceSpec,
    build_ptr_table,
    column_scope,
    line_token,
    wide_name,
    write_ptr_table,
)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "vcc2026").mkdir()
    pd.DataFrame({"gene_name": ["A", "B", "C", "D"]}).to_csv(tmp_path / "vcc2026" / "gene_names.csv", index=False)
    bridge_dir = tmp_path / "runs" / "probes" / "t102_utr_load" / "bridge"
    bridge_dir.mkdir(parents=True)
    # D is not bridged; C's id is bridged but the half-life source only knows it by symbol
    (bridge_dir / "symbol_to_ensg.json").write_text(json.dumps({"A": "ENSG1.4", "B": "ENSG2", "C": "ENSG3"}))

    ts = tmp_path / "derived" / "ts"
    ts.mkdir(parents=True)
    pd.DataFrame({"gene_id": ["ENSG1", "ENSG2", "ENSG9"], "symbol": ["A", "Bold", "D"],
                  "transcript_id": ["T1", "T2", "T9"], "utr_len": [100, 200, 900],
                  "n_cons_sites": [1, 0, 3]}).to_parquet(ts / "t.parquet", index=False)
    (ts / "LINEAGE.json").write_text(json.dumps({"entries": {"ts/t": {
        "sha256": "abc", "built": "2026-09-28", "columns": {"utr_len": "from lineage"}}}}))
    hl = tmp_path / "derived" / "hl"
    hl.mkdir(parents=True)
    pd.DataFrame({"gene_id": ["ENSG1", "ENSG33"], "symbol": ["A", "C"], "halflife_pc1": [1.5, 2.5],
                  "halflife_hek293_mean": [0.1, np.nan], "halflife_hek293_n": [14, 0],
                  "halflife_rpe_mean": [0.3, 0.4]}).to_parquet(hl / "h.parquet", index=False)
    pa = tmp_path / "derived" / "pa"
    pa.mkdir(parents=True)
    pd.DataFrame({"gene_id": ["ENSG2"], "symbol": ["B"], "transcript_id": ["T2"], "pa_tx_source": ["mane_select"],
                  "pa_n_sites": [3], "pa_has_apa": [True], "pa_prox_usage_hela": [0.5]}).to_parquet(pa / "p.parquet", index=False)
    ext = tmp_path / "external"
    for name, flags, lic, route in (("ts", (False, False), "Free-for-research-with-citation", "download"),
                                    ("hl", (False, False), "CC-BY-4.0", "download"),
                                    ("pa", (True, True), "UNSTATED", "stream")):
        (ext / name).mkdir(parents=True)
        (ext / name / "PROVENANCE.json").write_text(json.dumps({
            "route": route, "record": {"license": lic},
            "license_flags": {"noncommercial": flags[0], "redistribution_encumbered": flags[1]}}))
    registry = tmp_path / "data_sources.yaml"
    registry.write_text(yaml.safe_dump({"sources": [
        {"name": "targetscan", "dest": "external/ts", "license": "Free-for-research-with-citation"},
        {"name": "halflife_consensus", "dest": "external/hl", "license": "CC-BY-4.0"},
        {"name": "polyasite_2", "dest": "external/pa", "license": "UNSTATED", "route": "stream"},
    ]}))
    return tmp_path, registry


SOURCES = (
    SourceSpec(key="ts", block="targetscan", derived="derived/ts", table="t.parquet", lineage_key="ts/t",
               drop=("transcript_id",), describe={"n_cons_sites": "conserved sites"}),
    SourceSpec(key="hl", block="halflife_consensus", derived="derived/hl", table="h.parquet", lineage_key="hl/h",
               scope=(("^halflife_pc1$", "universal"), (r"^halflife_([a-z0-9]+)_(?:mean|n)$", "line:{0}"))),
    SourceSpec(key="pa", block="polyasite_2", derived="derived/pa", table="p.parquet", lineage_key="pa/p",
               drop=("transcript_id", "pa_tx_source"),
               scope=((r"^pa_(?:prox_usage|window_tpm)_(hela|hek293|k562)$", "line:{0}"),)),
)


def test_scope_and_names():
    hl = SOURCES[1]
    assert column_scope(hl, "halflife_pc1") == "universal"
    assert column_scope(hl, "halflife_hek293_mean") == "line:hek293"
    assert column_scope(hl, "halflife_rpe_n") == "line:rpe1"
    assert column_scope(hl, "something_else") == "universal"
    assert line_token("293T") == "hek293t" and line_token("H1ESC") == "h1" and line_token("Jurkat") == "jurkat"
    assert wide_name(SOURCES[0], "utr_len") == "ts_utr_len"
    assert wide_name(SOURCES[2], "pa_n_sites") == "pa_n_sites"       # prefixed once


def test_join_scope_flags_and_coverage(root):
    tmp, registry = root
    wide, d, entry = build_ptr_table(tmp, sources=SOURCES, registry=registry)
    assert wide["symbol"].tolist() == ["A", "B", "C", "D"]
    assert wide["gene_id"].tolist()[:3] == ["ENSG1", "ENSG2", "ENSG3"] and wide["gene_id"].isna().tolist()[3]
    # by id: A and B; D matches TargetScan by SYMBOL (its id is not bridged); C matches nothing there
    assert wide["ts_utr_len"].tolist()[:2] == [100, 200] and wide["ts_utr_len"].iloc[3] == 900
    assert np.isnan(wide["ts_utr_len"].iloc[2])
    # C matches the half-life table by symbol although its bridged id is ENSG3 and the table says ENSG33
    assert wide["hl_halflife_pc1"].tolist()[0] == 1.5 and wide["hl_halflife_pc1"].iloc[2] == 2.5
    assert np.isnan(wide["hl_halflife_pc1"].iloc[1]) and np.isnan(wide["hl_halflife_hek293_mean"].iloc[2])
    # a boolean enters as 0/1 with NaN where the source has no row
    assert wide["pa_has_apa"].iloc[1] == 1.0 and np.isnan(wide["pa_has_apa"].iloc[0])
    assert "pa_tx_source" not in wide.columns and "ts_transcript_id" not in wide.columns
    d = d.set_index("column")
    assert d.loc["ts_utr_len", "description"] == "from lineage" and d.loc["ts_n_cons_sites", "description"] == "conserved sites"
    assert d.loc["hl_halflife_rpe_mean", "scope"] == "line:rpe1" and d.loc["hl_halflife_rpe_mean", "line"] == "rpe1"
    assert d.loc["hl_halflife_pc1", "universal"] and not d.loc["pa_prox_usage_hela", "universal"]
    assert d.loc["pa_n_sites", "licence"] == "UNSTATED" and d.loc["pa_n_sites", "noncommercial"] and d.loc["pa_n_sites", "redistribution_encumbered"]
    assert d.loc["hl_halflife_pc1", "licence"] == "CC-BY-4.0" and not d.loc["hl_halflife_pc1", "noncommercial"]
    assert d.loc["ts_utr_len", "genes_covered"] == 3 and d.loc["hl_halflife_hek293_mean", "genes_covered"] == 1
    assert entry["sources"]["ts"]["matched"] == {"id": 2, "symbol": 1, "none": 1}
    assert entry["sources"]["hl"]["matched"] == {"id": 1, "symbol": 1, "none": 2}
    assert entry["inputs"]["derived/ts/t.parquet"]["source_sha256"] == "abc"
    assert entry["inputs"]["derived/pa/p.parquet"]["route"] == "stream"
    assert entry["counts"] == {"genes": 4, "columns": 9, "columns_universal": 5, "columns_line": 4,
                               "lines": ["hek293", "hela", "rpe1"], "genes_with_any_value": 4,
                               "genes_with_any_universal": 4}


def test_write_lands_table_dictionary_and_lineage(root):
    tmp, registry = root
    out = write_ptr_table(tmp, sources=SOURCES, registry=registry)
    assert sorted(p.name for p in out.iterdir()) == ["LINEAGE.json", "ptr_columns.json", "ptr_columns.parquet", "ptr_features.parquet"]
    lineage = json.loads((out / "LINEAGE.json").read_text())
    e = lineage["entries"]["ptr-feature-table/ptr_features"]
    assert e["sha256"] and e["dictionary_sha256"] and e["axis"]["genes"] == 4 and e["axis"]["bridged_to_ensembl"] == 3
    assert len(json.loads((out / "ptr_columns.json").read_text())) == 9
    assert len(pd.read_parquet(out / "ptr_features.parquet")) == 4


def test_duplicate_wide_columns_are_refused(root):
    tmp, registry = root
    twice = (SOURCES[0], SourceSpec(key="ts", block="targetscan", derived="derived/ts", table="t.parquet",
                                    lineage_key="ts/t", drop=("transcript_id",)))
    with pytest.raises(ValueError, match="duplicate"):
        build_ptr_table(tmp, sources=twice, registry=registry)
