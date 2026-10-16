"""Contract tests for the per-context expressed-factor family (T120; final-phase: controls) on two
synthetic control h5ads and a synthetic CisBP-RNA pair of tables.

Pinned: a factor is expressed at the pseudobulk CPM floor and the CPM and detected share are
written for re-thresholding; a factor off the axis is unknown and in neither sum; a gene's direct-RBP
sites split into expressed and silent per context, the expressed RBPs with a site are counted, the
expressed share is NaN where a gene has no site; the miRNA machinery is in the factor table and never
in the sums; the DepMap check is skipped, and says so, where the files are absent; the context letter
comes from the file name and is kept upper-case in the view's scope.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sidechain.priors.ctx_factors import (
    build_control_factors,
    context_letter,
    factor_catalogue,
    write_control_factors,
)
from sidechain.priors.ptr_table import SourceSpec, column_scope

AXIS = ["RBP1", "RBP2", "AGO1", "G1", "G2", "G3"]


def _h5ad(path, rows):
    import anndata as ad
    X = sp.csr_matrix(np.asarray(rows, dtype=np.float32))
    a = ad.AnnData(X=X, obs=pd.DataFrame({"context": [path.stem[-1]] * X.shape[0]},
                                         index=[f"c{i}" for i in range(X.shape[0])]),
                   var=pd.DataFrame(index=AXIS))
    a.write_h5ad(path)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "vcc2026").mkdir()
    # A: RBP1 and AGO1 on, RBP2 off; B: RBP2 on, RBP1 and AGO1 off
    _h5ad(tmp_path / "vcc2026" / "context_A.h5ad", [[5, 0, 2, 10, 0, 1], [3, 0, 1, 8, 0, 0], [4, 0, 0, 9, 0, 2]])
    _h5ad(tmp_path / "vcc2026" / "context_B.h5ad", [[0, 6, 0, 10, 0, 1], [0, 2, 0, 8, 0, 0]])
    cis = tmp_path / "derived" / "cisbp-rna-2.0"
    cis.mkdir(parents=True)
    pd.DataFrame({"rbp": ["RBP1", "RBP2", "RBP3", "RBPX"], "status": ["D", "D", "I", "D"],
                  "rbp_ensembl_id": ["ENSGR1", "ENSGR2", "ENSGR3", "ENSGRX"], "motif_id": ["M1", "M2", "M1", "M4"]}
                 ).to_parquet(cis / "rbp_motifs.parquet", index=False)
    pd.DataFrame({"symbol": ["G1", "G2", "G3"], "gene_id": ["ENSG1", "ENSG2", "ENSG3"],
                  "transcript_id": ["T1", "T2", "T3"], "seq_len": [100, 50, 80],
                  "rb_RBP1": [2, 0, 0], "rb_RBP2": [1, 0, 3], "rb_RBP3": [7, 7, 7], "rb_RBPX": [5, 0, 0],
                  "rb_sites_direct": [8, 0, 3]}).to_parquet(cis / "rbp_site_counts.parquet", index=False)
    return tmp_path


def test_context_letter_and_scope():
    from pathlib import Path
    assert context_letter(Path("x/context_D.h5ad")) == "D"
    with pytest.raises(ValueError):
        context_letter(Path("x/controls.h5ad"))
    spec = SourceSpec(key="cx", block="cisbp_rna_2", derived="d", table="t", lineage_key="l",
                      scope=((r"^cx_.*_([A-F])$", "ctx:{0}"),))
    assert column_scope(spec, "cx_sites_expressed_A") == "ctx:A"        # the letter stays as written
    assert column_scope(spec, "cx_frac_sites_expressed_F") == "ctx:F"


def test_catalogue_merges_the_lists():
    motifs = pd.DataFrame({"rbp": ["RBP1", "RBP3"], "status": ["D", "I"], "rbp_ensembl_id": ["E1", "E3"]})
    cat = factor_catalogue(motifs, ["RBP1", "ECL"]).set_index("symbol")
    assert cat.loc["RBP1", "cisbp"] == "direct" and cat.loc["RBP1", "eclip"] and cat.loc["RBP1", "ensembl_id"] == "E1"
    assert cat.loc["RBP3", "cisbp"] == "inferred" and cat.loc["ECL", "cisbp"] == "" and cat.loc["AGO1", "mirna_machinery"]
    assert len(cat) == 2 + 1 + 12


def test_expressed_sets_and_the_split_sums(root):
    factors, genes, entry = build_control_factors(root, depmap=False, eclip_manifest=root / "none.json")
    f = factors.set_index(["context", "symbol"])
    assert f.loc[("A", "RBP1"), "expressed"] and not f.loc[("A", "RBP2"), "expressed"]
    assert f.loc[("B", "RBP2"), "expressed"] and not f.loc[("B", "RBP1"), "expressed"]
    assert f.loc[("A", "AGO1"), "expressed"] and f.loc[("A", "AGO1"), "mirna_machinery"]
    assert not f.loc[("A", "RBPX"), "on_axis"] and not f.loc[("A", "RBPX"), "expressed"] and np.isnan(f.loc[("A", "RBPX"), "cpm"])
    assert abs(f.loc[("A", "RBP1"), "cpm"] - 12 / 45 * 1e6) < 1e-6 and abs(f.loc[("A", "AGO1"), "frac_cells"] - 2 / 3) < 1e-9
    g = genes.set_index("symbol")
    assert g.loc["G1", "cx_sites_expressed_A"] == 2 and g.loc["G1", "cx_sites_silent_A"] == 1        # RBPX off the axis: in neither
    assert g.loc["G1", "cx_n_rbps_expressed_A"] == 1 and abs(g.loc["G1", "cx_frac_sites_expressed_A"] - 2 / 3) < 1e-9
    assert g.loc["G1", "cx_sites_expressed_B"] == 1 and g.loc["G1", "cx_sites_silent_B"] == 2
    assert g.loc["G3", "cx_sites_expressed_A"] == 0 and g.loc["G3", "cx_frac_sites_expressed_A"] == 0.0
    assert g.loc["G3", "cx_sites_expressed_B"] == 3 and g.loc["G3", "cx_frac_sites_expressed_B"] == 1.0
    assert np.isnan(g.loc["G2", "cx_frac_sites_expressed_A"]) and g.loc["G2", "cx_n_rbps_expressed_B"] == 0
    assert "cx_sites_expressed_A" in entry["columns"]
    # RBP1 and RBP2 are each on in one context and off in the other: both variable; RBPX is off the axis
    assert entry["variable_direct_rbps"] == ["RBP1", "RBP2"]
    assert g.loc["G1", "cx_sites_variable_A"] == 2 and g.loc["G1", "cx_sites_variable_B"] == 1
    assert g.loc["G3", "cx_sites_variable_A"] == 0 and g.loc["G3", "cx_n_rbps_variable_B"] == 1
    assert entry["contexts"]["A"]["variable_rbps_expressed"] == ["RBP1"]
    a = entry["contexts"]["A"]
    assert a == {"cells": 3, "factors": 16, "factors_on_axis": 3, "expressed": 2, "direct_rbps_expressed": 1,
                 "direct_rbps_silent": 1, "direct_rbps_off_axis": 1, "mirna_machinery_expressed": ["AGO1"],
                 "median_frac_sites_expressed": pytest.approx(1 / 3), "variable_rbps_expressed": ["RBP1"]}
    assert entry["depmap_check"] == {"skipped": "not asked"}
    assert set(entry["inputs"]["contexts"]) == {"A", "B"} and entry["inputs"]["contexts"]["A"]["cells"] == 3


def test_write_lands_the_tables_and_skips_an_absent_depmap(root):
    out = write_control_factors(root, depmap=True, eclip_manifest=root / "none.json")
    assert sorted(p.name for p in out.iterdir()) == ["LINEAGE.json", "ctx_site_counts.parquet",
                                                      "depmap_check.json", "factor_expression.parquet"]
    check = json.loads((out / "depmap_check.json").read_text())
    assert "skipped" in check and "DepMap files" in check["skipped"]
    e = json.loads((out / "LINEAGE.json").read_text())["entries"]["control-factor-sets/ctx_site_counts"]
    assert e["sha256"] and e["factor_sha256"] and e["rule"]["expressed"].startswith("pseudobulk CPM >= 1")
    # one context alone, as on Oct 22
    out = write_control_factors(root, h5ads=[root / "vcc2026" / "context_B.h5ad"], depmap=False,
                                eclip_manifest=root / "none.json")
    genes = pd.read_parquet(out / "ctx_site_counts.parquet")
    assert "cx_sites_expressed_B" in genes.columns and "cx_sites_expressed_A" not in genes.columns
