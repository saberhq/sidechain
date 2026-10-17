"""Contract tests for `LineStabilitySource` (T120, phase 3) on synthetic copies of the four published
tables, plus the licence value the Schofield block introduces.

Pinned:

* Jurkat: the resting rate is read raw (negative), the half-life derived from it is ln2/-rate and NaN
  where the rate is missing, the HuR yes/no becomes 1/0, a non-Ensembl row is dropped;
* HCT116: hour and replicate are read off the file names whatever their order, the slope is fitted
  on every sample, a gene designed to halve per hour gets the most negative slope, a gene under the
  0 h TPM floor gets NaN rates but keeps its 0 h TPM, and the centred and raw slopes rank genes
  identically (centring shifts every gene by one constant);
* Schofield: the K562 sheet is read and the mouse one ignored, a symbol on the axis matches exactly,
  a retired symbol is carried to its one approved successor through HGNC, one that two genes once
  carried is left alone, an unknown one keeps its spelling with no id, and the per-gene
  `symbol_match` says which rule applied;
* Schueler: the quoted R header is checked, not split, both lines land in one table under their own
  columns, the mean is derived, and a wrong header is a refusal;
* every table leaves a LINEAGE entry with its column dictionary and `build()` aligns one column to
  the gene space behind a mask;
* the view scopes every column of the four sources to its line (`ptr_table.SOURCES`);
* `Publisher-terms-noncommercial` is accepted by the gate and carries both encumbrance flags.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sidechain.priors import ptr_table
from sidechain.priors.posttx_line_stability import (
    BLOCKS,
    HCT116_BLOCK,
    HEK293_BLOCK,
    JURKAT_BLOCK,
    K562_BLOCK,
    LineStabilitySource,
    actd_fit,
    hgnc_previous,
    resolve_symbols,
)

AXIS = ["A", "B", "C", "D"]            # D is not bridged; OLDB is B's retired symbol
BRIDGE = {"A": "ENSG1.4", "B": "ENSG2", "C": "ENSG3"}


def _provenance(dest: Path, names: list[str], licence: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "PROVENANCE.json").write_text(json.dumps({
        "route": "download", "record": {"license": licence},
        "license_flags": {"noncommercial": False, "redistribution_encumbered": False},
        "selected": [{"name": n, "size_bytes": (dest / n).stat().st_size, "checksum": "http-last-modified:x"}
                     for n in names]}))


def _rsem(path: Path, rows: list[tuple[str, float]]) -> None:
    lines = ["gene_id\ttranscript_id(s)\tlength\teffective_length\texpected_count\tTPM\tFPKM"]
    for gid, tpm in rows:
        lines.append(f"{gid}\tT\t1000\t900\t{tpm * 10:.1f}\t{tpm:.4f}\t{tpm * 2:.4f}")
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")


def _schueler(path: Path, line: str, rows: dict[str, tuple[float, float]]) -> None:
    with gzip.open(path, "wt") as fh:
        fh.write(f'"{line}_RNA half-life (min) rep1" "{line}_RNA half-life (min) rep2"\n')
        for sym, (a, b) in rows.items():
            fh.write(f'"{sym}" {a} {b}\n')


@pytest.fixture
def root(tmp_path):
    (tmp_path / "vcc2026").mkdir()
    pd.DataFrame({"gene_name": AXIS}).to_csv(tmp_path / "vcc2026" / "gene_names.csv", index=False)
    bridge_dir = tmp_path / "runs/probes/t102_utr_load/bridge"
    bridge_dir.mkdir(parents=True)
    (bridge_dir / "symbol_to_ensg.json").write_text(json.dumps(BRIDGE))
    hg = tmp_path / "external/hgnc-custom-uniprot-20261005"
    hg.mkdir(parents=True)
    pd.DataFrame({"HGNC ID": ["HGNC:1", "HGNC:2", "HGNC:3", "HGNC:9"],
                  "Approved symbol": ["A", "B", "C", "Z"], "Status": ["Approved"] * 4,
                  "Previous symbols": ["", "OLDB", "OLDC, OLDC2", "OLDC"],      # OLDC is ambiguous (B? no: C and Z)
                  "Locus type": ["gene with protein product"] * 4,
                  "UniProt ID(supplied by UniProt)": ["P1", "P2", "P3", "P9"]}
                 ).to_csv(hg / "hgnc_custom_uniprot.tsv", sep="\t", index=False)

    # Jurkat sheet
    jk = tmp_path / "external/jk"
    jk.mkdir(parents=True)
    pd.DataFrame({
        "Gene Symbol": ["A", "B", "C", "junk"], "Avg Tot RPKM": [10.0, 20.0, 30.0, 1.0],
        "EnsemblGene": ["ENSG1", "ENSG2", "ENSG3", "nope"], "iCLIP Target": ["yes", "no", "no", "no"],
        "LOD_HuR 293": [0.1, 0.2, 0.3, 0.4], "ELAVL1 3UTR": ["t", "n", "n", "n"], "HuR_Any": ["yes", np.nan, np.nan, np.nan],
        "HuR LOD 0hr": [-1.0, np.nan, -3.0, np.nan], "HuR LOD 4hr": [-1.5, np.nan, -2.5, np.nan],
        "Avg Opt DR U": [-np.log(2.0), -0.1, np.nan, -0.2], "StDev Opt DR U": [0.01, 0.02, 0.03, 0.04],
        "Avg Opt DR 1": [-0.5, -0.1, -0.2, -0.2], "Avg Opt DR 2": [-0.5, -0.1, -0.2, -0.2],
        "Avg Opt DR 3": [-0.5, -0.1, -0.2, -0.2], "Avg Opt DR 4": [-0.4, -0.1, -0.2, -0.2],
    }).to_excel(jk / "GSE71382_ProcessedData.xlsx", index=False)
    _provenance(jk, ["GSE71382_ProcessedData.xlsx"], "NCBI-no-restrictions")

    # HCT116: A halves every hour, B is flat, C rises a little, D is below the floor; three replicates with jitter
    hc = tmp_path / "external/hc"
    hc.mkdir(parents=True)
    rng = np.random.default_rng(0)
    names = []
    for hour in (0, 2, 4):
        for rep in (1, 2, 3):
            tag = f"HCT116{'ActD' if hour else ''}{hour}hr{rep}"
            name = f"GSM{hour}{rep}_{tag}_rsem.genes.results.gz"
            names.append(name)
            jit = lambda v: float(v * np.exp(rng.normal(0, 0.02)))          # noqa: E731
            _rsem(hc / name, [("ENSG1.3_A", jit(16.0 * 0.5 ** hour)), ("ENSG2.1_B", jit(8.0)),
                              ("ENSG3.7_C", jit(8.0 * 1.1 ** hour)), ("ENSG4.1_D", jit(0.2)),
                              ("ENSG5.1_E", jit(6.0))])
    _provenance(hc, names, "NCBI-no-restrictions")

    # Schofield workbook: K562 sheet plus a mouse sheet that must be ignored
    sf = tmp_path / "external/sf"
    sf.mkdir(parents=True)
    with pd.ExcelWriter(sf / "41592_2018_BFnmeth4582_MOESM5_ESM.xlsx") as xw:
        pd.DataFrame({"transcript": ["A", "OLDB", "OLDC", "ZZZ"], "theta_n1": [0.2, 0.5, 0.3, 0.4],
                      "theta_n2": [0.25, 0.45, 0.3, 0.4], "half_life_rep1": [2.0, 1.0, 3.0, 4.0],
                      "half_life_rep2": [2.4, 1.2, 3.0, 4.0], "mean_half_life": [2.2, 1.1, 3.0, 4.0]}
                     ).to_excel(xw, sheet_name="Table S2_K562", index=False)
        pd.DataFrame({"transcript": ["Rik1"], "theta_n1": [0.1], "theta_n2": [0.1], "half_life_rep1": [9.0],
                      "half_life_rep2": [9.0], "mean_half_life": [9.0]}).to_excel(xw, sheet_name="Table S2_MEF", index=False)
    _provenance(sf, ["41592_2018_BFnmeth4582_MOESM5_ESM.xlsx"], "Publisher-terms-noncommercial")

    # Schueler: two R tables
    sl = tmp_path / "external/sl"
    sl.mkdir(parents=True)
    _schueler(sl / "GSE49831_HEK293_halflives.txt.gz", "HEK293", {"A": (90.0, 70.0), "OLDB": (200.0, 220.0)})
    _schueler(sl / "GSE49831_MCF7_halflives.txt.gz", "MCF7", {"A": (300.0, 320.0), "C": (500.0, 480.0)})
    _provenance(sl, ["GSE49831_HEK293_halflives.txt.gz", "GSE49831_MCF7_halflives.txt.gz"], "NCBI-no-restrictions")

    registry = tmp_path / "data_sources.yaml"
    registry.write_text(yaml.safe_dump({"sources": [
        {"name": JURKAT_BLOCK, "kind": "node_feature", "layer": "posttx", "relation": "mrna_stability_line",
         "dest": "external/jk", "derived": "derived/jk", "license": "NCBI-no-restrictions",
         "files": [{"name": "GSE71382_ProcessedData.xlsx"}]},
        {"name": HCT116_BLOCK, "kind": "node_feature", "layer": "posttx", "relation": "mrna_stability_line",
         "dest": "external/hc", "derived": "derived/hc", "license": "NCBI-no-restrictions",
         "files": [{"name": n, "url": f"https://x/{n}"} for n in reversed(names)]},    # reversed on purpose
        {"name": K562_BLOCK, "kind": "node_feature", "layer": "posttx", "relation": "mrna_stability_line",
         "dest": "external/sf", "derived": "derived/sf", "license": "Publisher-terms-noncommercial",
         "files": [{"name": "41592_2018_BFnmeth4582_MOESM5_ESM.xlsx"}]},
        {"name": HEK293_BLOCK, "kind": "node_feature", "layer": "posttx", "relation": "mrna_stability_line",
         "dest": "external/sl", "derived": "derived/sl", "license": "NCBI-no-restrictions",
         "files": [{"name": "GSE49831_HEK293_halflives.txt.gz"}, {"name": "GSE49831_MCF7_halflives.txt.gz"}]},
    ]}))
    return tmp_path, registry


def _src(root, block, gene_index=None):
    tmp, registry = root
    spec = next(s for s in yaml.safe_load(registry.read_text())["sources"] if s["name"] == block)
    return LineStabilitySource(spec, gene_index or {}, root=tmp, registry=registry)


def _lineage(src):
    return json.loads((src.derived / "LINEAGE.json").read_text())["entries"][src.lineage_key]


# ------------------------------------------------------------------------- Jurkat --

def test_jurkat_rate_raw_halflife_derived_hur_as_number_and_non_ensembl_dropped(root):
    t = _src(root, JURKAT_BLOCK).stability_table().set_index("gene_id")
    assert list(t.index) == ["ENSG1", "ENSG2", "ENSG3"]                         # "nope" dropped
    assert t.loc["ENSG1", "decay_rate_jurkat_rest"] == pytest.approx(-np.log(2.0))
    assert t.loc["ENSG1", "halflife_jurkat_rest_h"] == pytest.approx(1.0)       # ln2 / ln2
    assert t.loc["ENSG2", "halflife_jurkat_rest_h"] == pytest.approx(np.log(2.0) / 0.1)
    assert np.isnan(t.loc["ENSG3", "halflife_jurkat_rest_h"])                   # no resting rate
    assert t["hur_iclip_target_jurkat"].tolist() == [1.0, 0.0, 0.0]
    assert "LOD_HuR 293" not in t.columns and not any(c.startswith("Avg UT") for c in t.columns)
    e = _lineage(_src(root, JURKAT_BLOCK))
    assert e["counts"]["genes_with_resting_rate"] == 2 and e["counts"]["hur_iclip_targets"] == 1
    assert "halflife_jurkat_rest_h" in e["columns"] and e["licence"] == "NCBI-no-restrictions"
    assert "GSE71382_ProcessedData.xlsx" in e["inputs"]


# ------------------------------------------------------------------------- HCT116 --

def test_actd_fit_recovers_a_planted_slope_and_its_se():
    hours = np.array([0, 0, 2, 2, 4, 4], dtype=float)
    y = np.vstack([-0.5 * hours + 3.0, 0.0 * hours + 1.0])
    fit = actd_fit(y, hours)
    assert fit["slope"] == pytest.approx([-0.5, 0.0])
    assert fit["se"] == pytest.approx([0.0, 0.0])


def test_hct116_samples_are_read_off_the_names_in_hour_order_and_the_rate_is_fitted(root):
    src = _src(root, HCT116_BLOCK)
    samples = src._hct116_samples()
    assert [(h, r) for _, h, r in samples] == [(0, 1), (0, 2), (0, 3), (2, 1), (2, 2), (2, 3), (4, 1), (4, 2), (4, 3)]
    t = src.stability_table().set_index("gene_id")
    assert t.loc["ENSG1", "actd_slope_hct116"] < t.loc["ENSG2", "actd_slope_hct116"] < t.loc["ENSG3", "actd_slope_hct116"]
    # A halves per hour: raw slope about -1 log2/h; the centred slope is the raw one shifted by one constant
    assert t.loc["ENSG1", "actd_slope_raw_hct116"] == pytest.approx(-1.0, abs=0.1)
    shift = (t["actd_slope_raw_hct116"] - t["actd_slope_hct116"]).dropna()
    assert shift.max() - shift.min() == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(t.loc["ENSG4", "actd_slope_hct116"]) and t.loc["ENSG4", "tpm0_hct116"] == pytest.approx(0.2, abs=0.02)
    assert t["symbol"].tolist() == ["A", "B", "C", "D", "E"]
    assert (t["actd_slope_hct116_se"].dropna() >= 0).all()
    e = _lineage(src)
    assert e["counts"]["genes_expressed"] == 4 and e["counts"]["tpm_floor_at_0h"] == 1.0
    assert len(e["counts"]["samples"]) == 9 and len(e["inputs"]) == 9


def test_hct116_refuses_a_block_without_three_time_points(root):
    src = _src(root, HCT116_BLOCK)
    src.spec = {**src.spec, "files": [f for f in src.spec["files"] if "4hr" not in f["name"]]}
    with pytest.raises(ValueError, match="three time points"):
        src._hct116_samples()


# -------------------------------------------------------------------- symbol keys --

def test_resolve_symbols_exact_then_one_hgnc_successor_else_nothing(root):
    tmp, _ = root
    prev = hgnc_previous(tmp / "external/hgnc-custom-uniprot-20261005/hgnc_custom_uniprot.tsv")
    assert prev["OLDB"] == {"B"} and prev["OLDC"] == {"C", "Z"}
    keys, counts = resolve_symbols(pd.Series(["A", "OLDB", "OLDC", "ZZZ", "D"]), AXIS, {k: v.split(".")[0] for k, v in BRIDGE.items()}, prev)
    assert keys["symbol"].tolist() == ["A", "B", "OLDC", "ZZZ", "D"]
    assert keys["symbol_match"].tolist() == ["exact", "hgnc_previous", "none", "none", "exact"]
    assert keys["gene_id"].tolist()[:2] == ["ENSG1", "ENSG2"] and keys["gene_id"].isna().tolist() == [False, False, True, True, True]
    assert counts == {"exact": 2, "hgnc_previous": 1, "none": 2}


# ------------------------------------------------------------------------ Schofield --

def test_schofield_reads_the_k562_sheet_only_and_bridges_symbols(root):
    src = _src(root, K562_BLOCK)
    t = src.stability_table()
    assert t["symbol_source"].tolist() == ["A", "OLDB", "OLDC", "ZZZ"]
    assert t["symbol"].tolist() == ["A", "B", "OLDC", "ZZZ"]
    assert t["gene_id"].tolist()[:2] == ["ENSG1", "ENSG2"] and t["gene_id"].isna().sum() == 2
    assert t["halflife_k562_timelapse_mean_h"].tolist() == [2.2, 1.1, 3.0, 4.0]
    assert t["frac_new_4h_k562_timelapse_rep1"].tolist() == [0.2, 0.5, 0.3, 0.4]
    assert "Rik1" not in t["symbol_source"].tolist()
    e = _lineage(src)
    assert e["counts"]["symbol_resolution"] == {"exact": 1, "hgnc_previous": 1, "none": 2, "with_gene_id": 2}
    assert e["licence"] == "Publisher-terms-noncommercial"
    assert any(k.endswith("hgnc_custom_uniprot.tsv") for k in e["inputs"])


# ------------------------------------------------------------------------- Schueler --

def test_schueler_two_lines_one_table_header_checked(root):
    src = _src(root, HEK293_BLOCK)
    t = src.stability_table().set_index("symbol")
    assert sorted(t.index) == ["A", "B", "C"]
    assert t.loc["A", "halflife_hek293_schueler_mean_min"] == pytest.approx(80.0)
    assert t.loc["B", "halflife_hek293_schueler_rep1_min"] == pytest.approx(200.0)      # OLDB -> B
    assert np.isnan(t.loc["B", "halflife_mcf7_schueler_mean_min"])                      # MCF7 has no B
    assert t.loc["C", "halflife_mcf7_schueler_mean_min"] == pytest.approx(490.0)
    assert t.loc["A", "gene_id"] == "ENSG1"
    e = _lineage(src)
    assert e["counts"]["lines"]["hek293"]["genes"] == 2 and e["counts"]["lines"]["mcf7"]["genes"] == 2
    assert e["counts"]["symbol_resolution"]["hgnc_previous"] == 1


def test_schueler_refuses_a_header_that_is_not_the_published_one(root):
    tmp, _ = root
    src = _src(root, HEK293_BLOCK)
    bad = tmp / "external/sl/GSE49831_HEK293_halflives.txt.gz"
    with gzip.open(bad, "wt") as fh:
        fh.write('"HEK293 half-life rep1" "HEK293 half-life rep2"\n"A" 1 2\n')
    with pytest.raises(ValueError, match="header names"):
        src._read_schueler(bad, "hek293")


# ---------------------------------------------------------------------------- build --

def test_build_aligns_the_primary_column_behind_a_mask(root):
    src = _src(root, K562_BLOCK, gene_index={"ENSG1": 0, "ENSG2": 1, "ENSG3": 2, "ENSGX": 3})
    art = src.build()
    assert art.kind == "node_feature" and art.features.shape == (4, 1)
    assert art.features[:, 0].tolist() == pytest.approx([2.2, 1.1, 0.0, 0.0])
    assert art.meta["has_value"][:, 0].tolist() == [True, True, False, False]
    assert art.meta["feature_names"] == ["halflife_k562_timelapse_mean_h"]


def test_loader_refuses_a_block_it_does_not_serve(root):
    tmp, registry = root
    with pytest.raises(ValueError, match="not a block this loader serves"):
        LineStabilitySource({"name": "halflife_consensus", "kind": "node_feature", "dest": "x", "derived": "y"}, {},
                            root=tmp, registry=registry)


# ----------------------------------------------------------------------- the view --

def test_the_view_scopes_every_phase3_column_to_its_line():
    by_key = {s.key: s for s in ptr_table.SOURCES}
    assert {s.block for s in ptr_table.SOURCES if s.key in ("bk", "pz", "sf", "sl")} == set(BLOCKS)
    assert ptr_table.column_scope(by_key["bk"], "halflife_jurkat_rest_h") == "line:jurkat"
    assert ptr_table.column_scope(by_key["pz"], "tpm0_hct116") == "line:hct116"
    assert ptr_table.column_scope(by_key["sf"], "frac_new_4h_k562_timelapse_rep1") == "line:k562"
    assert ptr_table.column_scope(by_key["sl"], "halflife_hek293_schueler_mean_min") == "line:hek293"
    assert ptr_table.column_scope(by_key["sl"], "halflife_mcf7_schueler_rep2_min") == "line:mcf7"
    assert "symbol_source" in by_key["sf"].drop and "symbol_match" in by_key["sl"].drop
    assert len({s.key for s in ptr_table.SOURCES}) == len(ptr_table.SOURCES)


# -------------------------------------------------------------------- the licence --

def test_publisher_terms_noncommercial_is_accepted_and_carries_both_flags():
    from sidechain.ingest.provenance import ACCEPTED_LICENSES, LICENSE_POLICY, NONCOMMERCIAL, REDISTRIBUTION_ENCUMBERED
    v = LICENSE_POLICY["publisher-terms-noncommercial"]
    assert v == "Publisher-terms-noncommercial" and v in ACCEPTED_LICENSES
    assert v in NONCOMMERCIAL and v in REDISTRIBUTION_ENCUMBERED
