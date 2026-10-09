"""Tests for the Cell Ranger feature-barcode h5 reader (Pan 2026's H1 screen, T114).

The sums are checked against a plain loop over cells rather than against a second
implementation, as in the parquet streamer's tests and for the same reason. What is new in
this reader, and so what most of these pin, is everything BEFORE the sums: the file has no
perturbation column, so the orientation of the matrix, the cell filter, the guide call and
the control arm are all derived here from rules a config block declares.

Every fixture mirrors the real layout as read from GEO on 2026-10-09
(`GSM8943725_..._filtered_feature_bc_matrix.h5`): the matrix is stored features x barcodes
and compressed by barcode, `data` is int32 and `indices` int64, gene features carry an
Ensembl id and a symbol, guide features carry the guide name twice and no target tag.
"""
import contextlib
import hashlib
import io
import json

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml

from sidechain.data import stream_10x_h5 as r
from sidechain.data.stream_pseudobulk import PseudobulkSums

GENES = [  # (ensembl id, symbol)
    ("ENSG0", "AAA"), ("ENSG1", "BBB"), ("ENSG2", "CCC"),
    ("ENSG3", "DUP"), ("ENSG4", "DUP"),          # one symbol on two ids, as 21 real names are
    ("ENSG5", "AARS1"),                          # the CURRENT symbol; the 2026 axis says AARS
    ("ENSG6", "MT-ND1"),
    ("ENSG7", "TGT1"), ("ENSG8", "TGT2"),
    ("ENSG9", "NEWX"),                           # the library spells this target OLDX
]
GUIDES = ["TGT1_+_100.23-P1P2", "TGT1_-_200.23-P1P2", "TGT2_+_300.23-P1",
          "OLDX_-_5.23-P1P2", "non-targeting_00001", "non-targeting_00002"]
N_GE, N_GUIDE = len(GENES), len(GUIDES)

SPEC = {
    "context": "h1", "pert_col": "target_gene", "control_label": "non-targeting",
    "modality": "crispri", "role": "analysis",
    "expression_feature_type": "Gene Expression",
    "guide_feature_type": "CRISPR Guide Capture",
    "min_genes": 3, "guide_min_umi": 20, "guide_rule": "exactly_one",
    "guide_target_regex": r"^(?P<target>.+)_[+-]_\d+\.\d+-.+$",
    "control_guide_regex": r"^non-targeting_\d+$",
    "n_control_guides": 2,
}
RULES = r.LaneRules.from_spec(SPEC)
CHALLENGE = ["AAA", "AARS", "DUP", "MISSING", "CCC", "TGT1", "TGT2", "MT-ND1"]


def _write_lane(path, dense, barcodes=None, *, genes=GENES, guides=GUIDES):
    """`dense` is cells x features. Written the way Cell Ranger does: features x barcodes,
    compressed by barcode."""
    dense = np.asarray(dense)
    stored = sp.csc_matrix(dense.T)
    barcodes = barcodes or [f"BC{i:04d}-1" for i in range(dense.shape[0])]
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g["data"] = stored.data.astype(np.int32)
        g["indices"] = stored.indices.astype(np.int64)
        g["indptr"] = stored.indptr.astype(np.int64)
        g["shape"] = np.asarray(stored.shape, dtype=np.int32)
        g["barcodes"] = np.asarray(barcodes, dtype="S")
        fg = g.create_group("features")
        fg["id"] = np.asarray([i for i, _ in genes] + list(guides), dtype="S")
        fg["name"] = np.asarray([n for _, n in genes] + list(guides), dtype="S")
        fg["feature_type"] = np.asarray(
            ["Gene Expression"] * len(genes) + ["CRISPR Guide Capture"] * len(guides), dtype="S")
    return path


def _cell(ge, guide=None, umi=50, extra=None):
    """One row: gene counts, then guide counts with `umi` on `guide` (an index) if given."""
    row = np.zeros(N_GE + N_GUIDE, dtype=np.int64)
    row[:N_GE] = ge
    if guide is not None:
        row[N_GE + guide] = umi
    for g, u in (extra or {}).items():
        row[N_GE + g] = u
    return row


def _random_lane(rng, n_cells):
    rows = []
    for _ in range(n_cells):
        ge = rng.integers(0, 6, size=N_GE)
        kind = rng.integers(0, 10)
        if kind == 0:                                    # no guide at threshold
            rows.append(_cell(ge, guide=int(rng.integers(0, N_GUIDE)), umi=19))
        elif kind == 1:                                  # two guides at threshold
            rows.append(_cell(ge, guide=0, umi=30, extra={4: 25}))
        else:
            rows.append(_cell(ge, guide=int(rng.integers(0, N_GUIDE)),
                              umi=int(rng.integers(20, 400)),
                              extra={int(rng.integers(0, N_GUIDE)): 0}))
    return np.vstack(rows)


def _ctx(lane, challenge=CHALLENGE, aliases=None):
    return r.LaneContext.from_lane(lane, RULES, list(challenge) if challenge else None,
                                   target_aliases=aliases)


# ---------------------------------------------------------------- the rules --


def test_rules_are_read_from_the_spec_and_never_defaulted():
    for key in r.RULE_KEYS:
        spec = {k: v for k, v in SPEC.items() if k != key}
        with pytest.raises(ValueError, match=key):
            r.LaneRules.from_spec(spec)
    with pytest.raises(ValueError, match="not implemented"):
        r.LaneRules.from_spec({**SPEC, "guide_rule": "most_abundant"})
    with pytest.raises(ValueError, match="named group"):
        r.LaneRules.from_spec({**SPEC, "guide_target_regex": r"^(.+)_[+-]_.*$"})
    with pytest.raises(TypeError, match="single string"):
        r.LaneRules.from_spec({**SPEC, "control_label": ["a", "b"]})


def test_the_registry_block_parses_into_rules():
    """The real block, not a fixture: every lane of Pan 2026 shares one spec and it carries
    every rule this reader needs."""
    from sidechain.ingest.fetch import load_datasets

    block = load_datasets()["pan2026_h1_crispri"]
    specs = {json.dumps(f["spec"], sort_keys=True) for f in block["files"]}
    assert len(block["files"]) == 85 and len(specs) == 1
    rules = r.LaneRules.from_spec(block["files"][0]["spec"])
    assert (rules.min_genes, rules.guide_min_umi, rules.n_control_guides) == (500, 20, 370)
    lib = r.build_guide_library(
        np.array(["GTF3C5_-_135906397.23-P1P2", "HLA-A_+_1.23-P1", "ENSG_X_+_7.20-P2",
                  # a hyphen against the strand's, and a versioned transcript id as the tail
                  "KRTAP13-3_-_31797823.23-ENST00000390690.2",
                  "ZNF469_+_88493915.23-ENST00000437464.1"]
                 + [f"non-targeting_{i:05d}" for i in range(370)], dtype=object), rules)
    assert lib.targets == ["ENSG_X", "GTF3C5", "HLA-A", "KRTAP13-3", "ZNF469"]


# ------------------------------------------------------------- reading a lane --


def test_a_lane_reads_as_cells_by_features(tmp_path):
    rng = np.random.default_rng(0)
    dense = rng.integers(0, 4, size=(7, N_GE + N_GUIDE))
    lane = r.read_lane(_write_lane(tmp_path / "a.h5", dense), "lane_a")
    assert lane.X.shape == (7, N_GE + N_GUIDE)
    assert np.array_equal(lane.X.toarray(), dense)
    assert lane.barcodes[3] == "BC0003-1" and lane.feature_names[5] == "AARS1"
    assert lane.feature_ids[0] == "ENSG0" and lane.feature_types[-1] == "CRISPR Guide Capture"


def test_a_matrix_compressed_the_other_way_is_refused(tmp_path):
    """The three arrays build a matrix either way round whenever the sizes allow it; only
    the lengths say which way the file was compressed."""
    dense = np.random.default_rng(1).integers(0, 4, size=(7, N_GE + N_GUIDE))
    path = _write_lane(tmp_path / "a.h5", dense)
    with h5py.File(path, "r+") as f:
        stored = sp.csr_matrix(dense.T)          # compressed by FEATURE instead
        for key, value in (("data", stored.data.astype(np.int32)),
                           ("indices", stored.indices.astype(np.int64)),
                           ("indptr", stored.indptr.astype(np.int64))):
            del f["matrix"][key]
            f["matrix"][key] = value
    with pytest.raises(ValueError, match="not compressed by barcode"):
        r.read_lane(path)


def test_non_integer_counts_are_refused(tmp_path):
    path = _write_lane(tmp_path / "a.h5", np.ones((3, N_GE + N_GUIDE), dtype=int))
    with h5py.File(path, "r+") as f:
        data = f["matrix"]["data"][:].astype(np.float32) * 0.5
        del f["matrix"]["data"]
        f["matrix"]["data"] = data
    with pytest.raises(ValueError, match="not integers"):
        r.read_lane(path)


# ------------------------------------------------------------ the guide table --


def test_guides_take_their_target_from_the_name_and_controls_pool():
    lib = r.build_guide_library(np.asarray(GUIDES, dtype=object), RULES)
    assert lib.targets == ["OLDX", "TGT1", "TGT2"]
    assert lib.guide_label.tolist() == ["TGT1", "TGT1", "TGT2", "OLDX",
                                        "non-targeting", "non-targeting"]
    assert lib.gene_labels == ["OLDX", "TGT1", "TGT2", "non-targeting"]
    assert lib.guide_level_labels("non-targeting") == sorted(GUIDES[:4] + ["non-targeting"])


def test_a_control_count_other_than_the_declared_one_is_refused():
    with pytest.raises(ValueError, match="n_control_guides: 2"):
        r.build_guide_library(np.asarray(GUIDES[:5], dtype=object), RULES)


def test_a_guide_matching_neither_pattern_is_refused_not_dropped():
    with pytest.raises(ValueError, match="match neither"):
        r.build_guide_library(np.asarray([*GUIDES, "safe-targeting_1"], dtype=object), RULES)
    # the control pattern must match the WHOLE name: `non-targeting_1_extra` is not a control
    with pytest.raises(ValueError, match="match neither"):
        r.build_guide_library(np.asarray([*GUIDES, "non-targeting_1_extra"], dtype=object), RULES)


# ----------------------------------------------------------------- the calls --


def test_the_cell_filter_and_the_guide_call_are_inclusive_at_both_thresholds(tmp_path):
    three = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    two = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    dense = np.vstack([
        _cell(three, guide=0, umi=20),                      # kept: 3 genes, one guide at 20
        _cell(three, guide=0, umi=19),                      # no guide: 19 is below 20
        _cell(two, guide=0, umi=99),                        # low genes: 2 is below 3
        _cell(three, guide=0, umi=40, extra={2: 20}),       # two guides at threshold
        _cell(three, guide=4, umi=25, extra={1: 19, 3: 7}),  # kept: control; 19 does not count
        _cell(two),                                         # low genes is read first
        _cell(three),                                       # no guide at all
    ])
    lane = r.read_lane(_write_lane(tmp_path / "a.h5", dense))
    res = r.process_lane(lane, _ctx(lane))
    c = res.cells
    assert c["drop_reason"].tolist() == ["", "no_guide", "low_genes", "multi_guide", "",
                                         "low_genes", "no_guide"]
    assert c["kept"].tolist() == [True, False, False, False, True, False, False]
    assert c["target_gene"].tolist() == ["TGT1", "", "", "", "non-targeting", "", ""]
    assert c["top_guide"].tolist()[:5] == [GUIDES[0], GUIDES[0], GUIDES[0], GUIDES[0], GUIDES[4]]
    assert c["top_guide"].tolist()[5:] == ["", ""]
    assert c["top_guide_umi"].tolist() == [20, 19, 99, 40, 25, 0, 0]
    assert c["second_guide_umi"].tolist() == [0, 0, 0, 20, 19, 0, 0]
    assert c["n_guides_called"].tolist() == [1, 0, 1, 2, 1, 0, 0]
    assert c["n_guides_detected"].tolist() == [1, 1, 1, 2, 3, 0, 0]
    assert c["n_guides_ge10"].tolist() == [1, 1, 1, 2, 2, 0, 0]
    assert c["n_guides_ge5"].tolist() == [1, 1, 1, 2, 3, 0, 0]
    assert c["barcode"].tolist() == [f"BC{i:04d}-1" for i in range(7)]
    assert res.gene_label.tolist() == ["TGT1", "non-targeting"]
    assert res.guide_label.tolist() == [GUIDES[0], "non-targeting"]


# --------------------------------------------------------------- the gene axis --


def test_the_axis_is_the_challenge_order_with_the_alias_table_read_in_reverse():
    names = np.asarray([n for _, n in GENES], dtype=object)
    ids = np.asarray([i for i, _ in GENES], dtype=object)
    axis = r.build_symbol_axis(names, ids, CHALLENGE)
    assert axis.genes.tolist() == ["AAA", "AARS", "DUP", "CCC", "TGT1", "TGT2", "MT-ND1"]
    assert axis.unmapped_challenge == ["MISSING"]
    assert axis.collided_symbols == ["DUP"]
    assert axis.alias_recovered == [
        {"axis_symbol": "AARS", "source_symbol": "AARS1", "ensembl_ids": ["ENSG5"]}]
    # feature 5 (AARS1) feeds column 1 (AARS); features 3 and 4 both feed DUP; BBB is dropped
    assert axis.col_of_feature.tolist() == [0, -1, 3, 2, 2, 1, 6, 4, 5, -1]
    dense = np.arange(1, 11)[None, :]
    assert (sp.csr_matrix(dense) @ axis.projector()).toarray().tolist() == [[1, 6, 9, 3, 8, 9, 7]]


def test_an_alias_is_not_applied_when_both_spellings_are_challenge_genes():
    """One feature must not feed two columns: with AARS and AARS1 both on the axis the
    feature named AARS1 is AARS1's, and AARS is recorded as unmapped."""
    names = np.asarray([n for _, n in GENES], dtype=object)
    ids = np.asarray([i for i, _ in GENES], dtype=object)
    axis = r.build_symbol_axis(names, ids, ["AARS", "AARS1"])
    assert axis.genes.tolist() == ["AARS1"] and axis.unmapped_challenge == ["AARS"]
    assert axis.alias_recovered == []


def test_all_genes_keeps_every_symbol_once():
    names = np.asarray([n for _, n in GENES], dtype=object)
    ids = np.asarray([i for i, _ in GENES], dtype=object)
    axis = r.build_symbol_axis(names, ids, None)
    assert axis.genes.tolist() == sorted({n for _, n in GENES})
    assert not axis.restricted and axis.collided_symbols == ["DUP"]


def test_a_targets_own_gene_is_found_by_symbol_then_by_alias():
    names = np.asarray([n for _, n in GENES], dtype=object)
    proj, has = r.own_gene_projector(names, ["OLDX", "TGT1", "TGT2"], aliases={})
    assert has.tolist() == [False, True, True]
    proj, has = r.own_gene_projector(names, ["OLDX", "TGT1", "TGT2"], aliases={"OLDX": "NEWX"})
    assert has.tolist() == [True, True, True]
    assert (sp.csr_matrix(np.arange(1, 11)[None, :]) @ proj).toarray().tolist() == [[10, 8, 9]]


# -------------------------------------------------------------------- the sums --


def _brute(lanes_dense, axis_genes, by):
    """Per-label sums by a plain loop over cells. `by` maps a guide index to its label."""
    col = {"AAA": [0], "AARS": [5], "DUP": [3, 4], "CCC": [2], "TGT1": [7], "TGT2": [8],
           "MT-ND1": [6]}
    out: dict[str, dict] = {}
    for dense in lanes_dense:
        for row in dense:
            ge, guides = row[:N_GE], row[N_GE:]
            called = np.flatnonzero(guides >= 20)
            if (ge > 0).sum() < 3 or len(called) != 1:
                continue
            x = np.array([ge[col[g]].sum() for g in axis_genes], dtype=np.float64)
            lib = x.sum()
            if lib <= 0:
                continue
            slot = out.setdefault(by(int(called[0])), {
                "count": np.zeros(len(axis_genes)), "cpm": np.zeros(len(axis_genes)),
                "cpm_sq": np.zeros(len(axis_genes)), "n": 0, "lib": 0.0})
            cpm = x / lib * 1e6
            slot["count"] += x
            slot["cpm"] += cpm
            slot["cpm_sq"] += cpm * cpm
            slot["n"] += 1
            slot["lib"] += lib
    return out


def _assert_matches(pb: PseudobulkSums, want: dict):
    assert pb.labels == sorted(want)
    for i, lab in enumerate(pb.labels):
        assert np.array_equal(pb.count_sum[i], want[lab]["count"]), lab
        assert np.allclose(pb.cpm_sum[i], want[lab]["cpm"], rtol=1e-12), lab
        assert np.allclose(pb.cpm_sq_sum[i], want[lab]["cpm_sq"], rtol=1e-12), lab
        assert pb.n_cells[i] == want[lab]["n"] and np.isclose(pb.libsize_sum[i], want[lab]["lib"])


@pytest.fixture
def three_lanes(tmp_path):
    rng = np.random.default_rng(7)
    dense = [_random_lane(rng, n) for n in (60, 45, 80)]
    lanes = [(f"L{i}", _write_lane(tmp_path / f"L{i}.h5", d, [f"L{i}BC{j}-1" for j in range(len(d))]))
             for i, d in enumerate(dense)]
    return dense, lanes


def test_gene_and_guide_sums_equal_a_loop_over_cells(three_lanes):
    dense, lanes = three_lanes
    res = r.stream_lanes(lanes, RULES, challenge_genes=CHALLENGE, guide_level=True)
    genes = res.gene_pb.genes.tolist()
    label = ["TGT1", "TGT1", "TGT2", "OLDX", "non-targeting", "non-targeting"]
    _assert_matches(res.gene_pb, _brute(dense, genes, lambda g: label[g]))
    guide = [*GUIDES[:4], "non-targeting", "non-targeting"]
    _assert_matches(res.guide_pb, _brute(dense, genes, lambda g: guide[g]))
    # every kept cell carries exactly one guide, so the guide rows sum to the gene rows
    collapsed = r.collapse_to_genes(res.guide_pb, res.ctx.library, "non-targeting")
    assert collapsed.labels == res.gene_pb.labels
    assert np.array_equal(collapsed.count_sum, res.gene_pb.count_sum)
    assert np.allclose(collapsed.cpm_sum, res.gene_pb.cpm_sum, rtol=1e-12)
    assert np.array_equal(collapsed.n_cells, res.gene_pb.n_cells)
    assert res.gene_pb.sources == ["L0", "L1", "L2"]


def test_lane_groups_are_written_as_they_finish_and_merge_to_the_whole(three_lanes, tmp_path):
    dense, lanes = three_lanes
    whole = r.stream_lanes(lanes, RULES, challenge_genes=CHALLENGE)
    groups = {"L0": "half1", "L1": "half0", "L2": "half1"}
    out = tmp_path / "out" / "src"
    out.parent.mkdir()
    split = r.stream_lanes(lanes, RULES, challenge_genes=CHALLENGE, groups=groups, group_out=out)
    assert sorted(split.group_pbs) == ["half0", "half1"]
    assert split.gene_pb.labels == whole.gene_pb.labels
    assert np.array_equal(split.gene_pb.count_sum, whole.gene_pb.count_sum)
    assert np.allclose(split.gene_pb.cpm_sq_sum, whole.gene_pb.cpm_sq_sum, rtol=1e-12)
    genes = whole.gene_pb.genes.tolist()
    label = ["TGT1", "TGT1", "TGT2", "OLDX", "non-targeting", "non-targeting"]
    half1 = PseudobulkSums.load(split.group_pbs["half1"])
    _assert_matches(half1, _brute([dense[0], dense[2]], genes, lambda g: label[g]))
    assert half1.sources == ["L0", "L2"]
    with pytest.raises(ValueError, match="group_out"):
        r.stream_lanes(lanes, RULES, challenge_genes=CHALLENGE, groups=groups)


def test_the_cell_table_keeps_every_barcode_and_the_sidecar_counts_them(three_lanes):
    dense, lanes = three_lanes
    res = r.stream_lanes(lanes, RULES, challenge_genes=CHALLENGE,
                         target_aliases={"OLDX": "NEWX"})
    cells = res.cells
    assert len(cells) == sum(len(d) for d in dense)
    assert cells["barcode"].is_unique and set(cells["lane"].astype(str)) == {"L0", "L1", "L2"}
    kept = cells[cells["kept"]]
    assert int(res.gene_pb.n_cells.sum()) == len(kept) == res.qc.cells_seen
    assert res.qc.cells_dropped_filter == int((~cells["kept"]).sum())
    assert res.qc.batches == ["L0", "L1", "L2"] and res.qc.batch_cells.sum() == len(kept)
    assert res.qc.labels == ["OLDX", "TGT1", "TGT2", "non-targeting"]
    assert dict(zip(res.qc.guide_labels, res.qc.guide_cells.tolist(), strict=True)) == (
        kept.assign(g=np.where(kept["target_gene"].astype(str) == "non-targeting",
                               "non-targeting", kept["top_guide"].astype(str)))
        .groupby("g").size().to_dict())
    # each kept cell's count of its own target gene, straight from the dense rows
    flat = np.vstack(dense)
    own_col = {"TGT1": 7, "TGT2": 8, "OLDX": 9}
    for i in kept.index[:40]:
        target = str(cells.at[i, "target_gene"])
        want = -1 if target == "non-targeting" else int(flat[i, own_col[target]])
        assert cells.at[i, "own_target_count"] == want
    # control cells x targets: the same columns, for the control arm, in table order
    ctrl = kept.index[kept["target_gene"].astype(str) == "non-targeting"].to_numpy()
    assert np.array_equal(res.control_cell_rows, ctrl)
    assert np.array_equal(res.control_targets.toarray(), flat[ctrl][:, [9, 7, 8]])
    assert res.ctx.target_has_gene.all()
    # per-gene detection over kept cells, on the emitted axis
    assert res.qc.gene_cells[0] == int((flat[kept.index, 0] > 0).sum())


def test_a_lane_on_another_feature_table_is_refused(tmp_path):
    rng = np.random.default_rng(3)
    a = _write_lane(tmp_path / "a.h5", _random_lane(rng, 20))
    other = [*GENES[:-1], ("ENSG9", "RENAMED")]
    b = _write_lane(tmp_path / "b.h5", _random_lane(rng, 20), genes=other)
    with pytest.raises(ValueError, match="feature names differ"):
        r.stream_lanes([("a", a), ("b", b)], RULES, challenge_genes=CHALLENGE)


def test_an_aggregate_with_no_control_cells_is_refused(tmp_path):
    dense = np.vstack([_cell([n, 1, 1, 0, 0, 0, 0, 0, 0, 0], guide=0, umi=50)
                       for n in (1, 2, 3, 4)])
    lane = _write_lane(tmp_path / "a.h5", dense)
    with pytest.raises(ValueError, match="control label 'non-targeting' accumulated zero"):
        r.stream_lanes([("a", lane)], RULES, challenge_genes=CHALLENGE)


def test_groups_must_name_every_lane_exactly_once(tmp_path):
    path = tmp_path / "g.tsv"
    path.write_text("# rule: by parity\nL0\thalf0\nL1\thalf1\n")
    assert r.read_groups(path, ["L0", "L1"]) == {"L0": "half0", "L1": "half1"}
    with pytest.raises(ValueError, match="have no group"):
        r.read_groups(path, ["L0", "L1", "L2"])
    path.write_text("L0\thalf0\nL0\thalf1\n")
    with pytest.raises(ValueError, match="listed twice"):
        r.read_groups(path, ["L0"])
    path.write_text("L0\thalf 0\n")
    with pytest.raises(ValueError, match="file name"):
        r.read_groups(path, ["L0"])
    for reserved in ("guide", "qc", "control_targets"):
        path.write_text(f"L0\t{reserved}\n")
        with pytest.raises(ValueError, match="uses for something else"):
            r.read_groups(path, ["L0"])
    path.write_text("L0 half0\n")
    with pytest.raises(ValueError, match="line 1 is not lane<TAB>group"):
        r.read_groups(path, ["L0"])


# ---------------------------------------------------------------- end to end --


def test_the_cli_writes_every_artifact_and_its_lineage(three_lanes, tmp_path, capsys):
    _dense, lanes = three_lanes
    root = tmp_path / "root"
    dest = root / "external" / "geo-TEST"
    dest.mkdir(parents=True)
    selected = []
    for name, path in lanes:
        target = dest / f"{name}.h5"
        target.write_bytes(path.read_bytes())
        selected.append({"name": f"{name}.h5", "size_bytes": target.stat().st_size})
    (dest / "PROVENANCE.json").write_text(json.dumps({"selected": selected}))
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(yaml.safe_dump({"datasets": [{
        "name": "tiny", "host": "https", "record": "https://x", "budget_gb": 1,
        "dest": "external/geo-TEST",
        "files": [{"name": s["name"], "spec": SPEC} for s in selected]}]}))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "gene_names.csv").write_text("gene_name\n" + "\n".join(CHALLENGE) + "\n")
    challenge = tmp_path / "challenge.yaml"
    challenge.write_text(yaml.safe_dump({
        "data_dir": str(bundle), "gene_names_file": "gene_names.csv", "n_genes": len(CHALLENGE)}))
    groups = tmp_path / "groups.tsv"
    groups.write_text("L0.h5\tday0\nL1.h5\tday1\nL2.h5\tday0\n")
    aliases = tmp_path / "aliases.tsv"
    aliases.write_text("target\treference_symbol\tensembl_id\nOLDX\tNEWX\tENSG9\nGONE\t\t\n")
    axis_aliases = tmp_path / "axis_aliases.tsv"
    axis_aliases.write_text("axis_symbol\treference_symbol\nMISSING\tBBB\nNOWHERE\t\n")
    out = root / "derived" / "tiny" / "h1_tiny"

    assert r.main(["--dataset", "tiny", "--config", str(cfg), "--challenge-config",
                   str(challenge), "--root", str(root), "--guide-level", "--groups",
                   str(groups), "--target-aliases", str(aliases), "--axis-aliases",
                   str(axis_aliases), "--out", str(out)]) == 0

    for suffix in (".npz", ".guide.npz", ".day0.npz", ".day1.npz", ".qc.npz", ".cells.parquet",
                   ".control_targets.npz", ".lanes.tsv"):
        assert out.with_name(out.name + suffix).exists(), suffix
    pb = PseudobulkSums.load(out.with_name("h1_tiny.npz"))
    # the caller's table widens the reverse read: MISSING is fed by the feature named BBB
    assert pb.genes.tolist() == ["AAA", "AARS", "DUP", "MISSING", "CCC", "TGT1", "TGT2", "MT-ND1"]

    lineage = json.loads((out.parent / "LINEAGE.json").read_text())
    assert sorted(lineage["entries"]) == ["tiny/h1_tiny", "tiny/h1_tiny.day0",
                                         "tiny/h1_tiny.day1", "tiny/h1_tiny.guide"]
    main_entry = lineage["entries"]["tiny/h1_tiny"]
    assert main_entry["accumulator"]["control_label"] == "non-targeting"
    assert main_entry["accumulator"]["gene_axis"] == "challenge-symbols"
    assert [a["axis_symbol"] for a in main_entry["coverage"]["alias_recovered"]] == [
        "AARS", "MISSING"]
    assert main_entry["coverage"]["unmapped_genes"] == []
    assert main_entry["accumulator"]["rules"]["guide_target_regex"] == SPEC["guide_target_regex"]
    assert main_entry["accumulator"]["rules"]["min_genes"] == 3
    # three targets cannot support a median, and the reading says so instead of passing
    assert main_entry["on_target"]["status"] == "not_applicable"
    for key, path in (("groups_file", groups), ("target_aliases_file", aliases),
                      ("axis_aliases_file", axis_aliases)):
        for entry in lineage["entries"].values():
            assert entry[key] == {"path": str(path),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert "groups_file" not in lineage["entries"]["tiny/h1_tiny.day1"]["artifacts"]
    assert main_entry["coverage"]["targets_with_no_gene_feature"] == 0
    assert main_entry["coverage"]["cells_accumulated"] == int(pb.n_cells.sum())
    assert set(main_entry["source_sha256"]) == {"L0.h5", "L1.h5", "L2.h5"}
    assert set(lineage["entries"]["tiny/h1_tiny.day1"]["source_sha256"]) == {"L1.h5"}
    assert main_entry["groups"] == {"day0": "h1_tiny.day0.npz", "day1": "h1_tiny.day1.npz"}

    saved = np.load(out.with_name("h1_tiny.control_targets.npz"), allow_pickle=True)
    assert saved["targets"].tolist() == ["OLDX", "TGT1", "TGT2"]
    assert tuple(saved["shape"]) == (len(saved["cell_row"]), 3)
    assert '"cells":' in capsys.readouterr().out


def test_the_cli_refuses_a_lane_that_is_not_the_recorded_size(three_lanes, tmp_path):
    _, lanes = three_lanes
    root = tmp_path / "root"
    dest = root / "external" / "geo-TEST"
    dest.mkdir(parents=True)
    (dest / "L0.h5").write_bytes(lanes[0][1].read_bytes())
    (dest / "PROVENANCE.json").write_text(
        json.dumps({"selected": [{"name": "L0.h5", "size_bytes": 1}]}))
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(yaml.safe_dump({"datasets": [{
        "name": "tiny", "host": "https", "record": "https://x", "budget_gb": 1,
        "dest": "external/geo-TEST", "files": [{"name": "L0.h5", "spec": SPEC}]}]}))
    with pytest.raises(SystemExit, match="not the recorded size"):
        r.main(["--dataset", "tiny", "--config", str(cfg), "--root", str(root),
                "--all-genes", "--out", str(root / "derived" / "x")])


# ------------------------------------------------- what is SAVED, read back --
#
# Everything above checks objects in memory. The tests below run the command line once on
# lanes that carry the edge cases (guide counts exactly on 2, 5 and 10, low-gene cells, a
# count above 127) and read every saved file back, because a column that is written wrong
# and never read by a test is a column nobody checked.

LAB = ["TGT1", "TGT1", "TGT2", "OLDX", "non-targeting", "non-targeting"]
AXIS = {"AAA": [0], "AARS": [5], "DUP": [3, 4], "CCC": [2], "TGT1": [7], "TGT2": [8], "MT-ND1": [6]}
NAMES = ["L0.h5", "L1.h5", "L2.h5"]
GROUP = {"L0.h5": "day0", "L1.h5": "day1", "L2.h5": "day0"}


def _lane(rng, n):
    ge = lambda: rng.integers(0, 6, size=N_GE)
    edge = [_cell(ge(), guide=0, umi=40, extra={1: 2, 2: 5, 3: 10}),      # guides ON 2, 5 and 10
            _cell(ge(), guide=4, umi=22, extra={5: 10, 0: 1}),
            _cell([1, 1, 1, 0, 0, 0, 2, 300, 0, 0], guide=5, umi=30),     # a control cell, TGT1 = 300
            _cell([3, 1, 0, 0, 0, 0, 0, 0, 0, 0], guide=0, umi=50),            # two genes: low
            _cell([0, 2, 0, 0, 0, 0, 0, 0, 0, 0])]                             # low, and no guide
    return np.vstack([_random_lane(rng, n), *edge])


def _fixture(root, dense, *, files=None, prov=True):
    dest = root / "external" / "geo-TEST"
    dest.mkdir(parents=True)
    sel = []
    for nm, d in zip(NAMES, dense, strict=True):
        _write_lane(dest / nm, d, [f"{nm}BC{j}-1" for j in range(len(d))])
        sel.append({"name": nm, "size_bytes": (dest / nm).stat().st_size})
    if prov:
        (dest / "PROVENANCE.json").write_text(json.dumps({"selected": sel}))
    cfg = root / "datasets.yaml"
    cfg.write_text(yaml.safe_dump({"datasets": [{
        "name": "tiny", "host": "https", "record": "https://x", "budget_gb": 1, "dest": "external/geo-TEST",
        "files": files or [{"name": n, "spec": SPEC} for n in NAMES]}]}))
    bundle = root / "bundle"
    bundle.mkdir()
    (bundle / "gene_names.csv").write_text("gene_name\n" + "\n".join(CHALLENGE) + "\n")
    ch = root / "challenge.yaml"
    ch.write_text(yaml.safe_dump({"data_dir": str(bundle), "gene_names_file": "gene_names.csv",
                                  "n_genes": len(CHALLENGE)}))
    return dest, ["--dataset", "tiny", "--config", str(cfg), "--challenge-config", str(ch), "--root", str(root)]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("root")
    rng = np.random.default_rng(11)
    dense = [_lane(rng, n) for n in (60, 45, 80)]
    dest, base = _fixture(root, dense)
    groups = root / "groups.tsv"
    groups.write_text("".join(f"{n}\t{g}\n" for n, g in GROUP.items()))
    aliases = root / "aliases.tsv"
    aliases.write_text("target\treference_symbol\nOLDX\tNEWX\n")
    out = root / "derived" / "tiny" / "h1_tiny"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r.main([*base, "--guide-level", "--groups", str(groups), "--target-aliases", str(aliases),
                "--out", str(out)])
    cells = pd.read_parquet(out.with_name("h1_tiny.cells.parquet"))
    order = list(dict.fromkeys(cells["lane"].astype(str)))
    flat = np.vstack([dense[NAMES.index(n)] for n in order])
    return {"dense": dense, "dest": dest, "base": base, "out": out, "stdout": buf.getvalue(),
            "cells": cells, "ge": flat[:, :N_GE], "gd": flat[:, N_GE:], "order": order}


def _truth(ge, gd):
    n_genes, called = (ge > 0).sum(1), (gd >= 20).sum(1)
    kept = (n_genes >= 3) & (called == 1)
    target = np.array([LAB[i] if k else "" for i, k in zip(gd.argmax(1), kept, strict=True)], dtype=object)
    pct = np.where(ge.sum(1) > 0, 100 * ge[:, 6] / np.maximum(ge.sum(1), 1), 0.0)
    return n_genes, called, kept, target, pct


def test_every_column_of_the_saved_cell_table_matches_the_dense_rows(run):
    cells, ge, gd = run["cells"], run["ge"], run["gd"]
    n_genes, _called, kept, target, pct = _truth(ge, gd)
    assert len(cells) == sum(len(d) for d in run["dense"])                      # KEPT OR NOT
    assert cells["barcode"].tolist() == [f"{n}BC{j}-1" for n in run["order"]
                                         for j in range(len(run["dense"][NAMES.index(n)]))]
    for col in ("lane", "top_guide", "drop_reason", "target_gene"):
        assert str(cells[col].dtype) == "category", col
    assert np.array_equal(cells["n_genes"], n_genes)
    assert np.array_equal(cells["total_umi"], ge.sum(1))
    assert np.allclose(cells["pct_mt"], pct) and pct.max() > 1.0                # a percent, of MT- genes
    assert np.array_equal(cells["guide_umi"], gd.sum(1))
    for k in (2, 5, 10):                                                        # inclusive: counts ON k
        assert (gd == k).any() and np.array_equal(cells[f"n_guides_ge{k}"], (gd >= k).sum(1)), k
    assert np.array_equal(cells["kept"], kept)
    libsize = np.where(kept, sum(ge[:, c].sum(1) for c in AXIS.values()), 0)
    assert np.array_equal(cells["libsize"], libsize) and (libsize[kept] != ge.sum(1)[kept]).any()
    assert np.array_equal(cells["target_gene"].astype(str).to_numpy(), target)


def test_every_member_of_the_sidecar_is_a_count_over_the_cell_table(run):
    ge, gd = run["ge"], run["gd"]
    _, _, kept, target, pct = _truth(ge, gd)
    qc = np.load(run["out"].with_name("h1_tiny.qc.npz"), allow_pickle=True)
    labels, batches = [str(x) for x in qc["labels"]], [str(x) for x in qc["batches"]]
    lane = run["cells"]["lane"].astype(str).to_numpy()
    assert np.array_equal(qc["n_cells"], [int((target == lab).sum()) for lab in labels])
    assert np.allclose(qc["pct_mt_sum"], [pct[target == lab].sum() for lab in labels])
    assert np.allclose(qc["pct_mt_sq_sum"], [(pct[target == lab] ** 2).sum() for lab in labels])
    assert np.allclose(qc["total_counts_sum"], [ge.sum(1)[target == lab].sum() for lab in labels])
    assert np.array_equal(qc["batch_cells"], [[int(((target == lab) & (lane == b)).sum())
                                               for b in batches] for lab in labels])
    genes = [str(g) for g in PseudobulkSums.load(run["out"].with_name("h1_tiny.npz")).genes]
    assert np.array_equal(qc["gene_cells"], [int((ge[kept][:, AXIS[g]].sum(1) > 0).sum()) for g in genes])


def test_the_lane_table_and_the_lineage_carry_the_counts_and_the_digest_of_the_bytes(run):
    tsv = pd.read_csv(run["out"].with_name("h1_tiny.lanes.tsv"), sep="\t").set_index("lane")
    total = {"barcodes": 0, "kept": 0, "low_genes": 0, "no_guide": 0, "multi_guide": 0}
    for nm, d in zip(NAMES, run["dense"], strict=True):
        ge, gd = d[:, :N_GE], d[:, N_GE:]
        n_genes, called, kept, target, _ = _truth(ge, gd)
        want = {"group": GROUP[nm], "bytes": (run["dest"] / nm).stat().st_size,
                "sha256": hashlib.sha256((run["dest"] / nm).read_bytes()).hexdigest(),
                "barcodes": len(d), "kept": int(kept.sum()), "low_genes": int((n_genes < 3).sum()),
                "no_guide": int(((n_genes >= 3) & (called == 0)).sum()),
                "multi_guide": int(((n_genes >= 3) & (called > 1)).sum()), "zero_libsize": 0,
                "control_cells": int((target == "non-targeting").sum()),
                "targets_seen": len(set(target[kept]) - {"non-targeting"}),
                "median_umi_kept": float(np.median(ge.sum(1)[kept])),
                "median_genes_kept": float(np.median(n_genes[kept]))}
        assert {c: tsv.loc[nm, c] for c in want} == want, nm
        for k in total:
            total[k] += want[k]
    assert total["low_genes"] and total["no_guide"] != total["multi_guide"]
    entries = json.loads((run["out"].parent / "LINEAGE.json").read_text())["entries"]
    cov = entries["tiny/h1_tiny"]["coverage"]
    assert (cov["barcodes_read"], cov["cells_accumulated"], cov["cells_dropped_low_genes"],
            cov["cells_dropped_no_guide"], cov["cells_dropped_multi_guide"]) == (
        total["barcodes"], total["kept"], total["low_genes"], total["no_guide"], total["multi_guide"])
    assert entries["tiny/h1_tiny.day1"]["coverage"]["cells_accumulated"] == int(tsv.loc["L1.h5", "kept"])
    assert entries["tiny/h1_tiny"]["source_sha256"]["L1.h5"] == tsv.loc["L1.h5", "sha256"]
    assert '"on_target": {"n_self_measurable"' in run["stdout"] and '"status"' in run["stdout"]


def test_a_groups_lineage_entry_counts_its_own_lanes(run):
    """A group's entry counts its own lanes: the first version reported the whole run's
    barcodes and drops beside the group's own cells."""
    tsv = pd.read_csv(run["out"].with_name("h1_tiny.lanes.tsv"), sep="\t").set_index("lane")
    entries = json.loads((run["out"].parent / "LINEAGE.json").read_text())["entries"]
    cov = entries["tiny/h1_tiny.day1"]["coverage"]
    assert cov["lanes"] == 1
    for entry in entries.values():
        c = entry["coverage"]
        if entry["accumulator"]["keyed_by"] == "target_gene":
            assert c["barcodes_read"] == c["cells_accumulated"] + sum(
                c[f"cells_dropped_{why}"] for why in ("low_genes", "no_guide", "multi_guide",
                                                      "zero_libsize"))
    assert cov["barcodes_read"] == int(tsv.loc["L1.h5", "barcodes"])
    assert cov["cells_dropped_no_guide"] == int(tsv.loc["L1.h5", "no_guide"])


def test_the_saved_guide_sums_and_control_matrix_are_read_back(run):
    out, ge = run["out"], run["ge"]
    guide = [*GUIDES[:4], "non-targeting", "non-targeting"]
    pb = PseudobulkSums.load(out.with_name("h1_tiny.guide.npz"))
    _assert_matches(pb, _brute(run["dense"], [str(g) for g in pb.genes], lambda i: guide[i]))
    ct = np.load(out.with_name("h1_tiny.control_targets.npz"), allow_pickle=True)
    m = sp.csr_matrix((ct["data"], ct["indices"], ct["indptr"]), shape=tuple(ct["shape"])).toarray()
    rows = np.flatnonzero((run["cells"]["target_gene"].astype(str) == "non-targeting").to_numpy())
    assert np.array_equal(ct["cell_row"], rows)                    # rows of the SAVED parquet
    assert np.array_equal(m, ge[rows][:, [9, 7, 8]]) and m.max() == 300
    assert ct["target_has_gene"].tolist() == [True, True, True]
    lib = r.build_guide_library(np.asarray(GUIDES, dtype=object), RULES)
    whole, col = PseudobulkSums.load(out.with_name("h1_tiny.npz")), r.collapse_to_genes(pb, lib, "non-targeting")
    assert np.allclose(col.cpm_sq_sum, whole.cpm_sq_sum, rtol=1e-12)
    assert np.allclose(col.libsize_sum, whole.libsize_sum)


def test_a_kept_cell_with_nothing_on_the_emitted_axis_is_dropped_as_zero_libsize(tmp_path):
    dense = np.vstack([_cell([2, 3, 0, 0, 0, 0, 0, 0, 0, 5], guide=4, umi=30),   # control, AAA on axis
                       _cell([0, 3, 1, 0, 0, 0, 0, 0, 0, 5], guide=0, umi=30),   # 3 genes, none on axis
                       _cell([1, 3, 1, 0, 0, 0, 0, 4, 0, 5], guide=0, umi=30)])
    path = _write_lane(tmp_path / "z.h5", dense)
    res = r.stream_lanes([("z", path)], RULES, challenge_genes=["AAA", "TGT1"])
    assert res.cells["drop_reason"].astype(str).tolist() == ["", "zero_libsize", ""]
    assert res.cells["target_gene"].astype(str).tolist() == ["non-targeting", "", "TGT1"]
    assert res.gene_pb.n_cells.tolist() == [1, 1] and np.isfinite(res.gene_pb.cpm_sum).all()
    assert (res.qc.cells_dropped_zero, res.qc.cells_dropped_filter) == (1, 0)
    assert res.lanes["zero_libsize"].tolist() == [1] and res.control_cell_rows.tolist() == [0]


def test_a_target_with_no_gene_feature_reads_minus_one_not_zero(tmp_path):
    rng = np.random.default_rng(2)
    path = _write_lane(tmp_path / "a.h5", _random_lane(rng, 80))
    res = r.stream_lanes([("a", path)], RULES, challenge_genes=CHALLENGE)     # no alias for OLDX
    kept = res.cells[res.cells["kept"]]
    oldx = kept[kept["target_gene"].astype(str) == "OLDX"]
    assert len(oldx) and set(oldx["own_target_count"]) == {-1}
    assert res.ctx.target_has_gene.tolist() == [False, True, True]


def test_own_gene_lookup_order_exact_symbol_then_the_callers_table_then_the_alias_table(tmp_path):
    names = np.asarray([n for _, n in GENES], dtype=object)
    ones = sp.csr_matrix(np.arange(1, 11)[None, :])
    proj, _ = r.own_gene_projector(names, ["TGT1"], aliases={"TGT1": "TGT2"})
    assert (ones @ proj).toarray().tolist() == [[8]]                 # the exact symbol wins
    guides = ["AARS_+_1.23-P1", "TGT1_+_100.23-P1P2", "non-targeting_00001", "non-targeting_00002"]
    dense = np.zeros((2, N_GE + 4), dtype=int)
    dense[:, :3] = 1
    lane = r.read_lane(_write_lane(tmp_path / "a.h5", dense, guides=guides))
    ctx = r.LaneContext.from_lane(lane, RULES, CHALLENGE)        # AARS -> AARS1 from gene_aliases
    assert ctx.target_has_gene.tolist() == [True, True]
    assert (sp.csr_matrix(np.arange(1, 11)[None, :]) @ ctx.own_proj).toarray().tolist() == [[6, 8]]
    ctx = r.LaneContext.from_lane(lane, RULES, CHALLENGE, target_aliases={"AARS": "NEWX"})
    assert (sp.csr_matrix(np.arange(1, 11)[None, :]) @ ctx.own_proj).toarray().tolist() == [[10, 8]]


def test_limit_files_reads_the_first_lanes_and_all_genes_widens_the_axis(run, tmp_path):
    out = tmp_path / "lim"
    with contextlib.redirect_stdout(io.StringIO()):
        r.main([*run["base"], "--limit-files", "2", "--out", str(out)])
    assert pd.read_csv(out.with_name("lim.lanes.tsv"), sep="\t")["lane"].tolist() == NAMES[:2]
    out = tmp_path / "allg"
    with contextlib.redirect_stdout(io.StringIO()):
        r.main([*run["base"], "--all-genes", "--out", str(out)])
    assert list(PseudobulkSums.load(out.with_name("allg.npz")).genes) == sorted({n for _, n in GENES})


def test_the_refusals_nothing_else_exercises(run, tmp_path):
    dense = _random_lane(np.random.default_rng(4), 12)
    path = _write_lane(tmp_path / "neg.h5", dense)
    with h5py.File(path, "r+") as f:
        f["matrix"]["data"][0] = -1
    with pytest.raises(ValueError, match="negative counts"):
        r.read_lane(path)
    with h5py.File(tmp_path / "nomatrix.h5", "w") as f:
        f["x"] = np.zeros(1)
    with pytest.raises(ValueError, match="no `matrix` group"):
        r.read_lane(tmp_path / "nomatrix.h5")
    flat = np.vstack([_cell([4, 4, 4, 0, 0, 0, 0, 0, 0, 0], guide=0, umi=50)] * 5)   # equal totals
    lane = r.read_lane(_write_lane(tmp_path / "flat.h5", flat))
    with pytest.raises(ValueError, match="do not read as raw counts"):
        r.process_lane(lane, _ctx(lane))
    with pytest.raises(ValueError, match="duplicate guide names"):
        r.build_guide_library(np.asarray([*GUIDES, GUIDES[0]], dtype=object), RULES)
    with pytest.raises(ValueError, match="the control label"):
        r.build_guide_library(np.asarray([*GUIDES, "non-targeting_+_1.23-P1"], dtype=object), RULES)
    a = r.read_lane(_write_lane(tmp_path / "a.h5", dense))
    ids = r.read_lane(_write_lane(tmp_path / "ids.h5", dense, genes=[("ENSGX", "AAA"), *GENES[1:]]))
    with pytest.raises(ValueError, match="feature ids differ"):
        r.process_lane(ids, _ctx(a))
    typed = _write_lane(tmp_path / "typed.h5", dense)
    with h5py.File(typed, "r+") as f:
        ft = f["matrix"]["features"]["feature_type"][:]
        ft[1] = b"Gene Expressiom"
        del f["matrix"]["features"]["feature_type"]
        f["matrix"]["features"]["feature_type"] = ft
    with pytest.raises(ValueError, match="feature types differ"):
        r.process_lane(r.read_lane(typed), _ctx(a))
    noguide = r.read_lane(_write_lane(tmp_path / "ng.h5", dense[:, :N_GE], guides=[]))
    with pytest.raises(ValueError, match="feature types present"):
        _ctx(noguide)
    al = tmp_path / "al.tsv"
    al.write_text("target\treference_symbol\nOLDX\tNEWX\nOLDX\tAAA\n")
    with pytest.raises(ValueError, match="listed twice"):
        r.read_target_aliases(al)
    g = tmp_path / "g.tsv"
    g.write_text("L0\th0\nL1\th1\n")
    with pytest.raises(ValueError, match="not in the run"):
        r.read_groups(g, ["L0"])
    sums = r.Sums(["a"], np.asarray(["g0"], dtype=object))
    with pytest.raises(KeyError, match="no row in the accumulator"):
        sums.add(np.asarray(["b"], dtype=object), sp.csr_matrix([[1.0]]), np.asarray([1.0]), "x")
    _, base = _fixture(tmp_path / "noprov", run["dense"], prov=False)
    with pytest.raises(SystemExit, match="no PROVENANCE.json"):
        r.main([*base, "--out", str(tmp_path / "o1")])
    _, base = _fixture(tmp_path / "absent", run["dense"],
                       files=[{"name": n, "spec": SPEC} for n in [*NAMES, "L3.h5"]])
    with pytest.raises(SystemExit, match="not in the recorded provenance"):
        r.main([*base, "--out", str(tmp_path / "o2")])
    _, base = _fixture(tmp_path / "specs", run["dense"],
                       files=[{"name": n, "spec": {**SPEC, "min_genes": 3 + i}} for i, n in enumerate(NAMES)])
    with pytest.raises(SystemExit, match="different specs"):
        r.main([*base, "--out", str(tmp_path / "o3")])


def test_a_group_that_keeps_no_cell_merges_and_keeps_its_lane_in_the_sources(tmp_path):
    """`merge` built its row index from an empty list as float64 and died after the pass."""
    rng = np.random.default_rng(5)
    good = _write_lane(tmp_path / "good.h5", _random_lane(rng, 40))
    dead = _write_lane(tmp_path / "dead.h5", np.vstack(
        [_cell([n, 1, 0, 0, 0, 0, 0, 0, 0, 0], guide=0, umi=50) for n in (1, 2, 3)]))
    out = tmp_path / "o"
    res = r.stream_lanes([("good", good), ("dead", dead)], RULES, challenge_genes=CHALLENGE,
                         groups={"good": "g1", "dead": "g0"}, group_out=out)
    alone = r.stream_lanes([("good", good)], RULES, challenge_genes=CHALLENGE)
    assert sorted(res.gene_pb.sources) == ["dead", "good"]
    assert np.array_equal(res.gene_pb.count_sum, alone.gene_pb.count_sum)
    assert PseudobulkSums.load(res.group_pbs["g0"]).labels == []


def test_stored_zeros_and_split_entries_are_canonicalised_before_anything_is_counted(tmp_path):
    """A file may store a zero, or one (cell, feature) count in two pieces; neither may
    count as a detected gene or change a guide call. Written entry by entry, because the
    dense fixture writer can store neither."""
    n_feat = N_GE + N_GUIDE
    cells = [  # (feature, count) pairs per cell, as stored
        [(0, 1), (1, 1), (2, 1), (3, 0), (N_GE + 0, 10), (N_GE + 0, 10)],  # 3 genes; guide 10+10
        [(0, 1), (1, 1), (2, 0), (7, 0), (N_GE + 0, 50)],                  # 2 genes: low
        [(2, 2), (0, 1), (0, 1), (1, 1), (N_GE + 1, 12), (N_GE + 1, 12), (N_GE + 4, 30)],  # two
        [(0, 3), (1, 1), (2, 1), (N_GE + 4, 30), (N_GE + 2, 0)],           # control; a stored 0
    ]
    data = np.array([c for cell in cells for _, c in cell], dtype=np.int32)
    indices = np.array([f for cell in cells for f, _ in cell], dtype=np.int64)
    indptr = np.cumsum([0, *[len(cell) for cell in cells]]).astype(np.int64)
    path = tmp_path / "raw.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g["data"], g["indices"], g["indptr"] = data, indices, indptr
        g["shape"] = np.asarray([n_feat, len(cells)], dtype=np.int32)
        g["barcodes"] = np.asarray([f"BC{i}-1" for i in range(len(cells))], dtype="S")
        fg = g.create_group("features")
        fg["id"] = np.asarray([i for i, _ in GENES] + GUIDES, dtype="S")
        fg["name"] = np.asarray([n for _, n in GENES] + GUIDES, dtype="S")
        fg["feature_type"] = np.asarray(
            ["Gene Expression"] * N_GE + ["CRISPR Guide Capture"] * N_GUIDE, dtype="S")
    lane = r.read_lane(path)
    c = r.process_lane(lane, _ctx(lane)).cells
    assert c["n_genes"].tolist() == [3, 2, 3, 3]
    assert c["total_umi"].tolist() == [3, 2, 5, 5]
    assert c["n_guides_detected"].tolist() == [1, 1, 2, 1]
    assert c["n_guides_called"].tolist() == [1, 1, 2, 1]
    assert c["top_guide_umi"].tolist() == [20, 50, 30, 30]
    assert c["drop_reason"].tolist() == ["", "low_genes", "multi_guide", ""]
    assert c["target_gene"].tolist() == ["TGT1", "", "", "non-targeting"]


def test_limit_files_below_one_is_refused(run, tmp_path):
    for bad in ("0", "-1"):
        with pytest.raises(SystemExit, match="at least 1"):
            r.main([*run["base"], "--limit-files", bad, "--out", str(tmp_path / "x")])


def test_a_lane_on_another_feature_table_stops_the_cli_before_the_pass(run, tmp_path):
    other = [*GENES[:-1], ("ENSG9", "RENAMED")]
    dest, base = _fixture(tmp_path / "mixed", run["dense"])
    _write_lane(dest / "L2.h5", run["dense"][2], genes=other)
    prov = json.loads((dest / "PROVENANCE.json").read_text())
    prov["selected"][2]["size_bytes"] = (dest / "L2.h5").stat().st_size
    (dest / "PROVENANCE.json").write_text(json.dumps(prov))
    out = tmp_path / "mixed_out" / "x"
    with pytest.raises(SystemExit, match=r"L2\.h5: feature names differ"):
        r.main([*base, "--out", str(out)])
    assert not out.parent.exists() or not list(out.parent.glob("x*.npz"))


def test_the_printed_fetch_plan_makes_the_directories_a_nested_name_needs(tmp_path, monkeypatch,
                                                                          capsys):
    from sidechain.ingest import fetch
    from sidechain.ingest.provenance import HostRecord, RemoteFile

    record = HostRecord(
        host="https", record_id="https://x/samples", api_url="u", title="t", license="unknown",
        retrieved="2026-10-09",
        files=(RemoteFile("GSM1/suppl/a.h5", 10, "http-last-modified:x", "https://x/a"),))
    monkeypatch.setitem(fetch.PROBES, "https", lambda base, names=None: record)
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(yaml.safe_dump({"datasets": [{
        "name": "tiny", "host": "https", "record": "https://x/samples", "budget_gb": 1,
        "license": "CC-BY-4.0", "license_override_source": "the publisher's page",
        "allow_missing_checksum": True, "dest": "external/geo-TEST",
        "files": [{"name": "GSM1/suppl/a.h5", "spec": SPEC}]}]}))
    assert fetch.main(["--dataset", "tiny", "--config", str(cfg), "--root", str(tmp_path)]) == 0
    assert "curl -sSL -C - --create-dirs -o GSM1/suppl/a.h5 'https://x/a'" in capsys.readouterr().out


def test_a_failed_on_target_control_is_recorded_and_is_not_a_clean_exit(run, tmp_path, monkeypatch):
    def fail(pb, control_label, **kwargs):
        raise ValueError("on-target knockdown check FAILED: median self log2FC is +0.300")

    monkeypatch.setattr(r.checks, "require_on_target_knockdown", fail)
    out = tmp_path / "bad" / "x"
    with contextlib.redirect_stdout(io.StringIO()):
        assert r.main([*run["base"], "--out", str(out)]) == 1
    entry = json.loads((out.parent / "LINEAGE.json").read_text())["entries"]["tiny/x"]
    assert entry["on_target"]["status"] == "FAILED" and "+0.300" in entry["on_target"]["detail"]
