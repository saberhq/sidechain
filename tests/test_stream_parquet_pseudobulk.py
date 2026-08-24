"""Tests for the long-format parquet streamer (X-Atlas/Orion).

The sums are checked against hand-computed numbers rather than against a second
implementation, because the failure this module can have is the expensive kind: a
per-perturbation aggregate that is quietly wrong is indistinguishable from one that is right
until it has been pooled into a submission, and rebuilding it costs 126 GB of network.

Every fixture below mirrors the real schema as measured on 2026-08-23:
`gene_token_id` and `gene_expression` are parallel LIST columns, `pass_guide_filter` is
int64 rather than bool, and counts are integers stored as float64.
"""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sidechain.data.stream_parquet_pseudobulk import (
    COLUMNS,
    CONTROL_LABEL,
    Keying,
    _columns_for,
    _discover_labels,
    build_gene_axis,
    multi_construct_labels,
    read_gene_names,
    stream_files,
    stream_keyings,
    write_lineage,
)
from sidechain.data.stream_pseudobulk import PseudobulkSums

# token -> symbol. Token 3 and 4 deliberately share a symbol (`DUP`), mirroring the 22
# real collisions among X-Atlas's 38,606 ids over 38,584 names.
GENE_MAP = pd.DataFrame({
    "ensembl_id": ["ENSG0", "ENSG1", "ENSG2", "ENSG3", "ENSG4"],
    "gene_name": ["AAA", "BBB", "CCC", "DUP", "DUP"],
    "gene_token_id": [0, 1, 2, 3, 4],
})


def _write(path, rows):
    """One parquet file in the real schema. `rows` are dicts of column -> value.

    `guide_target` defaults to one construct per target (`<TARGET>_c1`), which is the shape
    16,832 of X-Atlas's 18,330 targets actually have. A row that wants a second construct
    says so with `guide=`.
    """
    table = pa.table({
        "gene_token_id": pa.array([r["tokens"] for r in rows], type=pa.list_(pa.int64())),
        "gene_expression": pa.array([r["values"] for r in rows], type=pa.list_(pa.float64())),
        "gene_target": pa.array([r["target"] for r in rows], type=pa.string()),
        "guide_target": pa.array(
            [r.get("guide", f"{r['target']}_c1") for r in rows], type=pa.string()),
        "sample": pa.array([r.get("sample", "B1") for r in rows], type=pa.string()),
        "pass_guide_filter": pa.array([r.get("pass", 1) for r in rows], type=pa.int64()),
        "pct_counts_mt": pa.array([r.get("mt", 1.0) for r in rows], type=pa.float64()),
        "total_counts": pa.array(
            [r.get("total", float(sum(r["values"]))) for r in rows], type=pa.float64()),
    })
    pq.write_table(table, path)
    return str(path)


def _opener(tmp_path):
    def _open(name):
        return open(tmp_path / name, "rb")
    return _open


def _axis(challenge=("AAA", "BBB", "CCC", "MISSING")):
    return build_gene_axis(GENE_MAP, list(challenge))


# ------------------------------------------------------------- the gene axis --


def test_challenge_genes_the_source_lacks_are_recorded_not_silently_dropped():
    """"X-Atlas is blind to these 427 genes" is a fact about our coverage. Dropping it
    silently is how a blind spot becomes an unexplained score."""
    axis = _axis()
    assert list(axis.genes) == ["AAA", "BBB", "CCC"]
    assert axis.unmapped_challenge == ["MISSING"]


def test_the_emitted_axis_keeps_challenge_order():
    """`pooled_delta` remaps by symbol, so order is not load-bearing downstream -- but a
    stable, documented order makes two aggregates directly comparable."""
    axis = build_gene_axis(GENE_MAP, ["CCC", "AAA", "BBB"])
    assert list(axis.genes) == ["CCC", "AAA", "BBB"]


def test_two_tokens_sharing_a_symbol_are_summed_and_the_collision_is_recorded():
    axis = build_gene_axis(GENE_MAP, ["AAA", "DUP"])
    assert axis.collided_symbols == ["DUP"]
    assert axis.col_of_token[3] == axis.col_of_token[4] == 1


def test_all_genes_mode_emits_every_symbol_once():
    axis = build_gene_axis(GENE_MAP, None)
    assert list(axis.genes) == ["AAA", "BBB", "CCC", "DUP"]
    assert axis.unmapped_challenge == []


# ------------------------------------------------------------------ the sums --


def test_sums_are_arithmetically_right(tmp_path):
    """Two cells on one target, hand-computed.

    cell 1: AAA=2, BBB=8   -> libsize 10 -> cpm AAA 200,000  BBB 800,000
    cell 2: AAA=6, CCC=2   -> libsize  8 -> cpm AAA 750,000  CCC 250,000
    """
    _write(tmp_path / "f.parquet", [
        {"tokens": [0, 1], "values": [2.0, 8.0], "target": "TP53"},
        {"tokens": [0, 2], "values": [6.0, 2.0], "target": "TP53"},
    ])
    pb, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                          labels=["TP53"], progress=False)

    assert list(pb.genes) == ["AAA", "BBB", "CCC"]
    np.testing.assert_allclose(pb.count_sum[0], [8.0, 8.0, 2.0])
    np.testing.assert_allclose(pb.cpm_sum[0], [200_000 + 750_000, 800_000, 250_000])
    np.testing.assert_allclose(
        pb.cpm_sq_sum[0], [200_000**2 + 750_000**2, 800_000**2, 250_000**2])
    assert pb.n_cells[0] == 2
    np.testing.assert_allclose(pb.libsize_sum[0], 18.0)
    assert qc.cells_seen == 2


def test_mean_cpm_and_var_cpm_come_out_of_the_emitted_object(tmp_path):
    """The consumers call these two methods; the sums exist to serve them."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [1.0], "target": "TP53"},
        {"tokens": [0, 1], "values": [1.0, 1.0], "target": "TP53"},
    ])
    pb, _ = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                         labels=["TP53"], progress=False)
    # cell 1: AAA cpm 1e6; cell 2: AAA cpm 5e5 -> mean 7.5e5
    np.testing.assert_allclose(pb.mean_cpm()[0, 0], 750_000.0)
    np.testing.assert_allclose(pb.var_cpm()[0, 0], 250_000.0**2)


def test_genes_outside_the_emitted_axis_are_dropped_not_misplaced(tmp_path):
    """Token 3 is `DUP`, which is not on this challenge axis. It must vanish, not land in
    some other gene's column -- the silent-misalignment failure this project has already
    paid for once with gene_names.csv."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0, 3], "values": [5.0, 999.0], "target": "TP53"},
    ])
    pb, _ = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                         labels=["TP53"], progress=False)
    np.testing.assert_allclose(pb.count_sum[0], [5.0, 0.0, 0.0])
    np.testing.assert_allclose(pb.libsize_sum[0], 5.0)   # libsize is over the EMITTED axis


def test_a_token_beyond_the_map_does_not_crash(tmp_path):
    """A token id larger than the metadata table would index out of bounds."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0, 99], "values": [4.0, 7.0], "target": "TP53"},
    ])
    pb, _ = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                         labels=["TP53"], progress=False)
    np.testing.assert_allclose(pb.count_sum[0], [4.0, 0.0, 0.0])


# ------------------------------------------------------- the filter, and int64 --


def test_pass_guide_filter_is_compared_as_an_integer(tmp_path):
    """The trap: the column is int64, not bool. A cell with 0 must not be accumulated."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [10.0], "target": "TP53", "pass": 1},
        {"tokens": [0], "values": [99.0], "target": "TP53", "pass": 0},
    ])
    pb, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                          labels=["TP53"], progress=False)
    np.testing.assert_allclose(pb.count_sum[0], [10.0, 0.0, 0.0])
    assert pb.n_cells[0] == 1
    assert qc.cells_dropped_filter == 1


def test_a_cell_with_no_on_axis_counts_is_dropped_not_divided_by_zero(tmp_path):
    _write(tmp_path / "f.parquet", [
        {"tokens": [3], "values": [5.0], "target": "TP53"},   # DUP only: off-axis
        {"tokens": [0], "values": [7.0], "target": "TP53"},
    ])
    pb, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                          labels=["TP53"], progress=False)
    assert pb.n_cells[0] == 1
    assert qc.cells_dropped_zero == 1
    assert np.isfinite(pb.cpm_sum).all()


# ------------------------------------------------------------ the control arm --


def test_the_control_label_is_matched_exactly(tmp_path):
    """`Non-Targeting` is an equality test. A substring rule would sweep in anything
    containing it, which is the INTS1/"NT" bug in a different costume."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [1.0], "target": CONTROL_LABEL},
        {"tokens": [0], "values": [2.0], "target": "Non-Targeting-Control"},
        {"tokens": [0], "values": [3.0], "target": "TP53"},
    ])
    pb, _ = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                         labels=[CONTROL_LABEL, "TP53"], progress=False)
    assert pb.labels == [CONTROL_LABEL, "TP53"]
    assert pb.n_cells[pb.labels.index(CONTROL_LABEL)] == 1   # NOT 2
    assert pb.n_cells[pb.labels.index("TP53")] == 1


def test_labels_outside_the_keep_list_are_ignored(tmp_path):
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [1.0], "target": "TP53"},
        {"tokens": [0], "values": [1.0], "target": "NOT_IN_PANEL"},
    ])
    pb, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                          labels=["TP53"], progress=False)
    assert pb.n_cells.sum() == 1
    assert qc.cells_seen == 1


# ------------------------------------------------ captured at ingest or lost --


def test_pct_counts_mt_mean_and_spread_survive_the_stream(tmp_path):
    """Free now, another 126 GB later. A perturbation whose delta is a dying-cell
    signature is the thing this is kept to detect."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [1.0], "target": "TP53", "mt": 2.0},
        {"tokens": [0], "values": [2.0], "target": "TP53", "mt": 6.0},
    ])
    _, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                         labels=["TP53"], progress=False)
    np.testing.assert_allclose(qc.mean_pct_mt()[0], 4.0)
    np.testing.assert_allclose(qc.sd_pct_mt()[0], 2.0)


def test_per_batch_cell_counts_are_kept_although_the_sums_are_not(tmp_path):
    """The resolution decision in one test: sums pool across batches (1.7 cells per
    bucket is not a measurement), COUNTS stay per batch (~8 MB, keeps batch structure
    auditable)."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [1.0], "target": "TP53", "sample": "B1"},
        {"tokens": [0], "values": [2.0], "target": "TP53", "sample": "B2"},
        {"tokens": [0], "values": [3.0], "target": "TP53", "sample": "B2"},
    ])
    pb, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                          labels=["TP53"], progress=False)
    assert pb.n_cells[0] == 3                       # sums pooled
    assert qc.batches == ["B1", "B2"]
    np.testing.assert_array_equal(qc.batch_cells[0], [1, 2])   # counts split


def test_the_qc_sidecar_round_trips(tmp_path):
    _write(tmp_path / "f.parquet", [{"tokens": [0], "values": [1.0], "target": "TP53"}])
    _, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                         labels=["TP53"], progress=False)
    qc.save(tmp_path / "qc.npz")
    back = np.load(tmp_path / "qc.npz", allow_pickle=True)
    assert [str(x) for x in back["labels"]] == ["TP53"]
    assert int(back["cells_seen"]) == 1


# ------------------------------------------------- normalization, from the data --


def test_a_transformed_matrix_is_refused_rather_than_pseudobulked(tmp_path):
    """The card says raw UMI counts. `counts_state` asks the data instead -- that is the
    whole point of that function, and the expm1-on-counts bug is why."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0, 1], "values": [1.5, 2.7], "target": "TP53"},
    ])
    with pytest.raises(ValueError, match="not raw counts"):
        stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                     labels=["TP53"], progress=False)


# ------------------------------------------------------ the integration shape --


def test_the_emitted_object_is_exactly_what_the_scorer_consumes(tmp_path):
    """The constraint that makes this cheap: `pooled_delta` and `loco` take a
    `PseudobulkSums` and nothing else, so X-Atlas is a new --source and no model, emitter
    or scorer code changes. If this ever fails, the shape is wrong."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0, 1], "values": [4.0, 6.0], "target": CONTROL_LABEL},
        {"tokens": [0, 1], "values": [1.0, 8.0], "target": "TP53"},
    ])
    pb, _ = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                         labels=[CONTROL_LABEL, "TP53"], progress=False)
    assert isinstance(pb, PseudobulkSums)

    pb.save(tmp_path / "x.npz")
    back = PseudobulkSums.load(tmp_path / "x.npz")
    assert back.labels == pb.labels
    np.testing.assert_allclose(back.count_sum, pb.count_sum)

    from sidechain.submit.build import pooled_delta

    axis = np.array(["AAA", "BBB", "CCC", "MISSING"])
    delta = pooled_delta("TP53", [(back, CONTROL_LABEL)], axis)
    assert delta is not None and delta.shape == (4,)
    assert delta[3] == 0.0          # a gene the source never measured -> no change
    assert delta[0] < 0 < delta[1]  # AAA down, BBB up, as constructed


def test_two_files_accumulate_into_one_aggregate(tmp_path):
    _write(tmp_path / "a.parquet", [{"tokens": [0], "values": [3.0], "target": "TP53"}])
    _write(tmp_path / "b.parquet", [{"tokens": [0], "values": [4.0], "target": "TP53"}])
    pb, _ = stream_files(["a.parquet", "b.parquet"], opener=_opener(tmp_path),
                          axis=_axis(), labels=["TP53"], progress=False)
    np.testing.assert_allclose(pb.count_sum[0, 0], 7.0)
    assert pb.n_cells[0] == 2
    assert len(pb.sources) == 2


def test_batching_does_not_change_the_answer(tmp_path):
    """Resident memory is a tuning knob; the aggregate must not be."""
    rows = [{"tokens": [0, 1], "values": [float(i), float(i + 1)], "target": "TP53"}
            for i in range(1, 51)]
    _write(tmp_path / "f.parquet", rows)
    whole, _ = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                            labels=["TP53"], batch_rows=1000, progress=False)
    tiny, _ = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                           labels=["TP53"], batch_rows=7, progress=False)
    np.testing.assert_allclose(whole.count_sum, tiny.count_sum)
    np.testing.assert_allclose(whole.cpm_sq_sum, tiny.cpm_sq_sum)


# ------------------------------------------------------------------- lineage --


def test_lineage_records_the_resolution_that_costs_126_gb_to_re_answer(tmp_path):
    _write(tmp_path / "f.parquet", [{"tokens": [0], "values": [1.0], "target": "TP53"}])
    axis = _axis()
    pb, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=axis,
                          labels=["TP53"], progress=False)
    path = write_lineage(tmp_path / "derived", provenance=tmp_path / "P.json",
                         dataset="xatlas_orion", context="hct116", axis=axis, pb=pb, qc=qc,
                         scope="panel", artifacts={"pseudobulk": "x.npz"},
                         entry="hct116_panel")
    entry = json.loads(path.read_text())["entries"]["xatlas_orion/hct116_panel"]
    assert entry["accumulator"]["resolution"] == "per-perturbation"
    assert entry["accumulator"]["control_label"] == CONTROL_LABEL
    assert entry["accumulator"]["guide_filter"] == "pass_guide_filter == 1"
    assert entry["coverage"]["unmapped_genes"] == ["MISSING"]
    assert entry["derives_from"].endswith("P.json")
    assert entry["code_sha"]


def test_a_second_context_does_not_erase_the_first_lineage(tmp_path):
    """Two aggregates and a record for one of them is worse than no record: it looks
    complete. HCT116 and HEK293T land in the same derived directory."""
    _write(tmp_path / "f.parquet", [{"tokens": [0], "values": [1.0], "target": "TP53"}])
    axis = _axis()
    pb, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=axis,
                          labels=["TP53"], progress=False)
    common = {"provenance": tmp_path / "P.json", "dataset": "xatlas_orion", "axis": axis,
              "pb": pb, "qc": qc, "scope": "panel", "artifacts": {"pseudobulk": "x.npz"}}
    write_lineage(tmp_path / "d", context="hct116", entry="hct116_panel", **common)
    path = write_lineage(tmp_path / "d", context="hek293t", entry="hek293t_panel", **common)

    entries = json.loads(path.read_text())["entries"]
    assert sorted(entries) == ["xatlas_orion/hct116_panel", "xatlas_orion/hek293t_panel"]
    assert entries["xatlas_orion/hct116_panel"]["context"] == "hct116"
    assert entries["xatlas_orion/hek293t_panel"]["context"] == "hek293t"


# -------------------------------------------------------- the 2026 header trap --


def test_gene_names_header_trap_is_caught_by_the_count(tmp_path):
    """`header=None` on the 2026 file yields 18,534 rows whose first gene is the literal
    string `gene_name`, misaligning every gene after it. Nothing downstream would raise."""
    p = tmp_path / "gene_names.csv"
    p.write_text("gene_name\nTSPAN6\nTNMD\nDPM1\n")
    assert read_gene_names(p, expect=3) == ["TSPAN6", "TNMD", "DPM1"]
    with pytest.raises(ValueError, match="header trap"):
        read_gene_names(p, expect=4)


def test_total_counts_is_the_corpus_column_not_the_emitted_axis_sum(tmp_path):
    """These are different numbers and the sidecar promises the bigger one.

    `libsize_sum` sums the 18,106 genes we emit; `total_counts` is the cell's whole
    transcriptome across all 38,606. Recording the former under the latter's name would
    make the alternative CPM normalization cost another 126 GB to recover -- which is the
    one thing capturing it at ingest was meant to prevent.
    """
    _write(tmp_path / "f.parquet", [
        # 5 counts land on the emitted axis; the cell's real library size is 100.
        {"tokens": [0], "values": [5.0], "target": "TP53", "total": 100.0},
    ])
    pb, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                          labels=["TP53"], progress=False)
    np.testing.assert_allclose(pb.libsize_sum[0], 5.0)        # emitted axis
    np.testing.assert_allclose(qc.total_counts_sum[0], 100.0)  # whole transcriptome
    assert qc.total_counts_sum[0] != pb.libsize_sum[0]


# ------------------------------------- a label the corpus never perturbed --


def test_never_observed_labels_are_pruned_from_the_emitted_object(tmp_path):
    """`pooled_delta` decides whether a source votes with `target not in pb.labels`. A
    label present with zero cells passes that test and then votes "every gene silenced",
    because mean_cpm is 0/1 and log2fc becomes log2(1/(ctrl+1)).

    The h5ad streamer never had this problem -- it intersects the keep list with the labels
    actually in the file. This is that parity.
    """
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [5.0], "target": "TP53"},
        {"tokens": [0], "values": [9.0], "target": CONTROL_LABEL},
    ])
    pb, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                          labels=["TP53", "NEVER_SEEN", CONTROL_LABEL], progress=False)
    assert pb.labels == ["TP53", CONTROL_LABEL]
    assert (pb.n_cells > 0).all()
    # the sidecar keeps the zeros: that IS the coverage number
    assert qc.labels == ["TP53", "NEVER_SEEN", CONTROL_LABEL]
    assert list(qc.n_cells) == [1, 0, 1]


def test_a_pruned_source_abstains_instead_of_voting_wrongly(tmp_path):
    """End to end through the real consumer: the failure this prevents is a source
    confidently reporting a transcriptome-wide shutdown for a gene it never perturbed."""
    from sidechain.submit.build import pooled_delta

    _write(tmp_path / "f.parquet", [
        {"tokens": [0, 1], "values": [4.0, 6.0], "target": CONTROL_LABEL},
        {"tokens": [0, 1], "values": [1.0, 8.0], "target": "TP53"},
    ])
    pb, _ = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                         labels=[CONTROL_LABEL, "TP53", "NEVER_SEEN"], progress=False)
    axis = np.array(["AAA", "BBB", "CCC"])

    assert pooled_delta("TP53", [(pb, CONTROL_LABEL)], axis) is not None
    # abstains rather than returning a huge negative vector
    assert pooled_delta("NEVER_SEEN", [(pb, CONTROL_LABEL)], axis) is None


# ============================================================ review findings ==
# Each test below pins a defect a review found in this module. They are grouped
# because they share a property: every one of them failed SILENTLY, producing a
# plausible aggregate rather than an error.


def test_a_negative_gene_token_cannot_wrap_into_a_real_genes_column(tmp_path):
    """`tokens < len(col_of_token)` alone lets -1 index from the END of the lookup,
    landing a count silently in some unrelated gene. Gene misalignment is this
    project's most expensive recurring bug and it never announces itself."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0, -1], "values": [4.0, 999.0], "target": "TP53"},
        {"tokens": [0], "values": [7.0], "target": CONTROL_LABEL},
    ])
    pb, _ = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                         labels=["TP53", CONTROL_LABEL], progress=False)
    i = pb.labels.index("TP53")
    np.testing.assert_allclose(pb.count_sum[i], [4.0, 0.0, 0.0])   # 999 must vanish
    assert pb.count_sum.max() < 999


def test_the_raw_counts_guard_stays_armed_until_it_actually_runs(tmp_path):
    """It used to be disarmed by the FIRST batch regardless of whether the check ran.
    A first batch where every cell fails the guide filter returns early, so whether a
    transformed matrix was caught depended on --batch-rows and row order, not on the
    data. The module docstring claimed it ran 'on the first block of every file'."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [3.0], "target": "TP53", "pass": 0},   # dropped
        {"tokens": [0, 1], "values": [1.5, 2.7], "target": "TP53"},      # log1p-ish
    ])
    with pytest.raises(ValueError, match="not raw counts"):
        stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                     labels=["TP53"], batch_rows=1, progress=False)


def test_the_guard_survives_a_first_batch_carrying_no_kept_label(tmp_path):
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [3.0], "target": "NOT_IN_PANEL"},
        {"tokens": [0, 1], "values": [1.5, 2.7], "target": "TP53"},
    ])
    with pytest.raises(ValueError, match="not raw counts"):
        stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                     labels=["TP53"], batch_rows=1, progress=False)


def test_a_cp10k_matrix_that_lands_on_integers_is_still_caught(tmp_path):
    """The signal that needs ROW STRUCTURE. counts_state catches a normalized matrix by
    its constant row totals, and that signal is dead if every cell's values are handed
    over as one flat vector -- which is what long format naturally produces."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0, 1], "values": [2000.0, 8000.0], "target": "TP53"},
        {"tokens": [0, 1], "values": [4000.0, 6000.0], "target": "TP53"},
        {"tokens": [0, 1], "values": [5000.0, 5000.0], "target": "TP53"},
    ])
    with pytest.raises(ValueError, match="not raw counts"):
        stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                     labels=["TP53"], progress=False)


def test_a_control_arm_that_yielded_nothing_stops_the_run(tmp_path):
    """Without this the prune removes the control, the .npz and LINEAGE.json are still
    written, and the failure surfaces much later inside pooled_delta as
    `ValueError: 'Non-Targeting' is not in list` -- naming neither this corpus nor this
    run. Reproduces what checks.control_mask guarantees for an in-memory matrix."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [5.0], "target": "TP53"},
        {"tokens": [0], "values": [9.0], "target": "Non-Targeting-Control"},
    ])
    with pytest.raises(ValueError, match="accumulated zero cells"):
        stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                     labels=["TP53", CONTROL_LABEL], progress=False)


def test_the_control_guard_offers_the_near_miss_label(tmp_path):
    """The same hint checks.control_mask gives: if a near-miss label is the real
    control, declare it exactly rather than matching loosely."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [5.0], "target": "TP53"},
        {"tokens": [0], "values": [9.0], "target": "non-targeting"},
    ])
    with pytest.raises(ValueError, match="non-targeting"):
        stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                     labels=["TP53", CONTROL_LABEL], progress=False)


def test_lineage_keys_are_dataset_qualified(tmp_path):
    """Two datasets can share a derived directory; keying on the artifact name alone
    would let one erase the other's entry."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [1.0], "target": "TP53"},
        {"tokens": [0], "values": [4.0], "target": CONTROL_LABEL},
    ])
    axis = _axis()
    pb, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=axis,
                          labels=["TP53", CONTROL_LABEL], progress=False)
    common = {"provenance": tmp_path / "P.json", "axis": axis, "pb": pb, "qc": qc,
              "scope": "panel", "artifacts": {}, "context": "c1", "entry": "panel"}
    write_lineage(tmp_path / "d", dataset="xatlas_orion", **common)
    path = write_lineage(tmp_path / "d", dataset="feng_2026", **common)
    assert sorted(json.loads(path.read_text())["entries"]) == [
        "feng_2026/panel", "xatlas_orion/panel"]


def test_an_unrecognised_lineage_file_is_moved_aside_not_overwritten(tmp_path):
    """Losing lineage is worse than an extra file: the aggregate it described stays on
    disk looking trustworthy."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [1.0], "target": "TP53"},
        {"tokens": [0], "values": [4.0], "target": CONTROL_LABEL},
    ])
    axis = _axis()
    pb, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=axis,
                          labels=["TP53", CONTROL_LABEL], progress=False)
    d = tmp_path / "d"
    d.mkdir()
    (d / "LINEAGE.json").write_text('{"schema_version": 1, "something": "else"}')
    write_lineage(d, provenance=tmp_path / "P.json", dataset="x", context="c",
                  axis=axis, pb=pb, qc=qc, scope="panel", artifacts={}, entry="e")
    assert (d / "LINEAGE.json.bak").exists()
    assert json.loads((d / "LINEAGE.json.bak").read_text())["something"] == "else"


def test_lineage_records_the_axis_the_caller_asked_for(tmp_path):
    """A source covering every challenge gene must still be recorded as a
    challenge-restricted axis, not as the much larger all-symbols one."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [1.0], "target": "TP53"},
        {"tokens": [0], "values": [4.0], "target": CONTROL_LABEL},
    ])
    full = build_gene_axis(GENE_MAP, ["AAA", "BBB", "CCC"])   # nothing unmapped
    assert full.restricted is True
    pb, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=full,
                          labels=["TP53", CONTROL_LABEL], progress=False)
    path = write_lineage(tmp_path / "d", provenance=tmp_path / "P.json", dataset="x",
                         context="c", axis=full, pb=pb, qc=qc, scope="panel",
                         artifacts={}, entry="e")
    rec = json.loads(path.read_text())["entries"]["x/e"]
    assert rec["accumulator"]["gene_axis"] == "challenge-symbols"
    assert build_gene_axis(GENE_MAP, None).restricted is False


# ----------------------------------------------- the construct-keyed accumulator --
#
# `--guide-agreement` adds a second accumulator keyed by `guide_target` -- the dual-guide
# CONSTRUCT -- in the same pass. It exists to answer "do two independently delivered
# constructs against the same gene produce the same profile?", which prices what a
# single-construct estimate is worth. These tests hold the three properties that make the
# answer meaningful: the split is exact, the population is restricted to the targets that
# can answer it, and the control is not split.


def _keyings(labels, guide_labels=None):
    keyings = [Keying(name="", column="gene_target", labels=labels,
                      scope="test", resolution="per-perturbation")]
    if guide_labels is not None:
        keyings.append(Keying(name="guide", column="guide_target", labels=guide_labels,
                              scope="test", resolution="per-construct"))
    return keyings


def test_two_constructs_against_one_gene_split_exactly_into_that_genes_row(tmp_path):
    """The strong invariant: the construct rows must SUM to the gene row, cell for cell.

    If they do not, the two aggregates describe different populations and any concordance
    measured between them is measuring the bug instead of the biology.
    """
    _write(tmp_path / "f.parquet", [
        {"target": "AAA", "guide": "AAA_c1", "tokens": [0, 1], "values": [4.0, 6.0]},
        {"target": "AAA", "guide": "AAA_c1", "tokens": [0], "values": [2.0]},
        {"target": "AAA", "guide": "AAA_c2", "tokens": [1, 2], "values": [5.0, 1.0]},
        {"target": CONTROL_LABEL, "guide": "nt_1|nt_2", "tokens": [0], "values": [3.0]},
    ])
    (pb, _), (gpb, _) = stream_keyings(
        ["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
        keyings=_keyings(["AAA", CONTROL_LABEL], ["AAA_c1", "AAA_c2", CONTROL_LABEL]))

    gene_row = pb.count_sum[pb.labels.index("AAA")]
    c1 = gpb.count_sum[gpb.labels.index("AAA_c1")]
    c2 = gpb.count_sum[gpb.labels.index("AAA_c2")]
    np.testing.assert_allclose(c1 + c2, gene_row)
    np.testing.assert_allclose(c1, [6.0, 6.0, 0.0])
    np.testing.assert_allclose(c2, [0.0, 5.0, 1.0])
    assert pb.n_cells[pb.labels.index("AAA")] == 3
    assert gpb.n_cells[gpb.labels.index("AAA_c1")] == 2
    assert gpb.n_cells[gpb.labels.index("AAA_c2")] == 1


def test_the_control_is_pooled_under_the_construct_keying_not_split_by_it(tmp_path):
    """X-Atlas pairs its 1,026 non-targeting sgRNAs at random, so every control cell can
    carry a different construct string. Splitting on it would replace the one arm every
    delta is measured against with a cloud of one-cell labels."""
    _write(tmp_path / "f.parquet", [
        {"target": CONTROL_LABEL, "guide": "nt_1|nt_2", "tokens": [0], "values": [3.0]},
        {"target": CONTROL_LABEL, "guide": "nt_7|nt_9", "tokens": [0], "values": [5.0]},
        {"target": "AAA", "guide": "AAA_c1", "tokens": [1], "values": [1.0]},
        {"target": "AAA", "guide": "AAA_c2", "tokens": [1], "values": [1.0]},
    ])
    (_, _), (gpb, _) = stream_keyings(
        ["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
        keyings=_keyings(["AAA", CONTROL_LABEL], ["AAA_c1", "AAA_c2", CONTROL_LABEL]))

    assert "nt_1|nt_2" not in gpb.labels
    assert gpb.n_cells[gpb.labels.index(CONTROL_LABEL)] == 2
    np.testing.assert_allclose(gpb.count_sum[gpb.labels.index(CONTROL_LABEL)], [8.0, 0.0, 0.0])


def test_a_single_construct_targets_cells_are_left_out_of_the_construct_aggregate(tmp_path):
    """Its construct profile IS its gene profile, already accumulated next door. Carrying
    all ~20,890 constructs on the 38,584-gene axis would cost ~19 GB to duplicate them."""
    _write(tmp_path / "f.parquet", [
        {"target": "AAA", "guide": "AAA_c1", "tokens": [0], "values": [1.0]},
        {"target": "AAA", "guide": "AAA_c2", "tokens": [0], "values": [1.0]},
        {"target": "BBB", "guide": "BBB_c1", "tokens": [1], "values": [9.0]},
        {"target": CONTROL_LABEL, "tokens": [0], "values": [3.0]},
    ])
    (pb, _), (gpb, gqc) = stream_keyings(
        ["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
        keyings=_keyings(["AAA", "BBB", CONTROL_LABEL],
                         ["AAA_c1", "AAA_c2", CONTROL_LABEL]))

    assert "BBB" in pb.labels and pb.n_cells[pb.labels.index("BBB")] == 1
    assert not any(lab.startswith("BBB") for lab in gpb.labels)
    # The construct accumulator saw 3 cells: two AAA constructs and one control.
    assert gqc.cells_seen == 3


def test_one_pass_feeds_both_accumulators_rather_than_two_reads(tmp_path):
    """Two passes would read 126 GB twice and could not be compared cell for cell -- the
    second read is not guaranteed to land on the same revision."""
    reads = []
    real = _opener(tmp_path)

    def counting_opener(name):
        reads.append(name)
        return real(name)

    _write(tmp_path / "f.parquet", [
        {"target": "AAA", "guide": "AAA_c1", "tokens": [0], "values": [1.0]},
        {"target": "AAA", "guide": "AAA_c2", "tokens": [0], "values": [1.0]},
        {"target": CONTROL_LABEL, "tokens": [0], "values": [3.0]},
    ])
    stream_keyings(["f.parquet"], opener=counting_opener, axis=_axis(),
                   keyings=_keyings(["AAA", CONTROL_LABEL],
                                    ["AAA_c1", "AAA_c2", CONTROL_LABEL]))
    assert reads == ["f.parquet"]


def test_the_guide_filter_drops_the_same_cells_from_both_keyings(tmp_path):
    """`pass_guide_filter` is a property of the CELL, not of the keying, so the count is
    the same in both sidecars and neither aggregate contains a filtered cell."""
    _write(tmp_path / "f.parquet", [
        {"target": "AAA", "guide": "AAA_c1", "tokens": [0], "values": [7.0], "pass": 0},
        {"target": "AAA", "guide": "AAA_c2", "tokens": [0], "values": [1.0]},
        {"target": CONTROL_LABEL, "tokens": [0], "values": [3.0]},
    ])
    (pb, qc), (gpb, gqc) = stream_keyings(
        ["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
        keyings=_keyings(["AAA", CONTROL_LABEL], ["AAA_c1", "AAA_c2", CONTROL_LABEL]))

    assert qc.cells_dropped_filter == gqc.cells_dropped_filter == 1
    assert "AAA_c1" not in gpb.labels          # pruned: it accumulated zero cells
    np.testing.assert_allclose(pb.count_sum[pb.labels.index("AAA")], [1.0, 0.0, 0.0])


def test_the_raw_counts_guard_still_fires_when_only_the_construct_keying_wants_the_cells(
        tmp_path):
    """The guard asks about `gene_expression`, which no keying changes -- so an early
    return from the perturbation accumulator must not disarm it."""
    _write(tmp_path / "f.parquet", [
        # cp10k rows: not raw counts. The perturbation keying keeps none of these labels.
        {"target": "ZZZ", "guide": "ZZZ_c1", "tokens": [0, 1], "values": [4000.0, 6000.0]},
        {"target": "ZZZ", "guide": "ZZZ_c2", "tokens": [0, 1], "values": [1000.0, 9000.0]},
    ])
    with pytest.raises(ValueError, match="not raw counts"):
        stream_keyings(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                       keyings=_keyings([CONTROL_LABEL], ["ZZZ_c1", "ZZZ_c2"]))


def test_the_guide_column_is_fetched_only_when_a_keying_keys_on_it(tmp_path):
    """126 GB of columnar parquet: a column nobody asked for is a column never sent."""
    plain = _keyings(["AAA"])
    assert "guide_target" not in _columns_for(plain)
    assert "guide_target" in _columns_for(_keyings(["AAA"], ["AAA_c1", "AAA_c2"]))
    # and the base list is unchanged by the call
    assert "guide_target" not in COLUMNS


def test_construct_discovery_reads_both_columns_and_leaves_the_control_out(tmp_path):
    """The control's construct strings are unbounded (random sgRNA pairs) and nothing keys
    on them; the targeting ones are a designed library of ~20,890."""
    _write(tmp_path / "f.parquet", [
        {"target": "AAA", "guide": "AAA_c1", "tokens": [0], "values": [1.0]},
        {"target": "AAA", "guide": "AAA_c2", "tokens": [0], "values": [1.0]},
        {"target": "BBB", "guide": "BBB_c1", "tokens": [0], "values": [1.0]},
        {"target": CONTROL_LABEL, "guide": "nt_1|nt_2", "tokens": [0], "values": [1.0]},
        {"target": CONTROL_LABEL, "guide": "nt_3|nt_4", "tokens": [0], "values": [1.0]},
    ])
    found, pairs = _discover_labels(["f.parquet"], _opener(tmp_path), with_constructs=True)

    assert found == {"AAA", "BBB", CONTROL_LABEL}
    assert pairs == {("AAA", "AAA_c1"), ("AAA", "AAA_c2"), ("BBB", "BBB_c1")}
    assert not any(t == CONTROL_LABEL for t, _ in pairs)


def test_construct_discovery_stays_off_the_wire_when_it_is_not_asked_for(tmp_path):
    _write(tmp_path / "f.parquet", [
        {"target": "AAA", "guide": "AAA_c1", "tokens": [0], "values": [1.0]},
    ])
    found, pairs = _discover_labels(["f.parquet"], _opener(tmp_path))
    assert found == {"AAA"} and pairs == set()


def test_only_targets_with_more_than_one_construct_become_construct_labels():
    pairs = {("AAA", "AAA_c1"), ("AAA", "AAA_c2"), ("BBB", "BBB_c1"),
             ("CCC", "CCC_c1"), ("CCC", "CCC_c2"), ("CCC", "CCC_c3")}
    labels, n_targets = multi_construct_labels(pairs)
    assert n_targets == 2
    assert labels == ["AAA_c1", "AAA_c2", "CCC_c1", "CCC_c2", "CCC_c3", CONTROL_LABEL]


def test_the_construct_set_is_a_refinement_of_the_labels_being_accumulated():
    """Otherwise the two aggregates in one directory would describe different populations
    that merely share a filename stem."""
    pairs = {("AAA", "AAA_c1"), ("AAA", "AAA_c2"), ("CCC", "CCC_c1"), ("CCC", "CCC_c2")}
    labels, n_targets = multi_construct_labels(pairs, keep={"AAA", CONTROL_LABEL})
    assert n_targets == 1
    assert labels == ["AAA_c1", "AAA_c2", CONTROL_LABEL]


def test_the_construct_aggregate_records_its_own_resolution_in_lineage(tmp_path):
    """An aggregate whose resolution is unknown cannot be trusted or extended -- and a
    reader must know whether `labels` holds gene symbols or `sgRNA1|sgRNA2` strings before
    loading a multi-GB array."""
    _write(tmp_path / "f.parquet", [
        {"target": "AAA", "guide": "AAA_c1", "tokens": [0], "values": [1.0]},
        {"target": "AAA", "guide": "AAA_c2", "tokens": [0], "values": [1.0]},
        {"target": CONTROL_LABEL, "tokens": [0], "values": [3.0]},
    ])
    axis = _axis()
    (pb, qc), (gpb, gqc) = stream_keyings(
        ["f.parquet"], opener=_opener(tmp_path), axis=axis,
        keyings=_keyings(["AAA", CONTROL_LABEL], ["AAA_c1", "AAA_c2", CONTROL_LABEL]))

    common = {"provenance": tmp_path / "PROVENANCE.json", "dataset": "xatlas_orion",
              "context": "hct116", "axis": axis, "artifacts": {}}
    write_lineage(tmp_path, pb=pb, qc=qc, scope="all labels", entry="hct116_full", **common)
    path = write_lineage(tmp_path, pb=gpb, qc=gqc, scope="multi-construct targets only",
                         entry="hct116_full.guide", resolution="per-construct",
                         keyed_by="guide_target", **common)

    entries = json.loads(path.read_text())["entries"]
    assert set(entries) == {"xatlas_orion/hct116_full", "xatlas_orion/hct116_full.guide"}
    assert entries["xatlas_orion/hct116_full"]["accumulator"]["keyed_by"] == "gene_target"
    guide = entries["xatlas_orion/hct116_full.guide"]["accumulator"]
    assert guide["keyed_by"] == "guide_target"
    assert guide["resolution"] == "per-construct"


def test_labels_in_corpus_is_counted_under_each_keying_not_shared(tmp_path):
    """"18,294 targets" and "20,890 constructs" are different facts about the same corpus.
    Sharing one counter would report whichever accumulator ran last."""
    _write(tmp_path / "f.parquet", [
        {"target": "AAA", "guide": "AAA_c1", "tokens": [0], "values": [1.0]},
        {"target": "AAA", "guide": "AAA_c2", "tokens": [0], "values": [1.0]},
        {"target": "BBB", "guide": "BBB_c1", "tokens": [0], "values": [1.0]},
        {"target": CONTROL_LABEL, "guide": "nt_1|nt_2", "tokens": [0], "values": [3.0]},
    ])
    (_, qc), (_, gqc) = stream_keyings(
        ["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
        keyings=_keyings(["AAA", "BBB", CONTROL_LABEL],
                         ["AAA_c1", "AAA_c2", CONTROL_LABEL]))

    assert qc.labels_in_corpus == 3          # AAA, BBB, Non-Targeting
    assert gqc.labels_in_corpus == 4         # AAA_c1, AAA_c2, BBB_c1, Non-Targeting


def test_a_construct_keying_with_no_control_cells_stops_the_run(tmp_path):
    """Same guarantee as the perturbation keying, and the message says which keying."""
    _write(tmp_path / "f.parquet", [
        {"target": "AAA", "guide": "AAA_c1", "tokens": [0, 1], "values": [1.0, 4.0]},
        {"target": "AAA", "guide": "AAA_c2", "tokens": [0], "values": [7.0]},
    ])
    with pytest.raises(ValueError, match="keyed by guide_target"):
        stream_keyings(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                       keyings=[Keying(name="guide", column="guide_target",
                                       labels=["AAA_c1", "AAA_c2", CONTROL_LABEL],
                                       scope="test", resolution="per-construct")])


def test_stream_files_still_returns_one_pair(tmp_path):
    """The 30 callers that want the ordinary per-perturbation aggregate are untouched."""
    _write(tmp_path / "f.parquet", [
        {"target": "AAA", "tokens": [0], "values": [1.0]},
        {"target": CONTROL_LABEL, "tokens": [0], "values": [3.0]},
    ])
    pb, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                          labels=["AAA", CONTROL_LABEL])
    assert pb.labels == ["AAA", CONTROL_LABEL]
    assert qc.cells_seen == 2
