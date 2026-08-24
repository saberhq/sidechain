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
    CONTROL_LABEL,
    build_gene_axis,
    read_gene_names,
    stream_files,
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
    """One parquet file in the real schema. `rows` are dicts of column -> value."""
    table = pa.table({
        "gene_token_id": pa.array([r["tokens"] for r in rows], type=pa.list_(pa.int64())),
        "gene_expression": pa.array([r["values"] for r in rows], type=pa.list_(pa.float64())),
        "gene_target": pa.array([r["target"] for r in rows], type=pa.string()),
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
