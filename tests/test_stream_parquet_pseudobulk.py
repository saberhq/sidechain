"""`sidechain.data.stream_parquet_pseudobulk`: per-label sums from long-format parquet, checked by hand.

    .venv/bin/python -m pytest -q tests/test_stream_parquet_pseudobulk.py

Writes temporary files only. Notes: private/code-notes/tests/test_stream_parquet_pseudobulk.py.md
"""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp

from sidechain.data.stream_parquet_pseudobulk import (
    COLUMNS,
    CONTROL_LABEL,
    Keying,
    _columns_for,
    _discover_labels,
    _scatter_add,
    build_gene_axis,
    multi_construct_labels,
    read_gene_names,
    stream_files,
    stream_keyings,
    write_lineage,
)
from sidechain.data.stream_pseudobulk import PseudobulkSums

# token -> symbol; tokens 3 and 4 share the symbol `DUP`
GENE_MAP = pd.DataFrame({
    "ensembl_id": ["ENSG0", "ENSG1", "ENSG2", "ENSG3", "ENSG4"],
    "gene_name": ["AAA", "BBB", "CCC", "DUP", "DUP"],
    "gene_token_id": [0, 1, 2, 3, 4],
})


def _write(path, rows):
    """One parquet file; `rows` are dicts of column -> value, one construct per target unless `guide=`."""
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
    """A challenge gene the source lacks is recorded as unmapped."""
    axis = _axis()
    assert list(axis.genes) == ["AAA", "BBB", "CCC"]
    assert axis.unmapped_challenge == ["MISSING"]


def test_the_emitted_axis_keeps_challenge_order():
    """The emitted axis keeps the challenge order."""
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
    """`mean_cpm` and `var_cpm` come out of the emitted object."""
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
    """A token off the emitted axis is dropped, not written to another gene's column."""
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
    """`pass_guide_filter` is int64; a cell with 0 is not accumulated."""
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
    """The control label is matched by equality, not substring."""
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
    """The per-label mean and sd of `pct_counts_mt` are kept."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [1.0], "target": "TP53", "mt": 2.0},
        {"tokens": [0], "values": [2.0], "target": "TP53", "mt": 6.0},
    ])
    _, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                         labels=["TP53"], progress=False)
    np.testing.assert_allclose(qc.mean_pct_mt()[0], 4.0)
    np.testing.assert_allclose(qc.sd_pct_mt()[0], 2.0)


def test_per_batch_cell_counts_are_kept_although_the_sums_are_not(tmp_path):
    """Sums pool across batches; cell counts stay per batch."""
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
    """A matrix that does not read as raw counts is refused."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0, 1], "values": [1.5, 2.7], "target": "TP53"},
    ])
    with pytest.raises(ValueError, match="not raw counts"):
        stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                     labels=["TP53"], progress=False)


# ------------------------------------------------------ the integration shape --


def test_the_emitted_object_is_exactly_what_the_scorer_consumes(tmp_path):
    """The output is a `PseudobulkSums` that round-trips and that `pooled_delta` reads."""
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
    assert delta[3] == 0.0          # a gene the source lacks -> no change
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
    """`batch_rows` does not change the sums."""
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


def test_lineage_records_the_resolution(tmp_path):
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
    """A second context's lineage entry is added beside the first."""
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
    """A header row read as a gene is caught by the expected count."""
    p = tmp_path / "gene_names.csv"
    p.write_text("gene_name\nTSPAN6\nTNMD\nDPM1\n")
    assert read_gene_names(p, expect=3) == ["TSPAN6", "TNMD", "DPM1"]
    with pytest.raises(ValueError, match="header trap"):
        read_gene_names(p, expect=4)


def test_total_counts_is_the_corpus_column_not_the_emitted_axis_sum(tmp_path):
    """`total_counts` in the sidecar is the corpus column; `libsize_sum` is the emitted-axis sum."""
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
    """A kept label with zero cells is pruned from the sums and kept, at zero, in the sidecar."""
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
    """`pooled_delta` returns None for a label the source never perturbed."""
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


# ====================================================== tokens, guards, lineage ==


def test_a_negative_gene_token_cannot_wrap_into_a_real_genes_column(tmp_path):
    """A negative token is dropped, not indexed from the end of the lookup."""
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
    """The raw-counts guard still runs after a first batch whose cells all fail the guide filter."""
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
    """A matrix with constant row totals is refused even when every value is an integer."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0, 1], "values": [2000.0, 8000.0], "target": "TP53"},
        {"tokens": [0, 1], "values": [4000.0, 6000.0], "target": "TP53"},
        {"tokens": [0, 1], "values": [5000.0, 5000.0], "target": "TP53"},
    ])
    with pytest.raises(ValueError, match="not raw counts"):
        stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                     labels=["TP53"], progress=False)


def test_a_control_arm_that_yielded_nothing_stops_the_run(tmp_path):
    """A control label that accumulated no cells stops the run."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [5.0], "target": "TP53"},
        {"tokens": [0], "values": [9.0], "target": "Non-Targeting-Control"},
    ])
    with pytest.raises(ValueError, match="accumulated zero cells"):
        stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                     labels=["TP53", CONTROL_LABEL], progress=False)


def test_the_control_guard_offers_the_near_miss_label(tmp_path):
    """The refusal names a near-miss control label."""
    _write(tmp_path / "f.parquet", [
        {"tokens": [0], "values": [5.0], "target": "TP53"},
        {"tokens": [0], "values": [9.0], "target": "non-targeting"},
    ])
    with pytest.raises(ValueError, match="non-targeting"):
        stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                     labels=["TP53", CONTROL_LABEL], progress=False)


def test_lineage_keys_are_dataset_qualified(tmp_path):
    """Lineage entries are keyed `<dataset>/<entry>`."""
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
    """An unrecognised LINEAGE.json is moved to `.bak`, not overwritten."""
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
    """A challenge axis with nothing unmapped is still recorded as `challenge-symbols`."""
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
# `--guide-agreement` adds a second accumulator keyed by `guide_target` in the same pass.


def _keyings(labels, guide_labels=None):
    keyings = [Keying(name="", column="gene_target", labels=labels,
                      scope="test", resolution="per-perturbation")]
    if guide_labels is not None:
        keyings.append(Keying(name="guide", column="guide_target", labels=guide_labels,
                              scope="test", resolution="per-construct"))
    return keyings


def test_two_constructs_against_one_gene_split_exactly_into_that_genes_row(tmp_path):
    """The construct rows sum to the gene row, cell for cell."""
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
    """Control cells pool under the control label whatever their construct string."""
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
    """A target with one construct is left out of the construct sums."""
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
    """Both accumulators fill from one read of each file."""
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
    """`pass_guide_filter` drops the same cells from both keyings."""
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
    """The raw-counts guard fires when only the construct keying keeps the cells."""
    _write(tmp_path / "f.parquet", [
        # cp10k rows: not raw counts. The perturbation keying keeps none of these labels.
        {"target": "ZZZ", "guide": "ZZZ_c1", "tokens": [0, 1], "values": [4000.0, 6000.0]},
        {"target": "ZZZ", "guide": "ZZZ_c2", "tokens": [0, 1], "values": [1000.0, 9000.0]},
    ])
    with pytest.raises(ValueError, match="not raw counts"):
        stream_keyings(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                       keyings=_keyings([CONTROL_LABEL], ["ZZZ_c1", "ZZZ_c2"]))


def test_the_guide_column_is_fetched_only_when_a_keying_keys_on_it(tmp_path):
    """`guide_target` is read only when a keying keys on it."""
    plain = _keyings(["AAA"])
    assert "guide_target" not in _columns_for(plain)
    assert "guide_target" in _columns_for(_keyings(["AAA"], ["AAA_c1", "AAA_c2"]))
    # and the base list is unchanged by the call
    assert "guide_target" not in COLUMNS


def test_construct_discovery_reads_both_columns_and_leaves_the_control_out(tmp_path):
    """Discovery returns (target, construct) pairs and leaves the control's constructs out."""
    _write(tmp_path / "f.parquet", [
        {"target": "AAA", "guide": "AAA_c1", "tokens": [0], "values": [1.0]},
        {"target": "AAA", "guide": "AAA_c2", "tokens": [0], "values": [1.0]},
        {"target": "BBB", "guide": "BBB_c1", "tokens": [0], "values": [1.0]},
        {"target": CONTROL_LABEL, "guide": "nt_1|nt_2", "tokens": [0], "values": [1.0]},
        {"target": CONTROL_LABEL, "guide": "nt_3|nt_4", "tokens": [0], "values": [1.0]},
    ])
    found, pairs = _discover_labels(["f.parquet"], _opener(tmp_path),
                                    with_constructs=True, progress=False)

    assert found == {"AAA", "BBB", CONTROL_LABEL}
    assert pairs == {("AAA", "AAA_c1"), ("AAA", "AAA_c2"), ("BBB", "BBB_c1")}
    assert not any(t == CONTROL_LABEL for t, _ in pairs)


def test_construct_discovery_stays_off_the_wire_when_it_is_not_asked_for(tmp_path):
    _write(tmp_path / "f.parquet", [
        {"target": "AAA", "guide": "AAA_c1", "tokens": [0], "values": [1.0]},
    ])
    found, pairs = _discover_labels(["f.parquet"], _opener(tmp_path), progress=False)
    assert found == {"AAA"} and pairs == set()


def test_only_targets_with_more_than_one_construct_become_construct_labels():
    pairs = {("AAA", "AAA_c1"), ("AAA", "AAA_c2"), ("BBB", "BBB_c1"),
             ("CCC", "CCC_c1"), ("CCC", "CCC_c2"), ("CCC", "CCC_c3")}
    labels, n_targets = multi_construct_labels(pairs)
    assert n_targets == 2
    assert labels == ["AAA_c1", "AAA_c2", "CCC_c1", "CCC_c2", "CCC_c3", CONTROL_LABEL]


def test_the_construct_set_is_a_refinement_of_the_labels_being_accumulated():
    """`keep` restricts the construct labels to the kept targets."""
    pairs = {("AAA", "AAA_c1"), ("AAA", "AAA_c2"), ("CCC", "CCC_c1"), ("CCC", "CCC_c2")}
    labels, n_targets = multi_construct_labels(pairs, keep={"AAA", CONTROL_LABEL})
    assert n_targets == 1
    assert labels == ["AAA_c1", "AAA_c2", CONTROL_LABEL]


def test_the_construct_aggregate_records_its_own_resolution_in_lineage(tmp_path):
    """The construct aggregate's lineage entry records its resolution and key column."""
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
    """`labels_in_corpus` is counted per keying."""
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
    """A construct keying with no control cells stops the run and names the keying."""
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
    """`stream_files` returns one (sums, qc) pair."""
    _write(tmp_path / "f.parquet", [
        {"target": "AAA", "tokens": [0], "values": [1.0]},
        {"target": CONTROL_LABEL, "tokens": [0], "values": [3.0]},
    ])
    pb, qc = stream_files(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                          labels=["AAA", CONTROL_LABEL])
    assert pb.labels == ["AAA", CONTROL_LABEL]
    assert qc.cells_seen == 2


def test_the_sparse_scatter_is_bit_identical_to_the_dense_form():
    """`_scatter_add` is bit-identical to the dense form."""
    rng = np.random.default_rng(0)
    n_labels, n_genes, n_cells = 40, 500, 300
    codes = rng.integers(0, n_labels, n_cells)
    uniq, local = np.unique(codes, return_inverse=True)
    ind = sp.csr_matrix((np.ones(n_cells), (local, np.arange(n_cells))),
                        shape=(len(uniq), n_cells))
    dense_cells = rng.random((n_cells, n_genes)) * 1e6
    dense_cells[rng.random((n_cells, n_genes)) < 0.9] = 0.0
    sub = sp.csr_matrix(dense_cells)

    sparse_path = np.zeros((n_labels, n_genes))
    _scatter_add(sparse_path, uniq, ind @ sub)

    dense_path = np.zeros((n_labels, n_genes))
    dense_path[uniq] += (ind @ sub).toarray()

    assert np.array_equal(sparse_path, dense_path)


def test_the_scatter_survives_a_batch_whose_product_is_empty():
    """`_scatter_add` leaves the target unchanged on an empty product."""
    target = np.ones((3, 4))
    _scatter_add(target, np.array([0, 2]), sp.csr_matrix((2, 4)))
    np.testing.assert_array_equal(target, np.ones((3, 4)))


# ------------------------------------------------------- screen-coverage QC tallies --
#
# Per gene, the cells it was detected in; per construct, the cells that carried it.


def test_gene_detection_counts_cells_not_counts(tmp_path):
    """`gene_cells` counts cells with the gene detected, not counts."""
    _write(tmp_path / "f.parquet", [
        # AAA: one cell, 100 counts.  BBB: three cells, 1 count each.
        {"target": "T", "tokens": [0, 1], "values": [100.0, 1.0]},
        {"target": "T", "tokens": [1, 2], "values": [1.0, 7.0]},
        {"target": "T", "tokens": [1, 2], "values": [1.0, 5.0]},
        {"target": CONTROL_LABEL, "tokens": [0, 2], "values": [3.0, 9.0]},
    ])
    (pb, qc), = stream_keyings(
        ["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
        keyings=_keyings(["T", CONTROL_LABEL]), guide_coverage=True)

    col = {g: i for i, g in enumerate(pb.genes)}
    # AAA appears in 2 cells (one T, one control) but carries 103 counts.
    assert qc.gene_cells[col["AAA"]] == 2
    assert pb.count_sum[:, col["AAA"]].sum() == 103.0
    # BBB appears in 3 cells and carries 3 counts -- the opposite shape.
    assert qc.gene_cells[col["BBB"]] == 3
    assert pb.count_sum[:, col["BBB"]].sum() == 3.0
    assert qc.gene_cells[col["CCC"]] == 3


def test_two_tokens_colliding_into_one_gene_count_the_cell_once(tmp_path):
    """A cell carrying both tokens of `DUP` counts once."""
    axis = build_gene_axis(GENE_MAP, ["AAA", "DUP"])
    _write(tmp_path / "f.parquet", [
        {"target": "T", "tokens": [3, 4], "values": [2.0, 5.0]},
        {"target": CONTROL_LABEL, "tokens": [0], "values": [4.0]},
    ])
    (pb, qc), = stream_keyings(["f.parquet"], opener=_opener(tmp_path), axis=axis,
                               keyings=_keyings(["T", CONTROL_LABEL]), guide_coverage=True)
    col = {g: i for i, g in enumerate(pb.genes)}
    assert qc.gene_cells[col["DUP"]] == 1
    assert pb.count_sum[pb.labels.index("T"), col["DUP"]] == 7.0


def test_construct_coverage_counts_every_construct_not_just_multi(tmp_path):
    """The construct tally covers single-construct targets too, and pools the control."""
    _write(tmp_path / "f.parquet", [
        {"target": "AAA", "guide": "AAA_c1", "tokens": [0], "values": [1.0]},
        {"target": "AAA", "guide": "AAA_c2", "tokens": [0, 1], "values": [1.0, 2.0]},
        {"target": "BBB", "guide": "BBB_only", "tokens": [1], "values": [9.0]},
        {"target": "BBB", "guide": "BBB_only", "tokens": [1, 2], "values": [3.0, 1.0]},
        {"target": CONTROL_LABEL, "guide": "nt_1|nt_2", "tokens": [0], "values": [3.0]},
    ])
    (_, qc), = stream_keyings(
        ["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
        keyings=_keyings(["AAA", "BBB", CONTROL_LABEL]), guide_coverage=True)

    counts = dict(zip(qc.guide_labels, qc.guide_cells.tolist(), strict=True))
    # BBB has ONE construct, so no accumulator would ever key it -- but it is here.
    assert counts["BBB_only"] == 2
    assert counts["AAA_c1"] == 1 and counts["AAA_c2"] == 1
    # and the control stays pooled, exactly as under the guide keying
    assert counts[CONTROL_LABEL] == 1
    assert "nt_1|nt_2" not in counts


def test_coverage_is_off_unless_asked_for(tmp_path):
    """Coverage tallies are off unless `guide_coverage=True`."""
    _write(tmp_path / "f.parquet", [
        {"target": "T", "tokens": [0, 1], "values": [4.0, 1.0]},
        {"target": CONTROL_LABEL, "tokens": [0], "values": [3.0]},
    ])
    (_, qc), = stream_keyings(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                              keyings=_keyings(["T", CONTROL_LABEL]))
    assert qc.gene_cells is None
    assert qc.guide_labels == [] and qc.guide_cells is None
    assert "guide_target" not in _columns_for(_keyings(["T"]))
    assert "guide_target" in _columns_for(_keyings(["T"]), guide_coverage=True)


def test_coverage_tallies_round_trip_through_the_sidecar(tmp_path):
    _write(tmp_path / "f.parquet", [
        {"target": "T", "guide": "T_c1", "tokens": [0, 1], "values": [4.0, 1.0]},
        {"target": CONTROL_LABEL, "tokens": [0], "values": [3.0]},
    ])
    (_, qc), = stream_keyings(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                              keyings=_keyings(["T", CONTROL_LABEL]), guide_coverage=True)
    qc.save(tmp_path / "s.qc.npz")
    z = np.load(tmp_path / "s.qc.npz", allow_pickle=True)
    np.testing.assert_array_equal(z["gene_cells"], qc.gene_cells)
    assert [str(x) for x in z["guide_labels"]] == list(qc.guide_labels)
    np.testing.assert_array_equal(z["guide_cells"], qc.guide_cells)


def test_a_cell_dropped_for_zero_on_axis_counts_is_not_tallied(tmp_path):
    """A cell dropped for zero on-axis counts is not tallied."""
    _write(tmp_path / "f.parquet", [
        {"target": "T", "guide": "T_c1", "tokens": [3], "values": [50.0]},   # DUP: off-axis
        {"target": "T", "guide": "T_c1", "tokens": [0, 1], "values": [4.0, 1.0]},
        {"target": CONTROL_LABEL, "tokens": [0], "values": [3.0]},
    ])
    (_, qc), = stream_keyings(["f.parquet"], opener=_opener(tmp_path), axis=_axis(),
                              keyings=_keyings(["T", CONTROL_LABEL]), guide_coverage=True)
    assert qc.cells_dropped_zero == 1
    counts = dict(zip(qc.guide_labels, qc.guide_cells.tolist(), strict=True))
    assert counts["T_c1"] == 1          # the off-axis cell is not counted
    assert int(qc.gene_cells.sum()) == 3   # AAA+BBB from cell 2, AAA from the control
