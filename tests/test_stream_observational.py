"""Tests for the observational reader — the fourth streamer, and the first with no labels.

The theme running through these: an unlabelled corpus has no downstream check. A perturbation
pseudobulk that goes wrong shows up as a control arm that matches nothing or a delta that comes
out zero. Nothing here would notice, so the checks have to be at this end.
"""
import json

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sidechain.data.stream_observational import (
    Accumulator,
    ContextProfiles,
    gene_axis_map,
    read_counts,
    select_experiments,
)
from sidechain.data.stream_pseudobulk import PseudobulkSums

# --------------------------------------------------------------- the output --


def _profiles():
    acc = Accumulator(np.array(["A", "B", "C"]))
    # two cells, on an axis whose gene order is NOT the file's
    counts = sp.csr_matrix(np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 5.0]]))
    acc.add("k562", counts, np.array([2, 0, 1]))     # file col 0 -> axis "C", 1 -> "A", 2 -> "B"
    return acc.result()


def test_a_context_panel_cannot_be_loaded_as_a_perturbation_panel(tmp_path):
    """The one that matters most. `PseudobulkSums` is what `pooled_delta` consumes, and this
    corpus has no per-cell perturbation label -- so it has no deltas and must never vote. If
    both objects shared an npz layout, a context panel passed as `--source` would be pooled as
    if its rows were perturbations, and every number downstream would be fiction with no error
    anywhere. Different key names make that a crash."""
    p = tmp_path / "ctx.npz"
    _profiles().save(p)
    with pytest.raises(KeyError):
        PseudobulkSums.load(p)


def test_profiles_round_trip(tmp_path):
    p = tmp_path / "ctx.npz"
    before = _profiles()
    before.save(p)
    after = ContextProfiles.load(p)
    assert after.contexts == before.contexts
    assert list(after.genes) == list(before.genes)
    np.testing.assert_allclose(after.cpm_sum, before.cpm_sum)
    np.testing.assert_array_equal(after.detect_count, before.detect_count)


def test_columns_land_on_the_axis_the_map_names_not_the_files_order():
    """The gene axis is rebuilt per file rather than assumed shared. If the remap were
    ignored, the profile would be right-looking and wrong -- counts under the wrong symbols,
    with nothing downstream to catch it."""
    got = _profiles()
    # file counts summed per column: [1, 2, 8]; map sends them to axis [C, A, B]
    np.testing.assert_allclose(got.count_sum[0], [2.0, 8.0, 1.0])
    np.testing.assert_array_equal(got.detect_count[0], [1, 2, 1])


def test_offaxis_genes_are_dropped_not_wrapped():
    """A -1 in the map means the gene is not on our axis. Dropping is right; letting it index
    from the end (numpy's default for -1) would silently add a foreign gene's counts to the
    LAST gene on the axis."""
    acc = Accumulator(np.array(["A", "B"]))
    counts = sp.csr_matrix(np.array([[1.0, 9.0, 2.0]]))
    acc.add("x", counts, np.array([0, -1, 1]))
    got = acc.result()
    np.testing.assert_allclose(got.count_sum[0], [1.0, 2.0])
    assert 9.0 not in got.count_sum[0], "an off-axis gene reached the axis"


def test_cpm_uses_the_full_library_not_just_the_kept_genes():
    """CPM has to be per-cell counts over that cell's TOTAL library. Normalising by the
    on-axis subtotal instead would rescale every profile by how much of the corpus's gene
    space our axis happens to cover, which differs per file and would look like biology."""
    acc = Accumulator(np.array(["A"]))
    counts = sp.csr_matrix(np.array([[1.0, 9.0]]))   # library is 10, only "A" is on-axis
    acc.add("x", counts, np.array([0, -1]))
    got = acc.result()
    np.testing.assert_allclose(got.cpm_sum[0], [1e5])       # 1/10 * 1e6
    np.testing.assert_allclose(got.libsize_sum, [10.0])


def test_sums_are_additive_across_experiments():
    """Every field is a sum so two runs over disjoint experiments can be added. This is what
    makes a 26,952-experiment stream resumable rather than all-or-nothing."""
    acc = Accumulator(np.array(["A", "B"]))
    m = np.array([0, 1])
    acc.add("x", sp.csr_matrix(np.array([[1.0, 2.0]])), m)
    acc.add("x", sp.csr_matrix(np.array([[3.0, 4.0]])), m)
    got = acc.result()
    assert got.n_cells[0] == 2 and got.n_experiments[0] == 2
    np.testing.assert_allclose(got.count_sum[0], [4.0, 6.0])


def test_mean_and_detection_rate_divide_by_cells_not_experiments():
    acc = Accumulator(np.array(["A"]))
    acc.add("x", sp.csr_matrix(np.array([[1.0], [0.0], [3.0]])), np.array([0]))
    got = acc.result()
    assert got.n_cells[0] == 3
    np.testing.assert_allclose(got.detect_rate()[0], [2 / 3])


# ------------------------------------------------------------- the selection --


def _manifest():
    return pd.DataFrame({
        "srx_accession": ["S1", "S2", "S3", "S4", "S5"],
        "obs_count": [1000, 1000, 1000, 1000, 1000],
        "cell_prep": ["single_cell"] * 4 + ["single_nucleus"],
        "tech_10x": ["3_prime_gex", "3_prime_gex", "5_prime_gex", "3_prime_gex", "3_prime_gex"],
        "cell_line": ["k562", "jurkat", "k562", "k562", "k562"],
        "file_path": [f"gs://b/{s}.h5ad" for s in ["S1", "S2", "S3", "S4", "S5"]],
    })


CF = {"min_bytes_per_cell": 500, "min_umi_per_cell": 1000,
      "cell_prep": ["single_cell"], "tech_10x": ["3_prime_gex"]}
SIZES = {"S1": 5_000_000, "S2": 5_000_000, "S3": 5_000_000,
         "S4": 100_000,   # 100 B/cell -- a guide-capture sub-library
         "S5": 5_000_000}


def test_the_declared_filter_removes_every_kind_of_junk_it_names():
    got = select_experiments(_manifest(), CF, sizes=SIZES)
    assert list(got.srx_accession) == ["S1", "S2"], (
        "expected 5' (S3), the low-content file (S4) and the nucleus prep (S5) all gone"
    )


def test_an_empty_filter_selects_everything_which_is_why_the_config_forbids_one():
    """`select_experiments` is a pure function and does what it is told. The refusal lives
    one level up, in `stream_observational`, because that is where a config with an empty
    `content_filter` would actually be acted on."""
    assert len(select_experiments(_manifest(), {}, sizes=SIZES)) == 5


def test_missing_sizes_skip_the_bytes_rule_rather_than_dropping_everything():
    """Without PROVENANCE.json there are no per-file sizes. Silently dropping every row would
    look like a corpus with nothing in it; the caller raises instead, and this pins that the
    rule simply does not apply here."""
    got = select_experiments(_manifest(), CF, sizes=None)
    assert list(got.srx_accession) == ["S1", "S2", "S4"]


def test_exclusion_takes_the_independent_part():
    """How the 62.4 % of scBaseCount that is not a screen we already pool gets taken."""
    got = select_experiments(_manifest(), CF, sizes=SIZES, exclude={"S1"})
    assert list(got.srx_accession) == ["S2"]


def test_gene_axis_map_is_by_symbol_and_marks_the_unknown():
    got = gene_axis_map(np.array(["B", "ZZZ", "A"]), np.array(["A", "B"]))
    np.testing.assert_array_equal(got, [1, -1, 0])


# ------------------------------------------------------------------ reading --


def _h5(path, X, symbols, encoding="csc_matrix"):
    M = sp.csc_matrix(X) if encoding == "csc_matrix" else sp.csr_matrix(X)
    with h5py.File(path, "w") as f:
        g = f.create_group("X")
        g.attrs["encoding-type"] = encoding
        g.attrs["shape"] = np.array(X.shape)
        g.create_dataset("data", data=M.data.astype(np.float32))
        g.create_dataset("indices", data=M.indices.astype(np.int32))
        g.create_dataset("indptr", data=M.indptr.astype(np.int32))
        v = f.create_group("var")
        v.attrs["encoding-type"] = "dataframe"
        v.attrs["encoding-version"] = "0.2.0"
        v.attrs["_index"] = "_index"
        v.attrs["column-order"] = ["gene_symbols"]
        v.create_dataset("_index", data=np.array([f"ENSG{i}" for i in range(len(symbols))],
                                                 dtype="S16"))
        v.create_dataset("gene_symbols", data=np.array(symbols, dtype="S16"))
    return path


def test_a_csc_file_is_read_and_the_umi_floor_drops_shallow_cells(tmp_path):
    """The floor is per CELL. A per-experiment filter cannot remove a bad barcode, which is
    why `min_umi_per_cell` exists beside `min_bytes_per_cell`: corpus-wide only 71.8 % of
    scBaseCount's cells reach 500 UMI."""
    X = np.array([[600.0, 600.0], [1.0, 2.0], [900.0, 900.0]])   # libs 1200, 3, 1800
    p = _h5(tmp_path / "a.h5ad", X, ["A", "B"])
    with h5py.File(p) as f:
        M, symbols, dropped = read_counts(f, symbol_col="gene_symbols", min_umi=1000)
    assert M.shape[0] == 2 and dropped == 1
    assert list(symbols) == ["A", "B"]


def test_a_csr_file_reads_identically(tmp_path):
    X = np.array([[600.0, 600.0], [900.0, 900.0]])
    with h5py.File(_h5(tmp_path / "c.h5ad", X, ["A", "B"], encoding="csr_matrix")) as f:
        M, _, dropped = read_counts(f, symbol_col="gene_symbols", min_umi=1000)
    assert M.shape[0] == 2 and dropped == 0


def test_a_wrong_symbol_column_raises_rather_than_keying_on_nothing(tmp_path):
    """`gene_symbol_col` is declared in the config. Guessing wrong would key the profiles on
    an axis nothing else shares, and no downstream consumer would notice."""
    # two genes, not one: anndata reads a single-element var index as a scalar and raises
    # its own TypeError before this reader is ever reached.
    with h5py.File(_h5(tmp_path / "b.h5ad", np.array([[1.0, 2.0]]), ["A", "B"])) as f, \
            pytest.raises(KeyError, match="gene_symbol_col|no 'symbol'"):
        read_counts(f, symbol_col="symbol", min_umi=0)


def test_zero_umi_floor_keeps_every_cell(tmp_path):
    X = np.array([[1.0, 0.0], [0.0, 0.0]])
    with h5py.File(_h5(tmp_path / "d.h5ad", X, ["A", "B"])) as f:
        M, _, dropped = read_counts(f, symbol_col="gene_symbols", min_umi=0)
    assert M.shape[0] == 2 and dropped == 0


# ------------------------------------------------------------------ lineage --


def test_the_reader_refuses_a_block_with_an_empty_content_filter(tmp_path):
    """The rule the config header states, enforced where it is acted on. An unlabelled corpus
    with no filter builds a prior out of whatever the corpus happens to contain -- for
    scBaseCount, 20.8 % near-empty matrices -- and reports success."""
    from sidechain.data.stream_observational import stream_observational

    block = {"name": "x", "host": "lamin", "record": "o/i", "dest": "external/x",
             "files": [{"name": "f", "kind": "observational",
                        "spec": {"sample_from": "m.parquet", "sample_id_col": "srx",
                                 "context_col": "cell_line", "content_filter": {}}}]}
    with pytest.raises(ValueError, match="empty content_filter"):
        stream_observational(block, root=tmp_path, gene_axis=np.array(["A"]))


def test_a_missing_provenance_stops_the_run_when_a_bytes_rule_is_declared(tmp_path):
    """The declared filter has to actually run. PROVENANCE.json is where the per-file sizes
    live, so without it `min_bytes_per_cell` would quietly not happen."""
    from sidechain.data.stream_observational import stream_observational

    block = {"name": "x", "host": "lamin", "record": "o/i", "dest": "external/x",
             "files": [{"name": "f", "kind": "observational",
                        "spec": {"sample_from": "m.parquet", "sample_id_col": "srx",
                                 "context_col": "cell_line",
                                 "content_filter": {"min_bytes_per_cell": 500}}}]}
    with pytest.raises(FileNotFoundError, match="min_bytes_per_cell"):
        stream_observational(block, root=tmp_path, gene_axis=np.array(["A"]))


def test_the_real_block_is_wired_for_this_reader():
    """The registry block and the reader have to agree on every key the reader reads."""
    from sidechain.ingest.fetch import load_datasets

    block = load_datasets()["scbasecount_human_gene"]
    spec = next(f["spec"] for f in block["files"] if f.get("kind") == "observational")
    for key in ("sample_from", "sample_id_col", "context_col", "content_filter",
                "gene_symbol_col"):
        assert spec.get(key), f"the reader reads {key!r} and the block does not declare it"
    assert spec["sample_from"] in {f["name"] for f in block["files"]}
    assert json.dumps(spec["content_filter"])  # declarative, serialisable into LINEAGE.json


def test_non_labels_are_dropped_by_name_and_case_insensitively():
    """`unsure` / `none` / `not_applicable` are 4,358 scBaseCount experiments that share no
    biology. Pooling them into one row would manufacture the composition artefact the whole
    design exists to avoid. Named by the caller, never defaulted, so it lands in LINEAGE.json."""
    m = _manifest().copy()
    m["cell_line"] = ["k562", "unsure", "k562", "Not_Applicable", "k562"]
    got = select_experiments(m, CF, sizes=SIZES, drop_contexts={"unsure", "not_applicable"},
                             context_col="cell_line")
    assert list(got.srx_accession) == ["S1"]


def test_the_per_context_cap_keeps_the_deepest_and_gives_breadth():
    """A context panel wants many contexts each measured well, not one measured 4,000 times."""
    m = _manifest().copy()
    m["cell_line"] = ["k562", "k562", "k562", "jurkat", "jurkat"]
    m["cell_prep"] = ["single_cell"] * 5
    m["tech_10x"] = ["3_prime_gex"] * 5
    m["obs_count"] = [10, 30, 20, 5, 50]
    got = select_experiments(m, {"cell_prep": ["single_cell"]}, context_col="cell_line",
                             max_per_context=1)
    assert set(got.srx_accession) == {"S2", "S5"}, "expected the deepest of each context"


def test_the_cap_ranks_by_content_per_cell_not_by_cell_count():
    """Most cells is not best measured. 50,000 barcodes at 300 UMI is a worse profile than
    5,000 at 20,000 -- and ten times the bytes. The cap ranks on the bytes-per-cell proxy,
    which was validated against true depth on 2026-08-29."""
    m = _manifest().copy()
    m["cell_line"] = ["k562"] * 5
    m["cell_prep"] = ["single_cell"] * 5
    m["tech_10x"] = ["3_prime_gex"] * 5
    m["obs_count"] = [50_000, 5_000, 1_000, 1_000, 1_000]
    sizes = {"S1": 15_000_000, "S2": 100_000_000, "S3": 1, "S4": 1, "S5": 1}   # S2 is richest/cell
    got = select_experiments(m, {"cell_prep": ["single_cell"]}, sizes=sizes,
                             context_col="cell_line", max_per_context=1)
    assert list(got.srx_accession) == ["S2"], "the cap took the biggest, not the richest"


def test_a_cell_floor_stops_the_ranking_picking_a_deep_but_tiny_run():
    m = _manifest().copy()
    m["cell_line"] = ["k562"] * 5
    m["cell_prep"] = ["single_cell"] * 5
    m["tech_10x"] = ["3_prime_gex"] * 5
    m["obs_count"] = [10, 5_000, 1_000, 1_000, 1_000]
    sizes = {"S1": 10_000_000, "S2": 100_000_000, "S3": 1, "S4": 1, "S5": 1}   # S1 richest, 10 cells
    got = select_experiments(m, {"cell_prep": ["single_cell"]}, sizes=sizes,
                             context_col="cell_line", max_per_context=1,
                             min_cells_per_experiment=1000)
    assert list(got.srx_accession) == ["S2"]


def test_an_allow_list_keeps_only_the_named_contexts_and_keeps_spellings_apart():
    """Measured reason this exists: `cell_line` is 5,644 free-text strings, so "top N by cell
    mass" returns primary tissue and descriptive sentences, and none of the four rotation lines
    appears in the top 150. Spellings stay SEPARATE rows -- whether `MCF7` and `MCF-7` land on
    each other is a control, not something to assume away."""
    m = _manifest().copy()
    m["cell_line"] = ["K562", "MCF7", "MCF-7", "PBMC from a donor", "k562"]
    m["cell_prep"] = ["single_cell"] * 5
    m["tech_10x"] = ["3_prime_gex"] * 5
    got = select_experiments(m, {"cell_prep": ["single_cell"]}, context_col="cell_line",
                             keep_contexts={"k562", "MCF7", "MCF-7"})
    assert set(got.srx_accession) == {"S1", "S2", "S3", "S5"}
    assert set(got.cell_line) == {"K562", "MCF7", "MCF-7", "k562"}
