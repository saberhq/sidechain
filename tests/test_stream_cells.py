"""Contract tests for the per-cell writer (`sidechain.data.stream_cells.CellSink`).

One contract so far: a per-cell file keeps the SOURCE's own cell id. The X-Atlas fold files
written before 2026-10-09 dropped `cell_barcode`, and joining them to the corpus's per-cell guide
UMI calls then needed a replay of the stream's own selection (T111). The id rides as an obs
column; the index stays `cell_<i>` because readers use it positionally.
"""
import anndata as ad
import numpy as np
import pandas as pd

from sidechain.data.stream_cells import CellSink
from sidechain.data.stream_parquet_pseudobulk import build_gene_axis


def _frame() -> pd.DataFrame:
    """Six cells in the real long schema: parallel list columns, one row per cell."""
    return pd.DataFrame({
        "gene_token_id": [[0, 1], [1, 2], [0], [2, 3], [1], [0, 3]],
        "gene_expression": [[2.0, 1.0], [3.0, 1.0], [5.0], [1.0, 1.0], [4.0], [1.0, 2.0]],
        "gene_target": ["AATF", "AATF", "AATF", "Non-Targeting", "Non-Targeting", "ZNF100"],
        "guide_target": ["AATF_1|AATF_2"] * 3 + ["nt_1|nt_2", "nt_3|nt_4", "ZNF100_1|ZNF100_2"],
        "sample": ["B1"] * 6,
        "total_counts": [3.0, 4.0, 5.0, 2.0, 4.0, 3.0],
        "cell_barcode": [f"ACGT{i}-B1" for i in range(6)],
    })


def _axis():
    gene_map = pd.DataFrame({"ensembl_id": [f"ENSG{i}" for i in range(4)],
                             "gene_name": ["G0", "G1", "G2", "G3"], "gene_token_id": [0, 1, 2, 3]})
    return build_gene_axis(gene_map, None)


def test_cell_sink_keeps_the_source_barcode(tmp_path):
    frame = _frame()
    # AATF is over its budget (3 cells, 2 wanted), so the seeded pick inside a batch runs too
    sink = CellSink(_axis(), {("AATF", "B1"): 2, ("Non-Targeting", "B1"): 2},
                    relabel={"Non-Targeting": "non-targeting"})
    sink.fold(frame, "B1")
    out = tmp_path / "cells.h5ad"
    sink.write(out)
    obs = ad.read_h5ad(out).obs
    assert list(obs.index) == [f"cell_{i}" for i in range(len(obs))]
    assert len(obs) == 4 and "cell_barcode" in obs.columns
    barcodes = obs["cell_barcode"].astype(str)
    assert barcodes.is_unique and (barcodes.str.len() > 0).all()
    # every kept row carries the barcode of the source row it came from: the label, the guide
    # pair and the depth of that source row are the kept row's own
    src = frame.set_index("cell_barcode")
    assert (src.loc[barcodes, "gene_target"].to_numpy() == obs["gene_target"].astype(str).to_numpy()).all()
    assert (src.loc[barcodes, "guide_target"].to_numpy() == obs["guide_target"].astype(str).to_numpy()).all()
    assert np.array_equal(src.loc[barcodes, "total_counts"].to_numpy(), obs["total_counts"].to_numpy())
    assert "ZNF100" not in set(obs["gene_target"].astype(str))


def test_cell_sink_without_a_barcode_column_still_writes(tmp_path):
    """A corpus that publishes no cell id is written as before, with no invented column."""
    sink = CellSink(_axis(), {("AATF", "B1"): 3})
    sink.fold(_frame().drop(columns="cell_barcode"), "B1")
    out = tmp_path / "cells.h5ad"
    sink.write(out)
    obs = ad.read_h5ad(out).obs
    assert len(obs) == 3 and "cell_barcode" not in obs.columns
