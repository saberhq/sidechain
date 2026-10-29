"""`--gene-factor` with a `context` column: a per-context multiplier on the pooled delta.

    .venv/bin/python -m pytest -q tests/test_gene_factor_per_line.py

Writes temporary files only. Notes: private/code-notes/tests/test_gene_factor_per_line.py.md
"""
from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml

from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.submit import build
from sidechain.submit.gene_factor import (load_gene_factor, load_gene_factors, read_gene_factor_table,
                                          same_factor_groups)

GENES = ["TP53", "B", "C", "D"]


def _table(path, rows, context=True):
    cols = ["gene", "factor", "context"] if context else ["gene", "factor"]
    pd.DataFrame(rows, columns=cols).to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------------------- the loader

def test_rows_with_and_without_a_context(tmp_path):
    t = _table(tmp_path / "f.parquet", [("B", 0.5, "X"), ("C", 2.0, None), ("D", 0.0, "Y"), ("B", 0.25, "Y")])
    axis = np.array(GENES, dtype=object)
    fx, rx = load_gene_factor(t, axis, context="X")
    fy, _ = load_gene_factor(t, axis, context="Y")
    f0, r0 = load_gene_factor(t, axis)
    assert fx.tolist() == [1.0, 0.5, 2.0, 1.0] and fy.tolist() == [1.0, 0.25, 2.0, 0.0]
    assert f0.tolist() == [1.0, 1.0, 2.0, 1.0]                     # no context: the every-context rows only
    assert rx["context"] == "X" and rx["table_contexts"] == ["X", "Y"] and rx["rows"] == 2
    assert r0["context"] is None and r0["table_rows"] == 4
    of, rec = load_gene_factors(t, axis, ["X", "Y", "Z"])
    assert of["X"].tolist() == fx.tolist() and of["Y"].tolist() == fy.tolist() and of["Z"].tolist() == f0.tolist()
    assert rec["contexts"]["X"]["matched"] == 2 and rec["contexts"]["Z"]["matched"] == 1
    assert rec["rows_every_context"] == 1 and rec["table_contexts_not_in_build"] == []
    assert same_factor_groups(of, ["X", "Y", "Z"]) == [["X"], ["Y"], ["Z"]]


def test_a_table_without_the_column_reads_as_before(tmp_path):
    axis = np.array(GENES, dtype=object)
    t = _table(tmp_path / "f.parquet", [("B", 0.5)], context=False)
    of, rec = load_gene_factors(t, axis, ["X", "Y"])
    assert of["X"] is of["Y"] and of["X"].tolist() == [1.0, 0.5, 1.0, 1.0]
    assert rec == load_gene_factor(t, axis)[1] and "context" not in rec
    assert same_factor_groups(of, ["X", "Y"]) == [["X", "Y"]]


def test_csv_with_a_context_column(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("gene,factor,context\nB,0.5,X\nC,0.25,\n")
    tab = read_gene_factor_table(p)
    assert tab["context"].tolist() == ["X", None]
    f, _ = load_gene_factor(p, np.array(GENES, dtype=object), context="X")
    assert f.tolist() == [1.0, 0.5, 0.25, 1.0]


@pytest.mark.parametrize("rows, why", [
    ([("B", 0.5, "X"), ("B", 0.7, "X")], "more than once for context 'X'"),
    ([("B", 0.5, None), ("B", 0.7, None)], "more than once for every context"),
    ([("B", 0.5, None), ("B", 0.7, "X")], "for every context AND for context 'X'"),
    ([("B", -0.5, "X")], "negative"),
])
def test_refusals_of_the_table(tmp_path, rows, why):
    with pytest.raises(SystemExit, match=why):
        read_gene_factor_table(_table(tmp_path / "f.parquet", rows))


def test_refusals_of_the_context(tmp_path):
    axis = np.array(GENES, dtype=object)
    plain = _table(tmp_path / "plain.parquet", [("B", 0.5)], context=False)
    with pytest.raises(SystemExit, match="no `context` column"):
        load_gene_factor(plain, axis, context="X")
    per = _table(tmp_path / "per.parquet", [("B", 0.5, "X")])
    with pytest.raises(SystemExit, match="no row for it"):
        load_gene_factor(per, axis, context="x")
    with pytest.raises(SystemExit, match="pass --gene-factor-context"):
        load_gene_factor(per, axis)
    with pytest.raises(SystemExit, match="in any of its contexts"):
        load_gene_factors(_table(tmp_path / "z.parquet", [("B", 0.5, "Z")]), axis, ["X", "Y"])


# ------------------------------------------------------------------------------------- submit.build

def _cache(path, ctrl_label, ctrl_cpm, pert_cpm, n_cells=1000):
    m = np.asarray([ctrl_cpm, pert_cpm], dtype=float)
    n = np.full(2, n_cells, dtype=np.int64)
    PseudobulkSums(labels=[ctrl_label, "TP53"], genes=np.array(GENES, dtype=object),
                   count_sum=m * n[:, None], cpm_sum=m * n[:, None], cpm_sq_sum=(m**2 + 1.0) * n[:, None],
                   n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["test"]).save(path)


def _controls(path, base, seed):
    rng = np.random.default_rng(seed)
    lib = np.exp(rng.normal(0, 0.5, size=(40, 1))) * 3000
    X = rng.poisson(np.asarray(base, float) / sum(base) * lib).astype(np.float32)
    ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame(index=[f"c{i}" for i in range(len(X))]),
               var=pd.DataFrame(index=GENES)).write_h5ad(path)


@pytest.fixture
def ch(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    (data / "gene_names.csv").write_text("gene_name\n" + "\n".join(GENES) + "\n")
    (data / "pert_counts.csv").write_text("target_gene\nTP53\n")
    _controls(data / "ctx_x.h5ad", [300, 700, 700, 500], 1)
    _controls(data / "ctx_y.h5ad", [500, 505, 595, 900], 2)
    _cache(data / "h1.npz", "non-targeting", [100000.0, 300000.0, 300000.0, 300000.0],
           [200000.0, 400000.0, 150000.0, 250000.0])
    _cache(data / "gwps.npz", "control", [100000.0, 300000.0, 300000.0, 300000.0],
           [210000.0, 395000.0, 160000.0, 235000.0])
    _cache(data / "extra.npz", "control", [300000.0, 250000.0, 250000.0, 200000.0],
           [30000.0, 485000.0, 200000.0, 285000.0])
    cfg = {"data_dir": str(data), "gene_names_file": "gene_names.csv", "n_genes": len(GENES),
           "pert_counts_file": "pert_counts.csv", "pert_col": "target_gene", "context_col": "context",
           "control_label": "non-targeting", "phase": "p1", "phases": {"p1": {"contexts": ["X", "Y"]}},
           "control_files": {"X": "ctx_x.h5ad", "Y": "ctx_y.h5ad"},
           "submission": {"cells_per_pert": 50, "max_counts_per_cell": 1_000_000, "max_cells": 100_000,
                          "max_stored_entries": 10_000_000}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    return {"cfg": tmp_path / "config.yaml", "data": data, "out": tmp_path / "out", "tmp": tmp_path}


def _run(ch, stem, *extra):
    assert build.main(["--challenge-config", str(ch["cfg"]), "--emitter", "delta-transfer",
                       "--h1-cache", str(ch["data"] / "h1.npz"), "--gwps-cache", str(ch["data"] / "gwps.npz"),
                       "--out", str(ch["out"] / stem), "--no-pack", "--min-libsize", "10", "--no-shrink",
                       "--alpha", "1.2", "--alpha-bulk", "1.6", "--emit-lambda", "0.4", *extra]) == 0
    return ad.read_h5ad(ch["out"] / f"{stem}.h5ad")


def _block(pred, ctx):
    return pred[pred.obs["context"].astype(str) == ctx].X.toarray()


def _gf(path):
    return ["--gene-factor", str(path)]


def test_a_context_the_table_names_moves_and_the_other_is_the_plain_build(ch):
    t = ch["tmp"]
    per = _table(t / "per.parquet", [("B", 0.3, "X"), ("C", 2.5, "X"), ("D", 0.0, "X")])
    uni_x = _table(t / "uni_x.parquet", [("B", 0.3), ("C", 2.5), ("D", 0.0)], context=False)
    plain, on, ux = _run(ch, "plain"), _run(ch, "per", *_gf(per)), _run(ch, "uni_x", *_gf(uni_x))
    assert np.array_equal(_block(on, "Y"), _block(plain, "Y"))          # Y: no row applies, cell for cell
    assert np.array_equal(_block(on, "X"), _block(ux, "X"))             # X: the table's X rows, nothing else
    assert not np.array_equal(_block(on, "X"), _block(plain, "X"))      # and they do move X
    rec = json.loads((ch["out"] / "per.gene_factor.json").read_text())
    assert rec["factor_groups"] == [["X"], ["Y"]] and rec["contexts"]["X"]["matched"] == 3
    assert rec["contexts"]["Y"]["matched"] == 0 and len(rec["sha256"]) == 64


def test_two_contexts_two_tables(ch):
    t = ch["tmp"]
    per = _table(t / "per.parquet", [("B", 0.3, "X"), ("C", 2.0, "Y"), ("D", 0.5, None)])
    ux = _table(t / "ux.parquet", [("B", 0.3), ("D", 0.5)], context=False)
    uy = _table(t / "uy.parquet", [("C", 2.0), ("D", 0.5)], context=False)
    on, bx, by = _run(ch, "per", *_gf(per)), _run(ch, "ux", *_gf(ux)), _run(ch, "uy", *_gf(uy))
    assert np.array_equal(_block(on, "X"), _block(bx, "X")) and np.array_equal(_block(on, "Y"), _block(by, "Y"))


def test_every_context_rows_with_the_column_are_the_table_without_it(ch):
    t = ch["tmp"]
    with_col = _table(t / "w.parquet", [("B", 0.3, None), ("C", 2.0, None)])
    without = _table(t / "wo.parquet", [("B", 0.3), ("C", 2.0)], context=False)
    a, b = _run(ch, "w", *_gf(with_col)), _run(ch, "wo", *_gf(without))
    assert (a.X != b.X).nnz == 0
    assert "factor_groups" not in json.loads((ch["out"] / "w.gene_factor.json").read_text())


def test_routed_build_splits_each_pool_by_factor(ch):
    t = ch["tmp"]
    src = ["--source", f"{ch['data'] / 'extra.npz'}:control", "--source-context", "extra=X"]
    per = _table(t / "per.parquet", [("B", 0.3, "X"), ("C", 2.0, "Y")])
    ux = _table(t / "ux.parquet", [("B", 0.3)], context=False)
    uy = _table(t / "uy.parquet", [("C", 2.0)], context=False)
    on = _run(ch, "per", *src, *_gf(per))
    bx, by, routed = _run(ch, "ux", *src, *_gf(ux)), _run(ch, "uy", *src, *_gf(uy)), _run(ch, "routed", *src)
    assert np.array_equal(_block(on, "X"), _block(bx, "X")) and np.array_equal(_block(on, "Y"), _block(by, "Y"))
    assert not np.array_equal(_block(on, "Y"), _block(routed, "Y"))
    # the routing record still lists the distinct pools, not the factor's split
    assert (json.loads((ch["out"] / "per.routing.json").read_text())
            == json.loads((ch["out"] / "routed.routing.json").read_text()))


def test_one_pool_shared_by_contexts_with_different_factors(ch):
    """Unrouted, X and Y share one pooled map: neither context's factor reaches the other's cells."""
    t = ch["tmp"]
    per = _table(t / "per.parquet", [("B", 0.0, "X"), ("B", 3.0, "Y")])
    ux = _table(t / "ux.parquet", [("B", 0.0)], context=False)
    uy = _table(t / "uy.parquet", [("B", 3.0)], context=False)
    on, bx, by = _run(ch, "per", *_gf(per)), _run(ch, "ux", *_gf(ux)), _run(ch, "uy", *_gf(uy))
    assert np.array_equal(_block(on, "X"), _block(bx, "X")) and np.array_equal(_block(on, "Y"), _block(by, "Y"))


def test_under_gamma_each_context_takes_its_rows(ch):
    t = ch["tmp"]
    per = _table(t / "per.parquet", [("B", 0.3, "X"), ("C", 2.0, "X")])
    ux = _table(t / "ux.parquet", [("B", 0.3), ("C", 2.0)], context=False)
    g = ["--gamma", "0.5"]
    on, bx, plain = _run(ch, "per", *g, *_gf(per)), _run(ch, "ux", *g, *_gf(ux)), _run(ch, "plain", *g)
    assert np.array_equal(_block(on, "X"), _block(bx, "X")) and np.array_equal(_block(on, "Y"), _block(plain, "Y"))


def test_build_refusals(ch, capsys):
    t = ch["tmp"]
    with pytest.raises(SystemExit, match="in any of its contexts"):
        _run(ch, "z", *_gf(_table(t / "z.parquet", [("B", 0.3, "Z")])))
    with pytest.raises(SystemExit):
        _run(ch, "ser-99_fixture_v1", "--bulk-anchor", "pooled",
             *_gf(_table(t / "u.parquet", [("B", 0.3)], context=False)))
    assert "--gene-factor has no registered knob letter" in capsys.readouterr().err


def test_a_table_context_the_build_does_not_hold_is_recorded(ch):
    t = ch["tmp"]
    _run(ch, "folds", *_gf(_table(t / "f.parquet", [("B", 0.3, "X"), ("B", 0.5, "k562")])))
    rec = json.loads((ch["out"] / "folds.gene_factor.json").read_text())
    assert rec["table_contexts_not_in_build"] == ["k562"] and rec["table_contexts"] == ["X", "k562"]


# ---------------------------------------------------------------------------------------- eval.loco

def _fold(tmp_path):
    genes = np.array([f"g{i}" for i in range(30)], dtype=object)
    targets = [f"g{i}" for i in range(6)]
    rng = np.random.default_rng(3)

    def line(seed):
        r = np.random.default_rng(seed)
        labels = ["ctrl"] + targets
        mean = r.uniform(100, 2000, size=(len(labels), len(genes)))
        n = np.full(len(labels), 2000, dtype=np.int64)
        return (PseudobulkSums(labels=labels, genes=genes.copy(), count_sum=mean * n[:, None],
                               cpm_sum=mean * n[:, None], cpm_sq_sum=(mean**2 + mean) * n[:, None],
                               n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["test"]), "ctrl")
    basal = rng.uniform(100, 2000, size=len(genes))
    rows, labels = [], []
    for _ in range(30):
        rows.append(rng.poisson(basal * 40 / basal.sum() * 1000)); labels.append("non-targeting")
    for tg in targets[:2]:
        for _ in range(8):
            rows.append(rng.poisson(basal * 40 / basal.sum() * 1000)); labels.append(tg)
    real = ad.AnnData(X=sp.csr_matrix(np.asarray(rows, dtype=np.float32)),
                      obs=pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(len(rows))]),
                      var=pd.DataFrame(index=genes.astype(str)))
    real.write_h5ad(tmp_path / "real.h5ad")
    return tmp_path / "real.h5ad", [line(1), line(2)], genes


def test_loco_picks_the_named_context_rows(tmp_path):
    from sidechain.eval import loco

    real, srcs, genes = _fold(tmp_path)
    kw = dict(pert_col="perturbation", control="non-targeting", shrinkage=False, var_floor="poisson",
              min_libsize=0.0, seed=0)
    per = _table(tmp_path / "per.parquet", [(g, 0.0, "k562") for g in genes[2:]]
                 + [(g, 3.0, "hek293t") for g in genes[2:]])
    uni = _table(tmp_path / "uni.parquet", [(g, 0.0) for g in genes[2:]], context=False)
    on = loco.build_transfer_prediction(real, srcs, tmp_path / "on.h5ad", gene_factor=per,
                                        gene_factor_context="k562", **kw)
    loco.build_transfer_prediction(real, srcs, tmp_path / "uni.h5ad", gene_factor=uni, **kw)
    assert on["gene_factor"]["context"] == "k562" and on["gene_factor"]["zero"] == len(genes) - 2
    a, b = ad.read_h5ad(tmp_path / "on.h5ad"), ad.read_h5ad(tmp_path / "uni.h5ad")
    assert (a.X != b.X).nnz == 0
    other = loco.build_transfer_prediction(real, srcs, tmp_path / "hek.h5ad", gene_factor=per,
                                           gene_factor_context="hek293t", **kw)
    assert other["gene_factor"]["above_one"] == len(genes) - 2
    with pytest.raises(SystemExit, match="no row for it"):
        loco.build_transfer_prediction(real, srcs, tmp_path / "x.h5ad", gene_factor=per,
                                       gene_factor_context="K562", **kw)
    with pytest.raises(SystemExit, match="pass --gene-factor-context"):
        loco.build_transfer_prediction(real, srcs, tmp_path / "y.h5ad", gene_factor=per, **kw)
    with pytest.raises(SystemExit, match="pass gene_factor"):
        loco.build_transfer_prediction(real, srcs, tmp_path / "z.h5ad", gene_factor_context="k562", **kw)

