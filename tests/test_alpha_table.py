"""`--alpha-table` (T1): a per-target multiplier on alpha, in both channels, the target's own gene pinned after.

Pinned here: the loader's alignment and refusals; on `submit.build`, a named perturbation's block carries the
plain build's vector times its factor on every gene but the pinned target, in both channels, an unnamed one the
plain build's vector, and the record names the table; on `eval.loco`, an all-ones table is bit-identical to no
table and a real one moves the emitted cells and is recorded; a model-named stem refuses the flag (no letter yet).
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
from sidechain.models.count_emitters import PoissonEmitter
from sidechain.submit import build
from sidechain.submit.alpha_table import load_alpha_table, read_alpha_table

GENES = ["A", "B", "C", "D"]


def _table(path, rows):
    pd.DataFrame(rows, columns=["target", "factor"]).to_parquet(path, index=False)
    return path


def test_loader_aligns_and_defaults_to_one(tmp_path):
    t = _table(tmp_path / "f.parquet", [("B", 0.5), ("D", 0.0), ("ZZ", 2.0)])
    f, rec = load_alpha_table(t, ["A", "B", "C", "D"])
    assert f == {"A": 1.0, "B": 0.5, "C": 1.0, "D": 0.0}
    assert rec["rows"] == 3 and rec["matched"] == 2 and rec["unmatched_in_table"] == 1 and rec["perturbations"] == 4
    assert rec["below_one"] == 2 and rec["zero"] == 1 and rec["above_one"] == 0
    assert rec["mean_on_matched"] == 0.25 and len(rec["sha256"]) == 64


def test_loader_reads_csv_too(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("target,factor\nA,1.25\n")
    f, rec = load_alpha_table(p, ["A", "B"])
    assert f == {"A": 1.25, "B": 1.0} and rec["above_one"] == 1


@pytest.mark.parametrize("rows, why", [
    ([("A", -0.1)], "negative"),
    ([("A", float("nan"))], "non-finite"),
    ([("A", 0.5), ("A", 0.7)], "more than once"),
])
def test_loader_refuses_a_bad_table(tmp_path, rows, why):
    t = _table(tmp_path / "f.parquet", rows)
    with pytest.raises(SystemExit, match=why):
        read_alpha_table(t)


def test_loader_refuses_a_missing_column_and_an_unmatched_table(tmp_path):
    p = tmp_path / "f.parquet"
    pd.DataFrame({"target": ["A"], "weight": [0.5]}).to_parquet(p, index=False)
    with pytest.raises(SystemExit, match="factor"):
        load_alpha_table(p, ["A"])
    t = _table(tmp_path / "g.parquet", [("ZZ", 0.5)])
    with pytest.raises(SystemExit, match="no target"):
        load_alpha_table(t, ["A"])
    with pytest.raises(SystemExit, match="does not exist"):
        load_alpha_table(tmp_path / "none.parquet", ["A"])


# ---------------------------------------------------------------------------------- submit.build

def _cache(path, ctrl_label, ctrl_cpm, pert_cpm, n_cells=1000):
    m = np.asarray([ctrl_cpm, pert_cpm, pert_cpm], dtype=float)
    n = np.full(3, n_cells, dtype=np.int64)
    PseudobulkSums(
        labels=[ctrl_label, "A", "B"], genes=np.array(GENES, dtype=object),
        count_sum=m * n[:, None], cpm_sum=m * n[:, None],
        cpm_sq_sum=(m**2 + 1.0) * n[:, None],
        n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["test"],
    ).save(path)


def _controls_h5ad(path, counts_per_cell, depths):
    base = np.asarray(counts_per_cell, float)
    X = sp.csr_matrix(np.stack([np.rint(base * d / base.sum()) for d in depths]))
    ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(len(depths))]),
               var=pd.DataFrame(index=GENES)).write_h5ad(path)


@pytest.fixture
def challenge(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    (data / "gene_names.csv").write_text("gene_name\n" + "\n".join(GENES) + "\n")
    (data / "pert_counts.csv").write_text("target_gene\nA\nB\n")
    depths = [1500, 2000, 2500, 3000, 1800, 2200, 2700, 1600, 2400, 2900, 2100, 1900]
    _controls_h5ad(data / "ctx_x.h5ad", [10, 1000, 990, 500], depths)
    for name, label in (("h1.npz", "non-targeting"), ("gwps.npz", "control")):
        _cache(data / name, label, [100000.0, 300000.0, 300000.0, 300000.0],
               [200000.0, 400000.0, 200000.0, 200000.0])
    cfg = {
        "data_dir": str(data), "gene_names_file": "gene_names.csv", "n_genes": len(GENES),
        "pert_counts_file": "pert_counts.csv", "pert_col": "target_gene",
        "context_col": "context", "control_label": "non-targeting",
        "phase": "p1", "phases": {"p1": {"contexts": ["X"]}},
        "control_files": {"X": "ctx_x.h5ad"},
        "submission": {"cells_per_pert": 6, "max_counts_per_cell": 1_000_000,
                       "max_cells": 100_000, "max_stored_entries": 10_000_000},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return {"cfg": cfg_path, "data": data, "out": tmp_path / "out"}


def _argv(ch, stem, extra):
    return ["--challenge-config", str(ch["cfg"]), "--emitter", "delta-transfer",
            "--h1-cache", str(ch["data"] / "h1.npz"), "--gwps-cache", str(ch["data"] / "gwps.npz"),
            "--out", str(ch["out"] / stem), "--no-pack", "--min-libsize", "100", "--no-shrink",
            "--alpha", "1.35", "--alpha-bulk", "1.5", "--emit-lambda", "0.5", *extra]


def test_build_scales_a_named_target_in_both_channels_and_records_the_table(challenge, monkeypatch, tmp_path):
    seen = {}

    def spy(self, n, log2fc_cell, log2fc_bulk, **kw):
        seen.setdefault(self.last_stem, []).append((np.array(log2fc_cell), np.array(log2fc_bulk)))
        return self.emit(n, log2fc_cell)
    monkeypatch.setattr(PoissonEmitter, "emit_dual", spy)
    PoissonEmitter.last_stem = "plain"
    assert build.main(_argv(challenge, "plain", [])) == 0
    t = _table(tmp_path / "f.parquet", [("A", 1.5), ("ZZ", 0.2)])
    PoissonEmitter.last_stem = "fac"
    assert build.main(_argv(challenge, "fac", ["--alpha-table", str(t)])) == 0
    (cell0_a, bulk0_a), (cell0_b, bulk0_b) = seen["plain"]
    (cell1_a, bulk1_a), (cell1_b, bulk1_b) = seen["fac"]
    # A (named, 1.5): every gene but its own pinned one scaled by 1.5 in both channels; its own gene pinned
    assert np.allclose(cell1_a[1:], cell0_a[1:] * 1.5) and np.allclose(bulk1_a[1:], bulk0_a[1:] * 1.5)
    assert cell1_a[0] == cell0_a[0] and bulk1_a[0] == bulk0_a[0]
    assert np.abs(cell0_a[1:]).max() > 0                              # the contrast is not vacuous
    # B (unnamed): bit-identical to the plain build
    assert np.array_equal(cell1_b, cell0_b) and np.array_equal(bulk1_b, bulk0_b)
    rec = json.loads((challenge["out"] / "fac.alpha_table.json").read_text())
    assert rec["rows"] == 2 and rec["matched"] == 1 and rec["unmatched_in_table"] == 1 and rec["path"] == str(t)
    assert not (challenge["out"] / "plain.alpha_table.json").exists()
    assert json.loads((challenge["out"] / "fac.args.json").read_text())["alpha_table"] == str(t)


def test_build_refuses_the_flag_off_delta_transfer_and_under_a_model_name(challenge, tmp_path):
    t = _table(tmp_path / "f.parquet", [("A", 0.5)])
    argv = _argv(challenge, "null", ["--alpha-table", str(t)])
    argv[argv.index("delta-transfer")] = "control-null"
    with pytest.raises(SystemExit):
        build.main(argv)
    with pytest.raises(SystemExit):
        build.main(_argv(challenge, "ser-99a_delta_test_v1", ["--alpha-table", str(t)]))


# ------------------------------------------------------------------------------------ eval.loco

def test_loco_all_ones_is_bit_identical_and_a_real_table_moves_the_cells(tmp_path):
    from sidechain.eval import loco

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
    srcs = [line(1), line(2)]
    basal = rng.uniform(100, 2000, size=len(genes))
    rows, labels = [], []
    for _ in range(30):
        rows.append(rng.poisson(basal * 40 / basal.sum() * 1000)); labels.append("non-targeting")
    for t in targets[:2]:
        for _ in range(8):
            rows.append(rng.poisson(basal * 40 / basal.sum() * 1000)); labels.append(t)
    real = ad.AnnData(X=sp.csr_matrix(np.asarray(rows, dtype=np.float32)),
                      obs=pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(len(rows))]),
                      var=pd.DataFrame(index=genes.astype(str)))
    real_path = tmp_path / "real.h5ad"
    real.write_h5ad(real_path)
    kw = dict(pert_col="perturbation", control="non-targeting", shrinkage=False, var_floor="poisson",
              min_libsize=0.0, seed=0, alpha=1.35)
    off = loco.build_transfer_prediction(real_path, srcs, tmp_path / "off.h5ad", **kw)
    ones = _table(tmp_path / "ones.parquet", [(t, 1.0) for t in targets[:2]])
    same = loco.build_transfer_prediction(real_path, srcs, tmp_path / "same.h5ad", alpha_table=ones, **kw)
    assert off["alpha_table"] is None and same["alpha_table"]["matched"] == 2
    a, b = ad.read_h5ad(tmp_path / "off.h5ad"), ad.read_h5ad(tmp_path / "same.h5ad")
    assert (a.X != b.X).nnz == 0
    zero = _table(tmp_path / "zero.parquet", [(targets[0], 0.0)])
    on = loco.build_transfer_prediction(real_path, srcs, tmp_path / "on.h5ad", alpha_table=zero, **kw)
    assert on["alpha_table"]["zero"] == 1 and on["alpha_table"]["matched"] == 1
    c = ad.read_h5ad(tmp_path / "on.h5ad")
    assert c.shape == a.shape and (a.X != c.X).nnz > 0
