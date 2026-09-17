"""`submit.build --alpha-bulk` (letter b, ADR 0005): the pseudobulk channel at its own amplitude.

The emitter's numerics are pinned in `test_dual_moment.py`; what is pinned here is the
builder's wiring: unset, `emit_dual` is never touched; set, every (context, perturbation)
block goes through it with the SAME pooled vector at two amplitudes; the knob is recorded;
and the two configurations that cannot carry it are refused before any work.
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

GENES = ["A", "B", "C"]


def _cache(path, ctrl_label, ctrl_cpm, pert_cpm, n_cells=1000):
    m = np.asarray([ctrl_cpm, pert_cpm], dtype=float)
    n = np.full(2, n_cells, dtype=np.int64)
    PseudobulkSums(
        labels=[ctrl_label, "TP53"], genes=np.array(GENES, dtype=object),
        count_sum=m * n[:, None], cpm_sum=m * n[:, None],
        cpm_sq_sum=(m**2 + 1.0) * n[:, None],
        n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["test"],
    ).save(path)


def _controls_h5ad(path, counts_per_cell, depths):
    base = np.asarray(counts_per_cell, float)
    X = sp.csr_matrix(np.stack([np.rint(base * d / base.sum()) for d in depths]))
    obs = pd.DataFrame(index=[f"c{i}" for i in range(len(depths))])
    ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=GENES)).write_h5ad(path)


@pytest.fixture
def challenge(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    (data / "gene_names.csv").write_text("gene_name\n" + "\n".join(GENES) + "\n")
    (data / "pert_counts.csv").write_text("target_gene\nTP53\n")
    depths = [1500, 2000, 2500, 3000, 1800, 2200, 2700, 1600, 2400, 2900, 2100, 1900]
    _controls_h5ad(data / "ctx_x.h5ad", [10, 1000, 990], depths)
    _controls_h5ad(data / "ctx_y.h5ad", [1000, 505, 495], depths)
    for name, label in (("h1.npz", "non-targeting"), ("gwps.npz", "control")):
        _cache(data / name, label, [100000.0, 450000.0, 450000.0], [200000.0, 400000.0, 400000.0])
    cfg = {
        "data_dir": str(data), "gene_names_file": "gene_names.csv", "n_genes": 3,
        "pert_counts_file": "pert_counts.csv", "pert_col": "target_gene",
        "context_col": "context", "control_label": "non-targeting",
        "phase": "p1", "phases": {"p1": {"contexts": ["X", "Y"]}},
        "control_files": {"X": "ctx_x.h5ad", "Y": "ctx_y.h5ad"},
        "submission": {"cells_per_pert": 6, "max_counts_per_cell": 1_000_000,
                       "max_cells": 100_000, "max_stored_entries": 10_000_000},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return {"cfg": cfg_path, "data": data, "out": tmp_path / "out"}


def _argv(ch, stem, extra):
    return ["--challenge-config", str(ch["cfg"]), "--emitter", "delta-transfer",
            "--h1-cache", str(ch["data"] / "h1.npz"), "--gwps-cache", str(ch["data"] / "gwps.npz"),
            "--out", str(ch["out"] / stem), "--no-pack", "--min-libsize", "100", "--no-shrink", *extra]


def test_unset_never_touches_emit_dual(challenge, monkeypatch):
    def boom(self, *a, **k):
        raise AssertionError("emit_dual called without --alpha-bulk")
    monkeypatch.setattr(PoissonEmitter, "emit_dual", boom)
    assert build.main(_argv(challenge, "plain", ["--alpha", "1.35", "--emit-lambda", "0.5"])) == 0
    assert not (challenge["out"] / "plain.dual.json").exists()
    assert json.loads((challenge["out"] / "plain.args.json").read_text())["alpha_bulk"] is None


def test_set_every_block_carries_one_pooled_vector_at_two_amplitudes(challenge, monkeypatch):
    seen = []

    def spy(self, n, log2fc_cell, log2fc_bulk, **kw):
        seen.append((self.p.name, n, np.array(log2fc_cell), np.array(log2fc_bulk), kw))
        return self.emit(n, log2fc_cell)          # the fit's numerics are test_dual_moment's job
    monkeypatch.setattr(PoissonEmitter, "emit_dual", spy)
    rc = build.main(_argv(challenge, "dual", ["--alpha", "1.35", "--alpha-bulk", "1.5",
                                              "--emit-lambda", "0.5"]))
    assert rc == 0
    assert [(c, n) for c, n, *_ in seen] == [("X", 6), ("Y", 6)]      # one block per (context, pert)
    for _c, _n, cell, bulk, kw in seen:
        assert kw == {"on_fail": "fallback"}
        assert np.abs(cell).max() > 0
        assert np.allclose(bulk, cell * (1.5 / 1.35))                  # same vector, two amplitudes
    assert json.loads((challenge["out"] / "dual.args.json").read_text())["alpha_bulk"] == 1.5
    rec = json.loads((challenge["out"] / "dual.dual.json").read_text())
    assert rec == {"alpha": 1.35, "alpha_bulk": 1.5, "dual_fallbacks": {"X": 0, "Y": 0}}
    pred = ad.read_h5ad(challenge["out"] / "dual.h5ad")
    assert pred.n_obs == 12


def test_the_two_configurations_that_cannot_carry_it_are_refused(challenge):
    with pytest.raises(SystemExit):                                     # lambda 0: no depth spread
        build.main(_argv(challenge, "even", ["--alpha-bulk", "1.5"]))
    with pytest.raises(SystemExit):
        build.main(_argv(challenge, "lam0", ["--alpha-bulk", "1.5", "--emit-lambda", "0"]))
    argv = _argv(challenge, "null", ["--alpha-bulk", "1.5", "--emit-lambda", "0.5"])
    argv[argv.index("delta-transfer")] = "control-null"
    with pytest.raises(SystemExit):
        build.main(argv)
