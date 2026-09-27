"""The neighbour arm (T103): the geometry gate's cross-mode fusion in the shipping path.

What is pinned, in the order it would cost us if it slipped:

* **Off is off.** Without ``--neighbour-w`` neither entry point builds, loads or calls anything
  of the arm, so every existing arm is untouched; a half-set arm is refused before any work.
* **The port IS the gate.** On one footing, the direction of ``out - m`` equals the gate's
  ``unit(ser) + w * unit(knn_mean(...))`` computed with the gate's own functions, and its length
  is SER's residual length -- so a w read off the gate's sweep means the same thing here.
* **No self-knockdown leaks.** A neighbour's own-gene value (its knockdown) never reaches
  another target's prediction, and a target is never its own neighbour.
* **Never zero-filled.** A target the table lacks keeps SER's delta; pool members the table
  lacks or no source covers are dropped and counted; retired symbols resolve.
* **Both entry points fuse and record.** `submit.build` writes ``<out>.neighbour.json`` and
  emits the fused shift; `eval.loco` records the arm in its build info and passes the flags on.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
import yaml

from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.models import neighbour_arm as na
from sidechain.models.count_emitters import PoissonEmitter
from sidechain.submit import build

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "esm2_geometry_gate.py"


def load_gate():
    spec = importlib.util.spec_from_file_location("gate_for_neighbour_arm", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world(n_targets=40, n_genes=60, dim=8, seed=0, on_axis=10):
    """Pooled deltas for n_targets, an axis on which the first `on_axis` targets are genes."""
    rng = np.random.default_rng(seed)
    targets = [f"T{i:03d}" for i in range(n_targets)]
    axis = np.array(targets[:on_axis] + [f"G{i:03d}" for i in range(n_genes - on_axis)])
    deltas = {t: rng.normal(size=n_genes) + 0.3 for t in targets}      # +0.3: a shared mean
    for i, t in enumerate(targets[:on_axis]):
        deltas[t][i] = -2.0 - rng.random()                             # its own knockdown
    table = {t: rng.normal(size=dim) for t in targets}
    return targets, axis, deltas, table


# ------------------------------------------------------------------------ the port is the gate

@pytest.mark.parametrize("w", [0.05, 0.15, 0.5, 2.0])
@pytest.mark.parametrize("k", [1, 10, 25])
def test_port_equals_the_gate_fusion_term_for_term(w, k):
    g = load_gate()
    targets, axis, deltas, table = _world()
    arm = na.build_neighbour_arm(targets, deltas.get, axis, table, k=k, w=w)

    # the gate's own arithmetic on the same footing: own genes zeroed, centred over the targets
    A = np.stack([deltas[t].copy() for t in targets])
    pos = {s: i for i, s in enumerate(axis)}
    for r, t in enumerate(targets):
        if t in pos:
            A[r, pos[t]] = 0.0
    A = A - A.mean(0)
    e = np.stack([table[t] for t in targets])
    se = g.unit(e) @ g.unit(e).T
    gate_fused = g.unit(g.unit(A) + w * g.unit(g.knn_mean(se, A, k)))

    for r, t in enumerate(targets):
        out = arm.fuse(t, deltas[t])
        resid = out - arm.mean
        np.testing.assert_allclose(na.unit(resid), gate_fused[r], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(resid), np.linalg.norm(A[r]), rtol=1e-9)
    assert arm.stats["targets_fused"] == len(targets)
    assert arm.summary()["pool_used"] == len(targets)


def test_fusion_moves_direction_only_and_is_continuous_at_small_w():
    targets, axis, deltas, table = _world()
    arm = na.build_neighbour_arm(targets, deltas.get, axis, table, k=5, w=1e-9)
    t = targets[20]                       # not on the axis: its delta is untouched by the pin
    np.testing.assert_allclose(arm.fuse(t, deltas[t]), deltas[t], atol=1e-7)


# -------------------------------------------------------------------------- no leaks, no self

def test_a_neighbours_own_knockdown_never_reaches_another_target():
    targets, axis, deltas, table = _world()
    loud = {t: d.copy() for t, d in deltas.items()}
    for i, t in enumerate(targets[:10]):
        loud[t][i] = -50.0                                  # an absurd self-knockdown
    quiet = na.build_neighbour_arm(targets, deltas.get, axis, table, k=10, w=0.5)
    noisy = na.build_neighbour_arm(targets, loud.get, axis, table, k=10, w=0.5)
    for t in targets:
        np.testing.assert_allclose(noisy.fuse(t, loud[t]), quiet.fuse(t, deltas[t]), atol=1e-12)


def test_a_target_is_never_its_own_neighbour():
    targets, axis, deltas, table = _world()
    t, twin = targets[30], targets[31]
    table = dict(table)
    table[twin] = table[t] + 1e-6                            # t's nearest OTHER target
    arm = na.build_neighbour_arm(targets, deltas.get, axis, table, k=1, w=0.4)
    out = arm.fuse(t, deltas[t])
    r = deltas[t] - arm.mean
    n = arm.resid[arm.pool.index(twin)]
    expect = arm.mean + np.linalg.norm(r) * na.unit(na.unit(r) + 0.4 * na.unit(n))
    np.testing.assert_allclose(out, expect, atol=1e-12)
    assert arm.stats["own_in_pool"] == 1


def test_a_target_outside_the_pool_excludes_nothing():
    targets, axis, deltas, table = _world()
    table = dict(table)
    table[targets[1]] = table[targets[0]].copy()            # the pool member identical to t
    arm = na.build_neighbour_arm(targets[1:], deltas.get, axis, table, k=1, w=1.0)
    t = targets[0]                                          # on the axis, not in the pool
    out = arm.fuse(t, deltas[t])
    d = deltas[t].copy(); d[0] = 0.0                        # t's own gene zeroed before r
    r = d - arm.mean
    n = arm.resid[0]                                        # targets[1], nothing excluded
    np.testing.assert_allclose(out, arm.mean + np.linalg.norm(r) * na.unit(na.unit(r) + na.unit(n)),
                               atol=1e-12)
    assert arm.stats.get("own_in_pool", 0) == 0


# ----------------------------------------------------------------------------- never zero-filled

def test_unresolved_target_keeps_sers_delta_and_is_counted():
    targets, axis, deltas, table = _world()
    arm = na.build_neighbour_arm(targets, deltas.get, axis, table, k=5, w=0.2)
    d = np.arange(len(axis), dtype=float)
    assert arm.fuse("NOT_IN_TABLE", d) is d
    assert arm.stats["targets_unresolved"] == 1


def test_pool_drops_unresolved_and_uncovered_and_resolves_retired_symbols():
    targets, axis, deltas, table = _world()
    table = dict(table)
    table["QARS1"] = table.pop(targets[5])                  # the table speaks the current name
    deltas = dict(deltas)
    deltas["QARS"] = deltas.pop(targets[5])                 # the corpus speaks the retired one
    pool = targets[:5] + ["QARS"] + targets[6:] + ["NO_TABLE_ROW"]
    table_extra = dict(table, UNCOVERED=np.ones(8))
    arm = na.build_neighbour_arm(pool + ["UNCOVERED"], deltas.get, axis, table_extra, k=5, w=0.2)
    assert "QARS" in arm.pool
    assert arm.stats["pool_unresolved"] == 1                # NO_TABLE_ROW
    assert arm.stats["pool_uncovered"] == 1                 # UNCOVERED: no source measured it
    assert len(arm.pool) == len(targets)


def test_a_pool_no_larger_than_k_is_refused_and_w_must_be_positive():
    targets, axis, deltas, table = _world(n_targets=6)
    with pytest.raises(ValueError, match="not more than k"):
        na.build_neighbour_arm(targets, deltas.get, axis, table, k=6, w=0.2)
    with pytest.raises(ValueError, match="w must be finite and > 0"):
        na.build_neighbour_arm(targets, deltas.get, axis, table, k=2, w=0.0)


# --------------------------------------------------------------------------- submit.build wiring

N_PERTS = 12
GENES = [f"P{i:02d}" for i in range(N_PERTS)] + ["X1", "X2", "X3", "X4"]


def _cache(path, ctrl_label, perts, seed):
    rng = np.random.default_rng(seed)
    labels = [ctrl_label] + perts
    m = rng.uniform(2e4, 8e4, size=(len(labels), len(GENES)))
    m = m / m.sum(1, keepdims=True) * 1e6
    n = np.full(len(labels), 500, dtype=np.int64)
    PseudobulkSums(labels=labels, genes=np.array(GENES, dtype=object),
                   count_sum=m * n[:, None], cpm_sum=m * n[:, None],
                   cpm_sq_sum=(m**2 + 1.0) * n[:, None], n_cells=n,
                   libsize_sum=n.astype(float) * 2e4, sources=["test"]).save(path)


def _controls(path, seed):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(200, size=(12, len(GENES))).astype(float))
    ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(12)]),
               var=pd.DataFrame(index=GENES)).write_h5ad(path)


@pytest.fixture
def challenge(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    perts = GENES[:N_PERTS]
    (data / "gene_names.csv").write_text("gene_name\n" + "\n".join(GENES) + "\n")
    (data / "pert_counts.csv").write_text("target_gene\n" + "\n".join(perts[:4]) + "\n")
    _controls(data / "ctx_x.h5ad", 1)
    _cache(data / "h1.npz", "non-targeting", perts[:3], 2)
    _cache(data / "gwps.npz", "control", perts, 3)
    rng = np.random.default_rng(4)
    torch.save({p: torch.tensor(rng.normal(size=6), dtype=torch.float32) for p in perts},
               data / "table.pt")
    (data / "pool.csv").write_text("target_gene\n" + "\n".join(perts + ["NOWHERE"]) + "\n")
    cfg = {"data_dir": str(data), "gene_names_file": "gene_names.csv", "n_genes": len(GENES),
           "pert_counts_file": "pert_counts.csv", "pert_col": "target_gene",
           "context_col": "context", "control_label": "non-targeting",
           "phase": "p1", "phases": {"p1": {"contexts": ["X"]}},
           "control_files": {"X": "ctx_x.h5ad"},
           "submission": {"cells_per_pert": 4, "max_counts_per_cell": 1_000_000,
                          "max_cells": 100_000, "max_stored_entries": 10_000_000}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return {"cfg": cfg_path, "data": data, "out": tmp_path / "out"}


def _argv(ch, stem, extra=()):
    return ["--challenge-config", str(ch["cfg"]), "--emitter", "delta-transfer",
            "--h1-cache", str(ch["data"] / "h1.npz"), "--gwps-cache", str(ch["data"] / "gwps.npz"),
            "--out", str(ch["out"] / stem), "--no-pack", "--min-libsize", "100", "--no-shrink",
            "--alpha", "1.35", *extra]


def _arm_flags(ch, w="0.3", k="3"):
    return ["--neighbour-table", str(ch["data"] / "table.pt"),
            "--neighbour-pool", str(ch["data"] / "pool.csv"), "--neighbour-k", k,
            "--neighbour-w", w]


def _capture_emits(monkeypatch):
    seen = []
    real = PoissonEmitter.emit

    def spy(self, n, log2fc=None, **kw):
        seen.append(None if log2fc is None else np.array(log2fc))
        return real(self, n, log2fc, **kw)

    monkeypatch.setattr(PoissonEmitter, "emit", spy)
    return seen


def test_build_off_never_touches_the_arm(challenge, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the neighbour arm was built without --neighbour-w")
    monkeypatch.setattr(build, "neighbour_arm_for", boom)
    monkeypatch.setattr(na, "load_gene_table", boom)
    assert build.main(_argv(challenge, "plain")) == 0
    args = json.loads((challenge["out"] / "plain.args.json").read_text())
    assert args["neighbour_w"] == 0.0 and args["neighbour_table"] is None
    assert not (challenge["out"] / "plain.neighbour.json").exists()


@pytest.mark.parametrize("extra, why", [
    (["--neighbour-w", "0.2"], "needs both"),
    (["--neighbour-w", "0.2", "--neighbour-table", "t.pt"], "needs both"),
    (["--neighbour-table", "t.pt"], "do nothing"),
    (["--neighbour-w", "-0.1"], "finite and >= 0"),
    (["--neighbour-w", "inf"], "finite and >= 0"),
    (["--neighbour-w", "nan"], "finite and >= 0"),
])
def test_build_refuses_a_half_set_arm(challenge, capsys, extra, why):
    with pytest.raises(SystemExit):
        build.main(_argv(challenge, "bad", extra))
    assert why in capsys.readouterr().err


def test_build_refuses_the_arm_with_gamma(challenge, capsys):
    with pytest.raises(SystemExit):
        build.main(_argv(challenge, "bad", [*_arm_flags(challenge), "--gamma", "0.5"]))
    assert "gamma" in capsys.readouterr().err


def test_build_emits_the_fused_shift_and_records_the_arm(challenge, monkeypatch):
    seen_off = _capture_emits(monkeypatch)
    assert build.main(_argv(challenge, "off")) == 0
    off = list(seen_off); seen_off.clear()
    assert build.main(_argv(challenge, "on", _arm_flags(challenge))) == 0
    on = list(seen_off)
    rec = json.loads((challenge["out"] / "on.neighbour.json").read_text())
    assert rec["targets_fused"] == 4 and rec["k"] == 3 and rec["w"] == 0.3
    assert rec["pool_requested"] == N_PERTS + 1 and rec["pool_unresolved"] == 1   # NOWHERE
    assert len(rec["table_sha256"]) == 64 and len(rec["pool_sha256"]) == 64

    # the expected fused shift, rebuilt from the pieces the build used
    axis = np.array(GENES)
    h1 = PseudobulkSums.load(challenge["data"] / "h1.npz")
    gw = PseudobulkSums.load(challenge["data"] / "gwps.npz")
    sources = [(gw, "control"), (h1, "non-targeting")]

    def delta_of(lab):
        return build.pooled_delta(lab, sources, axis, shrinkage=False)

    table = na.load_gene_table(challenge["data"] / "table.pt")
    arm = na.build_neighbour_arm(na.read_pool(challenge["data"] / "pool.csv"), delta_of, axis,
                                 table, k=3, w=0.3)
    for i, p in enumerate(GENES[:4]):
        expect = arm.fuse(p, delta_of(p)) * 1.35
        expect[i] = build.TARGET_SELF_LOG2FC
        np.testing.assert_allclose(on[i], expect, atol=1e-12)
        assert not np.allclose(on[i], off[i])                       # the arm did something
        np.testing.assert_allclose(on[i][i], off[i][i])             # the pin is untouched


# ------------------------------------------------------------------------------ eval.loco wiring

def test_loco_passes_the_flags_through_and_logs_them(monkeypatch, tmp_path, challenge):
    from sidechain.eval import loco

    captured, logged = {}, {}
    monkeypatch.setattr(loco, "build_transfer_prediction",
                        lambda real, sources, out_path, **kw: captured.update(kw) or {})
    monkeypatch.setattr(loco, "attach_controls", lambda pred, real, out, **kw: out)
    monkeypatch.setattr(loco, "score", lambda *a, **kw: {"overall": 0.0, "members": {}})
    monkeypatch.setattr(loco, "log_run",
                        lambda params, results, artifacts=None: logged.update(params))
    monkeypatch.setattr(PseudobulkSums, "load", classmethod(lambda cls, p: type("PB", (), {})()))
    rc = loco.main(["--real", "r.h5ad", "--bundle", "b", "--out", str(tmp_path / "arm"),
                    "--source", "x.npz:ctl", *_arm_flags(challenge, w="0.15", k="2")])
    assert rc == 0
    assert captured["neighbour_w"] == 0.15 and captured["neighbour_k"] == 2
    assert Path(captured["neighbour_table"]).name == "table.pt"
    assert logged["neighbour_w"] == 0.15 and logged["neighbour_pool"].endswith("pool.csv")


def test_loco_refuses_the_arm_with_basal_slope(tmp_path, challenge, capsys):
    from sidechain.eval import loco

    with pytest.raises(SystemExit):
        loco.main(["--real", "r.h5ad", "--bundle", "b", "--out", str(tmp_path / "arm"),
                   "--source", "x.npz:ctl", "--basal-slope", "gene", *_arm_flags(challenge)])
    assert "basal" in capsys.readouterr().err


def test_loco_fuses_end_to_end_and_off_is_unchanged(tmp_path, challenge):
    from sidechain.eval.loco import build_transfer_prediction

    rng = np.random.default_rng(7)
    perts = GENES[:4]
    labels = ["non-targeting"] * 20 + [p for p in perts for _ in range(5)]
    X = sp.csr_matrix(rng.poisson(200, size=(len(labels), len(GENES))).astype(float))
    real = tmp_path / "real.h5ad"
    ad.AnnData(X=X, obs=pd.DataFrame({"target_gene": labels},
                                     index=[f"c{i}" for i in range(len(labels))]),
               var=pd.DataFrame(index=GENES)).write_h5ad(real)
    gw = PseudobulkSums.load(challenge["data"] / "gwps.npz")
    kw = dict(pert_col="target_gene", control="non-targeting", shrinkage=False, alpha=1.35,
              seed=0, min_libsize=100)
    info_off = build_transfer_prediction(real, [(gw, "control")], tmp_path / "off.h5ad", **kw)
    info_def = build_transfer_prediction(real, [(gw, "control")], tmp_path / "def.h5ad", **kw,
                                         neighbour_w=0.0)
    info_on = build_transfer_prediction(
        real, [(gw, "control")], tmp_path / "on.h5ad", **kw,
        neighbour_table=challenge["data"] / "table.pt",
        neighbour_pool=challenge["data"] / "pool.csv", neighbour_k=3, neighbour_w=0.3)
    assert info_off["neighbour"] is None and info_def["neighbour"] is None
    assert info_on["neighbour"]["targets_fused"] == 4
    assert info_on["neighbour"]["pool_used"] == N_PERTS
    x_off = ad.read_h5ad(tmp_path / "off.h5ad").X
    x_def = ad.read_h5ad(tmp_path / "def.h5ad").X
    x_on = ad.read_h5ad(tmp_path / "on.h5ad").X
    assert (x_off != x_def).nnz == 0                       # off is bit-identical
    assert (x_off != x_on).nnz > 0                         # on moves the emitted cells


def test_a_zero_or_non_finite_table_row_counts_as_absent():
    targets, axis, deltas, table = _world()
    table = dict(table)
    table[targets[3]] = np.zeros(8)
    table[targets[4]] = np.full(8, np.nan)
    arm = na.build_neighbour_arm(targets, deltas.get, axis, table, k=5, w=0.2)
    assert arm.stats["pool_unresolved"] == 2
    assert targets[3] not in arm.pool and targets[4] not in arm.pool
    assert arm.fuse(targets[3], deltas[targets[3]]) is deltas[targets[3]]


def test_the_pool_is_pooled_with_the_targets_own_knobs(challenge, monkeypatch):
    """Every pooled_delta call -- targets and pool members alike -- carries the same floor,
    tiers and bias correction; a pool pooled differently would put SER's residual and its
    neighbours' in different spaces."""
    seen = []
    real = build.pooled_delta

    def spy(target, sources, axis, **kw):
        seen.append((target, kw.get("var_floor"), kw.get("coverage_tiers"),
                     kw.get("log_bias_correct"), kw.get("shrinkage")))
        return real(target, sources, axis, **kw)

    monkeypatch.setattr(build, "pooled_delta", spy)
    assert build.main(_argv(challenge, "knobs", [*_arm_flags(challenge), "--var-floor", "poisson",
                                                 "--coverage-tiers", "3:0.1,10:0.5",
                                                 "--log-bias-correct"])) == 0
    pool_only = [x for x in seen if x[0] not in GENES[:4]]
    assert len(pool_only) == N_PERTS - 4                     # NOWHERE has no table row
    assert {x[1:] for x in seen} == {("poisson", ((3.0, 0.1), (10.0, 0.5)), True, False)}


def test_gwps_control_names_the_cache_control_and_defaults_to_control(challenge):
    assert build.main(_argv(challenge, "ctl")) == 0
    args = json.loads((challenge["out"] / "ctl.args.json").read_text())
    assert args["gwps_control"] == "control"
    with pytest.raises(ValueError):                  # the fixture cache has no such label
        build.main(_argv(challenge, "ctl2", ["--gwps-control", "non-targeting"]))
