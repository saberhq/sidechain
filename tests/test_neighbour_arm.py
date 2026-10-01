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
* **The default blend is the gate's, to the bit.** ``--neighbour-size median`` scales the
  neighbour arm by the pool's median residual length instead of normalising it; ``unit``, the
  default, runs the same floating-point operations in the same order as before the flag existed
  -- and the expectation re-derives the neighbour pick instead of calling the method the flag
  refactored, so the rows picked and their order are pinned too.
* **A median run can be read and replayed.** ``|n|/s`` is reported with its own value (per table
  in a mix), and the scale is recorded unrounded as well, because the rounded one cannot
  reproduce ``fuse`` to the 1e-10 the screen's gate asks of it.
"""
from __future__ import annotations

import argparse
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
    assert args["neighbour_w"] is None and args["neighbour_table"] is None
    assert not (challenge["out"] / "plain.neighbour.json").exists()


@pytest.mark.parametrize("extra, why", [
    (["--neighbour-w", "0.2"], "needs both"),
    (["--neighbour-w", "0.2", "--neighbour-table", "t.pt"], "needs both"),
    (["--neighbour-table", "t.pt"], "--neighbour-table without --neighbour-w does nothing"),
    (["--neighbour-table", "t.pt", "--neighbour-pool", "p.csv", "--neighbour-size", "median"],
     "--neighbour-table, --neighbour-pool, --neighbour-size median without --neighbour-w "
     "do nothing"),
    (["--neighbour-w", "-0.1"], "finite and > 0"),
    (["--neighbour-w", "inf"], "finite and > 0"),
    (["--neighbour-w", "nan"], "finite and > 0"),
    (["--neighbour-w", "0"], "finite and > 0"),
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
    assert captured["neighbour_w"] == [0.15] and captured["neighbour_k"] == 2
    assert Path(captured["neighbour_table"][0]).name == "table.pt"
    assert logged["neighbour_w"] == [0.15] and logged["neighbour_pool"].endswith("pool.csv")


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


# ---------------------------------------------------------------------- two tables at once (mix)

def test_a_one_table_mix_is_the_single_arm_bit_for_bit():
    targets, axis, deltas, table = _world()
    one = na.build_neighbour_arms(targets, deltas.get, axis, [table], k=10, ws=[0.3])
    ref = na.build_neighbour_arm(targets, deltas.get, axis, table, k=10, w=0.3)
    assert isinstance(one, na.NeighbourArm)
    for t in targets:
        assert np.array_equal(one.fuse(t, deltas[t]), ref.fuse(t, deltas[t]))


def test_two_tables_sum_their_unit_neighbour_arms_on_one_shared_pool():
    targets, axis, deltas, t1 = _world(seed=0)
    _, _, _, t2 = _world(seed=1)
    t2 = {k: v for k, v in t2.items() if k != targets[7]}          # table 2 lacks one target
    mix = na.build_neighbour_arms(targets, deltas.get, axis, [t1, t2], k=5, ws=[0.2, 0.4])
    assert targets[7] not in mix.arms[0].pool                       # the pool both tables resolve
    a1 = na.build_neighbour_arm(mix.arms[0].pool, deltas.get, axis, t1, k=5, w=0.2)
    a2 = na.build_neighbour_arm(mix.arms[0].pool, deltas.get, axis, t2, k=5, w=0.4)
    for t in targets[10:20]:                                        # off the axis: no pin
        r = deltas[t] - a1.mean
        u1, _ = a1.neighbour_unit(t, na.table_vector(t1, t))
        u2, _ = a2.neighbour_unit(t, na.table_vector(t2, t))
        want = a1.mean + np.linalg.norm(r) * na.unit(na.unit(r) + 0.2 * u1 + 0.4 * u2)
        np.testing.assert_allclose(mix.fuse(t, deltas[t]), want, atol=1e-12)
    out = mix.fuse(targets[7], deltas[targets[7]])                  # only table 1 resolves it
    assert mix.stats["table1_unresolved"] == 1 and not np.allclose(out, deltas[targets[7]])
    summ = mix.summary()
    assert summ["w"] == [0.2, 0.4] and "table0_cosine_mean" in summ


def test_build_blends_two_tables_and_refuses_unpaired_weights(challenge, capsys):
    rng = np.random.default_rng(9)
    t2 = challenge["data"] / "table2.pt"
    torch.save({p: torch.tensor(rng.normal(size=4), dtype=torch.float32)
                for p in GENES[:N_PERTS]}, t2)
    flags = ["--neighbour-table", str(challenge["data"] / "table.pt"), "--neighbour-w", "0.2",
             "--neighbour-table", str(t2), "--neighbour-w", "0.1",
             "--neighbour-pool", str(challenge["data"] / "pool.csv"), "--neighbour-k", "3"]
    assert build.main(_argv(challenge, "mix", flags)) == 0
    rec = json.loads((challenge["out"] / "mix.neighbour.json").read_text())
    assert rec["w"] == [0.2, 0.1] and len(rec["table_sha256"]) == 2 and rec["targets_fused"] == 4
    with pytest.raises(SystemExit):
        build.main(_argv(challenge, "bad", flags[:-6] + ["--neighbour-table", str(t2)]
                         + flags[-4:]))
    assert "one --neighbour-w per --neighbour-table" in capsys.readouterr().err


# ------------------------------------------------------- the size-aware blend (direction 1 (i))

def _median_scale(arm):
    """s, re-derived from the arm's own residual rows rather than read off the arm."""
    return float(np.median(np.linalg.norm(arm.resid, axis=1)))


def _mean_of_the_k_nearest(arm, table, t, k):
    """``n_t`` re-derived WITHOUT the arm's own pick, so a change to `neighbour_mean` moves
    `fuse` and not this expectation: the rows chosen, their order and the flat mean are all
    pinned, not only `fuse`'s own arithmetic.
    """
    sim = arm.pool_unit @ na.unit(na.table_vector(table, t))
    if t in arm.pool:
        sim[arm.pool.index(t)] = -np.inf                 # never its own neighbour
    return arm.resid[np.argsort(-sim)[:k]].mean(0)


def test_size_unit_default_is_bit_identical():
    """The default and an explicit size="unit" are the same numbers, and both are the
    pre-flag arithmetic operation for operation (np.array_equal, not allclose).

    The expectation re-derives the neighbour pick itself (`_mean_of_the_k_nearest`) rather than
    calling the method the size flag refactored, so the pick's row ORDER is pinned here too; the
    pick's agreement with an outside implementation is pinned by
    `test_port_equals_the_gate_fusion_term_for_term`, which reads the gate's own `knn_mean`.
    """
    targets, axis, deltas, table = _world()
    default = na.build_neighbour_arm(targets, deltas.get, axis, table, k=10, w=0.3)
    explicit = na.build_neighbour_arm(targets, deltas.get, axis, table, k=10, w=0.3, size="unit")
    for t in targets:
        out = default.fuse(t, deltas[t])
        assert np.array_equal(out, explicit.fuse(t, deltas[t]))
        d = np.array(deltas[t], dtype=float)
        if t in default.gene_pos:
            d[default.gene_pos[t]] = 0.0
        r = d - default.mean
        un = na.unit(_mean_of_the_k_nearest(default, table, t, 10))
        want = default.mean + float(np.linalg.norm(r)) * na.unit(na.unit(r) + 0.3 * un)
        assert np.array_equal(out, want)
    summ = default.summary()
    assert summ["size"] == "unit" and summ["scale"] == round(_median_scale(default), 6)
    assert summ["scale_exact"] == default.scale
    assert "arm_size_mean" not in summ and "arm_size_gt1_frac" not in summ


def test_size_median_matches_the_formula():
    targets, axis, deltas, table = _world()
    arm = na.build_neighbour_arm(targets, deltas.get, axis, table, k=5, w=0.7, size="median")
    unit_arm = na.build_neighbour_arm(targets, deltas.get, axis, table, k=5, w=0.7)
    s = _median_scale(arm)
    assert s > 0 and arm.scale == s
    moved, sizes = 0, []
    for t in targets[12:20]:                       # off the axis: no own-gene pin to undo
        n, _own = arm.neighbour_mean(t, na.table_vector(table, t))
        r = deltas[t] - arm.mean
        want = arm.mean + np.linalg.norm(r) * na.unit(na.unit(r) + 0.7 * n / s)
        np.testing.assert_allclose(arm.fuse(t, deltas[t]), want, atol=1e-12)
        moved += int(not np.allclose(want, unit_arm.fuse(t, deltas[t])))
        sizes.append(float(np.linalg.norm(n)) / s)
    assert moved == 8                              # the flag really changes every prediction
    summ = arm.summary()
    assert summ["size"] == "median" and "arm_size_mean" in summ
    assert summ["scale"] == round(s, 6)
    # the |n|/s reading itself, not just its presence: it is the number the w grid is read on,
    # so "the arm is 0.42 of a pool-median residual" must not be able to become 3.21 (the /s
    # dropped) or "always longer than SER's residual" (the > 1 threshold slipped)
    assert summ["arm_size_mean"] == pytest.approx(float(np.mean(sizes)), abs=1e-6)
    assert summ["arm_size_gt1_frac"] == round(float(np.mean(np.array(sizes) > 1.0)), 6)
    assert 0.0 <= summ["arm_size_gt1_frac"] <= 1.0
    # and the recorded scale reproduces `fuse` bit for bit, which the rounded one cannot
    t = targets[12]
    n, _own = arm.neighbour_mean(t, na.table_vector(table, t))
    r = deltas[t] - arm.mean
    exact = arm.mean + float(np.linalg.norm(r)) * na.unit(na.unit(r)
                                                          + 0.7 * (n / summ["scale_exact"]))
    assert np.array_equal(arm.fuse(t, deltas[t]), exact)
    assert round(summ["scale_exact"], 6) == summ["scale"]


def test_size_median_counts_arms_longer_than_the_residual_at_k_1():
    """At k = 1 the arm is one member's whole residual, so |n|/s is around 1 and the `> 1`
    counter is exercised on both sides -- the one thing the k >= 5 worlds never do (there the
    frac is exactly 0.000, which a broken threshold also produces)."""
    targets, axis, deltas, table = _world(on_axis=0)
    arm = na.build_neighbour_arm(targets, deltas.get, axis, table, k=1, w=0.3, size="median")
    s = _median_scale(arm)
    want = []
    for t in targets:
        n, _own = arm.neighbour_mean(t, na.table_vector(table, t))
        want.append(float(np.linalg.norm(n)) / s)
        arm.fuse(t, deltas[t])
    gt1 = int(sum(x > 1.0 for x in want))
    assert 0 < gt1 < len(targets)                     # both sides of the threshold are hit
    assert arm.stats["arm_size_gt1"] == gt1
    summ = arm.summary()
    assert summ["arm_size_gt1_frac"] == round(gt1 / len(targets), 6)
    assert summ["arm_size_mean"] == pytest.approx(float(np.mean(want)), abs=1e-6)


def test_size_median_reduces_to_unit_r_when_neighbours_cancel():
    """A neighbourhood that cancels exactly contributes nothing: n = 0 leaves SER's direction
    alone, and w cannot matter. (So does the unit blend here -- `unit`'s 1e-12 guard sends 0 to
    0, not to an arbitrary unit vector; the two part company just off exact cancellation, which
    is the next test's job -- this one pins the exact-zero edge and the w-invariance that only
    an arm adding exactly nothing has.)"""
    axis = np.array([f"G{i:03d}" for i in range(12)])
    rng = np.random.default_rng(11)
    v1, v2 = rng.normal(size=12), rng.normal(size=12)
    deltas = {"A": v1, "B": -v1, "C": v2, "D": -v2}              # a pool with mean exactly 0
    table = {"A": np.array([1.0, 0.0]), "B": np.array([1.0, 0.0]),
             "C": np.array([0.0, 1.0]), "D": np.array([0.0, 1.0]),
             "T": np.array([1.0, 0.0])}
    arm = na.build_neighbour_arm(["A", "B", "C", "D"], deltas.get, axis, table,
                                 k=2, w=0.9, size="median")
    assert np.allclose(arm.mean, 0.0, atol=0.0)
    n, _own = arm.neighbour_mean("T", table["T"])                # A and B, residuals +-v1
    assert np.allclose(n, 0.0, atol=0.0)
    d = rng.normal(size=12)
    out = arm.fuse("T", d)
    np.testing.assert_allclose(na.unit(out - arm.mean), na.unit(d - arm.mean), atol=1e-12)
    summ = arm.summary()
    assert summ["arm_size_mean"] == 0.0 and summ["arm_size_gt1_frac"] == 0.0
    # nothing is added, so a 50x heavier weight is the same prediction to the bit -- which fails
    # the moment the median branch contributes anything at all
    loud = na.build_neighbour_arm(["A", "B", "C", "D"], deltas.get, axis, table,
                                  k=2, w=50.0, size="median")
    assert np.array_equal(loud.fuse("T", d), out)
    assert np.array_equal(out, arm.mean + float(np.linalg.norm(d - arm.mean))
                          * na.unit(na.unit(d - arm.mean)))


def test_size_median_shrinks_a_neighbourhood_the_unit_blend_trusts_in_full():
    """Just off exact cancellation the two blends diverge, which is the point of the flag:
    the unit arm pulls with full weight w in whatever direction is left over, the size-aware
    arm pulls with w |n| / s, and |n| / s is tiny when the neighbours disagree."""
    axis = np.array([f"G{i:03d}" for i in range(12)])
    rng = np.random.default_rng(12)
    v1, v2, u = rng.normal(size=12), rng.normal(size=12), rng.normal(size=12)
    eps = 1e-3
    deltas = {"A": v1, "B": -v1 + eps * u, "C": v2, "D": -v2}    # A and B nearly cancel
    table = {"A": np.array([1.0, 0.0]), "B": np.array([1.0, 0.0]),
             "C": np.array([0.0, 1.0]), "D": np.array([0.0, 1.0]),
             "T": np.array([1.0, 0.0])}
    pool, d = ["A", "B", "C", "D"], rng.normal(size=12)
    med = na.build_neighbour_arm(pool, deltas.get, axis, table, k=2, w=0.9, size="median")
    uni = na.build_neighbour_arm(pool, deltas.get, axis, table, k=2, w=0.9)
    n, _own = med.neighbour_mean("T", table["T"])
    assert np.linalg.norm(n) / med.scale < 1e-3                  # a cancelling neighbourhood
    r = d - med.mean
    turn_med = float(na.unit(med.fuse("T", d) - med.mean) @ na.unit(r))
    turn_uni = float(na.unit(uni.fuse("T", d) - uni.mean) @ na.unit(r))
    assert turn_med > 0.999999 > turn_uni                        # median barely turns, unit does


def test_neighbour_unit_is_unit_of_neighbour_mean():
    targets, axis, deltas, table = _world()
    arm = na.build_neighbour_arm(targets, deltas.get, axis, table, k=7, w=0.2)
    for t in targets[:5] + targets[20:25]:
        e = na.table_vector(table, t)
        n, own = arm.neighbour_mean(t, e)
        un, own_u = arm.neighbour_unit(t, e)
        np.testing.assert_allclose(na.unit(n), un, atol=1e-15)
        assert own == own_u


def test_build_arm_refuses_an_unknown_size_and_a_flat_pool():
    targets, axis, deltas, table = _world(on_axis=0)
    with pytest.raises(ValueError, match="size must be one of"):
        na.build_neighbour_arm(targets, deltas.get, axis, table, k=5, w=0.2, size="mean")
    flat = {t: np.ones(len(axis)) for t in targets}      # every delta equals the pool mean
    with pytest.raises(ValueError, match="median .*residual length"):
        na.build_neighbour_arm(targets, flat.get, axis, table, k=5, w=0.2, size="median")
    # the unit blend does not divide by s, so a flat pool is still a legal (useless) arm
    assert na.build_neighbour_arm(targets, flat.get, axis, table, k=5, w=0.2).scale == 0.0
    # a non-finite member poisons s as well; `scale == 0` would have let a NaN scale through and
    # every fused vector would have come back all-NaN
    nan_pool = {t: d.copy() for t, d in deltas.items()}
    nan_pool[targets[0]][3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        na.build_neighbour_arm(targets, nan_pool.get, axis, table, k=5, w=0.2, size="median")


def test_a_hand_built_median_arm_refuses_to_run_without_its_scale():
    """The analytic screen cross-checks its own fusion by constructing `NeighbourArm` directly
    (its gate is 1e-10 against `fuse`), so the median path's preconditions have to hold at
    construction, not only in the factory -- otherwise a missing scale surfaces as a TypeError
    about dividing a float by None, deep inside `fuse`."""
    targets, axis, deltas, table = _world()
    ref = na.build_neighbour_arm(targets, deltas.get, axis, table, k=2, w=0.3, size="median")
    kw = dict(k=2, w=0.3, table=table, pool=ref.pool, mean=ref.mean, resid=ref.resid,
              pool_unit=ref.pool_unit, gene_pos=ref.gene_pos)
    for bad in ({}, {"scale": 0.0}, {"scale": float("nan")}, {"scale": -1.0}):
        with pytest.raises(ValueError, match="positive scale"):
            na.NeighbourArm(**kw, size="median", **bad)
    with pytest.raises(ValueError, match="size must be one of"):
        na.NeighbourArm(**kw, size="mean", scale=ref.scale)
    hand = na.NeighbourArm(**kw, size="median", scale=ref.scale)
    for t in targets[12:16]:
        assert np.array_equal(hand.fuse(t, deltas[t]), ref.fuse(t, deltas[t]))


def test_mix_respects_size():
    targets, axis, deltas, t1 = _world(seed=0)
    _, _, _, t2 = _world(seed=1)
    mix = na.build_neighbour_arms(targets, deltas.get, axis, [t1, t2], k=5, ws=[0.2, 0.4],
                                  size="median")
    a0, a1 = mix.arms
    s = _median_scale(a0)
    assert mix.size == "median" and a0.scale == s and a1.scale == s
    sizes = ([], [])
    for t in targets[12:18]:                             # off the axis: no own-gene pin
        n1, _ = a0.neighbour_mean(t, na.table_vector(t1, t))
        n2, _ = a1.neighbour_mean(t, na.table_vector(t2, t))
        r = deltas[t] - a0.mean
        want = a0.mean + np.linalg.norm(r) * na.unit(na.unit(r) + 0.2 * n1 / s + 0.4 * n2 / s)
        np.testing.assert_allclose(mix.fuse(t, deltas[t]), want, atol=1e-12)
        sizes[0].append(float(np.linalg.norm(n1)) / s)
        sizes[1].append(float(np.linalg.norm(n2)) / s)
    summ = mix.summary()
    assert summ["size"] == "median" and summ["scale"] == round(s, 6)
    assert summ["scale_exact"] == s
    # how hard EACH table's arm pulled, per table: a median mix that records only `size` says
    # nothing about whether the size term did anything
    for i in (0, 1):
        assert summ[f"table{i}_size_mean"] == pytest.approx(float(np.mean(sizes[i])), abs=1e-6)
        assert summ[f"table{i}_size_gt1_frac"] == round(
            float(np.mean(np.array(sizes[i]) > 1.0)), 6)
    unit_mix = na.build_neighbour_arms(targets, deltas.get, axis, [t1, t2], k=5, ws=[0.2, 0.4])
    for t in targets[12:18]:
        unit_mix.fuse(t, deltas[t])
    # the unit blend has no size term, so it reports no reading -- never 0.0, which would read
    # as "both neighbourhoods cancelled"
    assert not [k for k in unit_mix.summary() if "size" in k and k != "size"]


def test_a_one_table_median_mix_is_the_single_median_arm_bit_for_bit():
    targets, axis, deltas, table = _world()
    one = na.build_neighbour_arms(targets, deltas.get, axis, [table], k=10, ws=[0.3],
                                  size="median")
    ref = na.build_neighbour_arm(targets, deltas.get, axis, table, k=10, w=0.3, size="median")
    assert isinstance(one, na.NeighbourArm) and one.size == "median"
    for t in targets:
        assert np.array_equal(one.fuse(t, deltas[t]), ref.fuse(t, deltas[t]))


def test_the_flags_size_choices_are_the_modules_sizes():
    """build.py names the two spellings so that off never imports the arm module; if
    `SIZES` grows, this is what says the flag did not."""
    ap = argparse.ArgumentParser()
    build.add_neighbour_args(ap, twin="sidechain.eval.loco")
    action = next(a for a in ap._actions if a.dest == "neighbour_size")
    assert tuple(action.choices) == na.SIZES
    assert action.default == "unit"


def test_build_refuses_size_without_w(challenge, capsys):
    with pytest.raises(SystemExit):
        build.main(_argv(challenge, "bad", ["--neighbour-size", "median"]))
    err = capsys.readouterr().err
    # one flag, so a singular verb and "drop it": the message is pasted into logs as it stands
    assert "--neighbour-size median without --neighbour-w does nothing" in err
    assert "drop it" in err and "/--neighbour" not in err
    with pytest.raises(SystemExit):                       # argparse guards the spelling
        build.main(_argv(challenge, "bad", [*_arm_flags(challenge),
                                            "--neighbour-size", "mean"]))
    assert "invalid choice" in capsys.readouterr().err


def test_build_records_size(challenge):
    assert build.main(_argv(challenge, "szu", _arm_flags(challenge))) == 0
    assert build.main(_argv(challenge, "szm", [*_arm_flags(challenge),
                                               "--neighbour-size", "median"])) == 0
    unit_rec = json.loads((challenge["out"] / "szu.neighbour.json").read_text())
    med_rec = json.loads((challenge["out"] / "szm.neighbour.json").read_text())
    assert unit_rec["size"] == "unit" and med_rec["size"] == "median"
    assert np.isfinite(med_rec["scale"]) and med_rec["scale"] > 0
    assert med_rec["scale"] == unit_rec["scale"]          # one pool, one scale either way
    assert med_rec["targets_fused"] == 4 and "arm_size_mean" in med_rec
    assert "arm_size_mean" not in unit_rec
    # the exact scale rides along, so a replay can re-derive the blend from the record alone
    assert round(med_rec["scale_exact"], 6) == med_rec["scale"]
    for stem, want in (("szu", "unit"), ("szm", "median")):
        args = json.loads((challenge["out"] / f"{stem}.args.json").read_text())
        assert args["neighbour_size"] == want


def test_the_arm_record_carries_the_size_before_the_summary_merge(challenge):
    """Both consumers merge `arm.summary()` OVER this record, and the summary always carries
    `size`, so the record's own key can only be read here. The arm's value winning is the safe
    direction -- a record that disagrees with the arm that ran cannot be written."""
    from types import SimpleNamespace

    axis = np.array(GENES)
    gw = PseudobulkSums.load(challenge["data"] / "gwps.npz")

    def delta_of(lab):
        return build.pooled_delta(lab, [(gw, "control")], axis, shrinkage=False)

    def args_for(**extra):
        return SimpleNamespace(neighbour_table=challenge["data"] / "table.pt",
                               neighbour_pool=challenge["data"] / "pool.csv",
                               neighbour_k=3, neighbour_w=0.3, **extra)

    arm, rec = build.neighbour_arm_for(args_for(neighbour_size="median"), delta_of, axis)
    assert rec["size"] == "median" and arm.size == "median"
    assert "arm_size_mean" not in rec                     # pre-merge: the record's own keys only
    # and a caller from before the flag (eval.loco's older SimpleNamespace) still gets the gate's
    old_arm, old_rec = build.neighbour_arm_for(args_for(), delta_of, axis)
    assert old_rec["size"] == "unit" and old_arm.size == "unit"


def test_build_records_size_for_a_two_table_mix(challenge):
    """The mix path end to end under the flag: both tables' arms scale by n/s, and the sidecar
    says how hard each one pulled."""
    rng = np.random.default_rng(21)
    t2 = challenge["data"] / "table2.pt"
    torch.save({p: torch.tensor(rng.normal(size=4), dtype=torch.float32)
                for p in GENES[:N_PERTS]}, t2)
    flags = ["--neighbour-table", str(challenge["data"] / "table.pt"), "--neighbour-w", "0.2",
             "--neighbour-table", str(t2), "--neighbour-w", "0.1",
             "--neighbour-pool", str(challenge["data"] / "pool.csv"), "--neighbour-k", "3",
             "--neighbour-size", "median"]
    assert build.main(_argv(challenge, "mixmed", flags)) == 0
    rec = json.loads((challenge["out"] / "mixmed.neighbour.json").read_text())
    assert rec["size"] == "median" and rec["w"] == [0.2, 0.1] and rec["targets_fused"] == 4
    assert np.isfinite(rec["scale"]) and rec["scale"] > 0
    assert round(rec["scale_exact"], 6) == rec["scale"]
    for i in (0, 1):
        assert rec[f"table{i}_size_mean"] > 0 and 0.0 <= rec[f"table{i}_size_gt1_frac"] <= 1.0
    args = json.loads((challenge["out"] / "mixmed.args.json").read_text())
    assert args["neighbour_size"] == "median"


def test_loco_passes_size_through(monkeypatch, tmp_path, challenge):
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
                    "--source", "x.npz:ctl", *_arm_flags(challenge, w="0.15", k="2"),
                    "--neighbour-size", "median"])
    assert rc == 0
    assert captured["neighbour_size"] == "median"
    assert logged["neighbour_size"] == "median"


def test_loco_default_size_is_unit_and_reaches_the_arm(tmp_path, challenge):
    """The SimpleNamespace eval.loco hands `neighbour_arm_for` carries the flag, so a loco
    arm and a submission arm blend the same way."""
    from sidechain.eval.loco import build_transfer_prediction

    rng = np.random.default_rng(13)
    perts = GENES[:4]
    labels = ["non-targeting"] * 20 + [p for p in perts for _ in range(5)]
    X = sp.csr_matrix(rng.poisson(200, size=(len(labels), len(GENES))).astype(float))
    real = tmp_path / "real.h5ad"
    ad.AnnData(X=X, obs=pd.DataFrame({"target_gene": labels},
                                     index=[f"c{i}" for i in range(len(labels))]),
               var=pd.DataFrame(index=GENES)).write_h5ad(real)
    gw = PseudobulkSums.load(challenge["data"] / "gwps.npz")
    kw = dict(pert_col="target_gene", control="non-targeting", shrinkage=False, alpha=1.35,
              seed=0, min_libsize=100, neighbour_table=challenge["data"] / "table.pt",
              neighbour_pool=challenge["data"] / "pool.csv", neighbour_k=3, neighbour_w=0.3)
    info_u = build_transfer_prediction(real, [(gw, "control")], tmp_path / "u.h5ad", **kw)
    info_m = build_transfer_prediction(real, [(gw, "control")], tmp_path / "m.h5ad", **kw,
                                       neighbour_size="median")
    assert info_u["neighbour"]["size"] == "unit" and info_m["neighbour"]["size"] == "median"
    assert info_m["neighbour"]["scale"] == info_u["neighbour"]["scale"] > 0
    x_u = ad.read_h5ad(tmp_path / "u.h5ad").X
    x_m = ad.read_h5ad(tmp_path / "m.h5ad").X
    assert (x_u != x_m).nnz > 0                            # the blend moved the emitted cells
