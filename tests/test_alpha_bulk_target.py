"""`submit.build --alpha-bulk-target R` (T84) and `sidechain.submit.knee`.

Pinned here: the knee module's ratio is knee_built.py's arithmetic on the same cells; the step lowers the
strength as sqrt(R / ratio) and never below the floor; the flag is refused without --alpha-bulk, outside
(0, 1), with --alpha-bulk not above --alpha and on a model-named stem; off, the build writes what it wrote
before (no record, no new args key, the knee never read); on, a context whose ratio at the ceiling is above R
is lowered until it is at most R while a context below R keeps the ceiling, and each context's cells are
those of a plain build at its chosen strength.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml

from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.submit import build, knee

KNEE_BUILT = Path.home() / "data/sidechain/runs/knobs/t84_stack_20261019/box/entry/knee_built.py"
G = 24
GENES = [f"G{i}" for i in range(G)]
PERTS = GENES[:6]


# -- the knee module ---------------------------------------------------------------------------------------

def _blocks(seed=0, targets=PERTS, n=12, depth=400.0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.5, 5.0, size=G)
    out = {}
    for k, t in enumerate(targets):
        shift = np.exp(rng.normal(0, 0.4 + 0.1 * k, size=G))
        rate = base * shift / (base * shift).sum() * depth
        out[t] = sp.csr_matrix(rng.poisson(rate * rng.uniform(0.6, 1.4, size=(n, 1))).astype(np.float32))
    return out


def _reference(blocks, genes):
    """knee_built.py's loop, copied: its budget_and_weights and its per-knockdown jackknife."""
    from cell_eval2.metrics import delta as ce
    from cell_eval2.moments import jackknife_correction
    from cell_eval2.prep import bulk_lognorm_means
    targets = list(blocks)
    sums = np.zeros((len(targets), len(genes))); jk = np.zeros(len(targets))
    for i, t in enumerate(targets):
        sub = sp.csr_matrix(blocks[t].astype(np.float64))
        sums[i] += np.asarray(sub.sum(axis=0)).ravel()
        jk[i] = float(jackknife_correction(sp.csr_matrix(sub), np.zeros(sub.shape[0], dtype=np.intp), 1, 50_000.0,
                                           chunk=64)[0])
    pm = bulk_lognorm_means(sums, 50_000.0)
    labels = [str(t) for t in targets]
    ex = ce._exclusion_cols(labels, np.asarray(genes), target_gene_map=None, gate_labels=labels, who="knee",
                            n_genes=len(genes))
    w = ce._row_weights(len(labels), len(genes), ex)
    budget = float(ce._across_pert_budget(pm, ex, w))
    claim = float((w * jk).sum())
    return budget, claim, budget / claim


def test_the_meter_is_knee_built_arithmetic():
    blocks = _blocks()
    meter = knee.KneeMeter(PERTS, GENES)
    for t in reversed(PERTS):                                  # order of arrival does not matter
        meter.add(t, blocks[t])
    got = meter.read()
    b, c, r = _reference(blocks, GENES)
    assert got["budget"] == b and got["claim_uncapped"] == c and got["ratio_budget_over_claim"] == r
    assert got["knockdowns"] == len(PERTS) and 0 < r < math.inf
    with pytest.raises(ValueError):
        meter.add(PERTS[0], blocks[PERTS[0]])                    # a knockdown twice
    short = knee.KneeMeter(PERTS, GENES)
    short.add(PERTS[0], blocks[PERTS[0]])
    with pytest.raises(ValueError):
        short.read()                                           # a knockdown missing


@pytest.mark.skipif(not KNEE_BUILT.exists(), reason="knee_built.py is a run record on Saber's Mac")
def test_the_meter_matches_knee_built_py_on_a_file(tmp_path):
    """knee_built.py itself, on a small built file of three contexts."""
    rows, obs = [], []
    reads = {}
    for ci, c in enumerate("ABC"):
        blocks = _blocks(seed=ci, depth=300.0 * (ci + 1))
        meter = knee.KneeMeter(PERTS, GENES)
        for t, b in blocks.items():
            meter.add(t, b)
            rows.append(b)
            obs += [(c, t)] * b.shape[0]
        reads[c] = meter.read()
    X = sp.vstack(rows).tocsr()
    a = ad.AnnData(X=X, obs=pd.DataFrame(obs, columns=["context", "target_gene"],
                                         index=[f"c{i}" for i in range(len(obs))]),
                   var=pd.DataFrame(index=GENES))
    a.write_h5ad(tmp_path / "built.h5ad")
    subprocess.run([sys.executable, str(KNEE_BUILT), str(tmp_path / "built.h5ad"), str(tmp_path),
                    str(tmp_path / "out.json")], check=True, capture_output=True)
    theirs = json.loads((tmp_path / "out.json").read_text())["contexts"]
    for c in "ABC":
        assert theirs[c]["budget"] == pytest.approx(reads[c]["budget"], rel=1e-12)
        assert theirs[c]["claim_uncapped"] == pytest.approx(reads[c]["claim_uncapped"], rel=1e-12)
        assert theirs[c]["ratio_budget_over_claim"] == pytest.approx(reads[c]["ratio_budget_over_claim"], rel=1e-12)


def test_the_step_is_sqrt_of_target_over_ratio_floored():
    assert knee.next_alpha_bulk(1.9, 1.88, 0.9, 1.35) == 1.35             # the K562 fold's read: to the floor
    assert knee.next_alpha_bulk(2.0, 1.6, 0.9, 1.0, margin=0.0) == pytest.approx(2.0 * math.sqrt(0.9 / 1.6))
    assert knee.next_alpha_bulk(2.0, 0.9, 0.9, 1.0, margin=0.0) == 2.0     # at the target, no margin: unchanged
    a = knee.next_alpha_bulk(3.0, 4.0, 0.8, 1.0, margin=0.0)
    assert a == pytest.approx(3.0 * math.sqrt(0.2)) and (a / 3.0) ** 2 * 4.0 == pytest.approx(0.8)
    # the default aims AIM_MARGIN under the target: on a ratio that grows as the square it lands there
    m = knee.AIM_MARGIN
    a = knee.next_alpha_bulk(2.0, 0.972, 0.9, 1.35)                          # context A's read at 2.00
    assert a == pytest.approx(2.0 * math.sqrt((1 - m) * 0.9 / 0.972)) and (a / 2.0) ** 2 * 0.972 < 0.9
    # with A/B/C's measured exponent (1.96) it still lands under 0.9 from twice the target
    assert all((knee.next_alpha_bulk(2.0, r0, 0.9, 1.0) / 2.0) ** 1.96 * r0 <= 0.9 for r0 in (0.95, 1.2, 1.8))
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            knee.next_alpha_bulk(2.0, bad, 0.9, 1.0)
    with pytest.raises(ValueError):
        knee.next_alpha_bulk(2.0, 1.2, 0.9, 1.0, margin=1.0)


# -- the build -----------------------------------------------------------------------------------------------

def _cache(path, ctrl_label, rng):
    ctrl = rng.uniform(5_000, 80_000, size=G)
    ctrl *= 1e6 / ctrl.sum()
    rows = [ctrl]
    for k in range(len(PERTS)):
        r = ctrl * np.exp2(rng.normal(0, 0.05 + 0.03 * k, size=G))
        rows.append(r * 1e6 / r.sum())
    m = np.asarray(rows)
    n = np.full(len(rows), 500, dtype=np.int64)
    PseudobulkSums(labels=[ctrl_label, *PERTS], genes=np.array(GENES, dtype=object),
                   count_sum=m * n[:, None], cpm_sum=m * n[:, None], cpm_sq_sum=(m**2 + m) * n[:, None],
                   n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["test"]).save(path)


def _controls(path, depth, seed):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.5, 5.0, size=G)
    lib = np.exp(rng.normal(0, 0.9, size=(200, 1))) * depth
    X = rng.poisson(base / base.sum() * lib).astype(np.float32)
    ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame(index=[f"c{i}" for i in range(len(X))]),
               var=pd.DataFrame(index=GENES)).write_h5ad(path)


@pytest.fixture
def ch(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    (data / "gene_names.csv").write_text("gene_name\n" + "\n".join(GENES) + "\n")
    (data / "pert_counts.csv").write_text("target_gene\n" + "\n".join(PERTS) + "\n")
    _controls(data / "ctx_x.h5ad", 60, 1)             # deeper cells: a smaller claim, a higher ratio
    _controls(data / "ctx_y.h5ad", 20, 2)             # shallower cells: a larger claim, a lower ratio
    rng = np.random.default_rng(7)
    _cache(data / "h1.npz", "non-targeting", rng)
    _cache(data / "gwps.npz", "control", rng)
    cfg = {"data_dir": str(data), "gene_names_file": "gene_names.csv", "n_genes": G,
           "pert_counts_file": "pert_counts.csv", "pert_col": "target_gene", "context_col": "context",
           "control_label": "non-targeting", "phase": "p1", "phases": {"p1": {"contexts": ["X", "Y"]}},
           "control_files": {"X": "ctx_x.h5ad", "Y": "ctx_y.h5ad"},
           "submission": {"cells_per_pert": 100, "max_counts_per_cell": 1_000_000, "max_cells": 100_000,
                          "max_stored_entries": 10_000_000}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    return {"cfg": tmp_path / "config.yaml", "data": data, "out": tmp_path / "out"}


def _argv(ch, stem, *extra):
    return ["--challenge-config", str(ch["cfg"]), "--emitter", "delta-transfer",
            "--h1-cache", str(ch["data"] / "h1.npz"), "--gwps-cache", str(ch["data"] / "gwps.npz"),
            "--out", str(ch["out"] / stem), "--no-pack", "--min-libsize", "10", "--no-shrink",
            "--alpha", "1.0", "--emit-lambda", "0.5", "--bulk-anchor", "pooled", *extra]


def _cells(path, ctx):
    a = ad.read_h5ad(path)
    return a.X[np.flatnonzero(a.obs["context"].astype(str).to_numpy() == ctx)].toarray()


def _ratios(path):
    """Each context's budget over claim, read off the built file."""
    a = ad.read_h5ad(path)
    out = {}
    for c in ("X", "Y"):
        sub = a[a.obs["context"].astype(str) == c]
        meter = knee.KneeMeter(PERTS, GENES)
        for t in PERTS:
            meter.add(t, sub.X[np.flatnonzero(sub.obs["target_gene"].astype(str).to_numpy() == t)])
        out[c] = meter.read()["ratio_budget_over_claim"]
    return out


def test_refusals(ch, capsys):
    for extra in (["--alpha-bulk-target", "0.9"],                                 # no ceiling
                  ["--alpha-bulk", "3", "--alpha-bulk-target", "0"],
                  ["--alpha-bulk", "3", "--alpha-bulk-target", "1"],
                  ["--alpha-bulk", "3", "--alpha-bulk-target", "1.5"],
                  ["--alpha-bulk", "3", "--alpha-bulk-target", "-0.2"],
                  ["--alpha-bulk", "3", "--alpha-bulk-target", "nan"],
                  ["--alpha-bulk", "1.0", "--alpha-bulk-target", "0.9"],          # ceiling not above --alpha
                  ["--alpha-bulk", "0.8", "--alpha-bulk-target", "0.9"]):
        with pytest.raises(SystemExit):
            build.main(_argv(ch, "probe_refused", *extra))
        assert "--alpha-bulk-target" in capsys.readouterr().err
    with pytest.raises(SystemExit):                                               # no registered letter yet
        build.main(_argv(ch, "ser-99benw_t84_v1", "--alpha-bulk", "3", "--alpha-bulk-target", "0.9"))
    assert "no registered knob letter" in capsys.readouterr().err
    assert not (ch["out"] / "probe_refused.h5ad").exists()


def test_off_writes_what_it_wrote_and_never_reads_the_knee(ch, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the knee was read without --alpha-bulk-target")
    monkeypatch.setattr(build, "KneeMeter", boom)
    assert build.main(_argv(ch, "plain", "--alpha-bulk", "3")) == 0
    args = json.loads((ch["out"] / "plain.args.json").read_text())
    assert "alpha_bulk_target" not in args and args["alpha_bulk"] == 3.0
    dual = json.loads((ch["out"] / "plain.dual.json").read_text())
    assert set(dual) == {"alpha", "alpha_bulk", "bulk_anchor", "dual_fallbacks"}
    assert not (ch["out"] / "plain.alpha_bulk.json").exists()


def test_a_context_above_the_target_is_lowered_and_one_below_keeps_the_ceiling(ch):
    ceiling, target = 1.5, 0.9
    assert build.main(_argv(ch, "plain", "--alpha-bulk", str(ceiling))) == 0
    r = _ratios(ch["out"] / "plain.h5ad")
    assert r["Y"] < target < r["X"]                                 # the fixture's premise

    assert build.main(_argv(ch, "rule", "--alpha-bulk", str(ceiling), "--alpha-bulk-target", str(target))) == 0
    rec = json.loads((ch["out"] / "rule.alpha_bulk.json").read_text())
    x, y = rec["contexts"]["X"], rec["contexts"]["Y"]
    assert rec["alpha_bulk_ceiling"] == ceiling and rec["alpha_bulk_target"] == target and rec["alpha"] == 1.0
    # Y: one read, at the ceiling, the plain build's ratio
    assert y["alpha_bulk"] == ceiling and y["reads"] == 1 and y["met"]
    assert y["tries"][0]["ratio_budget_over_claim"] == pytest.approx(r["Y"], rel=1e-12)
    # X: the ceiling read is the plain build's; each next strength is the rule's; the last is at most R
    assert x["tries"][0]["alpha_bulk"] == ceiling
    assert x["tries"][0]["ratio_budget_over_claim"] == pytest.approx(r["X"], rel=1e-12)
    assert 1 < x["reads"] <= build.KNEE_MAX_READS and x["met"]
    assert x["ratio_budget_over_claim"] <= target and x["alpha_bulk"] < ceiling
    for a, b in zip(x["tries"], x["tries"][1:]):
        assert b["alpha_bulk"] == knee.next_alpha_bulk(a["alpha_bulk"], a["ratio_budget_over_claim"], target, 1.0)
        assert a["ratio_budget_over_claim"] > target
    assert x["written_cells_are_the_cells_read"] and y["written_cells_are_the_cells_read"]
    dual = json.loads((ch["out"] / "rule.dual.json").read_text())
    assert dual["alpha_bulk"] == ceiling and dual["alpha_bulk_target"] == target
    assert dual["alpha_bulk_by_context"] == {"X": x["alpha_bulk"], "Y": ceiling}
    assert json.loads((ch["out"] / "rule.args.json").read_text())["alpha_bulk_target"] == target
    # the file's own ratios are the ones recorded
    got = _ratios(ch["out"] / "rule.h5ad")
    assert got["X"] == pytest.approx(x["ratio_budget_over_claim"], rel=1e-12) and got["X"] <= target
    # Y's cells are the plain build's at the ceiling; X's are a plain build's at X's chosen strength
    assert np.array_equal(_cells(ch["out"] / "rule.h5ad", "Y"), _cells(ch["out"] / "plain.h5ad", "Y"))
    assert build.main(_argv(ch, "at_x", "--alpha-bulk", repr(x["alpha_bulk"]))) == 0
    assert np.array_equal(_cells(ch["out"] / "rule.h5ad", "X"), _cells(ch["out"] / "at_x.h5ad", "X"))


def test_a_context_the_floor_cannot_bring_under_is_written_at_the_floor(ch):
    assert build.main(_argv(ch, "low", "--alpha-bulk", "1.05", "--alpha-bulk-target", "0.01")) == 0
    rec = json.loads((ch["out"] / "low.alpha_bulk.json").read_text())
    for c in ("X", "Y"):
        assert rec["contexts"][c]["alpha_bulk"] == 1.0 and rec["contexts"][c]["at_floor"]
        assert not rec["contexts"][c]["met"] and rec["contexts"][c]["reads"] == 2


def test_the_ceiling_map_at_a_strength_is_finalizes_arithmetic():
    gene_pos = {g: i for i, g in enumerate(GENES)}
    rng = np.random.default_rng(3)
    m = build.BulkAtCeiling(gene_pos)
    for p, f in (("G1", 1.0), ("G2", 0.7), ("notagene", 1.3)):
        vec = rng.normal(size=G)
        b = vec * (2.5 * f)
        if p in gene_pos:
            b[gene_pos[p]] = build.TARGET_SELF_LOG2FC
        m[p] = b
        m.keep(p, vec, f)
    assert all(np.array_equal(m.at(2.5)[p], m[p]) for p in m)                  # at the ceiling: itself
    low = m.at(1.7)
    assert np.array_equal(low["G2"][[0, 1, 3]], (m.base["G2"] * (1.7 * 0.7))[[0, 1, 3]])
    assert low["G2"][2] == build.TARGET_SELF_LOG2FC and low["G1"][1] == build.TARGET_SELF_LOG2FC
    assert np.array_equal(low["notagene"], m.base["notagene"] * (1.7 * 1.3))


def test_with_ctrl_axes_the_lowered_map_is_deflated_as_a_plain_build_at_that_strength(ch):
    ax = ["--ctrl-axes", "1", "--ctrl-axes-cache", str(ch["out"] / "axes")]
    assert build.main(_argv(ch, "rule_x", "--alpha-bulk", "1.5", "--alpha-bulk-target", "0.9", *ax)) == 0
    rec = json.loads((ch["out"] / "rule_x.alpha_bulk.json").read_text())["contexts"]["X"]
    assert rec["alpha_bulk"] < 1.5 and rec["written_cells_are_the_cells_read"]
    assert json.loads((ch["out"] / "rule_x.ctrl_axes.json").read_text())["contexts"]["X"]["deflation"]["targets"] > 0
    assert build.main(_argv(ch, "at_x", "--alpha-bulk", repr(rec["alpha_bulk"]), *ax)) == 0
    assert np.array_equal(_cells(ch["out"] / "rule_x.h5ad", "X"), _cells(ch["out"] / "at_x.h5ad", "X"))
