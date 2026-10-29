"""`submit.build --alpha-bulk-target R` and `--alpha-bulk-floor`, and `sidechain.submit.knee`.

    .venv/bin/python -m pytest -q tests/test_alpha_bulk_target.py

Writes temporary builds only. Notes: private/code-notes/tests/test_alpha_bulk_target.py.md
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
    """cell-eval2's budget and per-knockdown jackknife, called directly."""
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
    """The reference script on a small built file of three contexts."""
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
    assert knee.next_alpha_bulk(2.0, 2.0, 0.8, 1.6) == 1.6               # to the floor
    assert knee.next_alpha_bulk(2.0, 1.6, 0.8, 1.0, margin=0.0) == pytest.approx(2.0 * math.sqrt(0.8 / 1.6))
    assert knee.next_alpha_bulk(2.0, 0.8, 0.8, 1.0, margin=0.0) == 2.0     # at the target, no margin: unchanged
    a = knee.next_alpha_bulk(3.0, 4.0, 0.8, 1.0, margin=0.0)
    assert a == pytest.approx(3.0 * math.sqrt(0.2)) and (a / 3.0) ** 2 * 4.0 == pytest.approx(0.8)
    # the default aims AIM_MARGIN under the target: on a ratio that grows as the square it lands there
    m = knee.AIM_MARGIN
    a = knee.next_alpha_bulk(2.0, 0.85, 0.8, 1.2)
    assert a == pytest.approx(2.0 * math.sqrt((1 - m) * 0.8 / 0.85)) and (a / 2.0) ** 2 * 0.85 < 0.8
    # on a ratio law a little under the square it still lands under the target
    assert all((knee.next_alpha_bulk(2.0, r0, 0.8, 1.0) / 2.0) ** 1.95 * r0 <= 0.8 for r0 in (0.85, 1.1, 1.6))
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            knee.next_alpha_bulk(2.0, bad, 0.8, 1.0)
    with pytest.raises(ValueError):
        knee.next_alpha_bulk(2.0, 1.2, 0.8, 1.0, margin=1.0)


def _run_rule(law, ceiling, target, floor, max_reads, rule):
    """The knee loop on a ratio law: `rule` 'new' is knee_next / knee_keep, 'old' the square-root step alone."""
    reads = [(ceiling, law(ceiling))]
    while True:
        a, r = reads[-1]
        if rule == "old":
            if r <= target or a <= floor or len(reads) >= max_reads or not math.isfinite(r):
                return reads, len(reads) - 1
            nxt = knee.next_alpha_bulk(a, r, target, floor)
        else:
            nxt = knee.knee_next(reads, target, floor, ceiling, max_reads)
            if nxt is None:
                return reads, knee.knee_keep(reads, target)
        reads.append((nxt, law(nxt)))


def test_the_step_back_up_after_an_undershoot():
    """A read far under the target steps back up inside the bracket, on the power law through the two reads."""
    target, floor, ceiling = 0.8, 1.2, 2.4
    reads = [(2.4, 2.0), (1.5026376808798587, 0.38840124464512915)]
    nxt = knee.knee_next(reads, target, floor, ceiling, 4)
    k = math.log(reads[0][1] / reads[1][1]) / math.log(reads[0][0] / reads[1][0])
    assert k == pytest.approx(3.5, abs=0.01)
    assert nxt == pytest.approx(reads[1][0] * ((1 - knee.AIM_MARGIN) * target / reads[1][1]) ** (1 / k))
    assert reads[1][0] < nxt < reads[0][0]
    # on that law the rule lands between (1 - UNDERSHOOT) * target and the target within the four reads
    law = lambda a: reads[1][1] * (a / reads[1][0]) ** k
    got, kept = _run_rule(law, ceiling, target, floor, 4, "new")
    assert (1 - knee.UNDERSHOOT) * target <= got[kept][1] <= target and len(got) == 3 and kept == 2


def test_the_rule_is_the_old_one_until_a_read_undershoots():
    """A read sequence that never undershoots below the ceiling is the old rule's, strength for strength."""
    for k in (1.95, 2.0):
        for c, ceiling, floor in ((0.2, 2.4, 1.2), (0.12, 2.4, 1.2), (0.35, 1.8, 1.0), (0.05, 2.4, 1.2)):
            law = lambda a, c=c, k=k: c * a ** k
            old, old_kept = _run_rule(law, ceiling, 0.8, floor, 4, "old")
            new, new_kept = _run_rule(law, ceiling, 0.8, floor, 4, "new")
            if all(r >= 0.72 or a >= ceiling or r > 0.8 for a, r in old):
                assert new == old and new_kept == old_kept
    # a read at the ceiling under the target stops there, however far under
    assert knee.knee_next([(2.4, 0.2)], 0.8, 1.2, 2.4, 4) is None
    # over the target with nothing under it: the square-root step, as before
    assert knee.knee_next([(2.4, 1.6)], 0.8, 1.2, 2.4, 4) == knee.next_alpha_bulk(2.4, 1.6, 0.8, 1.2)
    # over the target at the floor, at the read limit, or on a ratio that is not finite: stop
    assert knee.knee_next([(2.4, 3.0), (1.2, 1.2)], 0.8, 1.2, 2.4, 4) is None
    assert knee.knee_next([(2.4, 3.0), (2.0, 2.0), (1.8, 1.5), (1.6, 0.4)], 0.8, 1.2, 2.4, 4) is None
    assert knee.knee_next([(2.4, math.inf)], 0.8, 1.2, 2.4, 4) is None


def test_a_bracket_steps_inside_it_and_the_strongest_read_under_the_target_is_kept():
    # under, then over: the next read is inside the bracket, never on its ends
    reads = [(2.4, 2.0), (1.5, 0.4), (2.2, 1.4)]
    nxt = knee.knee_next(reads, 0.8, 1.2, 2.4, 5)
    assert 1.5 < nxt < 2.2
    # a bracket whose power law points past an end is held a twentieth of its width inside it
    assert knee.bracket_alpha_bulk((1.5, 0.79), (2.2, 50.0), 0.8) == pytest.approx(1.5 + 0.7 / 20)
    with pytest.raises(ValueError):
        knee.bracket_alpha_bulk((1.5, 0.85), (2.2, 1.4), 0.8)          # not straddling
    # the strongest read at or under the target is kept; with none under it, the last
    assert knee.knee_keep([(2.4, 2.0), (1.5, 0.4), (2.0, 0.75), (2.2, 1.1)], 0.8) == 2
    assert knee.knee_keep([(2.4, 2.0), (1.5, 0.4), (2.2, 1.4)], 0.8) == 1
    assert knee.knee_keep([(2.4, 2.0), (2.0, 1.4)], 0.8) == 1


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
            "--alpha", "1.0", "--emit-lambda", "0.4", "--bulk-anchor", "pooled", *extra]


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
    for extra in (["--alpha-bulk-target", "0.8"],                                 # no ceiling
                  ["--alpha-bulk", "3", "--alpha-bulk-target", "0"],
                  ["--alpha-bulk", "3", "--alpha-bulk-target", "1"],
                  ["--alpha-bulk", "3", "--alpha-bulk-target", "1.5"],
                  ["--alpha-bulk", "3", "--alpha-bulk-target", "-0.2"],
                  ["--alpha-bulk", "3", "--alpha-bulk-target", "nan"],
                  ["--alpha-bulk", "1.0", "--alpha-bulk-target", "0.8"],          # ceiling not above --alpha
                  ["--alpha-bulk", "0.8", "--alpha-bulk-target", "0.8"]):
        with pytest.raises(SystemExit):
            build.main(_argv(ch, "probe_refused", *extra))
        assert "--alpha-bulk-target" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        build.main(_argv(ch, "ser-99_fixture_v1", "--alpha-bulk", "3", "--alpha-bulk-target", "0.8"))
    assert "part of letter b's rule" in capsys.readouterr().err
    assert not (ch["out"] / "probe_refused.h5ad").exists()


def test_a_model_named_stem_with_b_carries_the_target(ch):
    """A model-named stem with b builds with the target, and writes what the same flags write under a freeform stem."""
    flags = ("--alpha-bulk", "1.8", "--alpha-bulk-target", "0.8")
    assert build.main(_argv(ch, "ser-99b_fixture_v1", *flags)) == 0
    assert build.main(_argv(ch, "freeform", *flags)) == 0
    a, b = ad.read_h5ad(ch["out"] / "ser-99b_fixture_v1.h5ad"), ad.read_h5ad(ch["out"] / "freeform.h5ad")
    assert (a.X != b.X).nnz == 0
    assert json.loads((ch["out"] / "ser-99b_fixture_v1.alpha_bulk.json").read_text())["alpha_bulk_target"] == 0.8


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
    ceiling, target = 1.8, 0.8
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
    assert build.main(_argv(ch, "rule_x", "--alpha-bulk", "1.8", "--alpha-bulk-target", "0.8", *ax)) == 0
    rec = json.loads((ch["out"] / "rule_x.alpha_bulk.json").read_text())["contexts"]["X"]
    assert rec["alpha_bulk"] < 1.8 and rec["written_cells_are_the_cells_read"]
    assert json.loads((ch["out"] / "rule_x.ctrl_axes.json").read_text())["contexts"]["X"]["deflation"]["targets"] > 0
    assert build.main(_argv(ch, "at_x", "--alpha-bulk", repr(rec["alpha_bulk"]), *ax)) == 0
    assert np.array_equal(_cells(ch["out"] / "rule_x.h5ad", "X"), _cells(ch["out"] / "at_x.h5ad", "X"))


# -- --alpha-bulk-floor ---------------------------------------------------------------------------------------

def test_without_the_floor_nothing_new_is_written_and_a_floor_at_alpha_is_the_same_rule(ch):
    assert build.main(_argv(ch, "low", "--alpha-bulk", "1.05", "--alpha-bulk-target", "0.01")) == 0
    rec = json.loads((ch["out"] / "low.alpha_bulk.json").read_text())
    assert "alpha_bulk_floor" not in rec and "never below alpha;" in rec["rule"]
    assert "alpha_bulk_floor" not in json.loads((ch["out"] / "low.args.json").read_text())
    assert build.main(_argv(ch, "low_f", "--alpha-bulk", "1.05", "--alpha-bulk-target", "0.01",
                            "--alpha-bulk-floor", "1.0")) == 0
    rec_f = json.loads((ch["out"] / "low_f.alpha_bulk.json").read_text())
    assert rec_f.pop("alpha_bulk_floor") == 1.0
    assert rec_f.pop("rule") == rec.pop("rule").replace("never below alpha;", "never below alpha_bulk_floor;")
    assert rec_f == rec
    a, b = ad.read_h5ad(ch["out"] / "low.h5ad"), ad.read_h5ad(ch["out"] / "low_f.h5ad")
    assert (a.X != b.X).nnz == 0


def test_a_floor_below_alpha_lets_the_knee_settle_below_the_per_cell_strength(ch):
    flags = ("--alpha", "1.55", "--alpha-bulk", "1.6", "--alpha-bulk-target", "0.8")
    assert build.main(_argv(ch, "cells_floor", *flags)) == 0                    # the default floor: --alpha
    x0 = json.loads((ch["out"] / "cells_floor.alpha_bulk.json").read_text())["contexts"]["X"]
    assert x0["alpha_bulk"] == 1.55 and x0["at_floor"] and not x0["met"]       # the fixture's premise
    assert build.main(_argv(ch, "low_floor", *flags, "--alpha-bulk-floor", "0.5")) == 0
    rec = json.loads((ch["out"] / "low_floor.alpha_bulk.json").read_text())
    assert rec["alpha_bulk_floor"] == 0.5 and rec["alpha"] == 1.55
    x = rec["contexts"]["X"]
    assert x["met"] and not x["at_floor"] and 0.5 < x["alpha_bulk"] < 1.55
    assert x["ratio_budget_over_claim"] <= 0.8 and all(t["dual_fallbacks"] == 0 for t in x["tries"])
    for a, b in zip(x["tries"], x["tries"][1:]):
        assert b["alpha_bulk"] == knee.next_alpha_bulk(a["alpha_bulk"], a["ratio_budget_over_claim"], 0.8, 0.5)
    assert x["written_cells_are_the_cells_read"]
    # the written cells are a plain build's with the summed strength under the per-cell one
    assert build.main(_argv(ch, "plain_low", "--alpha", "1.55", "--alpha-bulk", repr(x["alpha_bulk"]))) == 0
    assert np.array_equal(_cells(ch["out"] / "low_floor.h5ad", "X"), _cells(ch["out"] / "plain_low.h5ad", "X"))
    assert _ratios(ch["out"] / "low_floor.h5ad")["X"] == pytest.approx(x["ratio_budget_over_claim"], rel=1e-12)


def test_the_floor_flag_is_refused_where_it_cannot_act(ch, capsys):
    for extra in (["--alpha-bulk", "1.8", "--alpha-bulk-floor", "0.5"],                     # no target
                  ["--alpha-bulk", "1.8", "--alpha-bulk-target", "0.8", "--alpha-bulk-floor", "0"],
                  ["--alpha-bulk", "1.8", "--alpha-bulk-target", "0.8", "--alpha-bulk-floor", "nan"],
                  ["--alpha-bulk", "1.8", "--alpha-bulk-target", "0.8", "--alpha-bulk-floor", "1.8"]):
        with pytest.raises(SystemExit):
            build.main(_argv(ch, "probe_refused", *extra))
        assert "--alpha-bulk-floor" in capsys.readouterr().err
    assert not (ch["out"] / "probe_refused.h5ad").exists()
    # a model-named stem cannot carry the floor
    with pytest.raises(SystemExit):
        build.main(_argv(ch, "ser-99b_floor_v1", "--alpha-bulk", "1.8", "--alpha-bulk-target", "0.8",
                         "--alpha-bulk-floor", "0.5"))
    assert "--alpha-bulk-floor changes letter b's rule" in capsys.readouterr().err
    # with a floor the ceiling needs only to be above it, not above --alpha
    assert build.main(_argv(ch, "under", "--alpha", "1.9", "--alpha-bulk", "1.8", "--alpha-bulk-target", "0.8",
                            "--alpha-bulk-floor", "0.5")) == 0


def test_a_build_that_undershoots_keeps_the_strongest_read_under_the_target(ch, monkeypatch):
    """Steps forced to undershoot, then overshoot: the context is written at the read under the target."""
    monkeypatch.setattr(knee, "next_alpha_bulk", lambda a, r, t, floor, margin=0.0: floor)
    monkeypatch.setattr(knee, "bracket_alpha_bulk", lambda lo, hi, t, margin=0.0: hi[0] * 0.999)
    monkeypatch.setattr(build, "KNEE_MAX_READS", 3)
    assert build.main(_argv(ch, "rule", "--alpha-bulk", "1.8", "--alpha-bulk-target", "0.85")) == 0
    x = json.loads((ch["out"] / "rule.alpha_bulk.json").read_text())["contexts"]["X"]
    t = [(r["alpha_bulk"], r["ratio_budget_over_claim"]) for r in x["tries"]]
    assert t[1][0] == 1.0 and t[1][1] < (1 - knee.UNDERSHOOT) * 0.85 and t[2][1] > 0.85    # the premise
    assert t[2][0] == pytest.approx(1.8 * 0.999)
    assert x["kept_read"] == 2 and x["alpha_bulk"] == 1.0 and x["met"] and x["written_cells_are_the_cells_read"]
    assert build.main(_argv(ch, "at_x", "--alpha-bulk", "1.0")) == 0
    assert np.array_equal(_cells(ch["out"] / "rule.h5ad", "X"), _cells(ch["out"] / "at_x.h5ad", "X"))
