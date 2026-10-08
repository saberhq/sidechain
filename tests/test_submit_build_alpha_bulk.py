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


def test_pooled_anchor_rides_every_block_and_is_recorded(challenge, monkeypatch):
    """`--bulk-anchor pooled` (T84 round 2): the emitter carries the anchor; without --alpha-bulk
    the pseudobulk takes --alpha (a two-channel emission at one amplitude), with it the two
    amplitudes; the record names the anchor, and a mean_cpm build's record stays as written."""
    seen = []

    def spy(self, n, log2fc_cell, log2fc_bulk, **kw):
        seen.append((self.bulk_anchor, np.array(log2fc_cell), np.array(log2fc_bulk)))
        return self.emit(n, log2fc_cell)
    monkeypatch.setattr(PoissonEmitter, "emit_dual", spy)
    assert build.main(_argv(challenge, "anch", ["--alpha", "1.35", "--bulk-anchor", "pooled",
                                                "--emit-lambda", "0.5"])) == 0
    assert len(seen) == 2
    for anchor, cell, bulk in seen:
        assert anchor == "pooled" and np.abs(cell).max() > 0 and np.allclose(bulk, cell)
    rec = json.loads((challenge["out"] / "anch.dual.json").read_text())
    assert rec == {"alpha": 1.35, "alpha_bulk": None, "bulk_anchor": "pooled",
                   "dual_fallbacks": {"X": 0, "Y": 0}}
    assert json.loads((challenge["out"] / "anch.args.json").read_text())["bulk_anchor"] == "pooled"

    seen.clear()
    assert build.main(_argv(challenge, "anch2", ["--alpha", "1.35", "--alpha-bulk", "1.25",
                                                 "--bulk-anchor", "pooled", "--emit-lambda", "0.5"])) == 0
    for anchor, cell, bulk in seen:
        assert anchor == "pooled" and np.allclose(bulk, cell * (1.25 / 1.35))
    assert json.loads((challenge["out"] / "anch2.dual.json").read_text())["alpha_bulk"] == 1.25


def test_pooled_anchor_is_refused_where_it_cannot_ride(challenge):
    with pytest.raises(SystemExit):                                     # lambda 0: no depth spread
        build.main(_argv(challenge, "a_lam0", ["--bulk-anchor", "pooled", "--emit-lambda", "0"]))
    with pytest.raises(SystemExit):
        build.main(_argv(challenge, "a_even", ["--bulk-anchor", "pooled"]))
    argv = _argv(challenge, "a_null", ["--bulk-anchor", "pooled", "--emit-lambda", "0.5"])
    argv[argv.index("delta-transfer")] = "control-null"
    with pytest.raises(SystemExit):
        build.main(argv)


def test_the_fallback_rung_is_a_flag_and_the_record_names_who_fell_back(challenge, monkeypatch):
    """`--dual-fallback anchor` (T84, 2026-10-01): the emitter is told to keep the summed profile on
    the anchor when the two amplitudes cannot both be met; the record names the rung and the
    perturbations that fell back, per context. The default's record stays as it was written."""
    seen = []

    def spy(self, n, log2fc_cell, log2fc_bulk, **kw):
        seen.append(kw)
        block = self.emit(n, log2fc_cell)
        # pretend the first context's perturbation fell to the anchor rung
        self.last_dual, self.last_dual_reason = (("anchor", "fit") if self.p.name == "X" else ("dual", None))
        if self.p.name == "X":
            self.dual_fallbacks = getattr(self, "dual_fallbacks", 0) + 1
            self.dual_fallbacks_anchor = getattr(self, "dual_fallbacks_anchor", 0) + 1
        return block
    monkeypatch.setattr(PoissonEmitter, "emit_dual", spy)
    assert build.main(_argv(challenge, "rung", ["--alpha", "1.35", "--alpha-bulk", "1.0", "--bulk-anchor", "pooled",
                                                "--emit-lambda", "0.5", "--dual-fallback", "anchor"])) == 0
    assert seen == [{"on_fail": "anchor"}, {"on_fail": "anchor"}]
    rec = json.loads((challenge["out"] / "rung.dual.json").read_text())
    pert = json.loads((challenge["out"] / "rung.args.json").read_text())
    assert pert["dual_fallback"] == "anchor"
    assert rec["dual_fallback"] == "anchor" and rec["dual_fallbacks"] == {"X": 1, "Y": 0}
    (name, why), = rec["dual_fallback_targets"]["X"]["anchor"]
    assert why == "fit" and rec["dual_fallback_targets"]["X"]["template"] == [] and "Y" not in rec["dual_fallback_targets"]


def test_a_real_fallback_keeps_the_default_record_and_the_anchor_rung_names_it(challenge, capsys):
    """No spy: the real emitter. Under the default rung a build with a genuine fallback writes the
    same four-key record every shipped entry wrote; with `--dual-fallback anchor` the record names
    the rung and the perturbation, and the cells of that perturbation differ."""
    argv = ["--alpha", "1.35", "--alpha-bulk", "1.5", "--bulk-anchor", "pooled", "--emit-lambda", "0.5"]
    assert build.main(_argv(challenge, "old", argv)) == 0
    old = json.loads((challenge["out"] / "old.dual.json").read_text())
    assert set(old) == {"alpha", "alpha_bulk", "bulk_anchor", "dual_fallbacks"}
    n_fb = sum(old["dual_fallbacks"].values())
    assert n_fb >= 1, "the fixture is expected to produce a genuine fallback"
    assert "fell back:" in capsys.readouterr().out                 # the names go to the log
    assert build.main(_argv(challenge, "new", argv + ["--dual-fallback", "anchor"])) == 0
    new = json.loads((challenge["out"] / "new.dual.json").read_text())
    assert new["dual_fallback"] == "anchor" and new["dual_fallbacks"] == old["dual_fallbacks"]
    named = [t for rungs in new["dual_fallback_targets"].values() for rung in rungs.values() for t in rung]
    assert len(named) == n_fb
    a = ad.read_h5ad(challenge["out"] / "old.h5ad"); b = ad.read_h5ad(challenge["out"] / "new.h5ad")
    assert a.shape == b.shape and np.array_equal(np.asarray(a.X.sum(axis=1)), np.asarray(b.X.sum(axis=1)))
    kept = sum(len(r["anchor"]) for r in new["dual_fallback_targets"].values())
    assert ((a.X != b.X).nnz > 0) == (kept > 0)                     # only an anchor-rung rescue moves cells


def test_the_fallback_rung_is_refused_where_it_cannot_act(challenge):
    with pytest.raises(SystemExit):                                 # one channel: nothing to fall back from
        build.main(_argv(challenge, "inert", ["--alpha", "1.35", "--emit-lambda", "0.5",
                                              "--dual-fallback", "anchor"]))


# --emit-shape controls (T85): every block as real control cells re-rated to the prediction. What is
# pinned here is the builder's wiring; the emitter's numerics are test_dual_moment's.

def test_emit_shape_is_off_by_default_and_refused_on_one_channel(challenge, monkeypatch):
    seen = []
    real = build.ContextProfile.from_controls

    def spy_profile(path, name, **kw):
        seen.append(kw)
        return real(path, name, **kw)
    monkeypatch.setattr(build.ContextProfile, "from_controls", staticmethod(spy_profile))
    assert build.main(_argv(challenge, "off", ["--alpha", "1.35", "--bulk-anchor", "pooled",
                                               "--emit-lambda", "0.5"])) == 0
    assert all("keep_cells" not in kw for kw in seen)                  # the default call is the old call
    assert json.loads((challenge["out"] / "off.args.json").read_text())["emit_shape"] == "template"
    assert "emit_shape" not in json.loads((challenge["out"] / "off.dual.json").read_text())
    with pytest.raises(SystemExit):                                    # one channel: nothing to re-rate against
        build.main(_argv(challenge, "bad", ["--alpha", "1.35", "--emit-lambda", "0.5",
                                            "--emit-shape", "controls"]))


def test_emit_shape_controls_rides_every_block_and_is_recorded(challenge, monkeypatch):
    seen = []

    def spy(self, n, log2fc_cell, log2fc_bulk, **kw):
        seen.append((self.p.name, self.p.cells is not None, kw))
        block = self.emit(n, log2fc_cell)          # emit() clears last_shaped, so set it after
        self.last_shaped = self.p.name == "X"      # say context Y's block fell to the template rung
        return block
    monkeypatch.setattr(PoissonEmitter, "emit_dual", spy)
    assert build.main(_argv(challenge, "shp", ["--alpha", "1.35", "--bulk-anchor", "pooled",
                                               "--emit-lambda", "0.5", "--emit-shape", "controls"])) == 0
    assert seen == [("X", True, {"on_fail": "fallback", "shape": True}),
                    ("Y", True, {"on_fail": "fallback", "shape": True})]     # the cells are kept, the flag rides
    rec = json.loads((challenge["out"] / "shp.dual.json").read_text())
    none = {"cells_held_to_the_caps": 0, "genes_left_unmet_a_block": None}     # the spy re-rates nothing
    assert rec["emit_shape"] == {"shape": "controls", "contexts": {
        "X": {"control_cells_kept": 12, "blocks_in_the_controls_shape": 1, "blocks_left_on_the_template": [], **none},
        "Y": {"control_cells_kept": 12, "blocks_in_the_controls_shape": 0, "blocks_left_on_the_template": ["TP53"], **none}}}
    assert json.loads((challenge["out"] / "shp.args.json").read_text())["emit_shape"] == "controls"


def test_emit_shape_controls_runs_the_real_emitter_end_to_end(challenge, capsys):
    common = ["--alpha", "1.35", "--bulk-anchor", "pooled", "--emit-lambda", "0.5"]
    assert build.main(_argv(challenge, "tmpl", common)) == 0
    assert build.main(_argv(challenge, "real", [*common, "--emit-shape", "controls"])) == 0
    out = capsys.readouterr().out
    assert "X: emit-shape controls: 1 of 1 perturbations emitted as re-rated control cells (12 kept)" in out
    assert "WARNING" not in out
    ctx = json.loads((challenge["out"] / "real.dual.json").read_text())["emit_shape"]["contexts"]
    for c in ("X", "Y"):                           # the real emitter carries the fixture's block in the shape
        unmet = ctx[c].pop("genes_left_unmet_a_block")                 # and says what its solve left to the fit
        assert set(unmet) == {"median", "max"} and unmet["max"] >= 0
        assert ctx[c] == {"control_cells_kept": 12, "blocks_in_the_controls_shape": 1,
                          "blocks_left_on_the_template": [], "cells_held_to_the_caps": 0}
    shaped, drawn = (ad.read_h5ad(challenge["out"] / f"{s}.h5ad") for s in ("real", "tmpl"))
    assert shaped.n_obs == drawn.n_obs == 12
    assert not np.array_equal(shaped.X.toarray(), drawn.X.toarray())   # other cells than the same seed's template
    # a shaped block is control cells: its depths spread as theirs do, the template's sit near one depth
    depth = lambda a: np.asarray(a.X.sum(axis=1)).ravel()
    assert np.ptp(depth(shaped)[:6]) > 2 * np.ptp(depth(drawn)[:6])


def test_the_build_record_counts_the_cells_held_to_the_caps(challenge, monkeypatch, capsys):
    from sidechain.models import count_emitters

    monkeypatch.setattr(count_emitters, "SHAPE_DEPTH_GAIN_CAP", 1.0)       # a cap this fixture's cells do reach
    assert build.main(_argv(challenge, "held", ["--alpha", "1.35", "--bulk-anchor", "pooled", "--emit-lambda", "0.5",
                                                "--emit-shape", "controls"])) == 0
    ctx = json.loads((challenge["out"] / "held.dual.json").read_text())["emit_shape"]["contexts"]
    assert {c: ctx[c]["cells_held_to_the_caps"] for c in ctx} == {"X": 1, "Y": 3}, ctx
    assert "held to the caps" in capsys.readouterr().out


def test_a_block_the_fit_cannot_carry_is_named_and_warned_about(challenge, monkeypatch, capsys):
    from sidechain.models import count_emitters

    def fail(*a, **k):
        raise ValueError("moment fitting failed: forced by the test")
    monkeypatch.setattr(count_emitters, "dual_moment_counts", fail)
    assert build.main(_argv(challenge, "lost", ["--alpha", "1.35", "--bulk-anchor", "pooled",
                                                "--emit-lambda", "0.5", "--emit-shape", "controls"])) == 0
    ctx = json.loads((challenge["out"] / "lost.dual.json").read_text())["emit_shape"]["contexts"]
    for c in ("X", "Y"):                           # the real emitter's last rung, read off its own flag
        assert ctx[c]["blocks_in_the_controls_shape"] == 0 and ctx[c]["blocks_left_on_the_template"] == ["TP53"]
    assert "WARNING X: 1 block(s) fell to the drawn template" in capsys.readouterr().out


def test_a_model_named_build_in_the_controls_shape_needs_the_letter_r(challenge, capsys):
    """ADR 0005 (2026-10-07): `r` is what the emitted cells are built from, moved off the emitter's
    own drawn cells and named in the slug; no letter is the drawn template."""
    flags = ["--alpha", "1.35", "--bulk-anchor", "pooled", "--emit-lambda", "0.5"]
    shaped = flags + ["--emit-shape", "controls"]
    assert build.main(_argv(challenge, "ser-99aefkrw_ctrlshape_v1", shaped)) == 0
    rec = json.loads((challenge["out"] / "ser-99aefkrw_ctrlshape_v1.dual.json").read_text())
    assert rec["emit_shape"]["shape"] == "controls"
    capsys.readouterr()
    for stem, extra, why in (
            ("ser-99aefkw_ctrlshape_v1", shaped, "must carry the letter r"),      # shaped, no r
            ("ser-99aefkrw_template_v1", flags, "carries the letter r")):         # r, but the drawn template
        with pytest.raises(SystemExit):
            build.main(_argv(challenge, stem, extra))
        assert why in capsys.readouterr().err, stem
    # a freeform stem carries no claim, shaped or not
    assert build.main(_argv(challenge, "shapeprobe", shaped)) == 0


def test_a_target_no_source_covers_keeps_its_generic_shift_and_is_shaped_too(challenge, monkeypatch):
    """The builder gives a target no source covers the generic H1 shift (never no shift), so under
    --emit-shape controls it goes through the same re-rated path as every other block."""
    (challenge["data"] / "pert_counts.csv").write_text("target_gene\nTP53\nNOSOURCE1\n")
    seen = []

    def spy(self, n, log2fc_cell, log2fc_bulk, **kw):
        seen.append((self.p.name, log2fc_cell is None, log2fc_bulk is None, kw.get("shape")))
        return self.emit(n, log2fc_cell)
    monkeypatch.setattr(PoissonEmitter, "emit_dual", spy)
    assert build.main(_argv(challenge, "unc", ["--alpha", "1.35", "--bulk-anchor", "pooled",
                                               "--emit-lambda", "0.5", "--emit-shape", "controls"])) == 0
    assert seen == [("X", False, False, True)] * 2 + [("Y", False, False, True)] * 2
    rec = json.loads((challenge["out"] / "unc.dual.json").read_text())["emit_shape"]["contexts"]
    for c in ("X", "Y"):                           # the spy never shapes
        assert rec[c]["blocks_left_on_the_template"] == ["TP53", "NOSOURCE1"]


# --scatter-table (T85): a per-(context, perturbation, gene) dial on the emitted cell-to-cell scatter,
# the table sidechain.eval.loco reads with a context beside it. What is pinned here is the builder's
# wiring and its record; the dial's numerics are test_dual_moment's.

DUAL = ["--alpha", "1.35", "--bulk-anchor", "pooled", "--emit-lambda", "0.5"]


def _scatter_table(path, rows, columns=("context", "target", "feature", "scatter")):
    pd.DataFrame(rows, columns=list(columns)).to_parquet(path)
    return path


def _spy_on_emit_dual(monkeypatch):
    """Every emit_dual call as (context, keywords), the scatter vector as a list; the block is a plain
    emit, and the dial count is set as the emitter sets it."""
    seen = []

    def spy(self, n, log2fc_cell, log2fc_bulk, **kw):
        seen.append((self.p.name, {k: (v.tolist() if k == "scatter" else v) for k, v in kw.items()}))
        block = self.emit(n, log2fc_cell)          # emit() clears last_sharpened, so set it after
        self.last_sharpened = None if "scatter" not in kw else int((kw["scatter"] != 1.0).sum())
        return block
    monkeypatch.setattr(PoissonEmitter, "emit_dual", spy)
    return seen


def test_scatter_pairs_keeps_what_changes_something_and_counts_the_rest():
    tab = pd.DataFrame(
        [("P1", "g0", 0.0), ("P1", "g1", 2.5), ("P1", "g2", 1.0),          # 1 changes nothing: not a pair
         ("P2", "g0", 0.5), ("P2", "g1", 0.5), ("P2", "g2", 0.25),
         ("g0", "g1", 0.5), ("g0", "g2", 2.0), ("g0", "g3", 0.0), ("g0", "g0", 0.0),      # a target's own gene
         ("ZZ", "ZZ", 0.0), ("ZZ", "g1", 0.0),                              # off the panel (the first off the axis and its own gene too)
         ("P3", "P3", 0.0), ("P1", "QQ", 0.0)],                             # off the axis (the first its own gene too)
        columns=["target", "feature", "scatter"])
    out, rec = build.scatter_pairs(tab, np.array(["g0", "g1", "g2", "g3"]), ["P1", "P2", "g0", "P3"], "t.parquet")
    assert list(out) == ["P1", "P2", "g0"]
    assert out["P1"][0].tolist() == [0, 1] and out["P1"][1].tolist() == [0.0, 2.5]
    assert out["P2"][0].tolist() == [0, 1, 2] and out["P2"][1].tolist() == [0.5, 0.5, 0.25]
    assert out["g0"][0].tolist() == [1, 2, 3] and out["g0"][1].tolist() == [0.5, 2.0, 0.0]
    # a dropped row is counted once, by the first reason it meets; the median is over targets (2, 3, 3)
    assert rec == {"rows": 14, "pairs_applied": 8, "targets_with_a_pair": 3, "pairs_per_target_median": 3.0,
                   "rows_at_scatter_zero": 2, "rows_below_one": 6, "rows_above_one": 2,
                   "rows_dropped": {"target_not_in_this_file": 2, "gene_not_on_the_axis": 2, "the_targets_own_gene": 1}}
    for bad, why in ((pd.DataFrame({"target": ["P1", "P1"], "feature": ["g0", "g0"], "scatter": [0.0, 0.5]}), "listed twice"),
                     (pd.DataFrame({"target": ["P1"], "feature": ["g0"], "scatter": [-0.5]}), ">= 0"),
                     (pd.DataFrame({"target": ["P1"], "feature": ["g0"], "scatter": [np.nan]}), ">= 0")):
        with pytest.raises(SystemExit, match=why):
            build.scatter_pairs(bad, np.array(["g0", "g1", "g2", "g3"]), ["P1"], "t.parquet")


def test_scatter_table_is_off_by_default_and_refused_where_it_cannot_act(challenge, tmp_path, capsys):
    assert build.main(_argv(challenge, "off", DUAL)) == 0
    assert json.loads((challenge["out"] / "off.args.json").read_text())["scatter_table"] is None
    assert "scatter_table" not in json.loads((challenge["out"] / "off.dual.json").read_text())
    table = _scatter_table(tmp_path / "t.parquet", [("X", "TP53", "B", 0.0)])
    twice = _scatter_table(tmp_path / "twice.parquet", [("X", "TP53", "B", 0.0), ("X", "TP53", "B", 0.5)])
    bare = _scatter_table(tmp_path / "bare.parquet", [("TP53", "B")], columns=("target", "feature"))
    wild = _scatter_table(tmp_path / "wild.parquet", [("X", "TP53", "B", 0.0), ("Q", "TP53", "B", -1.0)])   # in a context not built
    hole = _scatter_table(tmp_path / "hole.parquet", [("X", "TP53", "B", 0.0), ("Q", "TP53", "B", np.nan)])
    gone = _scatter_table(tmp_path / "gone.parquet", [("X", "NOPE", "B", 0.0)])                             # another panel's target
    named = _scatter_table(tmp_path / "named.parquet", [("X", "TP53", "B", 0.0)], columns=("Context", "target", "feature", "scatter"))
    lost = _scatter_table(tmp_path / "lost.parquet", [("x", "TP53", "B", 0.0), ("context_X", "TP53", "C", 0.0)])
    idle = _scatter_table(tmp_path / "idle.parquet", [("TP53", "B", 1.0)], columns=("target", "feature", "scatter"))
    again = _scatter_table(tmp_path / "again.parquet", [("TP53", "B", 0.0), ("TP53", "B", 0.5)],
                           columns=("target", "feature", "scatter"))
    capsys.readouterr()
    for stem, flags, tab, why in (
            ("one", ["--alpha", "1.35", "--emit-lambda", "0.5"], table, "two-channel"),     # one channel: no fit to act in
            ("ser-99aefkw_head_v1", DUAL, table, "no registered knob letter"),              # a model name claims its letters
            ("ser-99aefkrw_ctrlshape_v1", [*DUAL, "--emit-shape", "controls"], table, "no registered knob letter"),
            ("twice", DUAL, twice, "twice.parquet (context X): a (target, feature) pair is listed twice"),
            ("bare", DUAL, bare, "no column"),
            ("wild", DUAL, wild, ">= 0"),
            ("hole", DUAL, hole, ">= 0"),
            ("gone", DUAL, gone, "none of its pairs applies"),
            ("named", DUAL, named, "spelled 'context'"),      # read as no context it would act on every context
            ("lost", DUAL, lost, "none of its pairs applies"),          # its contexts are not this build's
            ("idle", DUAL, idle, "none of its pairs applies"),          # every value is 1
            ("again", DUAL, again, "again.parquet: a (target, feature) pair is listed twice")):
        with pytest.raises(SystemExit) as err:
            build.main(_argv(challenge, stem, [*flags, "--scatter-table", str(tab)]))
        io = capsys.readouterr()
        assert why in io.err + str(err.value), stem
        assert "WARNING" not in io.out, stem               # a refused table warns of nothing
        # refused before any work: no record, no cells
        assert not (challenge["out"] / f"{stem}.args.json").exists() and not (challenge["out"] / f"{stem}.h5ad").exists(), stem
    # the same pair in two contexts is two rows, not one listed twice
    both = _scatter_table(tmp_path / "both.parquet", [("X", "TP53", "B", 0.0), ("Y", "TP53", "B", 0.5)])
    assert build.main(_argv(challenge, "both", [*DUAL, "--scatter-table", str(both)])) == 0
    # a context kept as the table's index is its context column all the same
    pd.read_parquet(both).set_index("context").to_parquet(tmp_path / "indexed.parquet")
    assert build.main(_argv(challenge, "indexed", [*DUAL, "--scatter-table", str(tmp_path / "indexed.parquet")])) == 0
    rec = json.loads((challenge["out"] / "indexed.dual.json").read_text())["scatter_table"]
    assert rec["context_column"] is True and [rec["contexts"][c]["pairs_applied"] for c in ("X", "Y")] == [1, 1]


def test_scatter_table_rides_the_listed_blocks_and_is_recorded(challenge, monkeypatch, tmp_path):
    import hashlib

    seen = _spy_on_emit_dual(monkeypatch)
    table = _scatter_table(tmp_path / "t.parquet", [
        ("X", "TP53", "B", 0.0), ("X", "TP53", "C", 1.0),            # 1 changes nothing: not a pair
        ("Y", "TP53", "A", 0.25),
        ("Z", "TP53", "A", 0.0),                                     # a context this build does not write
        ("X", "NOPE", "A", 0.0), ("X", "TP53", "Q", 0.0)])           # no such perturbation; no such gene
    assert build.main(_argv(challenge, "dial", [*DUAL, "--scatter-table", str(table)])) == 0
    assert seen == [("X", {"on_fail": "fallback", "scatter": [1.0, 0.0, 1.0]}),
                    ("Y", {"on_fail": "fallback", "scatter": [0.25, 1.0, 1.0]})]
    rec = json.loads((challenge["out"] / "dial.dual.json").read_text())["scatter_table"]
    assert rec["table"] == str(table) and rec["sha256"] == hashlib.sha256(table.read_bytes()).hexdigest()
    assert (rec["rows"], rec["context_column"], rec["rows_naming_a_context_not_built"]) == (6, True, 1)
    carried = {"targets_carrying_it": 1, "pairs_carried": 1, "targets_listed_but_not_carrying": []}
    assert rec["contexts"]["X"] == {
        "rows": 4, "pairs_applied": 1, "targets_with_a_pair": 1, "pairs_per_target_median": 1.0,
        "rows_at_scatter_zero": 1, "rows_below_one": 1, "rows_above_one": 0,
        "rows_dropped": {"target_not_in_this_file": 1, "gene_not_on_the_axis": 1, "the_targets_own_gene": 0}, **carried}
    assert rec["contexts"]["Y"] == {
        "rows": 1, "pairs_applied": 1, "targets_with_a_pair": 1, "pairs_per_target_median": 1.0,
        "rows_at_scatter_zero": 0, "rows_below_one": 1, "rows_above_one": 0,
        "rows_dropped": {"target_not_in_this_file": 0, "gene_not_on_the_axis": 0, "the_targets_own_gene": 0}, **carried}
    assert json.loads((challenge["out"] / "dial.args.json").read_text())["scatter_table"] == str(table)
    # the other knobs of the two-channel call ride beside it: the controls' shape, the anchor rung,
    # and shifts pooled per context (--gamma)
    for stem, extra, kw in (("shaped", ["--emit-shape", "controls"], {"on_fail": "fallback", "shape": True}),
                            ("rung", ["--dual-fallback", "anchor"], {"on_fail": "anchor"}),
                            ("gamma", ["--gamma", "0.5"], {"on_fail": "fallback"})):
        seen.clear()
        assert build.main(_argv(challenge, stem, [*DUAL, *extra, "--scatter-table", str(table)])) == 0
        assert seen == [("X", {**kw, "scatter": [1.0, 0.0, 1.0]}), ("Y", {**kw, "scatter": [0.25, 1.0, 1.0]})], stem


def test_each_target_takes_its_own_pairs_and_an_unlisted_one_none(challenge, monkeypatch, tmp_path, capsys):
    """Three perturbations (two have no source and carry the generic shift): each listed block gets
    its own genes at its own values in its own context, and an unlisted block's call is the call
    without a table."""
    (challenge["data"] / "pert_counts.csv").write_text("target_gene\nTP53\nNOSRC1\nNOSRC2\n")
    seen = _spy_on_emit_dual(monkeypatch)
    table = _scatter_table(tmp_path / "t.parquet", [
        (1, "TP53", "B", 0.0), (1, "TP53", "C", 0.5), (1, "NOSRC1", "A", 2.0), (1, "NOSRC1", "C", 0.25),
        (2, "NOSRC1", "B", 0.75)])
    cfg = yaml.safe_load(challenge["cfg"].read_text())               # contexts named by integers, as a table may write them
    cfg["phases"]["p1"]["contexts"] = ["1", "2"]
    cfg["control_files"] = {"1": "ctx_x.h5ad", "2": "ctx_y.h5ad"}
    challenge["cfg"].write_text(yaml.safe_dump(cfg))
    assert build.main(_argv(challenge, "three", [*DUAL, "--scatter-table", str(table)])) == 0
    plain = {"on_fail": "fallback"}
    assert seen == [("1", {**plain, "scatter": [1.0, 0.0, 0.5]}), ("1", {**plain, "scatter": [2.0, 1.0, 0.25]}), ("1", plain),
                    ("2", plain), ("2", {**plain, "scatter": [1.0, 0.75, 1.0]}), ("2", plain)]
    rec = json.loads((challenge["out"] / "three.dual.json").read_text())["scatter_table"]["contexts"]
    assert (rec["1"]["targets_carrying_it"], rec["1"]["pairs_carried"], rec["1"]["pairs_per_target_median"]) == (2, 4, 2.0)
    assert (rec["2"]["targets_carrying_it"], rec["2"]["pairs_carried"]) == (1, 1)
    assert "1: scatter table: 2 of 3 perturbations carry it (4 of the 4 pairs that apply to this context)" in capsys.readouterr().out
    # --limit-perts cuts the panel before the table is read: the second target's rows are another panel's
    seen.clear()
    assert build.main(_argv(challenge, "cut", [*DUAL, "--limit-perts", "1", "--scatter-table", str(table)])) == 0
    assert seen == [("1", {**plain, "scatter": [1.0, 0.0, 0.5]}), ("2", plain)]
    cut = json.loads((challenge["out"] / "cut.dual.json").read_text())["scatter_table"]["contexts"]
    assert cut["1"]["rows_dropped"]["target_not_in_this_file"] == 2 and cut["2"]["targets_with_a_pair"] == 0


def test_a_scatter_table_without_a_context_acts_on_every_context(challenge, monkeypatch, tmp_path, capsys):
    seen = []

    def spy(self, n, log2fc_cell, log2fc_bulk, **kw):
        seen.append((self.p.name, kw["scatter"].tolist()))
        return self.emit(n, log2fc_cell)
    monkeypatch.setattr(PoissonEmitter, "emit_dual", spy)
    table = _scatter_table(tmp_path / "t.parquet", [("TP53", "C", 0.5)], columns=("target", "feature", "scatter"))
    assert build.main(_argv(challenge, "all", [*DUAL, "--scatter-table", str(table)])) == 0
    assert seen == [("X", [1.0, 1.0, 0.5]), ("Y", [1.0, 1.0, 0.5])]
    rec = json.loads((challenge["out"] / "all.dual.json").read_text())["scatter_table"]
    assert rec["context_column"] is False and rec["rows_naming_a_context_not_built"] == 0
    # the spy's emit() leaves no dial count, so the record says the block did not carry it
    assert rec["contexts"]["X"]["targets_listed_but_not_carrying"] == ["TP53"]
    assert "WARNING" not in capsys.readouterr().out
    # a context none of the table's pairs applies to (it has no row for it, or its rows all drop) is emitted
    # without the table, and said so; the first context is no different from the last
    def spy2(self, n, log2fc_cell, log2fc_bulk, **kw):
        seen.append((self.p.name, "scatter" in kw))
        return self.emit(n, log2fc_cell)
    monkeypatch.setattr(PoissonEmitter, "emit_dual", spy2)
    for stem, rows, dialled, silent in (
            ("part", [("X", "TP53", "C", 0.5)], "X", "Y"),
            ("last", [("Y", "TP53", "C", 0.5)], "Y", "X"),
            ("drop", [("X", "TP53", "C", 0.5), ("Y", "NOPE", "C", 0.0), ("Y", "TP53", "Q", 0.0)], "X", "Y")):
        seen.clear()
        tab = _scatter_table(tmp_path / f"{stem}.parquet", rows)
        assert build.main(_argv(challenge, stem, [*DUAL, "--scatter-table", str(tab)])) == 0
        assert seen == [("X", dialled == "X"), ("Y", dialled == "Y")], stem
        out = capsys.readouterr().out
        assert f"WARNING --scatter-table {stem}.parquet: no pair of it applies to context {silent}" in out, stem
        assert f"applies to context {dialled}" not in out, stem


def test_scatter_table_runs_the_real_emitter_and_touches_only_the_listed_context(challenge, tmp_path, capsys):
    table = _scatter_table(tmp_path / "t.parquet", [("X", "TP53", "B", 0.0), ("X", "TP53", "C", 0.0)])
    assert build.main(_argv(challenge, "plain", DUAL)) == 0
    assert build.main(_argv(challenge, "dial", [*DUAL, "--scatter-table", str(table)])) == 0
    assert "X: scatter table: 1 of 1 perturbations carry it (2 of the 2 pairs that apply to this context)" in capsys.readouterr().out
    rec = json.loads((challenge["out"] / "dial.dual.json").read_text())["scatter_table"]["contexts"]
    assert rec["X"]["targets_carrying_it"] == 1 and rec["Y"]["targets_carrying_it"] == 0
    a, b = (ad.read_h5ad(challenge["out"] / f"{s}.h5ad") for s in ("plain", "dial"))
    A, B = a.X.toarray().astype(np.float64), b.X.toarray().astype(np.float64)
    x, y = (a.obs["context"] == "X").to_numpy(), (a.obs["context"] == "Y").to_numpy()
    assert np.array_equal(A[y], B[y])                                  # the unlisted context: the same cells, bit for bit
    assert np.array_equal(A[x].sum(axis=1), B[x].sum(axis=1))          # the listed one keeps every cell's depth
    spread = lambda M: (M[:, 1] / M.sum(axis=1)).std()                 # gene B's share of a cell, over the block
    assert spread(B[x]) < spread(A[x])                                 # and its dialled gene is narrower
    # the controls' shape carries the same table
    assert build.main(_argv(challenge, "shaped", [*DUAL, "--emit-shape", "controls", "--scatter-table", str(table)])) == 0
    assert json.loads((challenge["out"] / "shaped.dual.json").read_text())["scatter_table"]["contexts"]["X"]["targets_carrying_it"] == 1


def test_a_dialled_block_on_a_fallback_rung_is_recorded_as_it_ended(challenge, monkeypatch, tmp_path, capsys):
    """The template rung carries none of the table and is named; the anchor rung is fitted on the
    dialled cells and counts as carrying it."""
    from sidechain.models import count_emitters

    table = _scatter_table(tmp_path / "t.parquet", [("X", "TP53", "B", 0.0), ("Y", "TP53", "B", 0.0)])
    # the real emitter, two amplitudes this fixture cannot always meet: who lands on which rung is its own record
    two = ["--alpha", "1.35", "--alpha-bulk", "1.5", "--bulk-anchor", "pooled", "--emit-lambda", "0.5",
           "--dual-fallback", "anchor", "--scatter-table", str(table)]
    assert build.main(_argv(challenge, "rungs", two)) == 0
    rec = json.loads((challenge["out"] / "rungs.dual.json").read_text())
    fell = rec.get("dual_fallback_targets", {})
    on_anchor = {c for c, r in fell.items() if any(t == "TP53" for t, _ in r["anchor"])}
    on_template = {c for c, r in fell.items() if any(t == "TP53" for t, _ in r["template"])}
    assert on_anchor, "the fixture is expected to put a block on the anchor rung"
    for c in ("X", "Y"):
        st = rec["scatter_table"]["contexts"][c]
        assert st["targets_carrying_it"] == (0 if c in on_template else 1), (c, fell)
        assert st["targets_listed_but_not_carrying"] == (["TP53"] if c in on_template else [])

    def fail(*a, **k):
        raise ValueError("moment fitting failed: forced by the test")
    monkeypatch.setattr(count_emitters, "dual_moment_counts", fail)
    more = _scatter_table(tmp_path / "more.parquet", [("X", "TP53", "B", 0.0), ("X", "NOPE", "A", 0.0), ("X", "TP53", "Q", 0.0),
                                                      ("Y", "TP53", "B", 0.0), ("Y", "TP53", "C", 0.5)])
    capsys.readouterr()
    assert build.main(_argv(challenge, "lost", [*DUAL, "--scatter-table", str(more)])) == 0
    lost = json.loads((challenge["out"] / "lost.dual.json").read_text())["scatter_table"]["contexts"]
    for c in ("X", "Y"):
        assert lost[c]["targets_carrying_it"] == 0 and lost[c]["pairs_carried"] == 0
        assert lost[c]["targets_listed_but_not_carrying"] == ["TP53"]
    out = capsys.readouterr().out                 # the line counts what was carried, of the pairs that apply (rows dropped apart)
    assert "X: scatter table: 0 of 1 perturbations carry it (0 of the 1 pairs that apply to this context)" in out
    assert "Y: scatter table: 0 of 1 perturbations carry it (0 of the 2 pairs that apply to this context)" in out
