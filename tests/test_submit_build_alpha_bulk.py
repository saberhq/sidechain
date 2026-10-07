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
    assert rec["emit_shape"] == {"shape": "controls", "contexts": {
        "X": {"control_cells_kept": 12, "blocks_in_the_controls_shape": 1, "blocks_left_on_the_template": []},
        "Y": {"control_cells_kept": 12, "blocks_in_the_controls_shape": 0, "blocks_left_on_the_template": ["TP53"]}}}
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
        assert ctx[c] == {"control_cells_kept": 12, "blocks_in_the_controls_shape": 1,
                          "blocks_left_on_the_template": []}
    shaped, drawn = (ad.read_h5ad(challenge["out"] / f"{s}.h5ad") for s in ("real", "tmpl"))
    assert shaped.n_obs == drawn.n_obs == 12
    assert not np.array_equal(shaped.X.toarray(), drawn.X.toarray())   # other cells than the same seed's template
    # a shaped block is control cells: its depths spread as theirs do, the template's sit near one depth
    depth = lambda a: np.asarray(a.X.sum(axis=1)).ravel()
    assert np.ptp(depth(shaped)[:6]) > 2 * np.ptp(depth(drawn)[:6])


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


def test_a_model_named_stem_is_refused_until_the_knob_has_a_letter(challenge, capsys):
    with pytest.raises(SystemExit):
        build.main(_argv(challenge, "ser-99aefkw_shapeprobe_v1", ["--alpha", "1.35", "--bulk-anchor", "pooled",
                                                                 "--emit-lambda", "0.5", "--emit-shape", "controls"]))
    assert "--emit-shape off its default has no registered knob letter" in capsys.readouterr().err


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
