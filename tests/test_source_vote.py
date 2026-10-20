"""`--source-weight`, `--source-weight-table` and `--source-scale` (`sidechain.submit.source_vote`).

Pinned here: the parsers' refusals; in `pooled_delta`, a vote of 1 everywhere and a strength of 1 are
bit-identical to the plain pool, a uniform vote changes nothing (the general pooled variance reduces to
`1 / sum(w)`), a vote of 0 is the source's absence, and a strength scales one source's contribution while its
weight stays put; on `submit.build`, a vote of 0 in a context writes the file `--source-context` writes, a
per-knockdown table row acts on its knockdown, contexts that vote alike share one pool, and the records are
written only when the flags are on; on `eval.loco`, a vote reaches the targets' pool and a delta cache is
refused.
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
from sidechain.submit.source_vote import (SourceVote, parse_source_scales, parse_source_weights,
                                          read_source_weight_table, source_votes, vote_groups)

GENES = ["TP53", "B", "C", "D", "E"]
TARGETS = ["TP53", "B"]


def _cache(path, ctrl_label, ctrl, perts, n_cells=200):
    m = np.asarray([ctrl, *perts], dtype=float)
    n = np.full(len(m), n_cells, dtype=np.int64)
    PseudobulkSums(labels=[ctrl_label, *TARGETS], genes=np.array(GENES, dtype=object),
                   count_sum=m * n[:, None], cpm_sum=m * n[:, None], cpm_sq_sum=(m**2 * 1.3 + m) * n[:, None],
                   n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["test"]).save(path)


def _sources(tmp_path):
    rng = np.random.default_rng(5)
    out = []
    for name in ("h1", "gwps", "extra"):
        label = "non-targeting" if name == "h1" else "control"     # the build reads H1 at the challenge's label
        ctrl = rng.uniform(1e5, 3e5, size=len(GENES))
        perts = [ctrl * np.exp2(rng.normal(0, 0.6, size=len(GENES))) for _ in TARGETS]
        _cache(tmp_path / f"{name}.npz", label, ctrl, perts)
        pb = PseudobulkSums.load(tmp_path / f"{name}.npz")
        pb.sidechain_name = name
        out.append((pb, label))
    return out


# ------------------------------------------------------------------------------------------- parsers

def test_parsers_and_refusals(tmp_path):
    assert parse_source_weights(["cd4=A:1.25", "cd4=B:0"]) == {("cd4", "A"): 1.25, ("cd4", "B"): 0.0}
    for bad, why in ((["cd4=A"], "NAME=CTX:W"), (["cd4=A:-1"], ">= 0"), (["cd4=A:x"], "not a number"),
                     (["cd4=A:1", "cd4=A:2"], "given twice")):
        with pytest.raises(SystemExit, match=why):
            parse_source_weights(bad)
    names = ["h1", "gwps", "extra"]
    assert parse_source_scales(["gwps=0.8"], names) == {"gwps": 0.8}
    for bad, why in ((["gwps=0"], "> 0"), (["nope=2"], "matches 0"), (["gwps=1", "gwps=2"], "given twice")):
        with pytest.raises(SystemExit, match=why):
            parse_source_scales(bad, names)
    with pytest.raises(SystemExit, match="match no source"):
        source_votes({("nope", "A"): 1.0}, None, ["A"], names)
    with pytest.raises(SystemExit, match="no flag or table row names a context"):
        source_votes({("gwps", "Z"): 1.0}, None, ["A", "B"], names)
    p = tmp_path / "t.parquet"
    pd.DataFrame({"source": ["gwps", "gwps"], "context": ["A", "A"], "target": [None, "TP53"],
                  "weight": [1.0, 2.0]}).to_parquet(p)
    with pytest.raises(SystemExit, match="one or the other"):
        read_source_weight_table(p)
    pd.DataFrame({"source": ["gwps", "gwps"], "context": ["A", "A"], "target": ["B", "B"],
                  "weight": [1.0, 2.0]}).to_parquet(p)
    with pytest.raises(SystemExit, match="more than once"):
        read_source_weight_table(p)


def test_the_flag_and_the_table_multiply_and_group(tmp_path):
    p = tmp_path / "t.parquet"
    pd.DataFrame({"source": ["gwps", "extra"], "context": ["A", "A"], "target": ["TP53", None],
                  "weight": [0.5, 3.0]}).to_parquet(p)
    votes, rec = source_votes({("gwps", "A"): 2.0, ("gwps", "k562"): 1.5}, read_source_weight_table(p),
                              ["A", "B", "C"], ["h1", "gwps", "extra"])
    a = votes["A"]
    assert a.m("gwps", "TP53") == 1.0 and a.m("gwps", "B") == 2.0 and a.m("extra", "B") == 3.0
    assert a.m("h1", "TP53") == 1.0 and votes["B"].is_identity
    assert vote_groups(["A", "B", "C"], votes) == [("A",), ("B", "C")]
    assert rec["contexts_not_in_build"] == ["k562"] and len(rec["table"]["sha256"]) == 64


# -------------------------------------------------------------------------------------- pooled_delta

@pytest.mark.parametrize("kw", [dict(shrinkage=False),
                                dict(shrinkage=True, shrink_stage="pooled", shrink_rule="adaptive"),
                                dict(shrinkage=True, shrink_stage="pooled", shrink_rule="garrote"),
                                dict(shrinkage=True)])
def test_neutral_and_uniform_votes(tmp_path, kw):
    srcs = _sources(tmp_path)
    axis = np.array(GENES, dtype=object)
    plain = build.pooled_delta("TP53", srcs, axis, var_floor="poisson", **kw)
    ones = SourceVote("A", {"gwps": 1.0}, {}, {})
    same = build.pooled_delta("TP53", srcs, axis, var_floor="poisson", source_vote=ones,
                              source_scale={"h1": 1.0}, **kw)
    assert np.array_equal(plain, same)                                     # neutral: bit for bit
    uniform = SourceVote("A", {"h1": 2.5, "gwps": 2.5, "extra": 2.5}, {}, {})
    got = build.pooled_delta("TP53", srcs, axis, var_floor="poisson", source_vote=uniform, **kw)
    assert np.allclose(got, plain, rtol=1e-10, atol=1e-12)                # the general variance is 1/sum(w)
    no_extra = build.pooled_delta("TP53", srcs[:2], axis, var_floor="poisson", **kw)
    zero = build.pooled_delta("TP53", srcs, axis, var_floor="poisson",
                              source_vote=SourceVote("A", {"extra": 0.0}, {}, {}), **kw)
    assert np.array_equal(zero, no_extra)                                  # a zero vote is the absence
    half = build.pooled_delta("TP53", srcs, axis, var_floor="poisson",
                              source_vote=SourceVote("A", {"extra": 0.5}, {}, {}), **kw)
    assert not np.allclose(half, plain)


def test_a_vote_is_a_weight_and_a_strength_is_not(tmp_path):
    srcs = _sources(tmp_path)
    axis = np.array(GENES, dtype=object)
    fx, wx = [], []
    for pb, ctl in srcs:
        fc, var = build.as_delta_source((pb, ctl), var_floor="poisson").effect("TP53")
        fx.append(fc); wx.append(1.0 / np.maximum(var, 1e-12))
    m = [1.0, 2.0, 0.5]
    vote = SourceVote("A", {"h1": 1.0, "gwps": 2.0, "extra": 0.5}, {}, {})
    got = build.pooled_delta("TP53", srcs, axis, var_floor="poisson", shrinkage=False, source_vote=vote)
    want = sum(mi * w * f for mi, w, f in zip(m, wx, fx)) / sum(mi * w for mi, w in zip(m, wx))
    assert np.allclose(got, want, rtol=1e-12)
    got = build.pooled_delta("TP53", srcs, axis, var_floor="poisson", shrinkage=False,
                             source_scale={"extra": 3.0})
    want = (wx[0] * fx[0] + wx[1] * fx[1] + wx[2] * 3.0 * fx[2]) / (wx[0] + wx[1] + wx[2])
    assert np.allclose(got, want, rtol=1e-12)
    # one source at the pooled stage: the garrote reads (S x, S^2 v), so the strength scales the result
    one = srcs[2:]
    kw = dict(var_floor="poisson", shrinkage=True, shrink_stage="pooled", shrink_rule="garrote")
    base = build.pooled_delta("TP53", one, axis, **kw)
    assert np.allclose(build.pooled_delta("TP53", one, axis, source_scale={"extra": 2.0}, **kw), 2.0 * base,
                       rtol=1e-12, atol=1e-15)


# ------------------------------------------------------------------------------------- submit.build

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
    (data / "pert_counts.csv").write_text("target_gene\n" + "\n".join(TARGETS) + "\n")
    _controls(data / "ctx_x.h5ad", [300, 700, 700, 500, 400], 1)
    _controls(data / "ctx_y.h5ad", [500, 505, 595, 900, 300], 2)
    _controls(data / "ctx_z.h5ad", [400, 605, 395, 700, 600], 3)
    _sources(data)
    cfg = {"data_dir": str(data), "gene_names_file": "gene_names.csv", "n_genes": len(GENES),
           "pert_counts_file": "pert_counts.csv", "pert_col": "target_gene", "context_col": "context",
           "control_label": "non-targeting", "phase": "p1", "phases": {"p1": {"contexts": ["X", "Y", "Z"]}},
           "control_files": {"X": "ctx_x.h5ad", "Y": "ctx_y.h5ad", "Z": "ctx_z.h5ad"},
           "submission": {"cells_per_pert": 30, "max_counts_per_cell": 1_000_000, "max_cells": 100_000,
                          "max_stored_entries": 10_000_000}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    return {"cfg": tmp_path / "config.yaml", "data": data, "out": tmp_path / "out", "tmp": tmp_path}


def _run(ch, stem, *extra):
    d = ch["data"]
    assert build.main(["--challenge-config", str(ch["cfg"]), "--emitter", "delta-transfer",
                       "--h1-cache", str(d / "h1.npz"), "--gwps-cache", str(d / "gwps.npz"),
                       "--gwps-control", "control", "--source", f"{d / 'extra.npz'}:control",
                       "--out", str(ch["out"] / stem), "--no-pack", "--min-libsize", "10",
                       "--var-floor", "poisson", "--shrink-stage", "pooled", "--shrink-rule", "adaptive",
                       "--alpha", "1.35", "--alpha-bulk", "1.6", "--emit-lambda", "0.5", *extra]) == 0
    return ad.read_h5ad(ch["out"] / f"{stem}.h5ad")


def test_a_zero_vote_writes_the_routed_file(ch):
    routed = _run(ch, "routed", "--source-context", "extra=X,Z")
    voted = _run(ch, "voted", "--source-weight", "extra=Y:0")
    assert (routed.X != voted.X).nnz == 0
    rec = json.loads((ch["out"] / "voted.source_weight.json").read_text())
    assert rec["pools"] == [["X", "Z"], ["Y"]]
    args = json.loads((ch["out"] / "voted.args.json").read_text())
    assert args["source_weight"] == ["extra=Y:0"]
    assert "source_weight" not in json.loads((ch["out"] / "routed.args.json").read_text())
    assert not (ch["out"] / "routed.source_weight.json").exists()


def test_a_table_row_acts_on_its_knockdown_and_context(ch, monkeypatch):
    """The per-knockdown row moves only its (context, knockdown) prediction; compared on the vectors handed to
    the emitter, since a changed block shifts the random stream every later block in that context draws from."""
    from sidechain.models.count_emitters import PoissonEmitter
    seen: dict = {}
    orig = PoissonEmitter.emit_dual

    def spy(self, n, d_cell, d_bulk, **kw):
        seen.setdefault(PoissonEmitter.stem, []).append((self.p.name, np.array(d_cell)))
        return orig(self, n, d_cell, d_bulk, **kw)
    monkeypatch.setattr(PoissonEmitter, "emit_dual", spy)
    p = ch["tmp"] / "w.parquet"
    pd.DataFrame({"source": ["extra"], "context": ["Y"], "target": ["TP53"], "weight": [0.0]}).to_parquet(p)
    for stem, extra in (("plain", []), ("routed", ["--source-context", "extra=X,Z"]),
                        ("table", ["--source-weight-table", str(p)])):
        PoissonEmitter.stem = stem
        _run(ch, stem, *extra)

    def vec(stem, ctx, tgt):     # the per-context write pass emits the targets in order, once each
        rows = [v for c, v in seen[stem] if c == ctx]
        return rows[-len(TARGETS):][TARGETS.index(tgt)]
    for ctx, tgt, like in (("Y", "TP53", "routed"), ("Y", "B", "plain"), ("X", "TP53", "plain"),
                           ("Z", "B", "plain")):
        assert np.array_equal(vec("table", ctx, tgt), vec(like, ctx, tgt)), (ctx, tgt)
    assert not np.array_equal(vec("table", "Y", "TP53"), vec("plain", "Y", "TP53"))


def test_a_vote_moves_and_a_strength_moves(ch):
    plain = _run(ch, "plain")
    up = _run(ch, "up", "--source-weight", "extra=X:3", "--source-weight", "extra=Y:3", "--source-weight", "extra=Z:3")
    assert (plain.X != up.X).nnz > 0
    sc = _run(ch, "sc", "--source-scale", "extra=1.7")
    assert (plain.X != sc.X).nnz > 0
    assert json.loads((ch["out"] / "sc.source_scale.json").read_text()) == {"source_scale": {"extra": 1.7}}
    assert json.loads((ch["out"] / "up.source_weight.json").read_text())["pools"] == [["X", "Y", "Z"]]


def test_a_model_named_stem_refuses_the_flags(ch, capsys):
    with pytest.raises(SystemExit):
        _run(ch, "ser-99abefsw_vote_v1", "--bulk-anchor", "pooled", "--source-weight", "extra=Y:0")
    assert "no registered knob letter" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------- eval.loco

def test_loco_takes_the_vote_and_refuses_the_delta_cache(tmp_path):
    from sidechain.eval import loco

    srcs = _sources(tmp_path)
    rng = np.random.default_rng(3)
    basal = rng.uniform(100, 2000, size=len(GENES))
    rows, labels = [], []
    for _ in range(30):
        rows.append(rng.poisson(basal * 40 / basal.sum() * 1000)); labels.append("non-targeting")
    for tg in TARGETS:
        for _ in range(8):
            rows.append(rng.poisson(basal * 40 / basal.sum() * 1000)); labels.append(tg)
    real = ad.AnnData(X=sp.csr_matrix(np.asarray(rows, dtype=np.float32)),
                      obs=pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(len(rows))]),
                      var=pd.DataFrame(index=GENES))
    real.write_h5ad(tmp_path / "real.h5ad")
    kw = dict(pert_col="perturbation", control="non-targeting", shrinkage=False, var_floor="poisson",
              min_libsize=0.0, seed=0)
    loco.build_transfer_prediction(tmp_path / "real.h5ad", srcs[:2], tmp_path / "two.h5ad", **kw)
    loco.build_transfer_prediction(tmp_path / "real.h5ad", srcs, tmp_path / "zero.h5ad",
                                   source_vote=SourceVote("k562", {"extra": 0.0}, {}, {}), **kw)
    a, b = ad.read_h5ad(tmp_path / "two.h5ad"), ad.read_h5ad(tmp_path / "zero.h5ad")
    assert (a.X != b.X).nnz == 0
    with pytest.raises(SystemExit, match="no delta cache"):
        loco.build_transfer_prediction(tmp_path / "real.h5ad", srcs, tmp_path / "c.h5ad",
                                       source_scale={"extra": 2.0},
                                       delta_cache={"dir": tmp_path, "sources": []}, **kw)
