"""`--source-context NAME=CTX`: one source pooled in the named contexts only, on `submit.build` and `eval.loco`.

    .venv/bin/python -m pytest -q tests/test_source_context.py

Writes temporary files only. Notes: private/code-notes/tests/test_source_context.py.md
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
from sidechain.eval import loco
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


def _controls_h5ad(path, counts_per_cell, n=12):
    X = sp.csr_matrix(np.tile(np.asarray(counts_per_cell, float), (n, 1)))
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=GENES)).write_h5ad(path)


@pytest.fixture()
def challenge(tmp_path):
    """Two contexts, three sources; `extra` moves TP53's gene A the other way from the two caches."""
    data = tmp_path / "data"; data.mkdir()
    (data / "gene_names.csv").write_text("gene_name\n" + "\n".join(GENES) + "\n")
    (data / "pert_counts.csv").write_text("target_gene\nTP53\n")
    _controls_h5ad(data / "ctx_x.h5ad", [600, 700, 700])
    _controls_h5ad(data / "ctx_y.h5ad", [900, 505, 595])
    _cache(data / "h1.npz", "non-targeting",
           [100000.0, 450000.0, 450000.0], [200000.0, 400000.0, 400000.0])
    _cache(data / "gwps.npz", "control",
           [100000.0, 450000.0, 450000.0], [210000.0, 395000.0, 395000.0])
    _cache(data / "extra.npz", "control",
           [300000.0, 350000.0, 350000.0], [30000.0, 485000.0, 485000.0])
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


def _run(ch, stem, extra):
    rc = build.main([
        "--challenge-config", str(ch["cfg"]), "--emitter", "delta-transfer",
        "--h1-cache", str(ch["data"] / "h1.npz"), "--gwps-cache", str(ch["data"] / "gwps.npz"),
        "--out", str(ch["out"] / stem), "--no-pack", "--min-libsize", "100",
        "--no-shrink", "--emit-lambda", "0.4", *extra])
    assert rc == 0
    return ad.read_h5ad(ch["out"] / f"{stem}.h5ad")


def _block(pred, ctx):
    return pred[pred.obs["context"].astype(str) == ctx].X.toarray()


def _extra(ch):
    return ["--source", f"{ch['data'] / 'extra.npz'}:control"]


def test_a_routed_context_is_the_build_of_its_own_pool(challenge):
    """X pools all three sources and Y the two caches: each block is its own pool's build, cell for cell."""
    two = _run(challenge, "two", [])
    three = _run(challenge, "three", _extra(challenge))
    routed = _run(challenge, "routed", _extra(challenge) + ["--source-context", "extra=X"])
    assert np.array_equal(_block(routed, "X"), _block(three, "X"))
    assert np.array_equal(_block(routed, "Y"), _block(two, "Y"))
    assert not np.array_equal(_block(three, "X"), _block(two, "X"))     # the source does move X
    assert not np.array_equal(_block(three, "Y"), _block(two, "Y"))     # and would move Y


def test_the_routing_record_names_each_pool(challenge):
    _run(challenge, "rec", _extra(challenge) + ["--source-context", "extra=X"])
    rec = json.loads((challenge["out"] / "rec.routing.json").read_text())
    assert rec["source_context"] == {"extra": ["X"]}
    assert [(p["contexts"], p["sources"]) for p in rec["pools"]] == [
        (["X"], ["gwps", "h1", "extra"]), (["Y"], ["gwps", "h1"])]
    assert all(p["fallback"] == 0 and p["covered"] == 1 for p in rec["pools"])
    args = json.loads((challenge["out"] / "rec.args.json").read_text())
    assert args["source_context"] == ["extra=X"]


def test_routed_to_every_context_is_one_pool(challenge):
    three = _run(challenge, "three", _extra(challenge))
    every = _run(challenge, "every", _extra(challenge) + ["--source-context", "extra=X,Y"])
    assert np.array_equal(every.X.toarray(), three.X.toarray())
    rec = json.loads((challenge["out"] / "every.routing.json").read_text())
    assert [p["contexts"] for p in rec["pools"]] == [["X", "Y"]]


def test_off_the_records_are_what_a_build_before_the_flag_wrote(challenge):
    _run(challenge, "plain", _extra(challenge))
    assert "source_context" not in json.loads((challenge["out"] / "plain.args.json").read_text())
    assert not (challenge["out"] / "plain.routing.json").exists()


@pytest.mark.parametrize("route, needle", [
    ("nosuch=X", "match no source"),
    ("extra=Z", "does not write"),
    ("extra", "is not NAME=CTX"),
    ("extra=X,X", "names a context twice"),
])
def test_a_route_that_misses_stops_the_build(challenge, route, needle, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(challenge, "bad", _extra(challenge) + ["--source-context", route])
    assert needle in str(exc.value.code)
    assert not (challenge["out"] / "bad.h5ad").exists()


def test_a_context_left_with_no_source_is_refused():
    with pytest.raises(SystemExit, match="no source"):
        build.route_sources([_Named("one"), _Named("two")], {"one": ("X",), "two": ("X",)}, ["X", "Y"])


def test_a_stem_shared_by_two_sources_cannot_be_routed():
    with pytest.raises(SystemExit, match="more than one"):
        build.route_sources([_Named("h1"), _Named("h1")], {"h1": ("X",)}, ["X", "Y"])


def test_pools_keep_order_and_identity_and_group_by_pool():
    s = [_Named("k562"), _Named("other"), _Named("h1"), _Named("wide")]
    pools = build.route_sources(s, {"other": ("A",), "wide": ("A", "C")}, ["A", "B", "C"])
    assert [x.sidechain_name for x in pools["A"]] == ["k562", "other", "h1", "wide"]
    assert [x.sidechain_name for x in pools["B"]] == ["k562", "h1"]
    assert [x.sidechain_name for x in pools["C"]] == ["k562", "h1", "wide"]
    assert pools["B"][0] is s[0]
    groups = build.context_groups(pools, ["A", "B", "C"])
    assert [g[0] for g in groups] == [("A",), ("B",), ("C",)]
    groups = build.context_groups(build.route_sources(s, {"other": ("A",)}, ["A", "B", "C"]), ["A", "B", "C"])
    assert [g[0] for g in groups] == [("A",), ("B", "C")]


def test_a_model_named_stem_refuses_the_flag_until_it_has_a_letter(challenge, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(challenge, "ser-99_fixture_v1", _extra(challenge) + ["--source-context", "extra=X"])
    assert exc.value.code == 2
    assert "--source-context has no registered knob letter" in capsys.readouterr().err


class _Named:
    def __init__(self, name):
        self.sidechain_name = name


class _Stop(Exception):
    pass


def _loco(ch, tmp_path, extra, monkeypatch):
    seen = {}

    def stub(real, sources, *a, **k):
        seen["sources"] = [build.source_name(s) for s in sources]
        raise _Stop

    monkeypatch.setattr(loco, "build_transfer_prediction", stub)
    argv = ["--real", str(tmp_path / "real.h5ad"), "--bundle", str(tmp_path / "bundle"),
            "--out", str(tmp_path / "arm"),
            "--source", f"{ch['data'] / 'gwps.npz'}:control",
            "--source", f"{ch['data'] / 'extra.npz'}:control", *extra]
    with pytest.raises(_Stop):
        loco.main(argv)
    return seen["sources"]


def test_loco_pools_a_routed_source_only_for_its_context(challenge, tmp_path, monkeypatch):
    route = ["--source-context", "extra=A"]
    assert _loco(challenge, tmp_path, route + ["--context", "A"], monkeypatch) == ["gwps", "extra"]
    assert _loco(challenge, tmp_path, route + ["--context", "B"], monkeypatch) == ["gwps"]
    assert _loco(challenge, tmp_path, [], monkeypatch) == ["gwps", "extra"]


@pytest.mark.parametrize("extra", [["--source-context", "extra=A"], ["--context", "A"]])
def test_loco_needs_both_halves_of_the_switch(challenge, tmp_path, extra):
    with pytest.raises(SystemExit) as exc:
        loco.main(["--real", str(tmp_path / "real.h5ad"), "--bundle", str(tmp_path / "bundle"),
                   "--out", str(tmp_path / "arm"),
                   "--source", f"{challenge['data'] / 'extra.npz'}:control", *extra])
    assert exc.value.code == 2
