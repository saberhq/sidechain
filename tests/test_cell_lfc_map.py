"""`--cell-lfc-map` on `sidechain.eval.loco` and `sidechain.submit.build` (T134; `sidechain.submit.cell_map`).

Pinned here: the map replaces each per-cell change by its bin's size with the sign kept (0 and NaN stay); a bad file
is refused before any work; in both entry points the per-cell channel the emitter receives is the map of what it
received without the flag, the knockdown's own gene pinned after it, and the summed-profile channel is the same
vector; off, nothing is recorded and nothing moves; the flag is refused on a one-channel emission and on a
model-named stem.
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
from sidechain.models import count_emitters as ce
from sidechain.submit import build
from sidechain.submit.cell_map import CellMap

MAP = {"lower": [0.0, 0.05, 0.2, 0.5], "size": [0.1, 0.3, 0.45, 0.8]}


@pytest.fixture
def map_file(tmp_path):
    p = tmp_path / "map.json"
    p.write_text(json.dumps(MAP))
    return p


@pytest.fixture
def spy(monkeypatch):
    calls = []
    orig = ce.PoissonEmitter.emit_dual

    def emit_dual(self, n, d_cell, d_bulk, **kw):
        calls.append((None if d_cell is None else np.array(d_cell, dtype=np.float64),
                      None if d_bulk is None else np.array(d_bulk, dtype=np.float64)))
        return orig(self, n, d_cell, d_bulk, **kw)
    monkeypatch.setattr(ce.PoissonEmitter, "emit_dual", emit_dual)
    return calls


# -- the map ---------------------------------------------------------------------------------------------------

def test_the_map_is_a_step_on_the_size_with_the_sign_kept(map_file):
    m = CellMap.load(map_file)
    d = np.array([0.0, 0.01, -0.04, 0.05, -0.19, 0.2, 0.49, -0.5, 3.0, np.nan])
    assert np.array_equal(m.apply(d), np.array([0.0, 0.1, -0.1, 0.3, -0.3, 0.45, 0.45, -0.8, 0.8, np.nan]),
                          equal_nan=True)
    rec = m.record()
    assert rec["lower"] == MAP["lower"] and rec["size"] == MAP["size"] and len(rec["sha256"]) == 64


@pytest.mark.parametrize("spec, msg", [
    ({"lower": [0.1, 0.2], "size": [0.1, 0.2]}, "start at 0"),
    ({"lower": [0.0, 0.2, 0.2], "size": [0.1, 0.2, 0.3]}, "increase strictly"),
    ({"lower": [0.0, 0.2], "size": [0.1, -0.2]}, ">= 0"),
    ({"lower": [0.0, 0.2], "size": [0.1]}, "same"),
    ({"lower": [0.0]}, "needs `lower` and `size`"),
])
def test_a_bad_map_is_refused(tmp_path, spec, msg):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(spec))
    with pytest.raises(SystemExit, match=msg):
        CellMap.load(p)


# -- eval.loco -------------------------------------------------------------------------------------------------

G, N_LAB = 60, 6
KW = dict(pert_col="perturbation", control="non-targeting", var_floor="poisson", emit_lambda=0.5, alpha=1.35,
          bulk_anchor="pooled", min_libsize=0.0, shrink_stage="pooled", shrink_rule="adaptive",
          dual_fallback="anchor", emit_shape="controls", alpha_bulk=2.0, seed=2)


@pytest.fixture
def fold(tmp_path):
    rng = np.random.default_rng(5)
    genes = np.array([f"g{i}" for i in range(G)], dtype=object)
    labs = [f"g{i}" for i in range(N_LAB)]
    basal = rng.uniform(50, 3000, size=G)
    mean = np.stack([basal] + [basal * np.exp2(rng.normal(0, 0.3, G)) for _ in labs])
    n = np.full(len(labs) + 1, 300, dtype=np.int64)
    src = PseudobulkSums(labels=["ctrl", *labs], genes=genes.copy(), count_sum=mean * n[:, None],
                         cpm_sum=mean * n[:, None], cpm_sq_sum=(mean**2 + mean) * n[:, None],
                         n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["t"])
    rows, obs = [], []
    for lab, k in [("non-targeting", 120)] + [(x, 25) for x in labs]:
        for _ in range(k):
            rows.append(rng.poisson(basal / basal.sum() * rng.integers(2000, 9000)))
            obs.append(lab)
    real = ad.AnnData(X=sp.csr_matrix(np.asarray(rows, dtype=np.float32)),
                      obs=pd.DataFrame({"perturbation": obs}, index=[f"c{i}" for i in range(len(rows))]),
                      var=pd.DataFrame(index=genes.astype(str)))
    real.write_h5ad(tmp_path / "real.h5ad")
    return tmp_path, tmp_path / "real.h5ad", [(src, "ctrl")]


def _same_but_mapped(off, on, m, own, targets=None):
    assert len(off) == len(on) > 0
    for k, ((c0, b0), (c1, b1)) in enumerate(zip(off, on)):
        assert np.array_equal(b0, b1)                                   # the summed profile: the same vector
        want = m.apply(c0) if targets is None else m.apply(c0, target=targets[k])
        want[own[k]] = c0[own[k]]                                       # the own gene: pinned after the map
        assert np.allclose(c1, want, rtol=0, atol=1e-6)
        assert not np.allclose(c0, c1)


def test_loco_maps_the_per_cell_channel_only(fold, map_file, spy):
    tmp, real, sources = fold
    off = loco.build_transfer_prediction(real, sources, tmp / "off.h5ad", **KW)
    n_off = len(spy)
    on = loco.build_transfer_prediction(real, sources, tmp / "on.h5ad", **KW, cell_lfc_map=map_file)
    assert "cell_lfc_map" not in off and on["cell_lfc_map"]["sha256"] == CellMap.load(map_file).sha256
    _same_but_mapped(spy[:n_off], spy[n_off:], CellMap.load(map_file), own=list(range(N_LAB)))


def test_loco_refuses_the_map_on_a_one_channel_emission(fold, map_file):
    tmp, real, sources = fold
    kw = {**KW, "alpha_bulk": None, "bulk_anchor": "mean_cpm", "dual_fallback": "template", "emit_shape": "template"}
    with pytest.raises(SystemExit, match="per-cell channel alone"):
        loco.build_transfer_prediction(real, sources, tmp / "x.h5ad", **kw, cell_lfc_map=map_file)


# -- submit.build ------------------------------------------------------------------------------------------------

GB = 24
GENES = [f"G{i}" for i in range(GB)]
PERTS = GENES[:6]


@pytest.fixture
def ch(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "gene_names.csv").write_text("gene_name\n" + "\n".join(GENES) + "\n")
    (data / "pert_counts.csv").write_text("target_gene\n" + "\n".join(PERTS) + "\n")
    rng = np.random.default_rng(1)
    base = rng.uniform(0.5, 5.0, size=GB)
    lib = np.exp(rng.normal(0, 0.9, size=(200, 1))) * 40
    ad.AnnData(X=sp.csr_matrix(rng.poisson(base / base.sum() * lib).astype(np.float32)),
               obs=pd.DataFrame(index=[f"c{i}" for i in range(200)]),
               var=pd.DataFrame(index=GENES)).write_h5ad(data / "ctx_x.h5ad")
    for name, label in (("h1.npz", "non-targeting"), ("gwps.npz", "control")):
        ctrl = rng.uniform(5_000, 80_000, size=GB)
        ctrl *= 1e6 / ctrl.sum()
        m = np.asarray([ctrl] + [ctrl * np.exp2(rng.normal(0, 0.1 + 0.05 * k, size=GB)) for k in range(len(PERTS))])
        m = m * 1e6 / m.sum(axis=1, keepdims=True)
        n = np.full(len(m), 500, dtype=np.int64)
        PseudobulkSums(labels=[label, *PERTS], genes=np.array(GENES, dtype=object), count_sum=m * n[:, None],
                       cpm_sum=m * n[:, None], cpm_sq_sum=(m**2 + m) * n[:, None], n_cells=n,
                       libsize_sum=n.astype(float) * 2e4, sources=["test"]).save(data / name)
    cfg = {"data_dir": str(data), "gene_names_file": "gene_names.csv", "n_genes": GB,
           "pert_counts_file": "pert_counts.csv", "pert_col": "target_gene", "context_col": "context",
           "control_label": "non-targeting", "phase": "p1", "phases": {"p1": {"contexts": ["X"]}},
           "control_files": {"X": "ctx_x.h5ad"},
           "submission": {"cells_per_pert": 50, "max_counts_per_cell": 1_000_000, "max_cells": 100_000,
                          "max_stored_entries": 10_000_000}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    return {"cfg": tmp_path / "config.yaml", "data": data, "out": tmp_path / "out"}


def _argv(ch, stem, *extra, anchor="pooled"):
    return ["--challenge-config", str(ch["cfg"]), "--emitter", "delta-transfer",
            "--h1-cache", str(ch["data"] / "h1.npz"), "--gwps-cache", str(ch["data"] / "gwps.npz"),
            "--out", str(ch["out"] / stem), "--no-pack", "--min-libsize", "10", "--no-shrink",
            "--alpha", "1.35", "--emit-lambda", "0.5", "--bulk-anchor", anchor, *extra]


def test_build_maps_the_per_cell_channel_only(ch, map_file, spy):
    assert build.main(_argv(ch, "off", "--alpha-bulk", "2.0")) == 0
    n_off = len(spy)
    assert build.main(_argv(ch, "on", "--alpha-bulk", "2.0", "--cell-lfc-map", str(map_file))) == 0
    _same_but_mapped(spy[:n_off], spy[n_off:], CellMap.load(map_file), own=list(range(len(PERTS))))
    assert "cell_lfc_map" not in json.loads((ch["out"] / "off.args.json").read_text())
    assert "cell_lfc_map" not in json.loads((ch["out"] / "off.dual.json").read_text())
    rec = json.loads((ch["out"] / "on.dual.json").read_text())["cell_lfc_map"]
    assert rec["sha256"] == CellMap.load(map_file).sha256 and rec["size"] == MAP["size"]


def test_build_refuses_the_map_where_it_cannot_act(ch, map_file, capsys):
    with pytest.raises(SystemExit):
        build.main(_argv(ch, "one", "--cell-lfc-map", str(map_file), anchor="mean_cpm"))
    assert "per-cell channel alone" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        build.main(_argv(ch, "ser-99benw_t84_v1", "--alpha-bulk", "2.0", "--cell-lfc-map", str(map_file)))
    assert "no registered knob letter" in capsys.readouterr().err


# -- the scale: a per-gene and per-target factor on the mapped size (T134, the screen-depth map) ------------------

def _scaled(tmp_path, scale=None, contexts=None, name="scaled.json"):
    p = tmp_path / name
    p.write_text(json.dumps({**MAP, **({"scale": scale} if scale else {}), **({"contexts": contexts} if contexts else {})}))
    return p


def test_the_scale_multiplies_the_mapped_size_gene_by_target_clipped(tmp_path):
    m = CellMap.load(_scaled(tmp_path, {"gene": {"a": 2.0, "b": 0.5, "c": 8.0}, "target": {"K1": 1.0, "K2": 0.5},
                                        "clip": [0.25, 4.0], "gene_default": 1.0}))
    assert m.scaled and m.record()["scale"] == {"genes": 3, "gene_default": 1.0, "targets": 2, "clip": [0.25, 4.0]}
    with pytest.raises(SystemExit, match="bound"):
        m.apply(np.array([0.1, 0.1, 0.1, 0.1]))
    b = m.bind(["a", "b", "c", "d"], ["K1", "K2"])
    d = np.array([0.1, -0.1, 0.6, 0.0])
    assert np.allclose(b.apply(d, target="K1"), [0.6, -0.15, 3.2, 0.0])          # 0.3x2, -0.3x0.5, 0.8x(8 -> 4), 0 stays
    assert np.allclose(b.apply(d, target="K2"), [0.3, -0.075, 3.2, 0.0])         # x0.5 more; 0.25 not reached
    assert np.isnan(b.apply(np.array([np.nan, 0.1, 0.1, 0.1]), target="K1")[0])
    with pytest.raises(SystemExit, match="no target factor"):
        m.bind(["a", "b", "c", "d"], ["K1", "K3"])
    with pytest.raises(SystemExit, match="another gene space"):
        m.bind(["x", "y", "z", "a"], ["K1"])
    assert CellMap.load(_scaled(tmp_path, None, name="plain.json")).bind(["a"], ["K1"]).apply(d).tolist() == \
        [0.3, -0.3, 0.8, 0.0]                                                      # unscaled: as before, target ignored


def _pairs(tmp_path, rows, name="pairs.parquet"):
    p = tmp_path / name
    pd.DataFrame(rows, columns=["target", "feature", "factor"]).to_parquet(p)
    return name


def test_a_pair_table_multiplies_one_knockdown_s_gene_before_the_clip(tmp_path):
    rel = _pairs(tmp_path, [("K1", "a", 0.5), ("K1", "c", 3.0), ("K2", "b", 0.0)])
    m = CellMap.load(_scaled(tmp_path, {"gene": {"a": 2.0, "b": 1.0, "c": 1.0}, "target": 1.0, "pair": rel, "clip": [0.25, 4.0]}))
    rec = m.record()["scale"]["pair"]
    assert rec["pairs"] == 3 and rec["targets"] == 2 and len(rec["sha256"]) == 64
    b = m.bind(["a", "b", "c", "d"], ["K1", "K2", "K3"])
    d = np.array([0.1, -0.1, 0.6, 0.0])
    assert np.allclose(b.apply(d, target="K1"), [0.3, -0.3, 2.4, 0.0])           # 0.3x2x0.5, -0.3, 0.8x3, 0 stays
    assert np.allclose(b.apply(d, target="K2"), [0.6, -0.075, 0.8, 0.0])         # b x0 clipped up to 0.25
    assert np.allclose(b.apply(d, target="K3"), [0.6, -0.3, 0.8, 0.0])           # no pair rows: the gene factor alone
    only = CellMap.load(_scaled(tmp_path, {"pair": rel}, name="only.json")).bind(["a", "b", "c", "d"], ["K1", "K2"])
    assert np.allclose(only.apply(d, target="K2"), [0.3, 0.0, 0.8, 0.0])         # no clip: 0 stays 0
    with pytest.raises(SystemExit, match="another gene space"):
        CellMap.load(_scaled(tmp_path, {"pair": rel}, name="o2.json")).bind(["x", "y", "z"], ["K1", "K2"])


@pytest.mark.parametrize("rows, msg", [
    ([("K1", "a", -1.0)], ">= 0"),
    ([("K1", "a", float("nan"))], ">= 0"),
    ([("K1", "a", 1.0), ("K1", "a", 2.0)], "repeats"),
])
def test_a_bad_pair_table_is_refused(tmp_path, rows, msg):
    rel = _pairs(tmp_path, rows)
    with pytest.raises(SystemExit, match=msg):
        CellMap.load(_scaled(tmp_path, {"pair": rel}))
    with pytest.raises(SystemExit, match="does not exist"):
        CellMap.load(_scaled(tmp_path, {"pair": "missing.parquet"}, name="m.json"))


def test_a_signed_pair_table_may_flip_the_sign(tmp_path):
    rel = _pairs(tmp_path, [("K1", "a", -0.5), ("K1", "c", 2.0), ("K2", "b", -1.0)])
    m = CellMap.load(_scaled(tmp_path, {"pair": rel, "signed": True}))
    assert m.record()["scale"]["pair"]["signed"] is True and m.record()["scale"]["pair"]["pairs"] == 3
    b = m.bind(["a", "b", "c", "d"], ["K1", "K2", "K3"])
    d = np.array([0.1, -0.1, 0.6, 0.0])
    assert np.allclose(b.apply(d, target="K1"), [-0.15, -0.3, 1.6, 0.0])         # 0.3 x -0.5 flips; 0.8 x 2
    assert np.allclose(b.apply(d, target="K2"), [0.3, 0.3, 0.8, 0.0])            # -0.3 x -1 flips
    assert np.allclose(b.apply(d, target="K3"), [0.3, -0.3, 0.8, 0.0])           # no pair rows
    assert np.isnan(b.apply(np.array([np.nan, -0.1, 0.6, 0.0]), target="K1")[0])
    with pytest.raises(SystemExit, match=">= 0"):
        CellMap.load(_scaled(tmp_path, {"pair": rel}, name="unsigned.json"))
    with pytest.raises(SystemExit, match="`signed`"):
        CellMap.load(_scaled(tmp_path, {"pair": rel, "signed": True, "clip": [0.5, 2.0]}, name="c.json"))
    with pytest.raises(SystemExit, match="`signed`"):
        CellMap.load(_scaled(tmp_path, {"gene": {"a": 1.0}, "signed": True}, name="g.json"))
    with pytest.raises(SystemExit, match="`signed`"):
        CellMap.load(_scaled(tmp_path, {"pair": rel, "signed": "yes"}, name="y.json"))
    ok = CellMap.load(_scaled(tmp_path, {"pair": _pairs(tmp_path, [("K1", "a", 0.5)], name="pos.parquet"), "signed": False},
                              name="f.json")).bind(["a", "b"], ["K1"])
    assert np.allclose(ok.apply(np.array([0.1, 0.1]), target="K1"), [0.15, 0.3])


@pytest.mark.parametrize("extra, msg", [
    ({"scale": {"gene": {"a": -1.0}}}, ">= 0"),
    ({"scale": {"gene": {"a": 1.0}, "clip": [2.0, 1.0]}}, "clip"),
    ({"scale": {"gene": {"a": 1.0}, "spread": 2}}, "unknown keys"),
    ({"scale": {"clip": [0.5, 2.0]}}, "needs `gene`, `target` and/or `pair`"),
    ({"scale": {"gene": {"a": 1.0}}, "contexts": {"D": {"gene": {"a": 1.0}}}}, "not both"),
])
def test_a_bad_scale_is_refused(tmp_path, extra, msg):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({**MAP, **extra}))
    with pytest.raises(SystemExit, match=msg):
        CellMap.load(p)


def test_loco_applies_the_scale_on_the_per_cell_channel_only(fold, tmp_path, spy):
    tmp, real, sources = fold
    scale = {"gene": {f"g{i}": 0.5 + (i % 4) for i in range(G)}, "target": 0.8, "clip": [0.25, 4.0]}
    p = _scaled(tmp_path, scale)
    off = loco.build_transfer_prediction(real, sources, tmp / "off.h5ad", **KW)
    n_off = len(spy)
    on = loco.build_transfer_prediction(real, sources, tmp / "on.h5ad", **KW, cell_lfc_map=p)
    assert on["cell_lfc_map"]["scale"]["genes"] == G
    m = CellMap.load(p).bind([f"g{i}" for i in range(G)], [f"g{i}" for i in range(N_LAB)])
    _same_but_mapped(spy[:n_off], spy[n_off:], m, own=list(range(N_LAB)))
    with pytest.raises(SystemExit, match="one line"):
        loco.build_transfer_prediction(real, sources, tmp / "x.h5ad", **KW,
                                       cell_lfc_map=_scaled(tmp_path, contexts={"X": scale}, name="ctx.json"))


def test_loco_applies_a_pair_table_on_the_per_cell_channel_only(fold, tmp_path, spy):
    tmp, real, sources = fold
    rows = [(f"g{k}", f"g{i}", 0.25 + ((i + k) % 5) * 0.4) for k in range(N_LAB) for i in range(G)]
    p = _scaled(tmp_path, {"pair": _pairs(tmp_path, rows), "clip": [0.25, 4.0]})
    off = loco.build_transfer_prediction(real, sources, tmp / "off.h5ad", **KW)
    n_off = len(spy)
    on = loco.build_transfer_prediction(real, sources, tmp / "on.h5ad", **KW, cell_lfc_map=p)
    assert on["cell_lfc_map"]["scale"]["pair"]["pairs"] == N_LAB * G
    m = CellMap.load(p).bind([f"g{i}" for i in range(G)], [f"g{i}" for i in range(N_LAB)])
    _same_but_mapped(spy[:n_off], spy[n_off:], m, own=list(range(N_LAB)), targets=[f"g{k}" for k in range(N_LAB)])


def test_build_applies_the_context_s_own_scale(ch, tmp_path, spy):
    scale = {"gene": {g: 0.5 + (i % 3) for i, g in enumerate(GENES)}, "target": 0.9, "clip": [0.25, 4.0]}
    p = _scaled(tmp_path, contexts={"X": scale})
    assert build.main(_argv(ch, "off", "--alpha-bulk", "2.0")) == 0
    n_off = len(spy)
    assert build.main(_argv(ch, "on", "--alpha-bulk", "2.0", "--cell-lfc-map", str(p))) == 0
    _same_but_mapped(spy[:n_off], spy[n_off:], CellMap.load(p).bind(GENES, PERTS, context="X"),
                     own=list(range(len(PERTS))))
    rec = json.loads((ch["out"] / "on.dual.json").read_text())["cell_lfc_map"]
    assert rec["scale"]["X"]["genes"] == GB
    q = _scaled(tmp_path, contexts={"Y": scale}, name="other.json")
    with pytest.raises(SystemExit, match="no scale for context 'X'"):
        build.main(_argv(ch, "miss", "--alpha-bulk", "2.0", "--cell-lfc-map", str(q)))
