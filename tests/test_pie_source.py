"""`--prediction-source pie` (T85): PIE's predicted log2FC where SER's pooled delta stands.

Pinned here: the reader maps PIE's genes onto the axis (unscored genes 0, off-axis genes dropped), sets values
at or below the floor to 0 and clips to +-5, refuses a missing context or a doubled row; on `eval.loco` the
default `ser` is the run without the flag, a PIE arm emits PIE's map at alpha with the knockdown's own gene
pinned (whatever PIE says of it), a perturbation with no row is uncovered, and the record names the source; on
`submit.build` the default's outputs and records are the build without the flag, a PIE build carries PIE's map
per context with the generic shift for a missing perturbation, and a model-named stem or a pooled source is
refused.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp

from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.models.count_emitters import PoissonEmitter
from sidechain.submit import build
from sidechain.submit import pie_source as ps
from test_alpha_table import _argv, challenge  # noqa: F401  (the fixture)


def _parquet(path, genes, rows, dataset="ds"):
    """rows: [(context, perturbation, lfc list)]"""
    t = pa.table({
        "dataset": [dataset] * len(rows),
        "context": [r[0] for r in rows],
        "perturbation": [r[1] for r in rows],
        "p_de": [[0.5] * len(genes) for _ in rows],
        "lfc_pred": [list(map(float, r[2])) for r in rows],
        "delta_p_pred": [[0.0] * len(genes) for _ in rows],
    })
    t = t.replace_schema_metadata({"pie": json.dumps({"format_version": 1, "genes": {dataset: list(genes)}})})
    pq.write_table(t, path)
    return path


# ------------------------------------------------------------------------------------------ the reader

def test_reader_maps_genes_floors_clips_and_counts(tmp_path):
    f = _parquet(tmp_path / "p.parquet", ["D", "B", "A", "Z"], [
        ("X", "A", [7.0, -12.0, 0.5, 3.0]),
        ("X", "B", [-10.0, -9.5, np.nan, -6.0]),
        ("Y", "A", [1.0, 1.0, 1.0, 1.0]),
    ])
    m, rec = ps.pie_shift_map(f, "X", ["A", "B", "C", "D"], ["A", "B", "Q"])
    # axis A B C D <- PIE's A, B, (C unscored), D; Z is off the axis
    assert np.array_equal(m["A"], [0.5, 0.0, 0.0, 5.0])          # -12 floored to 0, 7 clipped to 5
    assert np.array_equal(m["B"], [0.0, -5.0, 0.0, 0.0])         # NaN to 0, -9.5 clipped, -10 floored
    assert set(m) == {"A", "B"}                                  # Q has no row: absent from the map
    assert rec["axis_genes_scored"] == 3 and rec["axis_genes_unscored"] == 1 and rec["pie_genes_off_axis"] == 1
    assert rec["values_floored"] == 2 and rec["values_clipped"] == 3 and rec["values_non_finite"] == 1
    assert rec["perturbations_missing"] == 1 and rec["missing"] == ["Q"] and rec["rows"] == 2
    assert rec["context"] == "X" and rec["dataset"] == "ds" and len(rec["sha256"]) == 64
    assert ps.pie_shift_map(f, "Y", ["A"])[0]["A"].tolist() == [1.0]


def test_reader_refuses_a_missing_context_and_a_doubled_row(tmp_path):
    f = _parquet(tmp_path / "p.parquet", ["A"], [("X", "A", [1.0]), ("X", "A", [2.0])])
    with pytest.raises(SystemExit, match="twice"):
        ps.pie_shift_map(f, "X", ["A"])
    with pytest.raises(SystemExit, match="no rows for context 'Y'"):
        ps.pie_shift_map(f, "Y", ["A"])


def test_build_context_specs():
    assert ps.parse_pie_contexts([], ["A", "B"]) == {"A": "A", "B": "B"}
    assert ps.parse_pie_contexts(["unnamed"], ["A", "B"]) == {"A": "unnamed", "B": "unnamed"}
    assert ps.parse_pie_contexts(["unnamed", "B=hela"], ["A", "B"]) == {"A": "unnamed", "B": "hela"}
    with pytest.raises(SystemExit):
        ps.parse_pie_contexts(["Q=x"], ["A"])


# ------------------------------------------------------------------------------------------- eval.loco

GL = np.array([f"g{i}" for i in range(12)], dtype=object)
TARGETS = ["g0", "g1", "g2"]


def _fold(tmp_path):
    rng = np.random.default_rng(3)

    def line(seed):
        r = np.random.default_rng(seed)
        labels = ["ctrl"] + TARGETS
        mean = r.uniform(100, 2000, size=(len(labels), len(GL)))
        n = np.full(len(labels), 2000, dtype=np.int64)
        return (PseudobulkSums(labels=labels, genes=GL.copy(), count_sum=mean * n[:, None],
                               cpm_sum=mean * n[:, None], cpm_sq_sum=(mean**2 + mean) * n[:, None],
                               n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["test"]), "ctrl")
    basal = rng.uniform(100, 2000, size=len(GL))
    rows, labels = [], []
    for i in range(40):
        rows.append(rng.poisson(basal * (20 + i) / basal.sum() * 1000)); labels.append("non-targeting")
    for t in TARGETS:
        for _ in range(6):
            rows.append(rng.poisson(basal * 40 / basal.sum() * 1000)); labels.append(t)
    real = ad.AnnData(X=sp.csr_matrix(np.asarray(rows, dtype=np.float32)),
                      obs=pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(len(rows))]),
                      var=pd.DataFrame(index=GL.astype(str)))
    real.write_h5ad(tmp_path / "real.h5ad")
    return tmp_path / "real.h5ad", [line(1), line(2)]


KW = dict(pert_col="perturbation", control="non-targeting", shrinkage=False, var_floor="poisson",
          min_libsize=0.0, seed=0, alpha=1.35, alpha_bulk=1.0, emit_lambda=0.5, bulk_anchor="pooled")


def test_loco_default_is_the_run_without_the_flag(tmp_path):
    from sidechain.eval import loco

    real, srcs = _fold(tmp_path)
    off = loco.build_transfer_prediction(real, srcs, tmp_path / "off.h5ad", **KW)
    ser = loco.build_transfer_prediction(real, srcs, tmp_path / "ser.h5ad", prediction_source="ser", **KW)
    assert "prediction_source" not in off and {k: v for k, v in off.items() if k != "pred"} == \
        {k: v for k, v in ser.items() if k != "pred"}
    a, b = ad.read_h5ad(tmp_path / "off.h5ad"), ad.read_h5ad(tmp_path / "ser.h5ad")
    assert (a.X != b.X).nnz == 0


def test_loco_pie_emits_pies_map_pinned_and_a_missing_row_uncovered(tmp_path, monkeypatch):
    from sidechain.eval import loco

    real, _ = _fold(tmp_path)
    r = np.random.default_rng(0)
    pie_genes = list(GL[::-1][:10])                       # reversed order, g0 and g1 unscored
    vals = {t: r.normal(0, 2, len(pie_genes)) for t in TARGETS[:2]}
    vals["g0"][3] = -11.0                                 # one floored value
    f1 = _parquet(tmp_path / "a.parquet", pie_genes, [("u", t, vals[t]) for t in TARGETS[:2]])
    seen = {}

    def spy(self, n, log2fc_cell, log2fc_bulk, **kw):
        seen.setdefault(PoissonEmitter.tag, []).append(
            (None if log2fc_cell is None else np.array(log2fc_cell),
             None if log2fc_bulk is None else np.array(log2fc_bulk)))
        return self.emit(n, log2fc_cell)
    monkeypatch.setattr(PoissonEmitter, "emit_dual", spy)
    PoissonEmitter.tag = "pie"
    info = loco.build_transfer_prediction(real, [], tmp_path / "pie.h5ad", prediction_source="pie",
                                          pie_parquet=f1, pie_context="u", **KW)
    assert info["covered_by_sources"] == 2 and info["prediction_source"]["source"] == "pie"
    assert info["prediction_source"]["perturbations_missing"] == 1
    assert info["prediction_source"]["values_floored"] == 1
    (c0, b0), (c1, b1), (c2, b2) = seen["pie"]            # targets sorted: g0, g1, g2
    pos = {g: i for i, g in enumerate(GL)}
    for t, (c, b) in zip(TARGETS[:2], ((c0, b0), (c1, b1))):
        want = np.zeros(len(GL))
        for j, g in enumerate(pie_genes):
            want[pos[g]] = vals[t][j]
        want, _ = ps.to_log2fc(want)
        mask = np.arange(len(GL)) != pos[t]
        assert np.allclose(c[mask], want[mask] * 1.35) and np.allclose(b[mask], want[mask] * 1.0)
        assert c[pos[t]] == b[pos[t]] == build.TARGET_SELF_LOG2FC
    assert c2 is None and b2 is None                      # g2 has no row: uncovered, as no source covering it
    # the knockdown's own gene: whatever PIE says of it, the emitted cells are the same
    monkeypatch.undo()
    pie_genes_own = list(GL[:10])                          # now g0 and g1 are scored: move only their own genes
    v_a = {t: r.normal(0, 1, 10) for t in TARGETS[:2]}
    v_b = {t: v.copy() for t, v in v_a.items()}
    v_b["g0"][0], v_b["g1"][1] = 4.0, -30.0
    fa = _parquet(tmp_path / "own_a.parquet", pie_genes_own, [("u", t, v_a[t]) for t in TARGETS[:2]])
    fb = _parquet(tmp_path / "own_b.parquet", pie_genes_own, [("u", t, v_b[t]) for t in TARGETS[:2]])
    loco.build_transfer_prediction(real, [], tmp_path / "oa.h5ad", prediction_source="pie", pie_parquet=fa,
                                   pie_context="u", **KW)
    loco.build_transfer_prediction(real, [], tmp_path / "ob.h5ad", prediction_source="pie", pie_parquet=fb,
                                   pie_context="u", **KW)
    a, b = ad.read_h5ad(tmp_path / "oa.h5ad"), ad.read_h5ad(tmp_path / "ob.h5ad")
    assert (a.X != b.X).nnz == 0


def test_loco_pie_refuses_a_pool(tmp_path):
    from sidechain.eval import loco

    real, srcs = _fold(tmp_path)
    f = _parquet(tmp_path / "a.parquet", ["g3"], [("u", "g0", [1.0])])
    with pytest.raises(SystemExit, match="pooled delta stands"):
        loco.build_transfer_prediction(real, srcs, tmp_path / "x.h5ad", prediction_source="pie",
                                       pie_parquet=f, pie_context="u", **KW)


class _Stop(Exception):
    pass


def _loco_kwargs(tmp_path, extra, monkeypatch):
    from sidechain.eval import loco

    seen = {}

    def stub(real, sources, out, **k):
        seen.update(sources=sources, kwargs=k)
        raise _Stop
    monkeypatch.setattr(loco, "build_transfer_prediction", stub)
    with pytest.raises(_Stop):
        loco.main(["--real", str(tmp_path / "real.h5ad"), "--bundle", str(tmp_path / "bundle"),
                   "--out", str(tmp_path / "arm"), *extra])
    return seen


def test_loco_main_default_and_switch(tmp_path, monkeypatch, challenge):  # noqa: F811
    src = ["--source", f"{challenge['data'] / 'gwps.npz'}:control"]
    plain = _loco_kwargs(tmp_path, src, monkeypatch)
    ser = _loco_kwargs(tmp_path, src + ["--prediction-source", "ser"], monkeypatch)
    assert plain["kwargs"] == ser["kwargs"] and "prediction_source" not in plain["kwargs"]
    f = _parquet(tmp_path / "a.parquet", ["g3"], [("u", "g0", [1.0])])
    pie = _loco_kwargs(tmp_path, ["--prediction-source", "pie", "--pie-parquet", str(f), "--pie-context", "u"],
                       monkeypatch)
    assert pie["sources"] == [] and pie["kwargs"]["prediction_source"] == "pie"
    assert pie["kwargs"]["pie_context"] == "u"
    from sidechain.eval import loco
    for bad in (src + ["--prediction-source", "pie", "--pie-parquet", str(f), "--pie-context", "u"],
                ["--prediction-source", "pie", "--pie-parquet", str(f)],
                src + ["--pie-context", "u"]):
        with pytest.raises(SystemExit) as exc:
            loco.main(["--real", str(tmp_path / "real.h5ad"), "--bundle", str(tmp_path / "bundle"),
                       "--out", str(tmp_path / "arm"), *bad])
        assert exc.value.code == 2


# ----------------------------------------------------------------------------------------- submit.build

def _pie_argv(ch, stem, extra):
    argv = _argv(ch, stem, extra)
    i = argv.index("--gwps-cache")
    return argv[:i] + argv[i + 2:]


def test_build_default_is_the_build_without_the_flag(challenge):  # noqa: F811
    assert build.main(_argv(challenge, "plain", ["--bulk-anchor", "pooled"])) == 0
    assert build.main(_argv(challenge, "ser", ["--bulk-anchor", "pooled", "--prediction-source", "ser"])) == 0
    o = challenge["out"]
    a, b = ad.read_h5ad(o / "plain.h5ad"), ad.read_h5ad(o / "ser.h5ad")
    assert (a.X != b.X).nnz == 0 and a.obs.equals(b.obs)
    ra, rb = (json.loads((o / f"{s}.args.json").read_text()) for s in ("plain", "ser"))
    assert {k: v for k, v in ra.items() if k != "out"} == {k: v for k, v in rb.items() if k != "out"}
    assert not any(k.startswith("pie") or k == "prediction_source" for k in ra)
    assert (o / "plain.dual.json").read_text() == (o / "ser.dual.json").read_text()
    assert not (o / "plain.pie.json").exists()


def test_build_pie_carries_pies_map_and_the_generic_shift_for_a_missing_row(challenge, monkeypatch, tmp_path):  # noqa: F811
    seen = []

    def spy(self, n, log2fc_cell, log2fc_bulk, **kw):
        seen.append((np.array(log2fc_cell), np.array(log2fc_bulk)))
        return self.emit(n, log2fc_cell)
    monkeypatch.setattr(PoissonEmitter, "emit_dual", spy)
    f = _parquet(tmp_path / "p.parquet", ["D", "C", "B", "Z"], [
        ("X", "A", [0.4, -0.3, 8.0, 1.0]), ("unnamed", "A", [1.0, 1.0, 1.0, 1.0]),
        ("unnamed", "B", [1.0, 1.0, 1.0, 1.0])])
    assert build.main(_pie_argv(challenge, "pie", ["--bulk-anchor", "pooled", "--prediction-source", "pie",
                                                   "--pie-parquet", str(f)])) == 0
    (ca, ba), (cb, bb) = seen
    # A: genes A B C D <- (own, pinned) 5.0 (clipped) -0.3 0.4, at alpha 1.35 / alpha-bulk 1.5
    assert ca[0] == ba[0] == build.TARGET_SELF_LOG2FC
    assert np.allclose(ca[1:], np.array([5.0, -0.3, 0.4]) * 1.35)
    assert np.allclose(ba[1:], np.array([5.0, -0.3, 0.4]) * 1.5)
    # B: no row under context X -> the generic (H1 mean) shift, as a target no source covers
    h1 = PseudobulkSums.load(challenge["data"] / "h1.npz")
    generic = build.h1_mean_shift(h1, "non-targeting", np.array(["A", "B", "C", "D"]))
    assert cb[1] == build.TARGET_SELF_LOG2FC
    assert np.allclose(np.delete(cb, 1), np.delete(generic, 1) * 1.35)
    rec = json.loads((challenge["out"] / "pie.pie.json").read_text())
    assert rec["pie_context"] == {"X": "X"} and rec["maps"]["X"]["perturbations_missing"] == 1
    args = json.loads((challenge["out"] / "pie.args.json").read_text())
    assert args["prediction_source"] == "pie" and args["pie_parquet"] == str(f)
    seen.clear()
    assert build.main(_pie_argv(challenge, "pie_u", ["--bulk-anchor", "pooled", "--prediction-source", "pie",
                                                     "--pie-parquet", str(f), "--pie-context", "unnamed"])) == 0
    assert np.allclose(seen[1][0][[0, 2, 3]], [0.0, 1.35, 1.35])      # B's row read (A unscored), own gene pinned
    assert seen[1][0][1] == build.TARGET_SELF_LOG2FC


def test_build_pie_refuses_a_pool_and_a_model_name(challenge, tmp_path):  # noqa: F811
    f = _parquet(tmp_path / "p.parquet", ["A"], [("X", "A", [1.0])])
    pie = ["--prediction-source", "pie", "--pie-parquet", str(f)]
    for argv in (_argv(challenge, "pool", pie),                               # --gwps-cache is a pool
                 _pie_argv(challenge, "ser-99_delta_test_v1", pie),
                 _pie_argv(challenge, "nopar", ["--prediction-source", "pie"]),
                 _pie_argv(challenge, "noflag", ["--pie-parquet", str(f)])):
        with pytest.raises(SystemExit) as exc:
            build.main(argv)
        assert exc.value.code == 2


def test_namespace_check_off_is_silent():
    ap = SimpleNamespace(error=lambda m: (_ for _ in ()).throw(SystemExit(m)))
    ps.check_prediction_source_args(ap, SimpleNamespace(prediction_source="ser", pie_parquet=None, pie_context=[]),
                                    per_context=True)
