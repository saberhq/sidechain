"""Contract tests for ``sidechain.data.stream_de_h5ad`` -- a remote DE-stats AnnData.

GWCD4i is the second table-type source and the first with a real standard error.
A table's variance BYPASSES the pool's Poisson floor, so the pinned things are the
ones that decide how loud it is:

* **The combined variance is exact for the stated error correlation.** rho = 0 is
  independent inverse-variance weighting, var = 1/sum(1/v); rho = 1 is no
  reduction at all. Three conditions sharing donors and guides are NOT independent,
  and the independent formula would report up to 3x too little variance -- the
  Feng failure (9-17x over-confident) by another door.
* **A row whose knockdown did not take ABSTAINS; it never votes zero.** 37.6 % of
  GWCD4i's rows have `ontarget_significant = False`.
* **Rows land where they belong.** A (condition, target) placed in the wrong slot is
  a silent misjoin; two rows for one slot is refused.

The fixture writes an h5ad by hand in AnnData's on-disk encoding (categorical
groups, bool datasets, contiguous layers) so the reader is exercised on the same
shapes the real object has, with no network.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from sidechain.data.stream_de_h5ad import (
    GATES,
    DEConditions,
    _runs,
    combine,
    read_de_h5ad,
)

GENES = ["G0", "G1", "G2", "G3", "G4"]


def _write_h5ad(path, rows, *, layers):
    """rows: list of dicts with target, cond, and the flag/num columns."""
    str_dt = h5py.string_dtype()
    with h5py.File(path, "w") as h:
        obs = h.create_group("obs")

        def categorical(name, values):
            cats = sorted(set(values))
            g = obs.create_group(name)
            g.attrs["encoding-type"] = "categorical"
            g.create_dataset("categories", data=np.array(cats, dtype=object), dtype=str_dt)
            g.create_dataset("codes", data=np.array([cats.index(v) for v in values], dtype=np.int8))

        categorical("target_contrast_gene_name", [r["target"] for r in rows])
        categorical("culture_condition", [r["cond"] for r in rows])
        for col in ("ontarget_significant", "distal_offtarget_flag", "low_target_gex",
                    "single_guide_estimate"):
            obs.create_dataset(col, data=np.array([r.get(col, False) for r in rows], dtype=bool))
        for col in ("n_cells_target", "n_guides", "ontarget_effect_size"):
            obs.create_dataset(col, data=np.array([r.get(col, 1.0) for r in rows], dtype=float))
        var = h.create_group("var")
        var.create_dataset("gene_name", data=np.array(GENES, dtype=object), dtype=str_dt)
        lay = h.create_group("layers")
        for name, mat in layers.items():
            lay.create_dataset(name, data=np.asarray(mat, dtype=np.float64))


def _fixture(tmp_path, *, kd_ok=True):
    rows = [
        {"target": "A", "cond": "Rest", "ontarget_significant": kd_ok},
        {"target": "A", "cond": "Stim8hr", "ontarget_significant": True},
        {"target": "B", "cond": "Rest", "ontarget_significant": True},
        {"target": "Z", "cond": "Rest", "ontarget_significant": True},   # not kept
        {"target": "B", "cond": "Stim8hr", "ontarget_significant": True},
    ]
    n = len(rows)
    lfc = np.arange(n * len(GENES), dtype=float).reshape(n, len(GENES)) / 10.0
    se = np.full((n, len(GENES)), 0.2)
    se[1] = 0.4                                          # A|Stim8hr noisier
    padj = np.full((n, len(GENES)), 0.5)
    path = tmp_path / "de.h5ad"
    _write_h5ad(path, rows, layers={"log_fc": lfc, "lfcSE": se, "adj_p_value": padj})
    return path, lfc, se


# ------------------------------------------------------------------ reading

def test_rows_land_in_their_condition_and_target_slot(tmp_path):
    path, lfc, se = _fixture(tmp_path)
    with h5py.File(path, "r") as h:
        de = read_de_h5ad(h, keep={"A", "B"})
    assert de.targets == ["A", "B"]
    assert de.conditions == ["Rest", "Stim8hr"]
    r, s = de.conditions.index("Rest"), de.conditions.index("Stim8hr")
    a, b = de.targets.index("A"), de.targets.index("B")
    np.testing.assert_allclose(de.lfc[r, a], lfc[0], rtol=1e-6)     # file row 0
    np.testing.assert_allclose(de.lfc[s, a], lfc[1], rtol=1e-6)     # file row 1
    np.testing.assert_allclose(de.lfc[r, b], lfc[2], rtol=1e-6)     # file row 2
    np.testing.assert_allclose(de.lfc[s, b], lfc[4], rtol=1e-6)     # file row 4, after Z
    np.testing.assert_allclose(de.se[s, a], 0.4, rtol=1e-6)
    assert de.present.all()


def test_only_kept_targets_are_read(tmp_path):
    path, _, _ = _fixture(tmp_path)
    with h5py.File(path, "r") as h:
        de = read_de_h5ad(h, keep={"A", "NOT_THERE"})
    assert de.targets == ["A"]
    assert de.notes["keep_absent"] == ["NOT_THERE"]


def test_flags_and_numbers_come_through_per_row(tmp_path):
    path, _, _ = _fixture(tmp_path, kd_ok=False)
    with h5py.File(path, "r") as h:
        de = read_de_h5ad(h, keep={"A", "B"})
    r, a = de.conditions.index("Rest"), de.targets.index("A")
    assert not de.flags["ontarget_significant"][r, a]
    assert de.flags["ontarget_significant"][r, de.targets.index("B")]


def test_two_rows_for_one_slot_is_refused(tmp_path):
    rows = [{"target": "A", "cond": "Rest"}, {"target": "A", "cond": "Rest"}]
    m = np.zeros((2, len(GENES)))
    path = tmp_path / "dup.h5ad"
    _write_h5ad(path, rows, layers={"log_fc": m, "lfcSE": m + 0.1, "adj_p_value": m + 0.5})
    with h5py.File(path, "r") as h, pytest.raises(ValueError, match="two rows"):
        read_de_h5ad(h, keep={"A"})


def test_runs_coalesce_contiguous_rows():
    assert _runs(np.array([0, 1, 2, 5, 6, 9])) == [(0, 3), (5, 7), (9, 10)]
    assert _runs(np.array([], dtype=int)) == []


def test_save_load_round_trip(tmp_path):
    path, _, _ = _fixture(tmp_path)
    with h5py.File(path, "r") as h:
        de = read_de_h5ad(h, keep={"A", "B"})
    de.save(tmp_path / "c.npz")
    back = DEConditions.load(tmp_path / "c.npz")
    assert back.targets == de.targets and back.conditions == de.conditions
    np.testing.assert_array_equal(back.lfc, de.lfc)
    np.testing.assert_array_equal(back.flags["ontarget_significant"],
                                  de.flags["ontarget_significant"])
    assert back.notes == de.notes


# ------------------------------------------------------------------ combining

def _de(values, ses, *, kd=None):
    """A DEConditions with one target over G genes, one row per condition."""
    C, G = len(values), len(values[0])
    kd = [True] * C if kd is None else kd
    return DEConditions(
        targets=["A"], conditions=[f"c{i}" for i in range(C)], genes=np.array(GENES[:G]),
        lfc=np.asarray(values, dtype=np.float32)[:, None, :],
        se=np.asarray(ses, dtype=np.float32)[:, None, :],
        padj=np.full((C, 1, G), 0.5, dtype=np.float32),
        present=np.ones((C, 1), dtype=bool),
        flags={"ontarget_significant": np.array(kd)[:, None],
               "distal_offtarget_flag": np.zeros((C, 1), bool),
               "low_target_gex": np.zeros((C, 1), bool)},
        nums={},
    )


def test_a_single_condition_passes_straight_through():
    de = _de([[1.0, -2.0]], [[0.1, 0.3]])
    tab = combine(de, conditions=["c0"], rho=0.7)       # rho is moot for one condition
    np.testing.assert_allclose(tab.lfc[0], [1.0, -2.0], rtol=1e-6)
    np.testing.assert_allclose(tab.var[0], [0.01, 0.09], rtol=1e-5)


def test_rho_zero_is_independent_inverse_variance():
    de = _de([[1.0], [3.0]], [[0.1], [0.2]])            # v = 0.01, 0.04
    tab = combine(de, rho=0.0)
    w1 = (1 / 0.01) / (1 / 0.01 + 1 / 0.04)
    assert tab.lfc[0, 0] == pytest.approx(w1 * 1.0 + (1 - w1) * 3.0, rel=1e-5)
    assert tab.var[0, 0] == pytest.approx(1.0 / (1 / 0.01 + 1 / 0.04), rel=1e-5)


def test_rho_one_gives_no_reduction_for_equal_variances():
    """Fully correlated errors: averaging three copies of one error learns nothing."""
    de = _de([[1.0], [1.2], [0.8]], [[0.2], [0.2], [0.2]])
    assert combine(de, rho=1.0).var[0, 0] == pytest.approx(0.04, rel=1e-5)
    assert combine(de, rho=0.0).var[0, 0] == pytest.approx(0.04 / 3, rel=1e-5)


def test_the_independent_formula_is_the_overconfident_one():
    """The whole reason rho exists: rho=0 is always the SMALLEST variance."""
    de = _de([[1.0], [2.0], [0.5]], [[0.1], [0.3], [0.2]])
    vs = [combine(de, rho=r).var[0, 0] for r in (0.0, 0.3, 0.7, 1.0)]
    assert vs == sorted(vs) and vs[0] < vs[-1]


def test_rho_one_matches_its_closed_form():
    de = _de([[1.0], [2.0]], [[0.1], [0.3]])
    v = np.array([0.01, 0.09])
    w = (1 / v) / (1 / v).sum()
    assert combine(de, rho=1.0).var[0, 0] == pytest.approx((w * np.sqrt(v)).sum() ** 2,
                                                          rel=1e-5)


def test_a_failed_knockdown_row_abstains_under_the_knockdown_gate():
    de = _de([[5.0], [1.0]], [[0.1], [0.1]], kd=[False, True])
    tab = combine(de, rho=0.0, gate="knockdown")
    assert tab.lfc[0, 0] == pytest.approx(1.0, rel=1e-6)   # only the good row votes
    assert tab.var[0, 0] == pytest.approx(0.01, rel=1e-5)
    both = combine(de, rho=0.0, gate="none")                # and without the gate, both do
    assert both.lfc[0, 0] == pytest.approx(3.0, rel=1e-6)


def test_no_usable_row_means_abstain_not_zero():
    de = _de([[5.0], [4.0]], [[0.1], [0.1]], kd=[False, False])
    tab = combine(de, rho=0.0, gate="knockdown")
    assert np.isinf(tab.var[0, 0])                          # weight 0 in the pool
    assert tab.lfc[0, 0] == 0.0
    assert tab.notes["targets_abstaining_entirely"] == 1


def test_a_nan_gene_in_one_condition_is_skipped_for_that_gene_only():
    de = _de([[1.0, np.nan], [3.0, 2.0]], [[0.1, 0.1], [0.1, 0.1]])
    tab = combine(de, rho=0.0)
    assert tab.lfc[0, 0] == pytest.approx(2.0, rel=1e-6)   # both vote on G0
    assert tab.lfc[0, 1] == pytest.approx(2.0, rel=1e-6)   # only c1 votes on G1
    assert tab.var[0, 1] == pytest.approx(0.01, rel=1e-5)


@pytest.mark.parametrize("kw,exc", [({"rho": 1.5}, ValueError),
                                    ({"conditions": ["nope"]}, KeyError),
                                    ({"gate": "nope"}, KeyError)])
def test_bad_arguments_are_refused(kw, exc):
    with pytest.raises(exc):
        combine(_de([[1.0]], [[0.1]]), **kw)


def test_the_view_is_a_delta_source_the_pool_accepts():
    """It must enter `pooled_delta` exactly as Feng does -- via duck typing."""
    from sidechain.submit.build import as_delta_source
    tab = combine(_de([[1.0, -1.0]], [[0.1, 0.1]]))
    src = as_delta_source(tab)
    fc, var = src.effect("A")
    np.testing.assert_allclose(fc, [1.0, -1.0], rtol=1e-6)
    assert src.effect("NOT_A_TARGET") is None


def test_every_gate_names_real_flags():
    known = {"ontarget_significant", "distal_offtarget_flag", "low_target_gex",
             "single_guide_estimate"}
    for name, checks in GATES.items():
        for col, _ in checks:
            assert col in known, f"gate {name!r} names unknown flag {col!r}"



# --------------------------------------------- hardening after review (2026-09-26)

def test_combining_several_conditions_without_rho_is_refused():
    """rho = 0 is the over-confident assumption here; nobody inherits it by accident."""
    de = _de([[1.0], [2.0]], [[0.1], [0.1]])
    with pytest.raises(ValueError, match="explicit rho"):
        combine(de)
    combine(de, conditions=["c0"])                  # one condition needs no rho


def test_string_flags_are_mapped_or_refused_never_all_true():
    from sidechain.data.stream_de_h5ad import _as_bool
    np.testing.assert_array_equal(_as_bool(np.array(["True", "False"], dtype=object), "f"),
                                  [True, False])
    with pytest.raises(ValueError, match="not boolean"):
        _as_bool(np.array(["yes", "no"], dtype=object), "f")
    with pytest.raises(ValueError, match="not boolean"):
        _as_bool(np.array([0.0, 1.0]), "f")


def test_a_missing_gene_name_in_a_var_categorical_is_refused(tmp_path):
    from sidechain.data.stream_de_h5ad import _obs_col_var
    path = tmp_path / "v.h5"
    with h5py.File(path, "w") as h:
        g = h.create_group("var").create_group("gene_name")
        g.create_dataset("categories", data=np.array(["G0", "G1"], dtype=object),
                         dtype=h5py.string_dtype())
        g.create_dataset("codes", data=np.array([0, -1, 1], dtype=np.int8))
    with h5py.File(path, "r") as h, pytest.raises(ValueError, match="missing value"):
        _obs_col_var(h, "gene_name")


def test_the_registry_block_describes_what_the_reader_reads():
    """One source of truth: the gwcd4i_de block's spec must supply every reader column."""
    import yaml

    from sidechain.data.stream_de_h5ad import reader_kwargs
    from sidechain.utils.paths import resolve_config
    cfg = yaml.safe_load(resolve_config("configs/datasets.yaml").read_text())
    block = next(b for b in cfg["datasets"] if b["name"] == "gwcd4i_de")
    spec = next(f for f in block["files"] if f["kind"] == "lfc_table")["spec"]
    kw = reader_kwargs(spec)
    assert kw == {"target_col": "target_contrast_gene_name", "condition_col": "culture_condition",
                  "gene_col": "gene_name", "effect_layer": "log_fc", "se_layer": "lfcSE",
                  "padj_layer": "adj_p_value"}


def test_a_spec_missing_a_column_is_refused_not_defaulted():
    from sidechain.data.stream_de_h5ad import reader_kwargs
    with pytest.raises(KeyError, match="se_col"):
        reader_kwargs({"target_col": "t", "condition_col": "c", "gene_symbol_col": "g",
                       "effect_col": "e", "pvalue_col": "p"})


def _registry(tmp_path, *, with_provenance=True, size=1000, etag="abc-2"):
    import json

    import yaml
    cfg = {"datasets": [{"name": "toy", "host": "s3", "record": "b/p/", "license": "MIT",
                         "dest": "external/s3-b-p", "derived": "derived/toy",
                         "files": [{"name": "de.h5ad", "kind": "lfc_table", "spec": {
                             "target_col": "t", "condition_col": "c", "gene_symbol_col": "g",
                             "effect_col": "e", "se_col": "s", "pvalue_col": "p"}}]}]}
    cfg_path = tmp_path / "datasets.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    if with_provenance:
        d = tmp_path / "external/s3-b-p"
        d.mkdir(parents=True)
        (d / "PROVENANCE.json").write_text(json.dumps({"selected": [{
            "name": "de.h5ad", "size_bytes": size, "checksum": f"s3-etag:{etag}",
            "url": "https://b.s3.amazonaws.com/p/de.h5ad"}]}))
    return cfg_path


def test_a_pull_without_the_gate_having_run_is_refused(tmp_path):
    from sidechain.data.stream_de_h5ad import resolve_pull
    cfg = _registry(tmp_path, with_provenance=False)
    with pytest.raises(SystemExit, match="no PROVENANCE.json"):
        resolve_pull("toy", str(cfg), tmp_path, head=lambda u: {"size_bytes": 1000, "etag": "abc-2"})


def test_a_file_that_changed_upstream_since_the_gate_is_refused(tmp_path):
    from sidechain.data.stream_de_h5ad import resolve_pull
    cfg = _registry(tmp_path)
    with pytest.raises(SystemExit, match="changed upstream"):
        resolve_pull("toy", str(cfg), tmp_path, head=lambda u: {"size_bytes": 1001, "etag": "abc-2"})
    with pytest.raises(SystemExit, match="changed upstream"):
        resolve_pull("toy", str(cfg), tmp_path, head=lambda u: {"size_bytes": 1000, "etag": "zzz-9"})


def test_an_unchanged_file_resolves_to_the_url_the_gate_admitted(tmp_path):
    from sidechain.data.stream_de_h5ad import resolve_pull
    cfg = _registry(tmp_path)
    r = resolve_pull("toy", str(cfg), tmp_path, head=lambda u: {"size_bytes": 1000, "etag": "abc-2"})
    assert r["selected"]["url"] == "https://b.s3.amazonaws.com/p/de.h5ad"
    assert r["reader_kwargs"]["se_layer"] == "s"


# --------------------------------------- math-review findings (2026-09-26)

def test_a_bool_flag_stored_as_strings_is_read_correctly_end_to_end(tmp_path):
    """A bool column round-tripped through object dtype arrives as a "True"/"False"
    categorical; astype(bool) made every one True and inverted the gate."""
    rows = [{"target": "A", "cond": "Rest"}]
    path = tmp_path / "s.h5ad"
    str_dt = h5py.string_dtype()
    with h5py.File(path, "w") as h:
        obs = h.create_group("obs")
        for name, vals in (("target_contrast_gene_name", ["A"]), ("culture_condition", ["Rest"]),
                           ("ontarget_significant", ["True"]), ("distal_offtarget_flag", ["False"]),
                           ("low_target_gex", ["False"]), ("single_guide_estimate", ["False"])):
            g = obs.create_group(name)
            g.attrs["encoding-type"] = "categorical"
            g.create_dataset("categories", data=np.array(sorted(set(vals)), dtype=object), dtype=str_dt)
            g.create_dataset("codes", data=np.zeros(1, dtype=np.int8))
        h.create_group("var").create_dataset("gene_name", data=np.array(GENES, dtype=object),
                                             dtype=str_dt)
        lay = h.create_group("layers")
        for nm, val in (("log_fc", 1.0), ("lfcSE", 0.2), ("adj_p_value", 0.5)):
            lay.create_dataset(nm, data=np.full((1, len(GENES)), val))
    with h5py.File(path, "r") as h:
        de = read_de_h5ad(h, keep={"A"})
    assert not de.flags["distal_offtarget_flag"][0, 0]
    assert int(combine(de, gate="knockdown").n_usable[0]) == len(GENES)
    assert rows  # (fixture parity)


def test_duplicate_rows_are_caught_before_any_layer_is_read(tmp_path):
    """The file has NO layers at all: the refusal must come from the row check, proving it
    runs before the reads that are the whole cost on the remote object."""
    str_dt = h5py.string_dtype()
    path = tmp_path / "dup_nolayers.h5ad"
    with h5py.File(path, "w") as h:
        obs = h.create_group("obs")
        for name, vals in (("target_contrast_gene_name", ["A", "A"]),
                           ("culture_condition", ["Rest", "Rest"])):
            g = obs.create_group(name)
            g.attrs["encoding-type"] = "categorical"
            g.create_dataset("categories", data=np.array(sorted(set(vals)), dtype=object), dtype=str_dt)
            g.create_dataset("codes", data=np.zeros(2, dtype=np.int8))
        h.create_group("var").create_dataset("gene_name", data=np.array(GENES, dtype=object),
                                             dtype=str_dt)
    with h5py.File(path, "r") as h, pytest.raises(ValueError, match="two rows"):
        read_de_h5ad(h, keep={"A"})


def test_duplicate_gene_names_are_refused(tmp_path):
    rows = [{"target": "A", "cond": "Rest"}]
    m = np.zeros((1, len(GENES)))
    path = tmp_path / "dupgene.h5ad"
    _write_h5ad(path, rows, layers={"log_fc": m, "lfcSE": m + 0.1, "adj_p_value": m + 0.5})
    with h5py.File(path, "r+") as h:
        del h["var/gene_name"]
        h["var"].create_dataset("gene_name", data=np.array(["G0", "G1", "G1", "G3", "G4"],
                                                           dtype=object), dtype=h5py.string_dtype())
    with h5py.File(path, "r") as h, pytest.raises(ValueError, match="duplicate gene names"):
        read_de_h5ad(h, keep={"A"})
    from sidechain.data.lfc_table import LfcTable
    tab = LfcTable(labels=["A"], genes=np.array(["G0", "G0"]), lfc=np.zeros((1, 2)),
                   var=np.ones((1, 2)))
    with pytest.raises(ValueError, match="duplicate gene names"):
        tab.subset(["A"], ["G0"])


# ---- controls: what IS right, pinned so the suite records it (math review, 2026-09-26)

def test_the_combined_variance_is_w_sigma_w_for_any_number_of_conditions():
    """Brute force: w' Sigma w with Sigma_cd = rho*sqrt(v_c v_d), for 2-5 conditions."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        C = int(rng.integers(2, 6))
        v = rng.uniform(1e-3, 0.5, C)
        rho = float(rng.uniform(0, 1))
        w = (1 / v) / (1 / v).sum()
        sigma = rho * np.sqrt(np.outer(v, v))
        np.fill_diagonal(sigma, v)
        de = _de([[0.0]] * C, [[float(np.sqrt(x))] for x in v])
        got = combine(de, rho=rho).var[0, 0]
        assert got == pytest.approx(float(w @ sigma @ w), rel=1e-5)   # float32 storage


def test_rows_land_in_their_slot_when_kept_rows_are_scattered_and_interleaved(tmp_path):
    """block[k] is ds[a+k]: three scattered runs, unkept rows between, targets interleaved."""
    order = [("TP53", "Rest"), ("DROP1", "Rest"), ("MYC", "Stim8hr"), ("TP53", "Stim8hr"),
             ("DROP2", "Stim8hr"), ("DROP3", "Rest"), ("MYC", "Rest"), ("TP53", "Stim48hr")]
    rows = [{"target": t, "cond": c, "ontarget_significant": True} for t, c in order]
    lfc = np.arange(len(rows) * len(GENES), dtype=float).reshape(len(rows), len(GENES)) / 10.0
    path = tmp_path / "scatter.h5ad"
    _write_h5ad(path, rows, layers={"log_fc": lfc, "lfcSE": lfc * 0 + 0.2,
                                    "adj_p_value": lfc * 0 + 0.5})
    with h5py.File(path, "r") as h:
        de = read_de_h5ad(h, keep={"TP53", "MYC"})
    assert de.notes["range_runs"] == 3 and int(de.present.sum()) == 5
    for r, (t, c) in enumerate(order):
        if t in ("TP53", "MYC"):
            np.testing.assert_allclose(de.lfc[de.conditions.index(c), de.targets.index(t)],
                                       lfc[r], rtol=1e-6)
    assert np.isnan(de.lfc[~de.present]).all()
