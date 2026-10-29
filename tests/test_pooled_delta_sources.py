"""`pooled_delta` over pseudobulk and fold-change-table sources, its knobs, and their `eval.loco` flags.

    .venv/bin/python -m pytest -q tests/test_pooled_delta_sources.py

Writes temporary files only. Notes: private/code-notes/tests/test_pooled_delta_sources.py.md
"""
from __future__ import annotations

import numpy as np
import pytest

from sidechain.data.lfc_table import LfcTable
from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.submit.build import as_delta_source, pooled_delta


class _StubPB(str):
    """A stand-in for a loaded `PseudobulkSums`: a `str` that takes attributes."""

AXIS = np.array(["A", "B", "C"])


def _pb(labels, cpm, n_cells=1000, genes=("A", "B", "C")):
    """A PseudobulkSums with the given mean CPM per label and a fixed spread."""
    m = np.asarray(cpm, dtype=float)
    n = np.full(len(labels), n_cells, dtype=np.int64)
    cpm_sum = m * n[:, None]
    # var_cpm = cpm_sq/n - mean^2; pick cpm_sq so the per-gene variance is 1.0
    cpm_sq_sum = (m**2 + 1.0) * n[:, None]
    return PseudobulkSums(
        labels=list(labels), genes=np.array(genes),
        count_sum=cpm_sum.copy(), cpm_sum=cpm_sum, cpm_sq_sum=cpm_sq_sum,
        n_cells=n, libsize_sum=n.astype(float) * 1e4, sources=["test"],
    )


def _lfc(labels, lfc, var, genes=("A", "B", "C")):
    return LfcTable(labels=list(labels), genes=np.array(genes),
                    lfc=np.asarray(lfc, float), var=np.asarray(var, float),
                    source="test-lfc")


# --------------------------------------------------- nothing existing changes --


def test_the_historical_tuple_form_still_works():
    """The `(PseudobulkSums, control)` pair pools as before."""
    pb = _pb(["control", "TP53"], [[100.0, 100.0, 100.0], [200.0, 100.0, 50.0]])
    out = pooled_delta("TP53", [(pb, "control")], AXIS, shrinkage=False)
    assert out is not None
    assert out[0] > 0        # 100 -> 200 CPM
    assert abs(out[1]) < 1e-9
    assert out[2] < 0        # 100 -> 50 CPM


def test_a_target_no_source_covers_is_still_none():
    """A target no source covers returns None."""
    pb = _pb(["control", "TP53"], [[100.0, 100.0, 100.0], [200.0, 100.0, 50.0]])
    assert pooled_delta("BRCA1", [(pb, "control")], AXIS) is None


def test_as_delta_source_rejects_something_that_is_neither():
    with pytest.raises(TypeError, match="not a delta source"):
        as_delta_source(object())


# --------------------------------------------------------- the new source type --


def test_an_lfc_table_pools_alongside_a_pseudobulk():
    """An LfcTable and a pseudobulk pool in one call."""
    pb = _pb(["control", "TP53"], [[100.0, 100.0, 100.0], [200.0, 100.0, 100.0]])
    tab = _lfc(["TP53"], [[2.0, 0.0, 0.0]], [[0.01, np.inf, np.inf]])

    both = pooled_delta("TP53", [(pb, "control"), tab], AXIS, shrinkage=False)
    only_pb = pooled_delta("TP53", [(pb, "control")], AXIS, shrinkage=False)
    assert both is not None and only_pb is not None
    # gene A: the table is far more certain (var 0.01) so it pulls the pool up
    assert both[0] > only_pb[0]
    # genes B and C: the table abstains, so the pool is unchanged
    assert np.allclose(both[1:], only_pb[1:])


def test_an_abstaining_gene_contributes_no_weight_at_all():
    """A gene with `var = inf` carries no weight in the pool."""
    pb = _pb(["control", "TP53"], [[100.0, 100.0, 100.0], [400.0, 400.0, 400.0]])
    abstains = _lfc(["TP53"], [[0.0, 0.0, 0.0]], [[np.inf, np.inf, np.inf]])

    with_it = pooled_delta("TP53", [(pb, "control"), abstains], AXIS, shrinkage=False)
    without = pooled_delta("TP53", [(pb, "control")], AXIS, shrinkage=False)
    assert np.allclose(with_it, without)


def test_a_target_only_a_fully_abstaining_source_covers_returns_zeros_not_none():
    """A target covered only by a source abstaining on every gene returns zeros, not None."""
    tab = _lfc(["TP53"], [[0.1, 0.2, 0.3]], [[np.inf, np.inf, np.inf]])
    out = pooled_delta("TP53", [tab], AXIS)
    assert out is not None
    assert np.allclose(out, 0.0)


def test_pooling_is_inverse_variance_across_two_lfc_tables():
    """Two tables, same gene, different certainty -> the certain one dominates,
    and the pooled value is the exact inverse-variance mean."""
    a = _lfc(["TP53"], [[1.0, 0.0, 0.0]], [[1.0, np.inf, np.inf]])
    b = _lfc(["TP53"], [[3.0, 0.0, 0.0]], [[1.0 / 3.0, np.inf, np.inf]])
    out = pooled_delta("TP53", [a, b], AXIS, shrinkage=False)
    expected = (1.0 * 1.0 + 3.0 * 3.0) / (1.0 + 3.0)
    assert np.isclose(out[0], expected)


def test_a_source_on_a_different_gene_axis_is_remapped_not_misaligned():
    """A source on another gene axis is matched by gene name, not by position."""
    tab = LfcTable(labels=["TP53"], genes=np.array(["C", "A"]),
                   lfc=np.array([[5.0, 1.0]]), var=np.array([[1.0, 1.0]]),
                   source="reordered")
    out = pooled_delta("TP53", [tab], AXIS, shrinkage=False)
    assert np.isclose(out[0], 1.0)     # A
    assert np.isclose(out[1], 0.0)     # B, absent from the source
    assert np.isclose(out[2], 5.0)     # C


def test_shrinkage_on_an_abstaining_gene_does_not_produce_nan():
    """`shrink` divides by fc^2 and subtracts var/fc^2; with var = inf that is
    -inf before the clip. It must clip to zero, not propagate a nan into the
    submission."""
    tab = _lfc(["TP53"], [[1.0, 2.0, 0.0]], [[np.inf, 0.01, np.inf]])
    out = pooled_delta("TP53", [tab], AXIS, shrinkage=True)
    assert np.isfinite(out).all()
    assert np.isclose(out[0], 0.0)
    assert out[1] > 0


# ----------------------------------------------------- the variance floor --


def _pb_var(labels, cpm, var, n_cells=150, lib_per_cell=20000.0, genes=("A", "B", "C")):
    """A PseudobulkSums with chosen per-gene variance; `n_cells` scalar or per label; `lib_per_cell` mean depth."""
    m = np.asarray(cpm, dtype=float)
    v = np.asarray(var, dtype=float)
    n = np.asarray(n_cells if np.ndim(n_cells) else [n_cells] * len(labels), dtype=np.int64)
    return PseudobulkSums(
        labels=list(labels), genes=np.array(genes),
        count_sum=m * n[:, None], cpm_sum=m * n[:, None],
        cpm_sq_sum=(m**2 + v) * n[:, None],
        n_cells=n, libsize_sum=n.astype(float) * lib_per_cell,
    )


def test_without_the_floor_a_zero_spread_gene_pins_the_pool():
    """By default a zero-spread gene sits at the 1e-6 clamp and erases another source's +2 effect."""
    flat = _pb_var(["control", "TP53"], [[100.0] * 3, [100.0] * 3], [[0.0] * 3, [0.0] * 3])
    # Poisson-realistic spread at 20k UMI: var_cpm = mean * 1e6/libsize = mean * 50
    real = _pb_var(["control", "TP53"], [[100.0] * 3, [400.0] * 3],
                   [[5000.0] * 3, [20000.0] * 3])
    stats: dict = {}
    out = pooled_delta("TP53", [(flat, "control"), (real, "control")], AXIS,
                       shrinkage=False, stats=stats)
    assert abs(out[0]) < 0.05                     # the +2 effect all but erased
    assert stats["gene_weights_var_le_1e-6"] == 3  # every zero-spread gene sat at the clamp


def test_the_poisson_floor_unpins_zero_spread_genes():
    """Same two sources with `var_floor='poisson'`: the flat source keeps a vote
    sized by its sampling variance instead of a 1e6 veto, so the pooled value
    lands between the sources -- and nothing sits at the old clamp any more."""
    flat = _pb_var(["control", "TP53"], [[100.0] * 3, [100.0] * 3], [[0.0] * 3, [0.0] * 3])
    real = _pb_var(["control", "TP53"], [[100.0] * 3, [400.0] * 3],
                   [[5000.0] * 3, [20000.0] * 3])
    stats: dict = {}
    out = pooled_delta("TP53", [(flat, "control"), (real, "control")], AXIS,
                       shrinkage=False, var_floor="poisson", stats=stats)
    fc_real = pooled_delta("TP53", [(real, "control")], AXIS, shrinkage=False,
                           var_floor="poisson")
    assert 0.5 < out[0] < fc_real[0]
    assert stats["gene_weights_var_le_1e-6"] == 0


def test_a_single_cell_arm_dominates_without_the_floor_and_abstains_with_it():
    """Without the floor a one-cell arm outvotes a 150-cell source; with it the arm abstains."""
    once = _pb_var(["control", "TP53"], [[100.0] * 3, [800.0] * 3], [[0.0] * 3, [0.0] * 3],
                   n_cells=[150, 1])
    real = _pb_var(["control", "TP53"], [[100.0] * 3, [200.0] * 3], [[1.0] * 3, [1.0] * 3])

    unfloored = pooled_delta("TP53", [(once, "control"), (real, "control")], AXIS, shrinkage=False)
    only_real = pooled_delta("TP53", [(real, "control")], AXIS, shrinkage=False)
    assert unfloored[0] > only_real[0] + 0.5      # the 1-cell arm dragged the pool its way

    floored = pooled_delta("TP53", [(once, "control"), (real, "control")], AXIS,
                           shrinkage=False, var_floor="poisson")
    floored_real = pooled_delta("TP53", [(real, "control")], AXIS, shrinkage=False,
                                var_floor="poisson")
    assert np.allclose(floored, floored_real)


def test_a_target_covered_only_by_a_single_cell_arm_reports_zero_weight_not_fallback():
    """A target covered only by a one-cell arm returns zeros under the floor, and the stats count it."""
    once = _pb_var(["control", "TP53"], [[100.0] * 3, [800.0] * 3], [[0.0] * 3, [0.0] * 3],
                   n_cells=[150, 1])
    stats: dict = {}
    out = pooled_delta("TP53", [(once, "control")], AXIS, shrinkage=False,
                       var_floor="poisson", stats=stats)
    assert out is not None and np.allclose(out, 0.0)
    assert stats["source_arms_abstained"] == 1
    assert stats["targets_zero_weight"] == 1


def test_var_floor_none_leaves_every_existing_number_untouched():
    """The knob defaults off, and off means bit-identical: the same pool with
    `var_floor='none'` passed explicitly equals the historical call."""
    pb = _pb(["control", "TP53"], [[100.0, 100.0, 100.0], [200.0, 100.0, 50.0]])
    a = pooled_delta("TP53", [(pb, "control")], AXIS, shrinkage=False)
    b = pooled_delta("TP53", [(pb, "control")], AXIS, shrinkage=False, var_floor="none")
    assert np.array_equal(a, b)


def test_an_unknown_var_floor_is_refused_not_ignored():
    import pytest

    pb = _pb(["control", "TP53"], [[100.0] * 3, [200.0] * 3])
    with pytest.raises(ValueError, match="var_floor"):
        pooled_delta("TP53", [(pb, "control")], AXIS, var_floor="possion")


# ------------------------------------------- per-source (depth-aware) shrinkage --


def _noisy_pb(genes=("A", "B", "C")):
    """A 100 -> 200 CPM effect whose estimate variance (~2.1 in log2^2) exceeds
    fc^2 (~0.99), so `shrink` zeroes it while the raw fold change is ~1."""
    k = len(genes)
    return _pb_var(["control", "TP53"], [[100.0] * k, [200.0] * k],
                   [[1.0] * k, [6e6] * k], genes=genes)


def test_a_shrink_flagged_source_is_shrunk_while_the_global_switch_is_off():
    """A source flagged True in the tuple is shrunk while the global switch is off."""
    noisy = _noisy_pb()
    raw = pooled_delta("TP53", [(noisy, "control")], AXIS, shrinkage=False)
    flagged = pooled_delta("TP53", [(noisy, "control", True)], AXIS, shrinkage=False)
    assert raw[0] > 0.9
    assert np.allclose(flagged, 0.0)


def test_a_shrink_false_source_stays_raw_while_the_global_switch_is_on():
    """The override runs both ways: False pins a source raw under a global
    shrink, it does not merely mean 'unset'."""
    noisy = _noisy_pb()
    raw = pooled_delta("TP53", [(noisy, "control")], AXIS, shrinkage=False)
    pinned = pooled_delta("TP53", [(noisy, "control", False)], AXIS, shrinkage=True)
    globally = pooled_delta("TP53", [(noisy, "control")], AXIS, shrinkage=True)
    assert np.allclose(globally, 0.0)
    assert np.array_equal(pinned, raw)


def test_the_two_tuple_form_inherits_the_global_flag_bit_for_bit():
    """A pair and a None third slot follow the global flag, bit for bit, in both positions."""
    noisy = _noisy_pb()
    for flag in (True, False):
        a = pooled_delta("TP53", [(noisy, "control")], AXIS, shrinkage=flag)
        b = pooled_delta("TP53", [(noisy, "control", None)], AXIS, shrinkage=flag)
        assert np.array_equal(a, b), flag


def test_only_the_flagged_source_is_shrunk_inside_a_mixed_pool():
    """The point of the knob: one pool, one arm shrunk, its neighbour raw.
    Disjoint gene axes make each source's contribution readable directly."""
    deep = _noisy_pb(genes=("A",))
    wide = _noisy_pb(genes=("C",))
    out = pooled_delta("TP53", [(deep, "control", True), (wide, "control")], AXIS,
                       shrinkage=False)
    assert np.isclose(out[0], 0.0)      # the flagged arm's noise-level effect: shrunk away
    assert out[2] > 0.9                 # the raw arm's identical effect: kept


def test_a_non_bool_shrink_slot_is_refused_not_coerced():
    """A non-bool third slot is refused."""
    import pytest

    pb = _pb(["control", "TP53"], [[100.0] * 3, [200.0] * 3])
    with pytest.raises(TypeError, match="shrink"):
        as_delta_source((pb, "control", "poisson"))


def test_sources_from_specs_binds_the_flag_and_the_default_control(monkeypatch):
    """`--shrink-source` specs parse to `(pb, control, True)`, `--source` to pairs; an empty suffix is `control`."""
    from sidechain.data.stream_pseudobulk import PseudobulkSums
    from sidechain.submit.build import sources_from_specs

    monkeypatch.setattr(PseudobulkSums, "load", classmethod(lambda cls, p: _StubPB(f"PB:{p}")))
    out = sources_from_specs(["a.npz:ctrlA", "b.npz:"], ["c.npz:deepctl"])
    assert out[0] == ("PB:a.npz", "ctrlA")
    assert out[1] == ("PB:b.npz", "control")      # empty suffix -> the default
    assert out[2] == ("PB:c.npz", "deepctl", True)


def test_loco_passes_the_triple_through_and_records_it(monkeypatch, tmp_path):
    """`loco` passes a `--shrink-source` arm as the True triple and records it in summary.json and log_run."""
    import json

    from sidechain.data.stream_pseudobulk import PseudobulkSums
    from sidechain.eval import loco

    monkeypatch.setattr(PseudobulkSums, "load", classmethod(lambda cls, p: _StubPB(f"PB:{p}")))
    captured, logged = {}, {}

    def fake_build(real, sources, out_path, **kw):
        captured["sources"] = sources
        return {}

    monkeypatch.setattr(loco, "build_transfer_prediction", fake_build)
    monkeypatch.setattr(loco, "attach_controls", lambda pred, real, out, **kw: out)
    monkeypatch.setattr(loco, "score", lambda *a, **kw: {"overall": 0.0, "members": {}})
    monkeypatch.setattr(loco, "log_run", lambda params, results, artifacts=None: logged.update(params))

    out = tmp_path / "arm"
    rc = loco.main(["--real", "r.h5ad", "--bundle", "b", "--out", str(out),
                    "--source", "plain.npz:ctl", "--shrink-source", "deep.npz:deepctl"])
    assert rc == 0
    assert captured["sources"][0] == ("PB:plain.npz", "ctl")
    assert captured["sources"][1] == ("PB:deep.npz", "deepctl", True)
    rec = json.loads((out / "summary.json").read_text())
    assert rec["sources"]["pseudobulk"] == ["plain.npz:ctl"]
    assert rec["sources"]["shrink_pseudobulk"] == ["deep.npz:deepctl"]
    assert logged["sources"] == ["plain.npz:ctl"]
    assert logged["shrink_sources"] == ["deep.npz:deepctl"]


def test_an_lfc_table_without_the_attribute_still_follows_the_global_flag():
    """`pooled_delta` reads the override with getattr; a source type that never
    grew a `shrink` attribute must keep its historical behaviour under both
    global settings."""
    tab = _lfc(["TP53"], [[2.0, 0.0, 0.0]], [[0.01, np.inf, np.inf]])
    on = pooled_delta("TP53", [tab], AXIS, shrinkage=True)
    off = pooled_delta("TP53", [tab], AXIS, shrinkage=False)
    assert np.isclose(off[0], 2.0)
    assert on[0] < off[0]               # var 0.01 against fc 2: shrunk a little, not zeroed


# ------------------------------------------------- the transfer exponent (gamma) --


def test_gamma_1_is_bit_identical_to_the_historical_call():
    """gamma 1 skips the transform: bit-identical to the call without it."""
    pb = _pb(["control", "TP53"], [[100.0, 100.0, 100.0], [200.0, 100.0, 50.0]])
    plain = pooled_delta("TP53", [(pb, "control")], AXIS, shrinkage=False)
    explicit = pooled_delta("TP53", [(pb, "control")], AXIS, shrinkage=False,
                            gamma=1.0, ctrl_tgt_cpm=np.array([10.0, 10.0, 10.0]))
    assert np.array_equal(plain, explicit)


def test_gamma_0_transfers_the_absolute_cpm_change():
    """gamma 0 transfers the source's absolute CPM change onto the target's own control level."""
    pb = _pb(["control", "TP53"], [[100.0, 100.0, 100.0], [200.0, 100.0, 50.0]])
    ctrl_tgt = np.array([10.0, 100.0, 400.0])
    out = pooled_delta("TP53", [(pb, "control")], AXIS, shrinkage=False,
                       gamma=0.0, ctrl_tgt_cpm=ctrl_tgt)
    assert np.isclose(out[0], np.log2(1 + 100.0 / 11.0))     # +100 CPM onto 10 CPM controls
    assert np.isclose(out[1], 0.0)                           # no change stays no change
    assert np.isclose(out[2], np.log2(1 - 50.0 / 401.0))     # -50 CPM onto 400 CPM controls


def test_a_negative_predicted_expression_clamps_to_the_floor_not_no_change():
    """A negative predicted expression clamps to a finite floor, counted in the stats."""
    from sidechain.submit.build import GAMMA_MULT_FLOOR

    pb = _pb(["control", "TP53"], [[100.0, 100.0, 100.0], [200.0, 100.0, 50.0]])
    stats: dict = {}
    out = pooled_delta("TP53", [(pb, "control")], AXIS, shrinkage=False,
                       gamma=0.0, ctrl_tgt_cpm=np.array([1.0, 1.0, 1.0]), stats=stats)
    # gene C: -50 CPM onto 1-CPM controls -> multiplier 1 - 50/2 = -24
    assert np.isfinite(out).all()
    assert np.isclose(out[2], np.log2(GAMMA_MULT_FLOOR))
    assert stats["gamma_mult_clamped"] == 1
    assert stats["gamma_genes_transformed"] == 3


def test_each_source_transforms_against_its_own_control():
    """ctrl_src is a per-source quantity. The mutant that reads one source's
    control profile for every source produces the wrong ratio on the second --
    disjoint gene axes make each source's contribution readable directly."""
    deep = _pb(["control", "TP53"], [[100.0], [200.0]], genes=("A",))
    wide = _pb(["control", "TP53"], [[400.0], [800.0]], genes=("C",))
    ctrl_tgt = np.array([50.0, 0.0, 50.0])
    out = pooled_delta("TP53", [(deep, "control"), (wide, "control")], AXIS,
                       shrinkage=False, gamma=0.5, ctrl_tgt_cpm=ctrl_tgt)
    assert np.isclose(out[0], np.log2(1 + (100.0 / 101.0) * (51.0 / 101.0) ** -0.5))
    assert np.isclose(out[2], np.log2(1 + (400.0 / 401.0) * (51.0 / 401.0) ** -0.5))


def test_genes_the_target_axis_lacks_do_not_poison_a_gamma_pool():
    """Genes off the target axis pass through gamma as identity, not nan, and are not counted as transformed."""
    pb = _pb(["control", "TP53"], [[100.0] * 4, [200.0] * 4], genes=("A", "B", "C", "D"))
    stats: dict = {}
    out = pooled_delta("TP53", [(pb, "control")], AXIS, shrinkage=False,
                       gamma=0.5, ctrl_tgt_cpm=np.array([50.0, 50.0, 50.0]), stats=stats)
    assert np.isfinite(out).all()
    assert stats["gamma_genes_transformed"] == 3      # D is off the target axis


def test_gamma_runs_after_shrinkage_not_before():
    """Gamma transforms the shrunk fold change: transform(shrink(fc))."""
    from sidechain.submit.build import gamma_transfer
    from sidechain.submit.build import shrink as shrink_fn

    noisy = _noisy_pb()
    ctrl_tgt = np.array([5.0, 5.0, 5.0])
    out = pooled_delta("TP53", [(noisy, "control")], AXIS, shrinkage=True,
                       gamma=0.0, ctrl_tgt_cpm=ctrl_tgt)
    assert np.allclose(out, 0.0)

    # and a solid effect composes exactly as transform(shrink(fc))
    solid = _pb(["control", "TP53"], [[100.0] * 3, [200.0] * 3])
    src = as_delta_source((solid, "control"))
    fc, var = src.effect("TP53")
    expected = gamma_transfer(shrink_fn(fc, var), src.control_cpm(), ctrl_tgt, 0.0)
    got = pooled_delta("TP53", [(solid, "control")], AXIS, shrinkage=True,
                       gamma=0.0, ctrl_tgt_cpm=ctrl_tgt)
    assert np.allclose(got, expected)


def test_gamma_leaves_the_pooling_weights_untouched():
    """Gamma leaves the inverse-variance weights as they were when two sources share a gene."""
    from sidechain.submit.build import gamma_transfer

    deep = _pb_var(["control", "TP53"], [[100.0] * 3, [200.0] * 3],
                   [[400.0] * 3, [900.0] * 3])
    wide = _pb_var(["control", "TP53"], [[400.0] * 3, [800.0] * 3],
                   [[10000.0] * 3, [40000.0] * 3])
    ctrl_tgt = np.array([50.0, 50.0, 50.0])

    num = np.zeros(3); den = np.zeros(3)
    for pb in (deep, wide):
        src = as_delta_source((pb, "control"))
        fc, var = src.effect("TP53")
        w = 1.0 / np.maximum(var, 1e-6)
        num += gamma_transfer(fc, src.control_cpm(), ctrl_tgt, 0.5) * w
        den += w
    got = pooled_delta("TP53", [(deep, "control"), (wide, "control")], AXIS,
                       shrinkage=False, gamma=0.5, ctrl_tgt_cpm=ctrl_tgt)
    assert np.allclose(got, num / den)


def test_gamma_without_the_target_control_profile_is_refused():
    import pytest

    pb = _pb(["control", "TP53"], [[100.0] * 3, [200.0] * 3])
    with pytest.raises(ValueError, match="ctrl_tgt_cpm"):
        pooled_delta("TP53", [(pb, "control")], AXIS, gamma=0.5)


def test_gamma_on_a_contrast_only_source_is_refused():
    """An LfcTable publishes the contrast already taken -- there is no control
    profile to ratio against, and inventing one (r = 1 everywhere) would silently
    hold that source at gamma = 1 inside a gamma arm. Refuse instead."""
    import pytest

    tab = _lfc(["TP53"], [[2.0, 0.0, 0.0]], [[0.01, np.inf, np.inf]])
    with pytest.raises(ValueError, match="control profile"):
        pooled_delta("TP53", [tab], AXIS, gamma=0.5,
                     ctrl_tgt_cpm=np.array([10.0, 10.0, 10.0]))


def test_loco_passes_gamma_through_and_records_it(monkeypatch, tmp_path):
    """`loco --gamma` reaches the builder and the log_run payload."""
    from sidechain.data.stream_pseudobulk import PseudobulkSums
    from sidechain.eval import loco

    monkeypatch.setattr(PseudobulkSums, "load", classmethod(lambda cls, p: _StubPB(f"PB:{p}")))
    captured, logged = {}, {}

    def fake_build(real, sources, out_path, **kw):
        captured.update(kw)
        return {}

    monkeypatch.setattr(loco, "build_transfer_prediction", fake_build)
    monkeypatch.setattr(loco, "attach_controls", lambda pred, real, out, **kw: out)
    monkeypatch.setattr(loco, "score", lambda *a, **kw: {"overall": 0.0, "members": {}})
    monkeypatch.setattr(loco, "log_run", lambda params, results, artifacts=None: logged.update(params))

    rc = loco.main(["--real", "r.h5ad", "--bundle", "b", "--out", str(tmp_path / "arm"),
                    "--source", "plain.npz:ctl", "--gamma", "0.25"])
    assert rc == 0
    assert captured["gamma"] == 0.25
    assert logged["gamma"] == 0.25


def test_gamma_reaches_the_emitted_cells_end_to_end(tmp_path):
    """gamma 0 moves the emitted share of a low-expressed gene well past the fold change alone, and not to 1."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    from sidechain.eval.loco import build_transfer_prediction

    genes = ["A", "B", "C"]
    n_ctrl, n_pert = 30, 8
    ctrl_counts = np.tile([10.0, 1000.0, 990.0], (n_ctrl, 1))       # 2,000 UMI/cell
    pert_counts = np.tile([10.0, 1000.0, 990.0], (n_pert, 1))
    X = sp.csr_matrix(np.vstack([ctrl_counts, pert_counts]))
    obs = pd.DataFrame({"target_gene": ["non-targeting"] * n_ctrl + ["TP53"] * n_pert},
                       index=[f"c{i}" for i in range(n_ctrl + n_pert)])
    real_path = tmp_path / "real.h5ad"
    ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(real_path)

    # source: gene A doubles, 100k -> 200k CPM; the target's controls sit at 5k CPM
    src = _pb(["control", "TP53"],
              [[100000.0, 450000.0, 450000.0], [200000.0, 400000.0, 400000.0]])

    shares, infos = {}, {}
    for gamma in (1.0, 0.0):
        out = tmp_path / f"pred_g{gamma:g}.h5ad"
        infos[gamma] = build_transfer_prediction(
            real_path, [(src, "control")], out, pert_col="target_gene",
            control="non-targeting", shrinkage=False, gamma=gamma, seed=0)
        pred = ad.read_h5ad(out)
        totals = np.asarray(pred.X.sum(axis=0)).ravel()
        shares[gamma] = totals[0] / totals.sum()
    assert infos[0.0]["gamma"] == 0.0
    assert infos[1.0]["gamma"] == 1.0
    # gamma=1 roughly doubles gene A's share; gamma=0 adds the source's absolute
    # +100k CPM onto 5k-CPM controls, a ~20x multiplier; bounded above too, where
    # a CPM units error at the call site drives the share to ~1.0
    assert shares[0.0] > 3 * shares[1.0]
    assert 0.02 < shares[0.0] < 0.2
    assert infos[0.0]["pool_stats"]["gamma_mult_clamped"] == 0


# ------------------------------------------------- the loco entry point --


def test_loco_requires_at_least_one_source_but_not_a_pseudobulk_one():
    """`loco` with no source of any kind is an argparse error."""
    import pytest

    from sidechain.eval import loco

    with pytest.raises(SystemExit) as exc:
        loco.main(["--real", "r.h5ad", "--bundle", "b", "--out", "o"])
    assert exc.value.code == 2      # argparse error, not a traceback


def test_loco_counts_a_shrink_source_as_a_complete_arm(tmp_path):
    """`--shrink-source` is a source, not a modifier of one: an arm built only
    from shrunk sources must pass the at-least-one-source check (it dies later
    on the missing file, which is the proof it got past argparse)."""
    import pytest

    from sidechain.eval import loco

    with pytest.raises(FileNotFoundError):
        loco.main(["--real", "r.h5ad", "--bundle", "b", "--out", str(tmp_path / "o"),
                   "--shrink-source", str(tmp_path / "missing.npz") + ":control"])


def test_row_wise_log2fc_matches_the_whole_matrix_form():
    """`_log2fc_with_var` row by row is bit-identical to the whole-matrix form."""
    from sidechain.models.count_emitters import log2fc_from_cpm
    from sidechain.submit.build import LN2_SQ, _log2fc_with_var

    rng = np.random.default_rng(7)
    n_labels, n_genes = 6, 40
    pb = PseudobulkSums(
        labels=[f"T{i}" for i in range(n_labels - 1)] + ["ctrl"],
        genes=np.array([f"g{j}" for j in range(n_genes)], dtype=object),
        count_sum=rng.random((n_labels, n_genes)) * 100,
        cpm_sum=rng.random((n_labels, n_genes)) * 1e6,
        cpm_sq_sum=rng.random((n_labels, n_genes)) * 1e12,
        n_cells=rng.integers(1, 500, n_labels),
        libsize_sum=rng.random(n_labels) * 1e5,
    )

    def matrix_form(label, control, pseudocount=1.0):
        i, c = pb.labels.index(label), pb.labels.index(control)
        m = pb.mean_cpm(); v = pb.var_cpm(); n = np.maximum(pb.n_cells, 1)
        fc = log2fc_from_cpm(m[i], m[c], pseudocount)
        var = (v[i] / n[i]) / (m[i] + pseudocount) ** 2 + (v[c] / n[c]) / (m[c] + pseudocount) ** 2
        return fc, var / LN2_SQ

    for label in pb.labels[:-1]:
        fc_new, var_new = _log2fc_with_var(pb, label, "ctrl")
        fc_old, var_old = matrix_form(label, "ctrl")
        assert np.array_equal(fc_new, fc_old), label
        assert np.array_equal(var_new, var_old), label


# --------------------------------------------- similarity-weighted pooling --


def test_control_similarity_is_a_log_cosine_over_shared_genes():
    from sidechain.submit.build import control_similarity

    a = np.arange(1, 201, dtype=float)
    assert control_similarity(a, a.copy()) == pytest.approx(1.0)
    # a profile with a different SHAPE scores below a matching one
    b = a[::-1].copy()
    assert control_similarity(a, b) < 1.0


def test_control_similarity_ignores_genes_only_one_side_measures():
    """`control_similarity` reads only the genes both profiles measure."""
    from sidechain.submit.build import control_similarity

    src = np.arange(1, 201, dtype=float)
    tgt = src.copy()
    padded_src = np.concatenate([src, np.full(300, np.nan)])
    padded_tgt = np.concatenate([tgt, np.arange(1, 301, dtype=float)])
    assert control_similarity(padded_src, padded_tgt) == pytest.approx(1.0)


def test_control_similarity_refuses_an_axis_mismatch_rather_than_scoring_it_low():
    """An empty overlap means the gene spaces do not line up. Returning a small cosine would
    weight that source toward zero and look like a modelling result."""
    from sidechain.submit.build import control_similarity

    a = np.concatenate([np.arange(1, 51, dtype=float), np.full(50, np.nan)])
    b = np.concatenate([np.full(50, np.nan), np.arange(1, 51, dtype=float)])
    with pytest.raises(ValueError, match="axis mismatch"):
        control_similarity(a, b)


def test_control_similarity_uses_log_not_raw_cpm():
    """`control_similarity` is a cosine on log1p CPM, which a dominant shared gene does not saturate."""
    from sidechain.submit.build import control_similarity

    n = 500
    base = np.full(n, 1.0)
    huge = base.copy(); huge[0] = 1e6           # one dominant gene, shared
    x = huge.copy(); y = huge.copy()
    x[1:250] = 50.0                              # the two differ only on modest genes
    y[250:] = 50.0
    log_cos = control_similarity(x, y)
    raw_cos = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    assert raw_cos > 0.999, "the raw cosine should be saturated by the dominant gene"
    assert log_cos < raw_cos, "log1p must expose the difference the raw cosine hides"


def test_similarity_beta_zero_is_bit_identical_to_uniform_pooling():
    """The endpoint property every knob in this codebase carries: the default cannot change a
    single historical number. x ** 0 == 1, exactly."""
    pb = _pb(["control", "T1"], [[100.0, 50.0, 10.0], [200.0, 50.0, 10.0]])
    tgt = np.array([120.0, 55.0, 11.0])
    a = pooled_delta("T1", [(pb, "control")], AXIS)
    b = pooled_delta("T1", [(pb, "control")], AXIS, similarity_beta=0.0, ctrl_tgt_cpm=tgt)
    np.testing.assert_array_equal(a, b)


def test_similarity_beta_needs_the_target_control_profile():
    pb = _pb(["control", "T1"], [[100.0, 50.0, 10.0], [200.0, 50.0, 10.0]])
    with pytest.raises(ValueError, match="similarity_beta"):
        pooled_delta("T1", [(pb, "control")], AXIS, similarity_beta=4.0)


def test_similarity_weighting_moves_the_pool_toward_the_more_alike_source():
    """The mechanism, on a case with a known answer: two sources disagree about T1, one has a
    control profile matching the target context and the other does not. Uniform pooling splits
    the difference; weighted pooling leans to the lookalike."""
    near = _pb(["control", "T1"], [[100.0, 10.0, 1.0], [200.0, 10.0, 1.0]])
    far = _pb(["control", "T1"], [[1.0, 10.0, 100.0], [0.5, 10.0, 100.0]])
    tgt = np.array([100.0, 10.0, 1.0])          # the target looks like `near`
    srcs = [(near, "control"), (far, "control")]

    uniform = pooled_delta("T1", srcs, AXIS, ctrl_tgt_cpm=tgt)
    weighted = pooled_delta("T1", srcs, AXIS, ctrl_tgt_cpm=tgt, similarity_beta=40.0)
    near_only = pooled_delta("T1", [(near, "control")], AXIS)

    assert abs(weighted[0] - near_only[0]) < abs(uniform[0] - near_only[0]), (
        "weighting should pull the pooled G1 effect toward the lookalike source"
    )


def test_a_source_with_no_control_profile_keeps_full_weight_and_is_counted():
    """An LfcTable publishes the contrast already taken, so it has no control arm and no
    similarity. Dropping it would silently delete a source; it is left alone and recorded."""
    lfc = _lfc(["T1"], [[1.0, 0.0, 0.0]], [[0.25, 0.25, 0.25]])
    pb = _pb(["control", "T1"], [[100.0, 10.0, 1.0], [200.0, 10.0, 1.0]])
    stats: dict = {}
    out = pooled_delta("T1", [(pb, "control"), lfc], AXIS,
                       ctrl_tgt_cpm=np.array([100.0, 10.0, 1.0]),
                       similarity_beta=8.0, stats=stats)
    assert out is not None
    assert stats["similarity_sources_unweighted"] == 1


# ── the log-bias correction ─────────────────────────────────────────────────────────────


def _two_arm_pb(n_pert: int, n_ctrl: int, mean_cpm: float, var_cpm: float):
    """A PseudobulkSums whose two arms have a KNOWN mean and per-cell variance, so the
    correction's size is arithmetic rather than a fit."""
    import numpy as np

    from sidechain.data.stream_pseudobulk import PseudobulkSums

    g = np.array(["gA", "gB"])
    m = np.array([mean_cpm, mean_cpm])
    # cpm_sq_sum / n - m^2 = var  =>  cpm_sq_sum = n (var + m^2)
    return PseudobulkSums(
        labels=["tgt", "ctl"], genes=g,
        count_sum=np.vstack([m, m]),
        cpm_sum=np.vstack([m * n_pert, m * n_ctrl]),
        cpm_sq_sum=np.vstack([n_pert * (var_cpm + m * m), n_ctrl * (var_cpm + m * m)]),
        n_cells=np.array([n_pert, n_ctrl]),
        libsize_sum=np.array([n_pert * 20_000.0, n_ctrl * 20_000.0]),
        sources=np.array(["t"]))


def test_the_log_bias_correction_is_off_by_default_and_changes_nothing():
    import numpy as np

    from sidechain.submit.build import _log2fc_with_var

    pb = _two_arm_pb(150, 150_000, 10.0, 400.0)
    base, _ = _log2fc_with_var(pb, "tgt", "ctl")
    same, _ = _log2fc_with_var(pb, "tgt", "ctl", log_bias_correct=False)
    assert np.array_equal(base, same)


def test_the_log_bias_correction_lifts_a_shallow_arm_by_the_delta_method_term():
    """`log2` of a noisy mean sits Var(m_hat)/(2(m+c)^2 ln2) below log2 of the true mean. The
    perturbed arm is shallow and the control is deep, so the two do not cancel and the
    correction is positive -- the size is checked against the closed form, not a snapshot."""
    import numpy as np

    from sidechain.submit.build import LN2, _log2fc_with_var

    n_p, n_c, m, v = 150, 150_000, 10.0, 400.0
    pb = _two_arm_pb(n_p, n_c, m, v)
    base, _ = _log2fc_with_var(pb, "tgt", "ctl")
    fixed, _ = _log2fc_with_var(pb, "tgt", "ctl", log_bias_correct=True)
    want = ((v / n_p) / (m + 1.0) ** 2 - (v / n_c) / (m + 1.0) ** 2) / (2.0 * LN2)
    assert np.allclose(fixed - base, want)
    assert (fixed > base).all()          # the shallow arm was biased DOWN; this lifts it


def test_two_equally_deep_arms_need_no_correction():
    """The bias is a depth asymmetry. Same cells, same spread, nothing left to remove."""
    import numpy as np

    from sidechain.submit.build import _log2fc_with_var

    pb = _two_arm_pb(500, 500, 10.0, 400.0)
    base, _ = _log2fc_with_var(pb, "tgt", "ctl")
    fixed, _ = _log2fc_with_var(pb, "tgt", "ctl", log_bias_correct=True)
    assert np.allclose(fixed, base)


def test_pooled_delta_threads_the_flag_to_its_sources():
    import numpy as np

    from sidechain.submit.build import pooled_delta

    pb = _two_arm_pb(150, 150_000, 10.0, 400.0)
    axis = np.array(["gA", "gB"])
    off = pooled_delta("tgt", [(pb, "ctl")], axis, shrinkage=False, var_floor="poisson")
    on = pooled_delta("tgt", [(pb, "ctl")], axis, shrinkage=False, var_floor="poisson",
                      log_bias_correct=True)
    assert not np.allclose(off, on) and (on > off).all()
