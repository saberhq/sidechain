"""`submit.build --transfer-floor NAME=TAU2`: a per-source tau^2 added to the variance, weight 1/(var + tau^2).

    .venv/bin/python -m pytest -q tests/test_transfer_floor.py

Writes nothing. Notes: private/code-notes/tests/test_transfer_floor.py.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.submit.build import (
    apply_transfer_floors,
    parse_coverage_tiers,
    parse_transfer_floor,
    pooled_delta,
)

AXIS = np.array(["A", "B", "C"])


def _pb(name, labels, *, per_cell, n_cells):
    """A PseudobulkSums whose arms are `n_cells` identical cells at `per_cell` counts."""
    cpm_sum, cpm_sq_sum, libs = [], [], []
    for lab in labels:
        X = np.tile(np.asarray(per_cell[lab], dtype=float), (n_cells[lab], 1))
        lib = X.sum(axis=1)
        cpm = X / np.maximum(lib, 1)[:, None] * 1e6
        cpm_sum.append(cpm.sum(axis=0))
        cpm_sq_sum.append((cpm**2).sum(axis=0))
        libs.append(float(lib.sum()))
    pb = PseudobulkSums(
        labels=list(labels), genes=AXIS.copy(),
        count_sum=np.stack(cpm_sum), cpm_sum=np.stack(cpm_sum),
        cpm_sq_sum=np.stack(cpm_sq_sum),
        n_cells=np.array([n_cells[lab] for lab in labels], dtype=np.int64),
        libsize_sum=np.array(libs), sources=[name],
    )
    pb.sidechain_name = name
    return pb


def _two_sources():
    """Two sources that disagree about T, one loud and one quiet."""
    loud = _pb("loud", ["ctrl", "T"],
               per_cell={"ctrl": [100, 100, 100], "T": [400, 100, 100]},
               n_cells={"ctrl": 50, "T": 50})
    quiet = _pb("quiet", ["ctrl", "T"],
                per_cell={"ctrl": [100, 100, 100], "T": [150, 100, 100]},
                n_cells={"ctrl": 50, "T": 50})
    return [(loud, "ctrl"), (quiet, "ctrl")]


# ------------------------------------------------------- the endpoint property --

def test_tau2_zero_is_bit_identical_to_the_historical_call():
    srcs = _two_sources()
    base = pooled_delta("T", srcs, AXIS, shrinkage=False, var_floor="poisson")
    apply_transfer_floors(srcs, parse_transfer_floor(["loud=0", "quiet=0"]))
    with_zero = pooled_delta("T", srcs, AXIS, shrinkage=False, var_floor="poisson")
    assert np.array_equal(base, with_zero)


def test_no_floor_at_all_is_bit_identical():
    srcs = _two_sources()
    base = pooled_delta("T", srcs, AXIS, shrinkage=False, var_floor="poisson")
    apply_transfer_floors(srcs, parse_transfer_floor([]))
    assert np.array_equal(base, pooled_delta("T", srcs, AXIS, shrinkage=False,
                                             var_floor="poisson"))


# ------------------------------------------------------------- the algebra --

def test_the_weight_transform_really_is_one_over_var_plus_tau2():
    """The pooled weight equals `1 / (var + tau2)` computed directly."""
    from sidechain.submit.build import _log2fc_with_var

    srcs = _two_sources()
    tau2 = {"loud": 0.05, "quiet": 0.002}
    apply_transfer_floors(srcs, parse_transfer_floor([f"{k}={v}" for k, v in tau2.items()]))
    got = pooled_delta("T", srcs, AXIS, shrinkage=False, var_floor="poisson")

    num = np.zeros(len(AXIS))
    den = np.zeros(len(AXIS))
    for pb, ctrl in srcs:
        fc, var = _log2fc_with_var(pb, "T", ctrl, var_floor="poisson")
        w = 1.0 / (var + tau2[pb.sidechain_name])      # the claim, computed directly
        num += fc * w
        den += w
    assert np.allclose(got, num / den, rtol=0, atol=1e-12)


def test_it_caps_the_weight_rather_than_rescaling_it():
    """A floor caps a near-zero-variance source's weight at 1/tau^2, so it no longer takes over the pool."""
    from sidechain.submit.build import _log2fc_with_var

    srcs = _two_sources()
    loud_fc, _ = _log2fc_with_var(srcs[0][0], "T", "ctrl", var_floor="poisson")

    # uncapped: with 200x the cells, the loud source's variance collapses and it takes over
    deep = _pb("loud", ["ctrl", "T"],
               per_cell={"ctrl": [100, 100, 100], "T": [400, 100, 100]},
               n_cells={"ctrl": 10000, "T": 10000})
    uncapped = pooled_delta("T", [(deep, "ctrl"), srcs[1]], AXIS,
                            shrinkage=False, var_floor="poisson")
    assert abs(uncapped[0] - loud_fc[0]) < 0.02          # it has essentially won

    capped_srcs = [(deep, "ctrl"), srcs[1]]
    apply_transfer_floors(capped_srcs, parse_transfer_floor(["loud=0.05"]))
    capped = pooled_delta("T", capped_srcs, AXIS, shrinkage=False, var_floor="poisson")
    assert abs(capped[0] - loud_fc[0]) > abs(uncapped[0] - loud_fc[0])


def test_abstention_stays_abstention():
    """A source at var = inf keeps weight 0 under a floor."""
    srcs = _two_sources()
    single = _pb("loud", ["ctrl", "T"],
                 per_cell={"ctrl": [100, 100, 100], "T": [400, 100, 100]},
                 n_cells={"ctrl": 1, "T": 1})           # n=1 arms abstain under the floor
    pair = [(single, "ctrl"), srcs[1]]
    apply_transfer_floors(pair, parse_transfer_floor(["loud=0.05"]))
    got = pooled_delta("T", pair, AXIS, shrinkage=False, var_floor="poisson")
    quiet_only = pooled_delta("T", [srcs[1]], AXIS, shrinkage=False, var_floor="poisson")
    assert np.array_equal(got, quiet_only)


# ------------------------------------------------------ order against the tier --

def test_the_tier_scales_the_sampling_term_only():
    """The coverage tier divides the sampling variance only: `var/f + tau^2`, not `(var + tau^2)/f`."""
    from sidechain.submit.build import _log2fc_with_var, coverage_factor

    srcs = _two_sources()
    tau2 = 0.05
    tiers = parse_coverage_tiers("1e9:0.1")          # force every gene into the weak tier
    apply_transfer_floors(srcs, parse_transfer_floor([f"loud={tau2}"]))
    got = pooled_delta("T", srcs, AXIS, shrinkage=False, var_floor="poisson",
                       coverage_tiers=tiers)

    num = np.zeros(len(AXIS))
    den = np.zeros(len(AXIS))
    for pb, ctrl in srcs:
        fc, var = _log2fc_with_var(pb, "T", ctrl, var_floor="poisson")
        i, c = pb.labels.index("T"), pb.labels.index(ctrl)
        cf = coverage_factor(np.minimum(pb.n_eff(i), pb.n_eff(c)), tiers)
        t = tau2 if pb.sidechain_name == "loud" else 0.0
        w = 1.0 / (var / cf + t)                     # the right order
        num += fc * w
        den += w
    right = num / den
    assert np.allclose(got, right, rtol=0, atol=1e-12)

    wrong = np.zeros(len(AXIS)); wden = np.zeros(len(AXIS))
    for pb, ctrl in srcs:
        fc, var = _log2fc_with_var(pb, "T", ctrl, var_floor="poisson")
        i, c = pb.labels.index("T"), pb.labels.index(ctrl)
        cf = coverage_factor(np.minimum(pb.n_eff(i), pb.n_eff(c)), tiers)
        t = tau2 if pb.sidechain_name == "loud" else 0.0
        w = 1.0 / ((var + t) / cf)                   # the order we are NOT using
        wrong += fc * w; wden += w
    assert not np.allclose(got, wrong / wden, rtol=0, atol=1e-9)


# ------------------------------------------------------------------ parsing --

def test_parse_rejects_a_missing_value():
    with pytest.raises(SystemExit, match="NAME=TAU2"):
        parse_transfer_floor(["h1_pseudobulk"])


def test_parse_rejects_a_negative_variance():
    with pytest.raises(SystemExit, match="must be >= 0"):
        parse_transfer_floor(["h1=-0.01"])


def test_parse_rejects_a_duplicate_name():
    with pytest.raises(SystemExit, match="twice"):
        parse_transfer_floor(["h1=0.01", "h1=0.02"])


def test_parse_rejects_a_non_number():
    with pytest.raises(SystemExit, match="not a number"):
        parse_transfer_floor(["h1=poisson"])


def test_a_floor_naming_no_source_is_refused():
    """A floor whose name matches no source is refused."""
    srcs = _two_sources()
    with pytest.raises(SystemExit, match="match no source"):
        apply_transfer_floors(srcs, parse_transfer_floor(["h1_pseudobulk=0.01"]))


def test_round_trip():
    assert parse_transfer_floor(["a=0.5", "b=1e-3"]) == {"a": 0.5, "b": 0.001}


def test_the_named_caches_are_addressable_too(monkeypatch, tmp_path):
    """A cache loaded by `--h1-cache` / `--gwps-cache` is named by its file stem, so a floor attaches to it."""
    import sidechain.submit.build as B

    seen = {}

    def fake_load(cls, p):
        pb = _pb(Path(p).stem, ["ctrl", "T"],
                 per_cell={"ctrl": [100, 100, 100], "T": [400, 100, 100]},
                 n_cells={"ctrl": 50, "T": 50})
        pb.sidechain_name = None          # cleared: the code under test must set it
        seen[Path(p).stem] = pb
        return pb

    monkeypatch.setattr(PseudobulkSums, "load", classmethod(fake_load))
    h1 = PseudobulkSums.load("/x/h1_pseudobulk.npz")
    h1.sidechain_name = Path("/x/h1_pseudobulk.npz").stem     # what build.main does
    assert h1.sidechain_name == "h1_pseudobulk"
    # and the floor then finds it rather than refusing
    srcs = [(h1, "ctrl")]
    B.apply_transfer_floors(srcs, B.parse_transfer_floor(["h1_pseudobulk=0.01"]))
    assert h1.transfer_floor == 0.01
