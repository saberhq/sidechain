"""Contract tests for ``sidechain.eval.transfer_tau2`` -- the pairwise tau^2 fitter.

The thing being pinned here is not the arithmetic, which is a 60-iteration golden
section and was already right twice. It is the GUARD, because the arithmetic being
right is exactly what makes the failure invisible:

    **A tau^2 is only comparable to another tau^2 fit on the same gene axis.**

On 2026-09-21 a hand-rolled version let each pair use its own intersection. The
X-Atlas pair got 38,584 genes where every other pair got ~8,000, and its tau^2 came
out FIVE TIMES too low (0.00100 against a known 0.0053) because the wide axis is
mostly genes neither line expresses, where both read ~0 and agree for free. Every
number was computed correctly. Nothing warned. So:

* `fit_pairs` must share one axis across every pair, by construction.
* Two fits on different axes must REFUSE to be compared, not silently divide.
* A wide axis must actually drag tau^2 down -- the test asserts the failure mode is
  real, so the guard is protecting against something rather than decorating.

Also pinned: tau^2 recovers a known injected floor; it is symmetric in its
arguments; a self-pair is refused; and `load_subset` returns what `load` would.
"""
from __future__ import annotations

import numpy as np
import pytest

from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.eval.transfer_tau2 import (
    Tau2Fit,
    axis_fingerprint,
    common_axis,
    common_axis_from_paths,
    fit_pair,
    fit_pairs,
    load_for_axis,
)

CONTROL = "control"


def _synthetic(genes, targets, *, tau: float, seed: int, n_cells: int = 200,
               quiet_genes: int = 0) -> PseudobulkSums:
    """A pseudobulk whose log2FCs sit `tau` away from a shared truth.

    Two sources built with the same `truth_seed` but different `seed` disagree by
    sqrt(2)*tau in expectation, so a fit over the pair recovers 2*tau^2. Genes past
    `len(genes) - quiet_genes` are given no signal at all in either source: they are
    the "neither line expresses this" genes that a too-wide axis is full of.
    """
    rng = np.random.default_rng(seed)
    truth = np.random.default_rng(12345).normal(0.0, 1.0, (len(targets), len(genes)))
    if quiet_genes:
        truth[:, len(genes) - quiet_genes:] = 0.0
    offset = rng.normal(0.0, tau, truth.shape)
    if quiet_genes:
        offset[:, len(genes) - quiet_genes:] = 0.0
    fc = truth + offset

    labels = [CONTROL] + list(targets)
    base = 50.0
    cpm_sum = np.empty((len(labels), len(genes)))
    cpm_sq = np.empty_like(cpm_sum)
    cpm_sum[0] = base * n_cells
    # tiny within-arm spread -> the sampling variance is negligible beside tau^2,
    # so the fit has to find tau^2 rather than absorb it into var.
    cpm_sq[0] = (base ** 2 + 1e-6) * n_cells
    for i, _ in enumerate(targets, start=1):
        m = (base + 1.0) * (2.0 ** fc[i - 1]) - 1.0
        cpm_sum[i] = m * n_cells
        cpm_sq[i] = (m ** 2 + 1e-6) * n_cells
    return PseudobulkSums(
        labels=labels, genes=np.asarray(genes, dtype=object),
        count_sum=np.zeros_like(cpm_sum), cpm_sum=cpm_sum, cpm_sq_sum=cpm_sq,
        n_cells=np.full(len(labels), n_cells, dtype=np.int64),
        libsize_sum=np.full(len(labels), 1e6 * n_cells, dtype=np.float64),
        sources=["synthetic"],
    )


GENES = [f"G{i:04d}" for i in range(400)]
TARGETS = [f"T{i:03d}" for i in range(60)]


# ------------------------------------------------------------------ the guard

def test_fits_on_different_axes_refuse_to_be_compared():
    """The whole reason this module exists: silently dividing is the bug."""
    wide = GENES
    narrow = GENES[:200]
    a_w = _synthetic(wide, TARGETS, tau=0.10, seed=1)
    b_w = _synthetic(wide, TARGETS, tau=0.10, seed=2)
    a_n = _synthetic(narrow, TARGETS, tau=0.10, seed=1)
    b_n = _synthetic(narrow, TARGETS, tau=0.10, seed=2)

    f_wide = fit_pair(a_w, CONTROL, b_w, CONTROL, genes=wide, targets=TARGETS)
    f_narrow = fit_pair(a_n, CONTROL, b_n, CONTROL, genes=narrow, targets=TARGETS)

    assert f_wide.axis != f_narrow.axis
    with pytest.raises(ValueError, match="DIFFERENT axes"):
        f_wide.assert_comparable(f_narrow)
    with pytest.raises(ValueError, match="DIFFERENT axes"):
        f_wide.ratio_to(f_narrow)


def test_a_wide_axis_really_does_drag_tau2_down():
    """The guard protects against a real effect, not a hypothetical one.

    Half the wide axis is genes neither source has signal on. Both read ~0 there
    and agree for free, so the fitted floor over the wide axis must come out BELOW
    the one over only the informative genes -- which is exactly how a real pair read
    five times too low on 2026-09-21.
    """
    informative = GENES[:200]
    wide = GENES                       # 200 informative + 200 silent
    a = _synthetic(wide, TARGETS, tau=0.20, seed=1, quiet_genes=200)
    b = _synthetic(wide, TARGETS, tau=0.20, seed=2, quiet_genes=200)

    f_wide = fit_pair(a, CONTROL, b, CONTROL, genes=wide, targets=TARGETS)

    sub_a = PseudobulkSums(
        labels=a.labels, genes=np.asarray(informative, dtype=object),
        count_sum=a.count_sum[:, :200], cpm_sum=a.cpm_sum[:, :200],
        cpm_sq_sum=a.cpm_sq_sum[:, :200], n_cells=a.n_cells,
        libsize_sum=a.libsize_sum, sources=a.sources)
    sub_b = PseudobulkSums(
        labels=b.labels, genes=np.asarray(informative, dtype=object),
        count_sum=b.count_sum[:, :200], cpm_sum=b.cpm_sum[:, :200],
        cpm_sq_sum=b.cpm_sq_sum[:, :200], n_cells=b.n_cells,
        libsize_sum=b.libsize_sum, sources=b.sources)
    f_informative = fit_pair(sub_a, CONTROL, sub_b, CONTROL,
                             genes=informative, targets=TARGETS)

    assert f_wide.tau2 < 0.5 * f_informative.tau2, (
        f"wide {f_wide.tau2:.5f} vs informative {f_informative.tau2:.5f} -- the "
        "dilution this module guards against did not reproduce")


def test_fit_pairs_stamps_one_axis_on_every_pair():
    """Even though the three sources have DIFFERENT gene sets of their own."""
    srcs = {
        "a": _synthetic(GENES, TARGETS, tau=0.10, seed=1),
        "b": _synthetic(GENES[:300], TARGETS, tau=0.10, seed=2),
        "c": _synthetic(GENES[:250], TARGETS, tau=0.30, seed=3),
    }
    ctrls = {k: CONTROL for k in srcs}
    genes, targets = common_axis(srcs, ctrls)
    assert len(genes) == 250                      # the narrowest wins, for all pairs

    subs = {k: PseudobulkSums(
        labels=v.labels, genes=np.asarray(genes, dtype=object),
        count_sum=v.count_sum[:, :250], cpm_sum=v.cpm_sum[:, :250],
        cpm_sq_sum=v.cpm_sq_sum[:, :250], n_cells=v.n_cells,
        libsize_sum=v.libsize_sum, sources=v.sources) for k, v in srcs.items()}

    fits = fit_pairs(subs, ctrls, [("a", "b"), ("a", "c"), ("b", "c")],
                     genes=genes, targets=targets)
    assert len({f.axis for f in fits}) == 1
    assert all(f.n_genes == 250 for f in fits)
    for f in fits[1:]:
        fits[0].assert_comparable(f)              # must not raise


# ------------------------------------------------------------------ the numbers

def test_recovers_an_injected_floor():
    """Two sources each `tau` off a shared truth disagree by 2*tau^2 in total."""
    for tau in (0.10, 0.25):
        a = _synthetic(GENES, TARGETS, tau=tau, seed=7)
        b = _synthetic(GENES, TARGETS, tau=tau, seed=8)
        f = fit_pair(a, CONTROL, b, CONTROL, genes=GENES, targets=TARGETS)
        assert f.tau2 == pytest.approx(2 * tau ** 2, rel=0.25), \
            f"tau={tau}: fit {f.tau2:.5f}, expected ~{2 * tau ** 2:.5f}"
        assert f.tau == pytest.approx(np.sqrt(f.tau2))


def test_is_symmetric_in_its_two_sources():
    """tau^2 is a property of the unordered pair, so swapping must not move it."""
    a = _synthetic(GENES, TARGETS, tau=0.15, seed=4)
    b = _synthetic(GENES, TARGETS, tau=0.15, seed=5)
    ab = fit_pair(a, CONTROL, b, CONTROL, genes=GENES, targets=TARGETS)
    ba = fit_pair(b, CONTROL, a, CONTROL, genes=GENES, targets=TARGETS)
    assert ab.tau2 == pytest.approx(ba.tau2, rel=1e-9)
    assert ab.axis == ba.axis


def test_ratio_of_two_pairs_tracks_their_injected_floors():
    srcs = {
        "near1": _synthetic(GENES, TARGETS, tau=0.10, seed=11),
        "near2": _synthetic(GENES, TARGETS, tau=0.10, seed=12),
        "far": _synthetic(GENES, TARGETS, tau=0.30, seed=13),
    }
    ctrls = {k: CONTROL for k in srcs}
    fits = {f.pair: f for f in fit_pairs(srcs, ctrls,
                                         [("near1", "near2"), ("near1", "far")],
                                         genes=GENES, targets=TARGETS)}
    near = fits[("near1", "near2")]
    far = fits[("near1", "far")]
    assert far.ratio_to(near) > 2.0


def test_bootstrap_brackets_the_point_estimate():
    srcs = {"a": _synthetic(GENES, TARGETS, tau=0.20, seed=21),
            "b": _synthetic(GENES, TARGETS, tau=0.20, seed=22)}
    ctrls = {k: CONTROL for k in srcs}
    (f,) = fit_pairs(srcs, ctrls, [("a", "b")], genes=GENES, targets=TARGETS,
                     bootstrap=25, seed=3)
    assert f.ci95 is not None
    lo, hi = f.ci95
    assert lo < f.tau2 < hi
    assert f.extra["bootstrap"] == 25


def test_bootstrap_is_reproducible_from_its_seed():
    srcs = {"a": _synthetic(GENES, TARGETS, tau=0.20, seed=21),
            "b": _synthetic(GENES, TARGETS, tau=0.20, seed=22)}
    ctrls = {k: CONTROL for k in srcs}
    kw = dict(genes=GENES, targets=TARGETS, bootstrap=12, seed=99)
    (one,) = fit_pairs(srcs, ctrls, [("a", "b")], **kw)
    (two,) = fit_pairs(srcs, ctrls, [("a", "b")], **kw)
    assert one.ci95 == two.ci95


# ------------------------------------------------------------------ the refusals

def test_a_self_pair_is_refused():
    srcs = {"a": _synthetic(GENES, TARGETS, tau=0.1, seed=1)}
    srcs["b"] = _synthetic(GENES, TARGETS, tau=0.1, seed=2)
    with pytest.raises(ValueError, match="against itself"):
        fit_pairs(srcs, {k: CONTROL for k in srcs}, [("a", "a")],
                  genes=GENES, targets=TARGETS)


def test_common_axis_excludes_every_control_label():
    srcs = {"a": _synthetic(GENES, TARGETS, tau=0.1, seed=1),
            "b": _synthetic(GENES, TARGETS, tau=0.1, seed=2)}
    _, targets = common_axis(srcs, {k: CONTROL for k in srcs})
    assert CONTROL not in targets
    assert len(targets) == len(TARGETS)


def test_a_missing_control_label_is_named_not_silently_dropped():
    srcs = {"a": _synthetic(GENES, TARGETS, tau=0.1, seed=1),
            "b": _synthetic(GENES, TARGETS, tau=0.1, seed=2)}
    with pytest.raises(KeyError, match="matches no label"):
        common_axis(srcs, {"a": CONTROL, "b": "non-targeting"})


def test_a_mismatched_gene_axis_is_refused_at_the_door():
    """fit_pair will not quietly fit sources whose axis is not the one asked for."""
    a = _synthetic(GENES, TARGETS, tau=0.1, seed=1)
    b = _synthetic(GENES, TARGETS, tau=0.1, seed=2)
    with pytest.raises(ValueError, match="does not match"):
        fit_pair(a, CONTROL, b, CONTROL, genes=GENES[:100], targets=TARGETS)


def test_family_must_be_one_we_implement():
    a = _synthetic(GENES, TARGETS, tau=0.1, seed=1)
    b = _synthetic(GENES, TARGETS, tau=0.1, seed=2)
    with pytest.raises(ValueError, match="family must be"):
        fit_pair(a, CONTROL, b, CONTROL, genes=GENES, targets=TARGETS, family="cauchy")


def test_fingerprint_ignores_order_but_not_content():
    assert axis_fingerprint(["a", "b"], ["t"]) == axis_fingerprint(["b", "a"], ["t"])
    assert axis_fingerprint(["a", "b"], ["t"]) != axis_fingerprint(["a", "c"], ["t"])
    assert axis_fingerprint(["a"], ["t"]) != axis_fingerprint(["a"], ["u"])


def test_different_var_floor_or_family_is_not_comparable():
    base = dict(pair=("a", "b"), tau2=0.01, n_targets=5, n_genes=10, n_cells=50,
                axis="deadbeefdeadbeef")
    g = Tau2Fit(family="gauss", var_floor="poisson", **base)
    t = Tau2Fit(family="t4", var_floor="poisson", **base)
    n = Tau2Fit(family="gauss", var_floor="none", **base)
    with pytest.raises(ValueError, match="likelihood family"):
        g.assert_comparable(t)
    with pytest.raises(ValueError, match="var_floor"):
        g.assert_comparable(n)


# ------------------------------------------------------------------ load_subset

def test_load_subset_matches_a_full_load(tmp_path):
    pb = _synthetic(GENES, TARGETS, tau=0.1, seed=1)
    path = tmp_path / "pb.npz"
    pb.save(path)

    want_labels = [CONTROL, TARGETS[5], TARGETS[0], TARGETS[40]]   # deliberately unsorted
    want_genes = [GENES[9], GENES[0], GENES[399], GENES[100]]
    sub = PseudobulkSums.load_subset(path, want_labels, want_genes)
    full = PseudobulkSums.load(path)

    assert sub.labels == want_labels
    assert [str(g) for g in sub.genes] == want_genes
    for si, lab in enumerate(want_labels):
        fi = full.labels.index(lab)
        for sj, gene in enumerate(want_genes):
            fj = list(full.genes).index(gene)
            assert sub.cpm_sum[si, sj] == pytest.approx(full.cpm_sum[fi, fj])
            assert sub.cpm_sq_sum[si, sj] == pytest.approx(full.cpm_sq_sum[fi, fj])
        assert sub.n_cells[si] == full.n_cells[fi]
        assert sub.libsize_sum[si] == pytest.approx(full.libsize_sum[fi])


def test_load_subset_names_what_is_missing(tmp_path):
    pb = _synthetic(GENES, TARGETS, tau=0.1, seed=1)
    path = tmp_path / "pb.npz"
    pb.save(path)
    with pytest.raises(KeyError, match="gene"):
        PseudobulkSums.load_subset(path, [CONTROL], ["NOT_A_GENE"])
    with pytest.raises(KeyError, match="label"):
        PseudobulkSums.load_subset(path, ["NOT_A_LABEL"], [GENES[0]])


def test_load_subset_feeds_a_fit_identical_to_the_full_one(tmp_path):
    """The path a real caller takes: save, load_subset, fit -- same number."""
    a = _synthetic(GENES, TARGETS, tau=0.2, seed=31)
    b = _synthetic(GENES, TARGETS, tau=0.2, seed=32)
    pa, pb_ = tmp_path / "a.npz", tmp_path / "b.npz"
    a.save(pa)
    b.save(pb_)
    direct = fit_pair(a, CONTROL, b, CONTROL, genes=GENES, targets=TARGETS)
    via = fit_pair(PseudobulkSums.load_subset(pa, [CONTROL] + TARGETS, GENES), CONTROL,
                   PseudobulkSums.load_subset(pb_, [CONTROL] + TARGETS, GENES), CONTROL,
                   genes=GENES, targets=TARGETS)
    assert via.tau2 == pytest.approx(direct.tau2, rel=1e-12)
    assert via.axis == direct.axis


# ------------------------------------------- the path-based route, for big artifacts

def test_common_axis_from_paths_matches_the_loaded_form(tmp_path):
    """The axis must be discoverable WITHOUT loading -- that is the point of it.

    On a full-corpus artifact `common_axis` is unreachable: you would have to load
    5.65 GB per matrix to find out which genes it has.
    """
    srcs = {"a": _synthetic(GENES, TARGETS, tau=0.1, seed=1),
            "b": _synthetic(GENES[:300], TARGETS[:40], tau=0.1, seed=2)}
    paths = {}
    for k, v in srcs.items():
        paths[k] = tmp_path / f"{k}.npz"
        v.save(paths[k])
    ctrls = {k: CONTROL for k in srcs}

    g_paths, t_paths = common_axis_from_paths(paths, ctrls)
    g_loaded, t_loaded = common_axis(srcs, ctrls)
    assert g_paths == g_loaded == GENES[:300]
    assert t_paths == t_loaded == sorted(TARGETS[:40])


def test_load_for_axis_then_fit_pairs_is_the_whole_real_workflow(tmp_path):
    srcs = {"near1": _synthetic(GENES, TARGETS, tau=0.10, seed=41),
            "near2": _synthetic(GENES, TARGETS, tau=0.10, seed=42),
            "far": _synthetic(GENES[:300], TARGETS, tau=0.35, seed=43)}
    paths = {}
    for k, v in srcs.items():
        paths[k] = tmp_path / f"{k}.npz"
        v.save(paths[k])
    ctrls = {k: CONTROL for k in srcs}

    genes, targets = common_axis_from_paths(paths, ctrls)
    loaded = load_for_axis(paths, ctrls, genes, targets)
    fits = fit_pairs(loaded, ctrls, [("near1", "near2"), ("near1", "far")],
                     genes=genes, targets=targets)

    assert len({f.axis for f in fits}) == 1          # one axis, as always
    assert all(f.n_genes == 300 for f in fits)
    by = {f.pair: f for f in fits}
    assert by[("near1", "far")].ratio_to(by[("near1", "near2")]) > 2.0


def test_common_axis_from_paths_names_a_bad_control(tmp_path):
    pb = _synthetic(GENES, TARGETS, tau=0.1, seed=1)
    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    pb.save(p1)
    pb.save(p2)
    with pytest.raises(KeyError, match="matches no label"):
        common_axis_from_paths({"a": p1, "b": p2}, {"a": CONTROL, "b": "nope"})
