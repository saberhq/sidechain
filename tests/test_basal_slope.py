"""Contract tests for T77's basal-expression slope (`models.basal_slope`).

What is pinned: the fit recovers a planted slope and its intercept is the pooled delta; a
line whose CPM is on a different gene universe changes nothing; no dependence means no
modifier; `off` leaves `eval.loco` bit-identical.
"""
from __future__ import annotations

import numpy as np
import pytest

from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.models.basal_slope import fit_basal_slopes, target_basal
from sidechain.submit.build import pooled_delta

GENES = np.array([f"g{i}" for i in range(40)], dtype=object)
TARGETS = [f"g{i}" for i in range(12)]
CTRL = "ctrl"


def _line(rng, basal_cpm, slopes, *, scale=1.0, n_cells=2000, genes=GENES, noise=0.0):
    """A pseudobulk source whose per-target log2FC is `beta + slope * x`, with `beta` shared
    across lines and `x` the line's basal on the fit's own footing (log1p CPM with the shared
    genes -- here all of them -- carrying 1e6); `scale` mimics a different gene universe. Basal
    CPM stays above 100 so the pool's 1-CPM pseudocount is a small part of every fold change."""
    G = len(genes)
    labels = [CTRL] + TARGETS
    mean = np.zeros((len(labels), G))
    x = np.log1p(basal_cpm * 1e6 / basal_cpm.sum())
    mean[0] = basal_cpm
    for i, t in enumerate(TARGETS):
        beta = 0.3 * np.sin(np.arange(G) + i)
        fc = beta + slopes[i] * x + noise * rng.normal(size=G)
        mean[i + 1] = basal_cpm * np.exp2(fc)
    mean = mean * scale
    n = np.full(len(labels), n_cells, dtype=np.int64)
    pb = PseudobulkSums(labels=labels, genes=genes.copy(), count_sum=mean * n[:, None],
                        cpm_sum=mean * n[:, None], cpm_sq_sum=(mean**2 + mean) * n[:, None],
                        n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["test"])
    return pb


def _sources(rng, slopes, scales=(1.0, 1.0, 1.0), noise=0.0):
    """Three lines with independent basal profiles; `scales` multiplies a whole line's CPM
    (control and targets alike), which is what a different gene universe does to a pseudobulk
    and what the fit's per-line rescaling exists to undo."""
    basals = [rng.uniform(100, 2000, size=len(GENES)) for _ in range(3)]
    return [(_line(rng, b, slopes, scale=s, noise=noise), CTRL) for b, s in zip(basals, scales)]


def test_the_intercept_at_the_weighted_basal_is_the_pooled_delta():
    rng = np.random.default_rng(0)
    slopes = np.full(len(TARGETS), 0.2)
    srcs = _sources(rng, slopes)
    fit = fit_basal_slopes(TARGETS, srcs, GENES, min_common=10)
    # predicting a line whose basal IS xbar adds nothing: the modifier is exactly zero there
    mod = fit.modifier(fit.xbar[0].astype(np.float64), mode="global")
    assert np.allclose(mod[0], 0.0)
    pooled = pooled_delta(TARGETS[0], srcs, GENES, shrinkage=False, var_floor="poisson")
    assert np.isfinite(pooled).all()


def test_a_planted_slope_is_recovered_and_kept_by_the_prior():
    rng = np.random.default_rng(1)
    slopes = rng.uniform(0.2, 0.5, size=len(TARGETS))
    fit = fit_basal_slopes(TARGETS, _sources(rng, slopes), GENES, min_common=10)
    assert (fit.n_lines >= 2).all()
    # every gene of every target carries the planted slope, in log1p-CPM units of the
    # rescaled basal (the rescaling is a per-line constant, so the slope survives it)
    assert np.allclose(fit.gamma_hat, slopes[:, None], atol=0.02)
    assert fit.tau2_global > 0
    assert (fit.tau2 > 0).all()
    f = fit.shrink("gene")
    assert f.min() > 0.8 and np.median(f) > 0.99


def test_the_shared_slope_is_the_pooled_per_target_slope():
    rng = np.random.default_rng(11)
    slopes = rng.uniform(0.2, 0.5, size=len(TARGETS))
    fit = fit_basal_slopes(TARGETS, _sources(rng, slopes), GENES, min_common=10)
    assert np.isfinite(fit.se2_shared).all()
    # one slope per gene, pooled over the targets: lands inside the planted range and near
    # its mean, and the prior keeps it (its spread across genes is real, not noise)
    assert fit.gamma_shared.min() > 0.15 and fit.gamma_shared.max() < 0.55
    assert abs(fit.gamma_shared.mean() - slopes.mean()) < 0.05
    x_t = target_basal(rng.uniform(100, 2000, size=len(GENES)), GENES, fit.common)
    mod = fit.modifier(x_t, mode="shared")
    assert mod.shape == (len(TARGETS), len(GENES)) and np.isfinite(mod).all()
    assert fit.stats("shared")["shared_slope_fitted_genes"] == len(GENES)


def test_a_line_on_another_gene_universe_changes_nothing():
    rng = np.random.default_rng(2)
    slopes = rng.uniform(0.2, 0.5, size=len(TARGETS))
    a = fit_basal_slopes(TARGETS, _sources(np.random.default_rng(7), slopes), GENES, min_common=10)
    b = fit_basal_slopes(TARGETS, _sources(np.random.default_rng(7), slopes, scales=(1.0, 4.0, 0.25)), GENES, min_common=10)
    # the pool's 1-CPM pseudocount is the only thing that still reads the universe: at a
    # quarter of the CPM (25-500 here) it moves a fold change by up to log2(1 + 1/25) = 0.06
    assert np.allclose(a.gamma_hat, b.gamma_hat, atol=0.06)
    assert np.abs(a.gamma_hat - b.gamma_hat).mean() < 0.02
    # each line's basal footing is unchanged; only the pool's weights (which read the CPM
    # scale through the Poisson floor) move, so the WEIGHTED mean basal may shift a little
    assert np.allclose(a.line_basal, b.line_basal, atol=1e-6, equal_nan=True)


def test_no_dependence_means_no_modifier():
    rng = np.random.default_rng(3)
    # noise at a third of the pool's own standard error (~0.014 log2 at these depths): the
    # method-of-moments prior variance comes out at exactly 0
    fit = fit_basal_slopes(TARGETS, _sources(rng, np.zeros(len(TARGETS)), noise=0.004), GENES, min_common=10)
    assert fit.tau2_global == 0.0
    x_t = target_basal(rng.uniform(100, 2000, size=len(GENES)), GENES, fit.common)
    assert np.abs(fit.modifier(x_t, mode="global")).max() == 0.0
    assert np.abs(fit.modifier(x_t, mode="gene")).max() < 1e-3
    # and no noise at all: the slopes are exactly zero, so the modifier is exactly zero
    exact = fit_basal_slopes(TARGETS, _sources(rng, np.zeros(len(TARGETS))), GENES, min_common=10)
    assert np.abs(exact.modifier(x_t, mode="gene")).max() < 1e-6


def test_a_gene_one_line_lacks_is_fitted_on_the_others_only():
    rng = np.random.default_rng(4)
    slopes = np.full(len(TARGETS), 0.3)
    srcs = _sources(rng, slopes)
    pb, ctrl = srcs[0]
    keep = np.ones(len(GENES), dtype=bool); keep[5] = False      # drop g5 from line 0
    short = PseudobulkSums(labels=pb.labels, genes=pb.genes[keep], count_sum=pb.count_sum[:, keep],
                           cpm_sum=pb.cpm_sum[:, keep], cpm_sq_sum=pb.cpm_sq_sum[:, keep],
                           n_cells=pb.n_cells, libsize_sum=pb.libsize_sum, sources=pb.sources)
    fit = fit_basal_slopes(TARGETS, [(short, ctrl)] + srcs[1:], GENES, min_common=10)
    assert fit.n_lines[0, 5] == 2 and fit.n_lines[0, 6] == 3
    assert not fit.common[5]


def test_fewer_than_two_profile_sources_is_refused():
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError):
        fit_basal_slopes(TARGETS, _sources(rng, np.zeros(len(TARGETS)))[:1], GENES, min_common=10)


def test_modifier_is_zero_off_the_fitted_pairs_and_finite_everywhere():
    rng = np.random.default_rng(6)
    slopes = np.full(len(TARGETS), 0.3)
    fit = fit_basal_slopes(TARGETS + ["never_measured"], _sources(rng, slopes), GENES, min_common=10)
    x_t = target_basal(rng.uniform(100, 2000, size=len(GENES)), GENES, fit.common)
    mod = fit.modifier(x_t)
    assert np.isfinite(mod).all()
    assert np.all(mod[-1] == 0.0)          # the unmeasured target has no fit
    assert np.abs(mod[:-1]).mean() > 0


def test_loco_off_is_bit_identical_and_the_knob_is_recorded(monkeypatch, tmp_path):
    """`--basal-slope off` never touches the fit; a mode is recorded in the build info."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    from sidechain.eval import loco

    rng = np.random.default_rng(8)
    slopes = np.full(len(TARGETS), 0.3)
    srcs = _sources(rng, slopes)
    n_ctrl, n_pert = 30, 8
    basal = rng.uniform(100, 2000, size=len(GENES))
    rows, labels = [], []
    for _ in range(n_ctrl):
        rows.append(rng.poisson(basal * 40 / basal.sum() * 1000)); labels.append("non-targeting")
    for t in TARGETS[:3]:
        for _ in range(n_pert):
            rows.append(rng.poisson(basal * 40 / basal.sum() * 1000)); labels.append(t)
    real = ad.AnnData(X=sp.csr_matrix(np.asarray(rows, dtype=np.float32)),
                      obs=pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(len(rows))]),
                      var=pd.DataFrame(index=GENES.astype(str)))
    real_path = tmp_path / "real.h5ad"
    real.write_h5ad(real_path)
    calls = []
    monkeypatch.setattr(loco, "fit_basal_slopes", lambda *a, **k: calls.append(1) or fit_basal_slopes(*a, min_common=10, **k))
    off = loco.build_transfer_prediction(real_path, srcs, tmp_path / "off.h5ad", pert_col="perturbation",
                                         control="non-targeting", shrinkage=False, var_floor="poisson",
                                         min_libsize=0.0)
    assert calls == [] and off["basal_slope"] == "off" and off["basal_slope_stats"] is None
    on = loco.build_transfer_prediction(real_path, srcs, tmp_path / "on.h5ad", pert_col="perturbation",
                                        control="non-targeting", shrinkage=False, var_floor="poisson",
                                        basal_slope="gene", min_libsize=0.0)
    assert calls == [1] and on["basal_slope"] == "gene"
    assert on["basal_slope_stats"]["fitted_pairs_frac"] > 0.9
    a = ad.read_h5ad(tmp_path / "off.h5ad"); b = ad.read_h5ad(tmp_path / "on.h5ad")
    assert a.shape == b.shape
    assert (a.X != b.X).nnz > 0            # the modifier moved the emitted cells
    with pytest.raises(SystemExit):
        loco.build_transfer_prediction(real_path, srcs, tmp_path / "bad.h5ad", pert_col="perturbation",
                                       control="non-targeting", basal_slope="sideways", min_libsize=0.0)
