import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sidechain.models.count_emitters import (
    ContextProfile,
    PoissonEmitter,
    log2fc_from_cpm,
    remap_to_axis,
)


def _controls(tmp_path, rng, n=200, g=40):
    base = rng.gamma(0.5, 2.0, size=g)
    X = rng.poisson(base[None, :] * rng.uniform(5, 15, size=(n, 1)))
    a = ad.AnnData(X=sp.csr_matrix(X.astype(np.float32)),
                   obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                   var=pd.DataFrame(index=[f"g{j}" for j in range(g)]))
    p = tmp_path / "ctrl.h5ad"
    a.write_h5ad(p)
    return p, X


def test_profile_and_emission_are_integral_and_scaled(tmp_path):
    rng = np.random.default_rng(0)
    p, _X = _controls(tmp_path, rng)
    prof = ContextProfile.from_controls(p, "A")
    assert abs(prof.fraction.sum() - 1) < 1e-9 and prof.n_cells == 200
    em = PoissonEmitter(prof, seed=1)
    M = em.emit(400)
    assert M.shape == (400, 40) and sp.isspmatrix_csr(M)
    d = M.data
    assert np.array_equal(d, np.round(d)) and d.min() >= 1
    lib = np.asarray(M.sum(axis=1)).ravel()
    assert abs(np.median(lib) / np.median(prof.libsizes) - 1) < 0.15


def test_log2fc_shift_moves_the_right_gene(tmp_path):
    rng = np.random.default_rng(2)
    p, _ = _controls(tmp_path, rng)
    prof = ContextProfile.from_controls(p, "A")
    em = PoissonEmitter(prof, seed=3)
    g = int(np.argmax(prof.fraction))
    shift = np.zeros(40); shift[g] = -3.0  # an 8x knockdown of the top gene
    base = np.asarray(em.emit(2000).mean(axis=0)).ravel()
    kd = np.asarray(em.emit(2000, shift).mean(axis=0)).ravel()
    assert kd[g] < 0.2 * base[g]


def test_log2fc_and_remap():
    fc = log2fc_from_cpm(np.array([7.0, 0.0]), np.array([3.0, 0.0]))
    assert np.isclose(fc[0], 1.0) and fc[1] == 0.0
    out = remap_to_axis(np.array([1.0, 2.0]), np.array(["b", "a"]), np.array(["a", "b", "c"]))
    assert out.tolist() == [2.0, 1.0, 0.0]


def test_even_mode_has_exact_means_and_minimal_spread(tmp_path):
    rng = np.random.default_rng(5)
    p, _X = _controls(tmp_path, rng)
    prof = ContextProfile.from_controls(p, "A")
    em = PoissonEmitter(prof, seed=6, dispersion="even")
    n = 400
    M = em.emit(n).toarray()
    assert np.array_equal(M, np.round(M)) and M.min() >= 0
    lib = M.sum(axis=1)
    assert lib.max() - lib.min() <= 40          # same depth for every cell, up to the remainders
    tot = M.sum(axis=0)
    expected = np.rint(n * np.median(prof.libsizes) * prof.fraction)
    np.testing.assert_array_equal(tot, expected)  # per-gene totals exact
    assert (M.max(axis=0) - M.min(axis=0)).max() <= 1  # spread of at most one count per gene
