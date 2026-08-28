import anndata as ad
import numpy as np
import pandas as pd
import pytest
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


# ------------------------------------------------- the emission-sharpening dial --


def _flat_controls(tmp_path, rng, n=300, g=30, depth=2000.0):
    """Controls at near-constant depth, so per-gene spread isolates the emission
    dial from library-size variation."""
    base = rng.gamma(2.0, 1.0, size=g)
    frac = base / base.sum()
    X = rng.poisson(depth * frac[None, :], size=(n, g))
    a = ad.AnnData(X=sp.csr_matrix(X.astype(np.float32)),
                   obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                   var=pd.DataFrame(index=[f"g{j}" for j in range(g)]))
    p = tmp_path / "flat_ctrl.h5ad"
    a.write_h5ad(p)
    return p


def test_the_lam_endpoints_are_the_two_modes_bit_for_bit(tmp_path):
    """even is lam=0 and poisson is lam=1 by construction, so a lam=0 control
    arm reproduces a shipped even-mode emission exactly, RNG stream included."""
    rng = np.random.default_rng(11)
    p, _ = _controls(tmp_path, rng)
    prof = ContextProfile.from_controls(p, "A")
    for mode, lam in (("even", 0.0), ("poisson", 1.0)):
        a = PoissonEmitter(prof, seed=7, dispersion=mode).emit(60)
        b = PoissonEmitter(prof, seed=7, lam=lam).emit(60)
        assert (a != b).nnz == 0, mode


def test_lam_beside_dispersion_or_outside_the_unit_interval_is_refused(tmp_path):
    """The two knobs are one dial; accepting both would let them disagree
    silently, and crash-to-wrong is the bad direction."""
    rng = np.random.default_rng(12)
    p, _ = _controls(tmp_path, rng)
    prof = ContextProfile.from_controls(p, "A")
    with pytest.raises(ValueError, match="not both"):
        PoissonEmitter(prof, dispersion="even", lam=0.5)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            PoissonEmitter(prof, lam=bad)


def test_an_interior_lam_keeps_the_means_and_scales_the_spread(tmp_path):
    """The dial's CONDITIONAL contract, isolated on a flat-depth pool
    (mean ~= median, CV(lib) ~2%): counts stay integral and non-negative,
    per-gene means sit at the profile, and cell-to-cell sd is ~lam times the
    Poisson sd on well-expressed genes. On pools with real depth spread the
    marginal law differs -- the class docstring carries it -- so this fixture
    is flat BY DESIGN and a skewed pool must not be swapped in."""
    rng = np.random.default_rng(13)
    p = _flat_controls(tmp_path, rng)
    prof = ContextProfile.from_controls(p, "A")
    n = 3000
    expected = np.median(prof.libsizes) * prof.fraction
    deep = expected >= 50      # where the Poisson part dwarfs the +-1 remainder noise
    assert deep.sum() >= 5
    sds = {}
    for lam in (0.15, 0.3, 0.7):
        M = PoissonEmitter(prof, seed=17, lam=lam).emit(n).toarray()
        assert np.array_equal(M, np.round(M)) and M.min() >= 0
        mu = M.mean(axis=0)
        np.testing.assert_allclose(mu[deep], expected[deep], rtol=0.05)
        sds[lam] = M.std(axis=0)
        np.testing.assert_allclose(sds[lam][deep], lam * np.sqrt(mu[deep]), rtol=0.15)
    assert (sds[0.15][deep] < sds[0.3][deep]).all()
    assert (sds[0.3][deep] < sds[0.7][deep]).all()


def test_an_interior_lam_carries_the_shift_in_both_components(tmp_path):
    """At lam=0.3 the even share holds 91% of the depth, so an 8x knockdown
    must survive emission essentially in full. A mixture whose even component
    were built from the UNSHIFTED profile (the mutation-test survivor this
    test was added to kill) would leave the gene at ~92% of baseline."""
    rng = np.random.default_rng(21)
    p, _ = _controls(tmp_path, rng)
    prof = ContextProfile.from_controls(p, "A")
    g = int(np.argmax(prof.fraction))
    shift = np.zeros(40); shift[g] = -3.0
    em = PoissonEmitter(prof, seed=22, lam=0.3)
    base = np.asarray(em.emit(2000).mean(axis=0)).ravel()
    kd = np.asarray(em.emit(2000, shift).mean(axis=0)).ravel()
    assert kd[g] < 0.2 * base[g]


def test_loco_passes_emit_lambda_through_and_records_it(monkeypatch, tmp_path):
    """The flag's wiring, with the heavy stages stubbed: --emit-lambda must
    reach build_transfer_prediction (deleting the pass-through made it a
    silent no-op under the full suite) and must land in the log_run payload;
    with neither flag given, the CLI's historical default 'even' must arrive."""
    from sidechain.data.stream_pseudobulk import PseudobulkSums
    from sidechain.eval import loco

    monkeypatch.setattr(PseudobulkSums, "load", classmethod(lambda cls, p: f"PB:{p}"))
    captured, logged = {}, {}

    def fake_build(real, sources, out_path, **kw):
        captured.update(kw)
        return {}

    monkeypatch.setattr(loco, "build_transfer_prediction", fake_build)
    monkeypatch.setattr(loco, "attach_controls", lambda pred, real, out, **kw: out)
    monkeypatch.setattr(loco, "score", lambda *a, **kw: {"overall": 0.0, "members": {}})
    monkeypatch.setattr(loco, "log_run", lambda params, results, artifacts=None: logged.update(params))

    base = ["--real", "r.h5ad", "--bundle", "b", "--source", "s.npz:ctl"]
    rc = loco.main(base + ["--out", str(tmp_path / "arm"), "--emit-lambda", "0.35"])
    assert rc == 0
    assert captured["emit_lambda"] == 0.35 and captured["dispersion"] is None
    assert logged["emit_lambda"] == 0.35

    captured.clear()
    rc = loco.main(base + ["--out", str(tmp_path / "arm2")])
    assert rc == 0
    assert captured["emit_lambda"] is None and captured["dispersion"] == "even"


def test_submit_build_passes_emit_lambda_to_the_emitter(monkeypatch, tmp_path):
    """The submission side of the same wiring, on a 3-gene, 2-perturbation
    toy config. This one has the worse silent failure mode: losing the
    `lam=` pass-through leaves dispersion=None and the constructor's
    historical default is POISSON, not the CLI's 'even'."""
    import yaml

    from sidechain.submit import build as submit_build

    genes = ["gA", "gB", "gC"]
    rng = np.random.default_rng(31)
    X = rng.poisson(50.0, size=(30, 3)) + 1
    ctrl = ad.AnnData(X=sp.csr_matrix(X.astype(np.float32)),
                      obs=pd.DataFrame(index=[f"c{i}" for i in range(30)]),
                      var=pd.DataFrame(index=genes))
    ctrl.write_h5ad(tmp_path / "ctrl_A.h5ad")
    pd.DataFrame({"gene_name": genes}).to_csv(tmp_path / "gene_names.csv", index=False)
    pd.DataFrame({"target_gene": ["gA", "gB"], "n_cells": [4, 4]}).to_csv(
        tmp_path / "pert_counts.csv", index=False)
    cfg = {"data_dir": str(tmp_path), "gene_names_file": "gene_names.csv", "n_genes": 3,
           "pert_counts_file": "pert_counts.csv", "pert_col": "target_gene",
           "context_col": "context", "control_label": "non-targeting",
           "phase": "validation", "phases": {"validation": {"contexts": ["A"]}},
           "control_files": {"A": "ctrl_A.h5ad"},
           "submission": {"cells_per_pert": 4, "max_counts_per_cell": 100000,
                          "max_cells": 100, "max_stored_entries": 10000}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))

    captured = {}
    real_cls = submit_build.PoissonEmitter

    class Spy(real_cls):
        def __init__(self, *a, **kw):
            captured.update(kw)
            super().__init__(*a, **kw)

    monkeypatch.setattr(submit_build, "PoissonEmitter", Spy)
    rc = submit_build.main(["--challenge-config", str(tmp_path / "cfg.yaml"),
                            "--emitter", "control-null", "--out", str(tmp_path / "toy_probe"),
                            "--no-pack", "--min-libsize", "0", "--emit-lambda", "0.35"])
    assert rc == 0
    assert captured["lam"] == 0.35 and captured["dispersion"] is None


def test_the_two_entry_points_refuse_dispersion_beside_emit_lambda():
    """Both CLIs resolve the pair before touching any file, so the refusal is
    testable without data on disk -- and the flag really exists on both, which
    is what lets a mirror-scored lam arm submit verbatim."""
    from sidechain.eval import loco
    from sidechain.submit import build as submit_build
    with pytest.raises(SystemExit):
        loco.main(["--real", "x.h5ad", "--bundle", "b", "--out", "lam_probe",
                   "--source", "s.npz:control", "--dispersion", "even", "--emit-lambda", "0.5"])
    with pytest.raises(SystemExit):
        submit_build.main(["--emitter", "control-null", "--out", "lam_probe_v1",
                           "--dispersion", "even", "--emit-lambda", "0.5"])
    # ...and an out-of-range lam dies at the argument parser, not an hour into
    # a build inside the write loop.
    with pytest.raises(SystemExit):
        loco.main(["--real", "x.h5ad", "--bundle", "b", "--out", "lam_probe",
                   "--source", "s.npz:control", "--emit-lambda", "1.5"])
    with pytest.raises(SystemExit):
        submit_build.main(["--emitter", "control-null", "--out", "lam_probe_v1",
                           "--emit-lambda", "-0.1"])


def test_shrinkage_pulls_noisy_genes_more_than_precise_ones():
    from sidechain.submit.build import shrink
    fc = np.array([1.0, 1.0, 0.0, -1.0])
    var = np.array([0.01, 1.0, 0.5, 0.01])
    out = shrink(fc, var)
    assert abs(out[0]) > abs(out[1])            # same effect, noisier estimate shrinks more
    assert out[2] == 0.0 and np.sign(out[3]) == -1
    assert np.all(np.abs(out) <= np.abs(fc))
