"""`sidechain.eval.delta_cache`: the pooled-delta cache that `eval.loco` builds and reads.

    .venv/bin/python -m pytest -q tests/test_delta_cache.py

Writes temporary files only. Notes: private/code-notes/tests/test_delta_cache.py.md
"""
from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.eval import delta_cache as dc, loco
from sidechain.models import adaptive_shrink as ash
from sidechain.models.neighbour_arm import SELECTS

G, N_LAB = 80, 9
IDS = [["src", "0" * 64, "ctrl"]]
KW = dict(pert_col="perturbation", control="non-targeting", var_floor="poisson", emit_lambda=0.4,
          alpha=1.2, alpha_bulk=1.2, bulk_anchor="pooled", min_libsize=0.0,
          shrink_stage="pooled", shrink_rule="adaptive", cells_per_pert=6)


@pytest.fixture
def fold(tmp_path):
    rng = np.random.default_rng(11)
    genes = np.array([f"g{i}" for i in range(G)], dtype=object)
    labs = [f"g{i}" for i in range(N_LAB)]
    basal = rng.uniform(100, 2000, size=G)
    mean = np.stack([basal] + [basal * np.exp2(rng.normal(0, 0.3, G)) for _ in labs])
    n = np.full(len(labs) + 1, 400, dtype=np.int64)
    src = PseudobulkSums(labels=["ctrl", *labs], genes=genes.copy(), count_sum=mean * n[:, None],
                         cpm_sum=mean * n[:, None], cpm_sq_sum=(mean**2 + mean) * n[:, None],
                         n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["t"])
    rows, obs = [], []
    # the fold predicts five of the labels and one the source lacks; the pool holds those five
    # and four more
    for lab, k in [("non-targeting", 40)] + [(x, 8) for x in [*labs[:5], "zz"]]:
        for _ in range(k):
            rows.append(rng.poisson(basal / basal.sum() * rng.integers(3000, 6000)))
            obs.append(lab)
    real = ad.AnnData(X=sp.csr_matrix(np.asarray(rows, dtype=np.float32)),
                      obs=pd.DataFrame({"perturbation": obs}, index=[f"c{i}" for i in range(len(rows))]),
                      var=pd.DataFrame(index=genes.astype(str)))
    real.write_h5ad(tmp_path / "real.h5ad")
    pd.DataFrame({"target_gene": [*labs, "absent"]}).to_csv(tmp_path / "pool.csv", index=False)
    torch.save({x: torch.tensor(rng.normal(size=6), dtype=torch.float32) for x in [*labs, "absent"]},
               tmp_path / "table.pt")
    nb = dict(neighbour_table=[tmp_path / "table.pt"], neighbour_pool=tmp_path / "pool.csv",
              neighbour_k=3, neighbour_w=[0.3])
    return tmp_path, tmp_path / "real.h5ad", [(src, "ctrl")], nb


def _arm(fold, name, **extra):
    tmp, real, sources, nb = fold
    info = loco.build_transfer_prediction(real, sources, tmp / f"{name}.h5ad", **{**KW, **nb, **extra})
    return info, ad.read_h5ad(tmp / f"{name}.h5ad")


def _build(fold, jobs=1, root="cache", **extra):
    tmp, real, sources, nb = fold
    return loco.build_transfer_prediction(
        real, sources, tmp / "unused.h5ad", **{**KW, **nb, **extra},
        delta_cache={"dir": tmp / root, "sources": IDS, "build_jobs": jobs})["delta_cache_built"]


def test_a_cached_arm_is_the_uncached_arm_bit_for_bit(fold):
    tmp = fold[0]
    plain, cells = _arm(fold, "plain")
    built = _build(fold)
    assert built["built"] and built["labels"] == N_LAB + 2 and built["covered"] == N_LAB
    assert not (tmp / "unused.h5ad").exists()                   # a build predicts nothing
    picks = tmp / "picks.json"
    picks.write_text(json.dumps({f"g{i}": {"members": [f"g{(i + j) % N_LAB}" for j in (1, 2, 5)],
                                           "weights": None} for i in range(5)}))
    arms = {s: {"neighbour_select": s, "neighbour_cand": 5} for s in SELECTS}
    arms["picks"] = {"neighbour_picks": picks}
    arms["none"] = {"neighbour_table": None, "neighbour_pool": None, "neighbour_w": None}
    for select, extra in arms.items():                          # a warm read by every kind of arm
        ref, ref_cells = (plain, cells) if select == "table" else _arm(fold, "plain_r", **extra)
        got, got_cells = _arm(fold, f"cached_{select}", **extra,
                              delta_cache={"dir": tmp / "cache", "sources": IDS})
        assert (got_cells.X != ref_cells.X).nnz == 0
        assert list(got_cells.obs["perturbation"]) == list(ref_cells.obs["perturbation"])
        rec = got.pop("delta_cache")
        assert rec["key"] == built["key"]
        assert rec["hits"] == (6 if select == "none" else 6 + N_LAB + 1)
        got.pop("pred"), ref.pop("pred")
        assert json.dumps(got) == json.dumps(ref)               # same keys, same order, same values
        assert got["pool_stats"]["adaptive_fits"] == 5          # the fold's covered targets
        assert got["adaptive_fit"]["neighbour_pool"].get("adaptive_fits") == (
            None if select == "none" else N_LAB)


def test_several_processes_write_the_cache_one_process_writes(fold):
    tmp = fold[0]
    one, two = _build(fold, root="one"), _build(fold, jobs=2, root="two")
    assert one["key"] == two["key"]
    a, b = (np.load(tmp / r / one["key"] / "deltas.npy") for r in ("one", "two"))
    assert np.array_equal(a, b)
    ma, mb = (json.loads((tmp / r / one["key"] / "meta.json").read_text()) for r in ("one", "two"))
    assert ma["row"] == mb["row"] and ma["stats"] == mb["stats"]
    assert ma["row"]["zz"] == -1 and ma["row"]["absent"] == -1  # uncovered labels hold no row
    again = _build(fold, root="one")                            # a second build leaves it alone
    assert not again["built"]


def test_a_read_the_cache_cannot_answer_is_refused(fold):
    tmp, real, sources, nb = fold
    _build(fold)
    use = {"dir": tmp / "cache", "sources": IDS}
    for extra in ({"var_floor": "none"}, {"shrink_rule": "garrote"}, {"log_bias_correct": True},
                  {"shrinkage": False, "shrink_rule": "garrote", "shrink_stage": "source"},
                  {"shrink_rule": "garrote", "shrink_stage": "source"},
                  {"shrink_rule": "garrote", "shrink_stage": "source", "shrink_k": 8.0}):
        with pytest.raises(SystemExit, match="no cache for this fold and these knobs"):
            _arm(fold, "bad", delta_cache=use, **extra)
    with pytest.raises(SystemExit, match="no cache"):           # another source file
        _arm(fold, "bad", delta_cache={"dir": tmp / "cache", "sources": [["src", "1" * 64, "ctrl"]]})
    with pytest.raises(SystemExit, match="no cache"):           # an arm never fills one
        _arm(fold, "bad", delta_cache={"dir": tmp / "empty", "sources": IDS})
    pd.DataFrame({"target_gene": ["g0", "g1", "g2", "g3", "other"]}).to_csv(tmp / "pool2.csv", index=False)
    torch.save({x: torch.tensor([1.0, float(i), 2.0]) for i, x in enumerate(
        ["g0", "g1", "g2", "g3", "g4", "other"])}, tmp / "table2.pt")
    with pytest.raises(SystemExit, match="'other' is not in the cache"):
        _arm(fold, "bad", delta_cache=use, neighbour_pool=tmp / "pool2.csv",
             neighbour_table=[tmp / "table2.pt"])
    with pytest.raises(SystemExit, match="plain pool only"):
        _arm(fold, "bad", delta_cache=use, gamma=0.5)
    with pytest.raises(SystemExit, match="plain pool only"):
        loco.build_transfer_prediction(real, [(sources[0][0], "ctrl", True)], tmp / "bad.h5ad",
                                       **{**KW, **nb}, delta_cache=use)


def test_the_key_moves_with_the_threads_and_the_axis(fold, monkeypatch):
    axis = np.array([f"g{i}" for i in range(G)])
    pooling = {"shrinkage": True}
    base = dc.key_of(dc.key_fields(IDS, axis, pooling))
    assert base == dc.key_of(dc.key_fields(IDS, axis, pooling))
    assert base != dc.key_of(dc.key_fields(IDS, axis[::-1], pooling))
    assert base != dc.key_of(dc.key_fields(IDS, axis, {"shrinkage": False}))
    assert base != dc.key_of(dc.key_fields([["src", "0" * 64, "other"]], axis, pooling))
    with monkeypatch.context() as m:
        m.setenv("OMP_NUM_THREADS", "7")
        assert base != dc.key_of(dc.key_fields(IDS, axis, pooling))
    for name in ("MAX_CYCLES", "TOL_PER_GENE", "CALM_CYCLES", "MIN_GENES"):
        with monkeypatch.context() as m:                        # the rule's constants
            m.setattr(ash, name, getattr(ash, name) * 2)
            assert base != dc.key_of(dc.key_fields(IDS, axis, pooling))
    with monkeypatch.context() as m:
        m.setattr(np, "__version__", "0.0")
        assert base != dc.key_of(dc.key_fields(IDS, axis, pooling))
    with monkeypatch.context() as m:                            # another machine
        m.setattr(dc.platform, "node", lambda: "elsewhere")
        assert base != dc.key_of(dc.key_fields(IDS, axis, pooling))
    # the code that does the arithmetic: these six files, and an edit to any moves the key
    assert set(dc.code_digests()) == {
        "sidechain.submit.build", "sidechain.models.adaptive_shrink",
        "sidechain.models.count_emitters", "sidechain.data.stream_pseudobulk",
        "sidechain.submit.variance_model", "sidechain.data.dispersion"}
    for mod in dc.CODE_MODULES:
        with monkeypatch.context() as m:
            real = dc.code_digests()
            m.setattr(dc, "code_digests", lambda real=real, mod=mod: {**real, mod: "edited"})
            assert base != dc.key_of(dc.key_fields(IDS, axis, pooling))
    assert base == dc.key_of(dc.key_fields(IDS, axis, pooling))


def test_the_code_digest_moves_with_the_code_and_never_with_its_text(tmp_path):
    mod = tmp_path / "m.py"
    mod.write_text('"""A module."""\nimport argparse\n\n\ndef f(x):\n    """What f does."""\n'
                   '    return x + 1  # a comment\n\n\nap = argparse.ArgumentParser(description=__doc__)\n'
                   'ap.add_argument("--n", type=int, help="how many")\n')
    base = dc.code_sha256(mod)
    mod.write_text('"""Another docstring,\n\nlonger, with a usage line."""\nimport argparse\n\n# a new comment\n'
                   'def f(x):\n    return x + 1\n\n\nap = argparse.ArgumentParser(description=__doc__)\n'
                   'ap.add_argument("--n", type=int, help="the count")\n')
    assert dc.code_sha256(mod) == base                      # docstrings, comments, help text, line numbers
    mod.write_text('"""A module."""\nimport argparse\n\n\ndef f(x):\n    """What f does."""\n'
                   '    return x + 2  # a comment\n\n\nap = argparse.ArgumentParser(description=__doc__)\n'
                   'ap.add_argument("--n", type=int, help="how many")\n')
    assert dc.code_sha256(mod) != base                      # the arithmetic
    mod.write_text('"""A module."""\nimport argparse\n\n\ndef f(x):\n    """What f does."""\n'
                   '    return x + 1  # a comment\n\n\nap = argparse.ArgumentParser(description=__doc__)\n'
                   'ap.add_argument("--n", type=int, default=3, help="how many")\n')
    assert dc.code_sha256(mod) != base                      # a default is code


def test_source_ids_read_the_file_the_label_and_the_order(tmp_path):
    (tmp_path / "a.npz").write_bytes(b"one")
    (tmp_path / "b.npz").write_bytes(b"two")
    a, b = f"{tmp_path / 'a.npz'}:ctrl", f"{tmp_path / 'b.npz'}:Non-Targeting"
    ids = dc.source_ids([a, b])
    assert ids == [["a", dc.file_sha256(tmp_path / "a.npz"), "ctrl"],
                   ["b", dc.file_sha256(tmp_path / "b.npz"), "Non-Targeting"]]
    assert dc.source_ids([b, a]) == ids[::-1] != ids
    assert dc.source_ids([f"{tmp_path / 'a.npz'}:other"])[0][2] == "other"
    assert dc.source_ids([str(tmp_path / "a.npz") + ":"])[0][2] == "control"   # loco's default
    (tmp_path / "a.npz").write_bytes(b"changed")
    assert dc.source_ids([a])[0][1] != ids[0][1]


def test_a_read_is_a_copy_and_the_main_entry_builds_what_an_arm_reads(fold, capsys):
    tmp, real, sources, nb = fold
    sources[0][0].save(tmp / "src.npz")
    spec = f"{tmp / 'src.npz'}:ctrl"
    argv = ["--real", str(real), "--pert-col", "perturbation", "--control", "non-targeting",
            "--source", spec, "--bundle", str(tmp), "--out", str(tmp / "never"),
            "--shrink-stage", "pooled", "--shrink-rule", "adaptive", "--var-floor", "poisson",
            "--neighbour-table", str(nb["neighbour_table"][0]), "--neighbour-w", "0.3",
            "--neighbour-pool", str(nb["neighbour_pool"]), "--neighbour-k", "3",
            "--delta-cache", str(tmp / "cli"), "--delta-cache-build", "--delta-cache-jobs", "2"]
    assert loco.main(argv) == 0
    built = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["delta_cache_built"]
    assert built["built"] and built["covered"] == N_LAB and not (tmp / "never").exists()
    use = {"dir": tmp / "cli", "sources": dc.source_ids([spec])}
    got, got_cells = _arm(fold, "cli_arm", delta_cache=use)
    ref, ref_cells = _arm(fold, "cli_ref")
    assert (got_cells.X != ref_cells.X).nnz == 0 and got["delta_cache"]["key"] == built["key"]
    axis = np.array([f"g{i}" for i in range(G)])
    cache = dc.DeltaCache(tmp / "cli", dc.key_fields(use["sources"], axis, {
        "shrinkage": True, "shrink_k": 1.0, "shrink_stage": "pooled", "shrink_rule": "adaptive",
        "var_floor": "poisson", "log_bias_correct": False,
        # the variance model and, per source, the fit it carries (none here)
        "variance_model": "shipped", "rule_variance": "model", "dispersion_fits": [None]}))
    one, two = cache.get("g0"), cache.get("g0")
    assert one is not two and not np.shares_memory(one, two) and one.flags.writeable
    one[:] = 0.0
    assert np.array_equal(two, cache.get("g0")) and two.any()
    assert cache.get("zz") is None


def test_a_cache_built_under_one_variance_model_is_refused_by_every_other(fold):
    """The variance model, `--rule-variance` and the fit are in the key: another of any finds no cache."""
    from sidechain.data.dispersion import fit_gene_dispersion
    from sidechain.submit.variance_model import VarianceModel

    tmp, real, sources, nb = fold
    src = sources[0][0]
    src.sidechain_name = "src"
    src.dispersion_fit = fit_gene_dispersion(src)
    src.dispersion_fit_sha256 = "ab" * 32
    vm = VarianceModel.parse("trend")
    built = _build(fold, root="vcache", variance_model=vm, rule_variance="model")
    assert built["built"]
    use = {"dir": tmp / "vcache", "sources": IDS}
    got, _ = _arm(fold, "trend_arm", delta_cache=use, variance_model=vm, rule_variance="model")   # the same arm reads it
    assert got["delta_cache"]["key"] == built["key"]
    for other in ({"variance_model": None, "rule_variance": "model"},                              # shipped
                  {"variance_model": VarianceModel.parse("trend:shuffle=1"), "rule_variance": "model"},
                  {"variance_model": vm, "rule_variance": "shipped"},
                  {"variance_model": VarianceModel.parse("flat"), "rule_variance": "model"}):
        with pytest.raises(SystemExit, match="no cache for this fold and these knobs"):
            _arm(fold, "refused", delta_cache=use, **other)
    src.dispersion_fit_sha256 = "cd" * 32                                                            # another fit, same spec
    with pytest.raises(SystemExit, match="no cache for this fold and these knobs"):
        _arm(fold, "refused_fit", delta_cache=use, variance_model=vm, rule_variance="model")
    # the category list's CONTENTS are in the key (its sha256 rides in identity()): the same path with other
    # genes in it is another model, so an edited list never reads a cache built for the old one
    src.dispersion_fit_sha256 = "ab" * 32
    genes = [str(g) for g in src.genes]
    lst = tmp / "cat.txt"
    lst.write_text("\n".join(genes[: len(genes) // 2]) + "\n")
    cat = VarianceModel.parse(f"category:list={lst}")
    built_c = _build(fold, root="ccache", variance_model=cat, rule_variance="model")
    use_c = {"dir": tmp / "ccache", "sources": IDS}
    got_c, _ = _arm(fold, "cat_arm", delta_cache=use_c, variance_model=cat, rule_variance="model")
    assert got_c["delta_cache"]["key"] == built_c["key"]
    lst.write_text("\n".join(genes[: len(genes) // 2 + 1]) + "\n")
    with pytest.raises(SystemExit, match="no cache for this fold and these knobs"):
        _arm(fold, "cat_refused", delta_cache=use_c, variance_model=VarianceModel.parse(f"category:list={lst}"),
             rule_variance="model")


def test_the_flags_are_refused_where_they_cannot_act(fold, capsys):
    tmp, real, _sources, _nb = fold
    argv = ["--real", str(real), "--source", "x.npz:ctrl", "--bundle", str(tmp), "--out", str(tmp / "o")]
    for extra, msg in ((["--delta-cache-build"], "needs --delta-cache DIR"),
                       (["--delta-cache", str(tmp), "--delta-cache-jobs", "2"], "only acts with"),
                       (["--delta-cache", str(tmp), "--transfer-floor", "x=0.01"], "plain pool only")):
        with pytest.raises(SystemExit):
            loco.main(argv + extra)
        assert msg in capsys.readouterr().err


def _lfc_table(seed=3):
    """A published-contrast source on the fold's genes for six of its labels, abstaining on a few cells."""
    from sidechain.data.lfc_table import LfcTable

    rng = np.random.default_rng(seed)
    labs = [f"g{i}" for i in range(3, N_LAB)]
    lfc = rng.normal(0, 0.4, size=(len(labs), G))
    var = rng.uniform(0.02, 0.2, size=(len(labs), G))
    var[rng.random(var.shape) < 0.1] = np.inf
    tab = LfcTable(labels=labs, genes=np.array([f"g{i}" for i in range(G)]), lfc=lfc, var=var, source="tab")
    tab.sidechain_name = "tab"
    return tab


def test_a_published_contrast_source_is_cached_bit_for_bit(fold):
    """A cached arm with an `--lfc-source` table is the uncached arm; a pooled table moves the key."""
    tmp, real, sources, nb = fold
    tab = _lfc_table()
    with_tab = [*sources, tab]
    ids = [*IDS, ["tab", "1" * 64, dc.LFC_TABLE]]
    ref = loco.build_transfer_prediction(real, with_tab, tmp / "tab_plain.h5ad", **{**KW, **nb})
    built = loco.build_transfer_prediction(real, with_tab, tmp / "unused.h5ad", **{**KW, **nb},
                                           delta_cache={"dir": tmp / "tcache", "sources": ids,
                                                        "build_jobs": 1})["delta_cache_built"]
    got = loco.build_transfer_prediction(real, with_tab, tmp / "tab_cached.h5ad", **{**KW, **nb},
                                         delta_cache={"dir": tmp / "tcache", "sources": ids})
    a, b = ad.read_h5ad(tmp / "tab_plain.h5ad"), ad.read_h5ad(tmp / "tab_cached.h5ad")
    assert (a.X != b.X).nnz == 0
    assert got.pop("delta_cache")["key"] == built["key"]
    got.pop("pred"), ref.pop("pred")
    assert json.dumps(got) == json.dumps(ref)
    axis = np.array([f"g{i}" for i in range(G)])
    with_t, without = dc.key_fields(ids, axis, {}), dc.key_fields(IDS, axis, {})
    assert dc.LFC_MODULE in with_t["code"] and dc.LFC_MODULE not in without["code"]
    assert dc.key_of(with_t) != dc.key_of(without)
    # the pool without the table cannot read the table's cache
    with pytest.raises(SystemExit, match="no cache for this fold and these knobs"):
        loco.build_transfer_prediction(real, sources, tmp / "x.h5ad", **{**KW, **nb},
                                       delta_cache={"dir": tmp / "tcache", "sources": IDS})


def test_lfc_source_ids_follow_the_source_ids(tmp_path):
    a, t = tmp_path / "a.npz", tmp_path / "t.npz"
    a.write_bytes(b"a"), t.write_bytes(b"t")
    ids = dc.source_ids([f"{a}:ctrl"], [str(t)])
    assert [x[0] for x in ids] == ["a", "t"] and ids[1][2] == dc.LFC_TABLE
    assert ids[1][1] == dc.file_sha256(t)
