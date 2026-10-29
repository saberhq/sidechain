"""The 2025 local mirror (`eval.local_mirror`): metric names, config precedence, paths, the guardrail.

    .venv/bin/python -m pytest -q tests/test_eval.py

Writes temporary files only. Notes: private/code-notes/tests/test_eval.py.md
"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

from sidechain.eval import local_mirror as lm
from sidechain.utils.paths import find_repo_root, resolve_config

# --------------------------------------------------------- metric mapping --


def test_challenge_metrics_map_to_the_cell_eval_registry_names():
    """DES/PDS/MAE map to cell-eval's registry names."""
    assert lm.CHALLENGE_TO_CELL_EVAL == {
        "des": "overlap_at_N",
        "pds": "discrimination_score_l1",
        "mae": "mae",
    }


def test_mapping_matches_cell_evals_own_vcc_profile():
    """The mapped names are cell-eval's 'vcc' profile."""
    from cell_eval._pipeline._runner import VCC_METRICS

    assert set(lm.CHALLENGE_TO_CELL_EVAL.values()) == set(VCC_METRICS)


def test_extras_profile_is_the_seven_metric_generalist_set():
    from cell_eval._pipeline._runner import MINIMAL_METRICS, VCC_METRICS

    assert len(MINIMAL_METRICS) == 7
    assert set(VCC_METRICS) <= set(MINIMAL_METRICS)


def test_mae_is_the_only_lower_is_better_challenge_metric():
    dirs = {m: lm.HIGHER_IS_BETTER[c] for m, c in lm.CHALLENGE_TO_CELL_EVAL.items()}
    assert dirs == {"des": True, "pds": True, "mae": False}


# ------------------------------------------------------- config precedence --


def test_challenge_config_overrides_eval_config():
    cfg = {
        "cell_eval": {"metrics": ["des", "pds", "mae"], "extra_state_metrics": True},
        "_challenge": {"metrics": ["mae"], "extra_state_metrics": False},
    }
    names, profile = lm.resolve_metrics(cfg)
    assert names == ["mae"]
    assert profile == "vcc", "extras off -> the 3-metric profile"


def test_eval_config_is_the_fallback_when_no_challenge_config():
    cfg = {"cell_eval": {"metrics": ["des", "pds"], "extra_state_metrics": True}, "_challenge": {}}
    names, profile = lm.resolve_metrics(cfg)
    assert names == ["overlap_at_N", "discrimination_score_l1"]
    assert profile == "minimal", "extras on -> the 7-metric profile"


def test_unknown_metric_name_raises():
    with pytest.raises(ValueError, match="Unknown challenge metrics"):
        lm.resolve_metrics({"_challenge": {"metrics": ["des", "auroc"]}})


def test_real_configs_load_and_resolve():
    cfg = lm.load_configs("configs/eval.yaml", "challenges/vcc2025/config.yaml")
    names, profile = lm.resolve_metrics(cfg)
    assert set(names) == set(lm.CHALLENGE_TO_CELL_EVAL.values())
    assert profile == "minimal"
    assert cfg["_challenge"]["control_label"] == "non-targeting"
    assert cfg["_challenge"]["pert_col"] == "target_gene"


def test_metrics_live_in_the_challenge_config_not_the_core():
    """The metric list lives in the challenge config."""
    import yaml

    challenge = yaml.safe_load(resolve_config("challenges/vcc2025/config.yaml").read_text())
    assert "metrics" in challenge and "extra_state_metrics" in challenge


# ------------------------------------------------------- path independence --


def test_config_resolves_from_an_unrelated_cwd(tmp_path, monkeypatch):
    """A config path resolves from a working directory outside the repo."""
    monkeypatch.chdir(tmp_path)
    assert resolve_config("configs/eval.yaml").is_file()


def test_missing_config_names_every_location_tried(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Tried:"):
        resolve_config("configs/does_not_exist.yaml")


def test_find_repo_root_locates_the_project():
    root = find_repo_root()
    assert root is not None and (root / "pyproject.toml").exists()


# ------------------------------------------------------------- guardrails --


def _pair(scale: float, n=200, g=25, seed=0):
    rng = np.random.default_rng(seed)
    def build(mult):
        base = rng.normal(0, 1, size=(n, g)) * mult
        obs = pd.DataFrame({"target_gene": ["p"] * (n // 2) + ["non-targeting"] * (n // 2)})
        obs.index = [f"c{i}" for i in range(n)]
        return ad.AnnData(X=base.astype(np.float32), obs=obs)
    return build(scale), build(1.0)


def test_variance_inflation_flags_an_inflated_prediction():
    pred, real = _pair(scale=4.0)
    out = lm.variance_inflation_check(pred, real, "target_gene", "non-targeting")
    assert out["flagged"] is True
    assert out["variance_ratio"] > 1.5


def test_variance_inflation_passes_a_well_scaled_prediction():
    pred, real = _pair(scale=1.0)
    out = lm.variance_inflation_check(pred, real, "target_gene", "non-targeting")
    assert out["flagged"] is False
    assert 0.5 < out["variance_ratio"] < 1.5


def test_similarity_beta_reaches_the_pool_with_a_control_profile_at_default_gamma(
        monkeypatch, tmp_path):
    """At the default gamma a non-zero `similarity_beta` reaches the pool with a control profile."""
    import anndata as ad
    import scipy.sparse as sp

    from sidechain.eval import loco

    genes = [f"G{i}" for i in range(60)]
    rng = np.random.default_rng(0)
    # libsize must clear loco's min_libsize=500 filter, or the control set empties
    X = sp.csr_matrix(rng.poisson(20, size=(80, 60)).astype(np.float32))
    obs = pd.DataFrame({"perturbation": ["non-targeting"] * 40 + ["T1"] * 40})
    real = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    real_path = tmp_path / "real.h5ad"
    real.write_h5ad(real_path)

    seen = {}

    def fake_pool(target, sources, axis, **kw):
        seen.update(kw)
        return np.zeros(len(axis))

    monkeypatch.setattr(loco, "pooled_delta", fake_pool)
    loco.build_transfer_prediction(
        real_path, [], tmp_path / "pred.h5ad",
        pert_col="perturbation", control="non-targeting",
        similarity_beta=20.0,          # gamma stays at its default 1.0
    )
    assert seen["similarity_beta"] == 20.0
    assert seen["ctrl_tgt_cpm"] is not None, (
        "loco must build the control profile for similarity_beta, not only for gamma"
    )
    assert len(seen["ctrl_tgt_cpm"]) == len(genes)
