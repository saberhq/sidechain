"""Contract tests for scripts/fusion_error_correlation.py.

Two ways this can be wrong while producing a plausible table, which is why they are pinned.

**The oracle direction.** Five of cell-eval2's per-target members score *downward* (`nmae`, the
three `mse` forms, `expr_distance_unbiased`). Taking a per-target max on those reports the worse
arm as a free gain, and the number looks entirely normal. So the oracle must take the min there.

**The shared-difficulty floor.** The flag exists because a raw correlation between two arms on one
fold is mostly a statement about the fold, not about the arms. If `--control-arm` were silently
ignored the report would still print, and the reading -- "is the candidate less correlated with
the incumbent than an information-free arm is?" -- would be gone.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fx = _load("fusion_error_correlation")


def _results_csv(path: Path, values: dict[str, np.ndarray], targets: list[str]) -> Path:
    rows = [{"perturbation": t, "metric": m, "value": v[i]}
            for m, v in values.items() for i, t in enumerate(targets)]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_oracle_takes_the_min_on_a_metric_that_scores_downward(tmp_path):
    targets = ["T1", "T2", "T3"]
    a = _results_csv(tmp_path / "a.csv", {"de_wilcoxon_lfc_nmae": np.array([1.0, 2.0, 3.0])}, targets)
    b = _results_csv(tmp_path / "b.csv", {"de_wilcoxon_lfc_nmae": np.array([3.0, 1.0, 1.0])}, targets)
    out = fx.compare(fx.per_target(a, "de_wilcoxon_lfc_nmae"),
                     fx.per_target(b, "de_wilcoxon_lfc_nmae"), "de_wilcoxon_lfc_nmae")
    assert out["lower_is_better"] is True
    assert out["oracle_mean"] == 1.0                       # min(1,3), min(2,1), min(3,1)
    # a downward metric improves by going DOWN: the better single arm is b at 5/3, and the
    # oracle beats it by 2/3. A max-taking oracle would have reported 7/3, i.e. a loss as a gain.
    assert out["oracle_gain_over_best_single"] == pytest.approx(-2 / 3)
    assert out["targets_where_b_wins"] == 2


def test_oracle_takes_the_max_on_a_metric_that_scores_upward(tmp_path):
    targets = ["T1", "T2"]
    a = _results_csv(tmp_path / "a.csv", {"pds_cosine": np.array([0.9, 0.2])}, targets)
    b = _results_csv(tmp_path / "b.csv", {"pds_cosine": np.array([0.3, 0.8])}, targets)
    out = fx.compare(fx.per_target(a, "pds_cosine"), fx.per_target(b, "pds_cosine"), "pds_cosine")
    assert out["lower_is_better"] is False
    assert out["oracle_mean"] == pytest.approx(0.85)
    assert out["oracle_gain_over_best_single"] == pytest.approx(0.30)   # better single arm is 0.55


def test_only_shared_targets_are_compared(tmp_path):
    a = _results_csv(tmp_path / "a.csv", {"pds_cosine": np.arange(4.0)}, ["A", "B", "C", "D"])
    b = _results_csv(tmp_path / "b.csv", {"pds_cosine": np.arange(3.0)}, ["B", "C", "E"])
    out = fx.compare(fx.per_target(a, "pds_cosine"), fx.per_target(b, "pds_cosine"), "pds_cosine")
    assert out["n_shared_targets"] == 2


def test_the_control_arm_reaches_the_report_as_a_difficulty_floor(tmp_path):
    rng = np.random.default_rng(0)
    targets = [f"T{i}" for i in range(60)]
    hard = rng.random(60)                       # the fold's own per-target difficulty
    a = _results_csv(tmp_path / "a.csv", {"pds_cosine": hard + 0.05 * rng.random(60)}, targets)
    b = _results_csv(tmp_path / "b.csv", {"pds_cosine": hard + 0.05 * rng.random(60)}, targets)
    c = _results_csv(tmp_path / "c.csv", {"pds_cosine": hard + 0.05 * rng.random(60)}, targets)
    out = tmp_path / "out"
    fx.main(["--arm-a", str(a), "--arm-b", str(b), "--control-arm", str(c),
             "--names", "incumbent", "cand", "null", "--out", str(out)])
    report = json.loads((out / "report.json").read_text())
    assert report["control_arm"]["name"] == "null"
    assert report["excess_over_difficulty_floor"] is not None
    # three arms driven by the same difficulty term: the candidate is no less correlated
    # with the incumbent than the information-free arm is.
    assert abs(report["excess_over_difficulty_floor"]) < 0.15
    per = pd.read_csv(out / "per_target_pds_cosine.csv")
    assert list(per.columns) == ["perturbation", "incumbent", "cand", "null"]
    assert len(per) == 60
