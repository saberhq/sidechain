"""`utils.logging.log_run`: what it records, and that it can never kill a run.

The contract under test is ADR 0001's: run logging is worth having only if it is
free to fail — a scoring run's validity comes from the mirror, and its numbers
already live in the caller's summary/report JSON, so the logger must degrade to a
warning, once, and never raise.
"""
from __future__ import annotations

import sys
import types
import warnings

import pytest

import sidechain.utils.logging as slog


@pytest.fixture(autouse=True)
def _reset_warn_once(monkeypatch):
    monkeypatch.setattr(slog, "_WARNED", False)


@pytest.fixture()
def fake_lamindb(monkeypatch):
    """A minimal lamindb double recording every call."""
    calls = {"track": [], "artifacts": [], "finish": 0}

    class _Artifact:
        def __init__(self, path, key=None):
            self.path, self.key = path, key

        def save(self):
            calls["artifacts"].append((self.path, self.key))

    mod = types.ModuleType("lamindb")
    mod.track = lambda params=None: calls["track"].append(params)
    mod.Artifact = _Artifact
    mod.finish = lambda: calls.__setitem__("finish", calls["finish"] + 1)
    monkeypatch.setitem(sys.modules, "lamindb", mod)
    return calls


def test_log_run_records_config_metrics_git_sha_and_artifacts(fake_lamindb, tmp_path):
    art = tmp_path / "summary.json"
    art.write_text("{}")
    slog.log_run({"alpha": 1.0}, {"overall": 0.2}, artifacts=[str(art)])

    (params,) = fake_lamindb["track"]
    assert params["config"] == {"alpha": 1.0}
    assert params["metrics"] == {"overall": 0.2}
    # In this working tree the SHA resolves; the contract is only that the key is
    # always stamped with one of the three shapes code_sha() can produce.
    sha = params["code_sha"]
    assert sha == "unknown" or len(sha.removeprefix("dirty:")) == 40
    assert fake_lamindb["artifacts"] == [(str(art), "runs/summary.json")]
    assert fake_lamindb["finish"] == 1


def test_a_lamindb_failure_is_one_warning_never_an_exception(monkeypatch):
    mod = types.ModuleType("lamindb")

    def _boom(params=None):
        raise RuntimeError("no instance connected")

    mod.track = _boom
    monkeypatch.setitem(sys.modules, "lamindb", mod)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        slog.log_run({}, {})          # warns
        slog.log_run({}, {})          # warn-once: silent
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len([w for w in caught if issubclass(w.category, RuntimeWarning)]) == 1


def test_log_run_survives_lamindb_not_being_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "lamindb", None)  # import lamindb -> ImportError
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slog.log_run({"x": 1}, {"y": 2}, artifacts=["nowhere.json"])


def test_code_sha_is_shared_with_the_stream_lineage():
    """One stamp for both consumers: LINEAGE.json and run logging must never
    disagree about which code produced an artifact."""
    from sidechain.data import stream_parquet_pseudobulk as spp

    assert spp.code_sha is slog.code_sha
