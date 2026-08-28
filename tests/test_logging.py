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
    calls = {"connect": [], "track": [], "artifacts": [], "finish": 0}

    class _Artifact:
        def __init__(self, path, key=None):
            self.path, self.key = path, key

        def save(self):
            calls["artifacts"].append((self.path, self.key))

    mod = types.ModuleType("lamindb")
    mod.connect = lambda slug: calls["connect"].append(slug)
    mod.track = lambda params=None: calls["track"].append(params)
    mod.Artifact = _Artifact
    mod.finish = lambda: calls.__setitem__("finish", calls["finish"] + 1)
    monkeypatch.setitem(sys.modules, "lamindb", mod)
    return calls


def test_log_run_records_config_metrics_git_sha_and_artifacts(fake_lamindb, tmp_path, monkeypatch):
    monkeypatch.setenv("SIDECHAIN_DATA_ROOT", str(tmp_path))
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
    assert fake_lamindb["artifacts"] == [(str(art), "summary.json")]
    assert fake_lamindb["finish"] == 1


def test_log_run_connects_to_the_hosted_instance_by_default(fake_lamindb, monkeypatch):
    """`ln.connect()` is process-local, so log_run must make the connection
    itself -- there is no machine default to inherit."""
    monkeypatch.delenv("SIDECHAIN_LAMIN_INSTANCE", raising=False)
    slog.log_run({}, {})
    assert fake_lamindb["connect"] == ["saberhq/sidechain"]
    assert len(fake_lamindb["track"]) == 1


def test_the_instance_env_var_overrides_and_empty_disables(fake_lamindb, monkeypatch):
    monkeypatch.setenv("SIDECHAIN_LAMIN_INSTANCE", "someone/else")
    slog.log_run({}, {})
    assert fake_lamindb["connect"] == ["someone/else"]

    # empty string: skip connecting entirely (offline / opted-out machine);
    # the rest of the pipeline still runs against whatever default exists.
    monkeypatch.setenv("SIDECHAIN_LAMIN_INSTANCE", "")
    slog.log_run({}, {})
    assert fake_lamindb["connect"] == ["someone/else"]
    assert len(fake_lamindb["track"]) == 2


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


def test_run_artifacts_are_keyed_by_their_path_so_distinct_runs_stay_distinct(
    fake_lamindb, tmp_path, monkeypatch
):
    """The bug this closes, verified on the hub 2026-08-28: `runs/<basename>` gave five
    scored runs the one key `runs/summary.json`, and lamindb folded them into versions
    0000-0004 of a single artifact — four runs demoted to history of a fifth, silently.
    ADR 0007 §1's rule (key = path under the data root) keeps them apart."""
    monkeypatch.setenv("SIDECHAIN_DATA_ROOT", str(tmp_path))
    keys = []
    for arm in ("h1_only", "h1_xatlas"):
        out = tmp_path / "runs" / "mirror" / "loco_k562gwps" / arm
        out.mkdir(parents=True)
        (out / "summary.json").write_text("{}")
        slog.log_run({}, {}, artifacts=[str(out / "summary.json")])
        keys.append(fake_lamindb["artifacts"][-1][1])

    assert keys == ["runs/mirror/loco_k562gwps/h1_only/summary.json",
                    "runs/mirror/loco_k562gwps/h1_xatlas/summary.json"]
    assert len(set(keys)) == 2


def test_an_artifact_outside_the_data_tree_still_gets_logged(fake_lamindb, tmp_path, monkeypatch):
    """Filing is not worth failing a scored run over — the flat key is the fallback,
    and log_run's non-fatal contract is what it is protecting."""
    monkeypatch.setenv("SIDECHAIN_DATA_ROOT", str(tmp_path / "tree"))
    (tmp_path / "tree").mkdir()
    stray = tmp_path / "stray.json"
    stray.write_text("{}")
    slog.log_run({}, {}, artifacts=[str(stray)])
    assert fake_lamindb["artifacts"] == [(str(stray), "runs/stray.json")]


def test_log_run_refuses_to_register_a_directory(fake_lamindb, tmp_path, monkeypatch):
    """A run OUTDIR is not a run artifact. `local_mirror` passed one until 2026-08-28,
    and a mirror outdir is ~21 GB per line — a scored run would have uploaded it as a
    folder artifact, and folder artifacts overwrite their own previous version's bytes.
    Registering a directory is `scripts/lamin_register.py`'s job: deliberate and loud."""
    monkeypatch.setenv("SIDECHAIN_DATA_ROOT", str(tmp_path))
    outdir = tmp_path / "runs" / "mirror" / "loco" / "arm"
    outdir.mkdir(parents=True)
    (outdir / "summary.json").write_text("{}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        slog.log_run({}, {}, artifacts=[str(outdir), str(outdir / "summary.json")])

    assert fake_lamindb["artifacts"] == [
        (str(outdir / "summary.json"), "runs/mirror/loco/arm/summary.json")
    ]
    assert any("refusing to register directory" in str(w.message) for w in caught)
    assert fake_lamindb["finish"] == 1   # the rest of the run still logs
