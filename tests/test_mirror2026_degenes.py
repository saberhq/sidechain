"""`mirror2026.score` passes `--write-degenes` to cell-eval2's run only when asked.

    uv run pytest tests/test_mirror2026_degenes.py
Reads nothing; writes a fake scored.csv under pytest's tmp_path. Notes: private/code-notes/tests/test_mirror2026_degenes.py.md
"""
from pathlib import Path

import pytest

from sidechain.eval import mirror2026 as mir


@pytest.mark.parametrize("flag", [False, True])
def test_write_degenes_reaches_the_run(tmp_path: Path, monkeypatch, flag: bool) -> None:
    calls = []

    def fake_run(cmd, log):
        calls.append(cmd)
        if "score" in cmd:
            out = Path(cmd[cmd.index("-o") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("metric,from_replicate\npds_cosine,0.5\n")

    monkeypatch.setattr(mir, "_run", fake_run)
    res = mir.score(tmp_path / "p.h5ad", tmp_path / "r.h5ad", tmp_path / "b", tmp_path / "o",
                    pert_col="perturbation", control="non-targeting", write_degenes=flag)
    run_cmd = next(c for c in calls if "run" in c)
    assert ("--write-degenes" in run_cmd) is flag
    assert res["members"] == {"pds_cosine": 0.5}
