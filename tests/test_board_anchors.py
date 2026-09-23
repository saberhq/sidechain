"""scripts/board_anchors.py fits one board and one anchor version at a time (T93, 2026-09-23)."""
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "board_anchors", Path(__file__).resolve().parent.parent / "scripts" / "board_anchors.py")
board_anchors = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(board_anchors)


def _snap(d: Path, stamp: str, live: list[dict], final: list[dict] | None = None) -> None:
    doc = {"fetched_utc": stamp, "live": {"entries": live, "total": len(live)}}
    if final is not None:
        doc["final"] = {"entries": final, "total": len(final)}
    (d / f"lb_{stamp}.json").write_text(json.dumps(doc))


def _entry(i: str, raw: float, version: str = "v1") -> dict:
    return {"id": i, "pdsCosine": raw, "scorePds": 2 * raw - 1, "anchorVersion": version}


def test_only_the_asked_board_is_read(tmp_path: Path) -> None:
    _snap(tmp_path, "20261024T1900Z", [_entry("a", 0.7), _entry("b", 0.8)],
          final=[_entry("f1", 0.6, "final-v1"), _entry("f2", 0.65, "final-v1")])
    assert set(board_anchors.load_entries(tmp_path)) == {"a", "b"}
    assert set(board_anchors.load_entries(tmp_path, "final")) == {"f1", "f2"}
    assert board_anchors.load_entries(tmp_path, "generalist") == {}


def test_two_anchor_versions_on_one_board_are_refused(tmp_path: Path) -> None:
    _snap(tmp_path, "20260917T2003Z", [_entry("a", 0.7, "r4")])
    _snap(tmp_path, "20261101T0000Z", [_entry("a", 0.7, "r4"), _entry("b", 0.8, "r5")])
    with pytest.raises(SystemExit, match="2 anchor versions"):
        board_anchors.load_entries(tmp_path)


def test_an_unknown_board_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--board must be one of"):
        board_anchors.load_entries(tmp_path, "validation")


def test_the_fit_recovers_the_anchors_of_one_board(tmp_path: Path) -> None:
    raws = [0.55, 0.6, 0.7, 0.8, 0.9, 0.95]          # 0.5 would scale to exactly 0, the clamp the fit skips
    _snap(tmp_path, "20261024T1900Z", [_entry(f"e{i}", r) for i, r in enumerate(raws)])
    b, r, n, resid = board_anchors.fit_anchors(board_anchors.load_entries(tmp_path), "pdsCosine", "scorePds")
    assert n == 6 and abs(b - 0.5) < 1e-9 and abs(r - 1.0) < 1e-9 and resid < 1e-9
