"""Contract tests for scripts/standings.py — the rank-when-scored rule.

The rule (scripts/standings.py docstring; private RESULTS.md header): rank is the rank
in the FIRST board snapshot that contains the entry, and the board size is that
snapshot's full team count (`live.total`) — never the embedded row count, because the
board page embeds only its top ~50 teams and the two diverged on 2026-08-21. An entry
ranked below the embed appears in no snapshot and falls back to its status record's
scoring-time rank, with the board size from the first snapshot after the submission.
"""
import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "standings", Path(__file__).resolve().parent.parent / "scripts" / "standings.py"
)
standings = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(standings)


def snap(snaps_dir, stamp, ranks, total):
    (snaps_dir / f"lb_{stamp}.json").write_text(json.dumps({
        "fetched_utc": stamp,
        "live": {"entries": [{"id": i, "rank": r} for i, r in ranks.items()],
                 "total": total},
    }))


def status(subs_dir, date, stem, entry_id, submission_date, score_avg=0.1, **extra):
    (subs_dir / f"{date}_{stem}.status.json").write_text(json.dumps({
        "entry_id": entry_id, "model_name": "Sidechain SER-9", "description": "",
        "submission_date": submission_date, "score_avg": score_avg, **extra,
    }))


def test_board_size_is_the_total_not_the_embed(tmp_path):
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260824T2051Z", {"e1": 25, "other": 1}, total=216)
    status(subs, "2026-08-24", "a_v1", "e1", "2026-08-24T20:10:41Z")
    (row,) = standings.load_rows(subs, snaps)
    assert (row["rank"], row["teams"]) == (25, 216)


def test_first_containing_snapshot_wins(tmp_path):
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260821T0812Z", {"e1": 2}, total=42)
    snap(snaps, "20260824T2051Z", {"e1": 31}, total=216)
    status(subs, "2026-08-21", "a_v1", "e1", "2026-08-21T08:31:00Z")
    (row,) = standings.load_rows(subs, snaps)
    assert (row["rank"], row["teams"]) == (2, 42)


def test_below_the_embed_falls_back_to_status_rank(tmp_path):
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260821T2318Z", {"other": 1}, total=95)
    snap(snaps, "20260824T2051Z", {"other": 1}, total=216)
    status(subs, "2026-08-22", "a_v1", "e1", "2026-08-22T00:05:00Z", rank=77)
    (row,) = standings.load_rows(subs, snaps)
    # teams from the first snapshot AFTER the submission, not an earlier one
    assert (row["rank"], row["teams"]) == (77, 216)


def test_no_rank_anywhere_renders_an_em_dash(tmp_path):
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260824T2051Z", {"other": 1}, total=216)
    status(subs, "2026-08-24", "a_v1", "e1", "2026-08-24T20:10:41Z")
    (row,) = standings.load_rows(subs, snaps)
    assert (row["rank"], row["teams"]) == (None, None)
    assert standings.rank_label(row["rank"], row["teams"]) == "—"


def test_unscored_submission_stays_out(tmp_path):
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    status(subs, "2026-08-24", "a_v1", "e1", "2026-08-24T20:10:41Z", score_avg=None)
    assert standings.load_rows(subs, snaps) == []


def test_pre_divergence_snapshot_without_total_uses_the_embed_count(tmp_path):
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    (snaps / "lb_20260820T2218Z.json").write_text(json.dumps({
        "fetched_utc": "20260820T2218Z",
        "live": {"entries": [{"id": "e1", "rank": 4}, {"id": "x", "rank": 1}]},
    }))
    status(subs, "2026-08-20", "a_v1", "e1", "2026-08-20T22:00:00Z")
    (row,) = standings.load_rows(subs, snaps)
    assert (row["rank"], row["teams"]) == (4, 2)
