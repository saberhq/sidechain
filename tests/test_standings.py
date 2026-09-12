"""Contract tests for scripts/standings.py — the rank-when-scored rule.

The rule (scripts/standings.py docstring; private RESULTS.md header): rank is the rank
in the FIRST board snapshot that contains the entry, and the board size is that
snapshot's full team count (`live.total`) — never the embedded row count, because the
board page embeds only its top ~50 teams and the two diverged on 2026-08-21. An entry
ranked below the embed appears in no snapshot and falls back to its status record's
scoring-time rank, with the board size from the first snapshot after the submission.

Also the `sidechain_class` rule (Saber, 2026-09-11): a record declares what the entry was FOR
— `contender` or `probe` — when it is logged, before its score is known. `contender` is the
default, so only the exception is written down, and only entries dated CLASS_REQUIRED_SINCE or
later owe the field at all. Nothing is filtered on it and it is not part of a model's name: it
decides only whether an entry shares the main axis or sits in the probe strip.
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


def status(subs_dir, date, stem, entry_id, submission_date, score_avg=0.1,
           klass="contender", **extra):
    """klass=None writes a record with no `sidechain_class` at all -- the pre-rule shape."""
    body = {"entry_id": entry_id, "model_name": "Sidechain SER-9", "description": "",
            "submission_date": submission_date, "score_avg": score_avg}
    if klass is not None:
        body["sidechain_class"] = klass
    (subs_dir / f"{date}_{stem}.status.json").write_text(json.dumps({**body, **extra}))


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


def test_a_submit_shaped_record_is_flattened_and_dated_from_its_filename(tmp_path):
    """A superseded probe's only record is the `vcc submit --wait --json` output --
    the status endpoint serves just a team's latest validation entry, so
    log_submission.py --status-file saves that shape verbatim (2026-08-30,
    SER-3afgn). Its members nest under "scores" and it has no submission_date;
    the row must still land, with the scoring-time rank and the board size from
    the first snapshot on the record's own filing date -- not the earliest
    snapshot ever, which is what an empty stamp used to select."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260820T0900Z", {"other": 1}, 42)
    snap(snaps, "20260830T1828Z", {"other": 1}, 476)
    (subs / "2026-08-30_probe_v1.status.json").write_text(json.dumps({
        "entry_id": "probe1", "model_name": "Sidechain SER-9", "final_status": "published",
        "sidechain_class": "contender", "scores": {"rank": 101, "score_avg": 0.0992},
    }))
    rows = standings.load_rows(subs, snaps)
    assert len(rows) == 1
    assert rows[0]["rank"] == 101
    assert rows[0]["teams"] == 476
    assert rows[0]["date"] == "2026-08-30"
    assert rows[0]["overall"] == 0.0992


def test_unscored_submission_stays_out(tmp_path):
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    status(subs, "2026-08-24", "a_v1", "e1", "2026-08-24T20:10:41Z", score_avg=None)
    assert standings.load_rows(subs, snaps) == []


def test_unparseable_board_name_without_alias_warns(tmp_path):
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260824T2051Z", {"e1": 25}, total=216)
    status(subs, "2026-08-24", "typo_v1", "e1", "2026-08-24T20:10:41Z",
           model_name="sidechain something v1")
    (row,) = standings.load_rows(subs, snaps)
    assert row["name"] == "typo_v1"  # the row still renders, under the raw stem
    assert len(standings.NAME_WARNINGS) == 1 and "typo_v1" in standings.NAME_WARNINGS[0]


def test_aliased_pre_adr_stem_stays_quiet(tmp_path):
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260821T0812Z", {"e1": 2}, total=42)
    status(subs, "2026-08-21", "r1_delta_even_v1", "e1", "2026-08-21T08:31:00Z",
           model_name="sidechain r1-delta-even v1")
    (row,) = standings.load_rows(subs, snaps)
    assert row["name"] == "SER-1"
    assert standings.NAME_WARNINGS == []


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


def test_a_cardless_entry_takes_the_sidecar_card_and_is_flagged_retro(tmp_path):
    """The board can't be backfilled (ADR 0005), so a `<record>.card.txt` beside the
    status record supplies the card for our own surfaces -- flagged, so the site can
    say the board entry itself is bare. A board card, when present, always wins."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260824T2051Z", {"e1": 25, "e2": 26}, total=216)
    status(subs, "2026-08-24", "a_v1", "e1", "2026-08-24T20:10:41Z")
    (subs / "2026-08-24_a_v1.card.txt").write_text("SER-9 = a retro card.\n")
    status(subs, "2026-08-24", "b_v1", "e2", "2026-08-24T21:00:00Z",
           description="SER-9 = a live board card.")
    (subs / "2026-08-24_b_v1.card.txt").write_text("must never be read")
    retro, live = standings.load_rows(subs, snaps)
    assert (retro["card"], retro["card_retro"]) == ("SER-9 = a retro card.", True)
    assert (live["card"], live["card_retro"]) == ("SER-9 = a live board card.", False)


def test_class_is_read_from_the_record_and_nothing_is_filtered(tmp_path):
    """Both kinds reach the rows. The class travels with the row so the surfaces can draw
    them apart; it never removes an entry -- hiding a submission after seeing its score is
    the failure this design exists to avoid (Saber, 2026-09-11)."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260824T2051Z", {"e1": 25, "e2": 700}, total=800)
    status(subs, "2026-08-24", "a_v1", "e1", "2026-08-24T20:10:41Z")
    status(subs, "2026-08-25", "b_v1", "e2", "2026-08-25T20:10:41Z",
           score_avg=-0.9807, klass="probe")
    rows = standings.load_rows(subs, snaps)
    assert [r["class"] for r in rows] == ["contender", "probe"]
    assert standings.CLASS_WARNINGS == []


def test_an_entry_that_owes_a_class_and_has_none_warns(tmp_path):
    """From CLASS_REQUIRED_SINCE on, log_submission.py demands the flag, so a record dated
    after it with no class was logged some other way -- read it as a contender, and let
    --check fail rather than letting an unlabelled entry onto the chart quietly."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260924T2051Z", {"e1": 25}, total=216)
    status(subs, "2026-09-24", "a_v1", "e1", "2026-09-24T20:10:41Z", klass=None)
    (row,) = standings.load_rows(subs, snaps)
    assert row["class"] == "contender"
    assert len(standings.CLASS_WARNINGS) == 1 and "a_v1" in standings.CLASS_WARNINGS[0]


def test_an_entry_predating_the_field_owes_nothing_and_says_nothing(tmp_path):
    """`contender` is the default, so only the exception is ever written down. Back-writing
    it over the ten records that predate the field would be noise, not provenance -- and a
    permanent --check failure if it were demanded of them (Saber, 2026-09-11)."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260824T2051Z", {"e1": 25}, total=216)
    status(subs, "2026-08-24", "a_v1", "e1", "2026-08-24T20:10:41Z", klass=None)
    (row,) = standings.load_rows(subs, snaps)
    assert row["class"] == "contender"
    assert standings.CLASS_WARNINGS == []


def test_an_unknown_class_is_not_trusted(tmp_path):
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260924T2051Z", {"e1": 25}, total=216)
    status(subs, "2026-09-24", "a_v1", "e1", "2026-09-24T20:10:41Z", klass="benchmark")
    (row,) = standings.load_rows(subs, snaps)
    assert row["class"] == "contender"
    assert len(standings.CLASS_WARNINGS) == 1


def test_class_retro_marks_the_entries_that_predate_the_field(tmp_path):
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260824T2051Z", {"e1": 25, "e2": 26}, total=216)
    status(subs, "2026-08-24", "a_v1", "e1", "2026-08-24T20:10:41Z",
           sidechain_class_retro=True)
    status(subs, "2026-08-25", "b_v1", "e2", "2026-08-25T20:10:41Z")
    old, new = standings.load_rows(subs, snaps)
    assert (old["class_retro"], new["class_retro"]) == (True, False)


def test_the_readme_marks_probes_and_leaves_contenders_bare(tmp_path):
    """Ten rows tagged "contender" would be noise for a word true by default; the prose
    above the table carries the rule, and only the exception is marked."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260824T2051Z", {"e1": 25, "e2": 700}, total=800)
    status(subs, "2026-08-24", "a_v1", "e1", "2026-08-24T20:10:41Z")
    status(subs, "2026-08-25", "b_v1", "e2", "2026-08-25T20:10:41Z",
           score_avg=-0.9807, klass="probe")
    contender, probe = standings.readme_block(standings.load_rows(subs, snaps)).splitlines()[2:]
    assert "probe" not in contender
    assert "· **probe**" in probe
    assert "-0.9807" in probe  # the number is never softened, only drawn apart
