"""Contract tests for scripts/standings.py — the rank-when-scored rule.

The rule (scripts/standings.py docstring; private RESULTS.md header): rank is the rank
in the FIRST board snapshot that contains the entry, and the board size is that
snapshot's full team count (`live.total`) — never the embedded row count, because the
board page embeds only its top ~50 teams and the two diverged on 2026-08-21. An entry
ranked below the embed appears in no snapshot and falls back to its status record's
scoring-time rank, with the board size from the first snapshot after the submission.

Also the `sidechain_class` rule (Saber, 2026-09-11, vocabulary settled 2026-09-12): the key is
present and reads `calibration` on an entry sent purely to test a new method on the official
board, and absent on every other one. There is no opposite label — every submission asks a
question. No entry is left out of the outputs on it and it is not part of a model's name. It
decides only how the site draws an entry: in the bars a calibration run sits in the strip
below instead of sharing the main axis, and the rank plot leaves it out (Saber, 2026-10-07).
"""
import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "standings", Path(__file__).resolve().parent.parent / "scripts" / "standings.py"
)
standings = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(standings)


def snap(snaps_dir, stamp, ranks, total, final=None, final_total=None):
    doc = {
        "fetched_utc": stamp,
        "live": {"entries": [{"id": i, "rank": r} for i, r in ranks.items()],
                 "total": total},
    }
    if final is not None:
        doc["final"] = {"entries": [{"id": i, "rank": r} for i, r in final.items()],
                        "total": final_total or len(final)}
    (snaps_dir / f"lb_{stamp}.json").write_text(json.dumps(doc))


def status(subs_dir, date, stem, entry_id, submission_date, score_avg=0.1,
           klass=None, **extra):
    """klass defaults to None -- no `sidechain_class` key at all, which is the normal shape.
    Only a calibration run carries one."""
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
    """The witness is the first snapshot AFTER the submission, never an earlier one -- and
    since 2026-09-11 it must also be within TEAMS_MAX_LAG_DAYS, so this fixture's second
    snapshot sits inside the window (the out-of-window case has its own test below)."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260821T2318Z", {"other": 1}, total=95)
    snap(snaps, "20260822T2051Z", {"other": 1}, total=216)
    status(subs, "2026-08-22", "a_v1", "e1", "2026-08-22T00:05:00Z", rank=77)
    (row,) = standings.load_rows(subs, snaps)
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
        "scores": {"rank": 101, "score_avg": 0.0992},
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


def test_a_calibration_run_is_marked_and_stays_in_the_outputs(tmp_path):
    """Both rows reach the generated outputs. The class travels with the row so the site can
    draw them apart (a strip in the bars, no point on the rank plot); it never removes a row
    from the JSON or the README."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260824T2051Z", {"e1": 25, "e2": 700}, total=800)
    status(subs, "2026-08-24", "a_v1", "e1", "2026-08-24T20:10:41Z")
    status(subs, "2026-08-25", "b_v1", "e2", "2026-08-25T20:10:41Z",
           score_avg=-0.9807, klass="calibration")
    rows = standings.load_rows(subs, snaps)
    assert [r["class"] for r in rows] == ["", "calibration"]
    assert standings.CLASS_WARNINGS == []


def test_an_absent_class_is_the_norm_and_never_warns(tmp_path):
    """There is no opposite label to write down, so an absent key is not a missing one --
    demanding it of every entry would be noise, not provenance (Saber, 2026-09-11)."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260924T2051Z", {"e1": 25}, total=216)
    status(subs, "2026-09-24", "a_v1", "e1", "2026-09-24T20:10:41Z")
    (row,) = standings.load_rows(subs, snaps)
    assert row["class"] == ""
    assert standings.CLASS_WARNINGS == []


def test_an_unknown_class_is_not_trusted(tmp_path):
    """A typo in a record must not quietly become a category: --check fails on it."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260924T2051Z", {"e1": 25}, total=216)
    status(subs, "2026-09-24", "a_v1", "e1", "2026-09-24T20:10:41Z", klass="contender")
    (row,) = standings.load_rows(subs, snaps)
    assert row["class"] == ""
    assert len(standings.CLASS_WARNINGS) == 1 and "contender" in standings.CLASS_WARNINGS[0]


def test_a_row_carries_the_class_and_nothing_else_about_it(tmp_path):
    """A generated row's only class key is `class`: any other key on a record does not travel."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260824T2051Z", {"e1": 25, "e2": 26}, total=216)
    status(subs, "2026-08-24", "a_v1", "e1", "2026-08-24T20:10:41Z",
           klass="calibration", sidechain_class_note="x")
    status(subs, "2026-08-25", "b_v1", "e2", "2026-08-25T20:10:41Z", klass="calibration")
    for row in standings.load_rows(subs, snaps):
        assert row["class"] == "calibration"
        assert [k for k in row if k.startswith("class")] == ["class"]


def test_the_readme_marks_only_the_calibration_run(tmp_path):
    """There is no opposite label, so every other row stays bare."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260824T2051Z", {"e1": 25, "e2": 700}, total=800)
    status(subs, "2026-08-24", "a_v1", "e1", "2026-08-24T20:10:41Z")
    status(subs, "2026-08-25", "b_v1", "e2", "2026-08-25T20:10:41Z",
           score_avg=-0.9807, klass="calibration")
    plain, calib = standings.readme_block(standings.load_rows(subs, snaps)).splitlines()[2:]
    assert "calibration" not in plain
    assert "· **calibration**" in calib
    assert "-0.9807" in calib  # the number is never softened, only drawn apart


def test_a_late_snapshot_is_not_a_witness_for_the_field_size(tmp_path):
    """The real case: PHE-2 scored 2026-09-07 below the page's embed, and the snapshot taken
    five days later for ANOTHER entry silently handed it a denominator of 903 — a field it
    never competed in (533 teams eleven days earlier). No denominator beats a flattering one
    (Saber, 2026-09-11), so the fallback only accepts a contemporary witness."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260903T2329Z", {"other": 1}, total=600)
    snap(snaps, "20260912T0216Z", {"other": 1}, total=903)
    status(subs, "2026-09-07", "phe2_v1", "e1", "2026-09-07T22:38:56Z", rank=746,
           klass="calibration")
    (row,) = standings.load_rows(subs, snaps)
    assert (row["rank"], row["teams"]) == (746, None)
    assert standings.rank_label(row["rank"], row["teams"]) == "#746"


def test_a_same_day_snapshot_still_supplies_the_field_size(tmp_path):
    """The guard must not break the normal path — log_submission.py takes the snapshot at
    record time, so the witness is minutes old, not days."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260912T0216Z", {"other": 1}, total=903)
    status(subs, "2026-09-12", "ser6_v1", "e1", "2026-09-12T00:10:00Z", rank=245)
    (row,) = standings.load_rows(subs, snaps)
    assert (row["rank"], row["teams"]) == (245, 903)


def test_the_frozen_scoring_time_rank_beats_the_records_live_rank(tmp_path):
    """SER-6aefn, 2026-09-12: re-fetched live hours after scoring to recover its
    submission_date, and the board had re-ranked it from 245 to 244 in the meantime. "Rank
    when scored" means when it scored, so the frozen first observation wins."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260912T0216Z", {"other": 1}, total=903)
    status(subs, "2026-09-12", "ser6_v1", "e1", "2026-09-12T01:56:42Z",
           rank=244, sidechain_rank_when_scored=245)
    (row,) = standings.load_rows(subs, snaps)
    assert (row["rank"], row["teams"]) == (245, 903)


def test_without_a_frozen_rank_the_records_own_rank_still_stands(tmp_path):
    """Nothing is written for the eleven records whose `rank` IS their first observation."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    snap(snaps, "20260912T0216Z", {"other": 1}, total=903)
    status(subs, "2026-09-12", "a_v1", "e1", "2026-09-12T01:56:42Z", rank=244)
    (row,) = standings.load_rows(subs, snaps)
    assert (row["rank"], row["teams"]) == (244, 903)


def test_a_final_entry_is_ranked_on_the_final_board_never_the_live_one(tmp_path):
    """The rounds are not comparable (Arc's FAQ): a final record reads the final board's rank
    and field size, and a validation record never sees the final board (T93, 2026-09-23)."""
    subs, snaps = tmp_path / "s", tmp_path / "l"
    subs.mkdir(); snaps.mkdir()
    status(subs, "2026-09-17", "ser-7_v1", "VAL1", "2026-09-17T19:43:00", partition="val",
           model_name="Sidechain SER-7")
    status(subs, "2026-10-24", "ser-8_v1", "FIN1", "2026-10-24T18:00:00", partition="final",
           model_name="Sidechain SER-8")
    snap(snaps, "20260917T2003Z", {"VAL1": 314}, 1022)
    snap(snaps, "20261024T1900Z", {"VAL1": 400, "FIN1": 3}, 1300, final={"FIN1": 41}, final_total=650)
    rows = {r["name"]: r for r in standings.load_rows(subs, snaps)}
    assert rows["SER-7"]["rank"] == 314 and rows["SER-7"]["teams"] == 1022 and rows["SER-7"]["partition"] == "val"
    assert rows["SER-8"]["rank"] == 41 and rows["SER-8"]["teams"] == 650 and rows["SER-8"]["partition"] == "final"
    block = standings.readme_block(list(rows.values()))
    assert "· **final**" in block.splitlines()[-1] and "**final**" not in block.splitlines()[-2]


def test_a_final_entry_below_the_embed_takes_the_final_boards_field_size(tmp_path):
    subs, snaps = tmp_path / "s", tmp_path / "l"
    subs.mkdir(); snaps.mkdir()
    status(subs, "2026-10-24", "ser-8_v1", "FIN1", "2026-10-24T18:00:00", partition="final", rank=120)
    snap(snaps, "20261024T1900Z", {"X": 1}, 1300, final={"Y": 1}, final_total=650)
    (row,) = standings.load_rows(subs, snaps)
    assert (row["rank"], row["teams"]) == (120, 650)          # never 1300, the validation field


def test_a_record_without_a_partition_is_a_validation_entry(tmp_path):
    subs, snaps = tmp_path / "s", tmp_path / "l"
    subs.mkdir(); snaps.mkdir()
    status(subs, "2026-08-21", "r1_v1", "OLD", "2026-08-21T08:31:00")
    snap(snaps, "20260821T0900Z", {"OLD": 2}, 42, final={"Z": 1}, final_total=5)
    (row,) = standings.load_rows(subs, snaps)
    assert (row["rank"], row["teams"], row["partition"]) == (2, 42, "val")


# -- the field file (T107): the leaderboard over time, from board_teams.py's ticks.json --------


def tick(stamp, field, source="api", origin="ours", at_rank=None, our=None):
    return {"stamp": stamp, "field": field, "source": source, "origin": origin,
            "at_rank": at_rank or {}, "our": our}


def test_the_field_file_is_the_series_headed_by_the_latest_whole_field_snapshot(tmp_path):
    ticks = tmp_path / "ticks.json"
    ours = {"rank": 294, "score_avg": 0.16264962, "model_name": "Sidechain SER-14aefksw"}
    ticks.write_text(json.dumps([
        tick("20260915T2116Z", 970, at_rank={"20": 0.21444, "100": 0.15759}, our={**ours, "rank": 280}),
        tick("20260820T2218Z", 6, source="page"),                       # out of order on disk
        tick("20260824T1806Z", 216, source="page", at_rank={"20": 0.0871}),
        tick("20260826T1456Z", 290, origin="wayback", at_rank={"20": 0.11921, "100": 0.05136}),
        tick("20261005T0030Z", 1314, at_rank={"20": 0.267277, "100": 0.225874}, our=ours),
        tick("20261005T0200Z", 1316, source="page", at_rank={"20": 0.27}),   # a page tick after it
    ]))
    f = standings.load_field(ticks)
    # the head is the last snapshot of the whole field that is ours -- never a page tick, which
    # cannot give rank 100, and never the archive's capture
    assert (f["as_of"], f["teams"], f["rank_20"], f["rank_100"]) == ("2026-10-05T00:30:00Z", 1314, 0.2673, 0.2259)
    assert f["ours"] == {"rank": 294, "overall": 0.1626, "name": "SER-14aefksw"}
    assert [p["t"] for p in f["series"]] == sorted(p["t"] for p in f["series"]) and len(f["series"]) == 6
    by_t = {p["t"]: p for p in f["series"]}
    assert by_t["2026-08-20T22:18:00Z"] == {"t": "2026-08-20T22:18:00Z", "teams": 6, "rank_20": None,
                                            "rank_100": None, "source": "page"}
    assert by_t["2026-08-24T18:06:00Z"]["rank_20"] == 0.0871 and by_t["2026-08-24T18:06:00Z"]["rank_100"] is None
    assert by_t["2026-08-26T14:56:00Z"]["source"] == "wayback" and by_t["2026-08-26T14:56:00Z"]["rank_100"] == 0.0514
    # no team's row passes through: a point is five finished numbers and nothing else
    assert all(set(p) == {"t", "teams", "rank_20", "rank_100", "source"} for p in f["series"])


def test_the_field_file_cut_at_its_own_as_of_ignores_newer_snapshots(tmp_path):
    ticks = tmp_path / "ticks.json"
    ours = {"rank": 300, "score_avg": 0.15, "model_name": "Sidechain SER-9"}
    old = [tick("20260915T2116Z", 970, at_rank={"20": 0.2, "100": 0.1}, our=ours)]
    ticks.write_text(json.dumps(old))
    had = standings.load_field(ticks)
    ticks.write_text(json.dumps(old + [tick("20260916T1630Z", 990, at_rank={"20": 0.21, "100": 0.11}, our=ours)]))
    assert standings.load_field(ticks, as_of=had["as_of"]) == had      # --check: no drift
    assert standings.load_field(ticks)["teams"] == 990                 # a rewrite: current


def test_no_series_on_this_machine_is_none_not_a_raise(tmp_path):
    assert standings.load_field(tmp_path / "absent.json") is None
    only_page = tmp_path / "ticks.json"
    only_page.write_text(json.dumps([tick("20260820T2218Z", 6, source="page")]))
    assert standings.load_field(only_page) is None       # nothing to head the file with yet


# --------------------------------------------------------------------------- the members and the Sidechain-only board (T79, 2026-10-08)


def members(**over):
    """The twelve member keys of a status record, as Arc writes them; override any by key."""
    base = {"score_pds": 0.6, "pds_cosine": 0.8, "score_mse": 0.15, "expr_mse_unbiased_capped_norm": 0.85,
            "score_jac": 0.01, "de_wilcoxon_sig_jaccard": 0.03, "score_nmae": 0.09, "de_wilcoxon_lfc_nmae": 0.94,
            "score_fid": -0.02, "de_wilcoxon_direction_fidelity_yield_raw": 0.5, "score_reach": 0.1,
            "de_wilcoxon_direction_reach_raw": 0.17}
    base.update(over)
    return base


def test_a_row_carries_the_six_members_scaled_over_raw(tmp_path):
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    status(subs, "2026-10-08", "a_v1", "e1", "2026-10-08T16:53:57Z", score_avg=0.1657, **members())
    (row,) = standings.load_rows(subs, snaps)
    assert list(row["members"]) == ["pds", "mse", "jac", "nmae", "fid", "reach"]
    assert row["members"]["pds"] == {"scaled": 0.6, "raw": 0.8}
    assert row["members"]["fid"] == {"scaled": -0.02, "raw": 0.5}


def test_a_submit_shaped_record_has_scaled_members_and_no_raw(tmp_path):
    """The `--wait --json` shape nests only the scaled members under `scores`; raw is None, never 0."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    (subs / "2026-08-30_probe_v1.status.json").write_text(json.dumps({
        "entry_id": "p1", "model_name": "Sidechain SER-3afgn",
        "scores": {"score_avg": 0.09, "rank": 12, "score_pds": 0.4, "score_mse": 0.0, "score_jac": 0.001,
                   "score_nmae": 0.05, "score_fid": -0.01, "score_reach": 0.06}}))
    (row,) = standings.load_rows(subs, snaps)
    assert row["members"]["pds"] == {"scaled": 0.4, "raw": None}
    assert row["members"]["mse"] == {"scaled": 0.0, "raw": None}


def test_the_board_is_sorted_by_overall_and_tinted_per_column_by_the_non_calibration_entries(tmp_path):
    """Per column: the best non-calibration score is the full green (+1), the worst negative non-calibration
    score the full red (-1); a calibration run is painted on that scale (clamped) and never sets it; a
    scaled 0 is the floor and reads full red."""
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    status(subs, "2026-08-21", "r1_delta_even_v1", "e1", "2026-08-21T07:51:06Z", score_avg=0.0788,
           model_name="sidechain r1-delta-even v1", **members(score_pds=0.3, score_mse=0.0, score_fid=-0.04))
    status(subs, "2026-10-03", "b_v1", "e2", "2026-10-03T10:00:00Z", score_avg=0.1626,
           model_name="Sidechain SER-14aefksw", **members(score_pds=0.6, score_mse=0.15, score_fid=-0.02))
    status(subs, "2026-10-08", "c_v1", "e3", "2026-10-08T15:23:27Z", score_avg=0.1423, klass="calibration",
           model_name="Sidechain SER-16aefhkrsw", **members(score_pds=0.9, score_mse=0.15, score_fid=-0.17))
    rows = standings.load_rows(subs, snaps)
    b = standings.board(rows)
    assert b["columns"] == ["pds", "mse", "jac", "nmae", "fid", "reach"]
    # by overall; n is the place on it
    assert [r["name"] for r in b["rows"]] == ["SER-14aefksw", "SER-16aefhkrsw", "SER-1"]
    assert [r["n"] for r in b["rows"]] == [1, 2, 3]
    assert b["scale"]["pds"] == {"ceiling": 0.6, "floor": None}      # the calibration run's 0.9 sets nothing
    assert b["scale"]["fid"] == {"ceiling": None, "floor": -0.04}
    assert b["fixed"] == {"ceiling": 1.0, "floors": {"pds": -1.0, "mse": 0.0, "jac": -1.0, "nmae": -6.0, "fid": -1.0, "reach": -1.0}}
    # the leaderboard page's scale: /1 above zero, /floor below, a scaled 0 the floor
    fixed = {(r["name"], c["key"]): c["tint"] for r in b["rows"] for c in r["cells"]}
    assert fixed[("SER-14aefksw", "pds")] == 0.6 and fixed[("SER-16aefhkrsw", "pds")] == 0.9
    assert fixed[("SER-1", "mse")] == -1.0 and fixed[("SER-14aefksw", "fid")] == -0.02
    assert fixed[("SER-16aefhkrsw", "fid")] == -0.17
    cell = {(r["name"], c["key"]): c["tint_own"] for r in b["rows"] for c in r["cells"]}
    assert cell[("SER-14aefksw", "pds")] == 1.0
    assert cell[("SER-1", "pds")] == 0.5
    assert cell[("SER-1", "mse")] == -1.0                            # a scaled 0 is the floor
    assert cell[("SER-1", "fid")] == -1.0 and cell[("SER-14aefksw", "fid")] == -0.5
    calib = next(r for r in b["rows"] if r["class"] == "calibration")
    assert {c["key"]: c["tint_own"] for c in calib["cells"]}["pds"] == 1.0     # clamped, not 1.5
    assert {c["key"]: c["tint_own"] for c in calib["cells"]}["fid"] == -1.0    # clamped
    assert calib["rank_when_scored"] is None and calib["name"] == "SER-16aefhkrsw"


def test_the_site_json_carries_the_board(tmp_path, monkeypatch):
    subs, snaps = tmp_path / "subs", tmp_path / "snaps"
    subs.mkdir(); snaps.mkdir()
    status(subs, "2026-10-08", "a_v1", "e1", "2026-10-08T16:53:57Z", score_avg=0.1657, **members())
    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- standings:begin -->\n<!-- standings:end -->\n")
    monkeypatch.setattr(standings, "README", readme)
    monkeypatch.setattr(standings, "SITE_JSON", tmp_path / "submissions.json")
    _, site = standings.render(standings.load_rows(subs, snaps))
    d = json.loads(site)
    assert d["board"]["rows"][0]["cells"][0] == {"key": "pds", "scaled": 0.6, "raw": 0.8, "tint": 0.6, "tint_own": 1.0}
