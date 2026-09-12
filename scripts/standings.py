"""Generate the scored-submissions table everywhere it appears, from the primary records.

    uv run python scripts/standings.py            # rewrite README.md + site/data/submissions.json
    uv run python scripts/standings.py --check    # exit 1 if either file is out of date

Inputs, both read-only:

- ``private/submissions/*.status.json`` — Arc's verbatim record per entry, written by
  ``private/research/protocol/log_submission.py`` (which runs this script afterwards).
- ``~/data/sidechain/vcc2026/leaderboards/lb_*.json`` — board snapshots, taken with
  ``python -m sidechain.eval.leaderboard``.

The one rule (Saber, 2026-08-23; board size clarified 2026-08-24): **rank-when-scored
is the rank in the first board snapshot that contains the entry, and the board size is
that snapshot's full team count** — ``live.total``, never the embedded row count. The
page embeds only its top ~50 teams and the two diverged on 2026-08-21 (95 teams, 50
embedded), which had this script quietly printing "of 50" against a 216-team board.
The board shows one row per team and re-ranks continuously, so a later look is not the
rank when scored. An entry ranked below the embed appears in no snapshot: it falls back
to its own status record's ``rank`` -- Arc's number when the record was fetched, which
``log_submission.py`` fetches right after scoring -- with the board size from the first
snapshot taken after the submission, **and only if that snapshot is within
``TEAMS_MAX_LAG_DAYS``**. A later one measures a field the entry never competed in, and no
denominator beats a flattering one (Saber, 2026-09-11). ``log_submission.py`` takes the
snapshot itself at record time, so the window is only ever missed when nobody was there.

Every record carries ``sidechain_class`` — ``contender`` (aimed at the score) or ``probe``
(spent to answer a question) — declared when the entry is logged, BEFORE its score comes
back (``log_submission.py --class``). Nothing is ever hidden: both kinds appear in the
table and in the site's JSON. The class only decides how an entry is *drawn*, because one
off-scale probe on a shared axis destroys the resolution of everything else — PHE-2 scored
-0.9807 against a field running 0.0730 to 0.1078 (2026-09-07). A record with no class is
``contender`` is the default, so ONLY the exception is ever written down — a record with no
class reads as a contender. Records dated ``CLASS_REQUIRED_SINCE`` or later owe the field and
warn without it, so ``--check`` fails rather than letting an unlabelled new entry through;
records that predate it owe nothing. PHE-2 carries ``sidechain_class_retro``, being the one
entry whose default would have been wrong and whose class was therefore written after its
score. The class is metadata about intent — it is NOT part of a model's name (ADR 0005 is
untouched) and NOT a prediction: an entry that set out to compete and failed stays a
contender, on the main axis, with its real number.

Outputs, both fully generated — never edit them by hand:

- ``README.md``: the table between ``<!-- standings:begin -->`` and ``<!-- standings:end -->``.
- ``site/data/submissions.json``: the same rows for saberhq.com/sidechain (a pushed
  change under site/ redeploys the page).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
SITE_JSON = ROOT / "site" / "data" / "submissions.json"
BEGIN, END = "<!-- standings:begin -->", "<!-- standings:end -->"

# Board names predating the ADR 0005 naming scheme, keyed by build stem. Entries are
# never renamed on the board, so this map only ever grows.
ALIASES = {"r1_delta_even_v1": "SER-1"}
SERIES_RE = re.compile(r"\b([A-Z]{3}-\d+[a-z]*)\b")

# Board names that neither parse nor sit in ALIASES land here (load_rows resets it).
# The row still renders under its raw stem so old data keeps flowing, but --check fails:
# a silent fallback is how a misnamed entry would reach the README and the site unnoticed.
NAME_WARNINGS: list[str] = []

# `contender` is the default, so only the exception is ever recorded -- a record without a
# class reads as a contender. From CLASS_REQUIRED_SINCE on, log_submission.py demands the
# flag, so a record dated after it with no class means the entry was logged some other way:
# warn, and let --check fail. Records that PREDATE the field owe nothing and say nothing;
# back-writing "contender" over ten of them would be noise, not provenance (Saber, 2026-09-11).
CLASS_WARNINGS: list[str] = []
CLASSES = ("contender", "probe")
CLASS_REQUIRED_SINCE = "2026-09-11"

# The fallback's board size must be a CONTEMPORARY witness, not merely a later one. The
# field grows fast — 533 teams on 2026-09-01, 903 on 2026-09-12, about 34 a day — so a
# snapshot taken days after an entry scored ranks it against a field it never competed in
# and flatters it. It happened: the snapshot taken for SER-6aefn on 2026-09-12 silently gave
# PHE-2 (2026-09-07, rank 746) a denominator of 903. Saber's call is that no denominator
# beats a flattering one. Two days is generous now that log_submission.py takes a snapshot
# at record time (`ensure_snapshot`) — anything slower means nobody was there.
TEAMS_MAX_LAG_DAYS = 2

DEFAULTS = {"deadline": "2026-11-05", "final_test_set": "2026-10-22"}
ABOUT = (
    "Generated by scripts/standings.py from private/submissions/*.status.json and the board "
    "snapshots in ~/data/sidechain/vcc2026/leaderboards/ — do not edit by hand; run the script "
    "(log_submission.py runs it after recording an entry). rank is from the first snapshot "
    "that contains the entry (the board re-ranks continuously, so a later look is not the rank "
    "when scored) and teams is that snapshot's full team count, not the ~50 rows the page "
    "embeds; an entry below the embed takes its status record's scoring-time rank. The README "
    "table between the standings markers is the same rows. class is the entry's declared "
    "intent -- contender (aimed at the score) or probe (spent to answer a question) -- "
    "recorded when the entry was logged, before its score was known; class_retro marks the "
    "entries classed after the fact, because the field postdates them. NOTHING is filtered "
    "on class: it decides only how an entry is drawn, because one off-scale probe on a "
    "shared axis destroys the resolution of every other bar."
)


def series_name(model_name: str, stem: str) -> str:
    m = SERIES_RE.search(model_name or "")
    if m:
        return m.group(1)
    return ALIASES.get(stem, stem)


def _lag_days(stamp: str, submitted: str) -> float | None:
    """Days between an entry being scored and a snapshot being taken, or None if unreadable."""
    fmt = "%Y%m%dT%H%MZ"
    try:
        taken = dt.datetime.strptime(stamp, fmt)
        scored = dt.datetime.strptime(submitted, fmt)
    except ValueError:
        return None
    return (taken - scored).total_seconds() / 86400


def _snapshot_stamp(iso_date: str) -> str:
    """An ISO submission date as a snapshot-filename stamp (YYYYMMDDTHHMMZ), for ordering."""
    d = iso_date or ""
    return f"{d[:4]}{d[5:7]}{d[8:10]}T{d[11:13]}{d[14:16]}Z" if len(d) >= 16 else ""


def load_rows(subs_dir: Path, snaps_dir: Path) -> list[dict]:
    NAME_WARNINGS.clear()
    CLASS_WARNINGS.clear()
    snapshots = []
    for f in sorted(snaps_dir.glob("lb_*.json")):
        d = json.loads(f.read_text())
        live = d.get("live", {})
        entries = live.get("entries", [])
        snapshots.append({
            "stamp": d.get("fetched_utc", ""),
            "ranks": {e["id"]: e["rank"] for e in entries},
            # The page embeds only its top ~50 teams; `total` is the whole field.
            # Old snapshots that predate the divergence have total == len(entries).
            "teams": live.get("total") or len(entries),
        })

    rows = []
    for f in sorted(subs_dir.glob("*.status.json")):
        s = json.loads(f.read_text())
        if s.get("score_avg") is None and isinstance(s.get("scores"), dict):
            # A superseded probe's record is the `vcc submit --wait --json` output --
            # the live status endpoint serves only a team's LATEST validation entry,
            # so log_submission.py saves that shape via --status-file (2026-08-30,
            # SER-3afgn). It nests the members under "scores" and omits
            # submission_date; flatten, and let the record's own filing date stand
            # in for the missing stamp (midnight, so the first same-day snapshot
            # supplies the board size).
            s = {**s, **s["scores"]}
            s.setdefault("submission_date", f.name.split("_", 1)[0] + "T00:00")
        if s.get("score_avg") is None:
            continue  # failed or unscored submissions stay out of the table
        stem = f.name.split("_", 1)[1].removesuffix(".status.json")
        rank = teams = None
        for snap in snapshots:
            if s["entry_id"] in snap["ranks"]:
                rank, teams = snap["ranks"][s["entry_id"]], snap["teams"]
                break
        # The frozen scoring-time rank wins over the record's live `rank`: a re-record
        # (a live `vcc status` fetch, to recover a missing submission_date) sees a board
        # that has re-ranked since. SER-6aefn drifted 245 -> 244 that way on 2026-09-12.
        scored_rank = s.get("sidechain_rank_when_scored")
        if scored_rank is None:
            scored_rank = s.get("rank")
        if rank is None and scored_rank is not None:
            # Ranked below the embed, so no snapshot will ever contain it. The
            # status record carries Arc's rank at fetch time (log_submission.py
            # fetches right after scoring); board size from the first snapshot
            # after the submission — but only if it is within TEAMS_MAX_LAG_DAYS, because a
            # later one measures a field the entry never competed in.
            rank = scored_rank
            submitted = _snapshot_stamp(s.get("submission_date") or "")
            witness = next((sn for sn in snapshots if sn["stamp"] >= submitted), None)
            lag = _lag_days(witness["stamp"], submitted) if witness else None
            teams = witness["teams"] if lag is not None and lag <= TEAMS_MAX_LAG_DAYS else None
        if not SERIES_RE.search(s.get("model_name") or "") and stem not in ALIASES:
            NAME_WARNINGS.append(
                f"{f.name}: board name {(s.get('model_name') or '(none)')!r} carries no "
                "series name and the stem has no ALIASES entry -- falling back to the raw "
                "stem; add the alias or fix the record (ADR 0005)")
        # Two entries reached the board before cards were required (ADR 0005) and the
        # board cannot be backfilled -- a `<record>.card.txt` beside the status record
        # supplies the card for OUR surfaces only, flagged so the site labels it.
        card = (s.get("description") or "").strip()
        card_retro = False
        if not card:
            side = f.with_name(f.name.replace(".status.json", ".card.txt"))
            if side.exists():
                card, card_retro = side.read_text().strip(), True
        klass = s.get("sidechain_class")
        if klass not in CLASSES:
            if (s.get("submission_date") or "")[:10] >= CLASS_REQUIRED_SINCE:
                CLASS_WARNINGS.append(
                    f"{f.name}: sidechain_class is {klass!r}, not one of {CLASSES}, on an entry "
                    f"dated {CLASS_REQUIRED_SINCE} or later -- reading it as a contender. Record "
                    "it with log_submission.py --class, which asks before the score is known")
            klass = "contender"
        rows.append({
            "_submitted": s.get("submission_date") or "",
            "date": (s.get("submission_date") or "")[:10],
            "name": series_name(s.get("model_name", ""), stem),
            "board_name": s.get("model_name") or stem,
            "overall": round(s["score_avg"], 4),
            "rank": rank,
            "teams": teams,
            "card": card,
            "card_retro": card_retro,
            "class": klass,
            "class_retro": bool(s.get("sidechain_class_retro")),
        })
    rows.sort(key=lambda r: r.pop("_submitted"))  # submission order; the key leaves the output
    return rows


def rank_label(rank, teams) -> str:
    return (f"#{rank} of {teams}" if teams else f"#{rank}") if rank else "—"


def readme_block(rows: list[dict]) -> str:
    # Only probes are marked. Tagging the other ten "contender" would be ten rows of noise
    # for a word that is true by default; the prose above the table carries the rule.
    out = ["| date (UTC) | submission | overall | rank when scored |", "|---|---|---|---|"]
    for r in rows:
        cell = f"`{r['board_name']}`"
        if r["name"] not in r["board_name"]:
            cell += f" (`{r['name']}`)"
        if r["class"] == "probe":
            cell += " · **probe**"
        out.append(f"| {r['date']} | {cell} | {r['overall']:.4f} | {rank_label(r['rank'], r['teams'])} |")
    return "\n".join(out)


def render(rows: list[dict]) -> tuple[str, str]:
    """(new README text, new site JSON text)."""
    text = README.read_text()
    if BEGIN not in text or END not in text:
        sys.exit(f"README.md is missing the {BEGIN} / {END} markers")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    readme = f"{head}{BEGIN}\n{readme_block(rows)}\n{END}{tail}"

    keep = DEFAULTS | {
        k: v for k, v in (json.loads(SITE_JSON.read_text()) if SITE_JSON.exists() else {}).items()
        if k in DEFAULTS
    }
    payload = {"_about": ABOUT, **keep, "entries": rows}
    return readme, json.dumps(payload, indent=2) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submissions", type=Path, default=ROOT / "private" / "submissions")
    ap.add_argument("--snapshots", type=Path,
                    default=Path.home() / "data" / "sidechain" / "vcc2026" / "leaderboards")
    ap.add_argument("--check", action="store_true", help="report drift instead of writing")
    args = ap.parse_args()

    rows = load_rows(args.submissions, args.snapshots)
    if not rows:
        sys.exit("no scored submissions found — refusing to write empty tables")
    for w in NAME_WARNINGS + CLASS_WARNINGS:
        print(f"warning: {w}", file=sys.stderr)
    readme, site = render(rows)

    drift = []
    if readme != README.read_text():
        drift.append("README.md")
    if not SITE_JSON.exists() or site != SITE_JSON.read_text():
        drift.append("site/data/submissions.json")
    if args.check:
        print(f"{len(rows)} scored entries; " + (f"OUT OF DATE: {', '.join(drift)}" if drift else "both outputs current"))
        return 1 if drift or NAME_WARNINGS or CLASS_WARNINGS else 0
    README.write_text(readme)
    SITE_JSON.write_text(site)
    for r in rows:
        print(f"{r['date']}  {r['name']:10s} {r['class']:9s} {r['overall']:8.4f}  "
              f"{rank_label(r['rank'], r['teams'])}")
    missing = [r["name"] for r in rows if r["rank"] and not r["teams"]]
    if missing:
        print(f"note: no field size for {', '.join(missing)} -- no board snapshot was taken "
              "after they scored, so the row reads '#<rank>' alone. Not recoverable later: a "
              "newer snapshot ranks them against a bigger field. log_submission.py takes the "
              "snapshot for every new entry (ensure_snapshot).")
    print(f"wrote README.md + site/data/submissions.json ({len(rows)} rows"
          + (f"; updated: {', '.join(drift)})" if drift else "; no change)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
