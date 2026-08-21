"""Turn the live leaderboard into a shields.io endpoint badge for one team.

    uv run python -m sidechain.eval.badge --team Sidechain --out badges/leaderboard.json
    uv run python -m sidechain.eval.badge --team Sidechain --snapshot lb_20260821T2303Z.json

Emits the JSON that https://img.shields.io/endpoint renders, e.g.
``{"schemaVersion": 1, "label": "live rank", "message": "#9 of 95", "color": "brightgreen"}``,
and prints the message. The GitHub Actions job in ``.github/workflows/leaderboard-badge.yml``
runs this hourly and commits the file to the ``badges`` branch, which the README badge reads.

Stdlib only, like ``leaderboard`` -- the job runs without installing the project.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sidechain.eval.leaderboard import LEADERBOARD_URL, fetch, parse_boards

LABEL = "live rank"


def find_entry(entries: list[dict], team: str) -> dict | None:
    """The team's best-ranked row, matched case-insensitively on ``teamName``."""
    rows = [e for e in entries if (e.get("teamName") or "").strip().lower() == team.strip().lower()]
    if not rows:
        return None
    return min(rows, key=lambda e: e.get("rank") or float("inf"))


def badge(board: dict, team: str) -> dict:
    """shields.io endpoint JSON for ``team`` on one board (``{'entries': [...], 'total': n}``)."""
    entries, total = board["entries"], board["total"]
    entry = find_entry(entries, team)
    if entry is not None and entry.get("rank"):
        rank = int(entry["rank"])
        message = f"#{rank} of {total}"
        color = "brightgreen" if rank <= 10 else "blue"
    elif total and len(entries) < total:
        # Only the top ~50 rows are embedded in the page; absence means "below them".
        message = f"below #{len(entries)} of {total}"
        color = "blue"
    else:
        message = "unranked"
        color = "lightgrey"
    return {"schemaVersion": 1, "label": LABEL, "message": message, "color": color}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", required=True, help="teamName as shown on the board")
    ap.add_argument("--board", default="live", choices=("live", "final", "generalist"))
    ap.add_argument("--out", type=Path, help="write the badge JSON here (default: stdout only)")
    ap.add_argument("--snapshot", type=Path, help="read a saved leaderboard snapshot instead of fetching")
    ap.add_argument("--url", default=LEADERBOARD_URL)
    args = ap.parse_args(argv)
    try:
        if args.snapshot:
            boards = json.loads(args.snapshot.read_text())
        else:
            boards = parse_boards(fetch(args.url))
        if args.board not in boards:
            raise ValueError(f"no {args.board!r} board in the payload")
        result = badge(boards[args.board], args.team)
    except Exception as exc:  # noqa: BLE001 - a CLI; report and exit non-zero, leave any old badge alone
        print(f"badge: {exc}", file=sys.stderr)
        return 1
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result) + "\n")
    print(result["message"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
