"""Snapshot the live Virtual Cell Challenge leaderboard.

    uv run python -m sidechain.eval.leaderboard                 # print top 20, save JSON
    uv run python -m sidechain.eval.leaderboard --top 50 --out ~/data/sidechain/vcc2026/leaderboards

The leaderboard page is a Next.js app that renders "Loading..." to any fetcher
that does not run JavaScript -- but the server still embeds the board in the
page as React Server Component (RSC) data, a series of
``self.__next_f.push([1, "..."])`` script calls. This module decodes those and
pulls out the three entry arrays (live / final / generalist). It is the same
trick that recovered the 2025 boards from the Wayback Machine
(``~/data/sidechain/vcc2025/leaderboards/README.txt``).

Only the top ~50 entries are embedded; ``*NumEntries`` carries the full count.
Scores are Arc's, untouched: ``scoreAvg`` is the overall (unweighted mean of
the six scaled members), ``scorePds`` ... ``scoreJac`` the scaled members, and
``pdsCosine`` ... ``deWilcoxonSigJaccard`` the raw cell-eval2 values.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path

LEADERBOARD_URL = "https://virtualcellchallenge.org/leaderboard"
USER_AGENT = "Mozilla/5.0 (sidechain leaderboard snapshot)"
DEFAULT_OUT = Path.home() / "data" / "sidechain" / "vcc2026" / "leaderboards"

BOARDS = ("Live", "Final", "Generalist")

# The six scaled members in the order the CLI prints them, plus their raw column.
SCALED = [
    ("pds", "scorePds", "pdsCosine"),
    ("mse", "scoreMse", "exprMseUnbiasedCappedNorm"),
    ("nmae", "scoreNmae", "deWilcoxonLfcNmae"),
    ("fid", "scoreFid", "deWilcoxonDirectionFidelityYieldRaw"),
    ("reach", "scoreReach", "deWilcoxonDirectionReachRaw"),
    ("jac", "scoreJac", "deWilcoxonSigJaccard"),
]

_PUSH = re.compile(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)')


def fetch(url: str = LEADERBOARD_URL, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def decode_rsc(html: str) -> str:
    """Concatenate the decoded RSC push chunks. Each is a JS string literal, which
    is close enough to a JSON string literal for json.loads to unescape it."""
    out = []
    for chunk in _PUSH.findall(html):
        try:
            out.append(json.loads(f'"{chunk}"'))
        except json.JSONDecodeError:
            out.append(chunk.encode().decode("unicode_escape", errors="replace"))
    return "\n".join(out)


def _extract_value(payload: str, key: str):
    """The JSON value following ``"key":`` in the decoded payload, or None."""
    m = re.search(rf'"{re.escape(key)}":', payload)
    if not m:
        return None
    value, _ = json.JSONDecoder().raw_decode(payload, m.end())
    return value


def parse_boards(html: str) -> dict:
    """{'live': {'entries': [...], 'total': int}, 'final': ..., 'generalist': ...}"""
    payload = decode_rsc(html)
    boards = {}
    for name in BOARDS:
        key = "initialLiveLeaderboardEntries" if name == "Live" else f"initial{name}LeaderboardEntries"
        if name == "Generalist":
            key = "initialGeneralistEntries"
        entries = _extract_value(payload, key)
        total = _extract_value(payload, key.replace("Entries", "NumEntries"))
        if entries is None:
            continue
        boards[name.lower()] = {"entries": entries, "total": total}
    if not boards:
        raise ValueError("no leaderboard arrays found in the page -- did the site's payload change?")
    return boards


def format_table(entries: list[dict], top: int = 20) -> str:
    head = f"{'#':>3} {'overall':>8} " + " ".join(f"{k:>7}" for k, _, _ in SCALED) + "  team / model"
    lines = [head, "-" * len(head)]
    for e in entries[:top]:
        cells = [f"{e.get('rank', '?'):>3}", f"{e.get('scoreAvg', float('nan')):>8.4f}"]
        for _, scaled, _ in SCALED:
            v = e.get(scaled)
            cells.append(f"{v:>7.3f}" if isinstance(v, (int, float)) else f"{'-':>7}")
        model = e.get("modelName") or ""
        lines.append(" ".join(cells) + f"  {e.get('teamName', '?')} / {model}")
    return "\n".join(lines)


def snapshot(out_dir: Path = DEFAULT_OUT, top: int = 20, url: str = LEADERBOARD_URL) -> Path:
    html = fetch(url)
    boards = parse_boards(html)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%MZ")
    out_dir = out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lb_{stamp}.json"
    path.write_text(json.dumps({"fetched_utc": stamp, "url": url, **boards}, indent=1))
    for name, board in boards.items():
        n = len(board["entries"])
        print(f"\n== {name} board: {board['total']} entries total, {n} embedded ==")
        if n:
            print(format_table(board["entries"], top))
    print(f"\nsaved {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory for the JSON snapshot")
    ap.add_argument("--top", type=int, default=20, help="rows to print per board")
    ap.add_argument("--url", default=LEADERBOARD_URL)
    args = ap.parse_args(argv)
    try:
        snapshot(args.out, args.top, args.url)
    except Exception as exc:  # noqa: BLE001 - a CLI; report and exit non-zero
        print(f"leaderboard: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
