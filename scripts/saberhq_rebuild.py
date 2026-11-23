"""Start saberh.com's rebuild when a post in this site's live feed has no card on saberh.com/blog/ yet.

    python3 scripts/saberhq_rebuild.py --guids site/public/posts/index.xml   # print a built feed's post URLs
    python3 scripts/saberhq_rebuild.py [--expect "URL ..."] [--dry-run]      # wait, compare, start if needed

Reads the live feed and saberh.com/blog/ over HTTPS; with --expect, waits until the feed lists those URLs. Starts
`gh workflow run` on saberhq/saberhq.github.io with GH_TOKEN. Writes nothing; always exits 0.
Notes: private/code-notes/scripts/saberhq_rebuild.py.md
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import urllib.request

FEED = "https://saberh.com/sidechain/posts/index.xml"
BLOG = "https://saberh.com/blog/"
REPO = "saberhq/saberhq.github.io"
WORKFLOW = "hugo.yml"
AGENT = "sidechain-site-deploy"


def guids(feed: str) -> list[str]:
    """The <guid> values of an RSS feed, in order."""
    return [g.strip() for g in re.findall(r"<guid[^>]*>([^<]+)</guid>", feed)]


def missing(feed_guids: list[str], blog_html: str) -> list[str]:
    """The guids that do not appear in the blog page's HTML."""
    return [g for g in feed_guids if g not in blog_html]


def fetch(url: str, timeout: float = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def say(msg: str, warn: bool = False) -> None:
    if warn and os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::warning::{msg}")
    else:
        print(("warning: " if warn else "") + msg)


def wait_for(expect: list[str], every: float, limit: float) -> list[str] | None:
    """The live feed's guids once they include every expected one; None after `limit` seconds."""
    start = time.monotonic()
    while True:
        try:
            live = guids(fetch(FEED))
            gap = [g for g in expect if g not in live]
            if not gap:
                return live
            say(f"the live feed lacks {len(gap)} of {len(expect)} posts; checking again in {every:.0f} s")
        except OSError as e:
            say(f"cannot read {FEED}: {e}")
        if time.monotonic() - start + every > limit:
            return None
        time.sleep(every)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--guids", metavar="FEED_FILE", help="print the post URLs of a built feed file, one line, and stop")
    ap.add_argument("--expect", default="", help="space-separated post URLs the live feed must list before comparing")
    ap.add_argument("--every", type=float, default=30, help="seconds between reads of the live feed")
    ap.add_argument("--limit", type=float, default=900, help="seconds to wait for the live feed before giving up")
    ap.add_argument("--dry-run", action="store_true", help="say what would start, start nothing")
    args = ap.parse_args()

    if args.guids:
        with open(args.guids, encoding="utf-8") as f:
            print(" ".join(guids(f.read())))
        return 0

    expect = args.expect.split()
    live = wait_for(expect, args.every, args.limit) if expect else None
    if expect and live is None:
        say(f"the live feed did not list every published post within {args.limit:.0f} s; nothing started", warn=True)
        return 0
    try:
        if live is None:
            live = guids(fetch(FEED))
        blog = fetch(f"{BLOG}?fresh={int(time.time())}")
    except OSError as e:
        say(f"cannot read the feed or the blog: {e}; nothing started", warn=True)
        return 0

    new = missing(live, blog)
    if not new:
        print("no new post, nothing to start")
        return 0
    print("not on saberh.com/blog/ yet: " + " ".join(new))
    cmd = ["gh", "workflow", "run", WORKFLOW, "--repo", REPO, "--ref", "main"]
    if args.dry_run:
        print("dry run, would start: " + " ".join(cmd))
        return 0
    if not os.environ.get("GH_TOKEN"):
        say("GH_TOKEN is not set (the SABERHQ_SITE_DISPATCH secret); saberh.com's rebuild not started", warn=True)
        return 0
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        say(f"starting saberh.com's rebuild failed: {e}", warn=True)
        return 0
    if r.returncode != 0:
        say(f"starting saberh.com's rebuild failed: {(r.stderr or r.stdout).strip()}", warn=True)
        return 0
    print(f"started {WORKFLOW} on {REPO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
