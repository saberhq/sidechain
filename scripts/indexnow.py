"""Tell the IndexNow search engines (Bing, Yandex and others) which of this site's pages a deploy published.

    python3 scripts/indexnow.py --urls site/public/sitemap.xml          # print a built sitemap's page URLs
    python3 scripts/indexnow.py --submit "URL ..." [--wait 600] [--dry-run]

Reads the key file in site/static/ and, with --submit, posts the URLs to api.indexnow.org. Writes nothing; always
exits 0.
Notes: private/code-notes/scripts/indexnow.py.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "site" / "static"
BASE = "https://saberh.com/sidechain/"
ENDPOINT = "https://api.indexnow.org/indexnow"


def locs(sitemap: str) -> list[str]:
    """The <loc> values of a sitemap, in order."""
    return [u.strip() for u in re.findall(r"<loc>([^<]+)</loc>", sitemap)]


def key(static: Path = STATIC) -> str | None:
    """The key whose file `<key>.txt` in site/static/ holds the key itself."""
    for f in sorted(static.glob("*.txt")):
        if re.fullmatch(r"[0-9a-f]{32}", f.stem) and f.read_text().strip() == f.stem:
            return f.stem
    return None


def payload(urls: list[str], k: str) -> dict:
    """The IndexNow request body for URLs under BASE."""
    return {"host": "saberh.com", "key": k, "keyLocation": f"{BASE}{k}.txt",
            "urlList": [u for u in urls if u.startswith(BASE)]}


def say(msg: str, warn: bool = False) -> None:
    if warn and os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::warning::{msg}")
    else:
        print(("warning: " if warn else "") + msg)


def submit(body: dict, timeout: float = 30) -> int:
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urls", metavar="SITEMAP", help="print the page URLs of a built sitemap.xml, space-separated")
    ap.add_argument("--submit", metavar="URLS", help="space-separated page URLs to submit")
    ap.add_argument("--wait", type=float, default=0, help="seconds to wait before submitting")
    ap.add_argument("--dry-run", action="store_true", help="print the request instead of sending it")
    a = ap.parse_args(argv)
    if a.urls:
        print(" ".join(locs(Path(a.urls).read_text())))
        return 0
    urls = (a.submit or "").split()
    k = key()
    if not urls or not k:
        say("no URLs" if not urls else "no key file in site/static/", warn=True)
        return 0
    body = payload(urls, k)
    if a.dry_run:
        print(json.dumps(body, indent=1))
        return 0
    if a.wait:
        say(f"waiting {a.wait:.0f} s for the deploy to reach the live site")
        time.sleep(a.wait)
    try:
        code = submit(body)
    except OSError as e:
        say(f"IndexNow unreachable: {e}", warn=True)
        return 0
    if code in (200, 202):
        say(f"IndexNow took {len(body['urlList'])} URLs (HTTP {code})")
    else:
        say(f"IndexNow answered HTTP {code}", warn=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
