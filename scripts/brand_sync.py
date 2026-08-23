"""Keep site/assets/tokens/ identical to saberhq.com's design tokens.

    uv run python scripts/brand_sync.py          # copy the four token files across
    uv run python scripts/brand_sync.py --check  # exit 1 if they differ, print the diff

The Sidechain page is a sibling of the personal site: same colours, type and spacing,
sourced from the same four files. They are copied rather than mounted because the site
builds in GitHub Actions from this repo alone. A copy can drift, so this script exists:
run it after any change to the brand on the personal site, or with --check to ask.
"""
from __future__ import annotations

import argparse
import difflib
import os
import sys
from pathlib import Path

TOKENS = ("fonts.css", "colors.css", "typography.css", "spacing.css")
HERE = Path(__file__).resolve().parent.parent
DEST = HERE / "site" / "assets" / "tokens"
SOURCE = Path(os.environ.get("SABERHQ_SITE", Path.home() / "code" / "saberhq.github.io")) / "assets" / "tokens"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report drift instead of copying")
    args = ap.parse_args()

    if not SOURCE.is_dir():
        print(f"personal site not found at {SOURCE.parent.parent} (set SABERHQ_SITE)", file=sys.stderr)
        return 2

    drift = 0
    for name in TOKENS:
        src, dst = SOURCE / name, DEST / name
        a = src.read_text()
        b = dst.read_text() if dst.exists() else ""
        if a == b:
            print(f"same     {name}")
            continue
        drift += 1
        if args.check:
            print(f"DIFFERS  {name}")
            sys.stdout.writelines(difflib.unified_diff(
                b.splitlines(True), a.splitlines(True),
                fromfile=f"site/assets/tokens/{name}", tofile=f"saberhq.github.io/assets/tokens/{name}",
            ))
        else:
            DEST.mkdir(parents=True, exist_ok=True)
            dst.write_text(a)
            print(f"updated  {name}")
    if args.check and drift:
        print(f"{drift} token file(s) differ; run without --check to copy them across")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
