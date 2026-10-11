#!/usr/bin/env python3
"""Is a box finished and safe to end? Reads the box's done marker over ssh and prints the exact commands that end it. It never runs them:
the delete or the stop is a decision for whoever holds the box (no timer, no kill, no self-delete), and a finished box is ended in the same
turn it is found finished.

    box_done.py <box> --marker ~/t111/t111b.done.json [--stoppable] [--fetched]

The marker is the JSON a kit writes when its queue ends. It is safe when `safe_to_delete` (or `safe_to_stop`) is true and, when the marker
lists what is not yet in Lamin (`not_in_lamin`), that list is empty. `--fetched` says the results are already home; without it the
commands are preceded by the reminder to fetch. Exit code 0 only when the box may be ended.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def verdict(marker, box, stoppable=False, fetched=False):
    """(ready, lines) for a parsed done marker."""
    flag = marker.get("safe_to_delete", marker.get("safe_to_stop"))
    missing = marker.get("not_in_lamin") or []
    lines = []
    if flag is not True:
        lines.append(
            f"NOT READY: the marker says safe={flag!r}; read it before touching {box}"
        )
    if missing:
        n = len(missing) if isinstance(missing, list) else missing
        lines.append(
            f"NOT READY: {n} file(s) are not in Lamin yet; they exist only on {box}"
        )
    if lines:
        return False, lines
    if not fetched:
        lines.append(
            f"first fetch the results and compare them with Lamin: the box's disk goes with it ({box})"
        )
    if stoppable:
        lines += [
            f"brev stop {box}",
            f"ledger.py box pause {box}    # stopped, disk kept; `box stop {box}` when it is finally deleted",
        ]
    else:
        lines += [
            f"scripts/brev_key.sh delete {box}",
            f"ledger.py box stop {box}",
            "brev ls    # must not list it",
        ]
    return True, lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("box")
    ap.add_argument(
        "--marker", required=True, help="path of the done marker on the box"
    )
    ap.add_argument("--stoppable", action="store_true")
    ap.add_argument("--fetched", action="store_true")
    a = ap.parse_args(argv)
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-T", a.box, f"cat {a.marker}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0 or not r.stdout.strip():
        print(
            f"NOT READY: no done marker at {a.marker} on {a.box} (ssh rc {r.returncode}): the queue is still running or the box is gone"
        )
        return 2
    try:
        marker = json.loads(r.stdout)
    except ValueError:
        print(f"NOT READY: {a.marker} is not JSON")
        return 2
    ready, lines = verdict(marker, a.box, a.stoppable, a.fetched)
    print("READY" if ready else "NOT READY")
    for ln in lines:
        print(ln)
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
