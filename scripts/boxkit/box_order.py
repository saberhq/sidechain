#!/usr/bin/env python3
"""Order a box queue longest-first within each layer of dependencies, so the widest arms start early and the last worker does not
finish 45 minutes after the rest (T98's tail). A layer is a group whose lines may start only after the layer before it (caches, then
template passes, then designs, then scored arms); inside a layer the line with the longest estimated minutes goes first.

    box_order.py queue_plan.txt --est loco_hek293t=59,loco_hct116=52,loco_k562gwps_union=21 [--layers tmpl,design,list]
                [--sep '|'] [--est-field 0] [--layer-field 3]

Reads one queue line per row (default `fold|name|arm|opts`); the estimate is looked up by the field `--est-field`, the layer by the first of
`--layers` that the field `--layer-field` starts with (`design:0/4` starts with `design`). Lines whose estimate is unknown keep their
place after the known ones of their layer. The order inside equal estimates is the file's own, so a rerun is stable. The output goes to stdout.
"""

from __future__ import annotations

import argparse
import sys


def longest_first(lines, minutes_of, layer_of):
    """Stable sort: layer ascending, then estimated minutes descending (unknown last), then the original position."""
    keyed = []
    for i, ln in enumerate(lines):
        m = minutes_of(ln)
        keyed.append((layer_of(ln), 0 if m is not None else 1, -(m or 0.0), i, ln))
    return [k[-1] for k in sorted(keyed)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("queue")
    ap.add_argument("--est", required=True, help="key=minutes,key=minutes")
    ap.add_argument(
        "--layers",
        default="",
        help="comma list of layer prefixes in order (default: one layer)",
    )
    ap.add_argument("--sep", default="|")
    ap.add_argument("--est-field", type=int, default=0)
    ap.add_argument("--layer-field", type=int, default=3)
    a = ap.parse_args(argv)
    est = {k: float(v) for k, v in (kv.split("=") for kv in a.est.split(",") if kv)}
    layers = [x for x in a.layers.split(",") if x]

    def field(ln, n):
        parts = ln.rstrip("\n").split(a.sep)
        return parts[n] if n < len(parts) else ""

    def minutes_of(ln):
        return est.get(field(ln, a.est_field))

    def layer_of(ln):
        f = field(ln, a.layer_field)
        for i, pre in enumerate(layers):
            if f.startswith(pre):
                return i
        return len(layers)

    with open(a.queue) as fh:
        lines = [
            ln for ln in fh.read().splitlines() if ln.strip() and not ln.startswith("#")
        ]
    sys.stdout.write("\n".join(longest_first(lines, minutes_of, layer_of)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
