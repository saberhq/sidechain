#!/usr/bin/env python3
"""Summarise a box run from box_profile.sh's procs.log (also the format of T111's second-pass kit).

    box_profile_report.py procs.log [--json out.json] [--usd-h 2.40]

Prints, in plain lines: how long the run was sampled, the 1-minute load and free memory, GPU use, and for each kind of process
(scoring step, cache build, list design, arm, submit build) how many there were, how long each lived, the CPU each used
(%CPU x elapsed at its last sample, so a process seen once is a lower bound), its cores and its peak resident memory.
Only the 60 largest processes of each minute are logged, so small processes are not in it.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict

KINDS = [  # first match wins
    ("score", re.compile(r"cell-eval2 run")),
    ("cache", re.compile(r"delta-cache-build")),
    ("design", re.compile(r"head_arm\.py")),
    ("build", re.compile(r"sidechain\.submit\.build")),
    ("arm", re.compile(r"sidechain\.eval\.loco|run_loco\.py")),
]
HEAD = re.compile(r"=== (\S+) load (\S+) (\S+) (\S+) memavail_GiB (\d+)")
PROC = re.compile(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(.*)")
CARD = re.compile(r"card (\d+), (\d+) MiB, (\d+) MiB, (\d+) %")


def parse(lines):
    blocks, cur = [], None
    for ln in lines:
        m = HEAD.match(ln)
        if m:
            cur = {
                "ts": m.group(1),
                "load": float(m.group(2)),
                "mem": int(m.group(5)),
                "procs": [],
                "cards": [],
            }
            blocks.append(cur)
            continue
        if cur is None:
            continue
        m = PROC.match(ln)
        if m:
            cur["procs"].append(
                (
                    int(m.group(1)),
                    int(m.group(3)) / 1048576,
                    float(m.group(4)),
                    int(m.group(5)),
                    m.group(6),
                )
            )
            continue
        m = CARD.match(ln)
        if m:
            cur["cards"].append(
                (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
            )
    return blocks


def kind_of(args):
    for name, rx in KINDS:
        if rx.search(args):
            return name
    return None


def summarise(blocks):
    out = {
        "minutes": len(blocks),
        "first": blocks[0]["ts"] if blocks else None,
        "last": blocks[-1]["ts"] if blocks else None,
    }
    if not blocks:
        return out
    loads, mem = [b["load"] for b in blocks], [b["mem"] for b in blocks]
    out["load"] = {
        "mean": st.mean(loads),
        "median": st.median(loads),
        "max": max(loads),
    }
    out["memavail_gib"] = {"median": st.median(mem), "min": min(mem)}
    cards = [b["cards"] for b in blocks if b["cards"]]
    if cards:
        used = [[c[3] for c in cs] for cs in cards]
        out["gpu"] = {
            "cards": len(cards[0]),
            "minutes_any_use": sum(1 for u in used if any(u)),
            "minutes": len(cards),
            "mean_util_per_card": st.mean(sum(u) / len(u) for u in used),
            "peak_card_mib": max(c[1] for cs in cards for c in cs),
            "card_total_mib": cards[0][0][2],
        }
    per = defaultdict(dict)  # kind -> pid -> (etimes, cpu_s, peak_rss)
    for b in blocks:
        for pid, rss, pcpu, et, args in b["procs"]:
            k = kind_of(args)
            if k is None:
                continue
            prev = per[k].get(pid, (0, 0.0, 0.0))
            per[k][pid] = (
                max(prev[0], et),
                max(prev[1], pcpu * et / 100.0) if et >= prev[0] else prev[1],
                max(prev[2], rss),
            )
    kinds = {}
    for k, d in per.items():
        v = list(d.values())
        wall, cpu, rss = [x[0] for x in v], [x[1] for x in v], [x[2] for x in v]
        kinds[k] = {
            "n": len(v),
            "wall_s_median": st.median(wall),
            "wall_s_max": max(wall),
            "cpu_s_median": st.median(cpu),
            "cores_median": st.median(
                c / max(w, 1) for c, w in zip(cpu, wall, strict=True)
            ),
            "rss_gib_median": st.median(rss),
            "rss_gib_max": max(rss),
        }
    out["kinds"] = kinds
    return out


def render(s, usd_h=None):
    if not s["minutes"]:
        return "no samples"
    ls = [
        f"sampled {s['minutes']} minutes, {s['first']} to {s['last']}",
        f"load: mean {s['load']['mean']:.1f}, median {s['load']['median']:.1f}, max {s['load']['max']:.1f}",
        f"free memory: median {s['memavail_gib']['median']:.0f} GiB, lowest {s['memavail_gib']['min']:.0f} GiB",
    ]
    if "gpu" in s:
        g = s["gpu"]
        ls.append(
            f"GPU: {g['cards']} cards, any use in {g['minutes_any_use']} of {g['minutes']} minutes, mean {g['mean_util_per_card']:.1f} % a card, "
            f"peak {g['peak_card_mib'] / 1024:.1f} of {g['card_total_mib'] / 1024:.0f} GiB on one card"
        )
    for k, v in sorted(s.get("kinds", {}).items()):
        ls.append(
            f"{k}: {v['n']} processes; wall median {v['wall_s_median']:.0f} s, max {v['wall_s_max']:.0f} s; CPU median {v['cpu_s_median']:.0f} s "
            f"({v['cores_median']:.1f} cores); peak memory median {v['rss_gib_median']:.1f} GiB, max {v['rss_gib_max']:.1f} GiB"
        )
    if usd_h:
        ls.append(
            f"sampled span costs about ${usd_h * s['minutes'] / 60:.2f} at ${usd_h}/h"
        )
    return "\n".join(ls)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("procs_log")
    ap.add_argument("--json")
    ap.add_argument("--usd-h", type=float)
    a = ap.parse_args(argv)
    with open(a.procs_log, errors="replace") as fh:
        s = summarise(parse(fh))
    print(render(s, a.usd_h))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(s, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
