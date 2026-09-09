#!/usr/bin/env python
"""Regenerate the local-mirror inventory report from what is on disk.

    .venv/bin/python scripts/mirror_inventory.py            # rewrite the report
    .venv/bin/python scripts/mirror_inventory.py --full     # every arm, not the top 15
    .venv/bin/python scripts/mirror_inventory.py --stdout    # print, write nothing

The report is GENERATED. Never hand-edit it -- fix this script or the run dirs
(the standings rule, `CLAUDE.md` § House rules). It reads only
`~/data/sidechain/runs/mirror/`, which lives outside both repos.

What a "mirror" is here: a directory holding a cell-eval2 `bundle/` (the scored
reference built from one held-out line's real cells) plus one subdirectory per
scored arm. Scores are NEVER comparable across mirrors -- each divides by its own
(replicate - baseline) gap -- so the gap is printed as the raw<->scaled converter:
`raw_delta = scaled_delta * gap`.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

MIRRORS = Path("~/data/sidechain/runs/mirror").expanduser()
OUT = Path(__file__).resolve().parents[1] / "private/reports/14_local_mirror_inventory.md"

# fold-name stem -> the line whose cells the bundle was built from
LINE = {
    "loco_hct116": "HCT116 (X-Atlas)", "loco_hek293t": "HEK293T (X-Atlas)",
    "loco_k562gwps": "K562 genome-wide", "loco_k562gwps_union": "K562 genome-wide",
    "loco_k562_essential": "K562 essential", "loco_jurkat": "Jurkat",
    "hepg2_flowtest": "HepG2",
}
SUFFIX = {"_pdex": "CPU/pdex", "_ch272": "challenge 272", "_union": "union panel",
          "_d3": "depth probe", "_rule": "rule probe"}
KNOBS = ("alpha", "gamma", "var_floor", "coverage_tiers", "similarity_beta",
         "transfer_floor", "dispersion", "emit_lambda", "shrinkage")


def held_out(name: str) -> str:
    for stem in sorted(LINE, key=len, reverse=True):
        if name.startswith(stem):
            return LINE[stem]
    return "—"


def shape(name: str) -> str:
    return ", ".join(v for k, v in SUFFIX.items() if name.endswith(k)) or "—"


def raw_mean(arm: Path, metric: str = "pds_cosine") -> float | None:
    f = arm / "run" / "agg_results.csv"
    if not f.exists():
        return None
    for row in csv.DictReader(f.open()):
        if row.get("statistic") == "mean" and row.get(metric):
            try:
                return float(row[metric])
            except ValueError:
                return None
    return None


def anchors(arm: Path, metric: str = "pds_cosine") -> tuple[float | None, float | None]:
    f = arm / "scored.csv"
    if not f.exists():
        return None, None
    for row in csv.DictReader(f.open()):
        if row.get("metric") == metric:
            try:
                return float(row["from_baseline"]), float(row["from_replicate"])
            except (ValueError, KeyError):
                return None, None
    return None, None


def knob_str(build: dict) -> str:
    bits = []
    for k in KNOBS:
        v = build.get(k)
        if v is None or v in ("", [], "none", False):
            continue                       # absent is absent, for `dispersion` too
        if k in ("alpha", "gamma") and v == 1.0:
            continue                       # 1.0 is "knob off" for both
        if k == "emit_lambda" and v == 0.0:
            continue
        if k == "similarity_beta" and v == 0.0:
            continue
        if k == "transfer_floor" and isinstance(v, dict):
            if not any(v.values()):
                continue               # every source floored at 0.0 is the knob switched off
            v = "/".join(f"{x:g}" for x in v.values())
        if k == "coverage_tiers" and isinstance(v, list):
            v = ",".join(f"{int(a)}:{b:g}" for a, b in v)
        bits.append(f"{k.split('_')[0]}={v}")
    return " ".join(bits) or "—"


def collect() -> list[dict]:
    folds = []
    for d in sorted(MIRRORS.iterdir()):
        man = d / "bundle" / "manifest.json"
        if not d.is_dir() or not man.exists():
            continue
        m = json.loads(man.read_text())
        arms = []
        for a in sorted(x for x in d.iterdir() if (x / "summary.json").exists()):
            s = json.loads((a / "summary.json").read_text())
            arms.append({"name": a.name, "overall": s.get("overall"),
                         "raw_pds": raw_mean(a), "build": s.get("build", {}),
                         "sources": s.get("sources", {})})
        arms.sort(key=lambda x: (x["overall"] is None, -(x["overall"] or 0)))
        ref = next((a for a in arms if a["name"] == "afn"), arms[0] if arms else None)
        b, r = anchors(d / ref["name"]) if ref else (None, None)
        folds.append({
            "name": d.name, "line": held_out(d.name), "shape": shape(d.name),
            "backend": m.get("resolved_de_backend", "?"), "device": m.get("resolved_device", "?"),
            "cell_eval2": m.get("cell_eval2_version", "?"),
            "targets": (ref or {}).get("build", {}).get("perturbations"),
            "genes": (ref or {}).get("build", {}).get("genes"),
            "cells": (ref or {}).get("build", {}).get("cells"),
            "baseline": b, "replicate": r,
            "gap": (r - b) if (b is not None and r is not None) else None,
            "arms": arms, "ref": (ref or {}).get("name"),
        })
    return folds


def num(v, nd=4):
    return "—" if v is None else (f"{v:,}" if isinstance(v, int) else f"{v:.{nd}f}")


def render(folds: list[dict], full: bool) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [
        "# Report 14 — Local mirror inventory",
        "",
        "> **GENERATED — never hand-edit.** Rebuild with",
        "> `.venv/bin/python scripts/mirror_inventory.py` (add `--full` for every arm).",
        f"> Source: `~/data/sidechain/runs/mirror/`. Last regenerated **{now}**.",
        "",
        "A *mirror* is one held-out line's cell-eval2 bundle plus the arms scored against it.",
        "**Scores are never comparable across mirrors** — each divides by its own",
        "(replicate − baseline) gap — so the gap is given as the converter:",
        "`raw_delta = scaled_delta × gap`. Corpus and dataset facts live in `reports/07`;",
        "this file is only about what has been *built and scored* locally.",
        "",
        "## The mirrors",
        "",
        "| mirror | held-out line | shape | targets | gene axis | cells | backend · device | pds gap (r−b) | arms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for f in folds:
        L.append(
            f"| `{f['name']}` | {f['line']} | {f['shape']} | {num(f['targets'])} | "
            f"{num(f['genes'])} | {num(f['cells'])} | {f['backend']} · {f['device']} | "
            f"{num(f['gap'])} | {len(f['arms'])} |"
        )
    L += ["", f"**{len(folds)} mirrors, {sum(len(f['arms']) for f in folds)} scored arms.** "
          f"cell-eval2 {sorted({f['cell_eval2'] for f in folds})[0]}.", "",
          "`—` in a cell means the value is not on disk, not that it is zero: the earliest arms "
          "(`loco_k562gwps`'s source comparison) predate the `scored.csv` convention, so they "
          "carry no anchors and no gap, and a mirror with 0 arms has a bundle and nothing scored "
          "against it yet.", ""]

    L += ["## Which lines pool into each mirror", "",
          "| mirror | pooled sources (the arm named in *ref*) |", "|---|---|"]
    for f in folds:
        ref = next((a for a in f["arms"] if a["name"] == f["ref"]), None)
        src = (ref or {}).get("sources") or {}
        names = [Path(s.split(":")[0]).stem for s in src.get("pseudobulk", [])]
        L.append(f"| `{f['name']}` | {', '.join(names) if names else '—'} |")
    L.append("")

    L += ["## Arms, per mirror", "",
          "`overall` is the scaled six-member mean on that fold; `raw pds` is",
          "`pds_cosine` before scaling. Sorted by `overall`.", ""]
    for f in folds:
        shown = f["arms"] if full else f["arms"][:15]
        L += [f"### `{f['name']}` — {f['line']}", "",
              "| arm | knobs | overall | raw pds |", "|---|---|---|---|"]
        for a in shown:
            L.append(f"| `{a['name']}` | {knob_str(a['build'])} | "
                     f"{num(a['overall'])} | {num(a['raw_pds'])} |")
        if len(f["arms"]) > len(shown):
            L.append(f"| … | *{len(f['arms']) - len(shown)} more — `--full`* | | |")
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="list every arm, not the top 15")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing the report")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if not MIRRORS.exists():
        raise SystemExit(f"no mirror directory at {MIRRORS}")
    text = render(collect(), args.full)
    if args.stdout:
        print(text)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
