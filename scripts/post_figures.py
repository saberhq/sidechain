#!/usr/bin/env python3
"""Draw the write-up figures as inline SVG, from the site's generated data.

Every number a figure draws comes from `site/data/submissions.json` (scored entries,
written by `scripts/standings.py`) or `site/data/coverage.json` (per-source coverage,
written by `python -m sidechain.data.coverage`) — never typed here. The one table that
lives in this file, `SCREENS`, holds paper numbers (cells per published screen), each
row with the paper it came from; those are not pipeline numbers, so they do not belong
in a generated data file.

Style: the site's own tokens, as CSS variables with light-theme fallbacks, so a figure
takes the dark theme on the page and still reads when the file is opened on its own.
One series per panel; carmine (`--mark`) for Sidechain's latest or own point, grey
(`--viz-8`) for the rest, blue (`--viz-2`) only where a second hue is unavoidable; thin
marks, direct labels, no legend for one series, no connecting line through board
entries. Calibration runs stay off any figure that reads as a standings plot (Saber,
2026-10-07) and appear only in the figure that is about them.

    python scripts/post_figures.py --out private/site/drafts/2026-10-the-knobs/ --post knobs
    python scripts/post_figures.py --out <dir> --post corpus
    python scripts/post_figures.py --out <dir> --post board

A post drops a figure in with `{{< svg src="fig-overall.svg" caption="…" >}}`
(`site/layouts/shortcodes/svg.html` inlines the file from the post's bundle).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "site" / "data" / "submissions.json"
COVERAGE = ROOT / "site" / "data" / "coverage.json"

# --- tokens -------------------------------------------------------------------------
INK = "var(--ink, #191613)"
INK2 = "var(--ink-2, #45403A)"
INK3 = "var(--ink-3, #6F6A62)"
RULE = "var(--rule, #E3E0DA)"
RULE_STRONG = "var(--rule-strong, #191613)"
MARK = "var(--mark, #A62B4D)"
GREY = "var(--viz-8, #999999)"
BLUE = "var(--viz-2, #0072B2)"
MONO = "var(--font-mono, 'Geist Mono', ui-monospace, Menlo, monospace)"

MEMBERS = ["pds", "mse", "nmae", "fid", "reach", "jac"]
MEMBER_TITLE = {
    "pds": "pds · which knockdown is this",
    "mse": "mse · distance of the summed profile",
    "nmae": "nmae · fold-change accuracy",
    "fid": "fid · called genes pointing the right way",
    "reach": "reach · how far down the real list",
    "jac": "jac · the same genes called",
}

# Cells per published screen, as released after the authors' own QC; one row per screen,
# the paper and the line in our reading note that states it. Atlases of unperturbed cells
# are a separate list: they are not screens.
SCREENS = [
    # (year, label, cells, source)
    (2022, "Replogle · K562 genome-wide", 1_989_578, "scPerturb's K562 genome-wide file; research/ingest-findings-scperturb.md"),
    (2025, "Nadig · Jurkat", 262_956, "research/reading/nadig-2025-trade-transcriptome-wide-impact.md"),
    (2025, "Nadig · HepG2", 145_473, "research/reading/nadig-2025-trade-transcriptome-wide-impact.md"),
    (2025, "Arc · H1 (VCC 2025)", 221_273, "research/master.md Appendix"),
    (2025, "X-Atlas/Orion · HCT116 + HEK293T", 7_943_468, "measured on the stream; reports/07"),
    (2025, "GWCD4i · primary CD4 T", 33_400_000, "QC-passing cells; research/reading/zhu-2025-gwcd4i.md"),
    (2025, "Song · Jurkat (374-gene readout)", 586_000, "research/ideas/jurkat-same-line-source.md"),
    (2026, "Feng · iPSC pool (genome-wide arm)", 719_204, "configs/datasets.yaml feng_2026_counts"),
    (2026, "Pan · H1", 928_267, "cells kept; research/ideas/pan-2026-h1-crispri-screen.md"),
    (2026, "X-Atlas/Pisces · 16 contexts", 25_600_000, "research/reading/wang-2026-x-cell-pisces.md"),
]
HELD = {"Replogle · K562 genome-wide", "Nadig · Jurkat", "Nadig · HepG2", "Arc · H1 (VCC 2025)",
        "X-Atlas/Orion · HCT116 + HEK293T", "GWCD4i · primary CD4 T", "Feng · iPSC pool (genome-wide arm)", "Pan · H1"}

# The board's tabs, when each was born and when it died (reports/12 addenda; the dates are
# the rounds that added or removed a tab).
TABS = [
    ("Board", "2026-09-02", None),
    ("Ledger", "2026-09-02", "2026-09-18"),
    ("Standing", "2026-09-02", None),
    ("Archive", "2026-09-04", "2026-10-04"),
    ("Tasks", "2026-09-11", None),
    ("Log", "2026-09-11", "2026-10-04"),
    ("Science", "2026-09-14", None),
    ("Docs", "2026-09-16", None),
    ("Ideas", "2026-10-04", None),
    ("Map", "2026-10-04", None),
]
TABS_END = "2026-10-10"


# --- tiny SVG kit -------------------------------------------------------------------
def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def text(x, y, s, size=10, fill=INK2, anchor="start", weight=400, rotate=None, dy=None, title=None):
    tr = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    dyat = f' dy="{dy}"' if dy is not None else ""
    t = f'<title>{esc(title)}</title>' if title else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}"{dyat} font-family="{MONO}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{tr}>{t}{esc(s)}</text>')


def line(x1, y1, x2, y2, stroke=RULE, width=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{d}/>'


def dot(x, y, r=3.2, fill=GREY, ring=True):
    ringat = ' stroke="var(--paper, #FAF9F7)" stroke-width="1.5"' if ring else ""
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"{ringat}/>'


def rect(x, y, w, h, fill=GREY, rx=0, opacity=None):
    op = f' fill-opacity="{opacity}"' if opacity is not None else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" rx="{rx}"{op}/>'


def svg(w, h, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" '
            f'style="max-width:{w}px;height:auto;display:block" role="img" aria-label="{esc(label)}">\n'
            + "\n".join(body) + "\n</svg>\n")


def fmt(v: float, nd=3) -> str:
    return f"{v:.{nd}f}"


def nice_ticks(lo, hi, n=4):
    span = hi - lo
    raw = span / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= n + 1:
            break
    start = math.floor(lo / step) * step
    ticks = []
    t = start
    while t <= hi + 1e-9:
        if t >= lo - 1e-9:
            ticks.append(round(t, 10))
        t += step
    return ticks


# --- data ---------------------------------------------------------------------------
def load_entries(path=SUBMISSIONS):
    d = json.loads(Path(path).read_text())
    return d["entries"]


def normal(entries):
    """Entries that are not calibration runs, in the order the file gives (date order)."""
    return [e for e in entries if not e.get("class")]


def short(name: str) -> str:
    """`SER-17aefhkrsw` → `SER-17`; the letters are the text's job, not the axis's."""
    head = name.split("-")
    if len(head) < 2:
        return name
    num = "".join(ch for ch in head[1] if ch.isdigit())
    return f"{head[0]}-{num}" if num else name


# --- the knobs post -----------------------------------------------------------------
def fig_overall(entries):
    """Overall score per normal entry, in entry order; the latest in carmine; no line."""
    es = normal(entries)
    W, H = 680, 318
    L, R, T, B = 52, 16, 22, 88
    xs = [L + (W - L - R) * (i + 0.5) / len(es) for i in range(len(es))]
    vals = [e["overall"] for e in es]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.12
    y0, y1 = lo - pad, hi + pad
    ticks = nice_ticks(y0, y1, 4)
    def Y(v): return T + (H - T - B) * (y1 - v) / (y1 - y0)
    body = []
    for t in ticks:
        body.append(line(L, Y(t), W - R, Y(t)))
        body.append(text(L - 6, Y(t), fmt(t, 2), 9.5, INK3, "end", dy="0.35em"))
    body.append(line(L, T, L, H - B, RULE_STRONG))
    latest = len(es) - 1
    for i, (e, x, v) in enumerate(zip(es, xs, vals)):
        is_latest = i == latest
        body.append(dot(x, Y(v), 4.2 if is_latest else 3.2, MARK if is_latest else GREY))
        body.append(text(x + 3, H - B + 12, e["name"], 8.5, INK if is_latest else INK3, "end", weight=500 if is_latest else 400, rotate=-50, title=f'{e["name"]} · {e["date"]} · {fmt(v, 4)}'))
    # direct labels on the points the text turns on
    named = {"SER-1", "SER-2", "SER-3fn", "SER-3afn", "SER-10abefnw", "SER-12aefkw", "SER-14aefksw", "SER-17aefhkrsw"}
    for e, x, v in zip(es, xs, vals):
        if e["name"] in named:
            body.append(text(x, Y(v) - 8, fmt(v, 3), 8.5, INK if e["name"].startswith("SER-17") else INK3, "middle"))
    body.append(text(L, 12, "overall score · the plain mean of six scaled members · 0 = the context-mean baseline", 9.5, INK3))
    return svg(W, H, body, "Overall score per Sidechain entry on the Virtual Cell Challenge 2026 board, in entry order")


def fig_members(entries):
    """Six small multiples: each member's scaled score per normal entry, in entry order."""
    es = normal(entries)
    cols, rows = 3, 2
    PW, PH = 214, 150
    GX, GY = 12, 26
    W = cols * PW + (cols - 1) * GX
    H = rows * PH + (rows - 1) * GY + 10
    body = []
    for k, m in enumerate(MEMBERS):
        cx, cy = k % cols, k // cols
        ox, oy = cx * (PW + GX), cy * (PH + GY)
        L, R, T, B = 34, 8, 18, 22
        vals = [e["members"][m]["scaled"] for e in es]
        lo, hi = min(vals + [0]), max(vals + [0])
        pad = (hi - lo) * 0.15 or 0.05
        y0, y1 = lo - pad, hi + pad
        def Y(v, y0=y0, y1=y1, oy=oy, T=T, B=B): return oy + T + (PH - T - B) * (y1 - v) / (y1 - y0)
        xs = [ox + L + (PW - L - R) * (i + 0.5) / len(es) for i in range(len(es))]
        body.append(text(ox + L, oy + 11, MEMBER_TITLE[m], 9, INK2, weight=500))
        for t in nice_ticks(y0, y1, 3):
            body.append(line(ox + L, Y(t), ox + PW - R, Y(t), RULE if abs(t) > 1e-9 else RULE_STRONG, 1 if abs(t) > 1e-9 else 0.8))
            body.append(text(ox + L - 4, Y(t), fmt(t, 2), 8.5, INK3, "end", dy="0.35em"))
        for i, (e, x, v) in enumerate(zip(es, xs, vals)):
            is_latest = i == len(es) - 1
            body.append(dot(x, Y(v), 3.6 if is_latest else 2.6, MARK if is_latest else GREY, ring=False))
        body.append(text(xs[0], oy + PH - 8, short(es[0]["name"]), 8, INK3, "start"))
        body.append(text(xs[-1], oy + PH - 8, short(es[-1]["name"]), 8, INK, "end"))
        body.append(text(xs[-1] + 2, Y(vals[-1]), fmt(vals[-1], 3), 8.5, INK, "start", dy="0.35em") if cx < cols - 1 else text(xs[-1] - 6, Y(vals[-1]) - 7, fmt(vals[-1], 3), 8.5, INK, "end"))
    return svg(W, H, body, "Each of the six scaled members per Sidechain entry, in entry order")


def fig_callset(entries):
    """The trade the shape story turns on: raw fid, reach and jac for SER-14 (artificial
    cells) and SER-15 → 16 → 17 (real-shaped cells with no list, a 300-gene list, a
    1,500-gene list). The three on the right are the pair of calibration runs and the
    entry that followed; this figure is about them, so they are on it."""
    by = {e["name"]: e for e in entries}
    order = [("SER-14aefksw", "SER-14\nartificial"), ("SER-15aefkrsw", "SER-15\nreal cells"),
             ("SER-16aefhkrsw", "SER-16\nlist 300"), ("SER-17aefhkrsw", "SER-17\nlist 1,500")]
    PW, PH, GX = 214, 170, 12
    W = 3 * PW + 2 * GX
    H = PH + 6
    body = []
    for k, m in enumerate(["fid", "reach", "jac"]):
        ox = k * (PW + GX)
        L, R, T, B = 32, 6, 18, 44
        vals = [by[n]["members"][m]["raw"] for n, _ in order]
        lo, hi = min(vals + [0]), max(vals)
        pad = (hi - lo) * 0.18 or 0.05
        y0, y1 = lo - pad, hi + pad
        def Y(v, y0=y0, y1=y1, T=T, B=B): return T + (PH - T - B) * (y1 - v) / (y1 - y0)
        xs = [ox + L + (PW - L - R) * (i + 0.5) / len(order) for i in range(len(order))]
        body.append(text(ox + L, 11, f"raw {m}", 9, INK2, weight=500))
        for t in nice_ticks(y0, y1, 3):
            body.append(line(ox + L, Y(t), ox + PW - R, Y(t), RULE if abs(t) > 1e-9 else RULE_STRONG))
            body.append(text(ox + L - 4, Y(t), fmt(t, 2), 8.5, INK3, "end", dy="0.35em"))
        ref = vals[0]
        body.append(line(xs[0], Y(ref), xs[-1], Y(ref), GREY, 1, dash="3 3"))
        for i, ((n, lab), x, v) in enumerate(zip(order, xs, vals)):
            is_latest = n.startswith("SER-17")
            body.append(dot(x, Y(v), 4 if is_latest else 3.2, MARK if is_latest else GREY))
            body.append(text(x, Y(v) - 8, fmt(v, 3), 8.5, INK if is_latest else INK3, "middle"))
            l1, l2 = lab.split("\n")
            body.append(text(x, PH - B + 14, l1, 8.5, INK if is_latest else INK3, "middle"))
            body.append(text(x, PH - B + 25, l2, 7.5, INK3, "middle"))
    return svg(W, H, body, "Raw fid, reach and jac for SER-14 and the three real-shaped entries SER-15, SER-16, SER-17")


# --- the corpus post ----------------------------------------------------------------
def fig_cells_per_target(coverage_path=COVERAGE):
    """Cells per covered panel target, min – median – max, per source that holds cells."""
    c = json.loads(Path(coverage_path).read_text())
    rows = [s for s in c["sources"] if s.get("cells_per_covered_target")]
    W = 680
    RH = 26
    L, R, T, B = 110, 150, 26, 30
    H = T + RH * len(rows) + B
    lo = min(s["cells_per_covered_target"]["min"] for s in rows)
    hi = max(s["cells_per_covered_target"]["max"] for s in rows)
    x0, x1 = math.log10(lo) - 0.12, math.log10(hi) + 0.12
    def X(v): return L + (W - L - R) * (math.log10(v) - x0) / (x1 - x0)
    body = [text(L, 12, "cells behind each covered target · min – median – max · log scale", 9.5, INK3)]
    for tick in (10, 100, 1000):
        if lo <= tick <= hi:
            body.append(line(X(tick), T, X(tick), H - B))
            body.append(text(X(tick), H - B + 12, f"{tick:,}", 8.5, INK3, "middle"))
    for i, s in enumerate(rows):
        y = T + RH * i + RH / 2
        q = s["cells_per_covered_target"]
        prior = s["role"] == "prior"
        body.append(text(L - 8, y, f'{s["line"]}', 9.5, INK if prior else INK3, "end", dy="0.35em", title=s["name"]))
        body.append(line(X(q["min"]), y, X(q["max"]), y, GREY if prior else RULE, 2))
        body.append(dot(X(q["median"]), y, 3.6, MARK if prior else GREY))
        body.append(text(W - R + 10, y, f'{q["min"]:,} – {q["median"]:,} – {q["max"]:,}', 8.5, INK3, "start", dy="0.35em"))
        body.append(text(X(q["median"]), y - 7, f'{s["panel_covered"]}/{c["panel_size"]}', 8, INK3, "middle"))
    return svg(W, H, body, "Cells per covered challenge target, per corpus: minimum, median and maximum")


def fig_screens(table=SCREENS):
    """Perturb-seq screens by cells released, against the year of the paper; log scale;
    the screens Sidechain holds in carmine."""
    W, H = 680, 300
    L, R, T, B = 54, 150, 22, 30
    years = sorted({y for y, *_ in table})
    y_lo, y_hi = min(years) - 0.6, max(years) + 0.6
    cells = [c for _, _, c, _ in table]
    c_lo, c_hi = math.log10(min(cells)) - 0.25, math.log10(max(cells)) + 0.35
    def X(y): return L + (W - L - R) * (y - y_lo) / (y_hi - y_lo)
    def Y(c): return T + (H - T - B) * (c_hi - math.log10(c)) / (c_hi - c_lo)
    body = [text(L, 12, "cells per published Perturb-seq screen, as released · log scale", 9.5, INK3)]
    for p in range(5, 9):
        v = 10 ** p
        if 10 ** c_lo <= v <= 10 ** c_hi:
            body.append(line(L, Y(v), W - R, Y(v)))
            body.append(text(L - 6, Y(v), f"{v:,.0f}" if p < 7 else f"{v // 1_000_000}M", 9, INK3, "end", dy="0.35em"))
    body.append(line(L, T, L, H - B, RULE_STRONG))
    for y in years:
        body.append(text(X(y), H - B + 14, str(y), 9.5, INK3, "middle"))
    # spread the labels within a year so they do not collide
    per_year = {}
    for row in table:
        per_year.setdefault(row[0], []).append(row)
    # a label sits right of its dot, except the small screens of a crowded year, which go
    # left where the empty years leave room; within each side the labels are nudged apart
    for y, rows in per_year.items():
        crowded = len(rows) >= 4
        sides = {"right": [], "left": []}
        for row in rows:
            sides["left" if (crowded and row[2] < 2_000_000) else "right"].append(row)
        for side, srows in sides.items():
            srows.sort(key=lambda r: -r[2])
            ys = []
            for _, _, c, _ in srows:
                yy = Y(c)
                if ys and yy < ys[-1] + 12:
                    yy = ys[-1] + 12
                ys.append(yy)
            left = side == "left"
            for (yr, label, c, src), ly in zip(srows, ys):
                held = label in HELD
                x = X(yr)
                body.append(dot(x, Y(c), 4 if held else 3.2, MARK if held else GREY))
                lx = x - 9 if left else x + 9
                if abs(ly - Y(c)) > 2:
                    body.append(line(x + (-5 if left else 5), Y(c), lx + (2 if left else -2), ly, RULE, 1))
                body.append(text(lx, ly, label, 8.5, INK if held else INK3, "end" if left else "start", dy="0.35em", title=src))
    return svg(W, H, body, "Published Perturb-seq screens by the number of cells released, against year; the ones Sidechain holds in carmine")


# --- the board post -----------------------------------------------------------------
def _day(s: str) -> int:
    y, m, d = (int(x) for x in s.split("-"))
    return (m - 9) * 31 + d if m >= 9 else d - 31


def fig_tabs(tabs=TABS, end=TABS_END):
    """The board's tabs: one row each, from the round that added it to the round that removed it."""
    W = 680
    RH = 22
    L, R, T, B = 70, 16, 22, 26
    H = T + RH * len(tabs) + B
    d0 = min(_day(b) for _, b, _ in tabs) - 1
    d1 = _day(end) + 1
    def X(d): return L + (W - L - R) * (d - d0) / (d1 - d0)
    body = [text(L, 12, "the board's tabs · when each was added and removed · 2026-09-02 to 2026-10-10", 9.5, INK3)]
    for lab, d in (("Sep 2", "2026-09-02"), ("Sep 15", "2026-09-15"), ("Oct 1", "2026-10-01"), ("Oct 10", "2026-10-10")):
        body.append(line(X(_day(d)), T, X(_day(d)), H - B))
        body.append(text(X(_day(d)), H - B + 12, lab, 8.5, INK3, "middle"))
    for i, (name, born, died) in enumerate(tabs):
        y = T + RH * i + RH / 2
        alive = died is None
        x0, x1 = X(_day(born)), X(_day(died or end))
        body.append(text(L - 8, y, name, 9.5, INK if alive else INK3, "end", dy="0.35em"))
        body.append(rect(x0, y - 5, x1 - x0, 10, MARK if alive else GREY, rx=2, opacity=None if alive else 0.55))
        body.append(text(x0, y - 8, born[5:], 7.5, INK3, "start"))
        if died:
            body.append(text(x1, y - 8, died[5:], 7.5, INK3, "end"))
    return svg(W, H, body, "The board's tabs from 2026-09-02 to 2026-10-10, each from the round that added it to the round that removed it")


# --- the two diagrams ---------------------------------------------------------------
def _box(x, y, w, h, title, sub=None, accent=False):
    out = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3" fill="var(--surface, #FFFFFF)" '
           f'stroke="{MARK if accent else INK2}" stroke-width="{1.4 if accent else 1}"/>']
    out.append(text(x + w / 2, y + (h / 2 if not sub else h / 2 - 5), title, 11, INK, "middle", weight=500, dy="0.35em"))
    if sub:
        out.append(text(x + w / 2, y + h / 2 + 9, sub, 8.5, INK3, "middle", dy="0.35em"))
    return out


def _arrow(x1, y1, x2, y2, label=None, above=True, stroke=INK2):
    out = [f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="1.1" marker-end="url(#arr)"/>']
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        out.append(text(mx, my + (-6 if above else 12), label, 8.5, INK3, "middle"))
    return out


ARROW_DEF = ('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK2}"/></marker></defs>')

RECIPE = [
    ("Borrow", "a log2 fold change per gene\nfrom every screen that silenced it", [("n", "shrinkage off"), ("d", "shrink scope, never sent"), ("g", "exponent γ")]),
    ("Pool", "one weighted average per gene\nsure sources count for more", [("f", "floor"), ("c", "tiers"), ("t", "τ², retired"), ("s", "adaptive rule")]),
    ("Adjust", "resize or nudge\nthe pooled change", [("a", "strength α"), ("k", "neighbours")]),
    ("Emit", "400 cells of counts per knockdown\non the context's own controls", [("p", "Poisson"), ("e", "dial λ"), ("b", "two strengths"), ("w", "pooled anchor"), ("r", "real cells"), ("h", "callable list")]),
]


def fig_recipe(recipe=RECIPE):
    """The SER recipe in four verbs, with the knob letters on the stage each one turns."""
    W, H = 680, 262
    n = len(recipe)
    BW, BH, GAP = 146, 54, 18
    L = (W - (n * BW + (n - 1) * GAP)) / 2
    y0 = 58
    body = [ARROW_DEF, text(L, 14, "the SER recipe · four verbs · the letters a model's name can carry, on the stage each one turns", 9.5, INK3)]
    body += _arrow(L + BW / 2, y0 - 10, L + BW / 2, y0 - 1)
    body.append(text(L + 2, y0 - 16, "H1 · K562 · HCT116 · HEK293T screens", 8.5, INK3, "start"))
    xe = L + 3 * (BW + GAP)
    body += _arrow(xe + BW / 2, y0 - 10, xe + BW / 2, y0 - 1)
    body.append(text(xe + BW / 2, y0 - 16, "the context's 18,400 control cells", 8.5, INK3, "middle"))
    for i, (verb, sub, knobs) in enumerate(recipe):
        x = L + i * (BW + GAP)
        body += _box(x, y0, BW, BH, verb, None, accent=(verb == "Emit"))
        for j, ln in enumerate(sub.split("\n")):
            body.append(text(x + BW / 2, y0 + BH + 14 + 11 * j, ln, 8, INK3, "middle"))
        ky = y0 + BH + 48
        for j, (letter, gloss) in enumerate(knobs):
            body.append(text(x + 10, ky + 13 * j, letter, 10, MARK, "start", weight=600))
            body.append(text(x + 24, ky + 13 * j, gloss, 8.5, INK2, "start"))
        if i < n - 1:
            body += _arrow(x + BW + 2, y0 + BH / 2, x + BW + GAP - 2, y0 + BH / 2)
    body += _arrow(xe + BW + 2, y0 + BH / 2, W - 4, y0 + BH / 2)
    body.append(text(W - 4, y0 + BH / 2 - 8, "scorer", 8, INK3, "end"))
    return svg(W, H, body, "The SER recipe: borrow, pool, adjust, emit; the knob letters placed on the stage each one turns")


LOOP = [
    ("a session", "ends its turn with a\nthree-line STATUS block"),
    ("the Stop hook", "status_hook.py reads the\nblock, refuses one that\nnames another session"),
    ("the stores", "asks · status · queue\nJSONL under agents/,\nledger.py the only writer"),
    ("the board", "one page on the Mac\nreads the stores, never\nkeeps a second record"),
    ("me", "answers on the board\nor in the session's chat"),
]


def fig_loop(loop=LOOP):
    """The reporting loop as it works today: a session's STATUS block, the hook, the stores, the board, Saber, back to the session."""
    W, H = 680, 236
    n = len(loop)
    BW, BH, GAP = 112, 48, 26
    L = (W - (n * BW + (n - 1) * GAP)) / 2
    y0 = 56
    body = [ARROW_DEF, text(L, 14, "how a session's state reaches me, and how my answer reaches it back", 9.5, INK3)]
    labels = ["STATUS · T<n> · state\n→ file · ASK: …", "lifts the block,\nfiles the ask as A<n>", "read on every request", "A12: b · seen · snooze"]
    for i, (name, sub) in enumerate(loop):
        x = L + i * (BW + GAP)
        body += _box(x, y0, BW, BH, name, None, accent=(name == "the board"))
        for j, ln in enumerate(sub.split("\n")):
            body.append(text(x + BW / 2, y0 + BH + 14 + 11 * j, ln, 8, INK3, "middle"))
        if i < n - 1:
            body += _arrow(x + BW + 2, y0 + BH / 2, x + BW + GAP - 2, y0 + BH / 2)
            for j, ln in enumerate(labels[i].split("\n")):
                body.append(text(x + BW + GAP / 2, y0 - 16 + 10 * j, ln, 7.5, INK3, "middle"))
    # the return path: Saber → ledger.py → the session
    xs, xe = L + (n - 1) * (BW + GAP) + BW / 2, L + BW / 2
    yb = y0 + BH + 70
    body.append(f'<path d="M {xs:.1f} {yb - 20:.1f} L {xs:.1f} {yb:.1f} L {xe:.1f} {yb:.1f} L {xe:.1f} {yb - 18:.1f}" fill="none" stroke="{MARK}" stroke-width="1.1" marker-end="url(#arr)"/>')
    body.append(text((xs + xe) / 2, yb + 13, "ledger.py ask answer · the owning session consumes the ask once it has acted", 8, INK2, "middle"))
    return svg(W, H, body, "The reporting loop: a session ends its turn with a STATUS block, the Stop hook lifts it into three stores, the board reads them, Saber answers, the answer returns through ledger.py")


# --- main ---------------------------------------------------------------------------
FIGS = {
    "knobs": {"fig-overall.svg": lambda: fig_overall(load_entries()),
              "fig-members.svg": lambda: fig_members(load_entries()),
              "fig-callset.svg": lambda: fig_callset(load_entries()),
              "fig-recipe.svg": lambda: fig_recipe()},
    "corpus": {"fig-cells-per-target.svg": lambda: fig_cells_per_target(),
               "fig-screens.svg": lambda: fig_screens()},
    "board": {"fig-tabs.svg": lambda: fig_tabs(),
              "fig-loop.svg": lambda: fig_loop()},
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="directory the SVGs are written into (a post's bundle or a draft's folder)")
    ap.add_argument("--post", choices=sorted(FIGS), required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS[a.post].items():
        (out / name).write_text(fn())
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
