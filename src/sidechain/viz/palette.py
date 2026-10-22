"""Sidechain's dataset colours: one colour per challenge context and per corpus study, the same in every figure.

    from sidechain.viz.palette import color, marker, PALETTE
    color("A")                    # '#2a78d6' (light theme); color("A", theme="dark")
    color("replogle22_rpe1")      # the study's colour: Replogle 2022 is teal in every chart
    .venv/bin/python -m sidechain.viz.palette --json    # the whole plan as JSON (what the board's Figures tab reads)

Two tiers. The six challenge contexts own six saturated hues (A blue, B orange, C aqua; D violet, E magenta,
F yellow): validation contexts draw as circles, test contexts as diamonds. A corpus line takes its STUDY's colour,
a darker, muted hue, and a marker shape of its own, and is always labelled. Sequential, diverging and within-line
category scales sit beside them. Every categorical set was run through the dataviz skill's validator
(`validate_palette.js`) on the board's own surfaces; the results are in `VALIDATION`. Plan and rules:
private/tasks/T129.md.
"""
from __future__ import annotations

import argparse
import json

# The board's chart surfaces (dashboard.py `--surface`), which the validator measured against.
SURFACES = {"light": "#FDFBF7", "dark": "#272421"}

# ── tier 1: the challenge's six contexts ───────────────────────────────────────────────────────────────────────
# A/B/C are the dataviz reference slots 1-3 (they pass all-pairs, so any scatter of the three is safe); D/E/F are
# slots 7, 5 and 4 (all-pairs among themselves too). All six in one scatter fail all-pairs only on A blue vs
# D violet, so a six-context scatter draws the test contexts as diamonds, or facets validation from test
# (final-phase: none).
CONTEXTS = {
    "A": {"light": "#2a78d6", "dark": "#3987e5", "marker": "circle", "round": "validation"},
    "B": {"light": "#eb6834", "dark": "#d95926", "marker": "circle", "round": "validation"},
    "C": {"light": "#1baf7a", "dark": "#199e70", "marker": "circle", "round": "validation"},
    "D": {"light": "#6250d6", "dark": "#9085e9", "marker": "diamond", "round": "test"},
    "E": {"light": "#e87ba4", "dark": "#d55181", "marker": "diamond", "round": "test"},
    "F": {"light": "#eda100", "dark": "#c98500", "marker": "diamond", "round": "test"},
}

# ── tier 2: the corpus, one colour per study ───────────────────────────────────────────────────────────────────
# A study shares its lab, chemistry and depth, which is most of what a QC plot shows, so a study is the unit of
# colour; the lines inside a study are told apart by marker and label. The five hues pass the adjacent checks in this
# order in both themes; they do NOT pass all-pairs (no five muted hues do), so a corpus line is never drawn without
# its label. Arc's 2025 H1 is an archive reference and stays neutral; the named-line panel is background.
STUDIES = {
    "replogle22": {"label": "Replogle 2022", "light": "#00826b", "dark": "#269e8b"},
    "nadig24": {"label": "Nadig 2024", "light": "#b44d34", "dark": "#d3694f"},
    "xatlas25": {"label": "X-Atlas Orion", "light": "#554e9b", "dark": "#7974c5"},
    "pan26": {"label": "Pan 2026", "light": "#b0a03c", "dark": "#a4932e"},
    "feng26": {"label": "Feng 2026", "light": "#9b548d", "dark": "#b86fa9"},
    "vcc25": {"label": "Arc VCC 2025", "light": "#8a837a", "dark": "#918c84"},
    "scbasecount": {"label": "scBaseCount named lines", "light": "#bdb6ac", "dark": "#5c5853"},
}

# corpus lines: id -> study, the line's name, its marker (Plotly symbol names, all drawable in WebGL)
LINES = {
    "replogle22_k562_ess": {"study": "replogle22", "line": "K562", "screen": "essential", "marker": "square"},
    "replogle22_k562_gwps": {"study": "replogle22", "line": "K562", "screen": "genome-wide", "marker": "square-open"},
    "replogle22_rpe1": {"study": "replogle22", "line": "RPE1", "screen": "essential", "marker": "triangle-up"},
    "nadig24_jurkat": {"study": "nadig24", "line": "Jurkat", "screen": "essential", "marker": "triangle-down"},
    "nadig24_hepg2": {"study": "nadig24", "line": "HepG2", "screen": "essential", "marker": "pentagon"},
    "xatlas25_hct116": {"study": "xatlas25", "line": "HCT116", "screen": "genome-wide", "marker": "cross"},
    "xatlas25_hek293t": {"study": "xatlas25", "line": "HEK293T", "screen": "genome-wide", "marker": "x"},
    "pan26_h1": {"study": "pan26", "line": "H1", "screen": "genome-scale", "marker": "star"},
    "feng26_ipsc": {"study": "feng26", "line": "iPSC lines", "screen": "targeted", "marker": "hexagon"},
    "vcc25_h1": {"study": "vcc25", "line": "H1", "screen": "VCC 2025", "marker": "star-open"},
}

# ── scales that are not identity ───────────────────────────────────────────────────────────────────────────────
# magnitude: the dataviz blue ramp from step 200 (step 100 vanishes into the surface under a point cloud); a dark
# theme reverses the anchor so "more" is still "more ink against the surface".
SEQUENTIAL = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# polarity: cosine similarity blue <-> red, a fold change violet <-> orange (two pairs, so the two heatmaps of a
# complex never read as one quantity); the midpoint is the neutral gray of each theme.
DIVERGING = {
    "cosine": {"neg": "#2a78d6", "pos": "#d03b3b", "mid": {"light": "#f0efec", "dark": "#383835"}},
    "lfc": {"neg": "#6250d6", "pos": "#e3721c", "mid": {"light": "#f0efec", "dark": "#383835"}},
}
# cell-cycle phase is ordinal (G1 -> S -> G2/M), so a one-hue ink ramp, never three identity hues: a phase colour
# must not be mistaken for a context.
PHASE = {"G1": "#cfc8bc", "S": "#8a8378", "G2M": "#2b2723"}
PHASE_DARK = {"G1": "#4a4641", "S": "#8f897f", "G2M": "#e9e4da"}
# clusters inside one line: local to that plot (the legend says so), muted so they never impersonate a context
CLUSTERS = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#9c755f", "#76b7b2", "#edc948", "#ff9da7", "#bab0ac",
            "#e15759", "#8cd17d", "#d4a6c8"]

# What the validator said (dataviz skill, scripts/validate_palette.js, 2026-10-21, session `5bb2ca10`).
VALIDATION = {
    "A-C all-pairs": "pass both themes; worst CVD dE 9.2 light / 9.4 dark; aqua is 2.7:1 on light, so labels",
    "D-F all-pairs": "pass both themes; worst CVD dE 16.3 light / 13.2 dark; magenta and yellow under 3:1 on light",
    "A-F adjacent": "pass both themes; worst CVD dE 9.2 light / 9.4 dark",
    "A-F all-pairs": "FAIL on A vs D (normal-vision dE 10.5): test contexts draw as diamonds, or facet by round",
    "studies adjacent": "pass both themes in the order listed; worst CVD dE 9.0 light / 9.9 dark",
    "studies all-pairs": "FAIL (no five muted hues pass): a corpus line always carries its label and marker",
}


def color(key: str, theme: str = "light") -> str:
    """The colour of a challenge context (`A`..`F`), a corpus line id, or a study id."""
    if key in CONTEXTS:
        return CONTEXTS[key][theme]
    if key in LINES:
        return STUDIES[LINES[key]["study"]][theme]
    if key in STUDIES:
        return STUDIES[key][theme]
    raise KeyError(f"no colour for {key!r}: not a context, a corpus line or a study")


def marker(key: str) -> str:
    """The marker of a context or a corpus line (Plotly symbol names; matplotlib maps them in `mpl_marker`)."""
    if key in CONTEXTS:
        return CONTEXTS[key]["marker"]
    if key in LINES:
        return LINES[key]["marker"]
    raise KeyError(f"no marker for {key!r}")


_MPL = {"circle": "o", "diamond": "D", "square": "s", "square-open": "s", "triangle-up": "^", "triangle-down": "v",
        "pentagon": "p", "cross": "P", "x": "X", "star": "*", "star-open": "*", "hexagon": "h"}


def mpl_marker(key: str) -> str:
    return _MPL[marker(key)]


def label(key: str) -> str:
    """A reader's name: `context A`, `K562 · Replogle 2022 essential`."""
    if key in CONTEXTS:
        return f"context {key}"
    if key in LINES:
        ln = LINES[key]
        return f"{ln['line']} · {STUDIES[ln['study']]['label']}" + (f" {ln['screen']}" if ln["study"] == "replogle22" else "")
    if key in STUDIES:
        return STUDIES[key]["label"]
    raise KeyError(key)


PALETTE = {"surfaces": SURFACES, "contexts": CONTEXTS, "studies": STUDIES, "lines": LINES,
           "sequential": SEQUENTIAL, "diverging": DIVERGING, "phase": {"light": PHASE, "dark": PHASE_DARK},
           "clusters": CLUSTERS, "validation": VALIDATION}


def as_json() -> dict:
    """The plan with every line's colour and label resolved, for a browser."""
    out = json.loads(json.dumps(PALETTE))
    for k in LINES:
        out["lines"][k]["light"], out["lines"][k]["dark"] = color(k), color(k, "dark")
        out["lines"][k]["label"] = label(k)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="print the plan as JSON")
    a = ap.parse_args()
    print(json.dumps(as_json(), indent=1) if a.json else "\n".join(f"{k:22s} {color(k)}  {label(k)}"
                                                                    for k in [*CONTEXTS, *LINES]))
