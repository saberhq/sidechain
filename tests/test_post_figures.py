"""Contract tests for scripts/post_figures.py: every figure parses as SVG, draws the numbers
the generated JSON holds and no others, and keeps calibration runs off the standings-shaped
figures (Saber, 2026-10-07)."""
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("post_figures", ROOT / "scripts" / "post_figures.py")
pf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pf)

ENTRIES = json.loads((ROOT / "site" / "data" / "submissions.json").read_text())["entries"]
COVERAGE = json.loads((ROOT / "site" / "data" / "coverage.json").read_text())


def texts(svg: str):
    root = ET.fromstring(svg)
    # a <text> may open with a <title> child (the hover); the label is the text that follows it
    return [((t.text or "") + "".join(c.tail or "" for c in t)).strip() for t in root.iter("{http://www.w3.org/2000/svg}text")]


@pytest.mark.parametrize("post", sorted(pf.FIGS))
def test_every_figure_parses(post):
    for name, fn in pf.FIGS[post].items():
        root = ET.fromstring(fn())
        assert root.tag.endswith("svg"), name
        assert root.get("role") == "img" and root.get("aria-label"), name


def test_overall_draws_only_normal_entries_and_their_scores():
    svg = pf.fig_overall(ENTRIES)
    labels = texts(svg)
    normal = [e for e in ENTRIES if not e.get("class")]
    assert sum(1 for e in normal if e["name"] in labels) == len(normal)
    for e in ENTRIES:
        if e.get("class"):
            assert e["name"] not in labels, f'{e["name"]} is a calibration run and must stay off the standings-shaped figure'
    overall = {f"{e['overall']:.3f}" for e in normal}
    for t in labels:
        if t.startswith(("0.", "-0.")) and len(t) == 5:
            assert t in overall, t


def test_members_use_scaled_values_from_the_file():
    svg = pf.fig_members(ENTRIES)
    labels = texts(svg)
    last = [e for e in ENTRIES if not e.get("class")][-1]
    for m in pf.MEMBERS:
        assert f"{last['members'][m]['scaled']:.3f}" in labels, m
    for e in ENTRIES:
        if e.get("class"):
            assert pf.short(e["name"]) not in labels


def test_callset_uses_raw_values_of_the_four_named_entries():
    svg = pf.fig_callset(ENTRIES)
    labels = texts(svg)
    by = {e["name"]: e for e in ENTRIES}
    for n in ("SER-14aefksw", "SER-15aefkrsw", "SER-16aefhkrsw", "SER-17aefhkrsw"):
        for m in ("fid", "reach", "jac"):
            assert f"{by[n]['members'][m]['raw']:.3f}" in labels, (n, m)


def test_cells_per_target_prints_every_range_in_the_json():
    svg = pf.fig_cells_per_target()
    labels = texts(svg)
    for s in COVERAGE["sources"]:
        q = s.get("cells_per_covered_target")
        if q:
            assert f'{q["min"]:,} – {q["median"]:,} – {q["max"]:,}' in labels, s["line"]
            assert s["line"] in labels


def test_screens_table_rows_each_carry_a_source_and_appear():
    for year, label, cells, source in pf.SCREENS:
        assert 2000 < year < 2100 and cells > 0 and source
    labels = texts(pf.fig_screens())
    for _, label, _, _ in pf.SCREENS:
        assert label in labels


def test_tabs_timeline_names_every_tab():
    labels = texts(pf.fig_tabs())
    for name, born, died in pf.TABS:
        assert name in labels
        assert born[5:] in labels


def test_diagrams_carry_every_letter_and_every_node():
    labels = texts(pf.fig_recipe())
    for letter in "ndgfctsakpebwrh":
        assert letter in labels, letter
    for verb, _, _ in pf.RECIPE:
        assert verb in labels
    labels = texts(pf.fig_loop())
    for name, _ in pf.LOOP:
        assert name in labels
