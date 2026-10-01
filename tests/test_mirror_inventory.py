"""Contract tests for the generated local-mirror inventory.

The report is generated, so the thing worth pinning is the arithmetic and the
refusal to invent numbers -- not the prose.
"""
from __future__ import annotations

import json

import pytest

from scripts.mirror_inventory import anchors, held_out, knob_str, raw_mean, render, shape


def test_held_out_prefers_the_longest_matching_stem():
    # 'loco_k562gwps_union' must not be resolved by the shorter 'loco_k562gwps'
    assert held_out("loco_k562gwps_union_ch272") == "K562 genome-wide"
    assert held_out("loco_k562_essential") == "K562 essential"
    assert held_out("something_unknown") == "—"


def test_shape_reports_every_matching_suffix():
    assert shape("loco_hek293t_ch272") == "challenge 272"
    assert shape("loco_hct116") == "—"


def test_missing_files_yield_none_never_zero(tmp_path):
    """A gap of 0.0 and a gap that was never recorded must not look alike."""
    assert raw_mean(tmp_path) is None
    assert anchors(tmp_path) == (None, None)


def test_raw_mean_reads_the_mean_row_not_the_first(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "agg_results.csv").write_text(
        "statistic,pds_cosine\ncount,272.0\nmean,0.75\nstd,0.1\n")
    assert raw_mean(tmp_path) == 0.75


def test_knob_str_hides_defaults_and_shows_set_knobs():
    assert knob_str({"alpha": 1.0, "gamma": 1.0, "var_floor": "none"}) == "—"
    out = knob_str({"alpha": 1.35, "var_floor": "poisson"})
    assert "alpha=1.35" in out and "var=poisson" in out


def test_render_prints_the_gap_and_refuses_to_fabricate_one():
    folds = [{"name": "f", "line": "L", "shape": "—", "backend": "pdex", "device": "cpu",
              "cell_eval2": "0.16.0", "targets": 10, "genes": 100, "cells": 1000,
              "baseline": 0.5, "replicate": 0.9, "gap": 0.4, "arms": [], "ref": None},
             {"name": "g", "line": "M", "shape": "—", "backend": "pdex", "device": "cpu",
              "cell_eval2": "0.16.0", "targets": None, "genes": None, "cells": None,
              "baseline": None, "replicate": None, "gap": None, "arms": [], "ref": None}]
    out = render(folds, full=False)
    assert "0.4000" in out
    assert "| `g` | M | — | — | — | — | pdex · cpu | — | 0 |" in out
    assert "GENERATED — never hand-edit" in out


def test_transfer_floor_of_all_zeros_is_the_knob_switched_off():
    off = {"transfer_floor": {"a": 0.0, "b": 0.0}, "dispersion": "even"}
    assert "transfer" not in knob_str(off)
    on = {"transfer_floor": {"a": 0.02033, "b": 0.0}, "dispersion": "even"}
    assert "transfer=0.02033/0" in knob_str(on)


def test_coverage_tiers_render_compactly():
    assert "coverage=3:0.1,10:0.5" in knob_str({"coverage_tiers": [[3.0, 0.1], [10.0, 0.5]]})


def test_a_zero_valued_knob_is_not_mistaken_for_an_absent_one():
    """`0.0 == False` in Python, so a membership test against False drops gamma = 0.

    `loco_k562gwps_pdex/ag_a100_g000` is a real arm at gamma = 0 -- the end of the transfer dial
    where the absolute CPM change transfers rather than the fold change. Dropped from the knob
    string it reads as gamma unset, which means gamma = 1: the other end. Found by carrying
    sidechain-11's own `or 1.0` bug (T59) into this file.
    """
    shown = knob_str({"alpha": 1.0, "gamma": 0.0})
    assert shown == "gamma=0.0", (
        "gamma = 0 must render; '—' reads as gamma unset, i.e. gamma = 1, the other end")
    # and the genuinely-off settings still vanish, "—" being this function's empty
    for off in ({"gamma": 1.0, "alpha": 1.0}, {"shrinkage": False}, {"var_floor": "none"},
                {"similarity_beta": 0.0}, {"emit_lambda": 0.0}, {"coverage_tiers": None}):
        assert knob_str(off) == "—", off


def test_the_anchor_amplitude_floor_neighbour_and_cell_count_are_shown():
    """T84 round 2: an arm's pseudobulk anchor and cell count decide whether it may be compared
    with another at all, so both must be visible; the default anchor stays hidden, since every
    arm scored before the flag existed carries no key for it."""
    base = {"alpha": 1.35, "alpha_bulk": 1.5, "emit_lambda": 0.5, "min_libsize": 1000.0,
            "cells": 50128, "perturbations": 272}
    out = knob_str(base)
    assert "ab=1.5" in out and "alpha=1.35" in out and "floor=1000" in out
    assert "anchor" not in out and "cells=" not in out            # mean_cpm default, real-side cells
    assert "anchor" not in knob_str({**base, "bulk_anchor": "mean_cpm"})
    pooled = knob_str({**base, "bulk_anchor": "pooled", "cells": 400 * 272})
    assert "anchor=pooled" in pooled and "cells=400/target" in pooled
    nb = {"table": ["/x/tahoex1_3b_gene_encoder_table.pt", "/y/string_node2vec_table.pt"],
          "w": [0.05, 0.3], "k": 25}
    assert "nb=tahoex1·0.05+string·0.3·k25" in knob_str({**base, "neighbour": nb})
    assert "nb=" not in knob_str({**base, "neighbour": None})
    # the blend shape, or a size-aware arm renders as the unit arm it is read against (T103)
    one = {"table": "/y/string_node2vec_table.pt", "w": 0.2, "k": 25, "scale": 31.4159}
    assert "nb=string·0.2·k25" in knob_str({**base, "neighbour": {**one, "size": "unit"}})
    assert "·med" not in knob_str({**base, "neighbour": {**one, "size": "unit"}})
    assert "nb=string·0.2·k25·med" in knob_str({**base, "neighbour": {**one, "size": "median"}})
    assert "·med" not in knob_str({**base, "neighbour": one})       # no key = scored pre-flag
    assert ("nb=tahoex1·0.05+string·0.3·k25·med"
            in knob_str({**base, "neighbour": {**nb, "size": "median"}}))
