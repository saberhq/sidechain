"""The challenge config is read per round: contexts, directory and control files follow `phase:`.

Born 2026-09-21 (T93): `phase: final` on the flat 2026 config died with a bare KeyError 'D' in
submit.build after the pooling had run, and unzipping the final bundle into the validation
directory would have overwritten the pert_counts.csv every cache was built from.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sidechain.data.loaders import (
    challenge_contexts,
    challenge_control_files,
    challenge_data_dir,
    challenge_phase,
    load_challenge_config,
)

ROOT = Path(__file__).resolve().parents[1]
FLAT = {"data_dir": "/d", "phase": "p1", "phases": {"p1": {"contexts": ["X", "Y"]}},
        "control_files": {"X": "ctx_x.h5ad", "Y": "ctx_y.h5ad"}}
PHASED = {"data_dir": "/d", "phase": "final",
          "phases": {"validation": {"contexts": ["A", "B", "C"], "dir": "/d"},
                     "final": {"contexts": ["D", "E", "F"], "dir": "/d/final"}},
          "control_files": {"validation": {"A": "context_A.h5ad", "B": "context_B.h5ad", "C": "context_C.h5ad"},
                            "final": {"D": "context_D.h5ad", "E": "context_E.h5ad", "F": "context_F.h5ad"}}}


def test_a_flat_map_still_works_for_a_single_round_config():
    assert challenge_contexts(FLAT) == ["X", "Y"]
    assert challenge_control_files(FLAT) == {"X": "ctx_x.h5ad", "Y": "ctx_y.h5ad"}
    assert challenge_data_dir(FLAT) == Path("/d")


def test_the_active_phase_selects_contexts_directory_and_files():
    assert challenge_phase(PHASED) == "final"
    assert challenge_contexts(PHASED) == ["D", "E", "F"]
    assert challenge_data_dir(PHASED) == Path("/d/final")
    assert challenge_control_files(PHASED) == {"D": "context_D.h5ad", "E": "context_E.h5ad", "F": "context_F.h5ad"}
    val = {**PHASED, "phase": "validation"}
    assert challenge_contexts(val) == ["A", "B", "C"] and challenge_data_dir(val) == Path("/d")
    assert list(challenge_control_files(val)) == ["A", "B", "C"]


def test_a_round_with_a_missing_control_file_is_refused_by_name():
    cfg = {**PHASED, "control_files": {"validation": PHASED["control_files"]["validation"]}}
    with pytest.raises(KeyError, match="D, E, F of phase 'final'"):
        challenge_control_files(cfg)


def test_a_2025_config_has_no_rounds():
    cfg = {"data_dir": "/d"}
    assert challenge_phase(cfg) == "" and challenge_contexts(cfg) == [] and challenge_data_dir(cfg) == Path("/d")


def test_the_2026_config_is_ready_for_both_rounds():
    cfg = load_challenge_config(ROOT / "challenges" / "vcc2026" / "config.yaml")
    assert challenge_phase(cfg) == "validation"
    assert challenge_contexts(cfg) == ["A", "B", "C"]
    assert challenge_data_dir(cfg) == Path("~/data/sidechain/vcc2026").expanduser()
    assert challenge_control_files(cfg) == {"A": "context_A.h5ad", "B": "context_B.h5ad", "C": "context_C.h5ad"}
    final = {**cfg, "phase": "final"}
    assert challenge_contexts(final) == ["D", "E", "F"]
    assert challenge_data_dir(final) == Path("~/data/sidechain/vcc2026/final").expanduser()
    assert challenge_control_files(final) == {"D": "context_D.h5ad", "E": "context_E.h5ad", "F": "context_F.h5ad"}
    assert challenge_data_dir(final) != challenge_data_dir(cfg)     # never unzipped over the validation bundle
