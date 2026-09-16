"""Contract tests for ``scripts/esm2_geometry_gate.py`` -- the ESM2 geometry gate.

The gate is a script, not a package module, and the batch wrappers under
``~/data/sidechain/runs/geometry_gate/`` drive it by loading it from its path and patching
module-level state (``EMB``, and here ``CACHE``). These tests do the same, so a refactor that
broke the wrappers breaks a test first.

Four things are pinned here, all of which cost real time when they slipped:

* **Every pre-existing output line is byte-identical with the bootstrap off.** The logged gate
  runs under ``~/data/sidechain/runs/geometry_gate/`` are parsed by tooling, so the 2026-09-15
  bootstrap had to be purely additive: the ``--bootstrap 0`` output must be a byte-for-byte
  prefix of the ``--bootstrap N`` output.
* **One source of truth for retired symbols.** Both scripts must hold the SAME dict object, not
  two copies that drift.
* **A retired corpus spelling still resolves.** A corpus that says ``QARS`` against a table keyed
  on ``QARS1`` must resolve the target, never drop it and never zero-fill it.
* **No alias can merge two genes.** No self-map, no chain, no two old symbols sharing a current
  one, and no pair with BOTH spellings on the 2026 gene axis.

The fixtures build a tiny synthetic pair of corpora and a tiny embedding table, so the whole file
runs in seconds with no cached pseudobulk and no 19,790-symbol ESM2 table on disk.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "esm2_geometry_gate.py"
BUILD = ROOT / "scripts" / "build_pert_features.py"

N_TARGETS = 40          # > 25, so the k=10 and k=25 rows (and their bootstraps) both exist
N_GENES = 30
DIM = 8
RETIRED, CURRENT = "QARS", "QARS1"      # corpus spelling -> the spelling the table is keyed on


def load_script(path: Path, name: str):
    """Load a script by path, the way the batch wrappers do."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def labels() -> list[str]:
    """One control row, then the targets -- one of them under its RETIRED symbol."""
    names = ["non-targeting"] + [f"GENE{i:03d}" for i in range(N_TARGETS)]
    names[1] = RETIRED
    return names


def write_corpus(path: Path, seed: int, names: list[str], genes: list[str]) -> None:
    rng = np.random.default_rng(seed)
    cpm_sum = rng.lognormal(mean=3.0, sigma=0.5, size=(len(names), len(genes))) * 100.0
    np.savez(path, labels=np.array(names), genes=np.array(genes),
             cpm_sum=cpm_sum, n_cells=np.full(len(names), 100.0))


@pytest.fixture
def corpora(tmp_path, monkeypatch):
    """Two tiny corpora in a fake CACHE, plus a tiny symbol-keyed embedding table."""
    import torch

    names, genes = labels(), [f"ENSGene{i:03d}" for i in range(N_GENES)]
    write_corpus(tmp_path / "a.npz", 1, names, genes)
    write_corpus(tmp_path / "b.npz", 2, names, genes)

    rng = np.random.default_rng(3)
    table = {(CURRENT if n == RETIRED else n): torch.tensor(rng.normal(size=DIM),
                                                            dtype=torch.float32)
             for n in names[1:]}
    emb = tmp_path / "emb.pt"
    torch.save(table, emb)

    g = load_script(GATE, "gate_under_test")
    monkeypatch.setattr(g, "CACHE", tmp_path)
    return g, emb


def run_gate(g, argv: list[str], capsys, monkeypatch) -> str:
    monkeypatch.setattr(sys, "argv", ["gate", *argv])
    assert g.main() == 0
    return capsys.readouterr().out


def within_argv(emb: Path, boot: int) -> list[str]:
    return ["within", "a.npz", "--embeddings", str(emb), "--bootstrap", str(boot)]


def cross_argv(emb: Path, boot: int) -> list[str]:
    return ["cross", "a.npz", "b.npz", "--embeddings", str(emb), "--bootstrap", str(boot)]


# --------------------------------------------------------------------------- one source of truth

def test_both_scripts_share_one_alias_object():
    gate = load_script(GATE, "gate_alias_identity")
    build = load_script(BUILD, "build_alias_identity")
    from sidechain.data.gene_aliases import RETIRED_SYMBOLS

    assert gate.ALIAS is RETIRED_SYMBOLS
    assert build.ALIAS is RETIRED_SYMBOLS
    assert build.RETIRED_ALIASES is gate.ALIAS          # the legacy name, same object
    assert RETIRED_SYMBOLS[RETIRED] == CURRENT
    assert list(RETIRED_SYMBOLS) == sorted(RETIRED_SYMBOLS)     # kept alphabetical


AXIS = Path("~/data/sidechain/vcc2026/gene_names.csv").expanduser()


def test_no_alias_can_merge_two_genes():
    """The four invariants that keep the table a bridge and not a collision.

    A retired symbol resolves through this table to ONE current symbol, which is then looked up in
    the embedding table. Break any of these and two distinct genes quietly become one row:

    * a self-map is a no-op that hides a typo;
    * a chain (a value that is also a key) means the answer depends on how many times you apply
      the table;
    * two old symbols sharing one current symbol merges two genes into one embedding;
    * an old symbol that is itself a current symbol elsewhere would rewrite a live gene.
    """
    from sidechain.data.gene_aliases import RETIRED_SYMBOLS as alias

    assert not [k for k, v in alias.items() if k == v]
    assert not (set(alias) & set(alias.values()))
    assert len(set(alias.values())) == len(alias)
    assert all(k.strip() == k and v.strip() == v for k, v in alias.items())


@pytest.mark.skipif(not AXIS.exists(), reason="the 2026 gene axis is not on this machine")
def test_no_pair_has_both_spellings_on_the_2026_axis():
    """If both spellings were on the axis they would be two genes, and aliasing would merge them.

    The 2026 axis speaks the OLD dialect (38 of the 48 old symbols are on it, none of the current
    ones), so the table maps a corpus label onto the embedding table -- never onto the axis.
    """
    from sidechain.data.gene_aliases import RETIRED_SYMBOLS as alias

    axis = {line.strip() for line in AXIS.read_text().splitlines()[1:] if line.strip()}
    assert len(axis) > 18_000                       # the real file, not a stub
    assert not [(k, v) for k, v in alias.items() if k in axis and v in axis]


def test_retired_symbol_resolves_against_a_current_keyed_table(corpora, capsys, monkeypatch):
    g, emb = corpora
    out = run_gate(g, within_argv(emb, 0), capsys, monkeypatch)
    # every target resolves, INCLUDING the one the corpus spells with the retired symbol
    assert f"{N_TARGETS}/{N_TARGETS} targets resolved" in out

    names = np.array(labels()[1:])
    ok, e = g.embeddings(names, emb)
    assert ok.all() and len(e) == N_TARGETS

    # drop the alias and that one target is dropped -- never zero-filled
    monkeypatch.setattr(g, "ALIAS", {})
    ok_no_alias, e_no_alias = g.embeddings(names, emb)
    assert ok_no_alias.sum() == N_TARGETS - 1
    assert len(e_no_alias) == N_TARGETS - 1


# ------------------------------------------------------------------------------ the new bootstrap

def test_within_prints_the_bootstrap_lines(corpora, capsys, monkeypatch):
    g, emb = corpora
    out = run_gate(g, within_argv(emb, 50), capsys, monkeypatch)
    assert "bootstrap over targets: 50 paired resamples, seed 20260903" in out
    boot = [l for l in out.splitlines() if l.startswith("bootstrap (50 paired resamples")]
    assert len(boot) == 2
    assert "k=10 margin" in boot[0] and "k=25 margin" in boot[1]
    for line in boot:
        assert "[" in line and "]" in line
        assert "distinguishable from zero" in line


def test_cross_prints_the_bootstrap_lines(corpora, capsys, monkeypatch):
    g, emb = corpora
    out = run_gate(g, cross_argv(emb, 50), capsys, monkeypatch)
    boot = [l for l in out.splitlines() if l.startswith("bootstrap (50 paired resamples")]
    assert len(boot) == 3
    assert "k=25 margin" in boot[0]
    assert "k=10 margin (embedding arm - scrambled arm)" in boot[2]

    # the fusion line is the selection-honest one: the gain at the best w is >= 0 by construction
    # (w = 0 is in the sweep), so what is bootstrapped is embedding maximum - scrambled maximum
    assert "fusion gain over the scrambled fusion (best w re-selected per resample)" in boot[1]
    assert "the raw gain at the best w is >= 0 by construction" in out
    assert "fusion gain at w=" not in out


def test_the_fusion_interval_is_centred_on_zero_for_a_meaningless_embedding(corpora, capsys,
                                                                            monkeypatch):
    """The selection bias the interval exists to remove: a random table must not read positive.

    The embedding here carries nothing about the responses (both are random draws), so the real
    fusion arm and the scrambled one are the same kind of object. An interval on the raw gain at
    the best w would still be strictly positive -- w = 0 is in the sweep, so the maximum cannot be
    negative -- while the difference of maxima may straddle zero.
    """
    g, emb = corpora
    out = run_gate(g, cross_argv(emb, 200), capsys, monkeypatch)
    line = next(l for l in out.splitlines() if "fusion gain over the scrambled fusion" in l)
    lo, hi = (float(x) for x in line.split("[")[1].split("]")[0].split(","))
    assert lo <= hi
    assert lo < 0.0 < hi                        # no geometry to find, and none manufactured
    assert "indistinguishable from zero" in line


@pytest.mark.parametrize("argv", ["within", "cross"])
def test_the_same_seed_gives_the_same_output_twice(corpora, capsys, monkeypatch, argv):
    g, emb = corpora
    build = within_argv if argv == "within" else cross_argv
    first = run_gate(g, build(emb, 50), capsys, monkeypatch)
    second = run_gate(g, build(emb, 50), capsys, monkeypatch)
    assert first == second
    assert "bootstrap (50 paired resamples" in first


@pytest.mark.parametrize("argv", ["within", "cross"])
def test_bootstrap_zero_is_the_pre_change_output(corpora, capsys, monkeypatch, argv):
    """--bootstrap 0 prints nothing new, and the bootstrap is APPENDED, never woven in."""
    g, emb = corpora
    build = within_argv if argv == "within" else cross_argv
    off = run_gate(g, build(emb, 0), capsys, monkeypatch)
    on = run_gate(g, build(emb, 50), capsys, monkeypatch)

    assert "bootstrap" not in off
    assert on.startswith(off)                    # every legacy line byte-identical, in place


def test_embeddings_option_defaults_to_the_module_level_table(corpora, capsys, monkeypatch):
    """The batch wrappers patch g.EMB and pass no --embeddings; that path must still work."""
    g, emb = corpora
    monkeypatch.setattr(g, "EMB", emb)
    out = run_gate(g, ["within", "a.npz", "--bootstrap", "0"], capsys, monkeypatch)
    assert f"{N_TARGETS}/{N_TARGETS} targets resolved" in out


def test_a_missing_embedding_table_is_refused(corpora, capsys, monkeypatch, tmp_path):
    g, _ = corpora
    monkeypatch.setattr(sys, "argv",
                        ["gate", "within", "a.npz", "--embeddings", str(tmp_path / "nope.pt")])
    with pytest.raises(SystemExit):
        g.main()


def test_boot_ci_is_a_paired_interval_around_the_mean():
    g = load_script(GATE, "gate_boot_ci")
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.05, scale=0.01, size=500)
    lo, hi = g.boot_ci(values, 500, g.boot_rng(20260903))
    assert lo < values.mean() < hi
    assert 0.0 < lo                              # a real effect reads as distinguishable

    zero = np.zeros(200)
    lo0, hi0 = g.boot_ci(zero, 200, g.boot_rng(20260903))
    assert lo0 == hi0 == 0.0                     # no spread to find, and no crash
