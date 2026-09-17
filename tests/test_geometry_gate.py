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

And since T89 (2026-09-16), two more:

* **More scramble draws only append.** ``--scrambles 1`` is the legacy output; ``M > 1`` adds the
  spread across permutations and a bootstrap that resamples the permutation, after everything
  else, so the 2026-09-15 logs stay a byte-for-byte prefix. Draw j is the same permutation
  whatever M is, so a sweep can be widened without re-running its early draws.
* **`compare` is paired and refuses an unpaired footing.** A row against itself reads exactly
  zero wherever the scramble is fixed; two rows with different targets are refused, never
  intersected; a table built from the responses beats a random one with the lead named.

The fixtures build a tiny synthetic pair of corpora and a tiny embedding table, so the whole file
runs in seconds with no cached pseudobulk and no 19,790-symbol ESM2 table on disk.
"""

from __future__ import annotations

import importlib.util
import re
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
    """--bootstrap 0 prints nothing new, and the bootstrap is APPENDED, never woven in.

    (--no-assay on both sides: the extraction assay is the last block of any run and has its own
    append test; here the property under test is the bootstrap's.)
    """
    g, emb = corpora
    build = within_argv if argv == "within" else cross_argv
    off = run_gate(g, build(emb, 0) + ["--no-assay"], capsys, monkeypatch)
    on = run_gate(g, build(emb, 50) + ["--no-assay"], capsys, monkeypatch)

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


# --------------------------------------------------- several scramble draws (T89, 2026-09-16)

def within_argv_scr(emb: Path, boot: int, scr: int, dump: Path | None = None) -> list[str]:
    argv = within_argv(emb, boot) + ["--scrambles", str(scr)]
    return argv + (["--dump", str(dump)] if dump else [])


def cross_argv_scr(emb: Path, boot: int, scr: int, dump: Path | None = None) -> list[str]:
    argv = cross_argv(emb, boot) + ["--scrambles", str(scr)]
    return argv + (["--dump", str(dump)] if dump else [])


SCR_PREFIX = "bootstrap over targets AND scramble draws"


GOLDEN = ROOT / "tests" / "data"


@pytest.mark.parametrize("mode", ["within", "cross"])
@pytest.mark.parametrize("boot", [0, 50])
def test_one_scramble_is_the_pre_change_output_and_more_draws_only_append(corpora, capsys,
                                                                             monkeypatch, mode,
                                                                             boot):
    """--scrambles 1 is the default and prints nothing new; M > 1 appends and never rewrites.

    (--no-assay on every side: the assay is the last block and is pinned separately.)
    """
    g, emb = corpora
    build = within_argv_scr if mode == "within" else cross_argv_scr
    one = run_gate(g, build(emb, boot, 1) + ["--no-assay"], capsys, monkeypatch)
    legacy = run_gate(g, (within_argv if mode == "within" else cross_argv)(emb, boot)
                      + ["--no-assay"], capsys, monkeypatch)
    assert one == legacy
    assert "scramble draws" not in one

    five = run_gate(g, build(emb, boot, 5) + ["--no-assay"], capsys, monkeypatch)
    assert five.startswith(one)                  # every fixed-scramble line, byte for byte
    assert "scramble draws: 5 permutations" in five
    if boot == 0:
        assert "bootstrap" not in five           # no bootstrap means none, in the new block too


@pytest.mark.parametrize("mode", ["within", "cross"])
@pytest.mark.parametrize("boot", [0, 50])
def test_legacy_output_matches_the_golden_file(corpora, capsys, monkeypatch, mode, boot):
    """The contract, pinned against a FIXED text and not against the module's own output.

    ``tests/data/geometry_gate_golden_<mode>_boot<N>.txt`` is what the gate committed before T89
    (2026-09-15) printed on this fixture at seed 20260903. Every later version must print it
    byte for byte as a PREFIX, whatever it appends after. Regenerate only if a legacy line is
    changed on purpose, and say so in the commit.
    """
    g, emb = corpora
    golden = (GOLDEN / f"geometry_gate_golden_{mode}_boot{boot}.txt").read_text()
    build = within_argv_scr if mode == "within" else cross_argv_scr
    out = run_gate(g, build(emb, boot, 4), capsys, monkeypatch)
    assert out.startswith(golden)
    assert len(golden.splitlines()) >= 10


def test_within_scramble_block_prints_spread_and_resampled_intervals(corpora, capsys,
                                                                     monkeypatch):
    g, emb = corpora
    out = run_gate(g, within_argv_scr(emb, 50, 6), capsys, monkeypatch)
    spread = [l for l in out.splitlines() if "scrambled arm" in l and "over 6 draws" in l]
    assert len(spread) == 2 and "k=10" in spread[0] and "k=25" in spread[1]
    for line in spread:
        assert "sd " in line and "min " in line and "max " in line
    boot = [l for l in out.splitlines() if l.startswith(SCR_PREFIX)]
    assert len(boot) == 2
    assert "(50 resamples; 6 permutations redrawn with replacement and averaged)" in boot[0]
    assert "k=10 margin" in boot[0] and "k=25 margin" in boot[1]
    assert all("distinguishable from zero" in l for l in boot)


def test_cross_scramble_block_prints_spread_and_resampled_intervals(corpora, capsys, monkeypatch):
    g, emb = corpora
    out = run_gate(g, cross_argv_scr(emb, 50, 6), capsys, monkeypatch)
    spread = [l for l in out.splitlines() if "over 6 draws" in l]
    assert len(spread) == 3                      # k=25 arm, k=10 arm, scrambled fusion at best w
    assert "k=25 scrambled arm" in spread[0]
    assert "k=10 scrambled arm" in spread[1]
    assert "scrambled fusion at w=" in spread[2]
    boot = [l for l in out.splitlines() if l.startswith(SCR_PREFIX)]
    assert len(boot) == 3
    assert "k=25 margin" in boot[0]
    assert "fusion gain over the scrambled fusion" in boot[1]
    assert "k=10 margin (embedding arm - scrambled arm)" in boot[2]


def test_extra_scramble_draws_are_distinct_seeded_and_stable_under_m():
    """Draw j is the same permutation whether M is 3 or 8, so a sweep can be widened later."""
    g = load_script(GATE, "gate_perms")
    first = np.random.default_rng(20260903).permutation(N_TARGETS)
    three = g.scramble_perms(N_TARGETS, 20260903, 3, first)
    eight = g.scramble_perms(N_TARGETS, 20260903, 8, first)
    assert len(three) == 3 and len(eight) == 8
    assert all(np.array_equal(three[j], eight[j]) for j in range(3))
    assert np.array_equal(eight[0], first)      # draw 0 is the caller's legacy permutation
    for i in range(8):
        for j in range(i + 1, 8):
            assert not np.array_equal(eight[i], eight[j])
    other_seed = g.scramble_perms(N_TARGETS, 1, 3, first)
    assert not np.array_equal(other_seed[1], three[1])


def test_the_scramble_resampled_interval_widens_with_the_spread_across_draws():
    """Identical draws give the fixed-scramble width; draws that disagree give a wider one."""
    g = load_script(GATE, "gate_boot_scr")
    rng = np.random.default_rng(0)
    emb = rng.normal(0.02, 0.05, size=400)
    same = np.tile(rng.normal(0.0, 0.05, size=400), (6, 1))
    lo1, hi1 = g.boot_ci_scr(emb, same, 400, g.boot_rng_scr(7))
    shifted = same + np.linspace(-0.01, 0.01, 6)[:, None]     # the draws disagree by +-0.01
    lo2, hi2 = g.boot_ci_scr(emb, shifted, 400, g.boot_rng_scr(7))
    assert lo1 < (emb - same[0]).mean() < hi1
    assert lo2 < (emb - shifted.mean(0)).mean() < hi2     # brackets the M-draw-MEAN margin
    assert (hi2 - lo2) > (hi1 - lo1)
    # the draws are redrawn with replacement and AVERAGED: the permutation term shrinks like
    # 1/sqrt(M), so with 6 draws +-0.01 apart the widening is far under the +-0.01 one draw carries
    assert (hi2 - lo2) - (hi1 - lo1) < 0.02


# ------------------------------------------------------------ the paired compare (T89)

def informative_table(g, corpus: Path, names: list[str], dim: int, path: Path) -> None:
    """An embedding built FROM the responses: neighbours in it share a response, so it carries
    real geometry, unlike the random fixture table."""
    import torch

    labels_, _, d = g.load_delta(corpus.name)
    r = d - d.mean(0)
    _, _, vt = np.linalg.svd(r, full_matrices=False)
    coords = r @ vt[:dim].T
    table = {(CURRENT if n == RETIRED else n): torch.tensor(coords[i], dtype=torch.float32)
             for i, n in enumerate(labels_)}
    torch.save(table, path)


COMPARE_LINE = re.compile(r"^  (.+?)\s+([+-]\d\.\d{4}) \[ ([+-]\d\.\d{4}), ([+-]\d\.\d{4}) \] -> "
                          r"([^|]+?)(?: \| A ahead on (\d+)% of targets)?$")


def compare_lines(out: str) -> dict[str, tuple[float, float, float, str]]:
    """label -> (difference, lo, hi, lead) for every difference line `compare` printed."""
    rows = {}
    for line in out.splitlines():
        m = COMPARE_LINE.match(line)
        if m:
            label, point, lo, hi, lead, _share = m.groups()
            rows[label] = (float(point), float(lo), float(hi), lead.strip())
    return rows


def test_dump_holds_the_per_target_cosines_on_the_footing(corpora, capsys, monkeypatch, tmp_path):
    g, emb = corpora
    dump = tmp_path / "row.npz"
    out = run_gate(g, cross_argv_scr(emb, 50, 4, dump), capsys, monkeypatch)
    assert f"per-target cosines -> {dump}" in out
    z = g.load_dump(dump)
    assert z["mode"] == "cross" and z["n_scrambles"] == 4 and z["k_header"] == 25
    assert len(z["targets"]) == N_TARGETS
    assert z["emb_k25"].shape == (N_TARGETS,)
    assert z["scr_k25"].shape == (4, N_TARGETS)
    assert z["gain_emb"].shape == (len(g.WS), N_TARGETS)
    assert z["gain_scr"].shape == (4, len(g.WS), N_TARGETS)
    # the printed k=25 margin is the dump's draw-0 margin, to the printed precision
    printed = next(l for l in out.splitlines() if "k=  25  embedding" in l)
    margin = float(printed.split("margin")[1])
    assert abs((z["emb_k25"] - z["scr_k25"][0]).mean() - margin) < 5e-5
    assert np.array_equal(z["emb_kh"], z["emb_k25"])       # header k is 25 by default


def test_compare_refuses_two_rows_that_are_not_on_one_footing(corpora, capsys, monkeypatch,
                                                              tmp_path):
    g, emb = corpora
    full = tmp_path / "full.npz"
    run_gate(g, within_argv_scr(emb, 20, 2, full), capsys, monkeypatch)
    monkeypatch.setattr(g, "ALIAS", {})                  # one target fewer resolves
    short = tmp_path / "short.npz"
    run_gate(g, within_argv_scr(emb, 20, 2, short), capsys, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["gate", "compare", str(full), str(short), "--bootstrap",
                                      "20"])
    with pytest.raises(SystemExit, match="not on the same footing"):
        g.main()


def test_compare_of_a_row_with_itself_is_exactly_zero_everywhere(corpora, capsys,
                                                                                  monkeypatch,
                                                                                  tmp_path):
    g, emb = corpora
    dump = tmp_path / "row.npz"
    run_gate(g, cross_argv_scr(emb, 20, 3, dump), capsys, monkeypatch)
    out = run_gate(g, ["compare", str(dump), str(dump), "--bootstrap", "50"], capsys,
                   monkeypatch)
    rows = compare_lines(out)
    assert len(rows) == 8            # per k: arm, fixed margin, resampled margin; fusion x 2
    assert sum("permutation resampled" in k for k in rows) == 3
    # one target draw AND one scramble draw serve both rows, so a row against itself is zero
    # everywhere -- a non-zero width on a resampled line would mean the draws were not shared
    for k, (point, lo, hi, lead) in rows.items():
        assert point == lo == hi == 0.0 and lead == "indistinguishable", (k, rows[k])


def test_compare_finds_the_informative_table_and_names_the_lead(corpora, capsys, monkeypatch,
                                                                tmp_path):
    """A table built from the responses beats a random one, PAIRED, at k=10 and k=25."""
    g, emb = corpora
    good = tmp_path / "good.pt"
    informative_table(g, tmp_path / "a.npz", labels()[1:], DIM, good)
    dump_good, dump_rand = tmp_path / "good.npz", tmp_path / "rand.npz"
    run_gate(g, within_argv_scr(good, 50, 3, dump_good), capsys, monkeypatch)
    run_gate(g, within_argv_scr(emb, 50, 3, dump_rand), capsys, monkeypatch)

    js = tmp_path / "cmp.json"
    out = run_gate(g, ["compare", str(dump_good), str(dump_rand), "--bootstrap", "200",
                       "--json", str(js)], capsys, monkeypatch)
    rows = compare_lines(out)
    for k in (10, 25):
        for label in (f"k={k} embedding arm", f"k={k} margin, scramble draw 0 fixed on both rows",
                      f"k={k} margin, permutation resampled (3 draws, shared)"):
            point, lo, hi, lead = rows[label]
            assert point > 0.0 and lo > 0.0 and lead == "A leads", (label, rows[label])

    # the other way round the same interval flips sign and B leads
    flipped = run_gate(g, ["compare", str(dump_rand), str(dump_good), "--bootstrap", "200"],
                       capsys, monkeypatch)
    frows = compare_lines(flipped)
    p, lo, hi, lead = frows["k=25 embedding arm"]
    q, lo2, hi2, _ = rows["k=25 embedding arm"]
    assert lead == "B leads" and abs(p + q) < 1e-9 and abs(lo + hi2) < 1e-9

    import json
    blob = json.loads(js.read_text())
    assert blob["a"] == "good" and blob["b"] == "rand" and blob["n_targets"] == N_TARGETS
    assert blob["mode"] == "within" and blob["n_boot"] == 200 and blob["seed"] == 20260903
    assert blob["scrambles"] == [3, 3] and blob["scrambles_paired"] == 3
    assert blob["seed_rows"] == 20260903 and blob["corpus_a"] == "a.npz"
    assert {r["what"] for r in blob["rows"]} == set(rows)
    for r in blob["rows"]:                       # the JSON IS the printed table, to 4 dp
        assert set(r) == {"what", "diff", "lo", "hi", "lead", "share_a"}
        assert r["share_a"] is None or 0.0 <= r["share_a"] <= 1.0
        point, lo, hi, lead = rows[r["what"]]
        assert (round(r["diff"], 4), round(r["lo"], 4), round(r["hi"], 4)) == (point, lo, hi)
        assert r["lead"] == lead and r["lo"] <= r["diff"] <= r["hi"]


def test_compare_is_the_same_with_the_same_seed_and_refuses_bootstrap_zero(corpora, capsys,
                                                                            monkeypatch, tmp_path):
    g, emb = corpora
    dump = tmp_path / "row.npz"
    run_gate(g, within_argv_scr(emb, 20, 2, dump), capsys, monkeypatch)
    one = run_gate(g, ["compare", str(dump), str(dump), "--bootstrap", "30"], capsys, monkeypatch)
    two = run_gate(g, ["compare", str(dump), str(dump), "--bootstrap", "30"], capsys, monkeypatch)
    assert one == two
    monkeypatch.setattr(sys, "argv", ["gate", "compare", str(dump), str(dump), "--bootstrap", "0"])
    with pytest.raises(SystemExit, match="paired difference IS the interval"):
        g.main()


def test_compare_refuses_two_rows_scored_on_different_corpora(corpora, capsys, monkeypatch,
                                                              tmp_path):
    """Same table, same 40 target names, a different truth corpus: a lead here would be a
    statement about the corpora wearing a table's name."""
    g, emb = corpora
    on_a, on_b = tmp_path / "on_a.npz", tmp_path / "on_b.npz"
    run_gate(g, within_argv_scr(emb, 20, 2, on_a), capsys, monkeypatch)
    run_gate(g, ["within", "b.npz", "--embeddings", str(emb), "--bootstrap", "20",
                 "--scrambles", "2", "--dump", str(on_b)], capsys, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["gate", "compare", str(on_a), str(on_b), "--bootstrap", "20"])
    with pytest.raises(SystemExit, match="different corpora"):
        g.main()


def test_compare_refuses_two_rows_scored_with_different_seeds(corpora, capsys, monkeypatch,
                                                              tmp_path):
    """Draw j is the same permutation only under one seed; under two, nothing is paired."""
    g, emb = corpora
    s1, s2 = tmp_path / "s1.npz", tmp_path / "s2.npz"
    run_gate(g, within_argv_scr(emb, 20, 2, s1), capsys, monkeypatch)
    run_gate(g, within_argv_scr(emb, 20, 2, s2) + ["--seed", "7"], capsys, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["gate", "compare", str(s1), str(s2), "--bootstrap", "20"])
    with pytest.raises(SystemExit, match="different seeds"):
        g.main()


def test_compare_pairs_on_the_common_draws_when_the_rows_differ_in_m(corpora, capsys,
                                                                     monkeypatch, tmp_path):
    g, emb = corpora
    m3, m5 = tmp_path / "m3.npz", tmp_path / "m5.npz"
    run_gate(g, cross_argv_scr(emb, 20, 3, m3), capsys, monkeypatch)
    run_gate(g, cross_argv_scr(emb, 20, 5, m5), capsys, monkeypatch)
    out = run_gate(g, ["compare", str(m3), str(m5), "--bootstrap", "30"], capsys, monkeypatch)
    assert "scramble draws: A 3, B 5 -> paired on the 3 common draws" in out
    assert "permutation resampled (3 draws, shared)" in out
    # draws 0-2 are the same permutations in both dumps, so the row is paired against itself
    # on those and every resampled line is exactly zero
    rows = compare_lines(out)
    for k, (point, lo, hi, lead) in rows.items():
        assert point == lo == hi == 0.0, (k, rows[k])


def test_cross_refuses_a_header_k_at_or_above_the_footing(corpora, capsys, monkeypatch):
    """At k >= n every target is its own neighbour and the arm is the zero vector; the k sweep
    skips such k, and the header arm must refuse rather than print zeros with intervals."""
    g, emb = corpora
    monkeypatch.setattr(sys, "argv", ["gate", *cross_argv(emb, 0), "-k", str(N_TARGETS)])
    with pytest.raises(SystemExit, match="needs more than"):
        g.main()


# ------------------------------------------------------------- the extraction assay (T89)

def paralog_table(g, dim: int, seed: int, structured: bool):
    """A table over the gate's own PARALOG_PAIRS plus filler genes. `structured` puts each pair's
    two genes on nearly the same vector; otherwise every vector is an independent draw."""
    import torch

    rng = np.random.default_rng(seed)
    table = {}
    for a, b in g.PARALOG_PAIRS:
        base = rng.normal(size=dim)
        table[a] = torch.tensor(base, dtype=torch.float32)
        table[b] = torch.tensor(base + 0.1 * rng.normal(size=dim) if structured
                                else rng.normal(size=dim), dtype=torch.float32)
    for i in range(200):
        table[f"FILLER{i:03d}"] = torch.tensor(rng.normal(size=dim), dtype=torch.float32)
    return table


def test_extraction_assay_separates_a_real_table_from_a_random_tensor():
    g = load_script(GATE, "gate_assay")
    good = g.extraction_assay(paralog_table(g, 16, 1, structured=True), 20260903)
    bad = g.extraction_assay(paralog_table(g, 16, 2, structured=False), 20260903)
    assert good["n_pairs"] == len(g.PARALOG_PAIRS) == bad["n_pairs"]
    assert good["ok"] and good["z"] > 5.0
    assert not bad["ok"] and abs(bad["z"]) < 2.0
    assert g.extraction_assay({"GENEA": np.ones(4), "GENEB": np.ones(4)}, 1) is None   # too few


def test_extraction_assay_resolves_a_pair_through_the_alias_table():
    g = load_script(GATE, "gate_assay_alias")
    table = paralog_table(g, 8, 3, structured=True)
    # spell one gene by a retired symbol the alias table maps onto a current one in the table
    retired = next(k for k, v in g.ALIAS.items() if v in {b for _, b in g.PARALOG_PAIRS})
    current = g.ALIAS[retired]
    n_before = g.extraction_assay(table, 1)["n_pairs"]
    table[retired] = table.pop(current)              # the table now only knows the retired name
    assert g.extraction_assay(table, 1)["n_pairs"] == n_before - 1       # not resolved that way
    table[current] = table.pop(retired)              # back: current name resolves directly
    assert g.extraction_assay(table, 1)["n_pairs"] == n_before


def test_the_assay_is_appended_and_can_be_switched_off(corpora, capsys, monkeypatch):
    g, emb = corpora
    on = run_gate(g, cross_argv(emb, 20), capsys, monkeypatch)
    off = run_gate(g, cross_argv(emb, 20) + ["--no-assay"], capsys, monkeypatch)
    assert on.startswith(off)                        # appended after everything else
    assert "extraction assay" in on and "extraction assay" not in off
    assert "not run: fewer than" in on               # the fixture's genes are not paralogues
