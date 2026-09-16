"""Contract tests for ``scripts/build_pert_features.py`` -- the perturbation featurizer.

The script turns a ``{symbol: vector}`` gene table into the ``perturbation_features_file`` a STATE
transition model reads instead of a one-hot code. Three things are pinned:

* **The default table is the ESM-2 3B one, and the sidecar says which bytes were used.** Since
  T90 (2026-09-16) a build without ``--embeddings`` reads the table under
  ``derived/transcriptformer-esm2-3b/``; the sidecar carries its path and sha256, so a feature
  file can always be traced to the bytes it came from. When the table is on this machine, its
  sha256 and shape are checked against the constants the script declares.
* **Nothing is zero-filled silently.** A label the table cannot resolve -- directly or through the
  retired-symbol alias table -- makes the script refuse to write; ``--allow-missing`` is the only
  way past, and the miss list is in the sidecar.
* **Controls are explicit zero vectors, keyed by the label as our data spells it.**

The fixtures build a tiny table, so the file runs in a second without the real 196 MB table.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "build_pert_features.py"
RETIRED, CURRENT = "QARS", "QARS1"


def load_script():
    spec = importlib.util.spec_from_file_location("build_pert_features_under_test", BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tiny(tmp_path):
    """A four-gene table keyed on CURRENT symbols, a label file that spells one gene RETIRED."""
    table = {name: torch.full((6,), float(i)) for i, name in enumerate(["GENEA", "GENEB", "GENEC", CURRENT], 1)}
    emb = tmp_path / "table.pt"
    torch.save(table, emb)
    labels = tmp_path / "labels.csv"
    labels.write_text("target_gene\nGENEA\nGENEB\n" + RETIRED + "\nnon-targeting\n")
    return emb, labels, tmp_path / "out" / "features.pt"


def run(mod, argv: list[str], monkeypatch) -> int:
    monkeypatch.setattr(sys, "argv", ["build", *argv])
    return mod.main()


def test_default_table_is_declared_once_and_is_the_esm2_3b_one():
    mod = load_script()
    assert mod.DEFAULT_EMBEDDINGS == Path("~/data/sidechain/derived/transcriptformer-esm2-3b/esm2_3b_raw_table.pt")
    assert len(mod.DEFAULT_EMBEDDINGS_SHA256) == 64


@pytest.mark.skipif(not Path("~/data/sidechain/derived/transcriptformer-esm2-3b/esm2_3b_raw_table.pt")
                    .expanduser().exists(), reason="the ESM-2 3B table is not on this machine")
def test_the_default_table_on_disk_is_the_registered_bytes():
    """The sha256 in the script, the file under derived/, and LINEAGE.json agree."""
    mod = load_script()
    p = mod.DEFAULT_EMBEDDINGS.expanduser()
    assert hashlib.sha256(p.read_bytes()).hexdigest() == mod.DEFAULT_EMBEDDINGS_SHA256
    lineage = json.loads((p.parent / "LINEAGE.json").read_text())
    entry = lineage["entries"]["transcriptformer_esm2_3b/gene_table"]
    assert entry["sha256"] == mod.DEFAULT_EMBEDDINGS_SHA256
    table = torch.load(p, weights_only=False, map_location="cpu")
    assert len(table) == entry["n_symbols"] == 18_618
    assert next(iter(table.values())).shape[-1] == 2560


def test_features_are_written_with_alias_control_and_sidecar(tiny, monkeypatch, capsys):
    mod = load_script()
    emb, labels, out = tiny
    rc = run(mod, ["--embeddings", str(emb), "--labels", str(labels), "--out", str(out),
                   "--control-label", "non-targeting"], monkeypatch)
    assert rc == 0
    feats = torch.load(out, weights_only=False)
    assert set(feats) == {"GENEA", "GENEB", RETIRED, "non-targeting"}
    assert torch.equal(feats[RETIRED], torch.full((6,), 4.0))      # keyed as OUR data spells it
    assert torch.equal(feats["non-targeting"], torch.zeros(6))      # an explicit zero vector

    side = json.loads(out.with_suffix(".pt.json").read_text())
    assert side["embeddings_source"] == str(emb)
    assert side["embeddings_sha256"] == hashlib.sha256(emb.read_bytes()).hexdigest()
    assert side["feature_dim"] == 6 and side["n_features_written"] == 4
    assert side["aliases"] == {RETIRED: CURRENT} and side["missing"] == []
    assert side["control_labels"] == ["non-targeting"]
    assert "sha256 " in capsys.readouterr().out


def test_an_unresolved_label_refuses_to_write_unless_allowed(tiny, monkeypatch, tmp_path):
    mod = load_script()
    emb, _, out = tiny
    labels = tmp_path / "bad.csv"
    labels.write_text("target_gene\nGENEA\nNOSUCHGENE\n")
    assert run(mod, ["--embeddings", str(emb), "--labels", str(labels), "--out", str(out)],
               monkeypatch) == 1
    assert not out.exists()                                          # nothing zero-filled, nothing written

    assert run(mod, ["--embeddings", str(emb), "--labels", str(labels), "--out", str(out),
                     "--allow-missing"], monkeypatch) == 0
    side = json.loads(out.with_suffix(".pt.json").read_text())
    assert side["missing"] == ["NOSUCHGENE"] and side["n_features_written"] == 1


def test_a_missing_table_is_refused_with_a_pointer(tiny, monkeypatch, tmp_path):
    mod = load_script()
    _, labels, out = tiny
    monkeypatch.setattr(sys, "argv", ["build", "--embeddings", str(tmp_path / "nope.pt"),
                                      "--labels", str(labels), "--out", str(out)])
    with pytest.raises(SystemExit, match="not found"):
        mod.main()
