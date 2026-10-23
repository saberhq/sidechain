"""Contract tests for scripts/site_controls.py: the site's control-cells data is built field by field, so nothing the
page does not name reaches it, and a string outside the page's vocabulary stops the build."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("site_controls", REPO / "scripts" / "site_controls.py")
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)

GENES = ["MT-CO1", "GAPDH", "ACTB", "TSPAN6", "EEF2"]
LEAK = "SomeLine-like"


def _stage(root: Path, n: int = 12) -> Path:
    rng = np.random.default_rng(0)
    src = root / "challenge"
    src.mkdir()
    metrics = {"total_umi": "UMIs per cell", "n_genes": "genes detected per cell", "pct_mt": "% mito",
               "pct_ribo": "% ribo", "S_score": "S-phase score"}
    shown = ["total_umi", "n_genes", "pct_mt", "S_score"]
    guides = [f"non-targeting-{i}" for i in (1, 2, 3)]
    summ = {"contexts": {}, "metrics": metrics, "targets": {"validation": ["GAPDH", "ACTB"], "final": ["EEF2"]},
            "private_note": LEAK}
    for c in ("A", "D"):
        summ["contexts"][c] = {"context": c, "unlisted_note": LEAK, "unlisted_list": [LEAK], "n_cells": n, "n_genes_axis": 5,
                               "n_guides": 3, "median": {m: 1.5 for m in shown}, "mean": {m: 1.0 for m in shown},
                               "phase_share": {"G1": .2, "S": .5, "G2M": .3}, "metrics_absent": ["pct_ribo"],
                               "n_clusters": 2, "n_clusters_resid": 2, "pca_var": [.1, .05], "guides_off": {}, "seconds": 1}
        cells = {"context": c, "n": n, "metrics": {m: rng.random(n).tolist() for m in shown},
                 "cat": {"guide": {"levels": guides, "codes": (np.arange(n) % 3).tolist()},
                         "phase": {"levels": ["G1", "S", "G2M"], "codes": (np.arange(n) % 3).tolist()},
                         "cluster": {"levels": ["0", "1"], "codes": (np.arange(n) % 2).tolist()},
                         "cluster_resid": {"levels": ["0", "1"], "codes": (np.arange(n) % 2).tolist()},
                         "label": {"levels": [LEAK], "codes": [0] * n}},
                 "emb": {k: rng.normal(size=(d, n)).tolist() for k, d in
                         (("umap", 2), ("pca", 4), ("umap_resid", 2), ("pca_resid", 4))},
                 "pca_var": [.1, .05], "pca_var_resid": [.08, .04], "unlisted_note": LEAK}
        (src / f"cells_{c}.json").write_text(json.dumps(cells))
        rows = {k: rng.random(3).tolist() for k in sc.GUIDE_FIELDS if k != "guide"}
        rows.update(guide=guides, n=[4, 4, 4], note=[LEAK] * 3)
        (src / f"guides_{c}.json").write_text(json.dumps({"context": c, "n_perm": 500, "rows": rows}))
        genes = {"context": c, **{k: rng.random(5).tolist() for k in ("mean_count", "detect", "mean_cp10k", "mean_log", "var_log",
                                                                      "dispersion_norm")},
                 "hvg": [True, False, True, False, False], "top20_share": [{"gene": "GAPDH", "q": [0, 1, 2, 3, 4]}]}
        (src / f"genes_{c}.json").write_text(json.dumps(genes))
    (src / "summary.json").write_text(json.dumps(summ))
    (src / "gene_index.json").write_text(json.dumps({"genes": GENES}))
    (src / "joint.json").write_text(json.dumps({"contexts": ["A", "D"], "ctx_codes": [0] * n + [1] * n,
                                                "umap": rng.normal(size=(2, 2 * n)).tolist(),
                                                "pca": rng.normal(size=(4, 2 * n)).tolist(), "pca_var": [.1, .05],
                                                "cluster": [0] * (2 * n), "genes": [LEAK]}))
    (src / "manifest.json").write_text(json.dumps({"built": "2026-10-21T23:48:01-07:00", "commit": "a695f44",
                                                   "notes": [LEAK]}))
    return src


def test_nothing_unnamed_reaches_the_site(tmp_path):
    files = sc.build(_stage(tmp_path))
    text = json.dumps(files)
    assert LEAK not in text
    for field in ("unlisted_note", "unlisted_list", "private_note", "label", "dispersion_norm", "notes", "mean_log"):
        assert f'"{field}"' not in text, field
    assert sc.check(files) == []
    assert set(files) == {"index.json", "genes.json", "joint.json", *(f"{k}_{c}.json" for k in ("cells", "emb", "guides", "genes")
                                                                       for c in ("A", "D"))}
    assert "pct_ribo" not in files["index.json"]["metrics"]          # absent in every context: not on the page
    assert files["index.json"]["contexts"]["D"]["round"] == "test"


def test_a_stray_string_stops_the_build(tmp_path):
    files = sc.build(_stage(tmp_path))
    files["cells_A.json"]["cat"]["guide"]["levels"][0] = LEAK
    assert sc.check(files) == [f"cells_A.json: {LEAK!r}"]


def test_check_mode_reads_a_written_directory(tmp_path):
    out = tmp_path / "out"
    assert sc.main(["--src", str(_stage(tmp_path)), "--out", str(out)]) == 0
    assert sc.main(["--check", "--out", str(out)]) == 0
    p = out / "guides_A.json"
    g = json.loads(p.read_text())
    g["rows"]["guide"][1] = LEAK
    p.write_text(json.dumps(g))
    assert sc.main(["--check", "--out", str(out)]) == 1


def test_coordinates_and_counts_are_slimmed(tmp_path):
    files = sc.build(_stage(tmp_path))
    umap = files["emb_A.json"]["umap"][0]
    assert all(round(v, 2) == v for v in umap)
    assert all(isinstance(v, int) for v in files["cells_A.json"]["metrics"]["n_genes"])


@pytest.mark.skipif(not (sc.OUT / "index.json").exists(), reason="the site's control-cells data is not built here")
def test_the_built_site_data_passes_the_vocabulary_check():
    assert sc.main(["--check"]) == 0
