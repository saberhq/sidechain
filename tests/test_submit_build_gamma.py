"""`submit.build` with the transfer exponent: gamma != 1 makes the shifts context-specific.

The mirror side (`eval.loco --gamma`) predicts one context, so a single pooled-shift pass was
enough. A submission emits into THREE contexts from one pass, and gamma reads each context's
own control profile -- so the builder had to move pooling inside the per-context write loop.
Two contracts are pinned here: the gamma = 1 path is byte-identical to the historical single
pass (every shipped build), and gamma != 1 really does consult each context's own profile
rather than computing one set of shifts and reusing it.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml

from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.submit import build

GENES = ["A", "B", "C"]


def _cache(path, ctrl_label, ctrl_cpm, pert_cpm, n_cells=1000):
    m = np.asarray([ctrl_cpm, pert_cpm], dtype=float)
    n = np.full(2, n_cells, dtype=np.int64)
    PseudobulkSums(
        labels=[ctrl_label, "TP53"], genes=np.array(GENES, dtype=object),
        count_sum=m * n[:, None], cpm_sum=m * n[:, None],
        cpm_sq_sum=(m**2 + 1.0) * n[:, None],
        n_cells=n, libsize_sum=n.astype(float) * 2e4, sources=["test"],
    ).save(path)


def _controls_h5ad(path, counts_per_cell, n=12):
    X = sp.csr_matrix(np.tile(np.asarray(counts_per_cell, float), (n, 1)))
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=GENES)).write_h5ad(path)


@pytest.fixture()
def challenge(tmp_path):
    """A 3-gene, 1-perturbation, 2-context challenge; context X barely expresses gene A
    (5,000 CPM) while context Y is dominated by it (500,000 CPM)."""
    data = tmp_path / "data"; data.mkdir()
    (data / "gene_names.csv").write_text("gene_name\n" + "\n".join(GENES) + "\n")
    (data / "pert_counts.csv").write_text("target_gene\nTP53\n")
    _controls_h5ad(data / "ctx_x.h5ad", [10, 1000, 990])       # 2,000 UMI/cell
    _controls_h5ad(data / "ctx_y.h5ad", [1000, 505, 495])
    _cache(data / "h1.npz", "non-targeting",
           [100000.0, 450000.0, 450000.0], [200000.0, 400000.0, 400000.0])
    _cache(data / "gwps.npz", "control",
           [100000.0, 450000.0, 450000.0], [200000.0, 400000.0, 400000.0])
    cfg = {
        "data_dir": str(data), "gene_names_file": "gene_names.csv", "n_genes": 3,
        "pert_counts_file": "pert_counts.csv", "pert_col": "target_gene",
        "context_col": "context", "control_label": "non-targeting",
        "phase": "p1", "phases": {"p1": {"contexts": ["X", "Y"]}},
        "control_files": {"X": "ctx_x.h5ad", "Y": "ctx_y.h5ad"},
        "submission": {"cells_per_pert": 6, "max_counts_per_cell": 1_000_000,
                       "max_cells": 100_000, "max_stored_entries": 10_000_000},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return {"cfg": cfg_path, "data": data, "out": tmp_path / "out"}


def _run(ch, stem, extra):
    rc = build.main([
        "--challenge-config", str(ch["cfg"]), "--emitter", "delta-transfer",
        "--h1-cache", str(ch["data"] / "h1.npz"), "--gwps-cache", str(ch["data"] / "gwps.npz"),
        "--out", str(ch["out"] / stem), "--no-pack", "--min-libsize", "100",
        "--no-shrink", *extra])
    assert rc == 0
    return ad.read_h5ad(ch["out"] / f"{stem}.h5ad")


def _share_of_A(pred, ctx):
    cells = pred[pred.obs["context"].astype(str) == ctx]
    totals = np.asarray(cells.X.sum(axis=0)).ravel()
    return totals[0] / totals.sum()


def test_gamma_1_is_identical_to_the_historical_single_pass(challenge):
    """The restructure (finalize + per-context hook) must leave every shipped build's
    bytes alone: an explicit --gamma 1.0 and the default must produce the identical
    cell matrix at the same seed."""
    a = _run(challenge, "default", [])
    b = _run(challenge, "explicit", ["--gamma", "1.0"])
    assert np.array_equal(a.X.toarray(), b.X.toarray())
    assert list(a.obs["context"]) == list(b.obs["context"])


def test_gamma_reads_each_context_s_own_control_profile(challenge):
    """The mutant this exists to kill: computing gamma shifts once (against whichever
    profile) and reusing them across contexts. At gamma = 0 the source's absolute
    +100k-CPM change on gene A lands as a ~21x multiplier in context X (A at 5k CPM)
    but only ~1.2x in context Y (A at 500k CPM) -- so X's emitted share of A must
    leap versus its gamma = 1 build while Y's must FALL. A shared-shift mutant moves
    both contexts the same way and fails one of the two."""
    g1 = _run(challenge, "g1", ["--gamma", "1.0"])
    g0 = _run(challenge, "g0", ["--gamma", "0.0"])
    assert _share_of_A(g0, "X") > 3 * _share_of_A(g1, "X")
    assert _share_of_A(g0, "Y") < _share_of_A(g1, "Y")


def test_gamma_on_a_non_transfer_emitter_is_refused(challenge):
    with pytest.raises(SystemExit) as exc:
        build.main(["--challenge-config", str(challenge["cfg"]), "--emitter", "h1-mean-shift",
                    "--h1-cache", str(challenge["data"] / "h1.npz"),
                    "--out", str(challenge["out"] / "bad"), "--no-pack", "--gamma", "0.5"])
    assert exc.value.code == 2      # argparse error, before any work


def test_gamma_is_recorded_in_the_args_sidecar(challenge):
    import json

    _run(challenge, "sidecar", ["--gamma", "0.75"])
    rec = json.loads((challenge["out"] / "sidecar.args.json").read_text())
    assert rec["gamma"] == 0.75
