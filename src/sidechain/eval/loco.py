"""Leave-one-context-out: predict a held-out line's perturbations from the other
lines, then score on that line's own competition bundle (`mirror2026`).

This is the local stand-in for "an unseen cell line": the held-out line
contributes only its control cells (its `ContextProfile`), exactly as A/B/C do
in the challenge; every per-gene effect comes from the OTHER lines' pseudobulks.
It measures *method* transfer on whatever perturbations the lines share -- not
the challenge panel, which the essential-gene screens barely cover
(private reports/06 s5).

    uv run python -m sidechain.eval.loco \
        --real ~/data/sidechain/cache/vcc2026/hepg2_flowtest_real.h5ad --pert-col perturbation --control control \
        --source ~/data/sidechain/cache/vcc2026/k562_essential_all_pseudobulk.npz:control \
        --source ~/data/sidechain/cache/vcc2026/rpe1_all_pseudobulk.npz:control \
        --source ~/data/sidechain/cache/vcc2026/jurkat_all_pseudobulk.npz:control \
        --bundle ~/data/sidechain/runs/mirror/hepg2_flowtest/bundle \
        --out ~/data/sidechain/runs/mirror/hepg2_flowtest/transfer_even --dispersion even

The prediction is built on the real file's own gene axis and cell counts
(so it scores against that file), with the same emitter and the same pooled,
shrunk deltas the challenge submissions use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd
import scipy.sparse as sp

from sidechain.data.lfc_table import LfcTable
from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.eval.mirror2026 import attach_controls, score
from sidechain.models.count_emitters import ContextProfile, PoissonEmitter
from sidechain.submit.build import pooled_delta


def build_transfer_prediction(
    real_path: Path,
    sources: list[tuple[PseudobulkSums, str]],
    out_path: Path,
    *,
    pert_col: str,
    control: str,
    dispersion: str = "even",
    shrinkage: bool = True,
    alpha: float = 1.0,
    cells_per_pert: int | None = None,
    seed: int = 0,
    min_libsize: float = 500.0,
) -> dict:
    """Predict every non-control perturbation of `real_path` from `sources`."""
    real = ad.read_h5ad(real_path)
    labels = real.obs[pert_col].astype(str).to_numpy()
    perts = sorted(set(labels) - {control})
    axis = real.var_names.astype(str).to_numpy()
    ctrl = real[labels == control].copy()
    ctrl_tmp = out_path.parent / f"{out_path.stem}.controls.h5ad"
    ctrl_tmp.parent.mkdir(parents=True, exist_ok=True)
    ctrl.write_h5ad(ctrl_tmp)
    prof = ContextProfile.from_controls(ctrl_tmp, real_path.stem, min_libsize=min_libsize)
    em = PoissonEmitter(prof, seed=seed, dispersion=dispersion)
    gene_pos = {g: i for i, g in enumerate(axis)}
    blocks, obs_labels, covered = [], [], 0
    for p in perts:
        d = pooled_delta(p, sources, axis, shrinkage=shrinkage)
        if d is not None:
            covered += 1
            d = d * alpha
            if p in gene_pos:
                d[gene_pos[p]] = -2.32
        n = cells_per_pert or int((labels == p).sum())
        blocks.append(em.emit(n, d))
        obs_labels += [p] * n
    X = sp.vstack(blocks, format="csr")
    pred = ad.AnnData(X=X, obs=pd.DataFrame({pert_col: obs_labels}, index=[f"pred_{i}" for i in range(len(obs_labels))]),
                      var=pd.DataFrame(index=real.var_names))
    pred.write_h5ad(out_path)
    return {"pred": str(out_path), "perturbations": len(perts), "covered_by_sources": covered,
            "cells": int(pred.n_obs), "genes": int(pred.n_vars), "dispersion": dispersion,
            "shrinkage": shrinkage, "alpha": alpha}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, type=Path)
    ap.add_argument("--pert-col", default="target_gene")
    ap.add_argument("--control", default="non-targeting")
    ap.add_argument("--source", action="append", default=[],
                    help="pseudobulk .npz:control_label (repeatable)")
    ap.add_argument("--lfc-source", action="append", default=[], metavar="NPZ",
                    help="cached LfcTable .npz -- a source publishing the contrast already "
                         "taken rather than cells (e.g. Feng 2026). Repeatable.")
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dispersion", choices=["poisson", "even"], default="even")
    ap.add_argument("--no-shrink", action="store_true")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--cells-per-pert", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--de-backend", default="pdex")
    args = ap.parse_args(argv)

    # `--source` is no longer required on its own: an LfcTable is a complete
    # source, so an arm built only from published contrasts is a legitimate run
    # and scoring one is how you find out what that corpus is worth alone.
    # At least one of the two is still mandatory -- an arm with no sources at
    # all would score the fallback shift and look like a model.
    if not args.source and not args.lfc_source:
        ap.error("need at least one --source or --lfc-source")
    sources = []
    for spec in args.source:
        path, _, ctrl = spec.rpartition(":")
        sources.append((PseudobulkSums.load(path), ctrl or "control"))
    sources += [LfcTable.load(path) for path in args.lfc_source]
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    info = build_transfer_prediction(args.real, sources, out / "pred.h5ad", pert_col=args.pert_col,
                                     control=args.control, dispersion=args.dispersion,
                                     shrinkage=not args.no_shrink, alpha=args.alpha,
                                     cells_per_pert=args.cells_per_pert, seed=args.seed)
    print(json.dumps(info), flush=True)
    with_ctrl = attach_controls(out / "pred.h5ad", args.real, out / "pred_with_controls.h5ad",
                                pert_col=args.pert_col, control=args.control)
    res = score(with_ctrl, args.real, args.bundle, out, pert_col=args.pert_col, control=args.control,
                de_backend=args.de_backend)
    res["build"] = info
    (out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k in ("members", "overall")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
