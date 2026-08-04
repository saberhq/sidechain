"""End-to-end rehearsal of the whole loop on 2025 data.

    load split -> normalize -> fit rung -> predict held-out perturbations
    -> score on the local cell-eval mirror

This is a PIPELINE CHECK, not a leaderboard estimate. Arc never released the 2025
public/private test AnnData to entrants, so the holdout here is carved out of the
training set by `loaders.carve_holdout`. Numbers from this script must never be
described as "what we would have placed in 2025".

    uv run python -m sidechain.eval.dry_run --rungs 0,0b,1 --n-holdout 25
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from sidechain.data import loaders
from sidechain.eval.local_mirror import score, summarize
from sidechain.models.baseline_stats import (
    PredictControl,
    PredictMeanPerturbation,
    StatisticalBackbone,
)

logger = logging.getLogger("sidechain.dry_run")

RUNGS = {
    "0": ("Rung 0  predict-control", PredictControl),
    "0b": ("Rung 0b predict-mean-perturbation", PredictMeanPerturbation),
    "1": ("Rung 1  statistical backbone", StatisticalBackbone),
}


def run(
    challenge_config: str = "challenges/vcc2025/config.yaml",
    eval_config: str = "configs/eval.yaml",
    rungs: tuple[str, ...] = ("0", "0b", "1"),
    n_holdout: int = 25,
    max_cells: int | None = None,
    outdir: str = "runs/dry_run_2025",
    seed: int = 0,
) -> dict:
    from sidechain.utils.paths import resolve_config

    t0 = time.time()
    cfg = loaders.load_challenge_config(resolve_config(challenge_config))
    pert_col = cfg.get("pert_col", "target_gene")
    control = cfg.get("control_label", "non-targeting")
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    # -- load --
    adata = loaders.load_challenge_split(cfg, split="all", dev=True)
    logger.info("Loaded %d cells x %d genes", adata.n_obs, adata.n_vars)

    if max_cells and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        keep = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[np.sort(keep)].copy()
        logger.info("Subsampled to %d cells for a fast loop", adata.n_obs)

    # -- gene ID space: prove the priors could align, and fail loudly if not --
    gi = loaders.gene_index(adata, "ensembl_gene_id")
    logger.info("gene_index: %d Ensembl IDs resolved", len(gi))

    # -- normalize into the space cell-eval expects --
    if loaders.is_discrete(adata):
        logger.info("X looks like raw counts -> normalize_total(1e4) + log1p")
        adata = loaders.normalize_counts(adata)

    # -- split --
    train_perts, holdout_perts = loaders.carve_holdout(
        adata,
        n_holdout=n_holdout,
        pert_col=pert_col,
        control_label=control,
        seed=cfg.get("splits", {}).get("seed", seed),
    )
    labels = adata.obs[pert_col].astype(str)
    train = adata[labels.isin(set(train_perts) | {control}).to_numpy()].copy()
    real = adata[labels.isin(set(holdout_perts) | {control}).to_numpy()].copy()
    logger.info(
        "train %d cells / %d perts | holdout %d cells / %d perts",
        train.n_obs,
        len(train_perts),
        real.n_obs,
        len(holdout_perts),
    )

    # cell-eval requires pred and real to carry the SAME perturbation set,
    # control included, so mirror the real cell counts exactly.
    real_labels = real.obs[pert_col].astype(str)
    counts = {p: int((real_labels == p).sum()) for p in holdout_perts}
    n_control = int((real_labels == control).sum())

    # -- ladder --
    report: dict = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "caveat": (
            "Self-carved holdout from the 2025 TRAINING set. Arc never released "
            "public/private test AnnData to entrants. This is a pipeline check, "
            "not a 2025 leaderboard estimate."
        ),
        "n_train_cells": int(train.n_obs),
        "n_holdout_cells": int(real.n_obs),
        "n_train_perts": len(train_perts),
        "n_holdout_perts": len(holdout_perts),
        "holdout_perturbations": holdout_perts,
        "rungs": {},
    }

    for key in rungs:
        if key not in RUNGS:
            raise ValueError(f"Unknown rung {key!r}; known: {sorted(RUNGS)}")
        label, cls = RUNGS[key]
        logger.info("=== %s ===", label)
        model = cls(pert_col=pert_col, control_label=control).fit(train)
        pred = model.predict(counts, seed=seed, include_control=n_control)

        res = score(
            pred,
            real,
            eval_config=eval_config,
            challenge_config=resolve_config(challenge_config),
            outdir=str(outdir_p / f"rung{key}"),
        )
        print(f"\n--- {label} ---")
        print(summarize(res))
        report["rungs"][key] = {
            "label": label,
            "metrics": res["metrics"],
            "challenge_metrics": res["challenge_metrics"],
            "guardrails": res.get("guardrails", {}),
        }

    report["elapsed_sec"] = round(time.time() - t0, 1)
    (outdir_p / "report.json").write_text(json.dumps(report, indent=2, default=float))
    logger.info("Wrote %s", outdir_p / "report.json")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--challenge-config", default="challenges/vcc2025/config.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--rungs", default="0,0b,1", help="comma-separated: 0,0b,1")
    ap.add_argument("--n-holdout", type=int, default=25)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--outdir", default="runs/dry_run_2025")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = run(
        challenge_config=args.challenge_config,
        eval_config=args.eval_config,
        rungs=tuple(r.strip() for r in args.rungs.split(",") if r.strip()),
        n_holdout=args.n_holdout,
        max_cells=args.max_cells,
        outdir=args.outdir,
        seed=args.seed,
    )

    print("\n================ SUMMARY ================")
    print(report["caveat"])
    hdr = f"{'rung':34s} {'DES':>9s} {'PDS':>9s} {'MAE':>9s}"
    print(hdr)
    print("-" * len(hdr))
    def fmt(x):
        return f"{x:9.5f}" if isinstance(x, (int, float)) else f"{'n/a':>9s}"

    for r in report["rungs"].values():
        cm = r["challenge_metrics"]
        print(f"{r['label']:34s} {fmt(cm.get('des'))} {fmt(cm.get('pds'))} {fmt(cm.get('mae'))}")
    print("\nDES higher better | PDS higher better | MAE lower better")


if __name__ == "__main__":
    main()
