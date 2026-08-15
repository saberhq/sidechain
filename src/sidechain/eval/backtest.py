"""True 2025 backtest: train on ALL training perturbations, score against Arc's
released held-out AnnData.

Unlike `dry_run` (which carves a holdout out of the training set to mirror
challenge-time conditions), this scores against the real answers Arc published
on 2025-12-16 -- the 50-perturbation validation set that drove the live
leaderboard, or the 100-perturbation test set that decided the final ranking.
Numbers from this script ARE comparable to the 2025 leaderboard, up to local-
mirror caveats (control subsampling, cell-eval version).

    uv run python -m sidechain.eval.backtest --holdout validation --rungs 0,0b,1

Memory notes (17 GB Mac): the training file is freed after fitting, before the
holdout is materialized; the fitted rungs share one control pool instead of
holding a ~2.5 GB copy each; and the holdout stays `backed` until its counts
have been read and its controls subsampled.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np

from sidechain.data import loaders
from sidechain.eval.dry_run import RUNGS, _log_mem
from sidechain.eval.local_mirror import score, summarize

logger = logging.getLogger("sidechain.backtest")


def run(
    holdout: str = "validation",
    challenge_config: str = "challenges/vcc2025/config.yaml",
    eval_config: str = "configs/eval.yaml",
    rungs: tuple[str, ...] = ("0", "0b", "1"),
    outdir: str | None = None,
    seed: int = 0,
    max_control_cells: int | None = 8000,
) -> dict:
    from sidechain.utils.paths import resolve_config

    t0 = time.time()
    cfg = loaders.load_challenge_config(resolve_config(challenge_config))
    pert_col = cfg.get("pert_col", "target_gene")
    control = cfg.get("control_label", "non-targeting")
    data_dir = Path(cfg["data_dir"]).expanduser()

    external = cfg.get("external_holdouts") or {}
    if holdout not in external:
        raise ValueError(
            f"No external holdout {holdout!r} in the challenge config; "
            f"available: {sorted(external)}"
        )
    holdout_path = data_dir / external[holdout]
    outdir_p = Path(outdir or f"~/data/sidechain/runs/backtest_2025_{holdout}").expanduser()
    outdir_p.mkdir(parents=True, exist_ok=True)

    # -- phase 1: fit every requested rung on the FULL training set --
    train = loaders.load_challenge_split(cfg, split="all", dev=False)
    logger.info("Training file: %d cells x %d genes", train.n_obs, train.n_vars)
    train_var = train.var.copy()
    n_train_perts = train.obs[pert_col].astype(str).nunique() - 1
    if loaders.is_discrete(train):
        train = loaders.normalize_counts(train, inplace=True)
    _log_mem("train normalized")

    models: dict[str, tuple[str, object]] = {}
    shared_pool = None
    for key in rungs:
        if key not in RUNGS:
            raise ValueError(f"Unknown rung {key!r}; known: {sorted(RUNGS)}")
        label, cls = RUNGS[key]
        logger.info("fit %s", label)
        model = cls(pert_col=pert_col, control_label=control).fit(train)
        # Every rung's fit slices an identical control pool out of the training
        # file. Keep one and share it -- three private copies would be ~7 GB.
        if shared_pool is None:
            shared_pool = model.control_cells_
        else:
            model.control_cells_ = shared_pool
        models[key] = (label, model)
        gc.collect()
        _log_mem(f"rung {key} fitted")
    del train
    gc.collect()
    _log_mem("training file freed")

    # -- phase 2: open the real holdout, subsample its controls, materialize --
    backed = loaders.load_h5ad(holdout_path, backed="r")
    labels = backed.obs[pert_col].astype(str)
    holdout_perts = sorted(set(labels) - {control})
    keep = np.ones(backed.n_obs, dtype=bool)
    ctrl_idx = np.flatnonzero((labels == control).to_numpy())
    if max_control_cells and len(ctrl_idx) > max_control_cells:
        rng = np.random.default_rng(seed)
        drop = rng.choice(ctrl_idx, len(ctrl_idx) - max_control_cells, replace=False)
        keep[drop] = False
        logger.info(
            "Subsampled holdout controls %d -> %d (eval cost; applied to truth "
            "and mirrored in the prediction)",
            len(ctrl_idx),
            max_control_cells,
        )
    real = backed[keep].to_memory()
    del backed
    gc.collect()

    # The released holdout files carry QC-style var columns instead of the
    # training var (no gene_id). var_names are identical and in identical order
    # -- asserted here -- so adopt the training var wholesale to keep pred and
    # real structurally interchangeable downstream.
    if not (real.var_names == train_var.index).all():
        raise ValueError(f"{holdout_path.name}: var_names differ from the training file")
    real.var = train_var.copy()

    if loaders.is_discrete(real):
        real = loaders.normalize_counts(real, inplace=True)
    real_labels = real.obs[pert_col].astype(str)
    counts = {p: int((real_labels == p).sum()) for p in holdout_perts}
    n_control = int((real_labels == control).sum())
    logger.info(
        "external holdout %r: %d perts, %d cells (%d control)",
        holdout,
        len(holdout_perts),
        real.n_obs,
        n_control,
    )
    _log_mem("holdout materialized")

    # -- phase 3: predict and score each rung against the real answers --
    report: dict = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "caveat": (
            f"TRUE EXTERNAL HOLDOUT: trained on all {n_train_perts} training "
            f"perturbations, scored against Arc's released {holdout} set "
            "(published 2025-12-16). Comparable to the 2025 leaderboard up to "
            "local-mirror caveats (control subsampling, cell-eval version)."
        ),
        "holdout": holdout,
        "holdout_file": str(holdout_path),
        "n_train_perts": n_train_perts,
        "n_holdout_cells": int(real.n_obs),
        "n_holdout_perts": len(holdout_perts),
        "n_control_cells_scored": n_control,
        "max_control_cells": max_control_cells,
        "holdout_perturbations": holdout_perts,
        "rungs": {},
    }

    de_real_cache: Path | None = None
    for key, (label, model) in models.items():
        logger.info("=== %s ===", label)
        t_rung = time.time()
        pred = model.predict(counts, seed=seed, include_control=n_control)
        _log_mem(f"rung {key} predicted")
        res = score(
            pred,
            real,
            eval_config=eval_config,
            challenge_config=resolve_config(challenge_config),
            outdir=str(outdir_p / f"rung{key}"),
            de_real=de_real_cache,
        )
        if de_real_cache is None:
            candidate = Path(res.get("de_real_path", ""))
            if candidate.exists():
                de_real_cache = candidate
                logger.info("Cached real-data DE for the remaining rungs: %s", candidate)
        print(f"\n--- {label} ---")
        print(summarize(res))
        report["rungs"][key] = {
            "label": label,
            "metrics": res["metrics"],
            "challenge_metrics": res["challenge_metrics"],
            "guardrails": res.get("guardrails", {}),
            "elapsed_sec": round(time.time() - t_rung, 1),
        }
        del pred, res
        gc.collect()

    report["elapsed_sec"] = round(time.time() - t0, 1)
    (outdir_p / "report.json").write_text(json.dumps(report, indent=2, default=float))
    logger.info("Wrote %s", outdir_p / "report.json")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout", default="validation", help="key under external_holdouts")
    ap.add_argument("--challenge-config", default="challenges/vcc2025/config.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--rungs", default="0,0b,1", help="comma-separated: 0,0b,1")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max-control-cells",
        type=int,
        default=8000,
        help="cap holdout control cells used for scoring (0 = keep all)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = run(
        holdout=args.holdout,
        challenge_config=args.challenge_config,
        eval_config=args.eval_config,
        rungs=tuple(r.strip() for r in args.rungs.split(",") if r.strip()),
        outdir=args.outdir,
        seed=args.seed,
        max_control_cells=args.max_control_cells or None,
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
