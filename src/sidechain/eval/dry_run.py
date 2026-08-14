"""End-to-end rehearsal of the whole loop on 2025 data.

    load split -> normalize -> fit rung -> predict held-out perturbations
    -> score on the local cell-eval mirror

This is a PIPELINE CHECK, not a leaderboard estimate. The holdout here is carved out
of the training set by `loaders.carve_holdout`, mirroring challenge-time conditions
(entrants only had names and cell counts for held-out perturbations). Numbers from
this script must never be described as "what we would have placed in 2025". Arc
published the real validation/test AnnData post-challenge (2025-12-16); scoring
against those is a separate external holdout, not this script.

    uv run python -m sidechain.eval.dry_run --rungs 0,0b,1 --n-holdout 25
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


def _log_mem(stage: str) -> None:
    """Report resident memory. The full 2025 file is 221k x 18k, so knowing where
    the peak lands is the difference between planning for 2026 and discovering a
    limit mid-run."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports kilobytes.
        gb = peak / 1e9 if peak > 1e7 else peak / 1e6
        logger.info("  [mem] %-22s peak RSS %.2f GB", stage, gb)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break a run
        logger.debug("mem probe unavailable: %s", exc)


def run(
    challenge_config: str = "challenges/vcc2025/config.yaml",
    eval_config: str = "configs/eval.yaml",
    rungs: tuple[str, ...] = ("0", "0b", "1"),
    n_holdout: int = 25,
    max_cells: int | None = None,
    outdir: str = "~/data/sidechain/runs/dry_run_2025",
    seed: int = 0,
    dev: bool = False,
    max_control_cells: int | None = 8000,
) -> dict:
    from sidechain.utils.paths import resolve_config

    t0 = time.time()
    cfg = loaders.load_challenge_config(resolve_config(challenge_config))
    pert_col = cfg.get("pert_col", "target_gene")
    control = cfg.get("control_label", "non-targeting")
    outdir_p = Path(outdir).expanduser()
    outdir_p.mkdir(parents=True, exist_ok=True)

    # -- load --
    # Full training file by default. Iterating on the subset hides exactly the
    # resource limits we need to hit before the 2026 kickoff, and its 4x-lower
    # cells-per-perturbation inflates holdout noise. `dev=True` opts into the
    # small file for a fast smoke test.
    adata = loaders.load_challenge_split(cfg, split="all", dev=dev)
    logger.info(
        "Loaded %d cells x %d genes (%s file)",
        adata.n_obs,
        adata.n_vars,
        "dev" if dev else "FULL",
    )
    _log_mem("after load")

    if max_cells and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        keep = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[np.sort(keep)].copy()
        logger.info("Subsampled to %d cells for a fast loop", adata.n_obs)

    # -- gene ID space: prove the priors could align, and fail loudly if not --
    gi = loaders.gene_index(adata, "ensembl_gene_id")
    logger.info("gene_index: %d Ensembl IDs resolved", len(gi))

    # -- normalize into the space cell-eval expects --
    # inplace: the un-normalized matrix is dead the moment this returns, and on
    # the full file the defensive copy costs ~3 GB and dominates the runtime.
    if loaders.is_discrete(adata):
        logger.info("X looks like raw counts -> normalize_total(1e4) + log1p")
        adata = loaders.normalize_counts(adata, inplace=True)
        _log_mem("after normalize")

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
    # `adata` still pins the whole matrix; train/real are independent copies, so
    # releasing it here is worth several GB on the full file.
    del adata, labels
    gc.collect()
    _log_mem("after split")

    # Controls are over half the evaluation cells (38,176 of 71,525 on the full
    # 2025 holdout) and their only role is to establish the baseline each
    # perturbation is measured against -- a job a few thousand cells do just as
    # well. Subsampling them is the single biggest lever on eval cost, and it is
    # applied to the TRUTH only; the prediction then mirrors the reduced count,
    # so both sides see the same control set size and nothing is biased.
    if max_control_cells:
        ctrl_idx = np.flatnonzero(real.obs[pert_col].astype(str).to_numpy() == control)
        if len(ctrl_idx) > max_control_cells:
            rng = np.random.default_rng(seed)
            drop = set(rng.choice(ctrl_idx, len(ctrl_idx) - max_control_cells, replace=False))
            keep = np.array([i not in drop for i in range(real.n_obs)])
            logger.info(
                "Subsampled holdout controls %d -> %d (eval cost; applied to truth "
                "and mirrored in the prediction)",
                len(ctrl_idx),
                max_control_cells,
            )
            real = real[keep].copy()
            gc.collect()

    # cell-eval requires pred and real to carry the SAME perturbation set,
    # control included, so mirror the real cell counts exactly.
    real_labels = real.obs[pert_col].astype(str)
    counts = {p: int((real_labels == p).sum()) for p in holdout_perts}
    n_control = int((real_labels == control).sum())
    logger.info("eval pair: %d real cells, %d predicted cells", real.n_obs, sum(counts.values()) + n_control)
    _log_mem("eval pair ready")

    # -- ladder --
    report: dict = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "caveat": (
            "Self-carved holdout from the 2025 TRAINING set, mirroring "
            "challenge-time conditions. This is a pipeline check, not a 2025 "
            "leaderboard estimate. (Arc released the real validation/test AnnData "
            "on 2025-12-16; this run does not use them.)"
        ),
        "dev_file": dev,
        "n_train_cells": int(train.n_obs),
        "n_holdout_cells": int(real.n_obs),
        "n_train_perts": len(train_perts),
        "n_holdout_perts": len(holdout_perts),
        "n_control_cells_scored": n_control,
        "max_control_cells": max_control_cells,
        "holdout_perturbations": holdout_perts,
        "rungs": {},
    }

    de_real_cache: Path | None = None
    for key in rungs:
        if key not in RUNGS:
            raise ValueError(f"Unknown rung {key!r}; known: {sorted(RUNGS)}")
        label, cls = RUNGS[key]
        logger.info("=== %s ===", label)
        t_rung = time.time()
        model = cls(pert_col=pert_col, control_label=control).fit(train)
        _log_mem(f"rung {key} fitted")
        pred = model.predict(counts, seed=seed, include_control=n_control)
        _log_mem(f"rung {key} predicted")

        # Every rung is scored against the SAME holdout, so the truth's DE is
        # identical each time -- and on the full file it is the single most
        # expensive step. Compute it on the first rung, reuse it thereafter.
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
        # cell-eval holds dense pseudobulk plus two DE frames per run; without
        # this the previous rung's are still resident when the next one starts.
        del model, pred, res
        gc.collect()

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
    ap.add_argument("--outdir", default="~/data/sidechain/runs/dry_run_2025")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dev",
        action="store_true",
        help="use the small dev_file instead of the full training set (smoke tests only)",
    )
    ap.add_argument(
        "--max-control-cells",
        type=int,
        default=8000,
        help="cap holdout control cells used for scoring (0 = keep all). "
        "Controls only set the baseline; capping them is the biggest eval-cost lever.",
    )
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
        dev=args.dev,
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
