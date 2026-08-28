"""Local mirror of Arc's cell-eval -- the highest-leverage piece. Nothing is
trusted until it clears this. Emits the 3 challenge metrics (DES/PDS/MAE) plus
STATE's extras (7 total, the Generalist-prize set) and runs the anti-hacking
guardrails.

The three challenge metrics map onto cell-eval's registry as (verified against
cell_eval._pipeline._runner.VCC_METRICS):

    DES  ->  overlap_at_N               higher is better
    PDS  ->  discrimination_score_l1    higher is better
    MAE  ->  mae                        lower is better

cell-eval's own `vcc` profile is exactly those three; its `minimal` profile is
those three plus pearson_delta, mse, precision_at_N and de_nsig_counts -- seven,
the set the 2025 Generalist prize averaged ranks over.

Config merge order:
    challenge config (year-specific: metrics, control label, pert column)
        overrides
    eval config (year-agnostic: holdout policy, guardrails, output)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import yaml

from sidechain.utils.paths import resolve_config

logger = logging.getLogger(__name__)

# Challenge metric name -> cell-eval registry name.
CHALLENGE_TO_CELL_EVAL: dict[str, str] = {
    "des": "overlap_at_N",
    "pds": "discrimination_score_l1",
    "mae": "mae",
}

# cell-eval registry name -> True if higher is better.
HIGHER_IS_BETTER: dict[str, bool] = {
    "overlap_at_N": True,
    "discrimination_score_l1": True,
    "precision_at_N": True,
    "pearson_delta": True,
    "mae": False,
    "mse": False,
}

# Profile that yields exactly the requested metric set.
_PROFILE_FOR = {"challenge_only": "vcc", "with_extras": "minimal"}


def load_configs(
    eval_config: str | Path = "configs/eval.yaml",
    challenge_config: str | Path | None = None,
) -> dict[str, Any]:
    """Read and merge the two configs. Challenge values win where they overlap."""
    merged: dict[str, Any] = yaml.safe_load(resolve_config(eval_config).read_text()) or {}
    merged["_challenge"] = {}
    if challenge_config is not None:
        challenge = yaml.safe_load(resolve_config(challenge_config).read_text()) or {}
        merged["_challenge"] = challenge
    return merged


def resolve_metrics(cfg: dict) -> tuple[list[str], str]:
    """Return (cell-eval metric names, profile) implied by the merged config."""
    challenge = cfg.get("_challenge", {})
    metrics = challenge.get("metrics") or cfg.get("cell_eval", {}).get(
        "metrics", ["des", "pds", "mae"]
    )
    extras = challenge.get(
        "extra_state_metrics",
        cfg.get("cell_eval", {}).get("extra_state_metrics", False),
    )

    unknown = [m for m in metrics if m.lower() not in CHALLENGE_TO_CELL_EVAL]
    if unknown:
        raise ValueError(
            f"Unknown challenge metrics {unknown}. Known: {sorted(CHALLENGE_TO_CELL_EVAL)}"
        )

    profile = _PROFILE_FOR["with_extras" if extras else "challenge_only"]
    names = [CHALLENGE_TO_CELL_EVAL[m.lower()] for m in metrics]
    return names, profile


# ------------------------------------------------------------- guardrails --


def variance_inflation_check(
    pred: ad.AnnData, real: ad.AnnData, pert_col: str, control_pert: str, tol: float = 1.5
) -> dict[str, Any]:
    """Flag predictions whose spread is inflated relative to the truth.

    PDS rewards making a perturbation distinguishable from the others. Scaling
    predicted deltas up is a cheap way to buy separation without predicting
    biology better, and it shows up as predicted per-gene variance well above the
    real variance. `tol` is the ratio above which we call it.
    """
    import scipy.sparse as sp

    def _gene_var(a: ad.AnnData) -> np.ndarray:
        mask = a.obs[pert_col].astype(str).to_numpy() != control_pert
        X = a.X[mask]
        if not sp.issparse(X):
            return np.asarray(X).var(axis=0)
        # var = E[X^2] - E[X]^2, computed without densifying: a holdout of 40k
        # cells x 18k genes would be ~3 GB as a dense array.
        n = X.shape[0]
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).sum(axis=0)).ravel() / n
        return np.maximum(mean_sq - mean**2, 0.0)

    v_pred, v_real = _gene_var(pred), _gene_var(real)
    denom = float(v_real.mean())
    ratio = float(v_pred.mean() / denom) if denom > 0 else float("inf")
    return {
        "variance_ratio": ratio,
        "tolerance": tol,
        "flagged": bool(ratio > tol),
        "note": (
            "Predicted per-gene variance exceeds the real data by more than the "
            "tolerance -- PDS gain may be inflation rather than skill."
            if ratio > tol
            else "Predicted spread is consistent with the real data."
        ),
    }


# ------------------------------------------------------------------ score --


def score(
    pred: str | Path | ad.AnnData,
    real: str | Path | ad.AnnData,
    *,
    eval_config: str | Path = "configs/eval.yaml",
    challenge_config: str | Path | None = None,
    outdir: str | Path = "~/data/sidechain/runs/cell-eval",
    baseline_agg_results: str | Path | None = None,
    allow_discrete: bool = False,
    skip_guardrails: bool = False,
    de_real: str | Path | None = None,
    log_run: bool = True,
) -> dict[str, Any]:
    """Run cell-eval on a prediction. Returns metrics, guardrails and raw frames.

    `pred` and `real` are AnnData (or paths). They must share gene order and the
    same set of perturbation labels, including the control.

    `de_real` is a path to a previously computed `real_de.csv`. Differential
    expression on the truth depends only on `real`, so when scoring several models
    against the same holdout it is the same computation every time -- and on the
    full 2025 file it dominates the run. Pass it to skip the recompute. The result
    dict always carries `de_real_path` so the next call can reuse it.

    `log_run=True` records the scored run in lamindb via `utils.logging.log_run`
    (ADR 0001) -- non-fatal by that function's contract, and skipped entirely
    (no lamindb import) when False.
    """
    from cell_eval import MetricsEvaluator

    cfg = load_configs(eval_config, challenge_config)
    challenge = cfg.get("_challenge", {})
    pert_col = challenge.get("pert_col", "target_gene")
    control_pert = challenge.get("control_label", "non-targeting")
    metric_names, profile = resolve_metrics(cfg)

    pred_ad = pred if isinstance(pred, ad.AnnData) else ad.read_h5ad(Path(pred).expanduser())
    real_ad = real if isinstance(real, ad.AnnData) else ad.read_h5ad(Path(real).expanduser())

    outdir = Path(outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Scoring %d pred cells vs %d real cells | profile=%s | pert_col=%s | control=%s",
        pred_ad.n_obs,
        real_ad.n_obs,
        profile,
        pert_col,
        control_pert,
    )

    if de_real is not None:
        de_real = Path(de_real).expanduser()
        if not de_real.exists():
            raise FileNotFoundError(f"de_real not found: {de_real}")
        logger.info("Reusing precomputed real-data DE: %s", de_real)

    evaluator = MetricsEvaluator(
        adata_pred=pred_ad,
        adata_real=real_ad,
        control_pert=control_pert,
        pert_col=pert_col,
        outdir=str(outdir),
        allow_discrete=allow_discrete,
        de_real=str(de_real) if de_real is not None else None,
    )
    results, agg = evaluator.compute(profile=profile, write_csv=True)

    # agg is a polars frame in pl.describe() shape: a 'statistic' column plus one
    # column per metric. The per-metric mean is the headline number.
    metrics = _extract_means(agg)

    out: dict[str, Any] = {
        "metrics": metrics,
        "challenge_metrics": {
            k: metrics.get(v) for k, v in CHALLENGE_TO_CELL_EVAL.items() if v in metrics
        },
        "profile": profile,
        "requested": metric_names,
        "outdir": str(outdir),
        "results": results,
        "agg_results": agg,
        # Present whether we computed it or reused it, so a caller scoring several
        # models against one holdout can thread it through without special-casing.
        "de_real_path": str(de_real) if de_real is not None else str(outdir / "real_de.csv"),
    }

    if not skip_guardrails and cfg.get("guardrails", {}).get("variance_inflation_check", True):
        out["guardrails"] = {
            "variance_inflation": variance_inflation_check(
                pred_ad,
                real_ad,
                pert_col,
                control_pert,
                tol=float(
                    cfg.get("guardrails", {}).get("variance_inflation_tolerance", 1.5)
                ),
            )
        }

    baseline = baseline_agg_results or challenge.get("baseline_agg_results")
    if baseline:
        from cell_eval import score_agg_metrics

        out["vs_baseline"] = score_agg_metrics(agg, str(baseline))

    if log_run:
        from sidechain.utils.logging import log_run as _log_run

        # The registered artifact is this one small file, NOT `outdir`. A mirror outdir
        # is ~21 GB per line (predictions plus cell-eval's own CSVs), all of it
        # reproducible; uploading it as a side effect of scoring would put tens of GB
        # in the managed bucket per run. Same shape as `eval.loco`, which has written a
        # summary.json and logged that since it was built. ADR 0007 §1.
        summary = outdir / "summary.json"
        summary.write_text(json.dumps(
            {"entry": "local_mirror.score", "profile": profile, "pert_col": pert_col,
             "control": control_pert, "metrics": metrics,
             "challenge_metrics": out["challenge_metrics"],
             "guardrails": out.get("guardrails"), "vs_baseline": out.get("vs_baseline")},
            indent=1, default=str) + "\n")
        _log_run(
            {"entry": "local_mirror.score", "profile": profile, "requested": metric_names,
             "pert_col": pert_col, "control": control_pert, "outdir": str(outdir)},
            metrics,
            artifacts=[str(summary)],
        )

    return out


def _extract_means(agg) -> dict[str, float]:
    """Pull the per-metric mean out of cell-eval's aggregated (describe) frame."""
    try:
        row = agg.filter(agg["statistic"] == "mean")
        if row.height == 0:
            return {}
        rec = row.to_dicts()[0]
        return {
            k: float(v)
            for k, v in rec.items()
            if k != "statistic" and isinstance(v, (int, float)) and v is not None
        }
    except Exception as exc:  # noqa: BLE001 - shape varies across cell-eval versions
        logger.warning("Could not extract means from agg frame: %s", exc)
        return {}


def summarize(result: dict[str, Any]) -> str:
    """One-screen human summary of a score() result."""
    lines = ["Sidechain local mirror -- cell-eval", f"  profile: {result['profile']}"]
    lines.append("  challenge metrics:")
    for name, val in result.get("challenge_metrics", {}).items():
        if val is None:
            continue
        arrow = "^" if HIGHER_IS_BETTER.get(CHALLENGE_TO_CELL_EVAL[name], True) else "v"
        lines.append(f"    {name.upper():4s} {arrow} {val:.5f}")
    extras = {
        k: v
        for k, v in result.get("metrics", {}).items()
        if k not in CHALLENGE_TO_CELL_EVAL.values()
    }
    if extras:
        lines.append("  extra metrics:")
        for k, v in sorted(extras.items()):
            lines.append(f"    {k:26s} {v:.5f}")
    g = result.get("guardrails", {}).get("variance_inflation")
    if g:
        flag = "FLAGGED" if g["flagged"] else "ok"
        lines.append(f"  guardrail variance-inflation: {flag} (ratio {g['variance_ratio']:.2f})")
    return "\n".join(lines)


def _cli(
    pred: str,
    real: str,
    eval_config: str = "configs/eval.yaml",
    challenge_config: str = "challenges/vcc2025/config.yaml",
    outdir: str = "~/data/sidechain/runs/cell-eval",
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    res = score(
        pred,
        real,
        eval_config=eval_config,
        challenge_config=challenge_config,
        outdir=outdir,
    )
    print(summarize(res))


if __name__ == "__main__":  # `python -m sidechain.eval.local_mirror --help`
    import typer

    typer.run(_cli)
