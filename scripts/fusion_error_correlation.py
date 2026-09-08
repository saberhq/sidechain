"""Are two scored arms independent enough to be worth fusing?

A second arm pays as a *voter* rather than as a replacement only if its per-target errors are
unlike the incumbent's. Where they are alike, ensemble theory puts the gain from combining at
roughly nothing, and the honest move is to keep the better single arm and stop. The bar for
"alike" belongs to the project, not to this script: `--gate` carries it and defaults to a
correlation of 0.5.

This reads two cell-eval2 `run/results.csv` files -- the per-target long table, one row per
(perturbation, metric) -- and reports the correlation on the shared targets, plus the two things
that decide what a positive answer would be *worth*:

**The oracle gate.** The mean score a perfect per-target chooser would reach by taking whichever
arm is better on that target. It is the ceiling of fusion-by-selection: no realizable gate beats
it, so if the oracle barely clears the better arm, no gate is worth building.

**A difficulty control, and it is the point of the `--control-arm` flag.** Two arms scored on the
same fold share a large nuisance term: some targets have a distinctive true profile and every
method retrieves them, some do not and nothing does. That alone produces a positive correlation
between arms that share no information at all -- so a low correlation is also what pure noise
looks like, and the gate above cannot be read on its own. Pass a third arm you have independent
reason to treat as uninformative (an ablation whose per-target ranks do not separate from uniform
is the usual one, and a failed ablation of the candidate is ideal, being wrong in the same family
of ways). Its correlation with the incumbent estimates the shared-difficulty floor, and the
candidate's excess over that floor is what the candidate's own behaviour contributes.

    uv run python scripts/fusion_error_correlation.py \
        --arm-a runs/mirror/<fold>/<incumbent>/run/results.csv \
        --arm-b runs/mirror/<fold>/<candidate>/run/results.csv \
        --control-arm runs/mirror/<fold>/<candidate_ablation>/run/results.csv \
        --names incumbent candidate ablation --out runs/fusion/<name>

Every metric in both files is reported; `--metric` picks the headline one for the gate line.
Metrics whose better direction is *down* (`nmae`, `mse`, distance) are flagged, and the oracle
takes the min rather than the max for those -- reading an oracle the wrong way round would make
a worse arm look like a free gain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# cell-eval2 members where a LOWER value is better. Everything else scores upward.
LOWER_IS_BETTER = (
    "de_wilcoxon_lfc_nmae",
    "expr_mse_unbiased",
    "expr_mse_unbiased_capped",
    "expr_mse_unbiased_capped_norm",
    "expr_distance_unbiased",
)


def per_target(path: Path, metric: str) -> pd.Series:
    """One metric's per-target column from a cell-eval2 `results.csv`."""
    df = pd.read_csv(path)
    need = {"perturbation", "metric", "value"}
    if not need <= set(df.columns):
        raise SystemExit(f"{path}: expected columns {sorted(need)}, got {list(df.columns)}")
    sub = df[df["metric"] == metric]
    if sub.empty:
        raise SystemExit(f"{path}: no rows for metric {metric!r}; have "
                         f"{sorted(df['metric'].unique())}")
    return sub.set_index("perturbation")["value"].astype(float)


def compare(a: pd.Series, b: pd.Series, metric: str, *, seed: int = 0,
            n_splits: int = 200) -> dict:
    """Correlation, oracle ceiling and split-half spread for one metric on shared targets."""
    joined = pd.concat({"a": a, "b": b}, axis=1).dropna()
    x, y = joined["a"].to_numpy(), joined["b"].to_numpy()
    lower = metric in LOWER_IS_BETTER
    oracle = np.minimum(x, y) if lower else np.maximum(x, y)
    picked_b = (y < x) if lower else (y > x)
    rng = np.random.default_rng(seed)
    halves = []
    for _ in range(n_splits):
        idx = rng.permutation(len(x))[: len(x) // 2]
        if len(idx) > 2:
            halves.append(float(stats.spearmanr(x[idx], y[idx]).statistic))
    return {
        "metric": metric,
        "lower_is_better": lower,
        "n_shared_targets": len(joined),
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "pearson": float(stats.pearsonr(x, y).statistic),
        "spearman": float(stats.spearmanr(x, y).statistic),
        "spearman_split_half_sd": float(np.std(halves)) if halves else None,
        "oracle_mean": float(oracle.mean()),
        "oracle_gain_over_best_single": float(oracle.mean() - (min(x.mean(), y.mean())
                                                              if lower else max(x.mean(), y.mean()))),
        "targets_where_b_wins": int(picked_b.sum()),
        "targets_where_b_wins_frac": float(picked_b.mean()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-a", required=True, type=Path, help="incumbent arm's run/results.csv")
    ap.add_argument("--arm-b", required=True, type=Path, help="candidate arm's run/results.csv")
    ap.add_argument("--control-arm", type=Path,
                    help="an uninformative arm's results.csv: its correlation with A is the "
                         "shared-difficulty floor the candidate has to clear")
    ap.add_argument("--names", nargs="*", default=[],
                    help="labels for arm A, arm B and the control, in that order")
    ap.add_argument("--metric", default="pds_cosine", help="the metric the gate line reads")
    ap.add_argument("--gate", type=float, default=0.5,
                    help="kill bar on the error correlation: above it, keep the better single "
                         "arm and stop. The default is the project's pre-registered value.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="directory for report.json and per_target.csv")
    args = ap.parse_args(argv)

    names = list(args.names) + ["arm_a", "arm_b", "control"][len(args.names):]
    metrics = sorted(set(pd.read_csv(args.arm_a)["metric"]) & set(pd.read_csv(args.arm_b)["metric"]))
    rows = [compare(per_target(args.arm_a, m), per_target(args.arm_b, m), m, seed=args.seed)
            for m in metrics]
    control = None
    if args.control_arm:
        control = [compare(per_target(args.arm_a, m), per_target(args.control_arm, m), m,
                           seed=args.seed) for m in metrics]

    head = next(r for r in rows if r["metric"] == args.metric)
    floor = next((r for r in (control or []) if r["metric"] == args.metric), None)
    report = {
        "arm_a": {"name": names[0], "results_csv": str(args.arm_a)},
        "arm_b": {"name": names[1], "results_csv": str(args.arm_b)},
        "control_arm": ({"name": names[2], "results_csv": str(args.control_arm)}
                        if args.control_arm else None),
        "headline_metric": args.metric,
        "gate": args.gate,
        "verdict": ("independent enough to fuse" if head["spearman"] <= args.gate
                    else "correlated: pick the better single arm"),
        "per_metric": rows,
        "control_per_metric": control,
        "excess_over_difficulty_floor": (None if floor is None
                                         else head["spearman"] - floor["spearman"]),
    }

    print(f"{'metric':44s} {'rho':>7s} {'r':>7s} {'mean ' + names[0]:>16s} "
          f"{'mean ' + names[1]:>16s} {'oracle':>8s} {'B wins':>7s}")
    for r in rows:
        print(f"{r['metric']:44s} {r['spearman']:7.3f} {r['pearson']:7.3f} "
              f"{r['mean_a']:16.4f} {r['mean_b']:16.4f} {r['oracle_mean']:8.4f} "
              f"{r['targets_where_b_wins_frac']:6.1%}")
    print()
    print(f"{args.metric}: spearman {head['spearman']:.3f} "
          f"(+/- {head['spearman_split_half_sd']:.3f} split-half) against a gate of {args.gate} "
          f"-> {report['verdict']}")
    if floor is not None:
        print(f"  shared-difficulty floor ({names[0]} vs {names[2]}): {floor['spearman']:.3f}; "
              f"excess {report['excess_over_difficulty_floor']:+.3f}")
    print(f"  oracle per-target pick: {head['oracle_mean']:.4f} "
          f"({head['oracle_gain_over_best_single']:+.4f} over the better single arm), "
          f"{names[1]} wins {head['targets_where_b_wins_frac']:.1%} of targets")

    if args.out:
        out = args.out.expanduser()
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
        per = pd.concat({names[0]: per_target(args.arm_a, args.metric),
                         names[1]: per_target(args.arm_b, args.metric)}, axis=1)
        if args.control_arm:
            per[names[2]] = per_target(args.control_arm, args.metric)
        per.to_csv(out / f"per_target_{args.metric}.csv")
        print(f"\nwrote {out}/report.json and per_target_{args.metric}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
