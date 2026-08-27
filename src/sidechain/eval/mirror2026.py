"""The 2026 local mirror: score a prediction for a held-out line exactly as Arc does.

Arc's score is cell-eval2's `vcc2026` rule: six metrics, each placed on a scale whose
0 is the held-out line's *generic response* (its mean perturbation effect, emitted
for every perturbation) and whose 1 is a five-split replicate of the line with
itself. cell-eval2 ships that whole construction as a "real bundle", so the mirror
is four of its own commands, run in order, on one machine and one config:

    baseline          -> the 0-end arm (a prediction file), from the real data alone
    prep-real-bundle  -> the bundle: baseline leg + replicate anchor + manifest
    run               -> the six raw metrics for OUR prediction vs the real data
    score             -> (u - b) / (r - b) per member, averaged; refused if the run's
                         identity (version, config, device, DE backend, ...) differs
                         from the bundle's

The bundle depends only on the real side, so it is built once per held-out line
and reused for every model (`bundle_dir`). Anything that changes the config --
DE backend (pdex on CPU, gpudge on the box), cell-eval2 version -- needs a new
bundle; the `score` step refuses a mismatch rather than mis-scoring.

    uv run python -m sidechain.eval.mirror2026 bundle --real real.h5ad --out ~/data/sidechain/runs/mirror/hepg2 \
        --pert-col perturbation --control control
    uv run python -m sidechain.eval.mirror2026 score --real real.h5ad --pred pred.h5ad \
        --bundle ~/data/sidechain/runs/mirror/hepg2/bundle --out ~/data/sidechain/runs/mirror/hepg2/<model>

The prediction must carry the real file's perturbation labels, the control label
included (the scorer needs the control cells on both sides; Arc's platform
concatenates the held-out controls onto a submission, and `attach_controls`
does the same here).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

from sidechain.utils.naming import check_out_leaf


def _ce2() -> list[str]:
    """The cell-eval2 console script next to this interpreter (the package has no __main__)."""
    exe = Path(sys.executable).parent / "cell-eval2"
    found = str(exe) if exe.exists() else shutil.which("cell-eval2")
    if not found:
        raise RuntimeError("cell-eval2 console script not found; is cell-eval2 installed in this env?")
    return [found]


CE2 = _ce2()
MEMBERS = ("pds_cosine", "expr_mse_unbiased_capped_norm", "de_wilcoxon_lfc_nmae",
           "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw",
           "de_wilcoxon_sig_jaccard")


def _common(pert_col: str, control: str, de_backend: str) -> list[str]:
    return ["--preset", "vcc2026", "--pert-col", pert_col, "--control", control,
            "--input-type", "counts", "--set", f"de.backend={de_backend}"]


def _run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        fh.write(" ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, check=False)
    if proc.returncode != 0:
        tail = log.read_text().splitlines()[-25:]
        raise RuntimeError(f"{cmd[3] if len(cmd) > 3 else cmd} failed ({proc.returncode}); tail of {log}:\n" + "\n".join(tail))


def build_bundle(real: Path, out: Path, *, pert_col: str, control: str, de_backend: str = "pdex",
                 bundle_id: str | None = None, n_splits: int = 5, force: bool = False) -> Path:
    """The 0-end arm and the bundle for one real file. Returns the bundle directory."""
    out = out.expanduser()
    bl, bundle = out / "baseline", out / "bundle"
    common = _common(pert_col, control, de_backend)
    if force and bl.exists():
        shutil.rmtree(bl)
    if not (bl / "baseline_pred.h5ad").exists():
        _run(CE2 + ["baseline", "-ar", str(real), "-o", str(bl), "--save-pred", str(bl / "baseline_pred.h5ad"),
                    "--seed", "0", *common], out / "logs" / "baseline.log")
    if force and bundle.exists():
        shutil.rmtree(bundle)
    if not (bundle / "manifest.json").exists():
        _run(CE2 + ["prep-real-bundle", "--real", str(real), "--baseline", str(bl / "baseline_pred.h5ad"),
                    "-o", str(bundle), "--id", bundle_id or out.name, "--anchor-splits", str(n_splits), *common],
             out / "logs" / "bundle.log")
    return bundle


def attach_controls(pred: Path, real: Path, out: Path, *, pert_col: str, control: str) -> Path:
    """Append the real control cells to a controls-free prediction, as the platform does."""
    p = ad.read_h5ad(pred)
    r = ad.read_h5ad(real)
    ctrl = r[r.obs[pert_col].astype(str) == control].copy()
    if list(p.var_names) != list(r.var_names):
        raise ValueError("prediction and real gene axes differ (set or order)")
    ctrl.obs = pd.DataFrame({pert_col: [control] * ctrl.n_obs}, index=[f"ctrl_{i}" for i in range(ctrl.n_obs)])
    p.obs = pd.DataFrame({pert_col: p.obs[pert_col].astype(str).to_numpy()}, index=[f"pred_{i}" for i in range(p.n_obs)])
    merged = ad.concat([p, ctrl], join="inner", index_unique=None)
    merged.var = pd.DataFrame(index=r.var_names)
    merged.write_h5ad(out)
    return out


def score(pred: Path, real: Path, bundle: Path, out: Path, *, pert_col: str, control: str,
          de_backend: str = "pdex") -> dict:
    """Raw metrics + bundle-scaled score for one prediction. Returns the scaled members."""
    out = out.expanduser()
    run = out / "run"
    _run(CE2 + ["run", "-ap", str(pred), "-ar", str(real), "-o", str(run), *_common(pert_col, control, de_backend)],
         out / "logs" / "run.log")
    scored = out / "scored.csv"
    _run(CE2 + ["score", "--user-agg", str(run / "agg_results.csv"), "--real-bundle", str(bundle), "-o", str(scored)],
         out / "logs" / "score.log")
    df = pd.read_csv(scored)
    result = {"scored_csv": str(scored), "columns": list(df.columns)}
    # The scaled column is `from_replicate`; keep the six members and the average.
    col = "from_replicate" if "from_replicate" in df.columns else None
    key = "metric" if "metric" in df.columns else df.columns[0]
    if col:
        members = {m: float(v) for m, v in zip(df[key], df[col]) if m in MEMBERS}
        result["members"] = members
        avg_rows = df[df[key].astype(str).str.contains("avg_score")]
        result["overall"] = float(avg_rows[col].iloc[0]) if len(avg_rows) else (
            sum(members.values()) / len(members) if members else None)
    (out / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle", help="build the baseline arm and the real bundle for one held-out line")
    b.add_argument("--real", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--pert-col", default="target_gene")
    b.add_argument("--control", default="non-targeting")
    b.add_argument("--de-backend", default="pdex")
    b.add_argument("--splits", type=int, default=5)
    b.add_argument("--force", action="store_true")
    s = sub.add_parser("score", help="score a prediction against a built bundle")
    s.add_argument("--real", required=True, type=Path)
    s.add_argument("--pred", required=True, type=Path)
    s.add_argument("--bundle", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--pert-col", default="target_gene")
    s.add_argument("--control", default="non-targeting")
    s.add_argument("--de-backend", default="pdex")
    s.add_argument("--attach-controls", action="store_true", help="append the real controls to the prediction first")
    args = ap.parse_args(argv)
    if args.cmd == "bundle":
        bundle = build_bundle(args.real, args.out, pert_col=args.pert_col, control=args.control,
                              de_backend=args.de_backend, n_splits=args.splits, force=args.force)
        print(json.dumps({"bundle": str(bundle), "manifest": json.loads((bundle / "manifest.json").read_text())}, indent=1, default=str))
    else:
        # A run named like a model must spell the name right: the run directory is the
        # model's identity in RESULTS.md, and a mirror-scored name is as permanent as a
        # board one. Freeform arm labels (h1_xatlas, baseline) pass untouched.
        check_out_leaf(args.out.expanduser().name, context="mirror2026.score")
        pred = args.pred
        if args.attach_controls:
            args.out.expanduser().mkdir(parents=True, exist_ok=True)
            pred = attach_controls(args.pred, args.real, args.out.expanduser() / "pred_with_controls.h5ad",
                                   pert_col=args.pert_col, control=args.control)
        res = score(pred, args.real, args.bundle, args.out, pert_col=args.pert_col, control=args.control,
                    de_backend=args.de_backend)
        print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
