"""The 2026 local mirror: score a prediction for one held-out line with cell-eval2's `vcc2026` preset.

    uv run python -m sidechain.eval.mirror2026 bundle --real REAL.h5ad --out OUT --pert-col perturbation --control non-targeting
    uv run python -m sidechain.eval.mirror2026 score --real REAL.h5ad --pred PRED.h5ad --bundle OUT/bundle --out RUN

`bundle` runs cell-eval2's `baseline` and `prep-real-bundle` once per line; `score` runs `run` and `score` against it,
and is refused on a bundle from another config (DE backend: pdex on CPU, the Mac; gpudge on a GPU box; version;
device). The prediction carries the truth's labels, control included (`--attach-controls` appends the controls).
Reads the truth, the prediction and the bundle; writes baseline/, bundle/, run/, scored.csv, summary.json, logs/.
Notes: private/code-notes/src/sidechain/eval/mirror2026.py.md
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sidechain.data.stream_pseudobulk import read_elem
from sidechain.utils.h5ad_stream import (
    CsrWriter,
    load_rows_csr,
    open_anndata_h5,
    write_frame,
)
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


def attach_controls(pred: Path, real: Path, out: Path, *, pert_col: str, control: str,
                    block_rows: int = 2000) -> Path:
    """Append the real control cells to a controls-free prediction, as the platform does, one row block at a time.

    Writes the prediction rows (`pred_<i>`), then the control rows (`ctrl_<i>`), one obs column, the real gene axis.
    """
    with open_anndata_h5(pred, "r") as p, open_anndata_h5(real, "r") as r:
        p_var = read_elem(p["var"]).index.astype(str).to_numpy()
        r_var = read_elem(r["var"]).index.astype(str).to_numpy()
        if list(p_var) != list(r_var):
            raise ValueError("prediction and real gene axes differ (set or order)")
        p_labels = read_elem(p["obs"])[pert_col].astype(str).to_numpy()
        r_labels = read_elem(r["obs"])[pert_col].astype(str).to_numpy()
        ctrl_rows = np.where(r_labels == control)[0]
        out.parent.mkdir(parents=True, exist_ok=True)
        with open_anndata_h5(out, "w") as o:
            w = CsrWriter(o, len(r_var))
            for lo in range(0, len(p_labels), block_rows):
                w.append(*load_rows_csr(p, np.arange(lo, min(lo + block_rows, len(p_labels)))))
            for lo in range(0, len(ctrl_rows), block_rows):
                w.append(*load_rows_csr(r, ctrl_rows[lo:lo + block_rows]))
            n_rows = w.close()
            index = np.array([f"pred_{i}" for i in range(len(p_labels))]
                             + [f"ctrl_{i}" for i in range(len(ctrl_rows))], dtype=object)
            labels = np.concatenate([p_labels, np.full(len(ctrl_rows), control, dtype=object)])
            assert n_rows == len(index), f"{n_rows} rows written, {len(index)} labels"
            write_frame(o, "obs", index, {pert_col: labels.astype(object)})
            write_frame(o, "var", r_var.astype(object), {})
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
        # An --out leaf named like a model must spell the name right; other labels pass.
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
