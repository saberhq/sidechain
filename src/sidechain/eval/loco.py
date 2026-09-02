"""Leave-one-context-out: predict a held-out line's perturbations from the other
lines, then score on that line's own competition bundle (`mirror2026`).

This is the local stand-in for "an unseen cell line": the held-out line
contributes only its control cells (its `ContextProfile`), exactly as A/B/C do
in the challenge; every per-gene effect comes from the OTHER lines' pseudobulks.
It measures *method* transfer on whatever perturbations the lines share -- not
the challenge panel, which the essential-gene screens barely cover
(private reports/06 s5).

    uv run python -m sidechain.eval.loco \
        --real ~/data/sidechain/cache/vcc2026/hepg2_flowtest_real.h5ad \
        --pert-col perturbation --control non-targeting \
        --source ~/data/sidechain/cache/vcc2026/k562_essential_all_pseudobulk.npz:control \
        --source ~/data/sidechain/cache/vcc2026/rpe1_all_pseudobulk.npz:control \
        --source ~/data/sidechain/cache/vcc2026/jurkat_all_pseudobulk.npz:control \
        --bundle ~/data/sidechain/runs/mirror/hepg2_flowtest_rule/bundle \
        --out ~/data/sidechain/runs/mirror/hepg2_flowtest_rule/transfer_even --dispersion even

THE TWO `control`S ON THAT COMMAND LINE ARE DIFFERENT LABELS, and this example used to get
one of them wrong. `--control` names the control arm inside the *truth* h5ad, and those are
harmonised to `non-targeting`. The `:control` suffix on each `--source` names the control
arm inside *that source's own* pseudobulk, and the Replogle-derived ones really do spell it
`control` (H1 spells it `non-targeting`) -- which is the entire reason the suffix is
per-source rather than one global flag. This example read `--control control` until
2026-08-25; three mirror bundles were built from it and record a control label their truth
file does not contain.

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
from sidechain.eval.mirror2026 import attach_controls, score
from sidechain.models.count_emitters import ContextProfile, PoissonEmitter
from sidechain.submit.build import (
    apply_transfer_floors,
    as_delta_source,
    parse_coverage_tiers,
    parse_transfer_floor,
    pooled_delta,
    sources_from_specs,
)
from sidechain.utils.logging import log_run
from sidechain.utils.naming import check_out_leaf


def build_transfer_prediction(
    real_path: Path,
    sources: list,
    out_path: Path,
    *,
    pert_col: str,
    control: str,
    dispersion: str | None = None,
    emit_lambda: float | None = None,
    shrinkage: bool = True,
    alpha: float = 1.0,
    gamma: float = 1.0,
    var_floor: str = "none",
    coverage_tiers: tuple[tuple[float, float], ...] | None = None,
    similarity_beta: float = 0.0,
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
    if dispersion is None and emit_lambda is None:
        dispersion = "even"    # this function's historical default
    em = PoissonEmitter(prof, seed=seed, dispersion=dispersion, lam=emit_lambda)
    gene_pos = {g: i for i, g in enumerate(axis)}
    blocks, obs_labels, covered = [], [], 0
    pool_stats: dict = {}
    # The transfer exponent reads the SAME control profile the emitter anchors on
    # (min_libsize-filtered, CPM within this file's own gene universe), so the
    # ratio and the replay are self-consistent by construction.
    #
    # BOTH knobs that need it build it here. An earlier version of this block guarded
    # `similarity_beta` BEFORE `ctrl_cpm` was ever assigned, so with the default gamma = 1 the
    # guard could never be satisfied and every similarity arm died in eight seconds. The guard
    # was right about the requirement and wrong about where the requirement is met.
    ctrl_cpm = None
    if gamma != 1.0 or similarity_beta != 0.0:
        if list(prof.genes) != list(axis):
            need = "gamma != 1" if gamma != 1.0 else "similarity_beta != 0"
            raise SystemExit(f"{need}: control profile genes differ from the real file's axis")
        ctrl_cpm = prof.fraction * 1e6
    for p in perts:
        d = pooled_delta(p, sources, axis, shrinkage=shrinkage, var_floor=var_floor,
                         gamma=gamma, ctrl_tgt_cpm=ctrl_cpm,
                         coverage_tiers=coverage_tiers,
                         similarity_beta=similarity_beta, stats=pool_stats)
        if d is not None:
            covered += 1
            d = d * alpha    # alpha scales the pooled vector; gamma acted per source inside the pool
            if p in gene_pos:
                d[gene_pos[p]] = -2.32
        n = cells_per_pert or int((labels == p).sum())
        blocks.append(em.emit(n, d))
        obs_labels += [p] * n
    X = sp.vstack(blocks, format="csr")
    pred = ad.AnnData(X=X, obs=pd.DataFrame({pert_col: obs_labels}, index=[f"pred_{i}" for i in range(len(obs_labels))]),
                      var=pd.DataFrame(index=real.var_names))
    pred.write_h5ad(out_path)
    # `shrinkage` alone under-describes a depth-aware run ('false' while one
    # arm was shrunk), so the per-source overrides are reported beside it,
    # aligned with the source list: None = followed the global flag.
    return {"pred": str(out_path), "perturbations": len(perts), "covered_by_sources": covered,
            "cells": int(pred.n_obs), "genes": int(pred.n_vars), "dispersion": em.dispersion,
            "emit_lambda": em.lam,
            "shrinkage": shrinkage,
            "shrink_overrides": [getattr(as_delta_source(s), "shrink", None) for s in sources],
            "alpha": alpha, "gamma": gamma, "var_floor": var_floor,
            "coverage_tiers": coverage_tiers,
            "similarity_beta": similarity_beta,
            # Recorded per source and by name, not as a bare list: a floor attached to the
            # wrong arm is the failure mode this knob has, so the run must say which arm got
            # which number rather than leaving it to the command line's order.
            "transfer_floor": {getattr(s[0] if isinstance(s, tuple) else s,
                                       "sidechain_name", f"src{i}"):
                               float(getattr(s[0] if isinstance(s, tuple) else s,
                                             "transfer_floor", 0.0) or 0.0)
                               for i, s in enumerate(sources)},
            "pool_stats": pool_stats}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, type=Path)
    ap.add_argument("--pert-col", default="target_gene")
    ap.add_argument("--control", default="non-targeting")
    ap.add_argument("--source", action="append", default=[],
                    help="pseudobulk .npz:control_label (repeatable)")
    ap.add_argument("--shrink-source", action="append", default=[], metavar="NPZ:CONTROL",
                    help="pseudobulk source whose transferred log2FCs are shrunk regardless "
                         "of --no-shrink (depth-aware shrinkage; same syntax and meaning as "
                         "sidechain.submit.build, so a scored arm submits verbatim)")
    ap.add_argument("--lfc-source", action="append", default=[], metavar="NPZ",
                    help="cached LfcTable .npz -- a source publishing the contrast already "
                         "taken rather than cells (e.g. Feng 2026). Repeatable.")
    ap.add_argument("--coverage-tiers", metavar="CUT:FACTOR,...",
                    help="weight each source's per-gene vote by how many cells' worth of "
                         "evidence stands behind that gene (n_eff), as cut:factor pairs, "
                         "e.g. '3:0.10,10:0.50'. Same knob in sidechain.submit.build, so a "
                         "scored arm submits verbatim.")
    ap.add_argument("--transfer-floor", action="append", default=[], metavar="NAME=TAU2",
                    help="per-source transfer-error floor tau^2 added to that source's "
                         "variance before it becomes a pooling weight, keyed by the source "
                         "file's basename stem (repeatable), e.g. 'h1_pseudobulk=0.0104'. "
                         "Same knob in sidechain.submit.build, so a scored arm submits "
                         "verbatim.")
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dispersion", choices=["poisson", "even"], default=None,
                    help="endpoint of the emission dial (default: even); exclusive with --emit-lambda")
    ap.add_argument("--emit-lambda", type=float, default=None, metavar="LAM",
                    help="emission-sharpening dial in [0, 1]: 0 = even cells, 1 = poisson cells, "
                         "interior values narrow the emitted cloud toward the mean (exact "
                         "variance law: count_emitters.PoissonEmitter). Same knob in "
                         "sidechain.submit.build, so a scored arm submits verbatim.")
    ap.add_argument("--no-shrink", action="store_true")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--similarity-beta", type=float, default=0.0,
                    help="exponent on each source's control-profile cosine to the held-out "
                         "context, applied to its pooling weight (submit.build."
                         "control_similarity). 0 is uniform pooling and bit-identical to the "
                         "historical call; cosines run ~0.9-0.99 so the exponent has to be "
                         "large to separate sources. Needs a control profile, like --gamma.")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="transfer exponent on the target/source control-CPM ratio: 1 = the "
                         "fold change transfers (today's emitter, bit-identical), 0 = the "
                         "absolute CPM change transfers (submit.build.gamma_transfer; "
                         "research/ideas/effect-size-from-control-features.md). NOT wired on "
                         "sidechain.submit.build yet: shifts there are pooled once for all "
                         "contexts and gamma makes them context-specific, so a gamma arm "
                         "cannot submit verbatim until that restructure lands")
    ap.add_argument("--var-floor", choices=["none", "poisson"], default="none",
                    help="floor each pseudobulk arm's variance at its Poisson sampling variance "
                         "(same knob as sidechain.submit.build, so a scored arm submits verbatim)")
    ap.add_argument("--cells-per-pert", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--de-backend", default="pdex")
    args = ap.parse_args(argv)
    cov_tiers = parse_coverage_tiers(args.coverage_tiers)
    if args.emit_lambda is not None and args.dispersion is not None:
        ap.error("--dispersion and --emit-lambda are one dial (even is 0, poisson is 1) -- pass one")
    if args.emit_lambda is not None and not 0.0 <= args.emit_lambda <= 1.0:
        # Same check as the emitter's, but before any work: the constructor
        # would only catch it after the prediction stage has started.
        ap.error(f"--emit-lambda must be in [0, 1], got {args.emit_lambda}")
    if args.emit_lambda is None and args.dispersion is None:
        args.dispersion = "even"    # the historical default of this entry point
    # Same rule as mirror2026.score: an arm named like a model must spell it right;
    # freeform ablation labels pass untouched.
    check_out_leaf(args.out.expanduser().name, context="loco")

    # `--source` is no longer required on its own: an LfcTable is a complete
    # source, so an arm built only from published contrasts is a legitimate run
    # and scoring one is how you find out what that corpus is worth alone.
    # At least one of the two is still mandatory -- an arm with no sources at
    # all would score the fallback shift and look like a model.
    if not args.source and not args.shrink_source and not args.lfc_source:
        ap.error("need at least one --source, --shrink-source or --lfc-source")
    sources = sources_from_specs(args.source, args.shrink_source)
    for path in args.lfc_source:
        tab = LfcTable.load(path)
        tab.sidechain_name = Path(path).expanduser().stem
        sources.append(tab)
    sources = apply_transfer_floors(sources, parse_transfer_floor(args.transfer_floor))
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    info = build_transfer_prediction(args.real, sources, out / "pred.h5ad", pert_col=args.pert_col,
                                     control=args.control, dispersion=args.dispersion,
                                     emit_lambda=args.emit_lambda,
                                     shrinkage=not args.no_shrink, alpha=args.alpha,
                                     gamma=args.gamma, var_floor=args.var_floor,
                                     coverage_tiers=cov_tiers,
                                     similarity_beta=args.similarity_beta,
                                     cells_per_pert=args.cells_per_pert, seed=args.seed)
    print(json.dumps(info), flush=True)
    with_ctrl = attach_controls(out / "pred.h5ad", args.real, out / "pred_with_controls.h5ad",
                                pert_col=args.pert_col, control=args.control)
    res = score(with_ctrl, args.real, args.bundle, out, pert_col=args.pert_col, control=args.control,
                de_backend=args.de_backend)
    res["build"] = info
    # The source list used to live only in the command line -- the 2026-08-26
    # session had to re-run three arms just to prove which sources produced them.
    # `shrink_pseudobulk` is listed separately: which arms were shrunk is part
    # of what produced the run.
    res["sources"] = {"pseudobulk": args.source, "shrink_pseudobulk": args.shrink_source,
                      "lfc": args.lfc_source}
    (out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    log_run(
        {"entry": "loco", "real": str(args.real), "bundle": str(args.bundle),
         "out": str(out), "sources": args.source, "shrink_sources": args.shrink_source,
         "lfc_sources": args.lfc_source,
         "dispersion": args.dispersion, "emit_lambda": args.emit_lambda,
         "shrinkage": not args.no_shrink,
         "alpha": args.alpha, "gamma": args.gamma, "var_floor": args.var_floor,
         "similarity_beta": args.similarity_beta,
         "coverage_tiers": args.coverage_tiers,
         "seed": args.seed, "de_backend": args.de_backend},
        {"overall": res.get("overall"), "members": res.get("members")},
        artifacts=[str(out / "summary.json")],
    )
    print(json.dumps({k: v for k, v in res.items() if k in ("members", "overall")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
