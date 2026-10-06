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
from types import SimpleNamespace

import anndata as ad
import numpy as np

from sidechain.data.lfc_table import LfcTable
from sidechain.eval.mirror2026 import attach_controls, score
from sidechain.models import adaptive_shrink
from sidechain.models.basal_slope import MODES as BASAL_MODES, fit_basal_slopes, target_basal
from sidechain.models.count_emitters import CONTROL_MIN_LIBSIZE, ContextProfile, PoissonEmitter
from sidechain.submit.build import (
    add_neighbour_args,
    add_shrink_args,
    apply_transfer_floors,
    as_delta_source,
    check_neighbour_args,
    check_shrink_args,
    check_shrink_rule,
    neighbour_arm_for,
    parse_coverage_tiers,
    parse_transfer_floor,
    pooled_delta,
    shrink_kwargs,
    sources_from_specs,
)
from sidechain.utils.h5ad_stream import CsrWriter, open_anndata_h5, write_frame
from sidechain.utils.logging import log_run
from sidechain.utils.naming import check_out_leaf


def read_scatter_table(path: Path, axis: np.ndarray, perts: list[str]) -> tuple[dict, dict]:
    """A per-(perturbation, gene) scatter table for `PoissonEmitter.emit_dual(scatter=)` (T85).

    A parquet with `target`, `feature` and `scatter` >= 0: 1 leaves the gene as the emitter's
    lambda emits it, 0 removes its sampling scatter, above 1 widens it. Rows whose target is not
    a perturbation of this file, whose gene is not on its axis, or whose gene IS the target (its
    pin is not the table's to move) are dropped and counted; a pair listed twice is refused.
    Returns `{target: (gene positions, values)}` for the pairs that change something (scatter
    other than 1) and the record that goes into the run's summary.
    """
    import hashlib

    import pandas as pd

    path = Path(path).expanduser()
    tab = pd.read_parquet(path, columns=["target", "feature", "scatter"])
    tab["target"], tab["feature"] = tab["target"].astype(str), tab["feature"].astype(str)
    val = tab["scatter"].to_numpy(dtype=np.float64)
    if not np.isfinite(val).all() or (val < 0).any():
        raise SystemExit(f"--scatter-table {path.name}: scatter must be finite and >= 0")
    if tab.duplicated(["target", "feature"]).any():
        raise SystemExit(f"--scatter-table {path.name}: a (target, feature) pair is listed twice")
    pos = {g: i for i, g in enumerate(axis)}
    known = set(perts)
    off_target = ~tab["target"].isin(known).to_numpy()
    off_axis = ~tab["feature"].isin(pos).to_numpy()
    own = (tab["target"] == tab["feature"]).to_numpy()
    keep = ~(off_target | off_axis | own) & (val != 1.0)
    out = {}
    for target, block in tab[keep].groupby("target", sort=False):
        out[target] = (np.array([pos[g] for g in block["feature"]], dtype=np.int64),
                       block["scatter"].to_numpy(dtype=np.float64))
    sizes = [len(v[0]) for v in out.values()]
    record = {"table": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "rows": len(tab), "pairs_applied": int(keep.sum()),
              "targets_with_a_pair": len(out), "pairs_per_target_median": float(np.median(sizes)) if sizes else 0.0,
              "rows_at_scatter_zero": int(((val == 0.0) & keep).sum()),
              "rows_below_one": int(((val < 1.0) & keep).sum()),
              "rows_above_one": int(((val > 1.0) & keep).sum()),
              "rows_dropped": {"target_not_in_this_file": int(off_target.sum()),
                               "gene_not_on_the_axis": int((off_axis & ~off_target).sum()),
                               "the_targets_own_gene": int((own & ~off_target & ~off_axis).sum())}}
    return out, record


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
    shrink_k: float = 1.0,
    shrink_stage: str = "source",
    shrink_rule: str = "garrote",
    alpha: float = 1.0,
    gamma: float = 1.0,
    var_floor: str = "none",
    coverage_tiers: tuple[tuple[float, float], ...] | None = None,
    similarity_beta: float = 0.0,
    basal_slope: str = "off",
    alpha_bulk: float | None = None,
    bulk_anchor: str = "mean_cpm",
    log_bias_correct: bool = False,
    cells_per_pert: int | None = None,
    seed: int = 0,
    min_libsize: float = CONTROL_MIN_LIBSIZE,
    neighbour_table: Path | list[Path] | None = None,
    neighbour_pool: Path | None = None,
    neighbour_k: int = 25,
    neighbour_w: float | list[float] | None = None,
    neighbour_size: str = "unit",
    neighbour_select: str = "table",
    neighbour_cand: int = 100,
    neighbour_picks: Path | None = None,
    dual_fallback: str = "template",
    delta_cache: dict | None = None,
    scatter_table: Path | None = None,
    emit_shape: str = "template",
) -> dict:
    """Predict every non-control perturbation of `real_path` from `sources`.

    `delta_cache` (T103 round three; `sidechain.eval.delta_cache`) is `{"dir": ..., "sources":
    [[stem, sha256, control], ...], "build_jobs": N or None}`: with `build_jobs` the pooled delta
    of every target and neighbour-pool label is written once and nothing is predicted; without,
    every pooled delta is read from that cache instead of pooled. None is the historical path.
    """
    if dual_fallback not in ("template", "anchor"):
        raise SystemExit(f"dual_fallback must be 'template' or 'anchor', got {dual_fallback!r}")
    if emit_shape not in ("template", "controls"):
        raise SystemExit(f"emit_shape must be 'template' or 'controls', got {emit_shape!r}")
    shaped = emit_shape == "controls"            # T85: every block is re-rated control cells
    check_shrink_rule(shrink_k, shrink_stage, shrink_rule)     # before any file is written
    # the neighbour pool's adaptive fits, counted apart from the targets'; None keeps the
    # pool's pooled_delta call exactly what it was for every other rule
    pool_fit_stats = {} if shrinkage and shrink_rule == "adaptive" else None
    # what a target whose two moments cannot both be met falls back to: the one-amplitude template
    # (every arm through 2026-10-01; it loses the pooled anchor too) or one amplitude with the
    # summed profile kept on the emitter's anchor (count_emitters.PoissonEmitter.emit_dual)
    on_fail = "fallback" if dual_fallback == "template" else "anchor"
    fell_back: dict[str, list] = {"anchor": [], "template": []}

    sharpened: dict[str, int] = {}               # target -> genes whose scatter the block carries
    in_shape: dict[str, bool] = {}               # target -> its block is control cells (not the template rung)

    def dual(n, d_cell, d_bulk, label, scatter=None):   # called only inside the write loop, after `em` exists
        block = em.emit_dual(n, d_cell, d_bulk, on_fail=on_fail, scatter=scatter, shape=shaped)
        how = getattr(em, "last_dual", "dual")
        if how in ("anchor", "template"):
            fell_back.setdefault(how, []).append([label, getattr(em, "last_dual_reason", None)])
        if scatter is not None:
            sharpened[label] = int(em.last_sharpened or 0)
        if shaped:
            in_shape[label] = bool(em.last_shaped)
        return block

    # Backed, and the control cells are the only rows brought into memory. The X-Atlas
    # folds are 726 M nonzeros: reading one whole would cost ~6 GB before a single cell
    # is emitted, and the emitted side costs as much again.
    real = ad.read_h5ad(real_path, backed="r")
    labels = real.obs[pert_col].astype(str).to_numpy()
    perts = sorted(set(labels) - {control})
    axis = real.var_names.astype(str).to_numpy()
    cache = None
    if delta_cache is not None:
        from sidechain.eval.delta_cache import DeltaCache, build_cache, key_fields
        from sidechain.models.neighbour_arm import read_pool

        plain = all(isinstance(s, tuple) and len(s) == 2
                    and not getattr(s[0], "transfer_floor", 0.0) for s in sources)
        if (gamma != 1.0 or similarity_beta != 0.0 or coverage_tiers is not None
                or basal_slope != "off" or not plain
                or len(sources) != len(delta_cache["sources"])):
            raise SystemExit("--delta-cache holds the plain pool only: gamma 1, no similarity "
                             "weight, no coverage tiers, no basal slope, --source arms alone")
        fields = key_fields(delta_cache["sources"], axis, {
            "shrinkage": shrinkage, "shrink_k": shrink_k, "shrink_stage": shrink_stage,
            "shrink_rule": shrink_rule, "var_floor": var_floor,
            "log_bias_correct": log_bias_correct})
        if delta_cache.get("build_jobs"):
            if real.isbacked:
                real.file.close()

            def pool_one(label, stats):
                return pooled_delta(label, sources, axis, shrinkage=shrinkage, shrink_k=shrink_k,
                                    shrink_stage=shrink_stage, shrink_rule=shrink_rule,
                                    var_floor=var_floor, log_bias_correct=log_bias_correct,
                                    gamma=gamma, ctrl_tgt_cpm=None, coverage_tiers=coverage_tiers,
                                    similarity_beta=similarity_beta, stats=stats)

            members = read_pool(neighbour_pool) if neighbour_pool is not None else []
            return {"delta_cache_built": build_cache(delta_cache["dir"], fields,
                                                     [*perts, *members], pool_one,
                                                     jobs=int(delta_cache["build_jobs"]))}
        cache = DeltaCache(delta_cache["dir"], fields)
    ctrl_tmp = out_path.parent / f"{out_path.stem}.controls.h5ad"
    ctrl_tmp.parent.mkdir(parents=True, exist_ok=True)
    real[labels == control].to_memory().write_h5ad(ctrl_tmp)
    if real.isbacked:
        real.file.close()
    prof = ContextProfile.from_controls(ctrl_tmp, real_path.stem, min_libsize=min_libsize,
                                        keep_cells=shaped)
    if dispersion is None and emit_lambda is None:
        dispersion = "even"    # this function's historical default
    em = PoissonEmitter(prof, seed=seed, dispersion=dispersion, lam=emit_lambda,
                        bulk_anchor=bulk_anchor)
    # T84 round 2: a pooled anchor is a two-channel emission even at one amplitude -- the
    # pseudobulk moves onto the depth-weighted control profile, the per-cell mean stays put.
    two_channel = alpha_bulk is not None or bulk_anchor != "mean_cpm"
    if dual_fallback != "template" and not two_channel:
        raise SystemExit("--dual-fallback only acts on a two-channel emission -- pass --alpha-bulk "
                         "or --bulk-anchor pooled")
    if two_channel and em.lam == 0.0:
        raise SystemExit("--alpha-bulk / --bulk-anchor pooled need a depth spread (--emit-lambda "
                         "> 0 or --dispersion poisson): at lambda 0 the pseudobulk and the "
                         "per-cell mean coincide")
    gene_pos = {g: i for i, g in enumerate(axis)}
    # T85: the per-gene scatter dial. It acts inside the two-moment fit, so it needs that path.
    scatter_of, scatter_record = None, None
    if scatter_table is not None:
        if not two_channel:
            raise SystemExit("--scatter-table acts on the two-channel emission -- pass --alpha-bulk "
                             "or --bulk-anchor pooled")
        scatter_of, scatter_record = read_scatter_table(scatter_table, axis, perts)
    if shaped and not two_channel:
        raise SystemExit("--emit-shape controls acts on the two-channel emission -- pass --alpha-bulk "
                         "or --bulk-anchor pooled")
    out_h5 = open_anndata_h5(out_path, "w")
    writer = CsrWriter(out_h5, len(axis))
    obs_labels, covered = [], 0
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
    if basal_slope not in BASAL_MODES:
        raise SystemExit(f"basal_slope must be one of {BASAL_MODES}, got {basal_slope!r}")
    if gamma != 1.0 or similarity_beta != 0.0 or basal_slope != "off":
        if list(prof.genes) != list(axis):
            need = ("gamma != 1" if gamma != 1.0 else
                    "similarity_beta != 0" if similarity_beta != 0.0 else "basal_slope")
            raise SystemExit(f"{need}: control profile genes differ from the real file's axis")
        ctrl_cpm = prof.fraction * 1e6
    # T77: Rhaister-O's term. The slope is fitted once over every target (the empirical-Bayes
    # prior needs all of them), read on this line through its control profile, and ADDED to
    # the pooled delta before alpha -- so alpha, the knockdown pin and the emitter see one
    # log2FC vector exactly as before. "off" touches nothing and is bit-identical.
    basal_mod, basal_stats = None, None
    if basal_slope != "off":
        fit = fit_basal_slopes(perts, sources, axis, var_floor=var_floor)
        basal_mod = fit.modifier(target_basal(ctrl_cpm, axis, fit.common), mode=basal_slope)
        basal_stats = fit.stats(basal_slope)
        basal_stats["modifier_mean_abs"] = float(np.abs(basal_mod).mean())
        basal_stats["modifier_nonzero_frac"] = float((basal_mod != 0).mean())
        del fit
    # T103: the neighbour arm. Its pool is pooled with every knob the targets get, so the pool's
    # residuals and SER's live in one space; off (w = 0) builds nothing and touches nothing.
    arm, arm_record = None, None
    if neighbour_w:
        if gamma != 1.0 or basal_slope != "off":
            raise SystemExit("the neighbour arm is wired for gamma = 1 and basal_slope off only, "
                             "as in sidechain.submit.build")

        def delta_of(label):
            if cache is not None:
                return cache.get(label, pool_fit_stats)
            return pooled_delta(label, sources, axis, shrinkage=shrinkage, shrink_k=shrink_k,
                                shrink_stage=shrink_stage, shrink_rule=shrink_rule,
                                var_floor=var_floor, stats=pool_fit_stats,
                                log_bias_correct=log_bias_correct, gamma=gamma,
                                ctrl_tgt_cpm=ctrl_cpm, coverage_tiers=coverage_tiers,
                                similarity_beta=similarity_beta)

        arm, arm_record = neighbour_arm_for(
            SimpleNamespace(neighbour_table=neighbour_table, neighbour_pool=neighbour_pool,
                            neighbour_k=neighbour_k, neighbour_w=neighbour_w,
                            neighbour_size=neighbour_size, neighbour_select=neighbour_select,
                            neighbour_cand=neighbour_cand, neighbour_picks=neighbour_picks),
            delta_of, axis)
    for p in perts:
        d = cache.get(p, pool_stats) if cache is not None else pooled_delta(
            p, sources, axis, shrinkage=shrinkage, shrink_k=shrink_k,
            shrink_stage=shrink_stage, shrink_rule=shrink_rule, var_floor=var_floor,
            log_bias_correct=log_bias_correct,
            gamma=gamma, ctrl_tgt_cpm=ctrl_cpm,
            coverage_tiers=coverage_tiers,
            similarity_beta=similarity_beta, stats=pool_stats)
        if d is not None:
            covered += 1
            if arm is not None:
                d = arm.fuse(p, d)
            if basal_mod is not None:
                d = d + basal_mod[perts.index(p)]
            d0 = d
            d = d * alpha    # alpha scales the pooled vector; gamma acted per source inside the pool
            if p in gene_pos:
                d[gene_pos[p]] = -2.32
        n = cells_per_pert or int((labels == p).sum())
        if not two_channel or (d is None and bulk_anchor == "mean_cpm" and not shaped):
            writer.append_csr(em.emit(n, d))
        elif d is None:
            # an uncovered target under the pooled anchor: control cells, bulk on the pooled profile.
            # --emit-shape controls sends an uncovered target here under either anchor, so that it
            # is emitted in the controls' shape like every other. Under mean_cpm that is one more
            # number drawn than `emit` draws, so the targets after it draw other cells than the
            # run without the flag; under the pooled anchor the path is the same either way.
            writer.append_csr(dual(n, None, None, p))
        else:
            # T84: the pseudobulk channel at its own amplitude, the per-cell channel at alpha;
            # the knockdown pin is the same on both, and everything upstream is untouched.
            d_bulk = d0 * (alpha_bulk if alpha_bulk is not None else alpha)
            if p in gene_pos:
                d_bulk[gene_pos[p]] = -2.32
            scatter = None
            if scatter_of is not None and p in scatter_of:
                scatter = np.ones(len(axis))
                scatter[scatter_of[p][0]] = scatter_of[p][1]
            writer.append_csr(dual(n, d, d_bulk, p, scatter))
        obs_labels += [p] * n
    n_rows = writer.close()
    assert n_rows == len(obs_labels), f"{n_rows} rows written, {len(obs_labels)} labels"
    write_frame(out_h5, "obs", np.array([f"pred_{i}" for i in range(n_rows)], dtype=object),
                {pert_col: np.asarray(obs_labels, dtype=object)})
    write_frame(out_h5, "var", axis.astype(object), {})
    out_h5.close()
    # `shrinkage` alone under-describes a depth-aware run ('false' while one
    # arm was shrunk), so the per-source overrides are reported beside it,
    # aligned with the source list: None = followed the global flag.
    return {"pred": str(out_path), "perturbations": len(perts), "covered_by_sources": covered,
            "cells": int(n_rows), "genes": len(axis), "dispersion": em.dispersion,
            "nonzeros": int(writer.nnz),
            "emit_lambda": em.lam,
            "shrinkage": shrinkage,
            # which rule and where (T84); the three defaults are the historical rule
            "shrink_k": shrink_k, "shrink_stage": shrink_stage, "shrink_rule": shrink_rule,
            # the adaptive rule's stopping constants are part of the model, and its fits on the
            # neighbour pool are counted apart from the targets' (those are in pool_stats)
            **({"adaptive_fit": {"max_cycles": adaptive_shrink.MAX_CYCLES,
                                 "tol_per_gene": adaptive_shrink.TOL_PER_GENE,
                                 "calm_cycles": adaptive_shrink.CALM_CYCLES,
                                 "min_genes": adaptive_shrink.MIN_GENES,
                                 "neighbour_pool": {k: v for k, v in (pool_fit_stats or {}).items()
                                                    if k.startswith("adaptive_")}}}
               if shrinkage and shrink_rule == "adaptive" else {}),
            "shrink_overrides": [getattr(as_delta_source(s), "shrink", None) for s in sources],
            "alpha": alpha, "alpha_bulk": alpha_bulk, "bulk_anchor": bulk_anchor,
            # targets whose two moments were jointly unreachable and carried one amplitude
            "dual_fallbacks": int(getattr(em, "dual_fallbacks", 0)) if two_channel else None,
            # which rung a failed target fell to, by name, with why the requested pair failed
            # ("envelope": refused before the fit; "fit": the moment fit did not converge)
            "dual_fallback": dual_fallback if two_channel else None,
            "dual_fallback_targets": fell_back if two_channel else None,
            # T85: present only with --scatter-table. `targets_carrying_it` counts the blocks whose
            # fit took the table's genes; a target that ended on the template rung carries none.
            **({"scatter_table": {**scatter_record,
                                  "targets_carrying_it": int(sum(v > 0 for v in sharpened.values())),
                                  "targets_listed_but_not_carrying": sorted(t for t, v in sharpened.items() if v == 0),
                                  "targets_listed_but_uncovered": sorted(set(scatter_of) - set(sharpened))}}
               if scatter_record is not None else {}),
            # T85: present only with --emit-shape controls. A target that ended on the template
            # rung is not control cells; it is named here and under dual_fallback_targets.
            **({"emit_shape": {"shape": emit_shape, "control_cells_kept": int(prof.n_cells),
                               "targets_in_the_controls_shape": int(sum(in_shape.values())),
                               "targets_left_on_the_template": sorted(t for t, v in in_shape.items() if not v)}}
               if shaped else {}),
            "gamma": gamma, "var_floor": var_floor,
            # Recorded because it moved on 2026-09-20 (T18 check 5) from 500 to the
            # submission's 1000: an arm scored before that date carries no floor in its
            # record and was built at 500. Two folds are affected and by under 2e-4 raw
            # pds (runs/probes/t18_check5_libsize_floor), but a knob that is not in the
            # record cannot be reproduced, and this one silently was not.
            "min_libsize": float(min_libsize),
            "log_bias_correct": bool(log_bias_correct),
            "coverage_tiers": coverage_tiers,
            "similarity_beta": similarity_beta,
            "basal_slope": basal_slope, "basal_slope_stats": basal_stats,
            # Recorded per source and by name, not as a bare list: a floor attached to the
            # wrong arm is the failure mode this knob has, so the run must say which arm got
            # which number rather than leaving it to the command line's order.
            "transfer_floor": {getattr(s[0] if isinstance(s, tuple) else s,
                                       "sidechain_name", f"src{i}"):
                               float(getattr(s[0] if isinstance(s, tuple) else s,
                                             "transfer_floor", 0.0) or 0.0)
                               for i, s in enumerate(sources)},
            "neighbour": None if arm is None else {**arm_record, **arm.summary()},
            # present only when the pooled deltas were read from a cache; the rest of the
            # record equals an uncached run's
            **({"delta_cache": cache.record()} if cache is not None else {}),
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
    ap.add_argument("--no-shrink", action="store_true",
                    help="switch the shrinkage of transferred log2FCs off entirely "
                         "(--shrink-stage and --shrink-rule are refused with it)")
    add_shrink_args(ap, twin="sidechain.submit.build")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--alpha-bulk", type=float, default=None, metavar="ALPHA_BULK",
                    help="T84: a second amplitude for the pseudobulk channel. The emitted cells' "
                         "equal-weight per-cell mean follows --alpha (what the four Wilcoxon "
                         "members read), their depth-weighted column sums follow this value "
                         "(what pds and mse read); count_emitters.PoissonEmitter.emit_dual. "
                         "Needs --emit-lambda > 0. Unset = one amplitude, bit-identical")
    ap.add_argument("--bulk-anchor", choices=["mean_cpm", "pooled"], default="mean_cpm",
                    help="T84 round 2: which control profile the pseudobulk channel is built on. "
                         "'mean_cpm' (default, bit-identical, every entry through SER-7abefn) is "
                         "the mean of per-cell CPM; 'pooled' is sum of counts / sum of depths, "
                         "the profile cell-eval2's control pseudobulk actually is. The per-cell "
                         "channel stays on mean_cpm either way. Needs --emit-lambda > 0 "
                         "(count_emitters.PoissonEmitter.bulk_anchor)")
    ap.add_argument("--dual-fallback", choices=["template", "anchor"], default="template",
                    help="what a target whose two moments cannot both be met falls back to. "
                         "'template' (default, bit-identical, every arm and entry through "
                         "SER-11abefknw): one amplitude on the mean-per-cell-CPM profile, so under "
                         "--bulk-anchor pooled it loses the anchor too. 'anchor': one amplitude "
                         "with the summed profile kept on the anchor, the template only if that "
                         "fails as well (under --bulk-anchor mean_cpm it re-pins the column sums "
                         "to their expectation instead). Same knob in sidechain.submit.build, so a "
                         "scored arm submits verbatim")
    ap.add_argument("--scatter-table", type=Path, default=None, metavar="PARQUET",
                    help="T85: a per-(perturbation, gene) dial on the emitted cell-to-cell scatter. A "
                         "parquet with target, feature, scatter >= 0: 1 = as --emit-lambda emits "
                         "the gene, 0 = no sampling scatter (the gene keeps its per-cell mean, its "
                         "column total and the depth tilt the two moments need), above 1 = wider. "
                         "The cells drawn, "
                         "their depths and every unlisted gene's draw are the run's without it. "
                         "Needs the two-channel emission (--alpha-bulk or --bulk-anchor pooled); "
                         "count_emitters.PoissonEmitter.emit_dual. Not a submit.build knob yet. With "
                         "--emit-shape controls the cells are control cells at other depths and the "
                         "factor acts on a gene's re-rated counts: 1 is the controls' shape, below 1 "
                         "narrower, 0 the predicted count in every cell")
    ap.add_argument("--emit-shape", choices=("template", "controls"), default="template",
                    help="T85: what the emitted cells are. template (the default, every scored arm) "
                         "is the emitter's own cells at --emit-lambda. controls emits real control "
                         "cells whose counts are re-rated to the prediction (thinned where less is "
                         "predicted, topped up at the cell's own rate where more is), every target "
                         "and every gene, the knockdown's own included: the rank test then calls a "
                         "gene for its predicted shift and not for being narrower than a real "
                         "cell's. The cells take real cells' depths. Both predicted moments are the "
                         "run's without it, and under --bulk-anchor pooled so is what every target "
                         "draws; under mean_cpm a target no source covers draws one number more, so "
                         "the targets after it draw other cells. An --alpha-bulk other than --alpha "
                         "is carried as a lean of every gene with the cell's depth, which is no "
                         "longer the controls' shape. Needs the two-channel emission; "
                         "count_emitters.PoissonEmitter.emit_dual(shape=True). Not a submit.build "
                         "knob yet")
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
    ap.add_argument("--basal-slope", choices=list(BASAL_MODES), default="off",
                    help="T77, Rhaister-O's term: add a per-(target, gene) slope on basal "
                         "expression to the pooled delta, fitted across the source lines and "
                         "read on the held-out line through its controls; 'gene' shrinks each "
                         "gene's slope by its own empirical-Bayes prior, 'global' by one prior "
                         "for all genes (models.basal_slope). 'off' is bit-identical to the "
                         "historical call. NOT wired on sidechain.submit.build yet: like gamma "
                         "it makes the shifts context-specific")
    ap.add_argument("--var-floor", choices=["none", "poisson"], default="none",
                    help="floor each pseudobulk arm's variance at its Poisson sampling variance "
                         "(same knob as sidechain.submit.build, so a scored arm submits verbatim)")
    ap.add_argument("--cells-per-pert", type=int)
    ap.add_argument("--min-libsize", type=float, default=CONTROL_MIN_LIBSIZE,
                    help="drop control cells below this depth before building the context "
                         "profile (same knob and same default as sidechain.submit.build, so a "
                         "scored arm submits verbatim). Arms scored before 2026-09-20 used 500, "
                         "which this entry point could not even be told to change; pass 500 to "
                         "reproduce one bit-for-bit.")
    ap.add_argument("--log-bias-correct", action="store_true",
                    help="add back the second-order bias of log2 of a noisy mean (`Var(m)/(2(m+c)^2 ln2)`), per arm, before pooling. The control arm is far deeper than any perturbed arm, so the two biases do not cancel and what is left is a shared negative shift on low-expression genes -- 9-12%% of a median delta on our genome-wide sources. Measured to cost 0.0027 raw pds; off by default (private research/ideas/batch-effect-diagnostics.md, T18 check 6)")
    add_neighbour_args(ap, twin="sidechain.submit.build")
    ap.add_argument("--delta-cache", type=Path, default=None, metavar="DIR",
                    help="read every pooled delta (targets and neighbour pool) from a per-fold "
                         "cache under DIR instead of pooling it (sidechain.eval.delta_cache): "
                         "the same deltas, without refitting the shrinkage rule in every arm. "
                         "The cache must have been built with --delta-cache-build from the same "
                         "sources, knobs and thread settings; an arm never fills it. Plain "
                         "--source pools only. Not a submit.build knob")
    ap.add_argument("--delta-cache-build", action="store_true",
                    help="with --delta-cache: pool every target of --real and every member of "
                         "--neighbour-pool once, write the cache, and stop -- nothing is "
                         "predicted or scored and --out is not written")
    ap.add_argument("--delta-cache-jobs", type=int, default=1, metavar="N",
                    help="--delta-cache-build: processes that pool side by side (default 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--de-backend", default="pdex")
    args = ap.parse_args(argv)
    cov_tiers = parse_coverage_tiers(args.coverage_tiers)
    check_neighbour_args(ap, args)
    check_shrink_args(ap, args)
    if args.delta_cache_build and args.delta_cache is None:
        ap.error("--delta-cache-build needs --delta-cache DIR")
    if args.delta_cache_jobs < 1 or (args.delta_cache_jobs != 1 and not args.delta_cache_build):
        ap.error("--delta-cache-jobs is a count >= 1 and only acts with --delta-cache-build")
    if args.delta_cache is not None and (args.shrink_source or args.lfc_source
                                         or args.transfer_floor):
        ap.error("--delta-cache holds the plain pool only: no --shrink-source, "
                 "--lfc-source or --transfer-floor")
    if args.neighbour_w and args.basal_slope != "off":
        ap.error("--neighbour-w with --basal-slope is not wired (sidechain.submit.build has no "
                 "basal slope to pair it with)")
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
    delta_cache = None
    if args.delta_cache is not None:
        from sidechain.eval.delta_cache import source_ids
        delta_cache = {"dir": args.delta_cache, "sources": source_ids(args.source),
                       "build_jobs": args.delta_cache_jobs if args.delta_cache_build else None}
    out = args.out.expanduser()
    if args.delta_cache_build:
        info = build_transfer_prediction(args.real, sources, out / "pred.h5ad",
                                         pert_col=args.pert_col, control=args.control,
                                         **shrink_kwargs(args), gamma=args.gamma,
                                         var_floor=args.var_floor, coverage_tiers=cov_tiers,
                                         similarity_beta=args.similarity_beta,
                                         basal_slope=args.basal_slope,
                                         log_bias_correct=args.log_bias_correct,
                                         neighbour_pool=args.neighbour_pool,
                                         delta_cache=delta_cache)
        print(json.dumps(info), flush=True)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    info = build_transfer_prediction(args.real, sources, out / "pred.h5ad", pert_col=args.pert_col,
                                     control=args.control, dispersion=args.dispersion,
                                     emit_lambda=args.emit_lambda,
                                     **shrink_kwargs(args), alpha=args.alpha,
                                     gamma=args.gamma, var_floor=args.var_floor,
                                     coverage_tiers=cov_tiers,
                                     similarity_beta=args.similarity_beta,
                                     basal_slope=args.basal_slope, alpha_bulk=args.alpha_bulk,
                                     bulk_anchor=args.bulk_anchor,
                                     cells_per_pert=args.cells_per_pert, seed=args.seed,
                                     min_libsize=args.min_libsize,
                                     log_bias_correct=args.log_bias_correct,
                                     neighbour_table=args.neighbour_table,
                                     neighbour_pool=args.neighbour_pool,
                                     neighbour_k=args.neighbour_k,
                                     neighbour_w=args.neighbour_w,
                                     neighbour_size=args.neighbour_size,
                                     neighbour_select=args.neighbour_select,
                                     neighbour_cand=args.neighbour_cand,
                                     neighbour_picks=args.neighbour_picks,
                                     dual_fallback=args.dual_fallback,
                                     delta_cache=delta_cache,
                                     scatter_table=args.scatter_table,
                                     emit_shape=args.emit_shape)
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
         "shrink_k": args.shrink_k, "shrink_stage": args.shrink_stage,
         "shrink_rule": args.shrink_rule,
         "alpha": args.alpha, "alpha_bulk": args.alpha_bulk, "bulk_anchor": args.bulk_anchor,
         "dual_fallback": args.dual_fallback,
         "scatter_table": None if args.scatter_table is None else str(args.scatter_table),
         "emit_shape": args.emit_shape,
         "gamma": args.gamma,
         "var_floor": args.var_floor,
         "similarity_beta": args.similarity_beta,
         "basal_slope": args.basal_slope,
         "coverage_tiers": args.coverage_tiers,
         "neighbour_table": None if args.neighbour_table is None else [
             str(t) for t in args.neighbour_table],
         "neighbour_pool": None if args.neighbour_pool is None else str(args.neighbour_pool),
         "neighbour_k": args.neighbour_k, "neighbour_w": args.neighbour_w,
         "neighbour_size": args.neighbour_size,
         "neighbour_select": args.neighbour_select, "neighbour_cand": args.neighbour_cand,
         "neighbour_picks": None if args.neighbour_picks is None else str(args.neighbour_picks),
         "seed": args.seed, "de_backend": args.de_backend},
        {"overall": res.get("overall"), "members": res.get("members")},
        artifacts=[str(out / "summary.json")],
    )
    print(json.dumps({k: v for k, v in res.items() if k in ("members", "overall")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
