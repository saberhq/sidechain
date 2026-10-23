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
    add_source_vote_args,
    apply_transfer_floors,
    as_delta_source,
    check_neighbour_args,
    check_shrink_args,
    check_shrink_rule,
    neighbour_arm_for,
    parse_coverage_tiers,
    parse_source_contexts,
    parse_transfer_floor,
    pooled_delta,
    read_scatter_frame,
    route_sources,
    scatter_pairs,
    shrink_kwargs,
    source_name,
    sources_from_specs,
)
from sidechain.submit.alpha_table import load_alpha_table
from sidechain.submit.ctrl_axes import DEFAULT_CACHE as CTRL_AXES_CACHE, add_ctrl_axes_args, check_ctrl_axes_args
from sidechain.submit.gene_factor import load_gene_factor
from sidechain.submit.source_vote import (parse_source_scales, parse_source_weights, pool_kwargs,
                                          read_source_weight_table, source_votes)
from sidechain.submit.pie_source import add_prediction_source_args, check_prediction_source_args, pie_shift_map
from sidechain.submit.variance_model import (
    VarianceModel,
    add_variance_args,
    apply_dispersion_fits,
    check_variance_args,
    parse_dispersion_fits,
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
    other than 1) and the record that goes into the run's summary. The rules are
    `sidechain.submit.build.scatter_pairs`, which a build's table goes through as well.
    """
    import hashlib

    path = Path(path).expanduser()
    tab = read_scatter_frame(path)
    if "context" in tab.columns:
        # a build's table (sidechain.submit.build): read here its contexts would be merged, or
        # refused as one pair listed twice with no word of why
        raise SystemExit(f"--scatter-table {path.name}: it carries a context column, which makes it "
                         "a build's table; a fold has one set of control cells and no context")
    out, record = scatter_pairs(tab[["target", "feature", "scatter"]], axis, perts, path.name)
    return out, {"table": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **record}


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
    variance_model: VarianceModel | None = None,
    rule_variance: str = "model",
    gene_factor: Path | None = None,
    gene_factor_context: str | None = None,
    source_vote=None,
    source_scale: dict[str, float] | None = None,
    alpha_table: Path | None = None,
    ctrl_axes: int = 0,
    ctrl_axes_eta: float = 1.0,
    ctrl_axes_basis: str = "pca",
    ctrl_axes_share: float = 1.0,
    ctrl_axes_cache: Path | None = CTRL_AXES_CACHE,
    prediction_source: str = "ser",
    pie_parquet: Path | None = None,
    pie_context: str | None = None,
    alpha_bulk_target: float | None = None,
    alpha_bulk_floor: float | None = None,
) -> dict:
    """Predict every non-control perturbation of `real_path` from `sources`.

    `gene_factor` (T102; `sidechain.submit.gene_factor`) is a table of per-gene multipliers applied to
    each target's pooled vector last before alpha, in both channels, after the neighbour arm and the
    basal slope; a gene it does not name keeps 1. It sits outside the delta cache's key, so an arm that
    differs by it alone reads the cache the shipped recipe built. None is the historical path. A per-line
    table (T120: a `context` column) applies its rows for `gene_factor_context` plus the rows with an empty
    context; the held-out line is not one of A/B/C, so the caller names which context's rows stand for it.

    `source_vote` and `source_scale` (`sidechain.submit.source_vote`; final-phase instruments items 3 and 4)
    reach every `pooled_delta` call, the targets' and the neighbour pool's, as `submit.build` passes them; a
    delta cache holds the plain pool only, so they are refused with one. None / {} is the historical path.

    `alpha_table` (T1; `sidechain.submit.alpha_table`) is a table of per-target multipliers on alpha:
    each named target's pooled vector is scaled by alpha times its factor, in both channels, the
    target's own gene pinned after it as always; a target it does not name keeps alpha. Outside the
    delta cache's key, like `gene_factor`. None is the historical path.

    `ctrl_axes` (T125; `sidechain.submit.ctrl_axes`) removes `ctrl_axes_eta` of each target's summed-profile
    change along the top `ctrl_axes` axes of this fold's own control cells (`ctrl_axes_basis` `random` is the
    twin), on the summed-profile channel of the pooled anchor only, after the pin; the per-cell channel is
    untouched. `ctrl_axes_share` below 1 deflates only the targets whose summed-profile change is smallest, so
    every target is predicted before the first is emitted (the same deltas, the same emission). Outside the
    delta cache's key. 0 is the historical path.

    `prediction_source` `pie` (T85; `sidechain.submit.pie_source`) takes each target's map before alpha from
    the `pie_context` rows of `pie_parquet` in place of the pooled delta; `sources` must be empty and the
    knobs that act inside the pool are refused. A target with no row is uncovered, as a target no source
    covers. `ser` is the historical path.

    `variance_model` and `rule_variance` (T98; `sidechain.submit.variance_model`) are passed to
    every `pooled_delta` call -- the targets, the neighbour pool and a delta-cache build -- and
    recorded; None / `shipped` is the historical path. Refused with a basal slope, whose own fits
    would read the shipped variance while the pool read the model's.

    `alpha_bulk_target` (T84's knee rule, `sidechain.submit.build --alpha-bulk-target`; T93 brought it to
    the folds) treats `alpha_bulk` as the ceiling: every target is emitted at it, cell-eval2's budget over
    claim is read on those cells (`sidechain.submit.knee`), and while the ratio is above the target the
    strength is lowered by `next_alpha_bulk` (never below alpha, at most `KNEE_MAX_READS` reads, the ceiling's
    included), each read a fresh emitter at `seed` into its own file; the read that stops is the prediction,
    so its cells are what a run at that `alpha_bulk` writes. None is the historical path.
    `alpha_bulk_floor` is the rule's lowest strength in place of alpha (it may sit below alpha); None is alpha.

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
    if prediction_source not in ("ser", "pie"):
        raise SystemExit(f"prediction_source must be 'ser' or 'pie', got {prediction_source!r}")
    pie = prediction_source == "pie"
    if pie and (pie_parquet is None or not pie_context):
        raise SystemExit("prediction_source pie needs pie_parquet and pie_context")
    if pie and (sources or delta_cache is not None or neighbour_w or basal_slope != "off" or gamma != 1.0
                or similarity_beta != 0.0 or coverage_tiers is not None):
        raise SystemExit("prediction_source pie stands where the pooled delta stands: no sources, delta cache, "
                         "neighbour arm, basal slope, gamma, similarity weight or coverage tiers")
    check_shrink_rule(shrink_k, shrink_stage, shrink_rule)     # before any file is written
    vote_kw = {**({"source_vote": source_vote} if source_vote is not None else {}),
               **({"source_scale": source_scale} if source_scale else {})}
    if vote_kw and (delta_cache is not None or pie):
        raise SystemExit("source_vote / source_scale act inside the pool: no delta cache (it holds the plain "
                         "pool) and no prediction_source pie")
    vm = variance_model if variance_model is not None else VarianceModel()
    if not vm.is_shipped and basal_slope != "off":
        raise SystemExit("--variance-model with --basal-slope is not wired: the slope's fits read the "
                         "shipped variance while the pool would read the model's")
    pool_variance = {"variance_model": vm, "rule_variance": rule_variance}
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
    shape_held: dict[str, int] = {}              # target -> cells of its block held to the emitter's caps
    shape_unmet: dict[str, int] = {}             # target -> genes the re-rating's solve left over 1 % off a moment

    def dual(n, d_cell, d_bulk, label, scatter=None):   # called only inside the write loop, after `em` exists
        block = em.emit_dual(n, d_cell, d_bulk, on_fail=on_fail, scatter=scatter, shape=shaped)
        how = getattr(em, "last_dual", "dual")
        if how in ("anchor", "template"):
            fell_back.setdefault(how, []).append([label, getattr(em, "last_dual_reason", None)])
        if scatter is not None:
            sharpened[label] = int(em.last_sharpened or 0)
        if shaped:
            in_shape[label] = bool(em.last_shaped)
            if em.last_shaped and getattr(em, "last_shape_unmet", None) is not None:
                shape_held[label], shape_unmet[label] = int(em.last_shape_held or 0), int(em.last_shape_unmet)
        return block

    # Backed, and the control cells are the only rows brought into memory. The X-Atlas
    # folds are 726 M nonzeros: reading one whole would cost ~6 GB before a single cell
    # is emitted, and the emitted side costs as much again.
    real = ad.read_h5ad(real_path, backed="r")
    labels = real.obs[pert_col].astype(str).to_numpy()
    perts = sorted(set(labels) - {control})
    axis = real.var_names.astype(str).to_numpy()
    if gene_factor_context is not None and gene_factor is None:
        raise SystemExit("gene_factor_context names which rows of a --gene-factor table apply: pass gene_factor")
    gf, gf_record = (load_gene_factor(gene_factor, axis, context=gene_factor_context) if gene_factor is not None
                     else (None, None))
    at, at_record = load_alpha_table(alpha_table, perts) if alpha_table is not None else (None, None)
    pie_map, pie_record = pie_shift_map(pie_parquet, pie_context, axis, perts) if pie else (None, None)
    cache = None
    if delta_cache is not None:
        from sidechain.eval.delta_cache import DeltaCache, build_cache, key_fields
        from sidechain.models.neighbour_arm import read_pool

        # a --source arm or a published-contrast table (T93), neither with a transfer floor
        plain = all((isinstance(s, tuple) and len(s) == 2 and not getattr(s[0], "transfer_floor", 0.0))
                    or (isinstance(s, LfcTable) and not getattr(s, "transfer_floor", 0.0)) for s in sources)
        if (gamma != 1.0 or similarity_beta != 0.0 or coverage_tiers is not None
                or basal_slope != "off" or not plain
                or len(sources) != len(delta_cache["sources"])):
            raise SystemExit("--delta-cache holds the plain pool only: gamma 1, no similarity "
                             "weight, no coverage tiers, no basal slope, --source and --lfc-source "
                             "arms alone")
        fields = key_fields(delta_cache["sources"], axis, {
            "shrinkage": shrinkage, "shrink_k": shrink_k, "shrink_stage": shrink_stage,
            "shrink_rule": shrink_rule, "var_floor": var_floor,
            "log_bias_correct": log_bias_correct,
            # T98: a cache built under one variance model never serves another; the fits' hashes
            # are in the key because the same spec on a refitted source is another model
            "variance_model": vm.identity(), "rule_variance": rule_variance,
            "dispersion_fits": [getattr(s[0] if isinstance(s, tuple) else s,
                                        "dispersion_fit_sha256", None) for s in sources]})
        if delta_cache.get("build_jobs"):
            if real.isbacked:
                real.file.close()

            def pool_one(label, stats):
                return pooled_delta(label, sources, axis, shrinkage=shrinkage, shrink_k=shrink_k,
                                    shrink_stage=shrink_stage, shrink_rule=shrink_rule,
                                    var_floor=var_floor, log_bias_correct=log_bias_correct,
                                    gamma=gamma, ctrl_tgt_cpm=None, coverage_tiers=coverage_tiers,
                                    similarity_beta=similarity_beta, stats=stats, **pool_variance)

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
    if alpha_bulk_target is not None:
        if alpha_bulk is None:
            raise SystemExit("--alpha-bulk-target lowers --alpha-bulk: pass --alpha-bulk (the ceiling)")
        if not 0.0 < alpha_bulk_target < 1.0:
            raise SystemExit(f"--alpha-bulk-target is a budget-over-claim ratio in (0, 1), got {alpha_bulk_target}")
        if alpha_bulk_floor is None and not alpha_bulk > alpha:
            raise SystemExit(f"--alpha-bulk-target lowers the summed strength from --alpha-bulk toward --alpha, so "
                             f"--alpha-bulk ({alpha_bulk:g}) must be above --alpha ({alpha:g})")
    if alpha_bulk_floor is not None:
        if alpha_bulk_target is None:
            raise SystemExit("--alpha-bulk-floor is the knee rule's lowest strength: pass --alpha-bulk-target")
        if not (np.isfinite(alpha_bulk_floor) and alpha_bulk_floor > 0):
            raise SystemExit(f"--alpha-bulk-floor must be a positive strength, got {alpha_bulk_floor}")
        if not alpha_bulk > alpha_bulk_floor:
            raise SystemExit(f"--alpha-bulk-target lowers the summed strength from --alpha-bulk toward "
                             f"--alpha-bulk-floor, so --alpha-bulk ({alpha_bulk:g}) must be above "
                             f"--alpha-bulk-floor ({alpha_bulk_floor:g})")
    # T125: the fold's own control-cell axes, read from the same control cells the profile was
    axes_v, ctrl_axes_per = None, {}
    if ctrl_axes:
        if bulk_anchor != "pooled":
            raise SystemExit("--ctrl-axes acts on the summed-profile channel of the pooled anchor: "
                             "pass --bulk-anchor pooled")
        from sidechain.submit.ctrl_axes import change_size, control_axes, deflate_bulk, summarize, weakest
        axes_v, axes_rec = control_axes(ctrl_tmp, k=ctrl_axes, min_libsize=min_libsize,
                                        basis=ctrl_axes_basis, cache_dir=ctrl_axes_cache, genes=axis)
    covered = 0
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
                                similarity_beta=similarity_beta, **pool_variance, **vote_kw)

        arm, arm_record = neighbour_arm_for(
            SimpleNamespace(neighbour_table=neighbour_table, neighbour_pool=neighbour_pool,
                            neighbour_k=neighbour_k, neighbour_w=neighbour_w,
                            neighbour_size=neighbour_size, neighbour_select=neighbour_select,
                            neighbour_cand=neighbour_cand, neighbour_picks=neighbour_picks),
            delta_of, axis)
    def predict(p):
        """The target's pooled vector at alpha with its pin, and the vector before alpha (None, None uncovered)."""
        if pie_map is not None:
            d = pie_map.pop(p, None)       # T85: PIE's map where the pooled delta stands; read once
        else:
            d = cache.get(p, pool_stats) if cache is not None else pooled_delta(
                p, sources, axis, shrinkage=shrinkage, shrink_k=shrink_k,
                shrink_stage=shrink_stage, shrink_rule=shrink_rule, var_floor=var_floor,
                log_bias_correct=log_bias_correct,
                gamma=gamma, ctrl_tgt_cpm=ctrl_cpm,
                coverage_tiers=coverage_tiers,
                similarity_beta=similarity_beta, stats=pool_stats, **pool_variance, **vote_kw)
        d0 = None
        if d is not None:
            if arm is not None:
                d = arm.fuse(p, d)
            if basal_mod is not None:
                d = d + basal_mod[perts.index(p)]
            if gf is not None:
                d = d * gf           # T102: last before alpha, so both channels carry it
            d0 = d
            # alpha scales the pooled vector; gamma acted per source inside the pool. The per-target
            # table (T1) rides on alpha: a factor of 1.0 leaves the product bit-identical.
            a_t = alpha * (at[p] if at is not None else 1.0)
            d = d * a_t
            if p in gene_pos:
                d[gene_pos[p]] = -2.32
        return d, d0

    def bulk_of(p, d0, a_bulk=None):
        # T84: the pseudobulk channel at its own amplitude, the per-cell channel at alpha;
        # the knockdown pin is the same on both, and everything upstream is untouched.
        # `a_bulk` is a knee read's strength (T93); None is alpha_bulk.
        a_b = alpha_bulk if a_bulk is None else a_bulk
        d_bulk = d0 * ((a_b if a_b is not None else alpha) * (at[p] if at is not None else 1.0))
        if p in gene_pos:
            d_bulk[gene_pos[p]] = -2.32
        return d_bulk

    def held(p):
        return () if p not in gene_pos else (gene_pos[p],)

    # T125: a share below 1 needs every target's summed-profile change before the first is emitted;
    # T93: so does a knee read, which emits every target more than once from the same predictions
    ahead = None
    if (axes_v is not None and ctrl_axes_share < 1.0) or alpha_bulk_target is not None:
        ahead = {p: predict(p) for p in perts}

    def emit_pass(path, a_bulk=None, meter=None, fresh=False):
        """Every target into `path` at the summed strength `a_bulk` (None: alpha_bulk). `fresh` starts a new
        emitter at `seed`, as a knee read after the first does; each pass's records start empty."""
        nonlocal em, covered
        if fresh:
            em = PoissonEmitter(prof, seed=seed, dispersion=dispersion, lam=emit_lambda, bulk_anchor=bulk_anchor)
        covered = 0
        for rec in (sharpened, in_shape, shape_held, shape_unmet, ctrl_axes_per):
            rec.clear()
        fell_back.clear()
        fell_back.update({"anchor": [], "template": []})
        deflated = None
        if axes_v is not None and ctrl_axes_share < 1.0:
            deflated = weakest({p: change_size(bulk_of(p, d0, a_bulk), prof.bulk_fraction, held(p))
                                for p, (d, d0) in ahead.items() if d is not None}, ctrl_axes_share)
        out_h5 = open_anndata_h5(path, "w")
        writer = CsrWriter(out_h5, len(axis))
        obs_labels = []
        for p in perts:
            d, d0 = (ahead[p] if alpha_bulk_target is not None else ahead.pop(p)) if ahead is not None else predict(p)
            if d is not None:
                covered += 1
            n = cells_per_pert or int((labels == p).sum())
            if not two_channel or (d is None and bulk_anchor == "mean_cpm" and not shaped):
                block = em.emit(n, d)
            elif d is None:
                # an uncovered target under the pooled anchor: control cells, bulk on the pooled profile.
                # --emit-shape controls sends an uncovered target here under either anchor, so that it
                # is emitted in the controls' shape like every other. Under mean_cpm that is one more
                # number drawn than `emit` draws, so the targets after it draw other cells than the
                # run without the flag; under the pooled anchor the path is the same either way.
                block = dual(n, None, None, p)
            else:
                d_bulk = bulk_of(p, d0, a_bulk)
                if axes_v is not None and (deflated is None or p in deflated):
                    d_bulk, ctrl_axes_per[p] = deflate_bulk(d_bulk, prof.bulk_fraction, axes_v, ctrl_axes_eta,
                                                            held=held(p))
                scatter = None
                if scatter_of is not None and p in scatter_of:
                    scatter = np.ones(len(axis))
                    scatter[scatter_of[p][0]] = scatter_of[p][1]
                block = dual(n, d, d_bulk, p, scatter)
            writer.append_csr(block)
            if meter is not None:
                meter.add(p, block)
            obs_labels += [p] * n
        n_rows = writer.close()
        assert n_rows == len(obs_labels), f"{n_rows} rows written, {len(obs_labels)} labels"
        write_frame(out_h5, "obs", np.array([f"pred_{i}" for i in range(n_rows)], dtype=object),
                    {pert_col: np.asarray(obs_labels, dtype=object)})
        write_frame(out_h5, "var", axis.astype(object), {})
        out_h5.close()
        return n_rows, writer.nnz

    knee_record = None
    if alpha_bulk_target is None:
        n_rows, nnz = emit_pass(out_path)
    else:
        # T84's rule on this fold's own emitted cells (sidechain.submit.build's choose_alpha_bulk): each read
        # writes its file; the read that stops is renamed to the prediction and the others are removed
        from sidechain.submit.build import KNEE_MAX_READS
        from sidechain.submit.knee import AIM_MARGIN, TARGET_SUM, KneeMeter, next_alpha_bulk

        a, reads, files = alpha_bulk, [], []
        floor = alpha if alpha_bulk_floor is None else alpha_bulk_floor
        while True:
            path = out_path.with_name(f"{out_path.stem}.knee{len(reads)}.h5ad")
            meter = KneeMeter(perts, axis)
            n_rows, nnz = emit_pass(path, a, meter, fresh=bool(reads))
            files.append(path)
            r = {"alpha_bulk": a, **meter.read(), "dual_fallbacks": int(getattr(em, "dual_fallbacks", 0))}
            reads.append(r)
            ratio = r["ratio_budget_over_claim"]
            print(f"  knee read {len(reads)}: alpha_bulk={a:.4f} budget {r['budget']:.5f} claim "
                  f"{r['claim_uncapped']:.5f} ratio {ratio:.4f} (target {alpha_bulk_target:g})", flush=True)
            if ratio <= alpha_bulk_target or a <= floor or len(reads) >= KNEE_MAX_READS or not np.isfinite(ratio):
                break
            a = next_alpha_bulk(a, ratio, alpha_bulk_target, floor)
            path.unlink(missing_ok=True)           # a read that does not stop is not kept: one file on disk at a time
        files[-1].replace(out_path)
        knee_record = {"alpha": alpha, "alpha_bulk_ceiling": alpha_bulk, "alpha_bulk_target": alpha_bulk_target,
                       "max_reads": KNEE_MAX_READS, "target_sum": TARGET_SUM, "aim_margin": AIM_MARGIN,
                       **({"alpha_bulk_floor": alpha_bulk_floor} if alpha_bulk_floor is not None else {}),
                       "alpha_bulk": a, "met": bool(ratio <= alpha_bulk_target), "reads": len(reads),
                       "at_floor": bool(a <= floor), "ratio_budget_over_claim": ratio, "tries": reads}
        if not knee_record["met"]:
            print(f"  WARNING: budget over claim {ratio:.4f} is above the target {alpha_bulk_target:g}; "
                  f"written at alpha_bulk={a:.4f}", flush=True)
    # `shrinkage` alone under-describes a depth-aware run ('false' while one
    # arm was shrunk), so the per-source overrides are reported beside it,
    # aligned with the source list: None = followed the global flag.
    return {"pred": str(out_path), "perturbations": len(perts), "covered_by_sources": covered,
            "cells": int(n_rows), "genes": len(axis), "dispersion": em.dispersion,
            "nonzeros": int(nnz),
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
            "alpha": alpha, "alpha_bulk": alpha_bulk if knee_record is None else knee_record["alpha_bulk"],
            "bulk_anchor": bulk_anchor,
            # T93: present only with alpha_bulk_target (the knee rule's reads, as build's .alpha_bulk.json)
            **({"alpha_bulk_knee": knee_record} if knee_record is not None else {}),
            "gene_factor": gf_record,
            "alpha_table": at_record,
            # T85: present only with prediction_source pie
            **({"prediction_source": {"source": "pie", **pie_record}} if pie else {}),
            # T125: present only with --ctrl-axes
            **({"ctrl_axes": {"k": ctrl_axes, "eta": ctrl_axes_eta, "basis": ctrl_axes_basis,
                              "share": ctrl_axes_share, "axes": axes_rec,
                              "deflation": {**summarize(ctrl_axes_per),
                                            "targets_left_as_they_were": covered - len(ctrl_axes_per)}}}
               if axes_v is not None else {}),
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
            # `cells_held_to_the_caps`: cells the emitter cut back to twice their own depth or to the
            # scorer's per-cell limit; `genes_left_unmet_a_target`: what its solve left to the fit.
            **({"emit_shape": {"shape": emit_shape, "control_cells_kept": int(prof.n_cells),
                               "targets_in_the_controls_shape": int(sum(in_shape.values())),
                               "targets_left_on_the_template": sorted(t for t, v in in_shape.items() if not v),
                               "cells_held_to_the_caps": int(sum(shape_held.values())),
                               "targets_with_a_held_cell": sorted(t for t, v in shape_held.items() if v),
                               "genes_left_unmet_a_target": ({"median": float(np.median(list(shape_unmet.values()))),
                                                              "max": int(max(shape_unmet.values()))}
                                                             if shape_unmet else None)}}
               if shaped else {}),
            "gamma": gamma, "var_floor": var_floor,
            # T98: the variance behind the pooling weight and the rule's standard error; per
            # source the fit file and its hash, or the multiplier's factors, so a scored arm says
            # which model it carried and on which fit
            "variance_model": {**vm.record(sources), "rule_variance": rule_variance,
                               # the neighbour pool is pooled under the same model: its floor-bound and
                               # unmodelled-source counts, kept apart from the targets' (in pool_stats)
                               "neighbour_pool_counts": {k: v for k, v in (pool_fit_stats or {}).items()
                                                         if k.startswith("variance_")}},
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
    ap.add_argument("--source-context", action="append", default=[], metavar="NAME=CTX[,CTX]",
                    help="the per-context source switch, as on sidechain.submit.build (T126): the source "
                         "whose file stem is NAME votes only in the named contexts. Needs --context, the "
                         "challenge context this fold's held-out line stands in for; a source not routed "
                         "to it is left out of the pool, so a build's routes score verbatim")
    ap.add_argument("--context", default=None, metavar="CTX",
                    help="the challenge context this fold stands in for, read by --source-context only")
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
    add_source_vote_args(ap, twin="sidechain.submit.build")
    ap.add_argument("--source-weight-context", default=None, metavar="CTX",
                    help="the context whose --source-weight / --source-weight-table rows stand for this fold's "
                         "held-out line (e.g. k562 on loco_k562gwps_wide); refused when no flag or row names it")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gene-factor", type=Path, default=None, metavar="TABLE",
                    help="a per-gene multiplier on each target's pooled log2FC vector, applied last before "
                         "--alpha and the knockdown pin, in both channels: a parquet or CSV with `gene` and "
                         "`factor` (finite, >= 0); unnamed genes keep 1; recorded in summary.json under "
                         "build.gene_factor with the table's sha256 (sidechain.submit.gene_factor; the same "
                         "flag on sidechain.submit.build). Outside the delta cache's key.")
    ap.add_argument("--gene-factor-context", default=None, metavar="CTX",
                    help="T120: with a per-line --gene-factor table (a `context` column), the context whose rows "
                         "apply on this fold, plus the rows with an empty context; e.g. k562 on "
                         "loco_k562gwps_wide. Refused on a table without the column or with no row for CTX. "
                         "Unset = the rows with an empty context only")
    ap.add_argument("--alpha-table", type=Path, default=None, metavar="TABLE",
                    help="T1: a per-target multiplier on --alpha, in both channels: a parquet or CSV with `target` "
                         "(perturbation labels) and `factor` (finite, >= 0); an unnamed target keeps --alpha; recorded "
                         "in summary.json under build.alpha_table with the table's sha256 (sidechain.submit.alpha_table; "
                         "the same flag on sidechain.submit.build). Outside the delta cache's key.")
    ap.add_argument("--alpha-bulk", type=float, default=None, metavar="ALPHA_BULK",
                    help="T84: a second amplitude for the pseudobulk channel. The emitted cells' "
                         "equal-weight per-cell mean follows --alpha (what the four Wilcoxon "
                         "members read), their depth-weighted column sums follow this value "
                         "(what pds and mse read); count_emitters.PoissonEmitter.emit_dual. "
                         "Needs --emit-lambda > 0. Unset = one amplitude, bit-identical")
    ap.add_argument("--alpha-bulk-target", type=float, default=None, metavar="R",
                    help="T84's knee rule on this fold (sidechain.submit.build's flag of the same name; T93): "
                         "--alpha-bulk is the ceiling; every target is emitted at it and cell-eval2's budget "
                         "over claim is read on those cells; while it is above R the strength is lowered "
                         "(sidechain.submit.knee.next_alpha_bulk, never below --alpha, at most 4 reads) and "
                         "the fold is emitted again; the read that stops is the prediction. Recorded under "
                         "build.alpha_bulk_knee in summary.json. Unset = bit-identical")
    ap.add_argument("--alpha-bulk-floor", type=float, default=None, metavar="FLOOR",
                    help="the lowest summed-profile strength --alpha-bulk-target may choose, in place of --alpha "
                         "(sidechain.submit.build's flag of the same name); a value below --alpha lets the summed "
                         "strength settle under the per-cell strength. Needs --alpha-bulk-target. Recorded under "
                         "build.alpha_bulk_knee. Unset = --alpha, bit-identical")
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
                         "count_emitters.PoissonEmitter.emit_dual. Same table in "
                         "sidechain.submit.build, where it may also carry a context column (a "
                         "table with one is refused here). With "
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
                         "count_emitters.PoissonEmitter.emit_dual(shape=True). The same knob "
                         "in sidechain.submit.build")
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
    add_variance_args(ap, twin="sidechain.submit.build")
    add_ctrl_axes_args(ap, twin="sidechain.submit.build")
    add_prediction_source_args(ap, twin="sidechain.submit.build", per_context=False)
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
    variance_model = check_variance_args(ap, args)       # T98
    check_ctrl_axes_args(ap, args)                       # T125
    check_prediction_source_args(ap, args, per_context=False)   # T85
    pie = args.prediction_source == "pie"
    if pie and (args.source or args.shrink_source or args.lfc_source or args.source_context
                or args.delta_cache is not None or args.neighbour_w or args.basal_slope != "off"
                or args.gamma != 1.0 or args.similarity_beta != 0.0 or args.coverage_tiers
                or args.transfer_floor or args.dispersion_fit or not variance_model.is_shipped
                or (args.shrink_k, args.shrink_stage, args.shrink_rule) != (1.0, "source", "garrote")
                or args.log_bias_correct):
        ap.error("--prediction-source pie stands where the pooled delta stands: no --source, --shrink-source, "
                 "--lfc-source, --source-context, --delta-cache, --neighbour-w, --basal-slope, --gamma, "
                 "--similarity-beta, --coverage-tiers, --transfer-floor, --dispersion-fit, --variance-model, "
                 "shrinkage rule or --log-bias-correct")
    if not variance_model.is_shipped and args.basal_slope != "off":
        ap.error("--variance-model with --basal-slope is not wired (the slope's fits would read the "
                 "shipped variance while the pool read the model's)")
    if args.delta_cache_build and args.delta_cache is None:
        ap.error("--delta-cache-build needs --delta-cache DIR")
    if args.delta_cache_jobs < 1 or (args.delta_cache_jobs != 1 and not args.delta_cache_build):
        ap.error("--delta-cache-jobs is a count >= 1 and only acts with --delta-cache-build")
    if args.delta_cache is not None and (args.shrink_source or args.transfer_floor or args.source_context):
        ap.error("--delta-cache holds the plain pool only: no --shrink-source, "
                 "--transfer-floor or --source-context")
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
    if not pie and not args.source and not args.shrink_source and not args.lfc_source:
        ap.error("need at least one --source, --shrink-source or --lfc-source")
    sources = sources_from_specs(args.source, args.shrink_source)
    for path in args.lfc_source:
        tab = LfcTable.load(path)
        tab.sidechain_name = Path(path).expanduser().stem
        sources.append(tab)
    sources = apply_transfer_floors(sources, parse_transfer_floor(args.transfer_floor))
    sources = apply_dispersion_fits(sources, parse_dispersion_fits(args.dispersion_fit))   # T98
    routes = parse_source_contexts(args.source_context)                                    # T126
    if routes and args.context is None:
        ap.error("--source-context needs --context: the challenge context this fold's held-out line "
                 "stands in for")
    if args.context is not None and not routes:
        ap.error("--context is read by --source-context only")
    if routes:
        sources = route_sources(sources, routes, [args.context], check_contexts=False)[args.context]
    # final-phase instruments items 3 and 4: the vote of the context standing for this fold, and the strengths
    weights = parse_source_weights(args.source_weight)
    weight_table = read_source_weight_table(args.source_weight_table)
    if (weights or weight_table is not None) and args.source_weight_context is None:
        ap.error("--source-weight / --source-weight-table need --source-weight-context: the context whose rows "
                 "stand for this fold's held-out line")
    if args.source_weight_context is not None and not weights and weight_table is None:
        ap.error("--source-weight-context is read by --source-weight / --source-weight-table only")
    names = [source_name(s) for s in sources]
    vote, vote_record = None, None
    if args.source_weight_context is not None:
        named = {c for _, c in weights} | (set(weight_table["context"]) if weight_table is not None else set())
        if args.source_weight_context not in named:
            ap.error(f"--source-weight-context {args.source_weight_context!r}: no flag or table row names it "
                     f"(they name {', '.join(sorted(named))})")
        votes, vote_record = source_votes(weights, weight_table, [args.source_weight_context], names)
        vote = pool_kwargs(votes, args.source_weight_context, {}).get("source_vote")
    scales = parse_source_scales(args.source_scale, names)
    if (vote_record is not None or scales) and (args.delta_cache is not None or pie):
        ap.error("--source-weight / --source-weight-table / --source-scale act inside the pool: no --delta-cache "
                 "(it holds the plain pool) and no --prediction-source pie")
    variance_model.check_sources(sources)
    delta_cache = None
    if args.delta_cache is not None:
        from sidechain.eval.delta_cache import source_ids
        delta_cache = {"dir": args.delta_cache, "sources": source_ids(args.source, args.lfc_source),
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
                                         delta_cache=delta_cache,
                                         variance_model=variance_model,
                                         rule_variance=args.rule_variance)
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
                                     alpha_bulk_target=args.alpha_bulk_target,
                                     alpha_bulk_floor=args.alpha_bulk_floor,
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
                                     emit_shape=args.emit_shape,
                                     variance_model=variance_model,
                                     rule_variance=args.rule_variance,
                                     gene_factor=args.gene_factor,
                                     **({"gene_factor_context": args.gene_factor_context}
                                        if args.gene_factor_context is not None else {}),
                                     **({"source_vote": vote} if vote is not None else {}),
                                     **({"source_scale": scales} if scales else {}),
                                     alpha_table=args.alpha_table,
                                     ctrl_axes=args.ctrl_axes, ctrl_axes_eta=args.ctrl_axes_eta,
                                     ctrl_axes_basis=args.ctrl_axes_basis,
                                     ctrl_axes_share=args.ctrl_axes_share,
                                     ctrl_axes_cache=args.ctrl_axes_cache,
                                     **({"prediction_source": "pie", "pie_parquet": args.pie_parquet,
                                         "pie_context": args.pie_context} if pie else {}))
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
    if vote_record is not None:
        res["sources"] |= {"source_weight": vote_record, "source_weight_context": args.source_weight_context}
    if scales:
        res["sources"] |= {"source_scale": scales}
    if routes:
        # off, the record is what an arm before the flag wrote (T126)
        res["sources"] |= {"source_context": {n: list(c) for n, c in routes.items()},
                           "context": args.context, "pooled": [source_name(s) for s in sources]}
    (out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    log_run(
        {"entry": "loco", "real": str(args.real), "bundle": str(args.bundle),
         "out": str(out), "sources": args.source, "shrink_sources": args.shrink_source,
         "lfc_sources": args.lfc_source,
         **({"source_context": list(args.source_context), "context": args.context} if routes else {}),
         **({"source_weight": list(args.source_weight), "source_weight_table": str(args.source_weight_table),
             "source_weight_context": args.source_weight_context} if vote_record is not None else {}),
         **({"source_scale": scales} if scales else {}),
         "dispersion": args.dispersion, "emit_lambda": args.emit_lambda,
         "shrinkage": not args.no_shrink,
         "shrink_k": args.shrink_k, "shrink_stage": args.shrink_stage,
         "shrink_rule": args.shrink_rule,
         "alpha": args.alpha, "alpha_bulk": args.alpha_bulk, "bulk_anchor": args.bulk_anchor,
         **({"alpha_bulk_target": args.alpha_bulk_target} if args.alpha_bulk_target is not None else {}),
         **({"alpha_bulk_floor": args.alpha_bulk_floor} if args.alpha_bulk_floor is not None else {}),
         "gene_factor": None if args.gene_factor is None else str(args.gene_factor),
         **({"gene_factor_context": args.gene_factor_context} if args.gene_factor_context is not None else {}),
         "alpha_table": None if args.alpha_table is None else str(args.alpha_table),
         **({"ctrl_axes": args.ctrl_axes, "ctrl_axes_eta": args.ctrl_axes_eta,
             "ctrl_axes_basis": args.ctrl_axes_basis, "ctrl_axes_share": args.ctrl_axes_share}
            if args.ctrl_axes else {}),
         **({"prediction_source": "pie", "pie_parquet": str(args.pie_parquet), "pie_context": args.pie_context}
            if pie else {}),
         "dual_fallback": args.dual_fallback,
         "scatter_table": None if args.scatter_table is None else str(args.scatter_table),
         "emit_shape": args.emit_shape,
         "gamma": args.gamma,
         "var_floor": args.var_floor,
         "variance_model": variance_model.spec(), "rule_variance": args.rule_variance,
         "dispersion_fit": list(args.dispersion_fit),
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
