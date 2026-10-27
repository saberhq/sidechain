"""Leave-one-context-out: predict a held-out line's knockdowns from the other lines' pseudobulks and score them.

    uv run python -m sidechain.eval.loco --real REAL.h5ad --pert-col perturbation --control non-targeting \\
        --source K562.npz:control --source RPE1.npz:control --bundle BUNDLE_DIR --out OUT_DIR --dispersion even

`--control` is the truth file's control label; each `--source`'s `:LABEL` is that source's own.
Reads the held-out h5ad, the sources and its mirror2026 bundle; writes pred.h5ad, pred_with_controls.h5ad and
summary.json under --out. Notes: private/code-notes/src/sidechain/eval/loco.py.md
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
from sidechain.submit.cell_map import CellMap
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
    """A per-(perturbation, gene) scatter table for `PoissonEmitter.emit_dual(scatter=)`, and its record."""
    import hashlib

    path = Path(path).expanduser()
    tab = read_scatter_frame(path)
    if "context" in tab.columns:
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
    cell_lfc_map: Path | None = None,
) -> dict:
    """Predict every non-control perturbation of `real_path` from `sources`; returns the build record."""
    if dual_fallback not in ("template", "anchor"):
        raise SystemExit(f"dual_fallback must be 'template' or 'anchor', got {dual_fallback!r}")
    if emit_shape not in ("template", "controls"):
        raise SystemExit(f"emit_shape must be 'template' or 'controls', got {emit_shape!r}")
    shaped = emit_shape == "controls"            # every block is re-rated control cells
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
    # the neighbour pool's adaptive fits, counted apart from the targets'
    pool_fit_stats = {} if shrinkage and shrink_rule == "adaptive" else None
    # what a target whose two moments cannot both be met falls back to: the one-amplitude template
    # or one amplitude with the summed profile kept on the emitter's anchor (PoissonEmitter.emit_dual)
    on_fail = "fallback" if dual_fallback == "template" else "anchor"
    fell_back: dict[str, list] = {"anchor": [], "template": []}

    sharpened: dict[str, int] = {}               # target -> genes whose scatter the block carries
    in_shape: dict[str, bool] = {}               # target -> its block is control cells (not the template rung)
    shape_held: dict[str, int] = {}              # target -> cells of its block held to the emitter's caps
    shape_unmet: dict[str, int] = {}             # target -> genes the re-rating's solve left outside its tolerance

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

    # backed: the control cells are the only rows brought into memory
    real = ad.read_h5ad(real_path, backed="r")
    labels = real.obs[pert_col].astype(str).to_numpy()
    perts = sorted(set(labels) - {control})
    axis = real.var_names.astype(str).to_numpy()
    if gene_factor_context is not None and gene_factor is None:
        raise SystemExit("gene_factor_context names which rows of a --gene-factor table apply: pass gene_factor")
    gf, gf_record = (load_gene_factor(gene_factor, axis, context=gene_factor_context) if gene_factor is not None
                     else (None, None))
    at, at_record = load_alpha_table(alpha_table, perts) if alpha_table is not None else (None, None)
    cmap = CellMap.load(cell_lfc_map) if cell_lfc_map is not None else None
    if cmap is not None:
        if cmap.contexts is not None:
            raise SystemExit("--cell-lfc-map: eval.loco reads one line; give the map one `scale`, not `contexts`")
        cmap = cmap.bind(axis, perts)
    pie_map, pie_record = pie_shift_map(pie_parquet, pie_context, axis, perts) if pie else (None, None)
    cache = None
    if delta_cache is not None:
        from sidechain.eval.delta_cache import DeltaCache, build_cache, key_fields
        from sidechain.models.neighbour_arm import read_pool

        # a --source arm or a published-contrast table, neither with a transfer floor
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
            # the variance model and its fits' hashes are in the key: a refitted source is another model
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
        dispersion = "even"    # this function's default
    em = PoissonEmitter(prof, seed=seed, dispersion=dispersion, lam=emit_lambda,
                        bulk_anchor=bulk_anchor)
    # a pooled anchor is a two-channel emission even at one amplitude: the pseudobulk moves
    # onto the depth-weighted control profile, the per-cell mean stays put
    two_channel = alpha_bulk is not None or bulk_anchor != "mean_cpm"
    if dual_fallback != "template" and not two_channel:
        raise SystemExit("--dual-fallback only acts on a two-channel emission -- pass --alpha-bulk "
                         "or --bulk-anchor pooled")
    if two_channel and em.lam == 0.0:
        raise SystemExit("--alpha-bulk / --bulk-anchor pooled need a depth spread (--emit-lambda "
                         "> 0 or --dispersion poisson): at lambda 0 the pseudobulk and the "
                         "per-cell mean coincide")
    gene_pos = {g: i for i, g in enumerate(axis)}
    # the per-gene scatter dial acts inside the two-moment fit
    scatter_of, scatter_record = None, None
    if scatter_table is not None:
        if not two_channel:
            raise SystemExit("--scatter-table acts on the two-channel emission -- pass --alpha-bulk "
                             "or --bulk-anchor pooled")
        scatter_of, scatter_record = read_scatter_table(scatter_table, axis, perts)
    if shaped and not two_channel:
        raise SystemExit("--emit-shape controls acts on the two-channel emission -- pass --alpha-bulk "
                         "or --bulk-anchor pooled")
    if cmap is not None and not two_channel:
        raise SystemExit("--cell-lfc-map acts on the per-cell channel alone -- pass --alpha-bulk "
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
    # the fold's own control-cell axes, read from the same control cells the profile was
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
    # gamma, the similarity weight and the basal slope read the control profile the emitter
    # anchors on (min_libsize-filtered, CPM within this file's own gene universe)
    ctrl_cpm = None
    if basal_slope not in BASAL_MODES:
        raise SystemExit(f"basal_slope must be one of {BASAL_MODES}, got {basal_slope!r}")
    if gamma != 1.0 or similarity_beta != 0.0 or basal_slope != "off":
        if list(prof.genes) != list(axis):
            need = ("gamma != 1" if gamma != 1.0 else
                    "similarity_beta != 0" if similarity_beta != 0.0 else "basal_slope")
            raise SystemExit(f"{need}: control profile genes differ from the real file's axis")
        ctrl_cpm = prof.fraction * 1e6
    # the basal slope: fitted once over every target, read on this line through its control
    # profile, and added to the pooled delta before alpha
    basal_mod, basal_stats = None, None
    if basal_slope != "off":
        fit = fit_basal_slopes(perts, sources, axis, var_floor=var_floor)
        basal_mod = fit.modifier(target_basal(ctrl_cpm, axis, fit.common), mode=basal_slope)
        basal_stats = fit.stats(basal_slope)
        basal_stats["modifier_mean_abs"] = float(np.abs(basal_mod).mean())
        basal_stats["modifier_nonzero_frac"] = float((basal_mod != 0).mean())
        del fit
    # the neighbour arm: its pool is pooled with every knob the targets get; w = 0 builds nothing
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
            d = pie_map.pop(p, None)       # the parquet's map where the pooled delta stands; read once
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
                d = d * gf           # last before alpha, so both channels carry it
            d0 = d
            # alpha scales the pooled vector (gamma acted per source inside the pool); the per-target
            # table rides on alpha
            a_t = alpha * (at[p] if at is not None else 1.0)
            d = d * a_t
            if cmap is not None:
                d = cmap.apply(d, target=p).astype(d.dtype, copy=False)     # the per-cell channel only
            if p in gene_pos:
                d[gene_pos[p]] = -2.32
        return d, d0

    def bulk_of(p, d0, a_bulk=None):
        # the pseudobulk channel at its own amplitude, the pin as on the per-cell channel;
        # `a_bulk` is a knee read's strength, None is alpha_bulk
        a_b = alpha_bulk if a_bulk is None else a_bulk
        d_bulk = d0 * ((a_b if a_b is not None else alpha) * (at[p] if at is not None else 1.0))
        if p in gene_pos:
            d_bulk[gene_pos[p]] = -2.32
        return d_bulk

    def held(p):
        return () if p not in gene_pos else (gene_pos[p],)

    # a ctrl-axes share below 1 and the knee rule need every target predicted before the first is emitted
    ahead = None
    if (axes_v is not None and ctrl_axes_share < 1.0) or alpha_bulk_target is not None:
        ahead = {p: predict(p) for p in perts}

    def emit_pass(path, a_bulk=None, meter=None, fresh=False):
        """Every target into `path` at summed strength `a_bulk` (None: alpha_bulk); `fresh` starts a new emitter."""
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
                # an uncovered target under the pooled anchor, or under either anchor with
                # --emit-shape controls: control cells, bulk on the anchor
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
        # the knee rule on this fold's emitted cells (as submit.build's choose_alpha_bulk): each read
        # writes its file; the read that stops is renamed to the prediction and the others are removed
        from sidechain.submit.build import KNEE_MAX_READS
        from sidechain.submit.knee import AIM_MARGIN, TARGET_SUM, UNDERSHOOT, KneeMeter, knee_keep, knee_next

        a, reads = alpha_bulk, []
        floor = alpha if alpha_bulk_floor is None else alpha_bulk_floor
        while True:
            path = out_path.with_name(f"{out_path.stem}.knee{len(reads)}.h5ad")
            meter = KneeMeter(perts, axis)
            n_rows, nnz = emit_pass(path, a, meter, fresh=bool(reads))
            r = {"alpha_bulk": a, **meter.read(), "dual_fallbacks": int(getattr(em, "dual_fallbacks", 0))}
            reads.append(r)
            ratio = r["ratio_budget_over_claim"]
            print(f"  knee read {len(reads)}: alpha_bulk={a:.4f} budget {r['budget']:.5f} claim "
                  f"{r['claim_uncapped']:.5f} ratio {ratio:.4f} (target {alpha_bulk_target:g})", flush=True)
            pairs = [(x["alpha_bulk"], x["ratio_budget_over_claim"]) for x in reads]
            nxt = knee_next(pairs, alpha_bulk_target, floor, alpha_bulk, KNEE_MAX_READS)
            if nxt is None:
                break
            a = nxt
            path.unlink(missing_ok=True)           # a read that does not stop is not kept: one file on disk at a time
        kept = knee_keep(pairs, alpha_bulk_target)     # an undershoot stepped back up may keep an earlier read
        if kept == len(reads) - 1:
            path.replace(out_path)
        else:
            # the last read is over the target and an earlier one is under it: that read is emitted again (the same
            # seed, so the same cells), so every record of the pass is its own
            path.unlink(missing_ok=True)
            n_rows, nnz = emit_pass(out_path, pairs[kept][0], None, fresh=True)
        a, ratio = pairs[kept]
        knee_record = {"alpha": alpha, "alpha_bulk_ceiling": alpha_bulk, "alpha_bulk_target": alpha_bulk_target,
                       "max_reads": KNEE_MAX_READS, "target_sum": TARGET_SUM, "aim_margin": AIM_MARGIN,
                       "undershoot": UNDERSHOOT,
                       **({"alpha_bulk_floor": alpha_bulk_floor} if alpha_bulk_floor is not None else {}),
                       "alpha_bulk": a, "met": bool(ratio <= alpha_bulk_target), "reads": len(reads),
                       "at_floor": bool(a <= floor), "ratio_budget_over_claim": ratio, "tries": reads,
                       **({"kept_read": kept + 1} if kept != len(reads) - 1 else {})}
        if not knee_record["met"]:
            print(f"  WARNING: budget over claim {ratio:.4f} is above the target {alpha_bulk_target:g}; "
                  f"written at alpha_bulk={a:.4f}", flush=True)
    # the per-source shrink overrides sit beside `shrinkage`, aligned with the sources
    # (None: followed the global flag)
    return {"pred": str(out_path), "perturbations": len(perts), "covered_by_sources": covered,
            "cells": int(n_rows), "genes": len(axis), "dispersion": em.dispersion,
            "nonzeros": int(nnz),
            "emit_lambda": em.lam,
            "shrinkage": shrinkage,
            # which shrinkage rule and where
            "shrink_k": shrink_k, "shrink_stage": shrink_stage, "shrink_rule": shrink_rule,
            # the adaptive rule's stopping constants, and its neighbour-pool fits apart from the
            # targets' (those are in pool_stats)
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
            # present only with alpha_bulk_target: the knee rule's reads, as build's .alpha_bulk.json
            **({"alpha_bulk_knee": knee_record} if knee_record is not None else {}),
            "gene_factor": gf_record,
            "alpha_table": at_record,
            # present only with --cell-lfc-map
            **({"cell_lfc_map": cmap.record()} if cmap is not None else {}),
            # present only with prediction_source pie
            **({"prediction_source": {"source": "pie", **pie_record}} if pie else {}),
            # present only with --ctrl-axes
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
            # present only with --scatter-table. `targets_carrying_it` counts the blocks whose
            # fit took the table's genes; a target that ended on the template rung carries none.
            **({"scatter_table": {**scatter_record,
                                  "targets_carrying_it": int(sum(v > 0 for v in sharpened.values())),
                                  "targets_listed_but_not_carrying": sorted(t for t, v in sharpened.items() if v == 0),
                                  "targets_listed_but_uncovered": sorted(set(scatter_of) - set(sharpened))}}
               if scatter_record is not None else {}),
            # present only with --emit-shape controls. A target that ended on the template
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
            # the variance behind the pooling weight and the rule's standard error; per source
            # the fit file and its hash, or the multiplier's factors
            "variance_model": {**vm.record(sources), "rule_variance": rule_variance,
                               # the neighbour pool is pooled under the same model: its floor-bound and
                               # unmodelled-source counts, kept apart from the targets' (in pool_stats)
                               "neighbour_pool_counts": {k: v for k, v in (pool_fit_stats or {}).items()
                                                         if k.startswith("variance_")}},
            "min_libsize": float(min_libsize),
            "log_bias_correct": bool(log_bias_correct),
            "coverage_tiers": coverage_tiers,
            "similarity_beta": similarity_beta,
            "basal_slope": basal_slope, "basal_slope_stats": basal_stats,
            # per source, by name
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
                    help="pseudobulk source whose transferred log2FCs are shrunk even under "
                         "--no-shrink, as on sidechain.submit.build (repeatable)")
    ap.add_argument("--lfc-source", action="append", default=[], metavar="NPZ",
                    help="cached LfcTable .npz: a source that publishes the contrast rather than "
                         "cells (repeatable)")
    ap.add_argument("--coverage-tiers", metavar="CUT:FACTOR,...",
                    help="weight each source's per-gene vote by the cells' worth of evidence "
                         "behind that gene (n_eff), as CUT:FACTOR pairs; as on sidechain.submit.build")
    ap.add_argument("--source-context", action="append", default=[], metavar="NAME=CTX[,CTX]",
                    help="the per-context source switch, as on sidechain.submit.build: the source "
                         "whose file stem is NAME votes only in the named contexts. Needs --context; a "
                         "source not routed to it is left out of the pool")
    ap.add_argument("--context", default=None, metavar="CTX",
                    help="the challenge context this fold stands in for, read by --source-context only")
    ap.add_argument("--transfer-floor", action="append", default=[], metavar="NAME=TAU2",
                    help="per-source transfer-error floor tau^2 added to that source's "
                         "variance before it becomes a pooling weight, keyed by the source "
                         "file's basename stem (repeatable); as on sidechain.submit.build")
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dispersion", choices=["poisson", "even"], default=None,
                    help="endpoint of the emission dial (default: even); exclusive with --emit-lambda")
    ap.add_argument("--emit-lambda", type=float, default=None, metavar="LAM",
                    help="emission-sharpening dial in [0, 1]: 0 = even cells, 1 = poisson cells, "
                         "interior values narrow the emitted cloud toward the mean "
                         "(count_emitters.PoissonEmitter); as on sidechain.submit.build")
    ap.add_argument("--no-shrink", action="store_true",
                    help="switch the shrinkage of transferred log2FCs off entirely "
                         "(--shrink-stage and --shrink-rule are refused with it)")
    add_shrink_args(ap, twin="sidechain.submit.build")
    add_source_vote_args(ap, twin="sidechain.submit.build")
    ap.add_argument("--source-weight-context", default=None, metavar="CTX",
                    help="the context whose --source-weight / --source-weight-table rows stand for this fold's "
                         "held-out line; refused when no flag or row names it")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gene-factor", type=Path, default=None, metavar="TABLE",
                    help="a per-gene multiplier on each target's pooled log2FC vector, applied last before "
                         "--alpha and the knockdown pin, in both channels: a parquet or CSV with `gene` and "
                         "`factor` (finite, >= 0); unnamed genes keep 1 (sidechain.submit.gene_factor)")
    ap.add_argument("--gene-factor-context", default=None, metavar="CTX",
                    help="with a per-line --gene-factor table (a `context` column), the context whose rows "
                         "apply on this fold, plus the rows with an empty context. Refused on a table without "
                         "the column or with no row for CTX. Unset = the rows with an empty context only")
    ap.add_argument("--alpha-table", type=Path, default=None, metavar="TABLE",
                    help="a per-target multiplier on --alpha, in both channels: a parquet or CSV with `target` "
                         "(perturbation labels) and `factor` (finite, >= 0); an unnamed target keeps --alpha "
                         "(sidechain.submit.alpha_table)")
    ap.add_argument("--cell-lfc-map", type=Path, default=None, metavar="JSON",
                    help="the per-cell channel, after --alpha, through a step map on its |log2FC| (JSON `lower`, "
                         "`size`; sidechain.submit.cell_map), signs kept, the own gene pinned after it; the summed "
                         "profile is untouched")
    ap.add_argument("--alpha-bulk", type=float, default=None, metavar="ALPHA_BULK",
                    help="a second amplitude for the pseudobulk channel: the emitted cells' "
                         "equal-weight per-cell mean follows --alpha, their depth-weighted column sums "
                         "follow this value (count_emitters.PoissonEmitter.emit_dual). Needs "
                         "--emit-lambda > 0. Unset = one amplitude")
    ap.add_argument("--alpha-bulk-target", type=float, default=None, metavar="R",
                    help="the knee rule, as on sidechain.submit.build: --alpha-bulk is the ceiling; every "
                         "target is emitted at it and cell-eval2's budget over claim is read on those cells; "
                         "while it is above R the strength is lowered (sidechain.submit.knee, never below "
                         "--alpha or --alpha-bulk-floor) and the fold is emitted again; the read that stops is the prediction")
    ap.add_argument("--alpha-bulk-floor", type=float, default=None, metavar="FLOOR",
                    help="the lowest summed-profile strength --alpha-bulk-target may choose, in place of --alpha "
                         "(as on sidechain.submit.build); it may sit below --alpha. Needs --alpha-bulk-target. "
                         "Unset = --alpha")
    ap.add_argument("--bulk-anchor", choices=["mean_cpm", "pooled"], default="mean_cpm",
                    help="which control profile the pseudobulk channel is built on: 'mean_cpm' "
                         "(default) the mean of per-cell CPM, 'pooled' sum of counts / sum of depths. "
                         "The per-cell channel stays on mean_cpm either way. Needs --emit-lambda > 0 "
                         "(count_emitters.PoissonEmitter.bulk_anchor)")
    ap.add_argument("--dual-fallback", choices=["template", "anchor"], default="template",
                    help="what a target whose two moments cannot both be met falls back to. "
                         "'template' (default): one amplitude on the mean-per-cell-CPM profile, so under "
                         "--bulk-anchor pooled it loses the anchor too. 'anchor': one amplitude "
                         "with the summed profile kept on the anchor, the template only if that "
                         "fails as well (under --bulk-anchor mean_cpm it re-pins the column sums "
                         "to their expectation instead). As on sidechain.submit.build")
    ap.add_argument("--scatter-table", type=Path, default=None, metavar="PARQUET",
                    help="a per-(perturbation, gene) dial on the emitted cell-to-cell scatter: a "
                         "parquet with target, feature, scatter >= 0 (1 = as emitted, 0 = no sampling "
                         "scatter, above 1 = wider). Needs the two-channel emission (--alpha-bulk or "
                         "--bulk-anchor pooled); a table with a context column is refused here")
    ap.add_argument("--emit-shape", choices=("template", "controls"), default="template",
                    help="what the emitted cells are: template (default) the emitter's own cells at "
                         "--emit-lambda; controls real control cells whose counts are re-rated to the "
                         "prediction (thinned where less is predicted, topped up where more is). Needs "
                         "the two-channel emission (count_emitters.PoissonEmitter.emit_dual(shape=True))")
    ap.add_argument("--similarity-beta", type=float, default=0.0,
                    help="exponent on each source's control-profile cosine to the held-out "
                         "context, applied to its pooling weight (submit.build.control_similarity); "
                         "0 = uniform pooling")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="transfer exponent on the target/source control-CPM ratio: 1 = the "
                         "fold change transfers, 0 = the absolute CPM change transfers "
                         "(submit.build.gamma_transfer). Not on sidechain.submit.build")
    ap.add_argument("--basal-slope", choices=list(BASAL_MODES), default="off",
                    help="add a per-(target, gene) slope on basal expression to the pooled delta, "
                         "fitted across the source lines and read on the held-out line through its "
                         "controls; 'gene' shrinks each gene's slope by its own empirical-Bayes prior, "
                         "'global' by one prior for all genes (models.basal_slope). Not on "
                         "sidechain.submit.build")
    ap.add_argument("--var-floor", choices=["none", "poisson"], default="none",
                    help="floor each pseudobulk arm's variance at its Poisson sampling variance "
                         "(as on sidechain.submit.build)")
    ap.add_argument("--cells-per-pert", type=int)
    ap.add_argument("--min-libsize", type=float, default=CONTROL_MIN_LIBSIZE,
                    help="drop control cells below this depth (total counts) before building the "
                         "context profile; the same default as sidechain.submit.build")
    ap.add_argument("--log-bias-correct", action="store_true",
                    help="add back the second-order bias of log2 of a noisy mean "
                         "(`Var(m)/(2(m+c)^2 ln2)`), per arm, before pooling")
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
    variance_model = check_variance_args(ap, args)
    check_ctrl_axes_args(ap, args)
    check_prediction_source_args(ap, args, per_context=False)
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
        # the emitter's own check, made before any work
        ap.error(f"--emit-lambda must be in [0, 1], got {args.emit_lambda}")
    if args.emit_lambda is None and args.dispersion is None:
        args.dispersion = "even"    # the default of this entry point
    # Same rule as mirror2026.score: an arm named like a model must spell it right;
    # freeform ablation labels pass untouched.
    check_out_leaf(args.out.expanduser().name, context="loco")

    # an LfcTable is a complete source; an arm with no source at all would score the fallback shift
    if not pie and not args.source and not args.shrink_source and not args.lfc_source:
        ap.error("need at least one --source, --shrink-source or --lfc-source")
    sources = sources_from_specs(args.source, args.shrink_source)
    for path in args.lfc_source:
        tab = LfcTable.load(path)
        tab.sidechain_name = Path(path).expanduser().stem
        sources.append(tab)
    sources = apply_transfer_floors(sources, parse_transfer_floor(args.transfer_floor))
    sources = apply_dispersion_fits(sources, parse_dispersion_fits(args.dispersion_fit))
    routes = parse_source_contexts(args.source_context)
    if routes and args.context is None:
        ap.error("--source-context needs --context: the challenge context this fold's held-out line "
                 "stands in for")
    if args.context is not None and not routes:
        ap.error("--context is read by --source-context only")
    if routes:
        sources = route_sources(sources, routes, [args.context], check_contexts=False)[args.context]
    # the source vote of the context standing for this fold, and the source strengths
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
                                     **({"cell_lfc_map": args.cell_lfc_map} if args.cell_lfc_map is not None else {}),
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
    # the sources, and which of them were shrunk, are part of what produced the run
    res["sources"] = {"pseudobulk": args.source, "shrink_pseudobulk": args.shrink_source,
                      "lfc": args.lfc_source}
    if vote_record is not None:
        res["sources"] |= {"source_weight": vote_record, "source_weight_context": args.source_weight_context}
    if scales:
        res["sources"] |= {"source_scale": scales}
    if routes:
        # without the flag the record is unchanged
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
         **({"cell_lfc_map": str(args.cell_lfc_map)} if args.cell_lfc_map is not None else {}),
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
