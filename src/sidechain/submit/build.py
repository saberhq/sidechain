"""Build a 2026 submission end to end: emitter -> out-of-core h5ad -> .vcc.

    uv run python -m sidechain.submit.build --challenge-config challenges/vcc2026/config.yaml \
        --emitter delta-transfer --h1-cache ~/data/sidechain/cache/vcc2026/h1_pseudobulk.npz \
        --gwps-cache ~/data/sidechain/cache/vcc2026/k562_gwps_targets_pseudobulk.npz \
        --out ~/data/sidechain/vcc2026/submissions/r1_delta_v1

Emitters (cells are integer counts at the target context's depth; --dispersion picks
Poisson or minimum-variance "even" cells):
  control-null     the context's control profile, no shift. A pipeline check; scores ~-0.3
                   because it calls no DE genes (fid charges silence).
  h1-mean-shift    one generic shift for every perturbation: the mean over the 300 H1
                   perturbations of their log2 fold change vs H1 controls. Rung 0b, transferred.
  delta-transfer   per-target log2 fold change pooled (inverse-variance, per gene) over the
                   sources that perturbed that gene -- K562 genome-wide and H1 -- re-anchored on
                   the target context's control profile. Targets no source covers fall back to
                   the H1 mean shift. Rung 1'.

`--source` adds a pseudobulk corpus beyond the two named caches, as `.npz:control_label`
(repeatable) -- the same syntax `sidechain.eval.loco` takes, so an arm scored on the mirror
is submitted with the identical source list. `--lfc-source` adds a corpus that publishes the
contrast already taken instead of cells (Feng 2026), built by
`python -m sidechain.data.lfc_table`. It pools identically; it just derives its per-gene
variance from an adjusted p-value rather than from CPM spread, and abstains (zero weight) on
genes whose p-value has saturated.

`--limit-perts N` builds a small panel (first N perturbations) and writes a matching
pert_counts CSV so `vcc prep --dry-run --perts <that>` can validate the layout locally.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sidechain.data.lfc_table import LfcTable
from sidechain.data.stream_pseudobulk import PseudobulkSums
from sidechain.models.count_emitters import (
    ContextProfile,
    PoissonEmitter,
    log2fc_from_cpm,
    remap_to_axis,
)
from sidechain.submit.writer import Contract, SubmissionWriter, pack_vcc, verify_h5ad
from sidechain.utils.paths import resolve_config

LN2_SQ = np.log(2) ** 2
TARGET_SELF_LOG2FC = -2.32  # > 80 % knockdown of the target itself; excluded from scoring, kept for realism


def _log2fc_with_var(pb: PseudobulkSums, label: str, control: str, pseudocount: float = 1.0,
                     var_floor: str = "none"):
    """Per-gene log2FC of mean CPM and its delta-method variance, for one source.

    Computed ROW-WISE, not by slicing `pb.mean_cpm()` / `pb.var_cpm()`. Those build the whole
    (labels x genes) matrix and this function needs exactly two of its rows -- which was
    invisible at 301 labels and fatal at 18,294: on a full-corpus X-Atlas artifact each call
    allocated ~5.6 GB per matrix, several times over, for two rows of 38,584 floats. Pooling
    272 targets from two such sources would have asked for hundreds of multi-GB temporaries.

    With `var_floor="none"` the arithmetic is unchanged -- `mean_cpm` and `var_cpm` are
    defined as exactly these per-row expressions, so this returns bit-identical values (held
    by `test_row_wise_log2fc_matches_the_whole_matrix_form`).

    `var_floor="poisson"` floors each arm's per-cell CPM variance at its Poisson sampling
    variance, `(m + pseudocount) * 1e6 / mean_libsize`: observed spread below what counting
    noise alone would produce is undersampling, not certainty. The `+ pseudocount` keeps
    never-expressed genes (observed variance exactly 0 at any n) off the pathological
    max-weight branch. An arm with a single cell has no observed spread at all, so it
    abstains outright (`var = inf`, weight 0) rather than letting the floor pretend one
    cell was a measurement.
    """
    i, c = pb.labels.index(label), pb.labels.index(control)
    ni = max(int(pb.n_cells[i]), 1)
    nc = max(int(pb.n_cells[c]), 1)
    mi = pb.cpm_sum[i] / ni
    mc = pb.cpm_sum[c] / nc
    vi = np.maximum(pb.cpm_sq_sum[i] / ni - mi * mi, 0.0)
    vc = np.maximum(pb.cpm_sq_sum[c] / nc - mc * mc, 0.0)
    if var_floor == "poisson":
        vi = np.maximum(vi, (mi + pseudocount) * 1e6 / (pb.libsize_sum[i] / ni))
        vc = np.maximum(vc, (mc + pseudocount) * 1e6 / (pb.libsize_sum[c] / nc))
    fc = log2fc_from_cpm(mi, mc, pseudocount)
    var = (vi / ni) / (mi + pseudocount) ** 2 + (vc / nc) / (mc + pseudocount) ** 2
    if var_floor == "poisson" and (ni < 2 or nc < 2):
        var = np.full_like(var, np.inf)
    return fc, var / LN2_SQ


def h1_mean_shift(h1: PseudobulkSums, control: str, axis: np.ndarray) -> np.ndarray:
    perts = [lab for lab in h1.labels if lab != control]
    fcs = np.stack([_log2fc_with_var(h1, p, control)[0] for p in perts])
    return remap_to_axis(fcs.mean(axis=0), h1.genes, axis)


def shrink(fc: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Per-gene positive-part shrinkage of log2FCs toward 0: fc * max(0, 1 - var/fc^2).

    A source measures each gene's fold change with its own sampling error; with
    ~170 cells per K562 target, a gene at 5 CPM carries roughly +-0.5 log2 of pure
    noise, which would be transferred as if it were signal and charged by the
    fold-change metric. This is the gene-wise James-Stein rule: a gene whose
    estimate is within one standard error of zero is set to zero, one at three
    standard errors keeps 89 % of its value, one at five keeps 96 %. A single
    global normal prior was tried first and erased the real effects too -- they
    are a few hundred genes among ~8,000 nulls, so any one-variance prior is
    dominated by the nulls.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(fc != 0, 1.0 - var / np.maximum(fc**2, 1e-12), 0.0)
    return fc * np.clip(factor, 0.0, 1.0)


class _PseudobulkDeltaSource:
    """Adapts `(PseudobulkSums, control_label)` to the `effect(target)` shape.

    Exists so `pooled_delta` iterates one kind of thing. Sources that ship
    counts and sources that ship a precomputed contrast then differ only in how
    they answer "what is this target's fold change and how well is it known",
    which is the only question the pooling actually asks.
    """

    __slots__ = ("control", "pb", "var_floor")

    def __init__(self, pb: PseudobulkSums, control: str, var_floor: str = "none"):
        self.pb, self.control, self.var_floor = pb, control, var_floor

    @property
    def genes(self) -> np.ndarray:
        return self.pb.genes

    def effect(self, target: str):
        if target not in self.pb.labels:
            return None
        return _log2fc_with_var(self.pb, target, self.control, var_floor=self.var_floor)


def as_delta_source(src, var_floor: str = "none"):
    """Normalise a source into something with `.genes` and `.effect(target)`.

    Accepts the historical `(PseudobulkSums, control_label)` tuple so every
    existing call site and command line keeps working unchanged, and passes an
    `LfcTable` (or anything else implementing the pair) straight through.
    `var_floor` only reaches the pseudobulk form: an LfcTable's variance comes
    from a p-value, already carries abstention (`inf`), and has no cell counts
    to floor against.
    """
    if isinstance(src, tuple):
        return _PseudobulkDeltaSource(*src, var_floor=var_floor)
    if hasattr(src, "effect") and hasattr(src, "genes"):
        return src
    raise TypeError(
        f"{type(src).__name__} is not a delta source: expected a "
        "(PseudobulkSums, control_label) tuple or an object with .genes and "
        ".effect(target) -> (fc, var) | None"
    )


def pooled_delta(target: str, sources: list, axis: np.ndarray,
                 *, shrinkage: bool = True, var_floor: str = "none",
                 stats: dict | None = None) -> np.ndarray | None:
    """Inverse-variance pool of the sources that perturbed `target`; None if none did.

    A source may be a `(PseudobulkSums, control_label)` tuple -- counts we
    accumulated, variance from the per-cell CPM spread -- or an `LfcTable`,
    which publishes the contrast already taken and derives its variance from an
    adjusted p-value. Both answer `effect(target)` with `(fc, var)` on their own
    gene axis, and everything below is indifferent to which it got.

    A source may also ABSTAIN per gene by returning `var = inf` there, which
    makes its weight exactly 0. Feng does this on the 98.9 % of rows whose
    p-value has saturated: it has a fold change but no evidence about it, and at
    a median of 25 cells per target "no detectable change" would otherwise drag
    real effects from better-powered sources toward zero. Note `any_src` is set
    on the source COVERING the target, not on it having a usable weight -- a
    target only Feng covers, and only saturated rows at that, returns a
    genuine all-zero delta rather than None, and does not silently fall through
    to the mean-shift fallback.

    `var_floor="poisson"` applies the sampling floor inside each pseudobulk
    source's variance (see `_log2fc_with_var`) and drops the weight clamp to a
    pure numerical guard -- with the floor on, no finite variance can be small
    for a pathological reason, so the clamp no longer has statistical work to do.
    `stats`, if given, accumulates the diagnostics across calls: how many gene
    weights sit at or below the historical 1e-6 clamp (the accidental-certainty
    arm this floor exists to remove), how many (target, source) arms fully
    abstained, and how many covered targets ended with zero total weight.
    """
    if var_floor not in ("none", "poisson"):
        raise ValueError(f"unknown var_floor {var_floor!r}: expected 'none' or 'poisson'")
    clamp = 1e-6 if var_floor == "none" else 1e-12
    num = np.zeros(len(axis)); den = np.zeros(len(axis)); any_src = False
    for src in (as_delta_source(s, var_floor=var_floor) for s in sources):
        got = src.effect(target)
        if got is None:
            continue
        any_src = True
        fc, var = got
        if shrinkage:
            fc = shrink(fc, var)
        # `1/inf` is 0, which is the abstention. The `maximum` floor only guards
        # the other end -- a variance so small it would swamp every other
        # source -- and must not be applied to inf, hence the divide as written.
        with np.errstate(divide="ignore"):
            w = 1.0 / np.maximum(var, clamp)
        w = np.where(np.isfinite(var), w, 0.0)
        fc = np.where(np.isfinite(fc), fc, 0.0)
        if stats is not None:
            finite = np.isfinite(var)
            stats["gene_weights"] = stats.get("gene_weights", 0) + int(finite.sum())
            stats["gene_weights_var_le_1e-6"] = (
                stats.get("gene_weights_var_le_1e-6", 0) + int((finite & (var <= 1e-6)).sum()))
            if not finite.any():
                stats["source_arms_abstained"] = stats.get("source_arms_abstained", 0) + 1
        num += remap_to_axis(fc * w, src.genes, axis, fill=0.0)
        den += remap_to_axis(w, src.genes, axis, fill=0.0)
    if not any_src:
        return None
    out = np.zeros(len(axis))
    nz = den > 0
    if stats is not None and not nz.any():
        stats["targets_zero_weight"] = stats.get("targets_zero_weight", 0) + 1
    out[nz] = num[nz] / den[nz]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--challenge-config", default="challenges/vcc2026/config.yaml")
    ap.add_argument("--emitter", choices=["control-null", "h1-mean-shift", "delta-transfer"], required=True)
    ap.add_argument("--h1-cache")
    ap.add_argument("--gwps-cache")
    ap.add_argument("--source", action="append", default=[], metavar="NPZ:CONTROL",
                    help="additional pseudobulk source for delta-transfer, as .npz:control_label "
                         "(repeatable) -- the syntax sidechain.eval.loco uses, so a mirror-scored "
                         "source list carries over verbatim")
    ap.add_argument("--lfc-source", action="append", default=[], metavar="NPZ",
                    help="cached LfcTable .npz -- a source publishing the contrast already "
                         "taken rather than cells (e.g. Feng 2026). Repeatable. Built by "
                         "`python -m sidechain.data.lfc_table`.")
    ap.add_argument("--alpha", type=float, default=1.0, help="scale applied to every transferred log2FC")
    ap.add_argument("--no-shrink", action="store_true", help="disable the per-gene empirical-Bayes shrinkage of transferred log2FCs")
    ap.add_argument("--var-floor", choices=["none", "poisson"], default="none",
                    help="floor each pseudobulk arm's per-gene variance at its Poisson sampling "
                         "variance and abstain on single-cell arms, so observed zero spread stops "
                         "counting as certainty in the pooling weights; 'none' reproduces the "
                         "historical weights bit-for-bit")
    ap.add_argument("--limit-perts", type=int, help="build only the first N perturbations (pipeline tests)")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--dispersion", choices=["poisson", "even"], default="even",
                    help="cell-to-cell spread of emitted counts (see count_emitters.PoissonEmitter)")
    ap.add_argument("--out", required=True, help="output stem; writes <out>.h5ad and <out>.vcc")
    ap.add_argument("--no-pack", action="store_true")
    ap.add_argument("--min-libsize", type=float, default=1000.0,
                    help="drop control cells below this depth from the library-size pool")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(resolve_config(args.challenge_config).read_text())
    data_dir = Path(cfg["data_dir"]).expanduser()
    genes = pd.read_csv(data_dir / cfg["gene_names_file"]).iloc[:, 0].astype(str).tolist()
    if len(genes) != cfg["n_genes"]:
        raise SystemExit(f"gene_names.csv read as {len(genes)} genes; config says {cfg['n_genes']} -- header handling?")
    perts = pd.read_csv(data_dir / cfg["pert_counts_file"])[cfg["pert_col"]].astype(str).tolist()
    if args.limit_perts:
        perts = perts[: args.limit_perts]
    contexts = [str(c) for c in cfg["phases"][cfg["phase"]]["contexts"]]
    sub = cfg["submission"]
    contract = Contract(
        genes=genes, perturbations=perts, contexts=contexts, cells_per_pert=int(sub["cells_per_pert"]),
        pert_col=cfg["pert_col"], context_col=cfg["context_col"], control_label=cfg["control_label"],
        max_counts_per_cell=int(sub["max_counts_per_cell"]), max_cells=int(sub["max_cells"]),
        max_stored_entries=int(sub["max_stored_entries"]),
    )
    axis = np.asarray(genes)

    # -- per-perturbation log2FC vectors (None = no shift)
    t0 = time.time()
    shifts: dict[str, np.ndarray | None] = {p: None for p in perts}
    fallback = 0
    pool_stats: dict = {}
    if args.emitter in ("h1-mean-shift", "delta-transfer"):
        if not args.h1_cache:
            raise SystemExit("--h1-cache is required for this emitter")
        h1 = PseudobulkSums.load(args.h1_cache)
        generic = h1_mean_shift(h1, cfg["control_label"], axis)
        shifts = {p: generic.copy() for p in perts}
    if args.emitter == "delta-transfer":
        if not args.gwps_cache:
            raise SystemExit("--gwps-cache is required for delta-transfer")
        gwps = PseudobulkSums.load(args.gwps_cache)
        sources = [(gwps, "control"), (h1, cfg["control_label"])]
        for spec in args.source:
            path, _, ctrl = spec.rpartition(":")
            sources.append((PseudobulkSums.load(path), ctrl or "control"))
        # Appended, not special-cased: `pooled_delta` normalises both forms, so a
        # source with no cells behind it enters the pool exactly like one that has
        # them and nothing downstream needs to know which it was.
        sources += [LfcTable.load(path) for path in args.lfc_source]
        for p in perts:
            d = pooled_delta(p, sources, axis, shrinkage=not args.no_shrink,
                             var_floor=args.var_floor, stats=pool_stats)
            if d is None:
                fallback += 1          # keep the generic shift
            else:
                shifts[p] = d
    gene_pos = {g: i for i, g in enumerate(genes)}
    for p, vec in shifts.items():
        if vec is not None:
            vec *= args.alpha
            if p in gene_pos:
                vec[gene_pos[p]] = TARGET_SELF_LOG2FC
    line = f"shifts ready in {time.time() - t0:.0f}s; fallback-to-generic: {fallback}"
    if pool_stats.get("gene_weights"):
        frac = pool_stats.get("gene_weights_var_le_1e-6", 0) / pool_stats["gene_weights"]
        line += (f"; var<=1e-6: {frac:.1%} of gene weights"
                 f"; abstained source-arms: {pool_stats.get('source_arms_abstained', 0)}"
                 f"; zero-weight targets: {pool_stats.get('targets_zero_weight', 0)}")
    print(line, flush=True)

    # -- write
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    h5ad = out.with_suffix(".h5ad")
    if args.limit_perts:
        pd.DataFrame({cfg["pert_col"]: perts}).to_csv(out.with_suffix(".pert_counts.csv"), index=False)
    t0 = time.time()
    with SubmissionWriter(h5ad, contract) as w:
        for ci, ctx in enumerate(contexts):
            prof = ContextProfile.from_controls(data_dir / cfg["control_files"][ctx], ctx, min_libsize=args.min_libsize)
            if list(prof.genes) != genes:
                raise SystemExit(f"context {ctx} var_names differ from gene_names.csv")
            em = PoissonEmitter(prof, seed=args.seed + ci, dispersion=args.dispersion)
            for k, p in enumerate(perts):
                w.add_block(em.emit(contract.cells_per_pert, shifts[p]), ctx, p)
                if (k + 1) % 50 == 0:
                    print(f"  {ctx}: {k + 1}/{len(perts)} perturbations  {time.time() - t0:.0f}s", flush=True)
    info = verify_h5ad(h5ad, contract)
    print(json.dumps({"h5ad": str(h5ad), **info, "write_seconds": round(time.time() - t0)}), flush=True)
    if not args.no_pack:
        t0 = time.time()
        vcc = pack_vcc(h5ad, out.with_suffix(".vcc"))
        print(json.dumps({"vcc": str(vcc), "vcc_gb": round(vcc.stat().st_size / 1e9, 2), "pack_seconds": round(time.time() - t0)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
