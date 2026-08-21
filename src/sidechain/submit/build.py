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


def _log2fc_with_var(pb: PseudobulkSums, label: str, control: str, pseudocount: float = 1.0):
    """Per-gene log2FC of mean CPM and its delta-method variance, for one source."""
    i, c = pb.labels.index(label), pb.labels.index(control)
    m = pb.mean_cpm(); v = pb.var_cpm(); n = np.maximum(pb.n_cells, 1)
    fc = log2fc_from_cpm(m[i], m[c], pseudocount)
    var = (v[i] / n[i]) / (m[i] + pseudocount) ** 2 + (v[c] / n[c]) / (m[c] + pseudocount) ** 2
    return fc, var / LN2_SQ


def h1_mean_shift(h1: PseudobulkSums, control: str, axis: np.ndarray) -> np.ndarray:
    perts = [lab for lab in h1.labels if lab != control]
    fcs = np.stack([_log2fc_with_var(h1, p, control)[0] for p in perts])
    return remap_to_axis(fcs.mean(axis=0), h1.genes, axis)


def pooled_delta(target: str, sources: list[tuple[PseudobulkSums, str]], axis: np.ndarray) -> np.ndarray | None:
    """Inverse-variance pool of the sources that perturbed `target`; None if none did."""
    num = np.zeros(len(axis)); den = np.zeros(len(axis)); any_src = False
    for pb, control in sources:
        if target not in pb.labels:
            continue
        any_src = True
        fc, var = _log2fc_with_var(pb, target, control)
        w = 1.0 / np.maximum(var, 1e-6)
        num += remap_to_axis(fc * w, pb.genes, axis, fill=0.0)
        den += remap_to_axis(w, pb.genes, axis, fill=0.0)
    if not any_src:
        return None
    out = np.zeros(len(axis))
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--challenge-config", default="challenges/vcc2026/config.yaml")
    ap.add_argument("--emitter", choices=["control-null", "h1-mean-shift", "delta-transfer"], required=True)
    ap.add_argument("--h1-cache")
    ap.add_argument("--gwps-cache")
    ap.add_argument("--alpha", type=float, default=1.0, help="scale applied to every transferred log2FC")
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
        for p in perts:
            d = pooled_delta(p, sources, axis)
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
    print(f"shifts ready in {time.time() - t0:.0f}s; fallback-to-generic: {fallback}", flush=True)

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
