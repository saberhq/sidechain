#!/usr/bin/env python
"""Run a trained ``state tx`` arm against the 2026 controls bundle and write a submission.

``scripts/emit_tx_prediction.py`` is the sibling of this file and the one to read first: it
turns a checkpoint into a prediction the **local mirror** scores, on the model's own gene axis,
against a truth file that tells it what to emit. This does the same forward passes for the
**competition**, where there is no truth file and the axis is not the model's:

    per context in A, B, C
      for each of the 300 target genes
        400 predicted cells, each a full 18,533-gene raw-count transcriptome

Everything specific to that lives here; the model-facing half (``predict_step`` over windows of
cloned control cells, the shifted-log inverse, integerisation) is imported from the sibling so
the two cannot drift, and the contract half (out-of-core CSR, the ``.vcc`` container) comes from
``sidechain.submit.writer``, which is what every statistical arm already ships through.

**Four decisions this file makes, none of them plumbing. State them when reporting a score.**

1. **The input axis does not match.** The model's ``basal_encoder`` is a ``Linear(38584 -> 768)``
   fitted where all 38,584 genes were measured; the challenge measures 18,533, of which 18,106
   are on the model's axis. So **20,478 coordinates (53.1 %) of every input vector are a
   structural zero the model has never seen.** Zero is the only honest fill -- there is no
   measurement to put there -- but it is a distribution shift, not a formatting step, and it has
   to be priced before this script is worth running: ``diagnose_tx_arm.py --mask-genes-to``
   zeroes exactly those genes on cells the model *has* seen and reports what it costs.

2. **How the surviving coordinates are scaled**, which is the same question seen from the other
   end. ``prep_tx_training.py --log1p`` fed the model ``log1p(x * L / D)`` with *D* the cell's
   whole library and *L* the corpus's median of it. A challenge cell's ``D`` is unobservable --
   the axis it was measured on omits ~30 % of its UMIs (``challenges/vcc2026/CLAUDE.md``, trap
   4) -- so the scale factor is a modelling choice with two defensible ends, and
   ``--on-axis-share`` is the dial between them:

   * ``--on-axis-share auto`` (the default; 0.7007 measured on X-Atlas HCT116, 0.7247 on
     HEK293T) reconstructs the cell the model would have seen and then drops the genes the
     challenge did not measure. The surviving values land where the masked diagnostic put them.
   * ``--on-axis-share 1.0`` treats the challenge axis as the whole library, which inflates
     every surviving value by ~1/0.70 and is the ``--mask-renorm`` arm of that same diagnostic.

   Both pass the diagnostic on *pseudobulk* statistics; they are not close on the *cells*. On
   real context-A controls, 1.0 emits a median 192 nonzero genes per cell with one gene taking
   68 % of it, against 0.70's 2,573 and 27 %. Measured 2026-09-07, not reasoned.

4. **The model's output needs an upper bound, and the bound is data-defined.** These predictions
   are shifted logarithms and ``expm1`` is unbounded: on out-of-distribution input the arm emits
   values of 17 (share 0.70) and 37 (share 1.0), which are 2.4e7 and 1e16 counts, and a single
   such gene *is* the cell. ``--clip-log auto`` bounds the output at the 99.9th percentile of the
   PER-CELL maximum in the training corpus -- a predicted cell may not carry a value larger than
   real cells of that corpus carry. It is the same guard ``state tx infer`` ships (it clips gene
   space to ``[0, 14]``); the constant is read from the data rather than hard-coded, because 14
   is a log-space bound and this model's cells top out near 6.9. It touches ~0.01 % of values
   and takes the median emitted cell from 21 % in one gene to 3 %.

3. **427 challenge genes are not on the model's axis and cannot be predicted.** They get that
   context's own control mean -- the honest encoding of "this model predicts no change here" --
   at the share of the library the controls give them, so the emitted cell's split between
   predicted and unpredictable genes matches the controls' and the remaining ``1 - q`` is the
   model's to distribute. ``--fill zero`` is available and is worse: it silently moves those
   genes' mass onto the 18,106 the model does emit.

Depth follows the sibling's ``--depth rescale``, and for the same reason (its module docstring
has the measurement): the model's *composition* is kept and each emitted cell's library size is
taken from the control cell it was built from. The literal count inverse of a log-trained model
is unbounded and cell-eval2 refuses it.

    uv run python scripts/emit_tx_submission.py \\
        --model-dir  ~/data/sidechain/runs/tx/phe2_hct116 \\
        --checkpoint ~/data/sidechain/runs/tx/phe2_hct116/checkpoints/last.ckpt \\
        --balanced-encoders --x-space log1p --depth rescale \\
        --out ~/data/sidechain/vcc2026/submissions/phe-2_tx_zeroshot_v1
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_tx_arm import _allow_numpy_globals, _pick_device, read_var_symbols
from emit_tx_prediction import emit, to_counts

from sidechain.models.count_emitters import CONTROL_MIN_LIBSIZE
from sidechain.submit.writer import Contract, SubmissionWriter, pack_vcc, verify_h5ad
from sidechain.utils.naming import CLAIMS_RE, check_out_leaf
from sidechain.utils.paths import resolve_config


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_context_controls(path: Path) -> tuple[sp.csr_matrix, np.ndarray, list[str]]:
    """One context's control pool as CSR, its per-cell raw depths, and its gene symbols.

    18,400 x 18,533 at ~6,000 nonzeros a cell is ~110 M entries: under a gigabyte held as CSR,
    and 120,000 random row reads per context if it is not held at all.
    """
    from anndata.io import read_elem

    with h5py.File(path, "r") as f:
        genes = [str(g) for g in read_elem(f["var"]).index]
        n, g = (int(v) for v in f["X"].attrs["shape"])
        indptr = f["X"]["indptr"][:]
        nnz = int(f["X"]["data"].shape[0])
        data = np.empty(nnz, dtype=np.float32)
        indices = np.empty(nnz, dtype=np.int32)
        step = 8_000_000
        for lo in range(0, nnz, step):
            data[lo:lo + step] = f["X"]["data"][lo:lo + step]
            indices[lo:lo + step] = f["X"]["indices"][lo:lo + step]
    X = sp.csr_matrix((data, indices, indptr), shape=(n, g))
    return X, np.asarray(X.sum(axis=1)).ravel(), genes


def measure_reference(path: Path, on_axis: np.ndarray, block: int = 20_000) -> dict:
    """The two constants a projection needs, read off the model's own training-space cells.

    Neither is a tuning knob and neither should be a literal in this file, because both are
    properties of a corpus that can change:

    * ``on_axis_share`` -- the median share of a cell's library that the challenge's 18,533-gene
      axis carries. It is what turns an observed on-axis depth back into the library the training
      prep divided by. (0.7007 on X-Atlas HCT116, 0.7247 on HEK293T, which is the same 29.5-32.5 %
      off-axis mass ``challenges/vcc2026/CLAUDE.md`` trap 4 measures in five unrelated
      experiments.)
    * ``clip_log`` -- the 99.9th percentile of the PER-CELL maximum value. A predicted cell has no
      business carrying a value larger than real cells of this corpus carry, and an unbounded
      regression output run on out-of-distribution input does exactly that. The per-cell maximum
      is the right statistic rather than the corpus maximum: the latter (8.06) licenses every one
      of 142,046 cells to be as extreme as the single most extreme one.

    One pass, in row blocks, so a 3-4.6 GB CSR never lands whole.
    """
    with h5py.File(path, "r") as f:
        indptr = f["X"]["indptr"][:]
        n = len(indptr) - 1
        per_cell_max = np.zeros(n, dtype=np.float64)
        share = np.zeros(n, dtype=np.float64)
        for lo in range(0, n, block):
            hi = min(lo + block, n)
            a, b = int(indptr[lo]), int(indptr[hi])
            data = f["X"]["data"][a:b]
            idx = f["X"]["indices"][a:b]
            ptr = indptr[lo:hi + 1] - a
            counts = np.expm1(data.astype(np.float64))
            keep = on_axis[idx]
            starts = ptr[:-1]
            nonempty = np.diff(ptr) > 0
            if nonempty.any():
                s = starts[nonempty]
                per_cell_max[lo:hi][nonempty] = np.maximum.reduceat(data, s)
                tot = np.add.reduceat(counts, s)
                kept = np.add.reduceat(np.where(keep, counts, 0.0), s)
                share[lo:hi][nonempty] = kept / np.maximum(tot, 1e-12)
    return {"n_cells": n,
            "on_axis_share_median": float(np.median(share)),
            "per_cell_max_median": float(np.median(per_cell_max)),
            "per_cell_max_p999": float(np.percentile(per_cell_max, 99.9)),
            "corpus_max": float(per_cell_max.max())}


def basal_window(pool: sp.csr_matrix, depths: np.ndarray, rows: np.ndarray,
                 chal_to_model: np.ndarray, n_model_genes: int, size_factor: float,
                 on_axis_share: float, x_space: str) -> np.ndarray:
    """Control cells on the challenge's axis -> the model's input, dense [len(rows), 38584].

    ``chal_to_model[j]`` is where challenge gene ``j`` sits on the model's axis, or -1 if the
    model does not carry it. Everything the model carries and the challenge does not stays zero:
    that is the structural mask decision 1 of the module docstring is about.

    The scale is ``L * on_axis_share / D_obs``: ``D_obs / on_axis_share`` estimates the library
    the model was trained to divide by, from the part of it the challenge measured.
    """
    block = pool[rows]
    keep = chal_to_model >= 0
    out = np.zeros((block.shape[0], n_model_genes), dtype=np.float32)
    cols = chal_to_model[block.indices]
    take = keep[block.indices]
    rowix = np.repeat(np.arange(block.shape[0]), np.diff(block.indptr))
    out[rowix[take], cols[take]] = block.data[take]
    scale = (size_factor * on_axis_share / np.maximum(depths[rows], 1.0)).astype(np.float32)
    out *= scale[:, None]
    return np.log1p(out) if x_space == "log1p" else out


def project_to_submission(comp: np.ndarray, model_to_chal: np.ndarray, fill_idx: np.ndarray,
                          fill_frac: np.ndarray, n_chal: int, depths: np.ndarray) -> np.ndarray:
    """Model-axis composition -> raw-count-scale values on the challenge's 18,533 genes.

    ``comp`` is non-negative and in count units but on the wrong axis and at the wrong total.
    The 18,106 genes both axes carry keep the model's relative composition and share
    ``1 - q`` of each cell; the 427 the model cannot emit take the controls' own ``q``. Each
    row is then multiplied by that cell's target library size, so the totals are exact before
    integerisation and nothing depends on the model's predicted depth.
    """
    on = model_to_chal >= 0
    shared = comp[:, on].astype(np.float64)
    shared /= np.maximum(shared.sum(1, keepdims=True), 1e-12)
    q = float(fill_frac.sum())
    out = np.zeros((comp.shape[0], n_chal), dtype=np.float64)
    out[:, model_to_chal[on]] = shared * (1.0 - q)
    if fill_idx.size:
        out[:, fill_idx] = fill_frac[None, :]
    return out * depths[:, None]


def control_mean_fraction(pool: sp.csr_matrix, depths: np.ndarray, min_libsize: float) -> np.ndarray:
    """Mean per-cell CPM/1e6 over the control pool -- the same quantity ContextProfile carries."""
    keep = depths > max(min_libsize, 0.0)
    cpm = sp.diags(1e6 / depths[keep]) @ pool[keep]
    mean = np.asarray(cpm.mean(axis=0)).ravel()
    return mean / mean.sum()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--challenge-config", default="challenges/vcc2026/config.yaml")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, help="default: <model-dir>/checkpoints/last.ckpt")
    ap.add_argument("--balanced-encoders", action="store_true",
                    help="the checkpoint was trained with sidechain.models.balanced_encoders")
    ap.add_argument("--x-space", choices=("counts", "log1p"), default="counts",
                    help="what the model consumes and emits (no default guess: getting it "
                         "wrong does not crash, it produces a confident meaningless file)")
    ap.add_argument("--depth", choices=("rescale",), default="rescale",
                    help="how an emitted cell gets its library size. Only `rescale` is offered "
                         "here: the sibling's `inverse` produced cells of 2.4e10 counts that "
                         "cell-eval2 refuses, and a submission has no truth file to notice on")
    ap.add_argument("--emit", choices=("round", "poisson"), default="round")
    ap.add_argument("--size-factor", type=float,
                    help="L, the median raw library the model was normalised against. Default: "
                         "read from --basal-reference, or the training median if that is absent")
    ap.add_argument("--basal-reference", type=Path,
                    help="the h5ad the model trained on, used only to read its median raw "
                         "depth (obs/total_counts) as L")
    ap.add_argument("--on-axis-share", default="auto",
                    help="what fraction of a cell's UMIs the challenge's 18,533-gene axis "
                         "carries. `auto` (default) measures it on --basal-reference; 1.0 "
                         "treats that axis as the whole library and inflates every surviving "
                         "value by ~1/0.70. Decision 2 in the module docstring")
    ap.add_argument("--clip-log", default="auto",
                    help="upper bound on the model's own output before expm1. `auto` (default) "
                         "is the 99.9th percentile of the PER-CELL maximum in --basal-reference; "
                         "`off` removes the bound; a float pins it. Decision 4 in the module "
                         "docstring -- without it a single gene takes a fifth of a cell")
    ap.add_argument("--fill", choices=("control-mean", "zero"), default="control-mean",
                    help="what the 427 challenge genes the model cannot emit receive")
    ap.add_argument("--min-libsize", type=float, default=CONTROL_MIN_LIBSIZE,
                    help="control cells below this depth are dropped from the basal pool "
                         "(same knob and same default as sidechain.submit.build)")
    ap.add_argument("--limit-perts", type=int, help="build only the first N perturbations")
    ap.add_argument("--limit-contexts", help="comma-separated subset of contexts (pipeline tests)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--out", required=True, help="output stem; writes <out>.h5ad and <out>.vcc")
    ap.add_argument("--no-pack", action="store_true")
    args = ap.parse_args(argv)

    stem = Path(args.out).name
    check_out_leaf(stem, context="emit_tx_submission", require_slug=True)
    if not CLAIMS_RE.match(stem):
        print(f"note: out stem '{stem}' carries no series tag -- fine for a probe, but a "
              "board submission's stem starts with its lowercased short name (ADR 0005)",
              flush=True)

    cfg = yaml.safe_load(resolve_config(args.challenge_config).read_text())
    data_dir = Path(cfg["data_dir"]).expanduser()
    genes = pd.read_csv(data_dir / cfg["gene_names_file"]).iloc[:, 0].astype(str).tolist()
    if len(genes) != cfg["n_genes"]:
        raise SystemExit(f"gene_names.csv read as {len(genes)} genes; config says "
                         f"{cfg['n_genes']} -- header handling?")
    perts = pd.read_csv(data_dir / cfg["pert_counts_file"])[cfg["pert_col"]].astype(str).tolist()
    if args.limit_perts:
        perts = perts[: args.limit_perts]
    contexts = [str(c) for c in cfg["phases"][cfg["phase"]]["contexts"]]
    if args.limit_contexts:
        want = [c.strip() for c in args.limit_contexts.split(",")]
        contexts = [c for c in contexts if c in want]
    sub = cfg["submission"]
    contract = Contract(
        genes=genes, perturbations=perts, contexts=contexts,
        cells_per_pert=int(sub["cells_per_pert"]), pert_col=cfg["pert_col"],
        context_col=cfg["context_col"], control_label=cfg["control_label"],
        max_counts_per_cell=int(sub["max_counts_per_cell"]), max_cells=int(sub["max_cells"]),
        max_stored_entries=int(sub["max_stored_entries"]),
    )

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".args.json").write_text(json.dumps(vars(args), indent=1, default=str) + "\n")

    # ---- the model's vocabulary and axis, asserted before any compute --------------------
    model_dir = args.model_dir.expanduser()
    ckpt = (args.checkpoint or model_dir / "checkpoints" / "last.ckpt").expanduser()
    _allow_numpy_globals()
    pert_map = torch.load(model_dir / "pert_onehot_map.pt", weights_only=False)
    with open(model_dir / "var_dims.pkl", "rb") as f:
        var_dims = pickle.load(f)
    key_of = {str(k): k for k in pert_map}
    _log(f"model dir     {model_dir}")
    _log(f"checkpoint    {ckpt.name}")
    _log(f"one-hot map   {len(key_of)} entries")

    # `state tx infer` substitutes the CONTROL one-hot for a perturbation missing from the map
    # and suppresses the warning under --quiet (_infer.py:756-806), which would produce a
    # confident file of control-like predictions. Fail instead, loudly, before the box time.
    missing = [p for p in perts if p not in key_of]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(perts)} challenge targets have no slot in the "
                         f"model's one-hot map, e.g. {missing[:5]} -- refusing to emit "
                         "control-like predictions under their names")
    zeros = [p for p in perts if float(pert_map[key_of[p]].abs().sum()) == 0]
    if zeros:
        raise SystemExit(f"{len(zeros)} targets map to an all-zero perturbation vector, "
                         f"e.g. {zeros[:5]} -- indistinguishable from no perturbation")
    _log(f"targets       {len(perts)}/{len(perts)} resolve in the one-hot map, 0 zero vectors")

    # ---- the two gene axes ---------------------------------------------------------------
    basal_ref = (args.basal_reference or
                 Path("~/data/sidechain/derived/tx-train-xatlas-log1p/loco_hct116_real.h5ad")
                 ).expanduser()
    model_genes = read_var_symbols(basal_ref)
    if len(model_genes) != int(var_dims["input_dim"]):
        raise SystemExit(f"{basal_ref.name} has {len(model_genes)} genes, model input_dim is "
                         f"{var_dims['input_dim']}")
    pos = {g: i for i, g in enumerate(model_genes)}
    chal_to_model = np.array([pos.get(g, -1) for g in genes], dtype=np.int64)
    model_to_chal = np.full(len(model_genes), -1, dtype=np.int64)
    for j, m in enumerate(chal_to_model):
        if m >= 0:
            model_to_chal[m] = j
    fill_idx = np.where(chal_to_model < 0)[0]
    _log(f"gene axes     challenge {len(genes):,} · model {len(model_genes):,} · shared "
         f"{int((chal_to_model >= 0).sum()):,} · model-only "
         f"{int((model_to_chal < 0).sum()):,} (zeroed on input) · challenge-only "
         f"{len(fill_idx):,} ({args.fill})")

    size_factor = args.size_factor
    if size_factor is None:
        with h5py.File(basal_ref, "r") as f:
            size_factor = float(np.median(f["obs/total_counts"][:]))

    ref: dict | None = None
    if args.on_axis_share == "auto" or args.clip_log == "auto":
        t_ref = time.time()
        on_axis = model_to_chal >= 0
        ref = measure_reference(basal_ref, on_axis)
        _log(f"reference     {basal_ref.name}: {ref['n_cells']:,} cells in {time.time() - t_ref:.0f}s"
             f" · on-axis share median {ref['on_axis_share_median']:.4f}"
             f" · per-cell max log1p median {ref['per_cell_max_median']:.3f}"
             f" p99.9 {ref['per_cell_max_p999']:.3f} corpus max {ref['corpus_max']:.3f}")
    on_axis_share = (ref["on_axis_share_median"] if args.on_axis_share == "auto"
                     else float(args.on_axis_share))
    if args.clip_log in ("off", "0", "none"):
        clip_log = None
    elif args.clip_log == "auto":
        clip_log = ref["per_cell_max_p999"]
    else:
        clip_log = float(args.clip_log)
    _log(f"size factor   L = {size_factor:,.1f}; on-axis share {on_axis_share:.4f}; "
         f"a control cell's on-axis total is scaled to {size_factor * on_axis_share:,.0f}")
    _log(f"clip          model output bounded at {clip_log if clip_log is None else round(clip_log, 3)}"
         " in the model's own (shifted-log) space")

    # ---- the model -----------------------------------------------------------------------
    if args.balanced_encoders:
        from sidechain.models.balanced_encoders import install

        install()
        _log("encoders      balanced class installed (settings come from the checkpoint)")
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    device = _pick_device(args.device)
    model = StateTransitionPerturbationModel.load_from_checkpoint(ckpt, map_location="cpu")
    model.eval()
    model.to(device)
    set_len = int(getattr(model, "cell_set_len", 512))
    _log(f"device        {device}, window {min(set_len, contract.cells_per_pert)} cells")

    # ---- write ---------------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    h5ad = out.with_suffix(".h5ad")
    if args.limit_perts:
        # so `vcc prep --dry-run --perts <this>` can validate a smoke build's layout
        pd.DataFrame({cfg["pert_col"]: perts}).to_csv(out.with_suffix(".pert_counts.csv"),
                                                      index=False)
    t0 = time.time()
    stats: dict[str, dict] = {}
    n_clipped = n_values = 0
    with SubmissionWriter(h5ad, contract) as w:
        for ctx in contexts:
            pool, depths, ctx_genes = load_context_controls(data_dir / cfg["control_files"][ctx])
            if ctx_genes != genes:
                raise SystemExit(f"context {ctx} var_names differ from {cfg['gene_names_file']}")
            keep = depths > max(args.min_libsize, 0.0)
            rows_ok = np.where(keep)[0]
            frac = control_mean_fraction(pool, depths, args.min_libsize)
            fill_frac = frac[fill_idx] if args.fill == "control-mean" else np.zeros(len(fill_idx))
            _log("")
            _log(f"context {ctx}     {len(rows_ok):,} control cells over {args.min_libsize:.0f} UMI, "
                 f"depth median {np.median(depths[rows_ok]):,.0f}; the {len(fill_idx)} "
                 f"unpredictable genes carry {fill_frac.sum():.3%} of a control cell")
            stats[ctx] = {"control_cells": len(rows_ok),
                          "control_depth_median": float(np.median(depths[rows_ok])),
                          "fill_share": float(fill_frac.sum())}
            emitted_depth: list[float] = []
            want = contract.cells_per_pert
            with torch.inference_mode():
                for k, p in enumerate(perts):
                    take = rng.choice(rows_ok, size=want, replace=want > len(rows_ok))
                    vec = pert_map[key_of[p]].float().to(device)
                    blocks = []
                    for lo in range(0, want, set_len):
                        idx = take[lo:lo + set_len]
                        X = basal_window(pool, depths, idx, chal_to_model, len(model_genes),
                                         size_factor, on_axis_share, args.x_space)
                        batch = {
                            "ctrl_cell_emb": torch.tensor(X, device=device),
                            "pert_emb": vec.unsqueeze(0).repeat(len(idx), 1),
                            "pert_name": [p] * len(idx),
                        }
                        preds = model.predict_step(batch, batch_idx=0,
                                                   padded=False)["preds"].float().cpu().numpy()
                        del batch, X
                        if clip_log is not None:
                            n_clipped += int((preds > clip_log).sum())
                            n_values += preds.size
                            np.clip(preds, None, clip_log, out=preds)
                        # `size-factor` is the sibling's "stop at expm1": the composition, with
                        # no depth applied. The depth is applied on the submission axis below,
                        # which is what --depth rescale means once the axis changes.
                        comp = to_counts(preds, np.ones(len(idx)), size_factor, args.x_space,
                                         "size-factor")
                        np.clip(comp, 0.0, None, out=comp)
                        vals = project_to_submission(comp, model_to_chal, fill_idx, fill_frac,
                                                     len(genes), depths[idx])
                        blocks.append(emit(vals, args.emit, rng).astype(np.float32))
                    block = np.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0]
                    emitted_depth.extend(block.sum(1).tolist())
                    w.add_block(block, ctx, p)
                    del blocks, block
                    if device.type == "mps" and (k + 1) % 25 == 0:
                        torch.mps.empty_cache()
                    if (k + 1) % 25 == 0 or k + 1 == len(perts):
                        _log(f"    {ctx}: {k + 1}/{len(perts)} perturbations  "
                             f"{time.time() - t0:.0f}s")
            stats[ctx]["emitted_depth_median"] = float(np.median(emitted_depth))
            _log(f"    {ctx}: emitted depth median {np.median(emitted_depth):,.0f} against "
                 f"control {np.median(depths[rows_ok]):,.0f}")
            del pool, depths
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    info = verify_h5ad(h5ad, contract)
    record = {"h5ad": str(h5ad), **info, "write_seconds": round(time.time() - t0),
              "model_dir": str(model_dir), "checkpoint": str(ckpt), "x_space": args.x_space,
              "depth": args.depth, "emit": args.emit, "size_factor": size_factor,
              "on_axis_share": on_axis_share, "clip_log": clip_log, "fill": args.fill,
              "clipped_values": n_clipped, "model_output_values": n_values,
              "clipped_frac": (n_clipped / n_values) if n_values else None,
              "basal_reference": str(basal_ref), "reference_measured": ref,
              "n_shared_genes": int((chal_to_model >= 0).sum()),
              "n_filled_genes": len(fill_idx), "contexts": stats}
    _log("")
    _log(json.dumps(record))
    out.with_suffix(".build.json").write_text(json.dumps(record, indent=1) + "\n")
    if not args.no_pack:
        t1 = time.time()
        vcc = pack_vcc(h5ad, out.with_suffix(".vcc"))
        _log(json.dumps({"vcc": str(vcc), "vcc_gb": round(vcc.stat().st_size / 1e9, 2),
                         "pack_seconds": round(time.time() - t1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
