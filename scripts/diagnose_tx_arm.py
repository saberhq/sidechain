"""Ask whether a trained `state tx` arm emits any per-perturbation signal at all.

    uv run python scripts/diagnose_tx_arm.py \
        --model-dir ~/data/sidechain/runs/tx/phe1_onehot_hct116 \
        --checkpoint ~/data/sidechain/runs/tx/phe1_onehot_hct116/checkpoints/best.ckpt \
        --adata ~/data/sidechain/derived/tx-train-xatlas/loco_hct116_real.h5ad \
        --n-cells 256 --out ~/data/sidechain/runs/tx/phe1_onehot_hct116/diagnose

`sidechain.eval.local_mirror` says *how well* a prediction scores. It cannot say *why* a
bad one is bad, and a random-looking score has several very different causes that call
for different fixes. This separates them, and it does so without a bundle, without truth
alignment and without a GPU.

**The one design choice that makes the answer clean: every perturbation is run on the
SAME basal cells.** In real inference each perturbation draws its own control sample, so
two predicted profiles differ both because the perturbation differs and because the cells
differ. Freeze the cells and the second term is gone: *all* remaining spread across
perturbations is the model's response to `pert_emb`, with no sampling floor to subtract
and no null to simulate. A predicted spread of zero then means the model is emitting one
profile, full stop.

Four measurements come out, in the order that discriminates:

1. **`pert_encoder` spread against `basal_encoder` norm.** The two encodings are *added*
   before the transformer sees them (`state_transition.py:421`). If the perturbation term
   is orders of magnitude smaller than the basal term, the perturbation is noise on the
   input and nothing downstream can recover it. This is residual collapse at the source
   and it needs no forward pass.
2. **Predicted delta magnitude**, ‖mean_p − mean_control‖ / ‖mean_control‖, against the
   same ratio measured on real cells. Near zero is collapse; comparable to real is not.
3. **Effective rank of the predicted deltas** (participation ratio of the singular
   values). A model that has learned one generic "something was perturbed" direction and
   only scales it is rank ~1 — which scores like noise per target while looking alive on
   magnitude alone. Distinguishing that from collapse matters: they have different fixes.
4. **Per-target cosine with the real delta**, the quantity `pds_cosine` is built on.

Read them together. Magnitude ~0 → the loss found the do-nothing optimum. Magnitude fine
but rank ~1 → the model learned the marginal response, the failure that killed β, `c` and
`t`. Magnitude and rank fine but cosine ~0 → the arm is learning *something* per target
and getting it wrong, which is a different problem entirely and the only one of the three
that more data plausibly fixes.

Three controls, because two of the four numbers are uninterpretable without one:

- **the sampling floor** for the real deltas — control cells cut into fake perturbations of
  the same size, so "the model moves 3 % as far as reality" is not an artefact of comparing
  a noiseless prediction to a noisy truth;
- **a permutation null** for the truth cosine — two delta sets that share a strong common
  direction have a positive cosine whatever the pairing, so the observed value is read
  against the same predictions re-paired to the wrong targets;
- **`--reinit-pert-encoder`**, which throws the trained perturbation encoder away and
  changes nothing else, so the run says what training bought rather than what the
  architecture can do.

Model-agnostic by construction: it reads `pert_onehot_map.pt` and `var_dims.pkl` from the
model directory and calls `predict_step` exactly as `state tx infer` does, so an ESM2-
featurised arm is the same command with a different `--model-dir`.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def _allow_numpy_globals() -> None:
    """Let `torch.load` read the maps `state tx train` wrote (`sidechain-brev` trap 3e).

    PyTorch 2.6 flipped `weights_only` to `True` and these files carry numpy scalars, so
    the checkpoint and both maps refuse to load. Allowlist the numpy globals rather than
    setting `weights_only=False`: these are our own files, and the narrow permission says
    so where the blanket one would not.
    """
    import torch.serialization as ts

    allow = [np.dtype, np.ndarray]
    for mod in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            allow.append(__import__(mod, fromlist=["scalar"]).scalar)
        except (ImportError, AttributeError):
            continue
    for name in dir(np.dtypes):
        obj = getattr(np.dtypes, name)
        if isinstance(obj, type):
            allow.append(obj)
    ts.add_safe_globals(allow)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _pick_device(requested: str) -> torch.device:
    """`auto` prefers MPS on the Mac, then CUDA, then CPU.

    Unlike `state emb transform` (which hard-codes cuda-or-cpu), nothing in the
    transition model's forward is CUDA-only, so the Mac's GPU is usable here.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_control_cells(adata_path: Path, control_pert: str, pert_col: str,
                       n_cells: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (basal counts [n_cells, G], the control cells' row indices).

    Reads the CSR blocks for the sampled rows only — the fold files are 1.5-2.4 GB and
    the diagnostic needs a few hundred cells.
    """
    import h5py

    with h5py.File(adata_path, "r") as f:
        cats = f[f"obs/{pert_col}/categories"][:].astype(str)
        codes = f[f"obs/{pert_col}/codes"][:]
        if control_pert not in set(cats):
            raise SystemExit(f"control label {control_pert!r} not in {pert_col}; saw {cats[:5]}…")
        ctrl_code = int(np.where(cats == control_pert)[0][0])
        ctrl_rows = np.where(codes == ctrl_code)[0]
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(ctrl_rows, size=min(n_cells, len(ctrl_rows)), replace=False))
        X = _read_csr_rows(f, take)
    return X, take


def _read_csr_rows(f, rows: np.ndarray) -> np.ndarray:
    """Dense [len(rows), G] float32 from an h5ad CSR group, one row at a time."""
    g = f["X"]
    n_genes = int(g.attrs["shape"][1])
    indptr = g["indptr"][:]
    out = np.zeros((len(rows), n_genes), dtype=np.float32)
    for i, r in enumerate(rows):
        lo, hi = int(indptr[r]), int(indptr[r + 1])
        if hi > lo:
            out[i, g["indices"][lo:hi]] = g["data"][lo:hi]
    return out


def true_pseudobulk(adata_path: Path, pert_col: str, control_pert: str,
                    labels: list[str], max_cells: int, seed: int) -> dict[str, np.ndarray]:
    """Mean count profile per label, from the real cells. Streams the CSR once."""
    import h5py

    rng = np.random.default_rng(seed)
    want = set(labels)
    with h5py.File(adata_path, "r") as f:
        cats = f[f"obs/{pert_col}/categories"][:].astype(str)
        codes = f[f"obs/{pert_col}/codes"][:]
        n_genes = int(f["X"].attrs["shape"][1])
        indptr = f["X"].indptr[:] if hasattr(f["X"], "indptr") else f["X"]["indptr"][:]
        out: dict[str, np.ndarray] = {}
        for ci, lab in enumerate(cats):
            if lab not in want and lab != control_pert:
                continue
            rows = np.where(codes == ci)[0]
            if len(rows) > max_cells:
                rows = np.sort(rng.choice(rows, size=max_cells, replace=False))
            acc = np.zeros(n_genes, dtype=np.float64)
            g = f["X"]
            for r in rows:
                lo, hi = int(indptr[r]), int(indptr[r + 1])
                if hi > lo:
                    acc[g["indices"][lo:hi]] += g["data"][lo:hi]
            out[lab] = (acc / max(len(rows), 1)).astype(np.float32)
    return out


def control_split_null(adata_path: Path, pert_col: str, control_pert: str,
                       group_size: int, max_groups: int, seed: int) -> np.ndarray:
    """The real deltas' sampling floor: control cells cut into fake perturbations.

    A real per-perturbation delta measured on `group_size` cells carries Poisson and
    cell-to-cell noise, so ‖delta‖ is never zero even for a gene that does nothing.
    Splitting the control arm into groups of the same size and measuring the same
    quantity says how much of the real spread is that floor — without it the
    predicted/real magnitude ratio is not interpretable.
    """
    import h5py

    with h5py.File(adata_path, "r") as f:
        cats = f[f"obs/{pert_col}/categories"][:].astype(str)
        codes = f[f"obs/{pert_col}/codes"][:]
        ctrl_rows = np.where(codes == int(np.where(cats == control_pert)[0][0]))[0]
        rng = np.random.default_rng(seed)
        rng.shuffle(ctrl_rows)
        n_groups = min(max_groups, len(ctrl_rows) // group_size)
        if n_groups < 2:
            return np.array([])
        n_genes = int(f["X"].attrs["shape"][1])
        indptr = f["X"]["indptr"][:]
        g = f["X"]
        means = np.zeros((n_groups, n_genes), dtype=np.float64)
        for k in range(n_groups):
            for r in ctrl_rows[k * group_size:(k + 1) * group_size]:
                lo, hi = int(indptr[r]), int(indptr[r + 1])
                if hi > lo:
                    means[k, g["indices"][lo:hi]] += g["data"][lo:hi]
            means[k] /= group_size
    log = cp10k_log1p(means.astype(np.float32))
    ref = log.mean(0)
    return np.linalg.norm(log - ref, axis=1) / max(float(np.linalg.norm(ref)), 1e-12)


def cp10k_log1p(profiles: np.ndarray) -> np.ndarray:
    """Depth-normalise then log1p. Rows are mean count profiles, not single cells."""
    depth = profiles.sum(axis=1, keepdims=True)
    depth[depth == 0] = 1.0
    return np.log1p(profiles / depth * 1e4)


def effective_rank(mat: np.ndarray) -> tuple[float, np.ndarray]:
    """Participation ratio of the singular-value spectrum, and the top-10 variance shares.

    (sum s^2)^2 / sum s^4 — 1.0 when one direction carries everything, n when the
    spectrum is flat. Reported instead of a hard rank because a near-collapse is what we
    are looking for, and a numerical rank would call it full.
    """
    s = np.linalg.svd(mat, compute_uv=False)
    p = s**2
    if p.sum() == 0:
        return 0.0, p
    return float(p.sum() ** 2 / (p**2).sum()), p[:10] / p.sum()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, help="default: <model-dir>/checkpoints/best.ckpt")
    ap.add_argument("--adata", required=True, type=Path, help="h5ad supplying basal cells and truth")
    ap.add_argument("--pert-col", default="gene_target")
    ap.add_argument("--control-pert", default="Non-Targeting")
    ap.add_argument("--n-cells", type=int, default=256, help="cells in the frozen basal set")
    ap.add_argument("--n-perts", type=int, default=0, help="0 = every perturbation in the map")
    ap.add_argument("--truth-max-cells", type=int, default=400,
                    help="cells per label for the real pseudobulk")
    ap.add_argument("--no-truth", action="store_true", help="skip the comparison with real cells")
    ap.add_argument("--reinit-pert-encoder", action="store_true",
                    help="untrained control: replace the trained pert_encoder with fresh random "
                         "weights and change nothing else, so the run measures what training "
                         "bought the perturbation pathway rather than what the pathway can do")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="directory for report.json + profiles.npz")
    ap.add_argument("--n-scramble", type=int, default=200,
                    help="permutations for the scrambled-target null on the truth cosine")
    ap.add_argument("--from-profiles", type=Path,
                    help="re-read an earlier run's profiles.npz instead of running the forward "
                         "pass again — the statistics are cheap, the 831 forwards are not")
    args = ap.parse_args(argv)

    model_dir = args.model_dir.expanduser()
    ckpt = (args.checkpoint or model_dir / "checkpoints" / "best.ckpt").expanduser()
    adata_path = args.adata.expanduser()

    _allow_numpy_globals()
    pert_map = torch.load(model_dir / "pert_onehot_map.pt", weights_only=False)
    with open(model_dir / "var_dims.pkl", "rb") as f:
        var_dims = pickle.load(f)
    pert_names = [str(k) for k in pert_map]
    _log(f"model dir     {model_dir}")
    _log(f"checkpoint    {ckpt.name}")
    _log(f"perturbations {len(pert_names)} in the map, control={args.control_pert!r}")
    if args.control_pert not in set(pert_names):
        raise SystemExit(f"control {args.control_pert!r} is not in the perturbation map")

    if args.from_profiles:
        cached = np.load(args.from_profiles.expanduser() / "profiles.npz", allow_pickle=True)
        order = [str(x) for x in cached["labels"]]
        profiles = cached["profiles"]
        pert_emb = cached["pert_emb"]
        prior = json.loads((args.from_profiles.expanduser() / "report.json").read_text())
        enc = prior["encoders"]
        # the device that produced the profiles, not the one recomputing statistics
        device = torch.device(prior.get("device", "cpu"))
        _log(f"reusing       {args.from_profiles}/profiles.npz — no forward pass")
        _log(f"              {profiles.shape[0]} perturbations x {profiles.shape[1]} genes")
        return _statistics(args, adata_path, order, profiles, pert_emb, enc, device, model_dir, ckpt)

    from state.tx.models.state_transition import StateTransitionPerturbationModel

    device = _pick_device(args.device)
    _log(f"device        {device}")
    model = StateTransitionPerturbationModel.load_from_checkpoint(ckpt, map_location="cpu")
    model.eval()
    if args.reinit_pert_encoder:
        torch.manual_seed(args.seed)
        for mod in model.pert_encoder.modules():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
        _log("pert_encoder  RE-INITIALISED (untrained control)")
    model.to(device)

    # ---- 1. the encoders, before any cell is pushed through the transformer -------------
    ident = torch.eye(len(pert_names), dtype=torch.float32, device=device)
    with torch.inference_mode():
        pert_emb = model.encode_perturbation(ident).float().cpu().numpy()  # [P, H]

    X, _ = load_control_cells(adata_path, args.control_pert, args.pert_col, args.n_cells, args.seed)
    _log(f"basal set     {X.shape[0]} control cells x {X.shape[1]} genes, "
         f"median depth {np.median(X.sum(1)):,.0f}")
    if X.shape[1] != int(var_dims["input_dim"]):
        raise SystemExit(f"gene axis {X.shape[1]} != model input_dim {var_dims['input_dim']}")

    with torch.inference_mode():
        basal_emb = model.encode_basal_expression(
            torch.tensor(X, device=device)).float().cpu().numpy()  # [S, H]

    pert_centred = pert_emb - pert_emb.mean(0, keepdims=True)
    enc = {
        "pert_emb_norm_median": float(np.median(np.linalg.norm(pert_emb, axis=1))),
        "pert_emb_spread_median": float(np.median(np.linalg.norm(pert_centred, axis=1))),
        "basal_emb_norm_median": float(np.median(np.linalg.norm(basal_emb, axis=1))),
        "basal_emb_spread_median": float(np.median(np.linalg.norm(
            basal_emb - basal_emb.mean(0, keepdims=True), axis=1))),
    }
    enc["pert_over_basal_spread"] = enc["pert_emb_spread_median"] / max(enc["basal_emb_spread_median"], 1e-12)
    _log("")
    _log("[1] transformer input, hidden space — pert_encoder vs basal_encoder")
    _log(f"    ||pert_emb||          median {enc['pert_emb_norm_median']:.4g}")
    _log(f"    spread across perts   median {enc['pert_emb_spread_median']:.4g}")
    _log(f"    ||basal_emb||         median {enc['basal_emb_norm_median']:.4g}")
    _log(f"    spread across cells   median {enc['basal_emb_spread_median']:.4g}")
    _log(f"    pert spread / basal spread  {enc['pert_over_basal_spread']:.4g}")

    # ---- 2. the frozen-basal forward, one homogeneous set per perturbation ---------------
    order = pert_names if args.n_perts <= 0 else (
        [args.control_pert] + [p for p in pert_names if p != args.control_pert][: args.n_perts]
    )
    key_of = {str(k): k for k in pert_map}
    X_t = torch.tensor(X, device=device)
    profiles = np.zeros((len(order), X.shape[1]), dtype=np.float32)
    _log("")
    _log(f"[2] forward pass, frozen basal set, {len(order)} perturbations")
    with torch.inference_mode():
        for i, p in enumerate(order):
            vec = pert_map[key_of[p]].float().to(device)
            batch = {
                "ctrl_cell_emb": X_t,
                "pert_emb": vec.unsqueeze(0).repeat(X.shape[0], 1),
                "pert_name": [p] * X.shape[0],
            }
            out = model.predict_step(batch, batch_idx=0, padded=False)
            profiles[i] = out["preds"].float().mean(0).cpu().numpy()
            del out, batch
            # 831 forwards, each holding a [S, 38,584] activation: without this the MPS
            # caching allocator grows until the machine swaps and the loop slows ~10x.
            if device.type == "mps" and (i + 1) % 25 == 0:
                torch.mps.empty_cache()
            if (i + 1) % 50 == 0 or i + 1 == len(order):
                _log(f"    {i + 1}/{len(order)}")

    return _statistics(args, adata_path, order, profiles, pert_emb, enc, device,
                       model_dir, ckpt)


def _statistics(args, adata_path, order, profiles, pert_emb, enc, device,
                model_dir, ckpt) -> int:
    """Everything downstream of the forward pass: the deltas, the geometry, the truth."""
    ctrl_i = order.index(args.control_pert)
    ctrl_profile = profiles[ctrl_i]
    mask = np.ones(len(order), dtype=bool)
    mask[ctrl_i] = False
    deltas = profiles[mask] - ctrl_profile
    labels = [p for j, p in enumerate(order) if mask[j]]

    rel = np.linalg.norm(deltas, axis=1) / max(float(np.linalg.norm(ctrl_profile)), 1e-12)
    er, _ = effective_rank(deltas - deltas.mean(0, keepdims=True))

    # the same two statistics in the space the DE members read
    log_profiles = cp10k_log1p(profiles)
    log_deltas = log_profiles[mask] - log_profiles[ctrl_i]
    rel_log = np.linalg.norm(log_deltas, axis=1) / max(float(np.linalg.norm(log_profiles[ctrl_i])), 1e-12)
    er_log, top_log = effective_rank(log_deltas - log_deltas.mean(0, keepdims=True))

    _log("")
    _log("[3] predicted deltas from the frozen basal set (no sampling noise by construction)")
    _log(f"    counts space  ||delta||/||control||  median {np.median(rel):.3e}  "
         f"q10 {np.quantile(rel, 0.1):.3e}  q90 {np.quantile(rel, 0.9):.3e}")
    _log(f"    log1p space   ||delta||/||control||  median {np.median(rel_log):.3e}")
    _log(f"    effective rank of the delta matrix   counts {er:.2f}  log1p {er_log:.2f}  "
         f"(of {min(len(labels), profiles.shape[1])})")
    _log(f"    top-5 variance share (log1p)  {np.array2string(top_log[:5], precision=3)}")
    _log(f"    predicted median depth  control {ctrl_profile.sum():,.0f}  "
         f"perturbed {np.median(profiles[mask].sum(1)):,.0f}")

    report: dict[str, object] = {
        "model_dir": str(model_dir),
        "checkpoint": str(ckpt),
        "adata": str(adata_path),
        "n_basal_cells": int(args.n_cells),
        "n_perturbations": len(labels),
        "device": str(device),
        "reinit_pert_encoder": bool(args.reinit_pert_encoder),
        "encoders": enc,
        "predicted": {
            "rel_delta_counts_median": float(np.median(rel)),
            "rel_delta_counts_q10": float(np.quantile(rel, 0.1)),
            "rel_delta_counts_q90": float(np.quantile(rel, 0.9)),
            "rel_delta_log1p_median": float(np.median(rel_log)),
            "effective_rank_counts": er,
            "effective_rank_log1p": er_log,
            "top10_variance_share_log1p": [float(x) for x in top_log],
            "control_depth": float(ctrl_profile.sum()),
            "perturbed_depth_median": float(np.median(profiles[mask].sum(1))),
        },
    }

    # ---- 4. against the real cells -------------------------------------------------------
    if not args.no_truth:
        _log("")
        _log(f"[4] real cells, up to {args.truth_max_cells} per label")
        tp = true_pseudobulk(adata_path, args.pert_col, args.control_pert, labels,
                             args.truth_max_cells, args.seed)
        have = [lab for lab in labels if lab in tp]
        t_ctrl = tp[args.control_pert]
        t_mat = np.stack([tp[lab] for lab in have])
        t_rel = np.linalg.norm(t_mat - t_ctrl, axis=1) / max(float(np.linalg.norm(t_ctrl)), 1e-12)
        t_log = cp10k_log1p(np.vstack([t_ctrl[None, :], t_mat]))
        t_log_deltas = t_log[1:] - t_log[0]
        t_rel_log = np.linalg.norm(t_log_deltas, axis=1) / max(float(np.linalg.norm(t_log[0])), 1e-12)
        t_er, t_top = effective_rank(t_log_deltas - t_log_deltas.mean(0, keepdims=True))

        idx = {lab: j for j, lab in enumerate(labels)}
        p_log_deltas = np.stack([log_deltas[idx[lab]] for lab in have])
        num = (p_log_deltas * t_log_deltas).sum(1)
        den = np.linalg.norm(p_log_deltas, axis=1) * np.linalg.norm(t_log_deltas, axis=1)
        cos = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)

        # The house control, and it is load-bearing here: a cosine between two delta sets
        # that both carry a strong shared direction is positive whatever the pairing, so
        # "mean cosine > 0" is not evidence of per-target signal. Re-pair the predictions
        # to the wrong targets many times and read the observed mean against that null.
        rng_s = np.random.default_rng(args.seed)
        p_norm = np.linalg.norm(p_log_deltas, axis=1)
        t_norm = np.linalg.norm(t_log_deltas, axis=1)
        null_means = np.empty(args.n_scramble, dtype=np.float64)
        for s in range(args.n_scramble):
            perm = rng_s.permutation(len(have))
            s_den = p_norm[perm] * t_norm
            s_cos = np.where(s_den > 0,
                             (p_log_deltas[perm] * t_log_deltas).sum(1) / np.maximum(s_den, 1e-12),
                             0.0)
            null_means[s] = s_cos.mean()
        p_value = float((null_means >= cos.mean()).mean())

        floor = control_split_null(adata_path, args.pert_col, args.control_pert,
                                   args.truth_max_cells, 50, args.seed)
        _log(f"    matched labels {len(have)}")
        _log(f"    real  ||delta||/||control||  counts median {np.median(t_rel):.3e}  "
             f"log1p median {np.median(t_rel_log):.3e}")
        if floor.size:
            _log(f"    sampling floor from {floor.size} control-only groups of "
                 f"{args.truth_max_cells}  log1p median {np.median(floor):.3e}")
            report.setdefault("truth_floor", {})
            report["truth_floor"] = {
                "n_groups": int(floor.size),
                "group_size": int(args.truth_max_cells),
                "rel_delta_log1p_median": float(np.median(floor)),
            }
        _log(f"    real  effective rank (log1p) {t_er:.2f}")
        _log(f"    magnitude ratio predicted/real (log1p, median)  "
             f"{np.median(rel_log) / max(np.median(t_rel_log), 1e-12):.4f}")
        _log(f"    per-target cosine(predicted delta, real delta)  "
             f"mean {cos.mean():+.4f}  median {np.median(cos):+.4f}  "
             f"frac>0 {float((cos > 0).mean()):.3f}")
        _log(f"    scrambled-target null ({args.n_scramble} permutations)          "
             f"mean {null_means.mean():+.4f}  sd {null_means.std():.4f}  "
             f"p(observed <= null) {p_value:.3f}")
        report["truth"] = {
            "cosine_scrambled_null_mean": float(null_means.mean()),
            "cosine_scrambled_null_sd": float(null_means.std()),
            "cosine_permutation_p": p_value,
            "n_scramble": int(args.n_scramble),
            "n_matched": len(have),
            "rel_delta_counts_median": float(np.median(t_rel)),
            "rel_delta_log1p_median": float(np.median(t_rel_log)),
            "effective_rank_log1p": t_er,
            "top10_variance_share_log1p": [float(x) for x in t_top],
            "magnitude_ratio_pred_over_real_log1p": float(
                np.median(rel_log) / max(np.median(t_rel_log), 1e-12)),
            "cosine_mean": float(cos.mean()),
            "cosine_median": float(np.median(cos)),
            "cosine_frac_positive": float((cos > 0).mean()),
        }

    if args.out:
        out = args.out.expanduser()
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "profiles.npz", profiles=profiles,
                            labels=np.array(order, dtype=object), pert_emb=pert_emb)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        _log("")
        _log(f"wrote {out / 'report.json'} and {out / 'profiles.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
