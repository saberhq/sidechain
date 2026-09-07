#!/usr/bin/env python
"""Turn a trained ``state tx`` checkpoint into a prediction h5ad the local mirror can score.

``state tx infer`` exists and does this, but it allocates the whole output densely --
``142,046 x 38,584`` float32 is **21.9 GB**, which does not fit on a 17 GB Mac. This walks
the same path (``predict_step`` on windows of cloned control cells, one perturbation per
window, exactly as ``_infer.py`` does) and writes the result straight out as CSR, so peak
memory is one window plus the control pool.

**The part that is not just a memory rewrite: the count space.** A model trained on
``prep_tx_training.py --log1p`` output consumes and emits the *shifted logarithm*
``log1p(x * L / depth_i)``, where ``L`` is the dataset's median raw depth. cell-eval2 scores
against raw counts and rejects fractional values under ``input_type='counts'``. So
``--x-space log1p`` applies the exact inverse of the prep transform, per cell::

    counts_i = expm1(pred_i) * total_counts_i / L

``total_counts_i`` is the raw depth of the basal cell that prediction was built from, read from
the basal file's own ``obs``; ``L`` is recomputed as the median of that column, which is what
``normalize_total(target_sum=None)`` used.

**And that literal inverse is not safe on its own, which is the trap this file exists to record.**
``expm1`` of an unbounded regression output has no upper bound, and a log-space prediction of 24
is 2.6e10 counts. Measured on PHE-2 arm A: median emitted depth 18,802, **99.9th percentile
38,709,950, maximum 2.4e10** -- 588 of 122,046 cells over a million counts, each one a single gene
taking 75% or more of the cell. A pseudobulk is a sum over cells, so one such cell *is* that
perturbation's profile and the score stops being about the model. Arc's own guard does not catch
it: ``state tx infer`` clips gene-space output to ``[0, 14]``, and at these dimensions that clips
673 values out of 562 million and leaves the 99.9th percentile at 6.4 million.

So ``--depth`` says how the emitted cell gets its library size, and the default is the stable one:

* ``rescale`` (default) -- keep the *composition* ``expm1(pred)`` and pin the cell's total to
  ``total_counts_i``. The depth then comes from the given control cell rather than from an
  exponential, which is what ``count_emitters.ContextProfile`` already does for the statistical
  arms. Uses nothing from the truth.
* ``inverse`` -- the literal ``expm1(pred) * total_counts_i / L``, kept so the instability above
  stays measurable rather than becoming folklore.
* ``size-factor`` -- stop at ``expm1``; every cell sits on ``L`` with no depth variation at all.

Getting the space wrong does not crash: the file scores, confidently and meaninglessly, which is
why ``--x-space`` is explicit and has no default guess.

**Then the emission choice, which is separable from the model.** cell-eval2 refuses fractional
values against ``input_type='counts'``, so the continuous prediction has to become integers.
``--emit round`` (the default) is what PHE-1's first scored run used -- deterministic, and it
cost 0.10% of total counts there. ``--emit poisson`` draws each value instead, preserving the
expectation but injecting cell-to-cell noise. The report records the count loss either way, so
the two are comparable.

The output carries **only** the perturbations the truth file holds, not the model's whole
one-hot vocabulary. A prediction over a wider label set is a different submission as far as
cell-eval2's ``source_fingerprint`` is concerned and will not pair with a bundle built from
the fold -- which is the refusal that cost PHE-1's scoring run a rebuild. The real control
cells are appended verbatim at the end (what ``mirror2026.attach_controls`` does, and what
Arc's platform does), streamed row by row so the 1.5 GB truth file is never held in memory.

Usage::

    uv run python scripts/emit_tx_prediction.py \\
        --model-dir  ~/data/sidechain/runs/tx/phe2_hct116 \\
        --checkpoint ~/data/sidechain/runs/tx/phe2_hct116/checkpoints/last.ckpt \\
        --balanced-encoders --x-space log1p \\
        --basal ~/data/sidechain/derived/tx-train-xatlas-log1p/loco_hct116_real.h5ad \\
        --truth ~/data/sidechain/cache/vcc2026/loco_hct116_real.h5ad \\
        --out   ~/data/sidechain/runs/tx/phe2_hct116/pred_loco_hct116.h5ad
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# The diagnostic already owns the two things a checkpoint reader needs on this Mac: the numpy
# allowlist PyTorch 2.6 made necessary, and the auto device pick that prefers MPS. It sits
# beside this file rather than in the package, so the path goes on first.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_tx_arm import _allow_numpy_globals, _pick_device


def _log(msg: str) -> None:
    print(msg, flush=True)


def read_categorical(f: h5py.File, col: str) -> tuple[np.ndarray, np.ndarray]:
    """(categories, codes) for one categorical obs column."""
    return f[f"obs/{col}/categories"][:].astype(str), f[f"obs/{col}/codes"][:]


def load_rows_csr(f: h5py.File, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR triple (data, indices, indptr) for `rows` of an h5ad CSR group, in the given order."""
    g = f["X"]
    indptr = g["indptr"][:]
    spans = [(int(indptr[r]), int(indptr[r + 1])) for r in rows]
    total = sum(hi - lo for lo, hi in spans)
    data = np.empty(total, dtype=np.float32)
    indices = np.empty(total, dtype=np.int32)
    out_ptr = np.zeros(len(rows) + 1, dtype=np.int64)
    at = 0
    for i, (lo, hi) in enumerate(spans):
        if hi > lo:
            data[at:at + hi - lo] = g["data"][lo:hi]
            indices[at:at + hi - lo] = g["indices"][lo:hi]
            at += hi - lo
        out_ptr[i + 1] = at
    return data, indices, out_ptr


class CsrWriter:
    """Append CSR row blocks to an h5ad-shaped `X` group without holding the matrix."""

    def __init__(self, h5: h5py.File, n_genes: int, chunk: int = 1 << 20):
        g = h5.create_group("X")
        g.attrs["encoding-type"] = "csr_matrix"
        g.attrs["encoding-version"] = "0.1.0"
        self.g = g
        self.n_genes = n_genes
        self.data = g.create_dataset("data", (0,), maxshape=(None,), dtype="float32",
                                     chunks=(chunk,), compression="gzip", compression_opts=1)
        self.indices = g.create_dataset("indices", (0,), maxshape=(None,), dtype="int32",
                                        chunks=(chunk,), compression="gzip", compression_opts=1)
        self.indptr: list[int] = [0]
        self.nnz = 0

    def append(self, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray) -> None:
        n = len(data)
        if n:
            self.data.resize((self.nnz + n,))
            self.data[self.nnz:] = data
            self.indices.resize((self.nnz + n,))
            self.indices[self.nnz:] = indices
        base = self.indptr[-1]
        self.indptr.extend((base + indptr[1:]).tolist())
        self.nnz += n

    def close(self) -> int:
        n_rows = len(self.indptr) - 1
        ptr = np.asarray(self.indptr, dtype=np.int64)
        dtype = "int32" if ptr[-1] < np.iinfo(np.int32).max else "int64"
        self.g.create_dataset("indptr", data=ptr.astype(dtype), compression="gzip", compression_opts=1)
        self.g.attrs["shape"] = np.array([n_rows, self.n_genes], dtype="int64")
        return n_rows


def write_frame(h5: h5py.File, name: str, index: np.ndarray, columns: dict[str, np.ndarray]) -> None:
    """An AnnData `dataframe` group with a string index and categorical string columns."""
    g = h5.create_group(name)
    g.attrs["encoding-type"] = "dataframe"
    g.attrs["encoding-version"] = "0.2.0"
    g.attrs["_index"] = "_index"
    g.attrs["column-order"] = np.array(list(columns), dtype=h5py.string_dtype(encoding="utf-8"))
    _string_array(g, "_index", index)
    for col, values in columns.items():
        cats, codes = np.unique(np.asarray(values, dtype=object), return_inverse=True)
        sub = g.create_group(col)
        sub.attrs["encoding-type"] = "categorical"
        sub.attrs["encoding-version"] = "0.2.0"
        sub.attrs["ordered"] = False
        _string_array(sub, "categories", cats)
        d = sub.create_dataset("codes", data=codes.astype("int32"), compression="gzip",
                               compression_opts=1)
        d.attrs["encoding-type"] = "array"
        d.attrs["encoding-version"] = "0.2.0"


def _string_array(g: h5py.Group, name: str, values: np.ndarray) -> None:
    """A string dataset carrying AnnData's encoding metadata.

    Without the two attributes anndata still reads it, but through its legacy path and with an
    `OldFormatWarning` — a route a later release is free to drop. The file is written by hand
    here (holding the matrix would cost 22 GB), so the metadata has to be written by hand too.
    """
    d = g.create_dataset(name, data=np.asarray(values, dtype=object),
                         dtype=h5py.string_dtype(encoding="utf-8"),
                         compression="gzip", compression_opts=1)
    d.attrs["encoding-type"] = "string-array"
    d.attrs["encoding-version"] = "0.2.0"


def to_counts(preds: np.ndarray, depths: np.ndarray, size_factor: float, x_space: str,
              depth_mode: str) -> np.ndarray:
    """Predictions in the model's own space -> raw-count scale, per cell.

    `log1p` inverts `prep_tx_training.py --log1p`: that script divided each row by `depth_i / L`
    and took `log1p`. `inverse` undoes exactly that and is unstable, because `expm1` of an
    unbounded prediction is unbounded; `rescale` keeps the same composition and pins each cell's
    total to its basal cell's raw depth instead. See the module docstring for the measurement.
    """
    if x_space != "log1p":
        return preds
    counts = np.expm1(preds)
    if depth_mode == "inverse":
        counts *= (depths / size_factor)[:, None]
    elif depth_mode == "rescale":
        total = counts.sum(1, keepdims=True)
        counts = counts / np.maximum(total, 1e-12) * depths[:, None]
    return counts


def emit(counts: np.ndarray, how: str, rng: np.random.Generator) -> np.ndarray:
    """Continuous count-scale values -> integers, because cell-eval2 refuses fractions."""
    counts = np.clip(counts, 0.0, None)
    if how == "round":
        return np.rint(counts)
    return rng.poisson(counts).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, help="default: <model-dir>/checkpoints/last.ckpt")
    ap.add_argument("--basal", required=True, type=Path,
                    help="h5ad supplying the control cells the model consumes, in the space it "
                         "was trained on, plus obs/total_counts (each cell's RAW depth)")
    ap.add_argument("--truth", required=True, type=Path,
                    help="the fold's raw-count h5ad: it names the perturbations to emit, how "
                         "many cells each gets, the gene axis, and the control cells appended")
    ap.add_argument("--x-space", choices=("counts", "log1p"), default="counts",
                    help="what the model consumes and emits; `log1p` inverts the shifted "
                         "logarithm per cell before anything is written")
    ap.add_argument("--emit", choices=("round", "poisson"), default="round")
    ap.add_argument("--depth", choices=("rescale", "inverse", "size-factor"), default="rescale",
                    help="with --x-space log1p: how the emitted cell gets its library size. "
                         "`rescale` pins it to the basal cell's raw depth (stable); `inverse` is "
                         "the literal expm1 inverse (unbounded); `size-factor` leaves every cell "
                         "on L")
    ap.add_argument("--pert-col", default="gene_target", help="column in --basal")
    ap.add_argument("--control-pert", default="Non-Targeting", help="control label in --basal")
    ap.add_argument("--truth-pert-col", default="perturbation", help="column in --truth")
    ap.add_argument("--truth-control", default="non-targeting", help="control label in --truth")
    ap.add_argument("--balanced-encoders", action="store_true",
                    help="the checkpoint was trained with sidechain.models.balanced_encoders")
    ap.add_argument("--max-set-len", type=int, default=0, help="0 = the model's own cell_set_len")
    ap.add_argument("--cells-per-pert", type=int, default=0,
                    help="0 = match the truth file's own cells-per-label")
    ap.add_argument("--n-perts", type=int, default=0, help="0 = every perturbation in --truth")
    ap.add_argument("--no-controls", action="store_true",
                    help="skip appending the real control cells (mirror2026 --attach-controls "
                         "would then have to add them)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    model_dir = args.model_dir.expanduser()
    ckpt = (args.checkpoint or model_dir / "checkpoints" / "last.ckpt").expanduser()
    basal_path, truth_path, out_path = args.basal.expanduser(), args.truth.expanduser(), args.out.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    _allow_numpy_globals()
    pert_map = torch.load(model_dir / "pert_onehot_map.pt", weights_only=False)
    with open(model_dir / "var_dims.pkl", "rb") as f:
        var_dims = pickle.load(f)
    key_of = {str(k): k for k in pert_map}
    _log(f"model dir     {model_dir}")
    _log(f"checkpoint    {ckpt.name}")
    _log(f"one-hot map   {len(key_of)} entries")

    # ---- what to emit, and on how many cells: the TRUTH decides both ---------------------
    with h5py.File(truth_path, "r") as t:
        t_cats, t_codes = read_categorical(t, args.truth_pert_col)
        genes = t["var/_index"][:].astype(str)
        counts_per_label = np.bincount(t_codes, minlength=len(t_cats))
    if args.truth_control not in set(t_cats):
        raise SystemExit(f"control {args.truth_control!r} not in {truth_path.name}/{args.truth_pert_col}")
    targets = [c for c in t_cats if c != args.truth_control]
    if args.n_perts > 0:
        targets = targets[: args.n_perts]
    missing = [p for p in targets if p not in key_of]
    if missing:
        raise SystemExit(f"{len(missing)} truth perturbations are not in the model's map, "
                         f"e.g. {missing[:5]} -- state tx infer would silently substitute the "
                         f"control one-hot for these")
    n_cells = {p: (args.cells_per_pert or int(counts_per_label[list(t_cats).index(p)])) for p in targets}
    _log(f"truth         {truth_path.name}: {len(targets)} targets, "
         f"{sum(n_cells.values()):,} perturbed cells to emit, {len(genes):,} genes")

    # ---- the basal pool ------------------------------------------------------------------
    with h5py.File(basal_path, "r") as b:
        b_cats, b_codes = read_categorical(b, args.pert_col)
        if args.control_pert not in set(b_cats):
            raise SystemExit(f"control {args.control_pert!r} not in {basal_path.name}/{args.pert_col}")
        ctrl_rows = np.where(b_codes == int(np.where(b_cats == args.control_pert)[0][0]))[0]
        if int(b["X"].attrs["shape"][1]) != len(genes):
            raise SystemExit("basal and truth gene axes differ in length")
        depths_all = b["obs/total_counts"][:].astype(np.float64)
        size_factor = float(np.median(depths_all))
        pool_data, pool_indices, pool_ptr = load_rows_csr(b, ctrl_rows)
    pool_depths = depths_all[ctrl_rows]
    _log(f"basal pool    {len(ctrl_rows):,} control cells, {len(pool_data):,} nonzeros, "
         f"raw depth median {np.median(pool_depths):,.0f}")
    _log(f"size factor   {size_factor:,.1f} (median raw depth; what --log1p normalised to)")
    if len(genes) != int(var_dims["input_dim"]):
        raise SystemExit(f"gene axis {len(genes)} != model input_dim {var_dims['input_dim']}")

    def pool_dense(rows: np.ndarray) -> np.ndarray:
        out = np.zeros((len(rows), len(genes)), dtype=np.float32)
        for i, r in enumerate(rows):
            lo, hi = int(pool_ptr[r]), int(pool_ptr[r + 1])
            out[i, pool_indices[lo:hi]] = pool_data[lo:hi]
        return out

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
    set_len = args.max_set_len or int(getattr(model, "cell_set_len", 512))
    _log(f"device        {device}, window {set_len} cells")

    # ---- the forward passes, one homogeneous window at a time ---------------------------
    out_h5 = h5py.File(out_path, "w")
    out_h5.attrs["encoding-type"] = "anndata"
    out_h5.attrs["encoding-version"] = "0.1.0"
    writer = CsrWriter(out_h5, len(genes))
    labels: list[str] = []
    emitted_depth: list[float] = []
    norm_depth: list[float] = []
    raw_total = 0.0
    emitted_total = 0.0
    _log("")
    with torch.inference_mode():
        for i, p in enumerate(targets):
            want = n_cells[p]
            take = rng.choice(len(ctrl_rows), size=want, replace=want > len(ctrl_rows))
            vec = pert_map[key_of[p]].float().to(device)
            for lo in range(0, want, set_len):
                idx = take[lo:lo + set_len]
                X = pool_dense(idx)
                batch = {
                    "ctrl_cell_emb": torch.tensor(X, device=device),
                    "pert_emb": vec.unsqueeze(0).repeat(len(idx), 1),
                    "pert_name": [p] * len(idx),
                }
                preds = model.predict_step(batch, batch_idx=0, padded=False)["preds"]
                preds = preds.float().cpu().numpy()
                del batch
                counts = to_counts(preds, pool_depths[idx], size_factor, args.x_space, args.depth)
                if args.x_space == "log1p":
                    norm_depth.extend(np.expm1(preds).sum(1).tolist())
                raw_total += float(counts.sum())
                block = emit(counts, args.emit, rng)
                emitted_total += float(block.sum())
                emitted_depth.extend(block.sum(1).tolist())
                # np.nonzero returns row-major order, which is exactly CSR order.
                r_i, c_i = np.nonzero(block)
                writer.append(block[r_i, c_i].astype(np.float32), c_i.astype(np.int32),
                              np.concatenate([[0], np.cumsum(np.bincount(r_i, minlength=len(idx)))]))
                labels.extend([p] * len(idx))
            if device.type == "mps" and (i + 1) % 25 == 0:
                torch.mps.empty_cache()
            if (i + 1) % 50 == 0 or i + 1 == len(targets):
                _log(f"    {i + 1}/{len(targets)} perturbations, {len(labels):,} cells, "
                     f"{writer.nnz:,} nonzeros")
    n_pred = len(labels)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    # ---- the real control cells, streamed straight across --------------------------------
    n_ctrl = 0
    if not args.no_controls:
        with h5py.File(truth_path, "r") as t:
            _, t_codes = read_categorical(t, args.truth_pert_col)
            rows = np.where(t_codes == int(np.where(t_cats == args.truth_control)[0][0]))[0]
            for lo in range(0, len(rows), 2000):
                d, ind, ptr = load_rows_csr(t, rows[lo:lo + 2000])
                writer.append(d, ind, ptr)
            n_ctrl = len(rows)
        labels.extend([args.truth_control] * n_ctrl)
        _log(f"controls      {n_ctrl:,} real cells appended verbatim")

    n_rows = writer.close()
    assert n_rows == len(labels), f"{n_rows} rows written, {len(labels)} labels"
    write_frame(out_h5, "obs", np.array([f"pred_{i}" for i in range(n_rows)], dtype=object),
                {args.truth_pert_col: np.array(labels, dtype=object)})
    write_frame(out_h5, "var", np.asarray(genes, dtype=object), {})
    out_h5.close()

    depth = np.asarray(emitted_depth)
    report = {
        "model_dir": str(model_dir), "checkpoint": str(ckpt), "basal": str(basal_path),
        "truth": str(truth_path), "out": str(out_path), "x_space": args.x_space,
        "emit": args.emit, "depth": args.depth,
        "balanced_encoders": bool(args.balanced_encoders),
        "device": str(device), "window": set_len, "seed": args.seed,
        "size_factor": size_factor, "n_perturbations": len(targets),
        "n_predicted_cells": n_pred, "n_control_cells": n_ctrl, "n_rows": n_rows,
        "nonzeros": int(writer.nnz),
        "count_loss_frac": (raw_total - emitted_total) / raw_total if raw_total else None,
        "predicted_depth_median": float(np.median(depth)) if len(depth) else None,
        "predicted_depth_mean": float(depth.mean()) if len(depth) else None,
        "predicted_depth_before_rescale_median": (
            float(np.median(norm_depth)) if norm_depth else None),
        "predicted_nnz_per_cell_median": float(np.median(np.diff(np.asarray(writer.indptr[:n_pred + 1])))),
    }
    (out_path.with_suffix(".json")).write_text(json.dumps(report, indent=1) + "\n")
    _log("")
    _log(f"wrote         {out_path}  ({n_rows:,} cells x {len(genes):,} genes, "
         f"{writer.nnz:,} nonzeros)")
    _log(f"emission      {args.emit}: {report['count_loss_frac'] * 100:+.3f}% of total counts")
    _log(f"depth         predicted median {report['predicted_depth_median']:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
