"""A context's own control-cell axes, projected out of each target's predicted summed profile.

    --ctrl-axes K [--ctrl-axes-eta ETA --ctrl-axes-basis {pca,random} --ctrl-axes-share F]   (submit.build, eval.loco)

Reads the context's control cells (h5ad); writes the axes to `--ctrl-axes-cache` (an .npz per input sha256).
Notes: private/code-notes/src/sidechain/submit/ctrl_axes.py.md
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

BULK_TARGET_SUM = 50_000.0     # cell-eval2's summed-profile space (bulk_lognorm), also each cell's here
N_COMPUTE = 100                # axes computed per context; K reads the first K
OVERSAMPLE = 20
POWER_ITERATIONS = 12
SEED = 0
TOLERANCE = 1e-6               # emitted along-axis part against its goal, relative to the removed part
MAX_CORRECTIONS = 12
LOG2FC_FLOOR = -10.0
VERSION = "ctrl_axes/1"        # in the cache key: a change to the arithmetic here is another cache
BASES = ("pca", "random")
DEFAULT_CACHE = Path("~/data/sidechain/cache/ctrl_axes")


def add_ctrl_axes_args(ap, *, twin: str) -> None:
    ap.add_argument("--ctrl-axes", type=int, default=0, metavar="K",
                    help="remove the context's top K control-cell axes (log1p CP50K, centred, from the "
                         "context's own control cells) from each target's predicted summed profile; the "
                         "per-cell channel is untouched. 0 (the default) is off. Needs --bulk-anchor pooled. "
                         f"Recorded with the axes' hash (sidechain.submit.ctrl_axes; the same flag on {twin})")
    ap.add_argument("--ctrl-axes-eta", type=float, default=1.0, metavar="ETA",
                    help="with --ctrl-axes: the share of the along-axis part removed, in (0, 1] (default 1)")
    ap.add_argument("--ctrl-axes-basis", choices=list(BASES), default="pca",
                    help="with --ctrl-axes: 'pca', the controls' principal axes, or 'random', K orthonormal "
                         "axes weighted by each gene's control variance")
    ap.add_argument("--ctrl-axes-share", type=float, default=1.0, metavar="F",
                    help="with --ctrl-axes: deflate only the share F of targets whose summed-profile change "
                         "is smallest (in (0, 1]; default 1, every target); the rest keep their summed profile")
    ap.add_argument("--ctrl-axes-cache", type=Path, default=DEFAULT_CACHE, metavar="DIR",
                    help="with --ctrl-axes: where the axes are cached by their inputs' sha256 "
                         f"(default {DEFAULT_CACHE})")


def check_ctrl_axes_args(ap, args) -> None:
    on = args.ctrl_axes != 0
    if args.ctrl_axes < 0:
        ap.error(f"--ctrl-axes is a count of axes >= 0, got {args.ctrl_axes}")
    if not on and (args.ctrl_axes_eta != 1.0 or args.ctrl_axes_basis != "pca" or args.ctrl_axes_share != 1.0):
        ap.error("--ctrl-axes-eta, --ctrl-axes-basis and --ctrl-axes-share act only with --ctrl-axes K > 0")
    if on and not 0.0 < args.ctrl_axes_eta <= 1.0:
        ap.error(f"--ctrl-axes-eta must be in (0, 1], got {args.ctrl_axes_eta}")
    if on and not 0.0 < args.ctrl_axes_share <= 1.0:
        ap.error(f"--ctrl-axes-share must be in (0, 1], got {args.ctrl_axes_share}")
    if on and args.bulk_anchor != "pooled":
        ap.error("--ctrl-axes acts on the summed-profile channel of the pooled anchor: pass --bulk-anchor pooled")


def _kept_cells(path: Path, min_libsize: float):
    a = ad.read_h5ad(path)
    X = sp.csr_matrix(a.X, dtype=np.float32)
    lib = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()
    keep = lib > max(min_libsize, 0)
    if not keep.any():
        raise ValueError(f"{path}: every control cell is at or below min_libsize={min_libsize:g}")
    genes = a.var_names.astype(str).to_numpy()
    del a
    if not keep.all():
        X = X[keep]
    X.eliminate_zeros()
    return X, lib[keep], genes


def _cache_key(X: sp.csr_matrix, genes: np.ndarray, min_libsize: float, basis: str, k: int) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"version": VERSION, "basis": basis, "k": k, "min_libsize": float(min_libsize),
                         "oversample": OVERSAMPLE, "power_iterations": POWER_ITERATIONS, "seed": SEED,
                         "target_sum": BULK_TARGET_SUM, "shape": list(X.shape)}, sort_keys=True).encode())
    h.update("\n".join(genes.tolist()).encode())
    for arr in (X.indptr.astype(np.int64), X.indices.astype(np.int32), X.data.astype(np.float32)):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _lognorm_inplace(X: sp.csr_matrix, lib: np.ndarray) -> sp.csr_matrix:
    for lo in range(0, X.shape[0], 2000):                 # rows in blocks: memory
        s, e = X.indptr[lo], X.indptr[min(lo + 2000, X.shape[0])]
        scale = np.repeat((BULK_TARGET_SUM / lib[lo:lo + 2000]).astype(np.float32),
                          np.diff(X.indptr[lo:lo + 2001]))
        X.data[s:e] = np.log1p(X.data[s:e] * scale)
    return X


def _column_moments(Y: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(Y.shape[1])
    square = np.zeros(Y.shape[1])
    for lo in range(0, Y.shape[0], 2000):
        s, e = Y.indptr[lo], Y.indptr[min(lo + 2000, Y.shape[0])]
        v = Y.data[s:e].astype(np.float64)
        total += np.bincount(Y.indices[s:e], weights=v, minlength=Y.shape[1])
        square += np.bincount(Y.indices[s:e], weights=v * v, minlength=Y.shape[1])
    n = Y.shape[0]
    mu = total / n
    var = np.maximum(square / n - mu * mu, 0.0) * n / max(n - 1, 1)
    return mu, var


def _centred_rsvd(Y: sp.csr_matrix, mu: np.ndarray, k: int, rng: np.random.Generator):
    """Top-k right singular vectors of Y - 1 mu^T, with the centred matrix never formed."""
    n, g = Y.shape
    mu32 = mu.astype(np.float32)

    def right(M):                      # (Y - 1 mu^T) M
        return Y @ M - (mu32 @ M)[None, :]

    def left(M):                       # (Y - 1 mu^T)^T M
        return Y.T @ M - np.outer(mu32, M.sum(axis=0))

    q, _ = np.linalg.qr(right(rng.standard_normal((g, min(k + OVERSAMPLE, n, g))).astype(np.float32)))
    for _ in range(POWER_ITERATIONS):
        z, _ = np.linalg.qr(left(q))
        q, _ = np.linalg.qr(right(z))
    _, s, vt = np.linalg.svd(left(q).T.astype(np.float64), full_matrices=False)
    vt = vt[:k]
    vt *= np.sign(vt[np.arange(k), np.abs(vt).argmax(axis=1)])[:, None]     # one sign per axis
    return s[:k] / np.sqrt(max(n - 1, 1)), vt


def control_axes(path: Path, *, k: int, min_libsize: float, basis: str = "pca",
                 cache_dir: Path | None = DEFAULT_CACHE, genes=None) -> tuple[np.ndarray, dict]:
    """(K, G) orthonormal axes of the controls in `path`, and their record. `genes` checks the axis."""
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    path = Path(path).expanduser()
    X, lib, file_genes = _kept_cells(path, min_libsize)
    if genes is not None and list(np.asarray(genes, dtype=str)) != list(file_genes):
        raise ValueError(f"{path}: the controls' genes differ from the build's axis")
    n_compute = min(max(N_COMPUTE, k), X.shape[0] - 1, X.shape[1])
    if k > n_compute:
        raise ValueError(f"{path}: {k} axes asked of {X.shape[0]} cells x {X.shape[1]} genes")
    key = _cache_key(X, file_genes, min_libsize, basis, n_compute)
    cached = None if cache_dir is None else Path(cache_dir).expanduser() / f"{key}.npz"
    if cached is not None and cached.exists():
        z = np.load(cached)
        axes, sd_axes, total_var = z["axes"], z["sd_axes"], float(z["total_var"])
    else:
        Y = _lognorm_inplace(X, lib)                    # X is spent: the key was read from it above
        n = Y.shape[0]
        mu, var = _column_moments(Y)
        total_var = float(var.sum())
        rng = np.random.default_rng(SEED)
        if basis == "pca":
            sd_axes, axes = _centred_rsvd(Y, mu, n_compute, rng)
        else:
            draw = rng.standard_normal((len(mu), n_compute)) * np.sqrt(var)[:, None]
            axes = np.linalg.qr(draw)[0].T
            proj = Y @ axes.T.astype(np.float32) - (mu.astype(np.float32) @ axes.T.astype(np.float32))[None, :]
            sd_axes = np.sqrt((np.asarray(proj, dtype=np.float64) ** 2).sum(axis=0) / max(n - 1, 1))
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            # one temp file per writer: arms sharing a cache may compute the same key side by side
            tmp = cached.with_name(f"{cached.stem}.{os.getpid()}.{uuid.uuid4().hex}.partial.npz")
            np.savez(tmp, axes=axes, sd_axes=sd_axes, total_var=total_var)
            tmp.replace(cached)
    axes = np.asarray(axes[:k], dtype=np.float64)
    var_axes = np.asarray(sd_axes[:k], dtype=np.float64) ** 2
    record = {"basis": basis, "k": int(k), "n_computed": int(n_compute), "key": key,
              "cache": None if cached is None else str(cached), "control_cells": int(len(lib)),
              "genes": int(axes.shape[1]), "variance_share_on_axes": float(var_axes.sum() / total_var),
              "variance_share_first_axis": float(var_axes[0] / total_var),
              "oversample": OVERSAMPLE, "power_iterations": POWER_ITERATIONS, "seed": SEED,
              "version": VERSION}
    return axes, record


def scorer_change(log2fc: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """The summed-profile change cell-eval2 reads for a pseudobulk emitted at `anchor * 2^log2fc`."""
    p = anchor * np.exp2(np.nan_to_num(log2fc, nan=0.0, posinf=0.0, neginf=0.0))
    p = p / p.sum()
    return np.log1p(BULK_TARGET_SUM * p) - np.log1p(BULK_TARGET_SUM * anchor)


def deflate_bulk(log2fc: np.ndarray, anchor: np.ndarray, axes: np.ndarray, eta: float = 1.0, *,
                 held=(), corrections: int = MAX_CORRECTIONS) -> tuple[np.ndarray, dict]:
    """`log2fc` with `eta` of its along-axis summed-profile change removed, Newton-corrected, and a record."""
    l0 = np.asarray(log2fc, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if l0.shape != anchor.shape or axes.shape[1] != l0.shape[0]:
        raise ValueError("log2fc, anchor and the axes must share one gene axis")
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0, 1], got {eta}")
    if eta == 0.0 or axes.shape[0] == 0:
        return l0.copy(), {"energy": None, "share_on_axes": None, "residual": 0.0, "corrections": 0,
                           "floored": 0}
    free = anchor > 0
    free[np.asarray(list(held), dtype=np.int64)] = False
    V = axes * free[None, :]                       # held and unseen genes take no part
    l0n = np.nan_to_num(l0, nan=0.0, posinf=0.0, neginf=0.0)
    logz = np.log2(np.sum(anchor * np.exp2(l0n)))
    lc = np.log1p(BULK_TARGET_SUM * anchor[free])
    base = BULK_TARGET_SUM * anchor[free]
    floor = np.minimum(l0n[free], LOG2FC_FLOOR)

    def back(D):                                   # a summed-profile change -> a log2FC on `free`, floored
        q = np.expm1(lc + D[free])
        out = np.full(len(q), -np.inf)
        pos = q > 0
        out[pos] = np.log2(q[pos] / base[pos]) + logz
        return np.maximum(out, floor), int(np.sum(out < floor))

    D0 = scorer_change(l0n, anchor)
    a0 = V @ D0
    goal = (1.0 - eta) * a0
    removed = float(np.linalg.norm(a0 - goal))
    D = D0 - V.T @ (a0 - goal)
    z0 = 2.0 ** logz
    l = l0n.copy()
    resid, first, n_corr, floored = np.inf, None, 0, 0
    Vf = V[:, free]
    for n_corr in range(corrections + 1):
        l[free], floored = back(D)
        err = V @ scorer_change(l, anchor) - goal
        resid = float(np.linalg.norm(err)) / removed if removed > 0 else 0.0
        first = resid if first is None else first
        if resid <= TOLERANCE or n_corr == corrections:
            break
        # the emitted change is D plus log(c) times w, c the renormalisation's factor
        c = z0 / np.sum(anchor * np.exp2(l))
        q = np.expm1(lc + D[free])
        u = Vf @ (c * q / (1.0 + c * q))
        g = (c / BULK_TARGET_SUM) * (Vf @ (1.0 + q))
        D = D - V.T @ (err + u * (g @ err) / (1.0 - g @ u))
    l[~free] = l0[~free]
    energy = float(D0[free] @ D0[free])
    return l, {"energy": energy, "share_on_axes": float(a0 @ a0) / energy if energy > 0 else 0.0,
               "residual": resid, "one_shot_residual": first, "corrections": int(n_corr), "floored": floored}


def change_size(log2fc: np.ndarray, anchor: np.ndarray, held=()) -> float:
    """How large a target's summed-profile change is, as the scorer reads it, its own gene left out."""
    D = scorer_change(np.asarray(log2fc, dtype=np.float64), anchor)
    keep = anchor > 0
    keep[np.asarray(list(held), dtype=np.int64)] = False
    return float(np.sqrt(D[keep] @ D[keep]))


def weakest(sizes: dict, share: float) -> set:
    """The targets whose change is at or below the `share` quantile of `sizes`; every target at share 1."""
    if share >= 1.0 or not sizes:
        return set(sizes)
    cut = np.quantile(np.array(list(sizes.values()), dtype=float), share)
    return {p for p, v in sizes.items() if v <= cut}


def deflate_map(bulk: dict, anchor: np.ndarray, axes: np.ndarray, eta: float,
                gene_pos: dict, share: float = 1.0) -> tuple[dict, dict]:
    """`deflate_bulk` over a {target: log2FC or None} map, each target's own gene held; a new map, and a summary.

    With `share` below 1 only the `weakest` targets are deflated; the others keep their vector.
    """
    def held(p):
        return () if gene_pos.get(p) is None else (gene_pos[p],)
    chosen = weakest({p: change_size(v, anchor, held(p)) for p, v in bulk.items() if v is not None}, share)
    out, per = {}, {}
    for p, v in bulk.items():
        if v is None or p not in chosen:
            out[p] = v
            continue
        out[p], per[p] = deflate_bulk(v, anchor, axes, eta, held=held(p))
    left = sum(1 for p, v in bulk.items() if v is not None and p not in chosen)
    return out, {**summarize(per), "share": share, "targets_left_as_they_were": left}


def summarize(per: dict) -> dict:
    """What the projection did over targets: the share of each change on the axes, the fit's residual."""
    if not per:
        return {"targets": 0}
    share = np.array([r["share_on_axes"] for r in per.values()], dtype=float)
    resid = np.array([r["residual"] for r in per.values()], dtype=float)
    one = np.array([r.get("one_shot_residual") or 0.0 for r in per.values()], dtype=float)
    return {"targets": len(per),
            "share_on_axes": {"median": float(np.median(share)), "min": float(share.min()),
                              "max": float(share.max())},
            "one_shot_residual_median": float(np.median(one)),
            "residual_max": float(resid.max()),
            "corrections_max": int(max(r["corrections"] for r in per.values())),
            "targets_not_converged": sorted(p for p, r in per.items() if r["residual"] > TOLERANCE),
            "floored_genes": int(sum(r["floored"] for r in per.values()))}
