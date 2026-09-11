"""Raw `pds_cosine` from a pooled delta, with no cells emitted and no scoring run.

**Why this is exact rather than a proxy.** Under `--dispersion even` the emitter lays
down a deterministic per-target group SUM -- `rint(n_p * median_libsize * frac_p)` --
and the RNG only chooses which cell carries the remainder
(`sidechain.models.count_emitters.PoissonEmitter._emit_even`). cell-eval2's
`pds_cosine` reads the group sums through `bulk_lognorm` and nothing else. So the
metric is computable from the delta alone, and the arithmetic here is cell-eval2's own
kernel, not a reimplementation of it.

    from sidechain.eval.analytic_pds import prep_fold, score_delta
    fold = prep_fold(real_h5ad, pert_col="perturbation")
    raw  = score_delta(deltas, targets, fold, alpha=1.35)

**Three boundaries, and they are not optional reading** (`RESULTS.md` § `T58`):

1. **`pds_cosine` only.** The other five members read emitted cells through a Wilcoxon
   test. Whether `expr_mse_unbiased_capped_norm` also survives is an open measurement --
   the bundle's estimator labels cannot settle it, because nine of the ten members carry
   `split_half_raw`, `pds_cosine` among them.
2. **`--dispersion even` (lambda = 0) only.** At any lambda > 0 a Poisson share is drawn
   and the group sum is exact only in expectation, so the identity BREAKS. This does not
   accelerate a lambda sweep (`T32`); it is the boundary a caller crosses by accident.
3. **No scoreable artifact.** There is no `run_meta.json`, no `config_digest` and no
   environment record here, and cell-eval2's `score` refuses a run whose identity differs
   from its bundle's. This is a SEARCH instrument: it ranks candidates cheaply, and
   whatever it selects still needs a real mirror run before the number is quoted.

Shipped under `T59` from the instrument `T58` built and `T60` exercised; module path and
public names fixed by session `271a46a8` (private `84570bd`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
from anndata.io import read_elem
from cell_eval2.metrics.discrimination import discrimination_score
from cell_eval2.prep import bulk_lognorm_means

BULK_TARGET_SUM = 50_000.0
PERT_COL = "perturbation"
CONTROL = "non-targeting"
KNOCKDOWN_LOG2FC = -2.32          # what the emitter pins the target's own gene to
MIN_LIBSIZE = 500.0               # `eval.loco`'s default control-cell floor

__all__ = ["FoldCache", "prep_fold", "emitted_sums", "pds_cosine", "score_delta",
           "pool_parts", "group_sums"]


@dataclass(frozen=True)
class FoldCache:
    """Everything the analytic path needs from one held-out line's real cells."""

    perts: np.ndarray            # every label, control included, sorted
    real_means: np.ndarray       # [L, G] bulk_lognorm means -- the pds reference
    genes: np.ndarray            # [G] the fold's gene axis, in file order
    n_cells: np.ndarray          # [L] cells per label
    frac: np.ndarray             # [G] control profile, sums to 1
    lib_median: float            # median control library size
    ctrl_n_cells: int

    def cells_for(self, targets) -> np.ndarray:
        n = dict(zip([str(p) for p in self.perts], self.n_cells))
        missing = [t for t in targets if t not in n]
        if missing:
            raise KeyError(f"{len(missing)} targets absent from the fold: {missing[:5]}")
        return np.array([n[t] for t in targets], dtype=np.int64)


def group_sums(path, pert_col: str = PERT_COL, block_rows: int = 4000):
    """(labels sorted, [L,G] float64 count sums, genes) streamed from a CSR h5ad."""
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        genes = read_elem(f["var"]).index.astype(str).to_numpy()
        labels = obs[pert_col].astype(str).to_numpy()
        wanted = np.array(sorted(set(labels)))
        code_of = {lab: i for i, lab in enumerate(wanted)}
        codes = np.array([code_of[lab] for lab in labels], dtype=np.int64)
        X = f["X"]
        n, g = (int(v) for v in X.attrs["shape"])
        if g != len(genes):
            raise ValueError(f"X has {g} columns against {len(genes)} var rows")
        indptr = X["indptr"][:]
        out = np.zeros((len(wanted), g))
        for r0 in range(0, n, block_rows):
            r1 = min(n, r0 + block_rows)
            s, e = int(indptr[r0]), int(indptr[r1])
            blk = sp.csr_matrix(
                (X["data"][s:e].astype(np.float64), X["indices"][s:e], indptr[r0:r1 + 1] - s),
                shape=(r1 - r0, g))
            c = codes[r0:r1]
            ind = sp.csr_matrix((np.ones(len(c)), (c, np.arange(len(c)))),
                                shape=(len(wanted), len(c)))
            out += (ind @ blk).toarray()
    return wanted, out, genes


def _control_profile(path, obs_labels, genes, control, min_libsize, block=2000):
    """The emitter's own anchor: mean per-cell CPM over the fold's control cells.

    Streamed in blocks rather than loaded: the X-Atlas folds run ~800 M nonzeros and a
    single slice of 20,000 x 38,584 peaks past what a 17 GB Mac has spare.
    """
    rows = np.where(np.asarray(obs_labels) == control)[0]
    if rows.size == 0:
        raise ValueError(f"no cells labelled {control!r}")
    cpm_sum = np.zeros(len(genes))
    libs = []
    with h5py.File(path, "r") as f:
        X = f["X"]
        indptr = X["indptr"][:]
        for lo in range(0, len(rows), block):
            chunk = rows[lo:lo + block]
            r0, r1 = int(chunk[0]), int(chunk[-1]) + 1
            s, e = int(indptr[r0]), int(indptr[r1])
            blk = sp.csr_matrix(
                (X["data"][s:e].astype(np.float64), X["indices"][s:e], indptr[r0:r1 + 1] - s),
                shape=(r1 - r0, len(genes)))[chunk - r0]
            lib = np.asarray(blk.sum(axis=1)).ravel()
            keep = lib > min_libsize
            blk, lib = blk[keep], lib[keep]
            if lib.size:
                cpm_sum += np.asarray((sp.diags(1e6 / lib) @ blk).sum(axis=0)).ravel()
                libs.append(lib)
    libs = np.concatenate(libs) if libs else np.array([])
    if libs.size == 0:
        raise ValueError(f"every control cell fell below min_libsize={min_libsize}")
    mean_cpm = cpm_sum / len(libs)
    return mean_cpm / mean_cpm.sum(), float(np.median(libs)), int(len(libs))


def prep_fold(path, pert_col: str = PERT_COL, control: str = CONTROL,
              min_libsize: float = MIN_LIBSIZE, cache: Path | None = None) -> FoldCache:
    """Build (or load) everything the analytic path needs from a fold's real h5ad.

    `T58` shipped the scorer without this, which left every caller re-deriving
    `real_means`, the control profile and `n_cells` by hand. Pass `cache` to write an
    `.npz` beside your run and skip the streaming pass next time.
    """
    path = Path(path)
    if cache is not None and Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        return FoldCache(z["perts"], z["real_means"], z["genes"].astype(str), z["n_cells"],
                         z["frac"], float(z["lib_median"]), int(z["ctrl_n_cells"]))

    labels, sums, genes = group_sums(path, pert_col=pert_col)
    real_means = bulk_lognorm_means(sums, BULK_TARGET_SUM)
    with h5py.File(path, "r") as f:
        obs_labels = read_elem(f["obs"])[pert_col].astype(str).to_numpy()
    if control not in set(labels):
        raise ValueError(f"control {control!r} not among the fold's labels")
    n_cells = np.array([(obs_labels == lab).sum() for lab in labels], dtype=np.int64)
    frac, lib_median, ctrl_n = _control_profile(path, obs_labels, genes, control, min_libsize)

    fold = FoldCache(labels.astype(object), real_means, genes.astype(str), n_cells,
                     frac, lib_median, ctrl_n)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, perts=fold.perts, real_means=fold.real_means,
                            genes=fold.genes.astype(object), n_cells=fold.n_cells,
                            frac=fold.frac, lib_median=fold.lib_median,
                            ctrl_n_cells=fold.ctrl_n_cells)
    return fold


def emitted_sums(deltas, frac, lib_median, n_cells):
    """The `even` emitter's per-target group sums, analytically.

    `deltas` is [P, G] log2 fold change on the fold's axis, `frac` the control profile.
    """
    d = np.nan_to_num(np.asarray(deltas, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    f = frac[None, :] * np.exp2(d)
    f = f / f.sum(axis=1, keepdims=True)
    return np.rint(np.asarray(n_cells, dtype=np.float64)[:, None] * lib_median * f)


def pds_cosine(pred_perts, pred_sums, real_perts, real_means, genes):
    """Raw `pds_cosine` through cell-eval2's own kernel, at the vcc2026 preset's settings."""
    pred_means = bulk_lognorm_means(np.asarray(pred_sums, dtype=np.float64), BULK_TARGET_SUM)
    out = discrimination_score(
        pred_bulk=(np.asarray(pred_perts, dtype=str), pred_means),
        real_bulk=(np.asarray(real_perts, dtype=str), real_means),
        pert_col=PERT_COL, control=CONTROL, distance="cosine",
        rank_denominator="n-1", tie_policy="midrank", exclude_target_gene=True,
        exclusion_scope="panel", control_source="real", genes=np.asarray(genes, dtype=str),
    )
    return float(np.mean(list(out.values()))) if isinstance(out, dict) else float(out)


def score_delta(deltas, targets, fold: FoldCache, alpha: float = 1.0,
                kd_value: float = KNOCKDOWN_LOG2FC, which=None, covered=None) -> float:
    """Raw `pds_cosine` for a [P, G] log2FC matrix. `deltas` is never mutated.

    `alpha` and the knockdown pin are applied here so a caller passes the pooled delta
    exactly as `submit.build.pooled_delta` returns it. `which` scores a subset of rows
    against the full retrieval pool.

    **`covered` is not optional bookkeeping -- it changes the number.** `eval.loco` pins
    the knockdown INSIDE `if d is not None`, so a target no source covers is emitted as
    the bare control profile with NO pin. Pinning it anyway renormalises all G
    coordinates for that target, not just the pinned gene. On `loco_hct116/afn_nosib`
    (802 of 830 covered) that one difference is the whole 4.4e-06 replay residual:
    pinning everything reads +4.359e-06 against the recorded value, honouring coverage
    reads -7.4e-10. Pass a boolean mask whenever any target may be uncovered; None means
    every target is covered.
    """
    d = np.asarray(deltas, dtype=np.float64) * alpha
    pos = {g: i for i, g in enumerate(fold.genes)}
    cov = np.ones(len(targets), dtype=bool) if covered is None else np.asarray(covered, dtype=bool)
    for i, t in enumerate(targets):
        j = pos.get(str(t))
        if j is not None and cov[i]:
            d[i, j] = kd_value
    idx = np.arange(len(targets)) if which is None else np.asarray(which)
    sums = emitted_sums(d[idx], fold.frac, fold.lib_median, fold.cells_for(targets)[idx])
    return pds_cosine([str(targets[i]) for i in idx], sums,
                      np.asarray(fold.perts, dtype=str), fold.real_means, fold.genes)


def pool_parts(targets, sources, axis, var_floor: str = "poisson", clamp: float = 1e-12,
               verify: int = 15, seed: int = 0):
    """Cache the inverse-variance numerator/denominator so a knob costs an array op.

    `pooled_delta` is a per-gene inverse-variance average, so caching `num` and `den`
    lets a caller re-weight without re-pooling. This CACHES `submit.build.pooled_delta`
    -- it never carries a second implementation of it -- and asserts so on a sample of
    `verify` targets before returning.
    """
    from sidechain.models.count_emitters import remap_to_axis
    from sidechain.submit.build import as_delta_source, pooled_delta

    axis = np.asarray(axis, dtype=str)
    G = len(axis)
    num = np.zeros((len(targets), G))
    den = np.zeros((len(targets), G))
    for i, p in enumerate(targets):
        for src in (as_delta_source(s, var_floor=var_floor) for s in sources):
            got = src.effect(str(p))
            if got is None:
                continue
            fc, var = got
            with np.errstate(divide="ignore"):
                w = 1.0 / np.maximum(var, clamp)
            w = np.where(np.isfinite(var), w, 0.0)
            fc = np.where(np.isfinite(fc), fc, 0.0)
            num[i] += remap_to_axis(fc * w, src.genes, axis, fill=0.0)
            den[i] += remap_to_axis(w, src.genes, axis, fill=0.0)

    if verify:
        rng = np.random.default_rng(seed)
        k = min(verify, len(targets))
        worst = 0.0
        for i in sorted(rng.choice(len(targets), size=k, replace=False).tolist()):
            ref = pooled_delta(str(targets[i]), sources, axis, shrinkage=False,
                               var_floor=var_floor)
            mine = np.zeros(G)
            nz = den[i] > 0
            mine[nz] = num[i][nz] / den[i][nz]
            worst = max(worst, float(np.abs(ref - mine).max()))
        if worst != 0.0:
            raise AssertionError(
                f"cached pool parts do not reproduce pooled_delta (max |diff| {worst})")
    return num, den


def delta_from_parts(num, den):
    """[P, G] pooled log2FC from cached parts; genes no source spoke for stay at 0."""
    d = np.zeros_like(num)
    nz = den > 0
    d[nz] = num[nz] / den[nz]
    return d
