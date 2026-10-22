"""Per-cell and per-gene QC numbers from a raw count matrix, and compact JSON for a browser to draw them.

    from sidechain.viz.qc import cell_qc, gene_stats, program_score, compact
    q = cell_qc(X, genes)          # X: cells x genes raw counts (CSR or dense); genes: symbols
    g = gene_stats(X)              # per gene: mean count, detection, mean and variance of log CP10K
    json.dumps(compact(obj))       # numpy arrays rounded to a few significant figures, lists out

The metrics are the scverse/Cell Ranger set a single-cell QC reads first: UMIs per cell, genes detected per cell,
the share of counts on mitochondrial (MT-) and ribosomal-protein (RPL/RPS) genes, the share held by the cell's 50
most-counted genes (a library dominated by few genes is a low-complexity or damaged cell) and the complexity ratio
log10(genes) / log10(UMIs). Read counts and sequencing saturation need the reads, which no count matrix carries.
Used by the board's Figures tab builders (private/scripts/figures/, T129).
"""
from __future__ import annotations

import re

import numpy as np
import scipy.sparse as sp

_RIBO = re.compile(r"^RP[SL]\d+[A-Z]?\d?$|^RPLP[0-2]$")

QC_LABELS = {
    "total_umi": "UMIs per cell",
    "n_genes": "genes detected per cell",
    "pct_mt": "% of UMIs on mitochondrial genes",
    "pct_ribo": "% of UMIs on ribosomal-protein genes",
    "pct_top50": "% of UMIs in the cell's top 50 genes",
    "complexity": "log10 genes / log10 UMIs",
}


def _csr(X):
    return X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))


def gene_masks(genes) -> dict[str, np.ndarray]:
    """Boolean masks over `genes` for the mitochondrial (MT-) and ribosomal-protein (RPL/RPS, RPLP0-2) sets, human
    symbols. The RPS6K kinases share the prefix and are not ribosomal proteins, so the rule is a pattern, not a prefix."""
    g = [str(x).upper() for x in genes]
    mt = np.array([s.startswith("MT-") for s in g], dtype=bool)
    ribo = np.array([bool(_RIBO.match(s)) for s in g], dtype=bool)
    return {"mt": mt, "ribo": ribo}


def cell_qc(X, genes, top_n: int = 50) -> dict[str, np.ndarray]:
    """The per-cell QC metrics (see QC_LABELS) of a raw count matrix, as float arrays keyed by metric."""
    X = _csr(X)
    total = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
    m = gene_masks(genes)
    safe = np.where(total > 0, total, np.nan)
    out = {"total_umi": total, "n_genes": n_genes.astype(np.float64)}
    for k in ("mt", "ribo"):
        out[f"pct_{k}"] = 100.0 * np.asarray(X[:, np.flatnonzero(m[k])].sum(axis=1)).ravel() / safe if m[k].any() \
            else np.full(X.shape[0], np.nan)
    top = np.empty(X.shape[0])
    for i in range(X.shape[0]):                       # the cell's own top-n genes, as scanpy's pct_counts_in_top_n
        row = X.data[X.indptr[i]:X.indptr[i + 1]]
        top[i] = np.sort(row)[-top_n:].sum() if row.size else 0.0
    out["pct_top50"] = 100.0 * top / safe
    with np.errstate(divide="ignore", invalid="ignore"):
        out["complexity"] = np.log10(np.maximum(n_genes, 1)) / np.log10(np.maximum(total, 2))
    return out


def gene_stats(X, target_sum: float = 1e4) -> dict[str, np.ndarray]:
    """Per gene: mean raw count, fraction of cells detecting it, mean CP10K, and mean and variance of log1p CP10K."""
    X = _csr(X).astype(np.float64)
    n = X.shape[0]
    lib = np.asarray(X.sum(axis=1)).ravel()
    scale = sp.diags(np.where(lib > 0, target_sum / np.maximum(lib, 1e-12), 0.0))
    N = scale @ X
    L = N.copy()
    L.data = np.log1p(L.data)
    mean_log = np.asarray(L.mean(axis=0)).ravel()
    sq = L.multiply(L)
    var_log = np.asarray(sq.mean(axis=0)).ravel() - mean_log ** 2
    return {"mean_count": np.asarray(X.mean(axis=0)).ravel(),
            "detect": np.asarray((X > 0).sum(axis=0)).ravel() / n,
            "mean_cp10k": np.asarray(N.mean(axis=0)).ravel(),
            "mean_log": mean_log, "var_log": np.maximum(var_log, 0.0)}


def program_score(L, genes, program, ctrl_size: int = 50, seed: int = 0) -> np.ndarray:
    """A gene-programme score per cell on log-normalised data `L` (Tirosh-style: the programme's mean minus the
    mean of a random control set of genes; scanpy's `score_genes` without the expression-bin matching, which is
    within noise for programmes of a dozen genes). NaN when none of the programme's genes is present."""
    L = _csr(L)
    idx = {g: i for i, g in enumerate(map(str, genes))}
    hit = [idx[g] for g in program if g in idx]
    if not hit:
        return np.full(L.shape[0], np.nan)
    rest = np.setdiff1d(np.arange(L.shape[1]), hit)
    ctrl = np.random.default_rng(seed).choice(rest, size=min(ctrl_size, rest.size), replace=False)
    a = np.asarray(L[:, hit].mean(axis=1)).ravel()
    b = np.asarray(L[:, ctrl].mean(axis=1)).ravel()
    return a - b


def _sig(a: np.ndarray, digits: int) -> list:
    a = np.asarray(a, dtype=np.float64)
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a) & (a != 0)
    mag = np.floor(np.log10(np.abs(a[ok])))
    f = 10.0 ** (digits - 1 - mag)
    out[ok] = np.round(a[ok] * f) / f
    out[np.isfinite(a) & (a == 0)] = 0.0
    return [None if not np.isfinite(v) else (int(v) if float(v).is_integer() and abs(v) < 2**53 else float(v))
            for v in out.ravel().tolist()]


def compact(obj, digits: int = 4):
    """Make `obj` JSON-ready: numpy arrays become lists rounded to `digits` significant figures, NaN becomes null,
    numpy scalars become Python numbers. Integer arrays are kept exact."""
    if isinstance(obj, dict):
        return {str(k): compact(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [compact(v, digits) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "iub":
            return obj.astype(np.int64).tolist() if obj.dtype.kind != "b" else obj.tolist()
        if obj.dtype.kind in "OUS":
            return [str(x) for x in obj.tolist()]
        if obj.ndim > 1:
            return [compact(r, digits) for r in obj]
        return _sig(obj, digits)
    if isinstance(obj, np.generic):
        return compact(np.asarray([obj]), digits)[0] if obj.dtype.kind == "f" else obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
