"""Trended, empirically shrunk Gamma-Poisson dispersion for a pseudobulk artifact: glmGamPoi's step 8, in numpy.

    python -m sidechain.data.dispersion SRC.npz --out FIT.npz

Written from Ahlmann-Eltze & Huber 2021 (Bioinformatics 36(24):5701-5702, doi 10.1093/bioinformatics/btaa1009,
Supplement S1): a method-of-moments dispersion per gene (`moment_dispersion`), a sliding-window median trend over
mean expression (`dispersion_trend`), the quasi-dispersion against it, an inverse-chi-squared prior fitted by ML
(`fit_variance_prior`) and the shrinkage toward it. Reads a saved `PseudobulkSums` (.npz); writes the fit (.npz)
and a .json summary beside it. Notes: private/code-notes/src/sidechain/data/dispersion.py.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import f as f_dist

from sidechain.data.stream_pseudobulk import PseudobulkSums, iter_npz_row_blocks

# the trend's floor: exactly zero (the Poisson case) breaks the ratios in `quasi_dispersion`
MIN_DISPERSION = 1e-8
# Groups thinner than this contribute no usable variance estimate (n = 1 has none at all).
MIN_CELLS_PER_GROUP = 2


@dataclass(frozen=True)
class GeneDispersion:
    """Per-gene dispersion for one arm: the raw `theta_ml`, its `theta_trend` and the shrunk `theta_sql`."""

    genes: np.ndarray        # (G,) gene names, the arm's own axis
    mean_count: np.ndarray   # (G,) cell-weighted mean RAW count per cell
    theta_ml: np.ndarray     # (G,) method-of-moments dispersion (supplement step 2)
    theta_trend: np.ndarray  # (G,) the local median curve through theta_ml
    theta_ql: np.ndarray     # (G,) theta_ML expressed relative to the trend
    theta_sql: np.ndarray    # (G,) theta_QL shrunk toward the fitted prior
    df: float                # residual degrees of freedom behind each theta_ML
    df0: float               # the prior's degrees of freedom -- its strength, fitted
    tau0_sq: float           # the prior's scale, fitted
    n_groups_used: int
    # provenance, set by `fit_gene_dispersion_file` and kept by `save`/`load`: the source artifact's stem and
    # sha256; empty for a fit made from an in-memory object
    source: str = ""
    fitted_on_sha256: str = ""
    # sha256 of the artifact's `sources` member (the corpus files it was accumulated from), shared by its label subsets
    fitted_on_sources_sha256: str = ""
    # every label of the fitted artifact with its cell count, to recognise a label subset of it
    fitted_labels: np.ndarray | None = None
    fitted_n_cells: np.ndarray | None = None

    def save(self, path: str | Path) -> None:
        """One `.npz` holding the arrays and the scalars; `load` is its exact inverse."""
        np.savez_compressed(
            Path(path).expanduser(),
            genes=np.asarray(self.genes, dtype=object), mean_count=self.mean_count,
            theta_ml=self.theta_ml, theta_trend=self.theta_trend, theta_ql=self.theta_ql,
            theta_sql=self.theta_sql,
            scalars=np.array([self.df, self.df0, self.tau0_sq, float(self.n_groups_used)]),
            provenance=np.array([self.source, self.fitted_on_sha256, self.fitted_on_sources_sha256],
                                dtype=object),
            fitted_labels=np.asarray([] if self.fitted_labels is None else self.fitted_labels, dtype=object),
            fitted_n_cells=np.asarray([] if self.fitted_n_cells is None else self.fitted_n_cells, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> GeneDispersion:
        z = np.load(Path(path).expanduser(), allow_pickle=True)
        df, df0, tau0_sq, n_used = (float(v) for v in z["scalars"])
        prov = [str(v) for v in z["provenance"]] + ["", "", ""]
        labels = z["fitted_labels"].astype(str) if "fitted_labels" in z.files and z["fitted_labels"].size else None
        n_cells = z["fitted_n_cells"] if "fitted_n_cells" in z.files and z["fitted_n_cells"].size else None
        return cls(genes=z["genes"].astype(str), mean_count=z["mean_count"],
                   theta_ml=z["theta_ml"], theta_trend=z["theta_trend"], theta_ql=z["theta_ql"],
                   theta_sql=z["theta_sql"], df=df, df0=df0, tau0_sq=tau0_sq,
                   n_groups_used=int(n_used), source=prov[0], fitted_on_sha256=prov[1],
                   fitted_on_sources_sha256=prov[2], fitted_labels=labels, fitted_n_cells=n_cells)

    def variance_counts(self, mean_count: np.ndarray) -> np.ndarray:
        """The paper's variance `theta_sql * (mu + theta_trend * mu^2)` at a mean RAW count per cell `mean_count`."""
        mu = np.asarray(mean_count, dtype=np.float64)
        return self.theta_sql * (mu + self.theta_trend * mu * mu)

    def variance_cpm(self, mean_cpm: np.ndarray, mean_libsize: float,
                     mean_inv_libsize: float | None = None) -> np.ndarray:
        """The same variance in CPM: the Poisson term scaled by `1e6 / mean_libsize`, or by `1e6 * mean_inv_libsize`.

        Pass `mean_inv_libsize` (the group's `mean(1/L_i)`) only with a fit made with `inv_libsize_sum`.
        """
        m = np.asarray(mean_cpm, dtype=np.float64)
        scale = 1e6 / float(mean_libsize) if mean_inv_libsize is None else 1e6 * float(mean_inv_libsize)
        return self.theta_sql * (m * scale + self.theta_trend * m * m)


def moment_dispersion(pb: PseudobulkSums, *, exclude: tuple[str, ...] = (),
                      min_cells: int = MIN_CELLS_PER_GROUP,
                      inv_libsize_sum: np.ndarray | None = None,
                      ) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-gene dispersion from `sigma^2 = mu + theta*mu^2`, one slope per gene pooled over every group.

    Each group's excess CPM variance over Poisson is regressed on `m_cpm^2`, weighted by `n - 1`. `inv_libsize_sum`
    (per group, `sum(1/L_i)`) scales the Poisson term by `mean(1/L)` in place of `1/mean(L)`.
    Returns `(mean_count, theta_ml, df)`, `df = sum(n_l - 1)` over the groups used.
    """
    keep = [i for i, lab in enumerate(pb.labels)
            if lab not in exclude and int(pb.n_cells[i]) >= min_cells]
    if len(keep) < 2:
        raise ValueError(
            f"need at least 2 groups with >= {min_cells} cells to fit a dispersion, "
            f"got {len(keep)} -- a single group cannot separate sampling noise from dispersion")
    rows = np.asarray(keep)
    n = pb.n_cells[rows].astype(np.float64)
    m_cpm = pb.cpm_sum[rows] / n[:, None]
    # Bessel's correction: `PseudobulkSums.var_cpm` is the population form (divides by n)
    v_cpm = np.maximum(pb.cpm_sq_sum[rows] / n[:, None] - m_cpm * m_cpm, 0.0)
    v_cpm = v_cpm * (n / np.maximum(n - 1.0, 1.0))[:, None]
    if inv_libsize_sum is None:
        scale = (1e6 / (pb.libsize_sum[rows] / n))[:, None]  # CPM per raw count, per group
    else:
        # Poisson noise per cell is m_cpm * 1e6 / L_i; its group mean is m_cpm * 1e6 * mean(1/L)
        scale = (1e6 * np.asarray(inv_libsize_sum, dtype=np.float64)[rows] / n)[:, None]

    # Var_cpm = mu*scale^2 + theta*(mu*scale)^2 with mu*scale = m_cpm, so the excess over Poisson is theta * m_cpm^2
    excess = v_cpm - m_cpm * scale
    w = np.maximum(n - 1.0, 0.0)[:, None]
    num = (w * excess).sum(axis=0)
    den = (w * m_cpm * m_cpm).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.where(den > 0, num / den, 0.0)
    # a negative slope (variances under Poisson) is clamped to zero, as the supplement clamps its ML fit
    theta = np.maximum(np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    mean_count = (pb.count_sum[rows].sum(axis=0) / n.sum())
    df = float(np.maximum(n - 1.0, 0.0).sum())
    return mean_count, theta, df


def dispersion_trend(mean_count: np.ndarray, theta_ml: np.ndarray, *,
                     window: int = 301, min_expressed: float = 0.0) -> np.ndarray:
    """A sliding-window median of `theta_ml` over genes ordered by mean count, `window` genes wide, edges flat.

    Genes at or below `min_expressed` take the trend of the lowest expressed gene above it.
    """
    mean_count = np.asarray(mean_count, dtype=np.float64)
    theta_ml = np.asarray(theta_ml, dtype=np.float64)
    if mean_count.shape != theta_ml.shape:
        raise ValueError(f"shape mismatch: {mean_count.shape} vs {theta_ml.shape}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    out = np.full(theta_ml.shape, MIN_DISPERSION, dtype=np.float64)
    usable = np.flatnonzero(mean_count > min_expressed)
    if usable.size == 0:
        return out
    order = usable[np.argsort(mean_count[usable], kind="stable")]
    vals = theta_ml[order]
    k = int(min(window, vals.size))
    half = k // 2
    # explicit windows: an O(G*k) pass of np.median over slices
    smoothed = np.empty(vals.size, dtype=np.float64)
    for i in range(vals.size):
        lo = max(0, min(i - half, vals.size - k))
        smoothed[i] = np.median(vals[lo:lo + k])
    out[order] = np.maximum(smoothed, MIN_DISPERSION)
    # Genes with no expression inherit the curve's low-expression end rather than a zero.
    out[mean_count <= min_expressed] = out[order[0]]
    return out


def quasi_dispersion(mean_count: np.ndarray, theta_ml: np.ndarray,
                     theta_trend: np.ndarray) -> np.ndarray:
    """`theta_QL = (1 + mu*theta_ML) / (1 + mu*theta_trend)`: 1.0 for a gene exactly on the trend."""
    mu = np.asarray(mean_count, dtype=np.float64)
    num = 1.0 + mu * np.asarray(theta_ml, dtype=np.float64)
    den = 1.0 + mu * np.maximum(np.asarray(theta_trend, dtype=np.float64), MIN_DISPERSION)
    return num / np.maximum(den, 1e-300)


def fit_variance_prior(theta_ql: np.ndarray, df: float) -> tuple[float, float]:
    """ML fit of the inverse-chi-squared prior's `(df0, tau0^2)`: `theta_QL / tau0^2` ~ F(df, df0), over logs.

    Returns `(0.0, 1.0)`, a prior of no strength, when there is nothing to fit.
    """
    x = np.asarray(theta_ql, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 2 or df <= 0:
        return 0.0, 1.0

    def nll(p):
        df0, tau0_sq = np.exp(p)
        if not np.isfinite(df0) or not np.isfinite(tau0_sq) or df0 <= 0 or tau0_sq <= 0:
            return np.inf
        # density of x = tau0_sq * F(df, df0), so the Jacobian contributes -log(tau0_sq)
        ll = f_dist.logpdf(x / tau0_sq, df, df0) - np.log(tau0_sq)
        return -float(np.sum(ll)) if np.all(np.isfinite(ll)) else np.inf

    start = np.log([max(df, 1.0), max(float(np.median(x)), 1e-6)])
    res = minimize(nll, start, method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
    if not np.isfinite(res.fun):
        return 0.0, 1.0
    df0, tau0_sq = (float(v) for v in np.exp(res.x))
    return df0, tau0_sq


def shrink_quasi_dispersion(theta_ql: np.ndarray, df: float, df0: float,
                            tau0_sq: float) -> np.ndarray:
    """`theta_SQL = (df0*tau0^2 + df*theta_QL) / (df0 + df)`; `df0 = 0` returns `theta_QL` unchanged."""
    total = float(df0) + float(df)
    if total <= 0:
        return np.asarray(theta_ql, dtype=np.float64).copy()
    return (df0 * tau0_sq + df * np.asarray(theta_ql, dtype=np.float64)) / total


def fit_gene_dispersion(pb: PseudobulkSums, *, exclude: tuple[str, ...] = (),
                        window: int = 301, min_cells: int = MIN_CELLS_PER_GROUP,
                        df0: float | None = None,
                        inv_libsize_sum: np.ndarray | None = None) -> GeneDispersion:
    """Step 8 on one arm: moments, trend, quasi-dispersion, prior, shrinkage.

    `exclude` drops label rows before the fit; `df0` sets the prior's strength in place of fitting it.
    """
    mean_count, theta_ml, df = moment_dispersion(pb, exclude=exclude, min_cells=min_cells,
                                                 inv_libsize_sum=inv_libsize_sum)
    theta_trend = dispersion_trend(mean_count, theta_ml, window=window)
    theta_ql = quasi_dispersion(mean_count, theta_ml, theta_trend)
    fitted_df0, tau0_sq = fit_variance_prior(theta_ql, df)
    if df0 is None:
        df0 = fitted_df0
    elif df0 < 0:
        raise ValueError(f"df0 must be non-negative, got {df0}")
    theta_sql = shrink_quasi_dispersion(theta_ql, df, df0, tau0_sq)
    n_used = int(sum(1 for i, lab in enumerate(pb.labels)
                     if lab not in exclude and int(pb.n_cells[i]) >= min_cells))
    return GeneDispersion(genes=np.asarray(pb.genes), mean_count=mean_count,
                          theta_ml=theta_ml, theta_trend=theta_trend, theta_ql=theta_ql,
                          theta_sql=theta_sql, df=df, df0=df0, tau0_sq=tau0_sq,
                          n_groups_used=n_used)


# ------------------------------------------------------------- the same fit, streamed from disk


def _small_members(path: Path) -> dict:
    """labels, genes, n_cells, libsize_sum, sources of a saved `PseudobulkSums` -- the cheap members."""
    import io
    import zipfile

    out = {}
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        for name in ("labels", "genes", "n_cells", "libsize_sum", "sources"):
            if f"{name}.npy" not in names:
                out[name] = np.array([], dtype=object)
                continue
            with z.open(f"{name}.npy") as fh:
                out[name] = np.load(io.BytesIO(fh.read()), allow_pickle=True)
    return out


def sources_sha256(sources) -> str:
    """The identity of the corpus files an artifact was accumulated from (its `sources` member)."""
    import hashlib

    return hashlib.sha256("\n".join(str(s) for s in sources).encode()).hexdigest()


def moment_dispersion_file(path: str | Path, *, exclude: tuple[str, ...] = (),
                           min_cells: int = MIN_CELLS_PER_GROUP, block_rows: int = 128,
                           progress: bool = False, inv_libsize_sum: np.ndarray | None = None,
                           ) -> tuple[np.ndarray, np.ndarray, float, int, np.ndarray]:
    """`moment_dispersion` on a saved artifact, one row block at a time, keeping only per-gene sums.

    Refuses an artifact whose `count_sum` is all zero (a `load_subset` product).
    Returns `(mean_count, theta_ml, df, n_groups_used, genes)`.
    """
    path = Path(path).expanduser()
    small = _small_members(path)
    labels = [str(x) for x in small["labels"]]
    genes = small["genes"].astype(str)
    n_all = small["n_cells"].astype(np.float64)
    lib_all = small["libsize_sum"].astype(np.float64)
    inv_all = None if inv_libsize_sum is None else np.asarray(inv_libsize_sum, dtype=np.float64)
    keep = np.array([lab not in exclude and int(n_all[i]) >= min_cells
                     for i, lab in enumerate(labels)])
    if int(keep.sum()) < 2:
        raise ValueError(
            f"need at least 2 groups with >= {min_cells} cells to fit a dispersion, "
            f"got {int(keep.sum())} -- a single group cannot separate sampling noise from dispersion")
    G = len(genes)
    num = np.zeros(G); den = np.zeros(G); count_total = np.zeros(G)
    n_total = 0.0; df = 0.0; count_any = False
    members = ("cpm_sum", "cpm_sq_sum", "count_sum")
    for r0, blk in iter_npz_row_blocks(path, [f"{m}.npy" for m in members], block_rows=block_rows):
        k = blk["cpm_sum.npy"].shape[0]
        rows = np.flatnonzero(keep[r0:r0 + k])
        if progress and (r0 // block_rows) % 20 == 0:
            print(f"  rows {r0}/{len(labels)}", flush=True)
        if rows.size == 0:
            continue
        n = n_all[r0 + rows]
        cs = blk["cpm_sum.npy"][rows]
        cq = blk["cpm_sq_sum.npy"][rows]
        ct = blk["count_sum.npy"][rows]
        if not count_any and ct.any():
            count_any = True
        m_cpm = cs / n[:, None]
        # the same Bessel correction as `moment_dispersion`
        v_cpm = np.maximum(cq / n[:, None] - m_cpm * m_cpm, 0.0)
        v_cpm = v_cpm * (n / np.maximum(n - 1.0, 1.0))[:, None]
        if inv_all is None:
            scale = (1e6 / (lib_all[r0 + rows] / n))[:, None]
        else:
            scale = (1e6 * inv_all[r0 + rows] / n)[:, None]   # as `moment_dispersion`
        excess = v_cpm - m_cpm * scale
        w = np.maximum(n - 1.0, 0.0)[:, None]
        num += (w * excess).sum(axis=0)
        den += (w * m_cpm * m_cpm).sum(axis=0)
        count_total += ct.sum(axis=0)
        n_total += float(n.sum())
        df += float(np.maximum(n - 1.0, 0.0).sum())
    if not count_any:
        raise ValueError(f"{path.name}: count_sum is all zero -- a load_subset product, not a "
                         "fit-able artifact (the trend would flatten to MIN_DISPERSION)")
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.where(den > 0, num / den, 0.0)
    theta = np.maximum(np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    mean_count = count_total / n_total
    return mean_count, theta, df, int(keep.sum()), genes


def file_sha256(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).expanduser().open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fit_gene_dispersion_file(path: str | Path, *, exclude: tuple[str, ...] = (), window: int = 301,
                             min_cells: int = MIN_CELLS_PER_GROUP, df0: float | None = None,
                             block_rows: int = 128, progress: bool = False,
                             sha256: str | None = None) -> GeneDispersion:
    """`fit_gene_dispersion` on a saved artifact, stamped with its stem and sha256 (`sha256` if already hashed)."""
    path = Path(path).expanduser()
    mean_count, theta_ml, df, n_used, genes = moment_dispersion_file(
        path, exclude=exclude, min_cells=min_cells, block_rows=block_rows, progress=progress)
    theta_trend = dispersion_trend(mean_count, theta_ml, window=window)
    theta_ql = quasi_dispersion(mean_count, theta_ml, theta_trend)
    fitted_df0, tau0_sq = fit_variance_prior(theta_ql, df)
    if df0 is None:
        df0 = fitted_df0
    elif df0 < 0:
        raise ValueError(f"df0 must be non-negative, got {df0}")
    theta_sql = shrink_quasi_dispersion(theta_ql, df, df0, tau0_sq)
    small = _small_members(path)
    return GeneDispersion(genes=np.asarray(genes), mean_count=mean_count, theta_ml=theta_ml,
                          theta_trend=theta_trend, theta_ql=theta_ql, theta_sql=theta_sql,
                          df=df, df0=df0, tau0_sq=tau0_sq, n_groups_used=n_used,
                          source=path.stem, fitted_on_sha256=sha256 or file_sha256(path),
                          fitted_on_sources_sha256=sources_sha256(small["sources"]),
                          fitted_labels=np.asarray([str(x) for x in small["labels"]], dtype=object),
                          fitted_n_cells=np.asarray(small["n_cells"], dtype=np.int64))


def main(argv: list[str] | None = None) -> int:
    """Fit one artifact and save the fit: `python -m sidechain.data.dispersion SRC.npz --out FIT.npz`."""
    import argparse
    import json
    import time

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("source", type=Path, help="a saved PseudobulkSums (.npz), read in row blocks")
    ap.add_argument("--out", type=Path, required=True, help="where the fit (.npz) is written")
    ap.add_argument("--exclude", action="append", default=[], metavar="LABEL",
                    help="label rows left out of the fit (repeatable); default: none")
    ap.add_argument("--window", type=int, default=301)
    ap.add_argument("--min-cells", type=int, default=MIN_CELLS_PER_GROUP)
    ap.add_argument("--block-rows", type=int, default=128)
    args = ap.parse_args(argv)
    t0 = time.time()
    gd = fit_gene_dispersion_file(args.source, exclude=tuple(args.exclude), window=args.window,
                                  min_cells=args.min_cells, block_rows=args.block_rows, progress=True)
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    gd.save(args.out)
    ok = gd.mean_count > 0
    q = np.quantile(gd.mean_count[ok], [0.1, 0.5, 0.9])
    trend_at = [float(np.interp(np.log(x), np.log(np.sort(gd.mean_count[ok])),
                                gd.theta_trend[ok][np.argsort(gd.mean_count[ok], kind="stable")]))
                for x in q]
    rec = {"source": str(args.source), "source_sha256": gd.fitted_on_sha256, "out": str(args.out),
           "exclude": args.exclude, "window": args.window, "min_cells": args.min_cells,
           "genes": int(len(gd.genes)), "genes_expressed": int(ok.sum()),
           "n_groups_used": gd.n_groups_used, "df": gd.df, "df0": gd.df0, "tau0_sq": gd.tau0_sq,
           "theta_trend_at_p10_p50_p90_mean_count": trend_at,
           "theta_sql_median": float(np.median(gd.theta_sql[ok])),
           "theta_sql_max_shift_from_ql": float(np.max(np.abs(gd.theta_sql[ok] - gd.theta_ql[ok]))),
           "seconds": round(time.time() - t0, 1)}
    args.out.expanduser().with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
