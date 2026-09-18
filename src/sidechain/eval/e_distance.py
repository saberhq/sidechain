"""E-distance and E-test (Peidli et al. 2024, *Nature Methods* -- scPerturb), built against the
paper and the authors' own `scperturb` package rather than from memory (`T94`, reversing `T21`'s
deletion of an unimplemented stub -- see `metrics_extra.py` for that history).

**Not a PDS proxy.** `metrics_extra.py` has the full argument for why this statistic cannot
substitute for `pds_cosine`: E-distance compares two clouds of real cells and needs no
prediction, PDS ranks a predicted pseudobulk delta against the truth. Its home here is grading
whether a *source* perturbation's own knockdown is distinguishable from control, before that
source's delta is pooled (`research/ideas/e-test-source-perturbation-gate.md`) -- not scoring
a prediction.

**`scperturb.edist_to_control` / `scperturb.etest` ARE the paper's formulas**, read against
Methods and confirmed, not `pertpy`'s (measured wrong in `metrics_extra.py`):
  * cell-wise distance: squared Euclidean -- `dist='sqeuclidean'`, the package default.
  * bias correction: sigma divided by N(N-1), not N^2 -- `sample_correct=True`, the default.
  * E(X,Y) = 2*delta_XY - sigma_X - sigma_Y, exactly the Methods formula.
  * E-test: Monte-Carlo permutation p-value, Holm-Sidak corrected per dataset -- the package's
    default `correction_method`, via `statsmodels`.
So this module does not reimplement the statistic itself -- it pins the arguments that must
never silently drift (so a future edit cannot reintroduce the pertpy trap) and supplies the
one thing `scperturb` does not: the paper's fixed preprocessing recipe.

**One measured departure from `scperturb.equal_subsampling`.** Called with `N_min=50` directly,
it computes the subsample size as `max(50, min(cell count over ALL groups, including ones about
to be dropped))` -- so a single near-empty group elsewhere in the dataset silently pins every
surviving group down to exactly 50 cells, not to "the smallest perturbation left after
filtering" as the paper states. Measured 2026-09-18 on a synthetic {3, 62, 70, 100}-cell
four-group example: `equal_subsampling(adata, key, N_min=50)` subsampled every surviving group
to 50; pre-filtering ourselves and calling `equal_subsampling(filtered, key)` (`N_min=None`, its
default) subsampled to 62 -- the smallest *surviving* group, matching the paper's own wording.
`prep_for_edistance` below does the filter itself for exactly this reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scperturb import edist_to_control, equal_subsampling, etest

# The paper's Methods (Data analysis), verbatim thresholds -- never silently retuned.
MIN_UMIS_PER_CELL = 1000
MIN_CELLS_PER_GENE = 50
N_HVG = 2000
N_PCS = 50
MIN_CELLS_PER_PERTURBATION = (
    50  # perturbations under this are dropped before anything else
)
DIST = "sqeuclidean"  # the Methods formula is squared Euclidean, not Euclidean
SAMPLE_CORRECT = True  # N(N-1) bias correction; "all calculations use" this form

__all__ = ["e_distance", "e_test", "prep_for_edistance"]


def prep_for_edistance(adata: AnnData, pert_col: str, *, seed: int = 0) -> AnnData:
    """The paper's fixed preprocessing recipe -- not a configurable one.

    Cells with >=1,000 UMIs, genes seen in >=50 cells, 2,000 seurat_v3 HVGs (selected on raw
    counts, before normalizing), `normalize_total` + `log1p` (no z-scaling, no `target_sum`
    override -- the paper names neither), 50-component PCA on the HVGs, then perturbations
    under 50 cells dropped and the rest equalized by subsampling to the smallest survivor.
    E-distance is not scale-free in cell count even after bias correction (Peidli 2024 Sec. 2),
    so any deviation from this recipe produces a number that is not comparable to the paper's.

    `adata.X` must be raw counts. Returns a new `AnnData` with `X_pca` in `.obsm`, cells already
    equalized per group in `pert_col` -- ready for `e_distance` / `e_test`.
    """
    out = adata.copy()
    sc.pp.filter_cells(out, min_counts=MIN_UMIS_PER_CELL)
    sc.pp.filter_genes(out, min_cells=MIN_CELLS_PER_GENE)
    sc.pp.highly_variable_genes(out, n_top_genes=N_HVG, flavor="seurat_v3")
    out = out[:, out.var["highly_variable"]].copy()
    sc.pp.normalize_total(out)
    sc.pp.log1p(out)
    sc.pp.pca(out, n_comps=N_PCS)

    counts = out.obs[pert_col].value_counts()
    survivors = counts.index[counts >= MIN_CELLS_PER_PERTURBATION]
    out = out[out.obs[pert_col].isin(survivors)].copy()

    rng_state = np.random.get_state()
    np.random.seed(seed)  # scperturb.equal_subsampling draws from the global numpy RNG
    try:
        out = equal_subsampling(
            out, pert_col
        )  # N_min=None: subsample to the smallest survivor
    finally:
        np.random.set_state(rng_state)
    return out


def e_distance(adata: AnnData, pert_col: str, control: str | list[str]) -> pd.Series:
    """E-distance of every group in `pert_col` to `control`, in `adata.obsm['X_pca']`.

    A thin, argument-pinned call onto `scperturb.edist_to_control` -- see the module docstring
    for why `dist` and `sample_correct` are fixed rather than exposed.
    """
    result = edist_to_control(
        adata,
        obs_key=pert_col,
        control=control,
        dist=DIST,
        sample_correct=SAMPLE_CORRECT,
        verbose=False,
    )
    return result["distance"]


def e_test(
    adata: AnnData,
    pert_col: str,
    control: str | list[str],
    *,
    n_permutations: int = 10_000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-group E-test against `control`: columns `edist`, `pvalue`, `pvalue_adj` (Holm-Sidak).

    10,000 permutations is the paper's own number (Peidli 2024 Sec. 2: "We repeated this
    process 10,000 times") -- drop it for a cheap first pass, never for the number that decides
    a real gate. `scperturb.etest` runs one shuffle at a time per `n_jobs`; a genome-wide corpus
    (thousands of targets x 10,000 permutations) is a box job, not a Mac one --
    `research/ideas/e-test-source-perturbation-gate.md`.
    """
    return etest(
        adata,
        obs_key=pert_col,
        control=control,
        dist=DIST,
        sample_correct=SAMPLE_CORRECT,
        runs=n_permutations,
        alpha=alpha,
        n_jobs=1,
        verbose=False,
    )
