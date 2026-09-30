"""E-distance and E-test (Peidli et al. 2024, *Nature Methods* -- scPerturb), built against the
paper and the authors' own `scperturb` package rather than from memory (`T94`, reversing `T21`'s
deletion of an unimplemented stub -- see `metrics_extra.py` for that history).

**Not a PDS proxy.** `metrics_extra.py` has the full argument for why this statistic cannot
substitute for `pds_cosine`: E-distance is computed from real cells and needs no prediction,
PDS ranks a predicted pseudobulk delta against the truth. Its home here is grading whether a
*source* perturbation's own knockdown is distinguishable from control, before that source's
delta is pooled (`research/ideas/e-test-source-perturbation-gate.md`) -- not scoring a
prediction.

**`scperturb.edist_to_control` / `scperturb.etest` ARE the paper's formulas**, read against
Methods and confirmed, not `pertpy`'s (measured wrong in `metrics_extra.py`):
  * cell-wise distance: squared Euclidean -- `dist='sqeuclidean'`, the package default.
  * bias correction: sigma divided by N(N-1), not N^2 -- `sample_correct=True`, the default
    (the paper uses it everywhere except its Fig. 5c and 5d).
  * E(X,Y) = 2*delta_XY - sigma_X - sigma_Y, exactly the Methods formula.
  * E-test: Monte-Carlo permutation p-value, Holm-Sidak corrected per dataset -- the package's
    default `correction_method`, via `statsmodels`.
So this module does not reimplement the statistic itself -- it pins the arguments that must
never silently drift (so a future edit cannot reintroduce the pertpy trap) and supplies the
one thing `scperturb` does not: the paper's fixed preprocessing recipe.

**What the number measures: the shift of the MEAN, not the shape of the cloud.** With squared
distances and the N(N-1) correction the formula is algebraically

    E(X,Y) = 2 * ||mean(X) - mean(Y)||^2  -  2 * tr(S_X) / N  -  2 * tr(S_Y) / M

(S the unbiased covariance), an unbiased estimate of 2 * ||mu_X - mu_Y||^2: the within-group
terms remove the sampling noise of the two means and add no spread (`tests/test_eval_e_distance.py`
pins the identity; measured on real K562 cells to 2.5e-12, T103, 2026-09-29). So a knockdown
that only widens the cloud scores zero, and "half the cells respond fully" reads the same as
"every cell responds halfway". The statistic that compares whole distributions is the
plain-Euclidean energy distance (Szekely-Rizzo; `scperturb` with `dist='euclidean'`) -- a
different number from the paper's, deliberately not exposed here. Two further traps, both
measured 2026-09-29 (`~/data/sidechain/runs/t103_directions_20260929/a_edist/`): a null from
random splits of the control cells understates the real one wherever controls span batches
(150 control cells drawn from 3 batches score 2.8 on X-Atlas HCT116 and about 4 on K562 GWPS,
means over 40 draws, against about 0 for a random split), so a threshold on E must come from a
batch-matched null; and a control given
as a list of labels is merged into one group first (`_one_control`), because
`edist_to_control` pools the list for delta but averages the per-label sigmas, which adds the
between-label spread to every knockdown (+0.317 on HCT116 split by batch).

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


def _one_control(
    adata: AnnData, pert_col: str, control: str | list[str]
) -> tuple[AnnData, str]:
    """A list of control labels becomes ONE group, named ``'+'.join(labels)``.

    `edist_to_control` pools a list's cells for the between-group term but averages the
    per-label sigmas, so batch-split controls add their between-label spread to every
    knockdown's E (module docstring). Merging first makes `e_distance` and `e_test` see the
    same pooled control. A single label is passed through untouched.
    """
    if isinstance(control, str):
        return adata, control
    labels = [str(c) for c in dict.fromkeys(control)]
    if len(labels) == 1:
        return adata, labels[0]
    col = adata.obs[pert_col].astype(str)
    missing = [c for c in labels if not (col == c).any()]
    if missing:
        raise ValueError(f"control labels not found in {pert_col!r}: {missing}")
    merged = "+".join(labels)
    obs = pd.DataFrame({pert_col: col.where(~col.isin(labels), merged)}, index=adata.obs_names)
    return AnnData(obs=obs, obsm={"X_pca": adata.obsm["X_pca"]}), merged


def e_distance(adata: AnnData, pert_col: str, control: str | list[str]) -> pd.Series:
    """E-distance of every group in `pert_col` to `control`, in `adata.obsm['X_pca']`.

    A thin, argument-pinned call onto `scperturb.edist_to_control` -- see the module docstring
    for why `dist` and `sample_correct` are fixed rather than exposed, and for what the number
    measures (the mean shift). A list `control` is merged into one group first; the result then
    carries one row for it, named ``'+'.join(control)``.
    """
    adata, control = _one_control(adata, pert_col, control)
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
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Per-group E-test against `control`: columns `edist`, `pvalue`, `pvalue_adj` (Holm-Sidak).

    10,000 permutations is the paper's own number (Peidli 2024 Sec. 2: "We repeated this
    process 10,000 times") -- drop it for a cheap first pass, never for the number that decides
    a real gate. `scperturb.etest` parallelizes over permutation runs via `n_jobs`
    (`joblib`) -- the default of 1 is the safe choice for a laptop; a genome-wide corpus
    (thousands of targets x 10,000 permutations) is a box job and should pass the box's core
    count, not stay at 1 -- `research/ideas/e-test-source-perturbation-gate.md`. A list
    `control` is merged into one group first, as in `e_distance`.
    """
    adata, control = _one_control(adata, pert_col, control)
    return etest(
        adata,
        obs_key=pert_col,
        control=control,
        dist=DIST,
        sample_correct=SAMPLE_CORRECT,
        runs=n_permutations,
        alpha=alpha,
        n_jobs=n_jobs,
        verbose=False,
    )
