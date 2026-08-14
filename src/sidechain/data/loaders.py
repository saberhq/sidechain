"""Thin wrappers around the challenge file format so the rest of Sidechain speaks
AnnData. Keep this module the ONLY place that touches the on-disk layout, so a
format change at the 2026 kickoff is a one-file edit.

The 2025 data (verified 2026-08-04 against adata_Training_subset.h5ad):
  obs: target_gene (control label 'non-targeting'), guide_id, batch
  var: index = HGNC symbols, var['gene_id'] = Ensembl gene IDs
  X:   raw integer counts, CSR sparse
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# A var column may be named any of these for a given logical ID space. Resolution
# is by name AND by value shape (see _ID_PATTERNS) -- a column called 'gene_id'
# holding symbols must not satisfy a request for Ensembl IDs.
_ID_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "ensembl_gene_id": ("ensembl_gene_id", "ensembl_id", "ensembl", "gene_id", "gene_ids"),
    "hgnc_symbol": ("hgnc_symbol", "gene_symbol", "symbol", "gene_name", "gene_names"),
}

_ID_PATTERNS: dict[str, re.Pattern | None] = {
    "ensembl_gene_id": re.compile(r"^ENSG\d{11}"),
    "hgnc_symbol": None,  # symbols have no reliable shape; accept any non-empty string
}


class GeneIDSpaceError(ValueError):
    """Raised when the requested gene ID space is absent or unrecognizable.

    Deliberately loud. The failure it replaces was silent: falling back to
    var_names yields an index that matches no prior, so every source returns zero
    edges and the run reports success on an empty graph.
    """


def _valid_fraction(values: Iterable, id_type: str) -> float:
    pattern = _ID_PATTERNS.get(id_type)
    vals = pd.Series(list(values), dtype="object").astype(str)
    if len(vals) == 0:
        return 0.0
    if pattern is None:
        ok = vals.str.len().gt(0) & ~vals.isin({"nan", "None", ""})
    else:
        ok = vals.str.match(pattern)
    return float(ok.mean())


def resolve_id_column(
    adata: ad.AnnData, id_type: str, min_valid_frac: float = 0.9
) -> tuple[str, pd.Index | pd.Series, float]:
    """Find the var column (or var_names) holding `id_type`.

    Returns (source_name, values, valid_fraction). Raises GeneIDSpaceError if no
    candidate carries values that actually look like `id_type`.
    """
    if id_type not in _ID_COLUMN_ALIASES:
        raise GeneIDSpaceError(
            f"Unknown id_type {id_type!r}. Known: {sorted(_ID_COLUMN_ALIASES)}"
        )

    candidates: list[tuple[str, pd.Index | pd.Series]] = [
        (f"var[{c!r}]", adata.var[c])
        for c in _ID_COLUMN_ALIASES[id_type]
        if c in adata.var.columns
    ]
    candidates.append(("var_names", adata.var_names))

    scored = [(name, vals, _valid_fraction(vals, id_type)) for name, vals in candidates]
    scored.sort(key=lambda t: t[2], reverse=True)
    best_name, best_vals, best_frac = scored[0]

    if best_frac < min_valid_frac:
        raise GeneIDSpaceError(
            f"No var column carries {id_type!r}. Best candidate {best_name} matched "
            f"{best_frac:.1%} of {adata.n_vars} genes (need {min_valid_frac:.0%}). "
            f"Available var columns: {list(adata.var.columns)}. "
            f"Pass allow_symbol_fallback=True only if you have verified the priors "
            f"key on the same space."
        )
    return best_name, best_vals, best_frac


def gene_index(
    adata: ad.AnnData,
    id_type: str = "ensembl_gene_id",
    *,
    allow_symbol_fallback: bool = False,
    min_valid_frac: float = 0.9,
) -> dict[str, int]:
    """Canonical {gene_id -> row position}. Every PriorSource aligns to this.

    Strict by default: if `id_type` cannot be located in `adata.var`, this raises
    rather than quietly falling back to var_names.
    """
    try:
        source, values, frac = resolve_id_column(adata, id_type, min_valid_frac)
    except GeneIDSpaceError:
        if not allow_symbol_fallback:
            raise
        logger.warning(
            "gene_index: falling back to var_names for %r (allow_symbol_fallback=True). "
            "Priors keyed on %r will match nothing.",
            id_type,
            id_type,
        )
        source, values, frac = "var_names", adata.var_names, 0.0

    logger.info("gene_index: using %s for %s (%.1f%% valid)", source, id_type, 100 * frac)

    idx: dict[str, int] = {}
    duplicates = 0
    for i, g in enumerate(values):
        key = str(g)
        if key in idx:
            duplicates += 1
            continue
        idx[key] = i
    if duplicates:
        logger.warning(
            "gene_index: %d duplicate IDs in %s; kept first occurrence of each.",
            duplicates,
            source,
        )
    return idx


def coverage(gene_index: dict[str, int], gene_ids: Iterable[str]) -> float:
    """Fraction of `gene_ids` present in the master gene space."""
    ids = [str(g) for g in gene_ids]
    if not ids:
        return 0.0
    return sum(g in gene_index for g in ids) / len(ids)


def assert_coverage(
    gene_index: dict[str, int],
    gene_ids: Iterable[str],
    *,
    source: str,
    min_frac: float = 0.5,
    raise_below: float = 0.0,
) -> float:
    """Warn (or raise) when a prior source maps poorly onto the master gene space.

    A source that maps ~0% is the signature of an ID-space mismatch, not of
    genuinely disjoint biology -- catch it here rather than downstream where it
    looks like "this prior just has no edges".
    """
    frac = coverage(gene_index, gene_ids)
    if frac <= raise_below:
        raise GeneIDSpaceError(
            f"{source}: {frac:.1%} of IDs map into the master gene space. "
            f"This is almost certainly an ID-space mismatch."
        )
    if frac < min_frac:
        logger.warning(
            "%s: only %.1f%% of IDs map into the master gene space (expected >=%.0f%%).",
            source,
            100 * frac,
            100 * min_frac,
        )
    return frac


# ---------------------------------------------------------------- loading --


def load_h5ad(path: str | Path, backed: str | None = None) -> ad.AnnData:
    """Load an AnnData. `backed='r'` keeps X on disk (schema inspection only)."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"No such AnnData: {path}")
    logger.info("Loading %s (backed=%s)", path, backed)
    return ad.read_h5ad(path, backed=backed)


def load_challenge_config(path: str | Path) -> dict:
    """Read a challenges/<year>/config.yaml."""
    return yaml.safe_load(Path(path).expanduser().read_text())


def load_challenge_split(
    challenge_config: str | Path | dict,
    split: str = "all",
    *,
    dev: bool = True,
    backed: str | None = None,
) -> ad.AnnData:
    """Load a challenge split: 'all' | 'train' | 'holdout'.

    For 2025 the scheme is `carve_from_training`: 'train'/'holdout' are carved
    locally and deterministically from the training file (see `carve_holdout`),
    mirroring challenge-time conditions when only names/counts existed for
    held-out perturbations. (Arc released the real validation/test AnnData on
    2025-12-16; an external-holdout scheme against those is pending.)

    `dev=True` uses the smaller `dev_file` for iteration.
    """
    cfg = (
        challenge_config
        if isinstance(challenge_config, dict)
        else load_challenge_config(challenge_config)
    )
    data_dir = Path(cfg["data_dir"]).expanduser()
    fname = cfg["dev_file"] if dev and cfg.get("dev_file") else cfg["source_file"]
    adata = load_h5ad(data_dir / fname, backed=backed)

    if split == "all":
        return adata

    scheme = cfg.get("splits", {}).get("scheme")
    if scheme != "carve_from_training":
        raise NotImplementedError(
            f"Split scheme {scheme!r} is not implemented. For 2026, add it here."
        )

    pert_col = cfg.get("pert_col", "target_gene")
    control = cfg.get("control_label", "non-targeting")
    sp = cfg["splits"]
    train_perts, holdout_perts = carve_holdout(
        adata,
        n_holdout=sp["holdout_perturbations"],
        pert_col=pert_col,
        control_label=control,
        seed=sp.get("seed", 0),
    )
    keep = train_perts if split == "train" else holdout_perts
    # Controls belong to BOTH sides: every perturbation is scored against them.
    mask = adata.obs[pert_col].astype(str).isin(set(keep) | {control})
    return adata[mask.to_numpy()].copy()


def carve_holdout(
    adata: ad.AnnData,
    n_holdout: int,
    *,
    pert_col: str = "target_gene",
    control_label: str = "non-targeting",
    seed: int = 0,
    min_cells: int = 30,
) -> tuple[list[str], list[str]]:
    """Deterministically split perturbations into (train, holdout).

    Only perturbations with >= `min_cells` cells are eligible for holdout: a
    9-cell perturbation gives a pseudobulk too noisy to score meaningfully, so
    holding it out measures sampling noise rather than model skill.
    """
    counts = adata.obs[pert_col].astype(str).value_counts()
    perts = [p for p in counts.index if p != control_label]
    eligible = sorted(p for p in perts if counts[p] >= min_cells)

    if len(eligible) < n_holdout:
        logger.warning(
            "carve_holdout: only %d perturbations have >=%d cells; holding out all of "
            "them (requested %d).",
            len(eligible),
            min_cells,
            n_holdout,
        )
        n_holdout = len(eligible)

    rng = np.random.default_rng(seed)
    holdout = sorted(rng.choice(eligible, size=n_holdout, replace=False).tolist())
    train = sorted(set(perts) - set(holdout))
    logger.info(
        "carve_holdout: %d train / %d holdout perturbations (seed=%d)",
        len(train),
        len(holdout),
        seed,
    )
    return train, holdout


def normalize_counts(
    adata: ad.AnnData, target_sum: float = 1e4, *, inplace: bool = False
) -> ad.AnnData:
    """Counts -> library-size-normalized log1p, the space cell-eval expects.

    cell-eval refuses discrete (raw count) input unless allow_discrete=True,
    because MAE on raw counts is dominated by library-size differences.

    `inplace=True` skips the defensive copy. On the full 2025 file that copy is
    ~3 GB and was the slowest single step in the pipeline; pass it whenever the
    caller discards the original immediately (as the dry run does).
    """
    import scanpy as sc

    out = adata if inplace else adata.copy()
    sc.pp.normalize_total(out, target_sum=target_sum)
    sc.pp.log1p(out)
    return out


def is_discrete(adata: ad.AnnData, n_check: int = 1000) -> bool:
    """True if X looks like raw integer counts."""
    import scipy.sparse as sp

    sub = adata.X[: min(n_check, adata.n_obs)]
    arr = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    return bool(np.allclose(arr, np.round(arr)))
