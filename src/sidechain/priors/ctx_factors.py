"""The per-context expressed-factor family: which RNA-binding proteins and which parts of the miRNA
machinery a challenge context's control cells express, and each gene's CisBP-RNA motif sites kept
only for the factors that context expresses (T120, the second hand-off; final-phase: controls).

    .venv/bin/python scripts/t120_build_control_factors.py                       # every vcc2026/context_*.h5ad
    .venv/bin/python scripts/t120_build_control_factors.py --h5ad ~/data/sidechain/vcc2026/context_D.h5ad

Reads a context's control h5ad (raw counts, cells x the 18,533-gene axis), the CisBP-RNA tables
(``derived/cisbp-rna-2.0/rbp_motifs.parquet`` for the factor catalogue with Ensembl ids,
``rbp_site_counts.parquet`` for the per-gene, per-RBP universal counts), the ENCODE eCLIP manifest
for its RBP names, and DepMap 24Q4's protein-coding expression for the check. Writes, under
``derived/control-factor-sets/``: ``factor_expression.parquet`` (one row per factor x context:
pseudobulk CPM, the share of cells detecting it, the expressed flag at ``CPM_MIN``),
``ctx_site_counts.parquet`` (one row per gene with a CisBP row; per context the direct-RBP sites
from expressed factors, from silent ones, the expressed RBPs with a site, the expressed share, and
the sites of the VARIABLE factors the context expresses -- those on in some of the contexts built
together and off in others, which is where the contexts differ: the shared factors dominate every
gene's sum, so the expressed sums of two contexts rank the genes alike),
``depmap_check.json`` and a LINEAGE.json entry.

**Expressed** means pseudobulk CPM at or above ``CPM_MIN`` over the context's control cells
(the counts of all cells summed, per million); the CPM and the detected share are written so a
reader can re-threshold. A factor absent from the axis is unknown there, counted in neither sum.
**The check**: the context's log CPM over the factor list is rank-correlated with every DepMap
model's log2(TPM+1); the line a context reads as (T88: A Jurkat, B HeLa, C CAL-33) should rank
near the top of 1,673 models, and the top five are listed for a context with no name yet.

The idea and the results: private ``research/ideas/ptr-feature-table.md`` § Outcome; the brief:
private ``handoffs/2026-10-16-t120-control-cell-factor-set.md``.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sidechain.priors.posttx_mirna import DATA_ROOT, sha256_of

DERIVED = "derived/control-factor-sets"
FACTOR_FILE = "factor_expression.parquet"
GENE_FILE = "ctx_site_counts.parquet"
CHECK_FILE = "depmap_check.json"
CISBP_DERIVED = "derived/cisbp-rna-2.0"
ECLIP_MANIFEST = "configs/manifests/encode_eclip.json"
DEPMAP_DIR = "external/figshare-27993248"
DEPMAP_EXPR = "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
DEPMAP_MODELS = "Model.csv"
CONTEXT_GLOB = "vcc2026/context_*.h5ad"

CPM_MIN = 1.0                 # pseudobulk CPM at or above this: the factor is expressed in the context
#: The miRNA pathway's protein parts, which carry no sequence motif and so never enter the site sums,
#: but are in the factor table because a context's miRNA machinery is read from them.
MIRNA_MACHINERY = ("AGO1", "AGO2", "AGO3", "AGO4", "DICER1", "DROSHA", "DGCR8", "TNRC6A", "TNRC6B",
                   "TNRC6C", "XPO5", "TARBP2")
#: What the challenge contexts read as on the named-line instrument (T88, 2026-10-13); a context not
#: listed has no name yet and the check lists its top matches instead (final-phase: controls).
NAMED_LINE = {"A": "JURKAT", "B": "HELA", "C": "CAL33"}

#: The per-context columns (``<X>`` is the context letter) and what each is.
CTX_COLUMNS = {
    "cx_sites_expressed_{X}": "direct-RBP motif sites (CisBP-RNA 2.0, rb_<RBP> columns) summed over the RBPs "
                              "context {X}'s control cells express at CPM >= 1",
    "cx_sites_silent_{X}": "the same sum over the direct RBPs context {X} does not express (on the axis, below the floor)",
    "cx_n_rbps_expressed_{X}": "expressed direct RBPs with at least one site on the gene, context {X}",
    "cx_frac_sites_expressed_{X}": "expressed / (expressed + silent) sites, context {X}; NaN where the gene has none",
    "cx_sites_variable_{X}": "sites of the VARIABLE direct RBPs context {X} expresses: the factors on in some of the "
                             "contexts built together and off in others (the list is in LINEAGE.json); the part of "
                             "the expressed sum that differs between contexts",
    "cx_n_rbps_variable_{X}": "variable direct RBPs expressed in context {X} with at least one site on the gene",
}


def _code_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:                                          # noqa: BLE001
        return "unknown"


def context_letter(h5ad: Path) -> str:
    m = re.search(r"context_([A-Za-z0-9]+)\.h5ad$", h5ad.name)
    if not m:
        raise ValueError(f"{h5ad.name}: not a context_<X>.h5ad file")
    return m.group(1).upper()


def control_expression(h5ad: Path) -> pd.DataFrame:
    """One row per axis gene: pseudobulk CPM and the share of cells detecting it, from raw counts."""
    import anndata as ad
    import scipy.sparse as sp

    a = ad.read_h5ad(h5ad)
    X = a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.tocsr()
    counts = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
    detected = np.asarray(X.getnnz(axis=0)).ravel().astype(np.float64) / max(X.shape[0], 1)
    total = counts.sum()
    if total <= 0:
        raise ValueError(f"{h5ad.name}: no counts")
    return pd.DataFrame({"symbol": list(a.var_names), "counts": counts, "cpm": counts / total * 1e6,
                         "frac_cells": detected, "n_cells": int(X.shape[0])})


def factor_catalogue(motifs: pd.DataFrame, eclip: list[str]) -> pd.DataFrame:
    """One row per factor symbol: its Ensembl id where CisBP gives one, and which lists carry it."""
    direct = set(motifs.loc[motifs["status"] == "D", "rbp"])
    inferred = set(motifs.loc[motifs["status"] == "I", "rbp"]) - direct
    ensembl = motifs.drop_duplicates("rbp").set_index("rbp")["rbp_ensembl_id"].to_dict()
    symbols = sorted(set(motifs["rbp"]) | set(eclip) | set(MIRNA_MACHINERY))
    return pd.DataFrame({
        "symbol": symbols,
        "ensembl_id": [ensembl.get(s, "") for s in symbols],
        "cisbp": ["direct" if s in direct else "inferred" if s in inferred else "" for s in symbols],
        "eclip": [s in set(eclip) for s in symbols],
        "mirna_machinery": [s in MIRNA_MACHINERY for s in symbols],
    })


def read_eclip_rbps(manifest: Path) -> list[str]:
    if not manifest.exists():
        return []
    names = set()
    for e in json.loads(manifest.read_text()):
        parts = str(e.get("note", "")).split()
        if len(parts) >= 3 and parts[0] == "eclip":
            names.add(parts[2])
    return sorted(names)


def depmap_check(factor_expr: pd.DataFrame, depmap: Path, *, named: dict[str, str] = NAMED_LINE,
                 top: int = 5) -> dict:
    """Rank agreement of each context's factor vector with every DepMap model's expression."""
    expr_path, model_path = depmap / DEPMAP_EXPR, depmap / DEPMAP_MODELS
    if not expr_path.exists() or not model_path.exists():
        return {"skipped": f"DepMap files not under {depmap}"}
    from scipy.stats import spearmanr

    header = pd.read_csv(expr_path, nrows=0).columns.tolist()
    want = set(factor_expr["symbol"])
    cols = [c for c in header[1:] if c.split(" (")[0] in want]
    table = pd.read_csv(expr_path, usecols=[header[0], *cols], index_col=0)
    table.columns = [c.split(" (")[0] for c in table.columns]
    models = pd.read_csv(model_path, usecols=["ModelID", "StrippedCellLineName"]).set_index("ModelID")
    names = models["StrippedCellLineName"].reindex(table.index).fillna("").str.upper()
    out = {"models": int(len(table)), "factors_shared": int(len(cols)), "contexts": {}}
    for ctx, g in factor_expr.groupby("context"):
        v = g.set_index("symbol")["log1p_cpm"].reindex(table.columns)
        ok = v.notna().to_numpy()
        rhos = np.array([spearmanr(v.to_numpy()[ok], table.iloc[i].to_numpy(dtype=float)[ok])[0]
                         for i in range(len(table))])
        order = np.argsort(-rhos)
        ranked = [(str(names.iloc[i]), float(rhos[i])) for i in order]
        entry = {"factors_used": int(ok.sum()),
                 "top": [{"line": n, "rho": round(r, 3)} for n, r in ranked[:top]]}
        line = named.get(ctx)
        if line:
            pos = next((k for k, (n, _) in enumerate(ranked) if n == line), None)
            entry["named_line"] = line
            entry["named_rank"] = None if pos is None else pos + 1
            entry["named_rho"] = None if pos is None else round(ranked[pos][1], 3)
        for other in sorted(set(named.values()) | {"K562", "HCT116", "HEPG2", "RPE1"}):
            pos = next((k for k, (n, _) in enumerate(ranked) if n == other), None)
            if pos is not None:
                entry.setdefault("our_lines", {})[other] = {"rank": pos + 1, "rho": round(ranked[pos][1], 3)}
        out["contexts"][ctx] = entry
    return out


def build_control_factors(root: Path = DATA_ROOT, *, h5ads: list[Path] | None = None, cpm_min: float = CPM_MIN,
                          depmap: bool = True, eclip_manifest: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The factor table, the per-gene per-context table and the LINEAGE entry (not written)."""
    root = Path(root).expanduser()
    h5ads = sorted(root.glob(CONTEXT_GLOB)) if h5ads is None else [Path(p).expanduser() for p in h5ads]
    if not h5ads:
        raise ValueError(f"no control h5ads: {root / CONTEXT_GLOB}")
    motifs = pd.read_parquet(root / CISBP_DERIVED / "rbp_motifs.parquet")
    sites = pd.read_parquet(root / CISBP_DERIVED / "rbp_site_counts.parquet")
    manifest = Path(eclip_manifest) if eclip_manifest else Path(ECLIP_MANIFEST)
    catalogue = factor_catalogue(motifs, read_eclip_rbps(manifest))
    direct = sorted(set(motifs.loc[motifs["status"] == "D", "rbp"]))
    direct_cols = {r: f"rb_{r}" for r in direct if f"rb_{r}" in sites.columns}

    factor_rows = []
    genes = sites[["symbol", "gene_id", "transcript_id"]].copy()
    counts = sites[list(direct_cols.values())].to_numpy(dtype=np.float64)
    columns: dict[str, str] = {"symbol": "TargetScan's gene symbol", "gene_id": "Ensembl gene id",
                               "transcript_id": "the representative transcript the sites were counted on"}
    inputs: dict = {"rbp_motifs": {"path": str(root / CISBP_DERIVED / "rbp_motifs.parquet"),
                                   "sha256": sha256_of(root / CISBP_DERIVED / "rbp_motifs.parquet")},
                    "rbp_site_counts": {"path": str(root / CISBP_DERIVED / "rbp_site_counts.parquet"),
                                        "sha256": sha256_of(root / CISBP_DERIVED / "rbp_site_counts.parquet")},
                    "contexts": {}}
    per_context: dict = {}
    expressed_sets: dict[str, set] = {}
    for h5ad in h5ads:
        ctx = context_letter(h5ad)
        expr = control_expression(h5ad).set_index("symbol")
        on_axis = catalogue["symbol"].isin(expr.index)
        cpm = catalogue["symbol"].map(expr["cpm"])
        frac = catalogue["symbol"].map(expr["frac_cells"])
        expressed = (cpm >= cpm_min).fillna(False)
        for sym, ax, c, f, e, ens, cb, ec, mm in zip(catalogue["symbol"], on_axis, cpm, frac, expressed,
                                                     catalogue["ensembl_id"], catalogue["cisbp"],
                                                     catalogue["eclip"], catalogue["mirna_machinery"]):
            factor_rows.append({"context": ctx, "symbol": sym, "ensembl_id": ens, "cisbp": cb, "eclip": bool(ec),
                                "mirna_machinery": bool(mm), "on_axis": bool(ax),
                                "cpm": float(c) if ax else np.nan, "log1p_cpm": float(np.log1p(c)) if ax else np.nan,
                                "frac_cells": float(f) if ax else np.nan, "expressed": bool(e) if ax else False})
        exp_set = set(catalogue.loc[expressed & on_axis, "symbol"])
        silent_set = set(catalogue.loc[(~expressed) & on_axis, "symbol"])
        expressed_sets[ctx] = exp_set
        is_exp = np.array([r in exp_set for r in direct_cols], dtype=bool)
        is_sil = np.array([r in silent_set for r in direct_cols], dtype=bool)
        e_sum = counts[:, is_exp].sum(axis=1)
        s_sum = counts[:, is_sil].sum(axis=1)
        genes[f"cx_sites_expressed_{ctx}"] = e_sum
        genes[f"cx_sites_silent_{ctx}"] = s_sum
        genes[f"cx_n_rbps_expressed_{ctx}"] = (counts[:, is_exp] > 0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            genes[f"cx_frac_sites_expressed_{ctx}"] = np.where(e_sum + s_sum > 0, e_sum / (e_sum + s_sum), np.nan)
        for key, desc in CTX_COLUMNS.items():
            columns[key.format(X=ctx)] = desc.format(X=ctx)
        n_cells = int(expr["n_cells"].iloc[0])
        inputs["contexts"][ctx] = {"h5ad": str(h5ad), "sha256": sha256_of(h5ad), "cells": n_cells,
                                   "genes": int(len(expr))}
        per_context[ctx] = {"cells": n_cells, "factors": int(len(catalogue)), "factors_on_axis": int(on_axis.sum()),
                            "expressed": int((expressed & on_axis).sum()),
                            "direct_rbps_expressed": int(is_exp.sum()), "direct_rbps_silent": int(is_sil.sum()),
                            "direct_rbps_off_axis": int(len(direct_cols) - is_exp.sum() - is_sil.sum()),
                            "mirna_machinery_expressed": sorted(exp_set & set(MIRNA_MACHINERY)),
                            "median_frac_sites_expressed": float(np.nanmedian(genes[f"cx_frac_sites_expressed_{ctx}"]))}
    # the variable factors: direct RBPs expressed in some of the contexts built together and not in
    # others; their sites are the part of a context's expressed sum that tells it from another
    on_somewhere = set().union(*expressed_sets.values()) if expressed_sets else set()
    on_everywhere = set.intersection(*expressed_sets.values()) if expressed_sets else set()
    variable = sorted((on_somewhere - on_everywhere) & set(direct_cols))
    is_var = np.array([r in set(variable) for r in direct_cols], dtype=bool)
    for ctx, exp_set in expressed_sets.items():
        is_exp = np.array([r in exp_set for r in direct_cols], dtype=bool) & is_var
        genes[f"cx_sites_variable_{ctx}"] = counts[:, is_exp].sum(axis=1)
        genes[f"cx_n_rbps_variable_{ctx}"] = (counts[:, is_exp] > 0).sum(axis=1)
        per_context[ctx]["variable_rbps_expressed"] = sorted(exp_set & set(variable))
    factors = pd.DataFrame(factor_rows)
    check = depmap_check(factors, root / DEPMAP_DIR) if depmap else {"skipped": "not asked"}
    entry = {
        "dataset": "control-factor-sets",
        "what": "per challenge context, from its control cells alone: the expressed set of RBPs and miRNA-machinery "
                "genes, and each gene's CisBP-RNA direct-RBP motif sites split into those of expressed and of silent "
                "factors (T120; final-phase: controls)",
        "inputs": inputs, "code": "src/sidechain/priors/ctx_factors.py::build_control_factors",
        "code_sha": _code_sha(), "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "rule": {"expressed": f"pseudobulk CPM >= {cpm_min} over the context's control cells",
                 "sites": "rb_<RBP> universal counts (relative score >= 0.85) of direct RBPs only; a factor off the "
                          "axis is unknown and in neither sum"},
        "contexts": per_context, "variable_direct_rbps": variable, "columns": columns, "depmap_check": check,
        "licence": "the site counts derive from CisBP-RNA 2.0 (UNSTATED, both encumbrance flags); the factor "
                   "expression from the challenge's control bundle",
    }
    return factors, genes, entry


def write_control_factors(root: Path = DATA_ROOT, **kw) -> Path:
    root = Path(root).expanduser()
    out = root / DERIVED
    out.mkdir(parents=True, exist_ok=True)
    factors, genes, entry = build_control_factors(root, **kw)
    factors.to_parquet(out / FACTOR_FILE, index=False)
    genes.to_parquet(out / GENE_FILE, index=False)
    (out / CHECK_FILE).write_text(json.dumps(entry["depmap_check"], indent=2) + "\n")
    entry = {**entry, "sha256": sha256_of(out / GENE_FILE), "bytes": (out / GENE_FILE).stat().st_size,
             "factor_file": str(out / FACTOR_FILE), "factor_sha256": sha256_of(out / FACTOR_FILE)}
    path = out / "LINEAGE.json"
    lineage = json.loads(path.read_text()) if path.exists() else {"schema_version": 2, "entries": {}}
    lineage.setdefault("entries", {})["control-factor-sets/ctx_site_counts"] = entry
    path.write_text(json.dumps(lineage, indent=2) + "\n")
    return out
