"""A per-gene multiplier on the pooled delta, applied before alpha and the knockdown pin.

`--gene-factor PATH` on `sidechain.submit.build` and `sidechain.eval.loco`: a parquet or CSV table with a
`gene` column (symbols on the build's axis) and a `factor` column (finite, non-negative). Every gene the
table does not name keeps 1.0, so the knob is exactly "multiply these genes' transferred log2FC by this
much" and nothing else; a gene named twice, a negative or non-finite factor, or a missing column is refused
before any work. The record written beside the build (`<stem>.gene_factor.json`, and `gene_factor` in
loco's summary) carries the file's sha256 and what matched the axis, so a run can be tied to its table.

An optional `context` column makes the table per line (T120): a row with a context applies only in that
context, a row with an empty context applies in every context, and a (gene, context) pair named twice is
refused -- which includes a gene with both an every-context row and a row of its own for some context
(override or multiply would be a guess). `submit.build` applies each context's rows to that context's
cells; `eval.loco` predicts one held-out line, so `--gene-factor-context CTX` names which context's rows
apply there (the rows with an empty context always do). A table without the column reads exactly as
before, record included.

Where it acts: the last thing before alpha, on the pooled vector every other knob has already shaped (the
neighbour arm's fusion, the basal slope), in both channels of a two-channel emission; the target's own gene
is pinned after it as before. It is not a shrinkage rule (`s`: a function of each fold change's own size
and spread) and not alpha (`a`: one constant for every gene). T102's trust shrink was its first arm:
private research/ideas/utr-layer-context-invariance.md; the per-line form: private tasks/T120.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

CONTEXT = "context"


def read_gene_factor_table(path: Path) -> pd.DataFrame:
    """`gene`, `factor` and, when the file has it, `context` (None where a row applies everywhere)."""
    path = Path(path).expanduser()
    if not path.exists():
        raise SystemExit(f"--gene-factor: {path} does not exist")
    if path.suffix == ".parquet":
        tab = pd.read_parquet(path)
    else:
        tab = pd.read_csv(path, dtype={CONTEXT: str} if _csv_has_context(path) else None)
    missing = [c for c in ("gene", "factor") if c not in tab.columns]
    if missing:
        raise SystemExit(f"--gene-factor: {path} lacks the column(s) {missing}; it needs `gene` and `factor`")
    per_line = CONTEXT in tab.columns
    tab = tab[["gene", "factor", *([CONTEXT] if per_line else [])]].copy()
    tab["gene"] = tab["gene"].astype(str)
    f = pd.to_numeric(tab["factor"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(f).all():
        raise SystemExit(f"--gene-factor: {path} carries a non-finite factor")
    if (f < 0).any():
        raise SystemExit(f"--gene-factor: {path} carries a negative factor; a multiplier on a log2FC is >= 0")
    tab["factor"] = f
    if not per_line:
        if tab["gene"].duplicated().any():
            dup = tab["gene"][tab["gene"].duplicated()].iloc[0]
            raise SystemExit(f"--gene-factor: {path} names {dup!r} more than once")
        return tab
    ctx = tab[CONTEXT].astype(object)
    ctx = ctx.where(ctx.notna(), None).map(lambda c: None if c is None or str(c).strip() == "" else str(c).strip())
    tab[CONTEXT] = ctx
    key = tab["gene"] + "\x00" + tab[CONTEXT].fillna("\x01")
    if key.duplicated().any():
        row = tab[key.duplicated()].iloc[0]
        where = "every context" if row[CONTEXT] is None else f"context {row[CONTEXT]!r}"
        raise SystemExit(f"--gene-factor: {path} names {row['gene']!r} more than once for {where}")
    everywhere = set(tab.loc[tab[CONTEXT].isna(), "gene"])
    both = tab[tab[CONTEXT].notna() & tab["gene"].isin(everywhere)]
    if len(both):
        row = both.iloc[0]
        raise SystemExit(f"--gene-factor: {path} names {row['gene']!r} for every context AND for context "
                         f"{row[CONTEXT]!r}; give it one or the other (override or multiply would be a guess)")
    return tab


def _csv_has_context(path: Path) -> bool:
    with open(path) as fh:
        return CONTEXT in fh.readline().strip().split(",")


def table_contexts(tab: pd.DataFrame) -> list[str]:
    """The contexts a per-line table names, sorted; [] for a table without the column."""
    if CONTEXT not in tab.columns:
        return []
    return sorted({c for c in tab[CONTEXT] if c is not None})


def _align(tab: pd.DataFrame, axis: np.ndarray) -> tuple[np.ndarray, dict]:
    pos = {g: i for i, g in enumerate(np.asarray(axis, dtype=object).astype(str))}
    factor = np.ones(len(axis), dtype=np.float64)
    hit = tab["gene"].map(pos)
    on_axis = hit.notna().to_numpy()
    factor[hit[on_axis].astype(int).to_numpy()] = tab["factor"].to_numpy()[on_axis]
    matched = factor[hit[on_axis].astype(int).to_numpy()] if on_axis.any() else np.array([])
    counts = {
        "rows": int(len(tab)), "matched": int(on_axis.sum()), "unmatched_in_table": int((~on_axis).sum()),
        "axis_genes": int(len(axis)),
        "mean_on_matched": float(matched.mean()) if len(matched) else None,
        "min": float(factor.min()), "max": float(factor.max()),
        "below_one": int((factor < 1.0).sum()), "above_one": int((factor > 1.0).sum()),
        "zero": int((factor == 0.0).sum()),
    }
    return factor, counts


def _rows_for(tab: pd.DataFrame, context: str | None) -> pd.DataFrame:
    if CONTEXT not in tab.columns:
        return tab
    keep = tab[CONTEXT].isna() if context is None else (tab[CONTEXT].isna() | (tab[CONTEXT] == context))
    return tab[keep]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()


def load_gene_factor(path: Path, axis: np.ndarray, context: str | None = None) -> tuple[np.ndarray, dict]:
    """The factor vector aligned to `axis` (1.0 where the table names no gene), and its record.

    `context` picks a per-line table's rows for one held-out line (`eval.loco --gene-factor-context`): that
    context's rows plus the every-context ones. It is refused on a table without a `context` column, and when
    the table has no row for it, so a misspelt context never scores as the plain recipe."""
    path = Path(path).expanduser()
    tab = read_gene_factor_table(path)
    named = table_contexts(tab)
    if context is not None:
        if CONTEXT not in tab.columns:
            raise SystemExit(f"--gene-factor-context {context!r}: {path} has no `context` column, so its rows "
                             "apply everywhere already; drop the flag")
        if context not in named:
            raise SystemExit(f"--gene-factor-context {context!r}: {path} has no row for it (its contexts: "
                             f"{', '.join(named) or 'none'})")
    factor, counts = _align(_rows_for(tab, context), axis)
    record = {"path": str(path), "sha256": _sha256(path), **counts}
    if CONTEXT in tab.columns:
        record = {**record, "context": context, "table_rows": int(len(tab)), "table_contexts": named}
    if record["matched"] == 0:
        hint = (f"; it names rows for {', '.join(named)}: pass --gene-factor-context"
                if context is None and named else "")
        raise SystemExit(f"--gene-factor: no gene of {path} is on the build's axis ({len(axis)} genes)"
                         + (f" for context {context!r}" if context is not None else "") + hint)
    return factor, record


def load_gene_factors(path: Path, axis: np.ndarray, contexts: list[str]) -> tuple[dict[str, np.ndarray], dict]:
    """One factor vector per build context, and the record.

    A table without a `context` column gives every context the SAME array and the record `load_gene_factor`
    writes (what a build before the column wrote). A per-line table gives each context its own rows plus the
    every-context ones; a context of the table the build does not hold is recorded, and a table none of whose
    rows applies to any context of the build is refused."""
    path = Path(path).expanduser()
    tab = read_gene_factor_table(path)
    if CONTEXT not in tab.columns:
        factor, record = load_gene_factor(path, axis)
        return {c: factor for c in contexts}, record
    named = table_contexts(tab)
    per, factors = {}, {}
    for c in contexts:
        factors[c], per[c] = _align(_rows_for(tab, c), axis)
    record = {"path": str(path), "sha256": _sha256(path), "rows": int(len(tab)),
              "rows_every_context": int(tab[CONTEXT].isna().sum()), "table_contexts": named,
              "table_contexts_not_in_build": [c for c in named if c not in contexts],
              "contexts": per}
    if not any(per[c]["matched"] for c in contexts):
        raise SystemExit(f"--gene-factor: no row of {path} applies to a gene of this build in any of its "
                         f"contexts ({', '.join(contexts)}); the table names {', '.join(named) or 'none'}")
    return factors, record


def same_factor_groups(factor_of: dict[str, np.ndarray], contexts) -> list[list[str]]:
    """`contexts` partitioned by equal factor vectors, in order; one group when every context reads the same."""
    groups: list[list[str]] = []
    for c in contexts:
        for g in groups:
            if factor_of[g[0]] is factor_of[c] or np.array_equal(factor_of[g[0]], factor_of[c]):
                g.append(c)
                break
        else:
            groups.append([c])
    return groups
