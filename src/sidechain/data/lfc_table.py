"""A delta source of published log fold changes with no cells behind them, pooled by `pooled_delta` like a pseudobulk.

    uv run python -m sidechain.data.lfc_table TABLE.tsv.gz --keep pert_counts.csv --out CACHE.npz

The variance comes from the adjusted p: invert Benjamini-Hochberg within one target's genes, take the two-sided
normal z, `var = (|lfc| / z)^2`; a row at `pval_adj >= SATURATED_P` abstains (`var = inf`, zero weight). Reads a
Feng et al. 2026-style TSV (`Target, Expressed_Gene_Symbol, lfc, pval_adj`); writes an `LfcTable` .npz.
Notes: private/code-notes/src/sidechain/data/lfc_table.py.md
"""
from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import norm

# adjusted p at or above which a row abstains (infinite variance)
SATURATED_P = 0.99

# floor on a recovered nominal p
MIN_NOMINAL_P = 1e-300


@dataclass
class LfcTable:
    """Per-(target, gene) log fold changes and their derived variances.

    `var` is `inf` wherever the source abstains, which makes `1/var` exactly 0
    and needs no special-casing at the pooling site.
    """

    labels: list[str]           # perturbation targets, row order of `lfc`
    genes: np.ndarray           # gene symbols, column order of `lfc`
    lfc: np.ndarray             # (L, G) float64
    var: np.ndarray             # (L, G) float64; inf = abstain
    source: str = ""
    context: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def n_usable(self) -> np.ndarray:
        """Per target, how many genes carry a finite weight."""
        return np.isfinite(self.var).sum(axis=1)

    def effect(self, target: str) -> tuple[np.ndarray, np.ndarray] | None:
        """(log2 fold change, variance) for `target`, or None if absent: `_log2fc_with_var`'s shape."""
        try:
            i = self.labels.index(target)
        except ValueError:
            return None
        return self.lfc[i], self.var[i]

    def subset(self, labels: list[str], genes) -> LfcTable:
        """A copy restricted to these labels and genes, in the order given; one it does not carry raises."""
        if len(set(map(str, self.genes))) != len(self.genes):
            raise ValueError(f"{self.source or 'LfcTable'}: duplicate gene names; a name "
                             "must identify one column")
        lpos = {x: i for i, x in enumerate(self.labels)}
        gpos = {str(g): i for i, g in enumerate(self.genes)}
        miss_l = [x for x in labels if x not in lpos]
        miss_g = [str(g) for g in genes if str(g) not in gpos]
        if miss_l or miss_g:
            raise KeyError(f"{self.source or 'LfcTable'}: {len(miss_l)} label(s) and "
                           f"{len(miss_g)} gene(s) not present, first few "
                           f"{miss_l[:3]} {miss_g[:3]}")
        r = np.array([lpos[x] for x in labels], dtype=np.int64)
        c = np.array([gpos[str(g)] for g in genes], dtype=np.int64)
        return LfcTable(labels=list(labels), genes=np.asarray([str(g) for g in genes]),
                        lfc=self.lfc[np.ix_(r, c)], var=self.var[np.ix_(r, c)],
                        source=self.source, context=self.context, notes=dict(self.notes))

    def scrambled(self, seed: int) -> LfcTable:
        """The same table with its voting targets' profiles permuted among themselves (seeded).

        A target with a finite variance still votes and one that abstained still abstains.
        """
        voting = np.flatnonzero(np.isfinite(self.var).any(axis=1))
        if voting.size < 2:
            raise ValueError("fewer than two voting targets; nothing to scramble")
        perm = voting.copy()
        np.random.default_rng(seed).shuffle(perm)
        lfc, var = self.lfc.copy(), self.var.copy()
        lfc[voting], var[voting] = self.lfc[perm], self.var[perm]
        return LfcTable(labels=list(self.labels), genes=np.asarray(self.genes), lfc=lfc,
                        var=var, source=f"{self.source} [scrambled seed={seed}]",
                        context=self.context, notes={**self.notes, "scrambled_seed": seed})

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path).expanduser(),
            labels=np.asarray(self.labels, dtype=object),
            genes=np.asarray(self.genes, dtype=object),
            lfc=self.lfc, var=self.var,
            source=np.asarray([self.source], dtype=object),
            context=np.asarray([self.context], dtype=object),
            # how the table was built (a view's conditions, rho and gate; a scramble's seed)
            notes=np.asarray([json.dumps(self.notes, default=str)], dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> LfcTable:
        z = np.load(Path(path).expanduser(), allow_pickle=True)
        return cls(
            labels=[str(x) for x in z["labels"]],
            genes=z["genes"].astype(str),
            lfc=z["lfc"], var=z["var"],
            source=str(z["source"][0]), context=str(z["context"][0]),
            notes=json.loads(str(z["notes"][0])) if "notes" in z.files else {},
        )


def _nominal_from_bh(p_adj: np.ndarray) -> np.ndarray:
    """Undo a Benjamini-Hochberg adjustment within one family: `p_(i) = p_adj_(i) * i / m`.

    Exact where BH's running minimum `min_{j>=i} (m/j) * p_(j)` does not bind; ties are where it stops being exact.
    """
    m = p_adj.size
    order = np.argsort(p_adj, kind="stable")
    rank = np.empty(m, dtype=np.float64)
    rank[order] = np.arange(1, m + 1, dtype=np.float64)
    return np.clip(p_adj * rank / m, MIN_NOMINAL_P, 1.0)


def variance_from_pvalue(lfc: np.ndarray, p_adj: np.ndarray,
                         *, saturated_p: float = SATURATED_P) -> np.ndarray:
    """Standard-error-squared implied by an effect size and its adjusted p; `inf` where the row abstains."""
    var = np.full(lfc.shape, np.inf, dtype=np.float64)
    usable = np.isfinite(p_adj) & (p_adj < saturated_p) & np.isfinite(lfc) & (lfc != 0.0)
    if not usable.any():
        return var

    p_nom = _nominal_from_bh(p_adj)[usable]
    # two-sided z by `isf`, not `ndtri(1 - p/2)`: `1 - p/2` rounds to 1.0 below p ~ 1e-16, so z = inf, var = 0
    z = norm.isf(p_nom / 2.0)
    good = np.isfinite(z) & (z > 0)
    idx = np.flatnonzero(usable)[good]
    se = np.abs(lfc[idx]) / z[good]
    v = se**2
    # a variance that underflows to 0 abstains
    v[v <= 0] = np.inf
    var[idx] = v
    return var


def _open(path: Path):
    return gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open(newline="")


def read_feng_lfc(
    path: str | Path,
    *,
    keep: set[str] | None = None,
    target_col: str = "Target",
    gene_col: str = "Expressed_Gene_Symbol",
    effect_col: str = "lfc",
    pvalue_col: str = "pval_adj",
    context_col: str | None = None,
    context: str = "",
    source: str = "",
    saturated_p: float = SATURATED_P,
) -> LfcTable:
    """Read a Feng-style per-gene LFC table (.tsv or .tsv.gz) into an `LfcTable`, keeping `keep`'s targets.

    `context_col` with `context` selects one line of a per-line table.
    """
    path = Path(path).expanduser()
    per_target: dict[str, dict[str, tuple[float, float]]] = {}
    genes: dict[str, None] = {}          # insertion-ordered set
    skipped_context = 0

    with _open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        missing = {target_col, gene_col, effect_col, pvalue_col} - set(reader.fieldnames or [])
        if missing:
            raise KeyError(
                f"{path.name} lacks column(s) {sorted(missing)}; header is "
                f"{reader.fieldnames}. The spec block in configs/datasets.yaml names "
                "these -- declared, not guessed, so a renamed column stops the run."
            )
        if context_col and context_col not in (reader.fieldnames or []):
            raise KeyError(f"{path.name} has no context column {context_col!r}")

        for row in reader:
            if context_col and row[context_col] != context:
                skipped_context += 1
                continue
            target = row[target_col]
            if keep is not None and target not in keep:
                continue
            try:
                fc, p = float(row[effect_col]), float(row[pvalue_col])
            except (TypeError, ValueError):
                continue
            gene = row[gene_col]
            genes.setdefault(gene, None)
            per_target.setdefault(target, {})[gene] = (fc, p)

    labels = sorted(per_target)
    gene_list = np.array(list(genes), dtype=object).astype(str)
    gidx = {g: i for i, g in enumerate(gene_list)}

    lfc = np.zeros((len(labels), len(gene_list)), dtype=np.float64)
    var = np.full((len(labels), len(gene_list)), np.inf, dtype=np.float64)
    for r, target in enumerate(labels):
        cols = per_target[target]
        j = np.fromiter((gidx[g] for g in cols), dtype=np.int64, count=len(cols))
        fcs = np.fromiter((v[0] for v in cols.values()), dtype=np.float64, count=len(cols))
        ps = np.fromiter((v[1] for v in cols.values()), dtype=np.float64, count=len(cols))
        lfc[r, j] = fcs
        # the BH family is this target's own rows in the file, not the padded matrix
        var[r, j] = variance_from_pvalue(fcs, ps, saturated_p=saturated_p)

    return LfcTable(
        labels=labels, genes=gene_list, lfc=lfc, var=var,
        source=source or path.name, context=context,
        notes={"rows_skipped_other_context": skipped_context, "saturated_p": saturated_p},
    )


def main(argv: list[str] | None = None) -> int:
    """Build a cached LfcTable from a published per-gene LFC table.

        uv run python -m sidechain.data.lfc_table \\
            ~/data/sidechain/external/figshare-26819743/GenomeWideScreen_LFC_byGene.tsv.gz \\
            --keep ~/data/sidechain/vcc2026/pert_counts.csv --out ~/data/sidechain/cache/vcc2026/feng_genomewide_lfc.npz
    """
    import argparse
    import json

    ap = argparse.ArgumentParser(description=main.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", type=Path, help="the .tsv/.tsv.gz LFC table")
    ap.add_argument("--keep", type=Path,
                    help="CSV whose first column lists the targets to keep (e.g. pert_counts.csv)")
    ap.add_argument("--target-col", default="Target")
    ap.add_argument("--gene-col", default="Expressed_Gene_Symbol")
    ap.add_argument("--effect-col", default="lfc")
    ap.add_argument("--pvalue-col", default="pval_adj")
    ap.add_argument("--context-col", help="for a per-line table, e.g. Cell_Line")
    ap.add_argument("--context", default="", help="which context to select, or its name if per-file")
    ap.add_argument("--saturated-p", type=float, default=SATURATED_P,
                    help="adjusted p at or above which a row abstains (infinite variance)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    keep = None
    if args.keep:
        with Path(args.keep).expanduser().open(newline="") as fh:
            rows = list(csv.reader(fh))
        keep = {r[0].strip() for r in rows[1:] if r} - {"non-targeting", ""}

    table = read_feng_lfc(
        args.table, keep=keep, target_col=args.target_col, gene_col=args.gene_col,
        effect_col=args.effect_col, pvalue_col=args.pvalue_col,
        context_col=args.context_col, context=args.context,
        source=args.table.name, saturated_p=args.saturated_p,
    )
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    table.save(out)

    usable = table.n_usable
    print(json.dumps({
        "out": str(out),
        "targets": len(table.labels),
        "genes": int(table.genes.size),
        "context": table.context or None,
        # per target, how many genes carry a finite weight
        "genes_with_finite_weight": {
            "min": int(usable.min()) if usable.size else 0,
            "median": int(np.median(usable)) if usable.size else 0,
            "max": int(usable.max()) if usable.size else 0,
        },
        "targets_abstaining_entirely": int((usable == 0).sum()),
        "rows_skipped_other_context": table.notes.get("rows_skipped_other_context", 0),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
