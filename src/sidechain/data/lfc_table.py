"""A source that ships log fold changes directly, with no cells behind them.

Every source `pooled_delta` consumed until now was a `PseudobulkSums`: per-cell
CPM sums we accumulated ourselves, from which both the fold change and its
sampling variance fall out. Feng et al. 2026 is the first corpus that publishes
the *contrast already taken* -- `Target, Expressed_Gene_Symbol,
Expressed_Gene_Ens_ID, lfc, pval_adj` -- and ships no counts at all.

The obvious shortcut is to fabricate a `PseudobulkSums` whose CPM sums happen to
back-compute to Feng's numbers. That is rejected deliberately: it would put
invented per-cell statistics into the one object the rest of Sidechain trusts as
measured, where `qc_report` and anything else downstream would later read them as
real. A source that has no cells should not claim to have cells. So this module
is a second source *type*, and `pooled_delta` takes either.

WHERE THE VARIANCE COMES FROM, and why this is the delicate part
----------------------------------------------------------------
`pooled_delta` weights each source per gene by `1 / var`, and Feng publishes no
variance. It publishes an adjusted p-value, which is a statement about the same
underlying standard error, so the variance is recoverable:

    p_adj  --(invert BH)-->  p_nominal  --(two-sided normal)-->  z
    se = |lfc| / z          var = se^2

That is an approximation and it is worth being precise about which parts:

  * **Inverting BH is exact only where the step-up's cumulative minimum did not
    bind.** Benjamini-Hochberg sets p_adj_(i) = min over j >= i of (m/j)*p_(j)
    for p sorted ascending over a family of m tests. Where that minimum is not
    binding it reduces to p_adj_(i) = (m/i)*p_(i), which inverts to
    p_(i) = p_adj_(i) * i / m -- what `_nominal_from_bh` computes. The minimum
    binds mostly in the saturated tail, which we discard anyway (below).
  * **The family is assumed to be one target's genes.** Feng does not document
    what it adjusted over. Per-target across the expressed axis is the usual
    choice and the only one that makes the rank meaningful here. If it was in
    fact adjusted globally, every nominal p shifts by a constant factor -- which
    rescales all of Feng's variances together and therefore barely moves an
    inverse-variance pool relative to the *other* sources' weights. Stated so
    the assumption is visible rather than buried.

WHY SATURATED ROWS ABSTAIN RATHER THAN VOTE TOWARD ZERO
-------------------------------------------------------
98.9 % of the genome-wide table's rows have `pval_adj` > 0.99 (measured
2026-08-23 over all 43,187,656 rows). Read naively, `se = |lfc|/z` turns those
into *tiny* standard errors whenever `lfc` is near zero, hence enormous weights
-- Feng would then dominate the pool on exactly the genes where it measured
nothing. That is the failure mode this module exists to avoid, and it is not
hypothetical arithmetic: it is what a floor-the-variance implementation does.

So a saturated row gets **infinite variance, i.e. zero weight**: Feng abstains.
The justification is the corpus itself. At a median of 25 cells per target the
screen has almost no power, so "no detectable change" is close to uninformative,
and letting it vote toward zero would shrink other sources' real effects using
evidence that does not exist. Feng votes where it has signal -- a median of 58
genes per panel target -- and stays silent elsewhere. `pooled_delta` already
divides only where the accumulated weight is positive, so abstention composes
correctly with the other sources rather than leaving a hole.
"""
from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import norm

# A row at or above this adjusted p carries no usable information about its
# standard error, and is given zero weight. Not a tuning knob so much as the
# point where the published number stops varying: Feng's saturated rows sit at
# 0.999999927233465.
SATURATED_P = 0.99

# Clamps p_nominal away from 0. With the stable upper-tail quantile this is a
# floor on how much certainty a single row may claim, not a numerical guard.
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
        """(log2 fold change, variance) for `target`, or None if absent.

        The same shape `_log2fc_with_var` returns for a PseudobulkSums, which is
        the whole point -- `pooled_delta` cannot tell the two apart.
        """
        try:
            i = self.labels.index(target)
        except ValueError:
            return None
        return self.lfc[i], self.var[i]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path).expanduser(),
            labels=np.asarray(self.labels, dtype=object),
            genes=np.asarray(self.genes, dtype=object),
            lfc=self.lfc, var=self.var,
            source=np.asarray([self.source], dtype=object),
            context=np.asarray([self.context], dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> LfcTable:
        z = np.load(Path(path).expanduser(), allow_pickle=True)
        return cls(
            labels=[str(x) for x in z["labels"]],
            genes=z["genes"].astype(str),
            lfc=z["lfc"], var=z["var"],
            source=str(z["source"][0]), context=str(z["context"][0]),
        )


def _nominal_from_bh(p_adj: np.ndarray) -> np.ndarray:
    """Undo a Benjamini-Hochberg adjustment within one family of tests.

    BH sorts p ascending over m tests and reports
    `p_adj_(i) = min_{j>=i} (m/j) * p_(j)`. Where that running minimum is not
    binding this is just `(m/i) * p_(i)`, so the nominal p is
    `p_adj_(i) * i / m`. Ties -- and the saturated tail is one enormous tie --
    are where it stops being exact, which is a reason to discard that tail
    rather than to trust a number recovered from it.
    """
    m = p_adj.size
    order = np.argsort(p_adj, kind="stable")
    rank = np.empty(m, dtype=np.float64)
    rank[order] = np.arange(1, m + 1, dtype=np.float64)
    return np.clip(p_adj * rank / m, MIN_NOMINAL_P, 1.0)


def variance_from_pvalue(lfc: np.ndarray, p_adj: np.ndarray,
                         *, saturated_p: float = SATURATED_P) -> np.ndarray:
    """Standard-error-squared implied by an effect size and its adjusted p.

    Returns `inf` for rows the source cannot speak to, so `1/var` is 0 there.
    See this module's docstring for why abstaining beats voting toward zero.
    """
    var = np.full(lfc.shape, np.inf, dtype=np.float64)
    usable = np.isfinite(p_adj) & (p_adj < saturated_p) & np.isfinite(lfc) & (lfc != 0.0)
    if not usable.any():
        return var

    p_nom = _nominal_from_bh(p_adj)[usable]
    # Two-sided: z is the upper-tail quantile at p/2, computed with `isf` and
    # NOT as `ndtri(1 - p/2)`.
    #
    # That difference is not stylistic. `1 - p/2` rounds to exactly 1.0 in
    # float64 once p drops below ~1e-16, so `ndtri` returns `inf`, `se` becomes
    # `|lfc| / inf` = 0, the variance is 0, and the weight `1/var` is INFINITE.
    # The source's MOST significant genes -- the ones it is most right about --
    # would each single-handedly override every other source on that gene.
    # Measured on the real Feng cache before this was fixed: 35 genes across 34
    # of 182 targets, and it cost ~0.01 of `reach` on the K562 mirror.
    #
    # It is the same failure this module's docstring is built around, arriving
    # through a numerical door instead of a statistical one: a near-zero
    # variance is a near-infinite weight, and a thin source must never be able
    # to claim one. `isf` evaluates the tail directly and stays finite.
    z = norm.isf(p_nom / 2.0)
    good = np.isfinite(z) & (z > 0)
    idx = np.flatnonzero(usable)[good]
    se = np.abs(lfc[idx]) / z[good]
    v = se**2
    # Belt and braces: a variance that underflows to 0 is still an infinite
    # weight, so anything non-positive abstains rather than dominating.
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
    """Read a Feng-style per-gene LFC table into an `LfcTable`.

    `keep` restricts to a set of targets while streaming, which is what makes
    this affordable: the genome-wide table is 43,187,656 rows and holding all
    6,673 targets as two float64 matrices is ~691 MB, while the 182 that touch
    the 2026 panel are ~19 MB. Filtering after the fact would defeat that.

    `context_col` is for the per-line table, where the context varies per ROW
    (`Cell_Line`, 19 lines) rather than per file. Pass it together with
    `context` to select one line; the config block declares which applies.
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
        # The BH family is this target's own tests -- inverted over the rows the
        # file actually carries for it, not over the padded matrix, so a target
        # measured on fewer genes is not given a family it never had.
        var[r, j] = variance_from_pvalue(fcs, ps, saturated_p=saturated_p)

    return LfcTable(
        labels=labels, genes=gene_list, lfc=lfc, var=var,
        source=source or path.name, context=context,
        notes={"rows_skipped_other_context": skipped_context, "saturated_p": saturated_p},
    )


def main(argv: list[str] | None = None) -> int:
    """Build a cached LfcTable from a published per-gene LFC table.

        uv run python -m sidechain.data.lfc_table \
            ~/data/sidechain/external/figshare-26819743/GenomeWideScreen_LFC_byGene.tsv.gz \
            --keep ~/data/sidechain/vcc2026/pert_counts.csv \
            --out ~/data/sidechain/cache/vcc2026/feng_genomewide_lfc.npz

    Mirrors `sidechain.data.stream_pseudobulk`'s shape on purpose: read the big
    thing once, cache the small thing, and let every later run load the cache.
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
        # The number that says whether the variance derivation produced anything
        # usable. Report 07 §3.3 asks for exactly this before Feng is trusted.
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
