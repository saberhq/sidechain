# CLAUDE.md — vcc2025 (backtest track)

You are in the **2025 backtest track**. Purpose: retrospective scoring only — run the
Sidechain core against the 2025 data to sanity-check the pipeline end to end. This is
NOT a competition submission.

- Use the frozen 2025 spec in `config.yaml`; do not leak 2026 assumptions here.
- Data: `~/data/sidechain/vcc2025/`.
- Score with `sidechain.eval.local_mirror`.

## Two things that will bite you here

1. **There is no official test split.** Arc never released public_test / private_test
   AnnData to entrants. All we hold is `adata_Training.h5ad` plus a locally derived
   subset, `gene_names.csv`, and `pert_counts_Validation.csv` (50 rows of perturbation
   *counts*, no expression). The backtest therefore carves its own holdout from
   training. A number produced here is a pipeline check, not a leaderboard estimate —
   never phrase it as "what we would have placed in 2025".

2. **Two ID spaces, both already in the file.** *(Corrected 2026-08-04 — the earlier
   version of this note was wrong.)* `gene_names.csv` is 18,080 bare HGNC symbols, and
   `var_names` matches it. But `var['gene_id']` holds 18,080 well-formed, unique Ensembl
   IDs, so the symbol→Ensembl mapping the prior registry needs **ships inside the h5ad** —
   no external mapping table is required.

   `loaders.gene_index(adata, 'ensembl_gene_id')` resolves against `var['gene_id']` and is
   strict: it raises rather than silently falling back to `var_names`. That fallback was
   the real hazard — it yields a gene index matching *no* prior, so every source returns
   zero edges and nothing errors. Use `loaders.assert_coverage` when wiring a new source.

   Note the two spaces are used for different things: the priors key on Ensembl, while
   `target_gene` values are **symbols** and match `var_names` — which is what lets the
   backbone find a perturbation's own transcript to apply the CRISPRi knockdown.

## Dev loop

**The full 221k-cell file is the default** (policy set 2026-08-04). Iterating on the subset
hid resource limits we need to meet before the 2026 kickoff, and its 4×-lower
cells-per-perturbation inflated holdout noise. `--dev` switches to
`adata_Training_subset.h5ad` for smoke tests only — **no `--dev` number is ever recorded as
a result.**

```bash
uv run python -m sidechain.eval.dry_run --rungs 0,0b,1 --n-holdout 25   # full, the real run
uv run python -m sidechain.eval.dry_run --dev --rungs 0 --max-cells 6000  # smoke test
```

Budget for the full run: peak ~9 GB RSS on a 17 GB machine, and the per-rung cost is
dominated by cell-eval's differential expression pass, not by our model. Each stage logs
peak RSS (`[mem]` lines) so a regression shows up immediately.

The subset was derived locally on 2025-11-23 and exists in exactly one place, with no
backup (`vcc_data.zip` was deleted on 2026-08-04 after verifying all three of its members
were already unpacked beside it at identical byte sizes — the subset was never in it).
Don't delete it casually.

Writes per-rung `results.csv` / `agg_results.csv` plus a `report.json` under
`runs/dry_run_2025/`. `notebooks/00_dry_run_2025.ipynb` is the same loop, narrated.

**Any number this produces is a pipeline check, not a leaderboard estimate.** The holdout
is carved from the training set (see `carve_holdout`), so never phrase a result as "what
we would have placed in 2025".
