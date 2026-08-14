# vcc2025 — backtest track

The 2025 data, used as an end-to-end check that the pipeline works. Same core
(`src/sidechain/`), 2025 settings in `config.yaml` beside this file.

**Not a competition submission.** During the challenge Arc released only names and cell counts
for its held-out perturbations, so the default holdout here is one we carve ourselves out of
the training set. A number from that split means "the pipeline runs and produces sane output" —
never "what we would have placed in 2025". Since 2025-12-16 Arc's public bucket
(`gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/`) also carries the real
answers — `adata_Validation.h5ad` (50 perturbations, 6.9 GB) and `adata_Test.h5ad` (100,
12 GB) — so a true backtest is possible once they are downloaded and wired in as an external
holdout.

## Two things that will bite you

**1. Two gene ID spaces, both already in the file.** `var_names` holds HGNC symbols; `var['gene_id']`
holds 18,080 Ensembl IDs. The prior sources key on Ensembl, while `target_gene` values are symbols —
which is what lets the model find a perturbation's own transcript.

`loaders.gene_index(adata, 'ensembl_gene_id')` resolves this and is **strict**: it raises rather
than falling back to `var_names`. That fallback is the real hazard — it produces a gene index
matching *no* prior, so every source returns zero edges and nothing errors. Use
`loaders.assert_coverage` when wiring a new source.

**2. `gene_names.csv` has no header row.** A default `pd.read_csv()` eats the first gene and
returns 18,079 rows, silently shifting every gene by one. Use `header=None` — or just read the
gene names from the h5ad and ignore the CSV.

## Running it

```bash
# the real run: full 221k-cell file, ~16 min, peaks around 9 GB
uv run python -m sidechain.eval.dry_run --rungs 0,0b,1 --n-holdout 25

# smoke test only
uv run python -m sidechain.eval.dry_run --dev --rungs 0 --max-cells 6000
```

The full file is the default. `--dev` switches to the smaller local subset and is for smoke tests
only — no `--dev` number is ever recorded as a result, because at 4× fewer cells per perturbation
the holdout noise is large enough to reverse rankings.

Output goes to `~/data/sidechain/runs/` — outside the repo, like all generated data. One
`report.json` per run, plus a directory per rung.

*"Rung" = a step on the model ladder, simplest first (0 = predict no change, 1 = a statistical
baseline, and up). Build the lowest, score it, and only climb if the next one wins.*
