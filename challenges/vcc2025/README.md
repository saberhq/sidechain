# VCC 2025 — backtest track

Run the Sidechain core against the 2025 data as an end-to-end pipeline check. Same core
(`src/sidechain/`), 2025 config.

**What this track can and can't tell you.** It exercises the whole pipeline — load,
pseudobulk, prior graph, score — on real perturbation data. It cannot tell you what you
would have placed in 2025: Arc never released the public/private test AnnData to
entrants, so the holdout here is one we carve ourselves. Treat the output as "the
pipeline runs and produces sane numbers", not as a leaderboard estimate.

**2025 challenge spec (Arc's year one, for reference):** H1 hESC, ~300 dual-guide CRISPRi
perturbations, NTC + safe-targeting controls, metrics DES / PDS / MAE.

**What we actually hold** in `~/data/sidechain/vcc2025/`:

| file | size | note |
|---|---|---|
| `adata_Training.h5ad` | 15.5 GB | full training set |
| `adata_Training_subset.h5ad` | 3.4 GB | locally derived; **not in the zip, no backup** — use for dev loops |
| `gene_names.csv` | 18,080 rows | HGNC **symbols**, not Ensembl IDs |
| `pert_counts_Validation.csv` | 50 rows | perturbation counts only — no expression matrix |
| `vcc_data.zip` | 4.0 GB | partial archive: 3 of the 4 files (no subset) |

## Run
1. Data is already in `~/data/sidechain/vcc2025/`.
2. `uv run python -m sidechain.eval.local_mirror --config challenges/vcc2025/config.yaml`
3. Compare Rung 0/1 first; climb only if the mirror rewards it.

Iterate against `adata_Training_subset.h5ad`, not the 15.5 GB file. And map HGNC symbols
to Ensembl IDs before touching the prior registry — see this track's CLAUDE.md.
