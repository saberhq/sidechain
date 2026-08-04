# VCC 2026 — live track

The competition entry. Spec drops **Aug 20, 2026**.

What Arc has signalled about round two: a new problem, wider scope, and an emphasis on
generalizing to new cell contexts. Everything beyond that — metrics, splits, cell types,
prize pool — is unconfirmed for 2026. (The ~$100k figure that circulates is the *2025*
prize; don't plan against it.) Keep this directory problem-agnostic until kickoff so we
don't overfit to the 2025 shape.

## At kickoff (Aug 20)
1. Drop the 2026 data in `~/data/sidechain/vcc2026/`.
2. Fill `config.yaml` from the official spec (metrics, splits, cell context).
3. Re-check `configs/data_sources.yaml:master_gene_space` against the challenge's actual
   feature space — the GENCODE v47 / Ensembl setting is provisional and was chosen to
   match 2025. Confirm both the release and whether IDs are Ensembl or symbols.
4. Re-read the task structure, then commit the architecture bet (which rungs, which priors).
