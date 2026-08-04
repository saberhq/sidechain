# Sidechain

Solo R&D entry for the **Arc Virtual Cell Challenge 2026**. Predict how a cell's
transcriptome shifts after a genetic (CRISPRi) perturbation, well enough to beat
Arc's STATE baseline on the official metrics (DES / PDS / MAE + STATE's extras).

See [`plan.md`](./plan.md) for strategy, budget, timeline, and the architecture ladder.

## Layout

```
CLAUDE.md           house rules + where things are (read first)
plan.md             strategy, budget, timeline
ARCHITECTURE.md     the rung ladder, the prior module, the eval loop
literature.md       idea log — the Researcher appends here
configs/            declarative config — edit these, not the code
  data_sources.yaml   << the flexible prior registry (add a source = add a block)
  model.yaml          rung selection + head hyperparameters
  eval.yaml           local cell-eval mirror settings
src/sidechain/
  data/             loaders (wrap cell-load), pseudobulk strategies, source registry
  priors/           the extensible prior layers  <-- Saber's edge lives here
    base.py           PriorSource ABC (the contract every prior implements)
    trans_grn.py      TF -> target edges (DoRothEA/TRRUST/STRING)
    cis_sequence.py   3'UTR sequence embeddings (Nucleotide Transformer; the modern ntEmbd)
    posttx_mirna.py   miRNA -> target repression edges (miRBind2 / TargetScan)
    posttx_rbp.py     RBP  -> target binding edges (POSTAR3 / eCLIP / ATtRACT)
    graph_builder.py  assemble one multi-relational sparse graph from enabled sources
  models/           baseline_stats (Rung 1), state_adapter (Rung 2),
                    residual_gnn (Rung 3), fusion (Rung 4 metric-aware loss)
  eval/             local_mirror (cell-eval wrapper) + extra dev metrics
  utils/            lamindb run logging
challenges/
  vcc2025/          frozen 2025 spec — the backtest track
  vcc2026/          the live track; fills at the Aug-20 spec drop
archive/
  moonshot-2025/    last year's preliminary scripts (reference only)
agents/             Claude Code subagent briefs
                    (researcher / developer / experiment_manager / critic)
notebooks/          scratch + the 2025 dry-run
tests/              contract tests (every PriorSource stays gene-index aligned)
scripts/reorg.sh    the one-shot repo reorganization, kept for the record
```

The split that makes this work: `src/sidechain/` is **year-agnostic** and knows nothing
about a specific challenge. Each `challenges/<year>/` is a thin adapter — a `config.yaml`
pointing the core at that year's data, splits and metrics. Same model, different
challenge config.

## Add a new prior in 3 steps (the whole point of this design)

1. Add a block to `configs/data_sources.yaml` (name, `layer`, `loader`, url, `enabled`).
2. If no existing loader fits, subclass `PriorSource` in `src/sidechain/priors/`.
3. `enabled: true`. The graph builder picks it up automatically — no model changes.

Set `enabled: false` to shelve a source without deleting it.

## Quickstart

```bash
uv sync                                               # incl. the dev group (pytest, ruff)
uv run pytest                                         # contract tests
uv run python -m sidechain.eval.local_mirror --help   # once the mirror is wired
```

## Data

Nothing bulky lives in this repo. Challenge data, caches and papers are under
`~/data/sidechain/` (`vcc2025/`, `vcc2026/`, `cache/`, `literature/`) — see CLAUDE.md.

Precompute-heavy priors (Nucleotide Transformer embeddings) run once on a rented GPU and
cache via lamindb, whose storage root is `~/data/sidechain/cache/`; everything downstream
runs on the Mac against frozen vectors.
