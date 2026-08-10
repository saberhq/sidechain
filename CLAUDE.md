# CLAUDE.md — Sidechain

Entry for the Arc **Virtual Cell Challenge 2026**: predict how a cell's transcriptome shifts when
a gene is silenced. Solo entry, human-in-the-loop (Saber) plus Claude Code.

This file is the **public** half of the house rules — the ones about the code. The strategy,
the research programme and the reading budget live in `private/CLAUDE.md`; see below.

## Two repos, one working directory

```
sidechain/            ← this repo. Public. The *what*: code, configs, tests, finished writing.
└── private/          ← github.com/saberhq/sidechain-private. The *why*.
```

`private/` is gitignored here, so this repo cannot see it and cannot accidentally publish it.
Both are checked out in the same directory, so a session reads across freely.

**Routing rule: if it is the *why*, it's private; if it is the *what*, it's public.** Code,
configs, tests and published posts are public. Strategy, measurements, half-formed ideas,
negative results and competitive reasoning are private. **When unsure, private** — promoting a
file later is one commit, and un-publishing is a history rewrite plus a force-push. That
asymmetry is the entire reason for the split.

Corollary that matters: **never move content from a private doc into a tracked file.** If a public
file needs to explain a design decision, write the decision — not the strategy behind it.

## The task, in one paragraph

Given expression data for genes that *have* been silenced, predict what happens when you silence a
gene **never seen silenced**. In 2026 it is harder still: predict how *multiple unseen cell lines*
respond, given only their untouched control cells. Arc provides no new training data this year, so
everyone starts from the same public corpus and whatever else they can bring.

## Where things are

- `src/sidechain/` — the **year-agnostic core**: `data/ priors/ models/ eval/ utils/`. Knows
  nothing about a specific challenge year.
- `challenges/vcc2025/`, `challenges/vcc2026/` — thin per-year adapters: a `config.yaml` pointing
  the core at that year's data, splits and metrics. Same model, different config.
- `configs/` — `data_sources.yaml` (the prior registry), `model.yaml`, `eval.yaml`.
- `tests/` — contract tests. A prior isn't done until these pass.
- `agents/` — the subagent briefs. One file per role, deliberately short.
- `notebooks/00_dry_run_2025.ipynb` — the 2025 loop end to end, narrated.
- `archive/moonshot-2025/` — last year's scripts. Reference only; don't build on it.

## Data lives OUTSIDE the repo

Everything bulky is under `~/data/sidechain/` — never in the working tree.

| path | holds |
|---|---|
| `~/data/sidechain/vcc2025/` | 2025 challenge data (a 15.5 GB h5ad, a local subset, gene names) |
| `~/data/sidechain/vcc2026/` | empty until the Aug-20 drop |
| `~/data/sidechain/cache/` | precomputed prior artifacts (NT embeddings etc.) |
| `~/data/sidechain/literature/` | papers and textbooks |

`.gitignore` blocks `*.h5ad`, `*.zip`, `*.pdf` and friends as defence in depth, but the real rule
is simply: don't put data in the repo.

## House rules

- **Climb the rung ladder.** An ordered sequence of models, simplest first. Build the lowest rung,
  **score it**, and only climb if the next one beats it. You always keep a working baseline and
  never add complexity the metric won't pay for. (Ladder detail: `private/ARCHITECTURE.md`.)
- **Metric-first.** A statistical backbone is the bar to clear. Graph and sequence heads are
  **residual-gated** — kept only if they beat that backbone on the local eval.
- **Nothing is trusted until `sidechain.eval.local_mirror` scores it.**
- **Sparse only** — `edge_index` / COO. Never allocate a dense gene × gene matrix. At 18,080
  genes a dense one is ~1.3 GB per copy.
- **Adding a new biological data source touches exactly two places:** a block in
  `configs/data_sources.yaml` describing it, and a `PriorSource` subclass that fetches it. The
  model never imports a specific source, so nothing else changes — which is what lets a source be
  swapped, ensembled, or shelved with `enabled: false` instead of deleted.
  - Every block declares `kind: edge | node_feature`. Declared, not inferred, so the registry can
    report the graph's shape without building anything — inferring it would mean running a GPU
    pass just to answer "what does this contain?".
  - Build edges with `PriorSource.to_edge_index`, never two separate `to_positions` calls.
    Filtering the two endpoint lists independently drops unknown IDs *per list*, so the survivors
    get re-paired into edges that were never in the source data. Silently wrong edges are worse
    than a crash.
- **Gene IDs are Ensembl.** The 2025 h5ad carries both spaces: `var_names` holds HGNC symbols and
  `var['gene_id']` holds 18,080 well-formed Ensembl IDs, so no external mapping table is needed.
  `loaders.gene_index(adata, 'ensembl_gene_id')` is **strict** — it raises rather than falling
  back to `var_names`, because that fallback produced an index matching no prior and every source
  returned zero edges with no error.
- **`gene_names.csv` has no header row.** `pd.read_csv()` with defaults eats the first gene and
  returns 18,079 rows — an off-by-one that misaligns every gene. Use `header=None`, or better,
  read gene names from the h5ad and ignore the CSV.
- **Check the minimum before writing "every".** This is about how we describe the *data*, not
  about which metrics we report. If you are about to write "every perturbation appears in all 48
  batches", compute the minimum first — the median and the max were both 48 and the minimum was
  22, so the claim was false and an argument had already been built on it. Any "every X has Y"
  statement needs the minimum, not a summary statistic that hides the tail.

## Env

- `uv sync` installs everything including the `dev` group (pytest, ruff). Python is pinned to 3.11
  (`.python-version`), and `requires-python` is capped below 3.13 so the Mac and the GPU box
  resolve to the same dependency versions.
- `arc-state`, `cell-eval`, `cell-load` and `pdex` come from PyPI — no git sources needed.
- Heavy priors (Nucleotide Transformer embeddings) **precompute once** on a rented GPU and are
  cached via lamindb, whose storage root is `~/data/sidechain/cache/`. Downstream runs on the Mac
  (PyTorch MPS).

## Reading budget

Only this file loads automatically. Everything else is on demand — read it because the task needs
it, not to warm up. **`private/CLAUDE.md` carries the reading budget for the private docs**, and is
the next thing to read for any research, strategy or planning task.

| when the task is… | read | where |
|---|---|---|
| touching the code | this file, plus the module you're in | public |
| touching a specific challenge year | `challenges/<year>/CLAUDE.md` — the data traps live there | public |
| understanding an agent's role | `agents/<role>.md` | public |
| **anything research, strategy, or "what should I work on"** | **`private/CLAUDE.md` — start there** | private |

Everything below is in the **private** repo, and `private/CLAUDE.md` is the index for it. Listed
here so a session knows these exist rather than concluding they don't:

| | |
|---|---|
| `private/TODO.md` | open work, and the agent task queue |
| `private/CHANGELOG.md` | completed work |
| `private/ARCHITECTURE.md` | the model ladder, the prior module, the eval loop |
| `private/RESULTS.md` | scored runs |
| `private/GLOSSARY.md` | metrics and statistical terms, with our own numbers as examples |
| `private/research/` | the research programme — ideas, decisions, reading notes |
| `private/reports/` | long-form analyses |
