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

**Always say which repo a file is in.** Two repos means "I changed `CLAUDE.md`" is ambiguous and
"I removed the task queue" could mean it left the public repo or left the project entirely. When
naming a file in writing, in a commit message, or in conversation, use the full path from this
directory: `private/CLAUDE.md` is the private one, `CLAUDE.md` is this one. Anything starting with
`private/` is private; everything else is public. When summarising a change, say which side moved.

**Routing rule: if it is the *why*, it's private; if it is the *what*, it's public.** Code,
configs, tests and published posts are public. Strategy, measurements, half-formed ideas,
negative results and competitive reasoning are private. **When unsure, private** — promoting a
file later is one commit, and un-publishing is a history rewrite plus a force-push. That
asymmetry is the entire reason for the split.

Corollary that matters: **never move content from a private doc into a tracked file.** If a public
file needs to explain a design decision, write the decision — not the strategy behind it.

## Committing

- **Stage by explicit path. Never `git add -A`, `git add .`, `git add -u`, or `git commit -a`.**
  More than one session works in this checkout at the same time, so the working tree holds
  other people's uncommitted work. On 2026-08-22 a `git add -A` for a one-file config change
  swept 44 files of another session's site work into a commit about the scPerturb budget, and
  pushed it; history had to be rewritten. Name what you changed:
  `git add configs/datasets.yaml && git commit -m "..."`. If `git status` shows changes you did
  not make, leave them alone and say so in your summary. Same in the private repo:
  `git -C private add <paths>`.
- **Push only when Saber says so.** Commits are local until he has reviewed them.

> **Status (2026-08-21):** the 2026 spec, evaluation criteria and the validation bundle are
> released and ingested. `challenges/vcc2026/CLAUDE.md` + `config.yaml` are authoritative for
> the data and the submission contract; `private/reports/05` (audit) and `06` (bundle QC) for
> the argument. Do not re-derive the spec from the web; diff it again on **Oct 22** (final bundle).

## The task, in one paragraph

Given expression data for genes that *have* been silenced, predict what happens when you silence a
gene **never seen silenced**. In 2026 it is harder still: predict how *multiple unseen cell lines*
respond, given only their untouched control cells. Arc provides no new training data this year, so
everyone starts from the same public corpus and whatever else they can bring.
Spec, as of the 2026-08-20 kickoff: six anonymized lines (three now, three on Oct 22), 300
knockdowns × 400 cells per context, raw counts, six cell-eval2 metrics — `challenges/vcc2026/CLAUDE.md`.

## Where things are

- `src/sidechain/` — the **year-agnostic core**: `data/ ingest/ priors/ models/ eval/ utils/`.
  Knows nothing about a specific challenge year.
- `challenges/vcc2025/`, `challenges/vcc2026/` — thin per-year adapters: a `config.yaml` pointing
  the core at that year's data, splits and metrics. Same model, different config.
- `configs/` — `data_sources.yaml` (the prior registry), `model.yaml`, `eval.yaml`.
- `tests/` — contract tests. A prior isn't done until these pass.
- `agents/` — the subagent briefs. One file per role, deliberately short.
- `notebooks/00_dry_run_2025.ipynb` — the 2025 loop end to end, narrated.
- `site/` — the public page at [saberhq.com/sidechain](https://saberhq.com/sidechain/): themeless
  Hugo on the saberhq.com brand tokens, deployed by `.github/workflows/site.yml`. Posts go in
  `site/content/posts/`; `site/data/submissions.json` feeds the standings. `site/README.md`.

## Data lives OUTSIDE the repo

Everything bulky is under `~/data/sidechain/` — never in the working tree.

| path | holds |
|---|---|
| `~/data/sidechain/vcc2025/` | 2025 challenge data (a 15.5 GB h5ad, a local subset, gene names) |
| `~/data/sidechain/vcc2026/` | 2026 controls bundle (contexts A/B/C; D/E/F from Oct 22), fetched with the `vcc` CLI; leaderboard snapshots |
| `~/data/sidechain/external/` | third-party corpora **exactly as published** — one directory per source, named `<host>-<record id>` |
| `~/data/sidechain/derived/` | artifacts **we computed** from `external/` — one directory per source |
| `~/data/sidechain/cache/` | precomputed prior artifacts (NT embeddings etc.) |
| `~/data/sidechain/literature/` | papers and textbooks |
| `~/data/sidechain/runs/` | eval and mirror outputs |

`.gitignore` blocks `*.h5ad`, `*.zip`, `*.pdf` and friends as defence in depth, but the real rule
is simply: don't put data in the repo.

### Reading the tree: `external/` vs `derived/`, and what "streamed" means

If you are working out what some directory under `~/data/sidechain/` *is*, this is the answer.
The two trees are split because they have **opposite recovery properties**, and that is what you
want to know when the disk is full:

- **`external/` is someone else's immutable bytes.** Checksummed, re-downloadable from a recorded
  URL. First thing to delete under disk pressure, never needs backing up. `rm -rf external/<x>`
  is always safe.
- **`derived/` is ours and expensive.** A streamed pseudobulk over 126 GB of parquet is hours of
  network. Back this up; delete it last.

**Every directory under either tree carries its own explanation — read it rather than guessing:**

| file | sits in | answers |
|---|---|---|
| `PROVENANCE.json` | `external/<host>-<record>/` | which host, which pinned version, every file's size + checksum, the licence, and **`route`** |
| `LINEAGE.json` | `derived/<source>/` | which `PROVENANCE.json` this came from, which code SHA built it, and what the accumulator's resolution was |

**`route` is the field that explains a directory that looks empty.** `configs/datasets.yaml`
declares it per dataset:

- **`route: download`** (the default, and what every block before X-Atlas is) — the bytes land in
  `external/<host>-<record>/` beside their `PROVENANCE.json`. `budget_gb` bounds *that selection*.
- **`route: stream`** — the corpus is read once over the network and **never lands**. Its
  `external/<host>-<record>/` holds `PROVENANCE.json` **and nothing else** — that is the intended
  shape, *not* a download that failed. The only output is an aggregate in `derived/`, and there
  `budget_gb` bounds the **output** instead. A streamed block declares both `dest:` (where the
  provenance goes) and `derived:` (where the aggregate goes); the gate refuses a stream that
  omits `derived:`.

So `external/hf-Xaira-Therapeutics-X-Atlas-Orion/` containing a single JSON file is correct and
complete: X-Atlas/Orion is 126.26 GB we stream and never store.

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
  `configs/data_sources.yaml` describing it, and a `PriorSource` subclass that fetches it. (A
  brand-new loader *class* is one line more — registering it in `registry.LOADERS`; reusing an
  existing class keeps it at two.) The
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
- **`gene_names.csv` differs by year — check the row count against the h5ad.** The 2025 file has
  **no header**: a default `pd.read_csv()` eats the first gene and returns 18,079 rows. The 2026
  file **has one** (`gene_name`): `header=None` returns 18,534 rows with a bogus first gene.
  Either mistake misaligns every gene silently. Read gene names from the h5ad where possible;
  `sidechain.data.profile` prints both reads and marks the one that matches.
- **Read the methods for the control definition; never infer it from the labels.** Which values
  count as controls is a fact about the *experiment*, not a property of the strings in the column,
  and `control_label` in `configs/datasets.yaml` is a **list** for that reason. Feng 2026 has 48
  cells whose guide call reads `NonTarget_*` and a control arm of **499,998** — its paper defines
  controls as "either no guide or a non-targeting gRNA", so `unassigned` is a control, not a
  discarded cell. Reading the label instead of the design undercounted the arm 10,000-fold and put
  a wrong conclusion into a report before it was caught (2026-08-23). The sibling of the exact-match
  rule in `ingest/checks.py`: that one says *match the declared label exactly*, this one says
  *the declaration itself has to come from the paper*.
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
- **The cloud data plane** (ADR 0007) is lamindb's hosted instance `saberhq/sidechain`
  (Lamin-managed S3, us-west-2). It is transport and backup for the things that are *ours and
  expensive* — `derived/`, `cache/`, mirror bundles and truth files. `external/` never goes there:
  its recovery story is the upstream host. **An artifact's key is its path under
  `~/data/sidechain/`**, so the instance is a mirror of the local tree, not a second naming
  scheme — `scripts/lamin_register.py` (path → key) and `scripts/lamin_pull.py` (key → path, with
  a hash check) are exact reciprocals over `utils/lamin.py`. That identity is also the exit plan:
  leaving costs an `s3 sync`, not a migration.
  - Heavy priors (Nucleotide Transformer embeddings) **precompute once** on a rented GPU, are
    registered from the box, and are pulled on the Mac (PyTorch MPS) — no `brev copy` either way.
  - A box authenticates once at bootstrap: `scripts/brev_lamin_key.sh <box>` ships the one
    long-lived key from `~/.lamin/current_user.env` as a 0600 file, and `brev_bootstrap.sh` logs
    in with it. No key is not an error — pulls fall back to `brev copy`.
  - Scored runs log themselves there too (`utils/logging.log_run`; set `SIDECHAIN_LAMIN_INSTANCE=`
    empty to disable) — non-fatal on machines without credentials, unlike registration and pulls,
    which fail loudly because they *are* the task.

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
