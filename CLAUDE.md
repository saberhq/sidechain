# CLAUDE.md — Sidechain

Entry for the Arc **Virtual Cell Challenge 2026**: predict how a cell's transcriptome shifts when
a gene is silenced. Solo entry: Saber, the one human, plus Claude Code sessions working together.

This file is **the rules every session must follow — code and coordination.** The *why* lives in
`private/CLAUDE.md`. Size: complete beats short (Saber, A43) — ~180 lines / ~3.4k tokens today,
against the audit's ~120 / ~2k. Trim what is stale, never what a session needs; anything loaded
every session counts, so a new rule earns its lines.

## What you already have when you start

- This file; the auto-memory index (Saber's standing corrections — open the note behind a line
  before acting on it); the **SessionStart brief** (your name and session id, the child-brief
  rule, the STATUS protocol), with the checkers' flags and the open asks beside it — live protocol,
  not background; the skills list (project skills are symlinks into `private/research/protocol/`).
- `challenges/<year>/CLAUDE.md` loads only when you Read a file there: open it yourself before
  touching that tree. `private/CLAUDE.md` never auto-loads; the router at the end says when.

## Two repos, one working directory

`sidechain/` (this repo, public) is the *what*: code, configs, tests, finished writing. `private/`
(github.com/saberhq/sidechain-private, gitignored here) is the *why*. Both are checked out here.

- **Always say which repo**, by writing the path from this directory: `private/CLAUDE.md` versus
  `CLAUDE.md`. Two exceptions to "no `private/` prefix means public": `.claude/skills/*` are
  symlinks into the private repo, so a skill edit is **private**; and private docs drop their own
  prefix. Never collapse a file list (`agents/*.md`) while quoting a count.
- **Shorthand** used here and in private docs: `reports/NN` = `private/reports/NN_*.md`;
  `ADR NNNN` = `private/research/decisions/NNNN-*.md`; `T<n>` a task in `private/TODO.md`;
  `A<n>` an ask on the board; `[[slug]]` = `private/research/ideas/<slug>.md`.
  `private/research/protocol/` holds `ledger.py`, `status_hook.py` and the `check_*.py` checkers;
  run them as `.venv/bin/python <path>`.
- **The why is private, the what is public; when unsure, private.** Never move content from a
  private doc into a tracked file: publishing is one commit, un-publishing is a history rewrite.

## Committing and pushing

- **Stage by explicit path.** Never `git add -A`, `.`, `-u` or `commit -a`; in private,
  `git -C private add <paths>`. 2026-08-22: one `add -A` swept 44 files of another session's work
  into a pushed commit. Other sessions' uncommitted work is always in this tree.
- **Shared files** (`private/TODO.md`, `CHANGELOG.md`, `RESULTS.md`): never `git commit <path>`,
  which commits the working tree with other sessions' hunks. Build the blob from HEAD plus your
  edit and run a bare `git commit` (`/sidechain-commit`, step 2). Never `git stash -u`.
- If `git status` shows changes you did not make, leave them alone and say so.
- **Push, publish or deploy only on Saber's go; one go covers one push.** First
  `git log --oneline origin/main..HEAD`, and say whose commits ride along.
- Messages: `<scope>: <what>` plus a pointer to the doc that explains it; public short, private
  may narrate briefly. Reasoning lives in editable files, never in a message.
- **Never delegated:** `vcc submit`, pushes, history rewrites, Saber's prose in `private/research/master.md`.
- **In files, commit messages and prose, name a session by its session id**, written
  ``session `0badcafe` `` (8 chars), never by its auto-name (`sidechain-xx`, which changes on
  restart). The auto-name is the *address* — `SendMessage`, `--name`, the STATUS line — and
  belongs nowhere else. The brief tells you both; a `ListAgents` bracket ref is not an id.

## The task, in one paragraph

Given expression data for genes that *have* been silenced, predict what happens when you silence a
gene **never seen silenced**, in cell lines never seen: six anonymized lines (A/B/C now, D/E/F on
**Oct 22**; only D/E/F decide prizes), 300 knockdowns × 400 cells per line, raw counts, 18,533
genes, six cell-eval2 metrics. Arc ships no training data. `challenges/vcc2026/CLAUDE.md` +
`config.yaml` are authoritative for the spec and the submission contract — never re-derive it from
the web; the argument: `reports/05`, `06`, `09`.

## Where things are

- `src/sidechain/` — the year-agnostic core: `data/ ingest/ priors/ models/ eval/ submit/ utils/`.
- `challenges/vcc2025/`, `vcc2026/` — thin adapters: `config.yaml` + `CLAUDE.md`.
- `configs/` — `datasets.yaml` (**the corpus registry**: datasets, `route`, `control_label`,
  budgets); `data_sources.yaml` (the prior registry); `eval.yaml` and `model.yaml` (2025-era).
- `tests/` — contract tests. `agents/` — the four role briefs (researcher, verifier, analyst,
  critic), each wired in `.claude/agents/`. `scripts/` — `standings.py`, `lamin_*.py`, `brev_*.sh`.
- `site/` — saberhq.com/sidechain, themeless Hugo (`site/README.md`). `notebooks/` — the 2025 loop.

## Data lives OUTSIDE the repo

Everything bulky is under `~/data/sidechain/`: `external/<host>-<record>/` is third-party bytes as
published (`PROVENANCE.json`; re-downloadable, delete first under disk pressure); `derived/<source>/`
is ours (`LINEAGE.json`; expensive, back it up); plus `vcc2025/`, `vcc2026/`, `cache/`, `runs/`.
**A streamed dataset's `external/` dir holds only `PROVENANCE.json`: correct, not a failed
download.** `route:` and the stream gate: the `configs/datasets.yaml` header and ADR 0003.

## House rules

- **Climb the rung ladder** (`private/ARCHITECTURE.md`): build the lowest rung, score it, climb
  only if the next beats it. **Metric-first**; graph and sequence heads are residual-gated.
- **Nothing is trusted until the local mirror scores it:** `sidechain.eval.mirror2026` (one
  held-out line, cell-eval2's six metrics) and `sidechain.eval.loco` (the rotation over lines);
  a pdex/CPU bundle scores on the Mac, gpudge and full-corpus builds need the box (`/sidechain-brev`).
  `sidechain.eval.local_mirror` is the frozen 2025 backtest and scores nothing for 2026.
- **Sparse only** (`edge_index`/COO); never a dense gene × gene matrix (18,533² is ~1.4 GB).
- **A new corpus** is a `configs/datasets.yaml` block behind the ADR 0003 gate
  (`sidechain.ingest.fetch`). **A new prior** is a `data_sources.yaml` block plus a `PriorSource`
  subclass (`src/sidechain/priors/base.py`) registered in `src/sidechain/data/registry.py`; blocks
  declare `kind: edge | node_feature`; build edges with `PriorSource.to_edge_index`, never two
  `to_positions` calls (why: `private/ARCHITECTURE.md`).
- **Two gene-ID spaces**, declared per year in `challenges/<year>/config.yaml`: 2025 carries
  Ensembl in `var['gene_id']` (18,080 genes); 2026 is symbols only, `var` empty, 18,533 genes.
  `loaders.gene_index` is strict and raises rather than falling back to `var_names` — that fallback
  once matched no prior and returned zero edges silently.
- **`gene_names.csv` differs by year** (2025 no header, 2026 has one): read names from the h5ad;
  `sidechain.data.profile` marks the read that matches.
- **Read the methods for the control definition; never infer it from the labels.** `control_label`
  is a list: Feng 2026's arm is `[NonTarget, unassigned]`, 499,998 cells, not 48.
- **Check the minimum before writing "every"** (median 48, minimum 22).
- **Standings numbers are generated** (`scripts/standings.py`): never hand-edit the README table,
  `site/data/submissions.json` or `RESULTS.md` ranks.
- **Model names carry their knobs at birth and are never renamed after scoring** (ADR 0005);
  `check_modelname.py --propose` gates every submit.
- **The GPU box is shared like this checkout:** `brev ls` first, message the owning session, box
  lifetime is Saber's call, pull → register → verify → delete (`/sidechain-brev`).

## Replying to Saber

Saber is a research scientist in bioinformatics, not an ML engineer: biology as technical as it
needs, ML plumbing in plain words. Name a thing, then say what it does; a concrete example beats a
category; brief wins. The full register: `private/CLAUDE.md` → How to write for Saber.

A reply that ends a unit of work or asks him something is **three headed sections and nothing
else**:

1. **What changed** — each path with its repo, and one or two sentences of substance per item:
   enough to digest the work without opening the file. Pointers carry the reasoning.
2. **What I need from you** — or `Nothing`.
3. **Next steps** — one line each.

An answer to a question is the answer first, then the pointers that carry the evidence — no
sections; a yes/no is one sentence. Everywhere: plain words, one number per sentence, no tables, no
why-paragraphs; a correction to your own earlier claim is one sentence plus a pointer. The STATUS
block follows when the state, task or pointer changed or an ASK is raised: `Nothing` in section 2
means `fyi`; a question you cannot proceed without means `waiting · expect_by HH:MM` (end of the
working day unless you know better, with `waiting_on: Saber`) plus a typed `decision` ASK.

## The STATUS block, subagents and the board

The SessionStart brief owns the protocol; this is the stub. Three consecutive plain-text lines, last
in the reply, unfenced, flush left, no blank line inside: `STATUS · <name> · T<n> · state ·
expect_by` / `→ <file>` / `ASK: type — question? rec: …`. You write `running`, `waiting` or
`returned` (the task is handed back); the hook sets `idle` and `gone`. The Stop hook
(`status_hook.py`) lifts the block to the board and refuses one naming another session, a state
outside those five, or a `review` ASK without a question mark; the ASK type is never optional (an
untyped one is filed as `decision` today). A fenced block is documentation. No brief at session
start means no hooks are wired here. **A subagent (an Agent or Workflow child in your context) sends
no block; a peer (a separate session you message) files its own. Every brief you hand a child opens
with `T<n> · from <your session id> · one line`, or `— · throwaway` for work you will discard; no
T-id yet → mint one first (`check_todo.py` prints the next free, add the line to `private/TODO.md`).**
**Models:** a mother runs Saber's session model (Fable 5.1, 1M window) — the only place the whole
picture is held; a child runs Opus by default (`CLAUDE_CODE_SUBAGENT_MODEL`), its wrapper's
`model:` when wired (Researcher on Sonnet), or the `model` a Workflow stage passes: sweeps and
schema-bound reading on Sonnet, counting and converting on Haiku, verify / judge / synthesis on
Opus. Say why when you override. The queue, the mother session and dispatch: `private/CLAUDE.md` → The mother session.

## Env

`uv sync` installs everything incl. `dev`; Python 3.11, capped below 3.13, so the Mac and a box
resolve alike. PyPI: `arc-state`, `cell-eval2` (the 2026 scorer), `cell-eval` (2025), `cell-load`,
`pdex`. The hosted lamindb `saberhq/sidechain` backs up `derived/`, `cache/` and bundles, never
`external/`: register every derived artifact in the session that creates it
(`scripts/lamin_register.py` ⇄ `lamin_pull.py`, key = path under `~/data/sidechain/`; ADR 0007).

## Where to look, by the kind of ask

| when the ask is … | read first | then |
|---|---|---|
| code: a module, a test, a config | this file + the module you are in | — |
| model or prior design, the rung ladder | `private/ARCHITECTURE.md` | — |
| score a prediction, build a bundle | `src/sidechain/eval/mirror2026.py` (docstring: `bundle` → `score`; Mac vs box) | `/sidechain-brev` |
| a challenge year's data, packaging, `vcc submit` | `challenges/<year>/CLAUDE.md` | `check_modelname.py --propose`; a submit is Saber's |
| QC on our h5ads, the controls bundle | `challenges/<year>/CLAUDE.md` · `sidechain.data.profile` · `sidechain.ingest.checks` | `reports/06` |
| a generic single-cell method: QC, scVI, nf-core, instrument data | `/bio-research:single-cell-rna-qc`, `:scvi-tools`, `:nextflow-development`, `:instrument-data-to-allotrope` | their `SKILL.md` under `private/research/protocol/plugins/bio-research/skills/` when the skill is not listed |
| a research question, "what have we tried" | `private/CLAUDE.md` | `private/research/master.md` · `private/research/INDEX.md` · `private/RESULTS.md` |
| "what should I work on", `pick up T<n>` | `private/TODO.md` `## Now` | `private/CLAUDE.md` → The mother session |
| spawn a Researcher, a Verifier, a workflow | `private/CLAUDE.md` → The mother session | every spawn names a task — there is no standing brief · write contracts in `private/research/README.md` |
| a paper, a literature review | `/paper-intake` (one paper) · the Researcher (a sweep) | `private/research/reading/` · `private/literature.md` |
| commit, push | this file § Committing · `/sidechain-commit` | — |
| the GPU box, more RAM, a full-corpus stream | `/sidechain-brev` | — |
| a data download, disk, a new corpus, a licence | this file § Data · `configs/datasets.yaml` | ADR 0003 |
| the website, a post, LinkedIn | `site/README.md` · `/post` | `private/site/NOTES.md` · `private/site/VOICE.md` |
| the board, `ledger.py`, asks, status, the queue | `private/HOWTO-desk.md` §1 and §8 (the desk is Saber's own session; §2–§7 are his vocabulary) | `reports/12` |
| a metric or statistical term | `private/GLOSSARY.md` (grep it) | — |
| a checker fired: why | that `private/research/protocol/check_<x>.py` (docstring, `--help`) | `reports/11` §8 |
| wiring hooks, skills, `.claude/settings.local.json` (gitignored; template in `private/research/protocol/machine-local/`) | `private/README.md` § After cloning | — |
| an agent's role | `agents/<role>.md` (researcher · verifier · analyst · critic) | its wrapper in `.claude/agents/`, which pins its model |
