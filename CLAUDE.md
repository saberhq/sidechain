# CLAUDE.md — Sidechain

Entry for the Arc **Virtual Cell Challenge 2026**: predict how a cell's transcriptome shifts when
a gene is silenced. Solo entry: Saber, the one human, plus Claude Code sessions working together.

This file is **the rules every session must follow — code and coordination.** The *why* lives in
`private/CLAUDE.md`. Size: complete beats short (Saber, A43) — ~280 lines / ~4.9k tokens today,
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
  `ADR NNNN` = `private/research/decisions/NNNN-*.md`; `T<n>` a task, `private/tasks/T<n>.md`;
  `A<n>` an ask on the board; `[[slug]]` = `private/research/ideas/<slug>.md`.
  `private/research/protocol/` holds `ledger.py`, `status_hook.py` and the `check_*.py` checkers;
  run them as `.venv/bin/python <path>`.
- **The why is private, the what is public; when unsure, private.** Never move content from a
  private doc into a tracked file: publishing is one commit, un-publishing is a history rewrite.

## Committing and pushing

- **Stage by explicit path.** Never `git add -A`, `.`, `-u` or `commit -a`; in private,
  `git -C private add <paths>`. 2026-08-22: one `add -A` swept 44 files of another session's work
  into a pushed commit. Other sessions' uncommitted work is always in this tree.
- **Shared files** (`CHANGELOG.md`, `RESULTS.md`): never `git commit <path>`,
  which commits the working tree with other sessions' hunks. Build the blob from HEAD plus your
  edit and run a bare `git commit` (`/sidechain-commit`, step 2). Never `git stash -u`.
- If `git status` shows changes you did not make, leave them alone and say so.
- **Push, publish or deploy only on Saber's go; one go covers one push.** First
  `git log --oneline origin/main..HEAD`, and say whose commits ride along.
- Messages: `<scope>: <what>` plus a pointer to the doc that explains it; public short, private
  may narrate briefly. Reasoning lives in editable files, never in a message.
- **Never delegated to a child agent:** `vcc submit` (a session runs it only on Saber's GO for that entry —
  `private/SUBMISSION-QUEUE.md` rule 3; since 2026-10-08 his GO can also be written into the entry's file,
  `go:` with `slot: auto`, and then the daily launchd job sends it at the UTC reset, rule 7 and ADR 0012),
  pushes, history rewrites, Saber's prose in `private/research/master.md`.
- **In files, commit messages and prose, name a session by its session id**, written
  ``session `0badcafe` `` (8 chars), never by its auto-name (`sidechain-xx`, which changes on
  restart). The auto-name is the *address* — `SendMessage`, `--name`, the STATUS line — and
  belongs nowhere else. The brief tells you both; a `ListAgents` bracket ref is not an id.
  Two open sessions can wear one auto-name, so a session you know by id is messaged at the
  address `ledger.py session address <id>` prints, never by bare name (2026-10-16).

## The task, in one paragraph

Given expression data for genes that *have* been silenced, predict what happens when you silence a
gene **never seen silenced**, in cell lines never seen: six anonymized lines (A/B/C now, D/E/F on
**Oct 22**; only D/E/F decide prizes), 300 knockdowns × 400 cells per line, raw counts, 18,533
genes, six cell-eval2 metrics. Arc ships no training data. `challenges/vcc2026/CLAUDE.md` +
`config.yaml` are authoritative for the spec and the submission contract — never re-derive it from
the web; the argument: `reports/05`, `06`, `09`. **What happens on Oct 22 is one document,
`private/FINAL-PHASE.md`** (T93): only the last final entry counts and no final score is shown (final-phase: none).

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
**A file that should be there and is not** was archived, not lost: read `~/data/sidechain/README.md`
§ Backups (Lamin nightly, MEDIA archive, the SSD) before rebuilding or re-downloading anything.

## House rules

- **Climb the rung ladder** (`private/ARCHITECTURE.md`): build the lowest rung, score it, climb
  only if the next beats it. **Metric-first**; graph and sequence heads are residual-gated.
- **Nothing is trusted until the local mirror scores it:** `sidechain.eval.mirror2026` (one
  held-out line, cell-eval2's six metrics) and `sidechain.eval.loco` (the rotation over lines);
  a pdex/CPU bundle scores on the Mac, gpudge and full-corpus builds need the box (`/sidechain-brev`).
  `sidechain.eval.local_mirror` is the frozen 2025 backtest and scores nothing for 2026.
- **Sparse only** (`edge_index`/COO); never a dense gene × gene matrix (18,533² is ~1.4 GB).
- **A script's docstring says what, never why** (Saber, 2026-10-15): one line of purpose, how to run it, what it reads
  and writes, and a pointer to the idea or task file. No hypotheses, kill criteria, review rounds or findings: they go
  stale in code. A task's probe scripts live in private `scripts/`, not here (T88's and T102's were moved).
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
- **A–F name the challenge's six anonymized contexts and nothing else** (Saber, 2026-09-21): never
  letter a **cell line, fold, bundle or corpus** of ours. Name it — the fold by its directory
  (`loco_hek293t_ch272`), the line by its name plus the study where that is ambiguous (Jurkat from
  Nadig 2024). "A source for context A" is the only shape a letter takes. **A two-arm comparison is
  exempt** (`arm A` vs `arm B`): that letters a contrast, not a context, and renaming the ~141 lines
  that use it would rewrite shipped submission records. D/E/F are unreleased until Oct 22, so a
  letter written today can only mean A, B or C (Saber, 2026-09-16).
- **Standings numbers are generated** (`scripts/standings.py`): never hand-edit the README table,
  `site/data/submissions.json`, `site/assets/data/field.json` or `RESULTS.md` ranks.
- **Model names carry their knobs at birth and are never renamed after scoring** (ADR 0005);
  `check_modelname.py --propose` gates every submit.
- **A GPU box belongs to one task** (`sidechain-gpu-<tid>`): `brev ls` first, and never a job on a
  box another session's job is using. Before Saber's GO a box request card: the queue, the predicted
  hours and billed dollars on two machines (`private/research/protocol/box_plan.py`), the plan for
  the end and who closes the box if the session stalls. The machine follows the queue's size, AMD
  by `lscpu` before anything is pulled; a second task's lines ride on a running box only on his
  word, one ledger note each. Box lifetime is his call: no timer, no kill, no self-delete; a stop
  you set never ends running work, and a renewed `brev login` cancels it; pull → register → verify →
  delete, or stop where the type allows (`/sidechain-brev`, trap 17). CPU-only tails leave the GPU
  box. Builds go on a GPU box and are scored on the same machine (a scoring bundle is bound to its backend
  and device); past that, box use is flexible (Saber, 2026-10-18; the CPU standby was retired, T118).

## Review tiers — how much review, and when none

A reviewer agent re-reads what the mother already holds, so review goes where an error would
otherwise survive, never by habit (Saber, 2026-10-14; the measured cost:
`private/research/token-usage-audit-2026-10-14.md`). The tier is set by what is claimed or done,
and it is the most review that work gets:

- **0 · none.** Code covered by its tests, pipeline output (a run, a scoring pass, a generated
  table), mechanical edits, ledger and file housekeeping. The tests and the mirror are the review.
- **1 · one cheap pass.** A reading whose numbers sit in a generated table or `results.json`: the
  mother, or one Sonnet child, checks each sentence against that table. Nobody re-derives numbers
  from the run directory.
- **2 · one critic round on Opus.** A new statistic or instrument, a pre-registration, box scripts
  before a paid run, a "this will tell us whether X" put to Saber for a spend: one critic, two
  lenses at most; a refuter only for a finding marked blocking; a second round only when a fix
  changed the design.
- **3 · full adversarial round.** A `vcc submit`, and anything published under Saber's name (site,
  README prose, posts): lensed critics and a refuter per serious finding.

After a review the mother checks her rewrite against the findings list herself; a rewrite is not a
new round. Write a reading's summary last, from its body, and lead with the registered contrast.
Workflows run when Saber asks for one or tier 3 calls for it. Ultracode stays off unless he turns it
on for a phase (`/effort ultracode`), and a hand-off never tells the next session to use either.

## When to proceed and when to ask

**Saber's GO is needed for four things** (Saber, 2026-10-16): a Brev box (creating one, or a paid run
on one), `vcc submit`, a push to either remote, and anything published under his name (the site, a
post, README prose). **His word stays his** on a box's lifetime, deleting data under `~/data/sidechain/`,
a task's priority or retirement (A43), and his own prose in `private/research/master.md`. **Everything
else proceeds:** a Mac reading or screen, a probe, a table, an idea file, a plan, a commit without a
push. A session that reaches a fork says which option it takes and why, takes it, and reports the fork
in the same reply; the `decision` ASK records the fork and never parks the work behind it. It stops
only at the items above, or when every option would waste the work if wrong. A session that waits for
a GO on anything else has left the work undone.

## Replying to Saber

Saber is a research scientist in bioinformatics (an MSc and a PhD in it, researching since 2013), not
an ML engineer, and he runs several research threads at once. Write to a colleague who knows the
field's words but not this corner of it: name a thing, say what it does and how it bears on the thing
beside it, before using it again; a concrete example beats a category; ML plumbing in plain words.
**Biology and bioinformatics are explained, never assumed** (Saber, 2026-10-16): a mechanism, a data
type or a method gets a plain clause on what it is and why it matters to our score the first time a
reply names it, and the link between two topics is said, never left for him to infer. The full
register: `private/CLAUDE.md` → How to write for Saber.

**A reply that opens a session on a task, ends a unit of work, or asks him something has one shape**
(Saber, 2026-10-16), in this order:

1. **Where we are** — the refresher, in mostly the same words each time: what this task asks in plain
   words and why it would move the score, and how it sits beside its neighbour tasks and ideas (read
   from the task file's `## In plain words`); then the trail to here, the forks taken in a few lines,
   and what the last step found. The full two paragraphs open a session's first reply and the reply
   that ends a unit of work; a reply between them carries one sentence of it (Saber, 2026-10-16).
2. **What this step found** — in plain groups: what works, what does not, what we could not tell. A
   sentence carries one number, and every number says what it is (a score, or a difference between
   two named scores) and where it was scored (which fold, or the leaderboard); a comparison of several
   numbers is a small table. Review bookkeeping is one line at most.
3. **What I need from you** — or `Nothing`. Each option says what it costs and what it gets, in words
   he already uses; a word coined during the work is spelled out in the sentence or dropped.
4. **Next steps** — one line each.
5. **Files** — last: one line per path with its repo and a verb only — new · edited · committed
   `abc1234`, not pushed · uncommitted. No sentences; the substance is above.

**Length follows the reader, not a cap** (Saber, 2026-10-16). Read the chat: when he has asked for a
repeat, asked several questions in a row, or asked what a word means, he is not following, so the
next reply explains more — a fuller refresher, the biology spelled out, an example; when he is briefed
and the work is moving, keep to the least that still says what was found and what it means. Short
never means vague: cut bookkeeping, restated numbers and review chatter, never the sentence that says
what a result means. The detail lives in a file and the reply points at it. An answer to a question is
the answer first, then the pointers that carry the evidence; a yes/no is one sentence; a correction to
your own earlier claim is one sentence plus a pointer. **One report per unit of work** (Saber,
2026-10-07): a child's return, a waiter's exit or a background notice is an input to you, not an
occasion to write to him. The STATUS block follows when the state, task or pointer changed or an ASK is
raised: `Nothing` in section 3 means `fyi`; a question you cannot proceed without means
`waiting · expect_by HH:MM` (end of the working day unless you know better, with `waiting_on: Saber`)
plus a typed `decision` ASK.

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
with `T<n> · from <your session id> · one line`; a session with no task writes `—` in place of the
id (Saber, 2026-09-21 — not every session works a pre-determined task), and work you will discard
is `— · throwaway`. The mother is the part the board cannot do without.**
**Keep-alive** (Saber, 2026-10-14): the prompt cache lives one hour, and he often reads a report
later than that. So a reply that ends a unit of work with a `returned` or `waiting` block first
schedules one ping: `CronCreate`, `recurring: false`, at now + 50 min (read the clock in the turn's
last Bash call: `date -v+50M '+%M %H %d %m'`), prompt `keep-alive (auto): reply with the single
word ok. No tools, no STATUS block.` It first deletes this session's earlier keep-alive that has not
fired, and is skipped while `/away` pings are pending. A keep-alive is answered `ok` and schedules nothing.

**Models and effort** (Saber, 2026-10-14): a mother runs Saber's session model at `xhigh` — the
only place the whole picture is held; a child runs at effort `high` (pass `effort: "high"` to an
unwired Agent or a Workflow `agent()`, which otherwise inherit `xhigh`). The model follows the
kind of work: code, box scripts, measurement and design review on Opus 5.5 (the default,
`CLAUDE_CODE_SUBAGENT_MODEL=claude-opus-5-5`; Analyst, Critic); a write-up checked against its
artefacts, documents and literature on Sonnet 5.5 (Researcher, Verifier, or `model: "sonnet"` on
a Critic); counting and converting on Haiku. Say why when you override. The queue, the mother session and dispatch: `private/CLAUDE.md` → The mother session.

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
| a challenge year's data, packaging, `vcc submit` | `challenges/<year>/CLAUDE.md` | `check_modelname.py --propose`; a submit goes up on Saber's GO |
| what goes up on the board next — approved and proposed entries, the daily auto-submit | `private/SUBMISSION-QUEUE.md` (the rules; its list is **generated** from `private/queue/Q<n>.md`, one file per entry — `private/queue/README.md`; never edit between its markers) | propose with `submission_queue.py --new`; `private/submissions/` records what went up; the job: ADR 0012 |
| QC on our h5ads, the controls bundle | `challenges/<year>/CLAUDE.md` · `sidechain.data.profile` · `sidechain.ingest.checks` | `reports/06` |
| a generic single-cell method: QC, scVI, nf-core, instrument data | `/bio-research:single-cell-rna-qc`, `:scvi-tools`, `:nextflow-development`, `:instrument-data-to-allotrope` | their `SKILL.md` under `private/research/protocol/plugins/bio-research/skills/` when the skill is not listed |
| a research question, "what have we tried" | `private/CLAUDE.md` | `private/research/master.md` · `private/research/INDEX.md` · `private/RESULTS.md` |
| **the final phase** — Oct 22, D/E/F, the final bundle, a "revisit on Oct 22" clause, a submission slot | `private/FINAL-PHASE.md` (T93; its generated half is `final_phase.py --write`, never hand-edited) | write the clause where it belongs — the idea's Revisit-if, or `(final-phase: <step>)` beside a code comment — and regenerate (final-phase: none) |
| "what should I work on" | `private/TODO.md` `## p1` (generated) | `private/CLAUDE.md` → The mother session |
| proceed or ask: a fork, a GO | this file § When to proceed · § Replying | — |
| `pick up T<n>` | `private/tasks/T<n>.md`: the lede, `## Next step`, the last five Log lines (the prompt hook names them) | the newest hand-off the hook names; other sections on demand |
| spawn a Researcher, a Verifier, a workflow | `private/CLAUDE.md` → The mother session | every spawn names a task — there is no standing brief · write contracts in `private/research/README.md` |
| a paper, a literature review | `/paper-intake` (one paper) · the Researcher (a sweep) | `private/research/reading/` · `private/literature.md` |
| commit, push | this file § Committing · `/sidechain-commit` | — |
| a hand-off to the next session (a context window ending, or Saber's ask) | `private/handoffs/README.md` — the shape, the rules, the index | — |
| the GPU box, more RAM, a full-corpus stream | `/sidechain-brev` | `box_plan.py` prices a queue before the GO; `scripts/boxkit/` |
| a data download, disk, a new corpus, a licence | this file § Data · `configs/datasets.yaml` | ADR 0003 |
| the website, a post, LinkedIn | `site/README.md` · `/post` | `private/site/NOTES.md` · `private/site/VOICE.md` |
| the board, `ledger.py`, asks, status, the queue | `private/HOWTO-desk.md` §1 and §8 (the desk is Saber's own session; §2–§7 are his vocabulary) | `reports/12` |
| Claude Science: what goes there, handing it a task, its hooks and commits | `CLAUDE-SCIENCE.md` | `private/HOWTO-science.md` · ADR 0011 · `/science-brief` |
| a metric or statistical term | `private/GLOSSARY.md` (grep it) | — |
| a checker fired: why | that `private/research/protocol/check_<x>.py` (docstring, `--help`) | `reports/11` §8 |
| wiring hooks, skills, `.claude/settings.local.json` (gitignored; template in `private/research/protocol/machine-local/`) | `private/README.md` § After cloning | — |
| an agent's role | `agents/<role>.md` (researcher · verifier · analyst · critic) | its wrapper in `.claude/agents/`, which pins its model |
