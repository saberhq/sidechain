# CLAUDE-SCIENCE.md — Claude Science beside Claude Code

Companion to `CLAUDE.md`. That file is the rules for sessions working *in this checkout*. This one
says which asks leave the checkout and go to a Claude Science session instead, what comes back, and
what must never move. Written against the platform as measured 2026-09-12; the lines marked
**(setup)** are things not yet wired.

## The seam, in one line

**Claude Code owns state that gets committed or submitted. Claude Science owns measurements and
arguments** — work whose value is the number, the figure and its provenance, not the file's place in
the tree. A Science session has no git, no hooks, no board and no `vcc`; it has an artifact store
that records how every result was made, approval-gated remote compute, and biology data connectors.

## What a Science session has that this checkout does not

- **Provenance without a register step.** Every saved file carries the exact code that made it, the
  environment snapshot, and the input artifacts it consumed (`host.lineage[vid]["code"]` replays it;
  `host.artifacts(search="mirror loco SER-6")` finds it across every session in the project). This is
  the Analyst's `~/data/sidechain/runs/<slug>_<date>/` contract, enforced by the platform rather than
  by the brief. **lamindb stays canonical for derived *data*;** Science holds the *argument* layer —
  score comparisons, diagnostics, the figures that go in a report or a post.
- **Remote dispatch with an approval card.** A session prepares inputs locally, submits a job to a
  configured host, parks, and the outputs land back in the artifact store. The box-sharing rule
  becomes a modal you click rather than a `brev ls` convention. **(setup)** No compute target is
  configured yet — see § Setup.
- **Biology connectors that return resolvable identifiers.** Exercised 2026-09-12 and returning
  records: PubMed (`search_articles` → PMIDs), GEO/ArrayExpress (`geo_search_series` → GSE
  accessions with titles and summaries), UniBind (`unibind_search_tfbs` → 982 CTCF datasets).
  Answered correctly but empty for the probe query: GTEx (`gtex_median_expression`), CELLxGENE
  CellGuide. **Attached but not exercised** — the host validates calls against each one's schema,
  which is evidence the server is wired, not that a query returns what you want: Ensembl BioMart,
  bioRxiv/medRxiv, MyGene/GO/Reactome, InterPro, Open Targets, gnomAD/ClinVar, arXiv/OpenAlex
  (OpenAlex additionally needs a key — see § Setup). Plus `fetch_article_fulltext` by DOI, which
  saves the text and any PMC figures to the workspace. Method names and argument schemas differ from
  the obvious guess; read the connector's skill doc before writing a loop.
- **Method skills that match rungs already on the ladder:** `fair-esm2` (the ESM-2 gate),
  `borzoi` (sequence → predicted RNA-seq/DNase track delta — a cis-prior candidate), `scgpt`
  (gene-level representations for perturbation/GRN tasks), `scvi-tools`. Writing surfaces:
  `literature-review`, `figure-style`, `figure-composer`, `paper-narrative`.
- **Sub-agents that run code and return typed output**, briefed one task at a time — the same
  "a spawn with no task is a spawn to refuse" rule, and the same model policy (a child on the
  reasoning default; counting and schema-bound reading on the cheap one).

## What never leaves Claude Code

- **git, in both repos.** Science has no git surface. Staging by explicit path, the shared-file
  blob rule, `/sidechain-commit` — unchanged, and a Science session cannot break them because it
  cannot reach them.
- **Anything `vcc`.** The PAT lives in the login keychain and the ACL names the interpreter that
  created it; a Science sandbox is a different interpreter and would re-trigger exactly the
  2026-09-07 prompt-on-every-call failure. Also measured: `virtualcellchallenge.org` is **not** on
  Science's network allowlist (the proxy returns 403), so board snapshots and `vcc submit` are yours.
- **`saberhq.com` and the site deploy** — same allowlist result.
- **Writing to the protocol machinery**: hooks, the STATUS block, `ledger.py`, the checkers,
  `private/TODO.md`. A Science session files no STATUS block; **the mother session records what it
  did**, the same way it records a subagent's work. The one thing it can put on the board itself
  is an ask, and it does that without touching the store — § Asks from a Science session. Reading
  the board is a different matter — see § The board.
- **The private/public split.** Do not paste `private/` reasoning into a Science session that will
  save it as an artifact. Same rule, new surface: publishing is one save, un-publishing is not.

## The handoff, both directions

**Code → Science** is the Analyst brief, unchanged in shape: `T<n>` · the one question · the data
path · where results land. Paste it as the first message of a Science session. Keep the two house
rules that matter most there — **every number carries its null** (scramble, permuted labels,
context-mean, replicate ceiling) and **check the minimum before writing "every"**; a Science session
will not know them otherwise until they are in project memory or a skill.

**Science → Code** is an artifact plus one line of result. Ask the session for the dated
`## Outcome` paragraph as text; the mother session quotes it verbatim and commits it, so path
discipline and the public/private call stay with the agent that has the repo, and a reader can always
tell Science's words from Code's (the landing rule, § Asks and results). Default to Science handing
back text and files — grant it write access into the checkout only when you specifically want that.

## Asks and results from a Science session (ADR 0009, ADR 0010)

A Science session cannot append to `asks.jsonl` — the checkout is read-only to it, and a second
writer would race `ledger.py` for the same `A<n>`. It writes an **outbox** under the data root
instead, and a Claude Code session folds it in through the same `ask_open` every other ask uses.
`ledger.py` stays the only writer; the outbox is transport, not record.

- **Where:** `~/data/sidechain/science/outbox/<frame_id>.jsonl` — one file per Science session,
  named by that session's own frame id, append-only. The directory exists once ingest has run
  once; a session may create it.
- **What a line is** — JSON, one per line, `src_id` minted by the session (any short unique
  string) and the ask fields the board already knows:

  ```
  {"src_id": "brief-1", "event": "open", "type": "decision", "tid": "T71",
   "question": "…?", "recommendation": "…", "options": ["a", "b"], "blocking": false,
   "pointer": "~/data/sidechain/runs/…", "read_minutes": 2}
  {"src_id": "brief-1-done", "event": "consume", "id": "A87"}
  ```

  `type` is `decision` or `fyi` (a `review` needs a `?`). Unknown keys are dropped, strings are
  capped, a bad line is skipped and named — never a broken store.
- **How it lands:** `ledger.py science ingest` — a launchd timer runs it every three minutes on
  the Mac, and `/desk` runs it first thing, so a line is on the board within minutes. The ask
  appears as `sci-xxxx`, joined to the session's card because `from_id` **is** the frame id.
  `src_id` makes ingest idempotent: the same line never opens twice, so a session may re-emit.
- **The answer comes back the way it always did:** Saber answers at the desk, `ledger.py ask
  answer A87 "b"` writes it, and the session reads its own answer out of
  `private/agents/asks.jsonl` under the read-only grant. When it has acted, it writes the
  `consume` line above; a consume is honoured only from the frame that raised the ask.
- **Discipline:** one open ask per dispatch. A session that would raise a second one returns
  instead — a Science session is not sitting in Saber's editor, and returning costs less than
  blocking. If asks average more than one per dispatch over a week, the channel goes back to
  `fyi` only (the ADR's revisit trigger).
- **A result goes back the same way** (ADR 0010) — the dated `## Outcome` paragraph as `text`,
  with its evidence beside it, so the Code side verifies instead of trusting prose:

  ```
  {"src_id": "brief-1-result", "event": "result", "tid": "T71", "idea": "<idea slug>",
   "text": "2026-09-14 · <the paragraph>", "artifact": "<artifact version id>",
   "run_dir": "~/data/sidechain/runs/<slug>_<date>/"}
  ```

  `artifact` or `run_dir` is required — a result with neither is a claim and is skipped. `tid`
  is the brief's; a session started without a brief leaves it empty and a Code session attaches
  one at landing. Ingest lists it as `R<n>` under `/desk`'s RESULTS. The ledger never writes a
  research file, and ingest never mints a task id.
- **Landing a result keeps the two authors apart** (2026-09-16). It is a Code session's job, in
  this order:
  1. Check the paragraph against its evidence.
  2. **Quote it verbatim** under the destination's `## Outcome` — the idea file's, or wherever
     Saber sends a result that carries no idea slug — as a `>` block identical to the ledger's
     `text`, headed `**<date> — R<n>, Claude Science session <frame id, 8 chars> (artifact <id>,
     run <run_dir>), verbatim:**`. The frame id is the Science session's own (`from_id`
     in `results.jsonl`), not the Code session that wrote the brief.
  3. **Anything the Code side adds is its own dated entry**, headed with its session id: the
     re-derivation, a correction, a reconciliation with another count. Never unheaded text under
     the quote, and never an edit to the quote — a correction is a new entry naming what it corrects.
  4. Commit, then `ledger.py science triage R<n> --into <file>`.

## The board, from a Science session

**The dashboard's port is unreachable and cannot be made reachable.** `dashboard.py --serve` binds
127.0.0.1 on the Mac; inside a Science sandbox `127.0.0.1` is the sandbox, and a loopback or
private-range target is a hard security refusal, not an allowlist decision. There is nothing to
grant.

**The stores are reachable, and they are the same source of truth.** `dashboard.py` re-reads
`private/agents/{asks,status,queue,box}.jsonl` on every request precisely so the page cannot go
stale; a session with the repo mounted read-only reads those files directly.
`dashboard.py --json` itself does *not* run here — it calls `registry_sessions()`, which globs
`~/.claude/sessions`, outside the grant (and outside what a Science session should be given).

`board_snapshot.py` (beside this file) is the read-only subset: open asks, last state per session,
queue counts, last box event. **Its session states are store-derived, not live** — the `idle`/`gone`
transitions come from the Stop hook and the process registry, so a session that died without firing
the hook still reads `running` here. Treat the count as an upper bound; the dashboard is
authoritative on liveness, this is authoritative on what was written.

## Routing table — when the ask is …

| when the ask is … | where it goes | why there |
|---|---|---|
| score two emitter or dispersion arms against one mirror bundle | **Science** | one artifact per arm, each carrying its own code and env; the comparison is reproducible a month later without a run README |
| a diagnostic that answers one number (coverage, depth, off-axis UMI share, fallback rate) | **Science** | the Analyst contract, with lineage instead of a results dir |
| a corpus or prior candidate: does it exist, what does it cover, **what is its control arm** | **Science** | connectors return accessions; `fetch_article_fulltext` returns the methods text a control-arm quote must come from |
| "is this claim true" on a specific paper | **Science** | verbatim quote + DOI that resolves today, which is the Verifier's output contract |
| a figure for a report, a post, or the eventual write-up | **Science** | `figure-style` / `figure-composer`, and the figure keeps its data lineage |
| profile an h5ad, gene-axis or header QC | either | Science if you want the provenance kept; Claude Code if the answer only decides the next command |
| a module, a contract test, a `datasets.yaml` or `data_sources.yaml` block | **Claude Code** | the code lands in the repo |
| commit, push, `vcc prep`/`submit`, the site | **Claude Code** | see § What never leaves |
| "what is open / what is stuck" | **either** | the dashboard for liveness; a Science session can read the same stores read-only (§ The board) when it needs the board state to decide something |
| "what have we tried" | **both** | `private/research/master.md` is the argument; `host.artifacts(search=…)` is the measurement layer. Two indexes, two different layers — do not merge them |

## Setup — five things, in order of payoff

1. **Grant `~/data/sidechain/` read-write.** The repo grant (`~/code/sidechain`, read-only,
   2026-09-12) covers both trees including `private/`, but **the data is not in the repo** — until
   it is granted a session sees no h5ad, no bundle, no cache. Read-write, not read-only, because
   the outbox (§ Asks from a Science session) and an Analyst's `runs/<slug>_<date>/` live there
   (ADR 0009); nothing under it is tracked, so no single-writer rule is at stake. Keep the repo
   grant read-only: commits belong to the agent that owns the path discipline.
2. **Add the 64 GB / GPU host as a compute target.** This is the biggest one. Science's sandbox runs
   on the same 16 GB Mac — measured 10 cores, 16 GiB — so it inherits the same ceiling and **cannot
   package a submission either**. With the box configured, `vcc prep`-sized packaging, `gpudge`
   scoring and full-corpus streams become jobs a session submits and waits on.
3. **Allowlist `virtualcellchallenge.org`** if you want a session to pull board snapshots itself.
   Optional: `sidechain.eval.leaderboard` already does this from your terminal.
4. **Add an OpenAlex API key** if citation-graph sweeps are worth it. Measured: no credential is
   stored, and OpenAlex rejects keyless calls. PubMed, bioRxiv and GEO need nothing.
5. **Save the Analyst and Critic briefs as specialist profiles** so a Science session starts in role
   instead of being handed the role every time.

## Project memory — what to seed, what to withhold

Seed the handful of facts a Science session will otherwise re-derive or guess wrong: the gene axis is
**18,533 symbols, `var` empty**; the axis is curated and off-axis genes carry ~30 % of every cell's
UMIs, so **normalise to the measured library, never the on-axis subtotal**; control arms come from the
methods, never the labels; raw integer counts, never log1p; the 16 GB ceiling; data lives outside the
repo. Withhold everything from `private/` that is reasoning rather than fact.

## What this does not change

The rung ladder, metric-first, nothing-trusted-until-the-mirror-scores-it, sparse-only, ADR 0005
naming, two submissions a day. Science is another place to do the measuring. It is not another place
to decide what counts as a win.

## Project Agent Context — last synced 2026-09-16

This block is canonical; the Claude Science project-settings box holds a copy of it. Edits are
made here first and re-pasted into that box, and the date on this heading is updated to the day of
the paste — nothing reads that box but every Claude Science session in the project inherits it, so
a block that has quietly diverged is invisible until it misleads a session, and worse than no copy.
Fenced so it copies without reflowing; keep it byte-identical to the box.

```markdown
## Sidechain — Claude Science side

**The project.** Sidechain is Saber's solo entry to Arc Institute's Virtual Cell Challenge 2026,
run as a multi-agentic workflow with Saber as the only human in the loop. Two repos, both at
`~/code/sidechain` (read-only here): the public core, and the gitignored `private/` tree that
holds the research argument, the protocol and the board.

**The seam.** Claude Code owns state that gets committed or submitted — git, the `vcc` CLI, the
site, the hooks and ledgers. Claude Science owns measurements and arguments: scoring comparisons,
diagnostics, literature and corpus sweeps, figures. Read `CLAUDE-SCIENCE.md` (public root) for the
routing and `private/HOWTO-science.md` for the operating page.

**Read the repo; do not recall it.** Spec numbers, the gene axis, control-arm definitions, the
rung ladder and the naming rule come from `private/research/protocol/facts.json`,
`challenges/vcc2026/CLAUDE.md`, `private/ARCHITECTURE.md` and `private/GLOSSARY.md` each session.
Never mirror them into memory — the git is canonical.

**House rules for anything reported.** Every number carries its null (scramble, permuted labels,
context-mean, or the replicate ceiling). Check the minimum before writing "every". A control arm
comes from a paper's methods, never from its labels. Raw integer counts, never log1p. Sparse only:
never allocate a dense gene-by-gene matrix. Nothing is trusted until the local mirror scores it.

**Where output lands** (A88). Both a `runs/<slug>_<date>/` directory under `~/data/sidechain`
(read-write; the interop surface the Code side reads) and a saved artifact, which carries the code
and environment that produced it. Artifact-only is for a one-off diagnostic nothing downstream
consumes.

**Talking to the Code side.** Append to `~/data/sidechain/science/outbox/<frame_id>.jsonl`:
`open`/`consume` for asks, `result` for a finished measurement (dated Outcome paragraph as `text`,
plus the artifact version id or run_dir). `ledger.py science ingest` folds it in; `ledger.py` stays
the only writer of the ledgers. The Code side lands that paragraph as a verbatim quote and writes
its own reading in a separate entry, so write it to stand alone: the date, what was measured and
how, the null, the numbers, the one surprise, the caveat — nothing that needs this conversation
to make sense. One open ask per dispatch — a session that would raise a second
returns instead. Announce every outbox line in the reply that writes it, with its `src_id`, and
name the `A<n>`/`R<n>` it became once the board shows it.

**Walls.** `virtualcellchallenge.org` and `saberhq.com` are off the network allowlist, and the
board's `127.0.0.1:7391` is unreachable from the sandbox — read `private/agents/*.jsonl` instead.
The GPU box is the SSH target `sidechain-gpu`; use it, never create, stop or delete one, and start
every remote command with `export PATH="$HOME/.local/bin:$PATH"` and `cd ~/code/sidechain`.

**Register.** Saber is a computational biologist, not an ML engineer: biology as technical as it
needs to be, ML and infrastructure in plain words. Lead with the number, then the pointer, then the
caveat. One number per sentence.
```
