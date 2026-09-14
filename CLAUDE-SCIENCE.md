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
  `private/TODO.md`. A Science session files no STATUS block and appears on no ledger; **the mother
  session records what it did**, the same way it records a subagent's work. Reading the board is a
  different matter — see § The board.
- **The private/public split.** Do not paste `private/` reasoning into a Science session that will
  save it as an artifact. Same rule, new surface: publishing is one save, un-publishing is not.

## The handoff, both directions

**Code → Science** is the Analyst brief, unchanged in shape: `T<n>` · the one question · the data
path · where results land. Paste it as the first message of a Science session. Keep the two house
rules that matter most there — **every number carries its null** (scramble, permuted labels,
context-mean, replicate ceiling) and **check the minimum before writing "every"**; a Science session
will not know them otherwise until they are in project memory or a skill.

**Science → Code** is an artifact plus one line of result. Ask the session for the dated
`## Outcome` paragraph as text; the mother session appends and commits it, so path discipline and the
public/private call stay with the agent that has the repo. Default to Science handing back text and
files — grant it write access into the checkout only when you specifically want that.

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

1. **Grant `~/data/sidechain/` read-only.** The repo grant (`~/code/sidechain`, read-only,
   2026-09-12) covers both trees including `private/`, but **the data is not in the repo** — until
   it is granted a session sees no h5ad, no bundle, no cache. Read-only is enough for profiling,
   scoring and plotting; grant `derived/`, `cache/` and `vcc2026/` rather than the whole tree if you
   prefer. Keep the repo grant read-only: commits belong to the agent that owns the path discipline.
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
