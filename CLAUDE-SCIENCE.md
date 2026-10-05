# CLAUDE-SCIENCE.md — Claude Science beside Claude Code

Companion to `CLAUDE.md`. That file is the rules for every session working in this checkout, and
since 2026-09-29 that includes Claude Science sessions (ADR 0011). This one says what a Science
session is for, how it gets its hooks, what stays with Claude Code, and how the two hand work to
each other. Written against the platform as measured 2026-09-29.

## The seam, in one line

**Claude Science takes the research; Claude Code takes the implementation-heavy work — and both
record it the same way.** A Science session told `pick up T<n>` is that task's mother, like a Code
session: it reads the task file, claims it, works it, writes the Log line, the idea file's
`## Outcome` and the `CHANGELOG.md` entry, and commits by explicit path. What it lacks is Claude
Code's hooks, so it runs them as commands (`private/research/protocol/science_session.py`, the
`sidechain-pickup` skill). What stays in Claude Code is what needs the Mac's keychain or loopback:
`vcc`, pushing, the site deploy, the board's page — § What stays with Claude Code.

## What a Science session has that this checkout does not

- **Provenance without a register step.** Every saved file carries the exact code that made it, the
  environment snapshot, and the input artifacts it consumed (`host.lineage[vid]["code"]` replays it;
  `host.artifacts(search="mirror loco SER-6")` finds it across every session in the project). This is
  the Analyst's `~/data/sidechain/runs/<slug>_<date>/` contract, enforced by the platform rather than
  by the brief. **lamindb stays canonical for derived *data*;** Science holds the *argument* layer —
  score comparisons, diagnostics, the figures that go in a report or a post.
- **Remote dispatch with an approval card.** A session prepares inputs locally, submits a job to a
  configured host, parks, and the outputs land back in the artifact store. The box-sharing rule
  becomes a modal you click rather than a `brev ls` convention. The GPU box is configured as an SSH
  target; a box made for a Science task is named `sidechain-gpu-sci-T<n>` (ADR 0011 § 6).
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

## What stays with Claude Code

- **Pushing, in both repos.** A Science session commits (§ Committing from Science); the remotes
  authenticate through the login keychain, which the sandbox cannot reach, and a push is Saber's go
  in any window.
- **Anything `vcc`.** The PAT lives in the login keychain and the ACL names the interpreter that
  created it; a Science sandbox is a different interpreter and would re-trigger exactly the
  2026-09-07 prompt-on-every-call failure. Also measured: `virtualcellchallenge.org` is **not** on
  Science's network allowlist (the proxy returns 403), so board snapshots and `vcc submit` are Code's.
- **`saberhq.com` and the site deploy** — same allowlist result.
- **Creating or deleting a GPU box** — not built for Science yet (ADR 0011 § 6): a Code session
  creates `sidechain-gpu-sci-T<n>` and Science submits jobs to it.
- **The four Code-only skills**: `sidechain-brev` (the `brev` CLI), `desk`, `wrap`, `compete`.
  Every other project skill is a procedure a Science session follows by reading its `SKILL.md`.
- **The private/public split.** Do not paste `private/` reasoning into an artifact or a public file.
  Same rule, new surface: publishing is one save, un-publishing is not.

## The hooks, as commands

Claude Science does not read `.claude/settings.local.json`, so none of the SessionStart,
PostToolUse, UserPromptSubmit or Stop hooks fire there. `science_session.py` is the same machinery:

- `start T<n> --frame <id>` — the claim `pick up T<n>` writes in Code, the task's fields and ledger
  rows, and the SessionStart checkers.
- `check` — the PostToolUse checkers, including `check_todo.py --write`, which regenerates
  `private/TODO.md` from `private/tasks/`.
- `changelog --entry <file>` — the `/sidechain-commit` blob routine for `CHANGELOG.md`.
- `commit {public|private} -m … <paths>` — explicit paths only (see below).
- `return T<n>` — hands the task back; its claim closes.
- `ledger -- <args>` — any `ledger.py` command.

The commands that run `ledger.py` or the checkers export the Science frame id as
`CLAUDE_CODE_SESSION_ID`, the variable `ledger.py` falls back to, so claims and asks carry the Science session's id. There is no STATUS block: the board draws a
Science card from Science's own database (`science_panel.py`), and a claim stays live while the
frame is running or has been active within a day (`dashboard.science_live`). Run everything with the
repo's `.venv/bin/python`; the sandbox refuses joblib's worker processes, so set
`JOBLIB_MULTIPROCESSING=0` for code that uses them.

## Committing from Science

`science_session.py commit` stages the named files only, and refuses `.`, a directory, a `private/`
path in the public repo, a shared file (`CHANGELOG.md`, `RESULTS.md`) not staged through
`changelog`, and an index that already holds another session's staged change. The author is the
repo's own, as its history has it; the message carries a `Co-Authored-By` trailer, and a private
commit a `Claude-Science-Session: <frame id>` trailer (the public `commit-msg` hook strips session
trailers). New public material still needs Saber's green light before it is pushed (`/sidechain-commit`).

## The handoff, both directions

**Code → Science** is a task file. Its Next step names the one question, the data path and **the
null** (scramble, permuted labels, context-mean, replicate ceiling) — the house rule a fresh session
most needs written down. Saber says `pick up T<n>` in a Science session; `/science-brief` is now
those two steps (ADR 0011 § 8).

**Science → Code** is the commit. The Science session writes its own Outcome entries, headed with its
own session id (the frame id's first 8 characters), and commits them; a Code session picking the task
up next reads them the way it reads any mother's.

## Asks from a Science session

A Science session holding a task opens an ask through the ledger, with its id attached:
`science_session.py ledger --frame <id> -- ask open --type decision --from science-<8 chars> --tid
T<n> --question "…?"`. One open ask per task at a time; the reply that opens it names its `A<n>`.
Saber answers at the desk as for any ask, and the session reads the answer from
`private/agents/asks.jsonl`.

## Results from a session with no task (ADR 0009, ADR 0010)

A Science session Saber started without a task is exploration, and the outbox stays its channel.

- **Where:** `~/data/sidechain/science/outbox/<frame_id>.jsonl` — one file per session, named by that
  session's frame id, append-only. `ledger.py science ingest` folds it in (a launchd timer every
  three minutes, and `/desk`); `src_id` makes ingest idempotent.
- **What a line is** — JSON, one per line, `src_id` minted by the session:

  ```
  {"src_id": "brief-1", "event": "open", "type": "decision", "tid": "",
   "question": "…?", "recommendation": "…", "options": ["a", "b"], "blocking": false,
   "pointer": "~/data/sidechain/runs/…", "read_minutes": 2}
  {"src_id": "brief-1-result", "event": "result", "tid": "", "idea": "<idea slug>",
   "text": "2026-09-14 · <the paragraph>", "artifact": "<artifact version id>",
   "run_dir": "~/data/sidechain/runs/<slug>_<date>/"}
  {"src_id": "brief-1-done", "event": "consume", "id": "A87"}
  ```

  `type` is `decision` or `fyi`. A `result` needs `artifact` or `run_dir`; with neither it is a claim
  and is skipped. Unknown keys are dropped, strings are capped, a bad line is skipped and named.
- **Landing a result** is a Code session's job, keeping the two authors apart (2026-09-16): check the
  paragraph against its evidence; quote it verbatim under the destination's `## Outcome` as a `>`
  block headed `**<date> — sci-<n>, Claude Science session <frame id, 8 chars> (artifact <id>, run
  <run_dir>), verbatim:**`; put anything the Code side adds in its own dated entry headed with its
  session id; commit; then `ledger.py science triage sci-<n> --into <file>`.

## The board, from a Science session

**The dashboard's port is unreachable and cannot be made reachable.** `dashboard.py --serve` binds
127.0.0.1 on the Mac; inside a Science sandbox `127.0.0.1` is the sandbox, and a loopback or
private-range target is a hard security refusal, not an allowlist decision. There is nothing to
grant.

**The stores are reachable, and they are the same source of truth.** `dashboard.py` re-reads
`private/agents/{asks,status,queue,box}.jsonl` on every request precisely so the page cannot go
stale; a Science session reads those files directly (`ledger.py open`, `ledger.py show T<n>`).
`dashboard.py --json` itself does *not* run here — it calls `registry_sessions()`, which globs
`~/.claude/sessions`, outside the grant (and outside what a Science session should be given).

**Session states read from the stores are not live** — the `idle`/`gone` transitions come from the
Stop hook and the process registry, so a session that died without firing the hook still reads
`running`. The dashboard is authoritative on liveness; the stores are authoritative on what was
written.

## Routing table — when the ask is …

| when the ask is … | where it goes | why there |
|---|---|---|
| score two emitter or dispersion arms against one mirror bundle | **Science** | one artifact per arm, each carrying its own code and env; the comparison is reproducible a month later without a run README |
| a diagnostic that answers one number (coverage, depth, off-axis UMI share, fallback rate) | **Science** | the Analyst contract, with lineage instead of a results dir |
| a corpus or prior candidate: does it exist, what does it cover, **what is its control arm** | **Science** | connectors return accessions; `fetch_article_fulltext` returns the methods text a control-arm quote must come from |
| "is this claim true" on a specific paper | **Science** | verbatim quote + DOI that resolves today, which is the Verifier's output contract |
| a figure for a report, a post, or the eventual write-up | **Science** | `figure-style` / `figure-composer`, and the figure keeps its data lineage |
| profile an h5ad, gene-axis or header QC | either | Science if you want the provenance kept; Claude Code if the answer only decides the next command |
| a research task end to end — measure, read, decide, write up (`pick up T<n>`) | **Science** | it records and commits like any mother (ADR 0011) |
| a module, a contract test, a `datasets.yaml` or `data_sources.yaml` block — implementation-heavy work | **Claude Code** | Science can commit code too, but a long build is Code's strength |
| commit | **either** | by explicit path, in whichever window did the work |
| push, `vcc prep`/`submit`, the site | **Claude Code** | see § What stays with Claude Code |
| "what is open / what is stuck" | **either** | the dashboard for liveness; a Science session reads the same stores (§ The board) |
| "what have we tried" | **both** | `private/research/master.md` is the argument; `host.artifacts(search=…)` is the measurement layer. Two indexes, two different layers — do not merge them |

## Setup state

- **Grants** (Science app → the folder grants): `~/code/sidechain` read-write, both repos;
  `~/data/sidechain` read-write; `~/.local/share/uv`, so `.venv/bin/python` starts in the sandbox;
  `~/.lamin` read-write, for lamindb registration.
- **The GPU box** as an SSH compute target — configured; each new box needs
  `scripts/brev_ssh_mirror.sh <name>` once and its own host entry in the app.
- **OpenAlex key** — stored.
- **Profiles**: Sidechain Analyst and Sidechain Critic, generated from `agents/analyst.md` and
  `agents/critic.md`; the repo files stay canonical.
- **Skills**: `sidechain-pickup` is canonical at `private/research/protocol/sidechain-pickup/SKILL.md`;
  the Science skill of the same name points there.
- **Not built**: creating or deleting a GPU box from Science (ADR 0011 § 6).

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

## Project Agent Context — last synced 2026-10-05

This block is canonical; the Claude Science project-settings box holds a copy of it. It is kept to
pointers on purpose (Saber, 2026-10-05), so it only changes when a path does: everything else a
Science session needs is read from the git, starting with `CLAUDE.md`. Edit here first, re-paste
into the box, and set this heading's date to the day of the paste.

```markdown
Sidechain: Saber's entry to Arc's Virtual Cell Challenge 2026. The git is canonical; read it, never recall it. Repos: ~/code/sidechain (public) and ~/code/sidechain/private, both read-write. Data: ~/data/sidechain. Every session first reads ~/code/sidechain/CLAUDE.md, then CLAUDE-SCIENCE.md; for research or planning, also private/CLAUDE.md. "pick up T<n>": follow private/research/protocol/sidechain-pickup/SKILL.md. Any other project skill: read private/research/protocol/<name>/SKILL.md and follow it. Commit only through science_session.py commit; never push, never run vcc.
```
