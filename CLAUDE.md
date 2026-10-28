# CLAUDE.md — Sidechain

Sidechain is an entry to the Arc **Virtual Cell Challenge 2026**: predict how a cell's transcriptome shifts when a
gene is silenced. This file holds the rules for this repo's code: what is where, where the data lives, the house
rules, how a change is committed, and the environment.

## The task, in one paragraph

Given expression data for genes that *have* been silenced, predict what happens when you silence a gene **never seen
silenced**, in cell lines never seen: six anonymized lines (A/B/C for validation, closed Oct 21; D/E/F released
Oct 22, the only three that decide prizes), 300 knockdowns × 400 cells per line, raw counts, 18,533 genes, six
cell-eval2 metrics. Arc ships no training data. `challenges/vcc2026/CLAUDE.md` + `config.yaml` are authoritative for
the spec and the submission contract — never re-derive it from the web. `challenges/<year>/CLAUDE.md` loads only
when you read a file in that folder: open it yourself before touching that tree.

## Where things are

- `src/sidechain/` — the year-agnostic core: `data/ ingest/ priors/ models/ eval/ submit/ utils/ viz/`.
- `challenges/vcc2025/`, `vcc2026/` — thin adapters: `config.yaml` + `CLAUDE.md`.
- `configs/` — `datasets.yaml` (**the corpus registry**: datasets, `route`, `control_label`, budgets);
  `data_sources.yaml` (the prior registry); `eval.yaml` and `model.yaml` (2025-era).
- `tests/` — contract tests. `agents/` — the four role briefs (researcher, verifier, analyst, critic), each wired in
  `.claude/agents/`. `scripts/` — `standings.py`, `lamin_*.py`, `brev_*.sh`, `boxkit/`.
- `site/` — saberhq.com/sidechain, themeless Hugo (`site/README.md`). `notebooks/` — the 2025 loop.

## Data lives OUTSIDE the repo

Everything bulky is under `~/data/sidechain/`: `external/<host>-<record>/` is third-party bytes as published
(`PROVENANCE.json`; re-downloadable, delete first under disk pressure); `derived/<source>/` is ours (`LINEAGE.json`;
expensive, back it up); plus `vcc2025/`, `vcc2026/`, `cache/`, `runs/`. **A streamed dataset's `external/` dir holds
only `PROVENANCE.json`: correct, not a failed download.** `route:` and the stream gate: the `configs/datasets.yaml`
header. **A file that should be there and is not** was archived, not lost: read `~/data/sidechain/README.md`
§ Backups (Lamin nightly, MEDIA archive, the SSD) before rebuilding or re-downloading anything.

## House rules

- **This repo holds the pipeline:** what a build, a fold score, an ingest, the box, Lamin or the site runs, and its
  tests. Research code (a screen, gate, probe, diagnostic, board analysis, or a feature builder no build reads) does
  not start here; it moves here only when a build or fold score starts reading it.
- **Nothing is trusted until the local mirror scores it:** `sidechain.eval.mirror2026` (one held-out line, cell-eval2's
  six metrics) and `sidechain.eval.loco` (the rotation over lines); a pdex/CPU bundle scores on the Mac, gpudge and
  full-corpus builds need a GPU box. `sidechain.eval.local_mirror` is the frozen 2025 backtest and scores nothing for
  2026.
- **Sparse only** (`edge_index`/COO); never a dense gene × gene matrix (18,533² is ~1.4 GB).
- **Public text is the least that says what the code does.** A module or script docstring: one line of what, a usage
  line, what it reads and writes, and one pointer, `Notes: private/code-notes/<this file's path>.md`. A function: one
  line, or none. A comment: only where the code cannot say it. A help string: what the flag does, and its unit. No
  why, finding, measured number, score, entry name or setting, context identity, competitor method, review round or
  plan in a docstring, comment, help string, error message, test fixture or doc in this repo.
- **A new corpus** is a `configs/datasets.yaml` block behind the stream gate (`sidechain.ingest.fetch`). **Priors**
  subclass the contract in `src/sidechain/priors/base.py`; registry blocks declare `kind: edge | node_feature`; build
  edges with `PriorSource.to_edge_index`, never two `to_positions` calls.
- **Two gene-ID spaces**, declared per year in `challenges/<year>/config.yaml`: 2025 carries Ensembl in
  `var['gene_id']` (18,080 genes); 2026 is symbols only, `var` empty, 18,533 genes. `loaders.gene_index` is strict and
  raises rather than falling back to `var_names` — that fallback once matched no prior and returned zero edges
  silently.
- **`gene_names.csv` differs by year** (2025 no header, 2026 has one): read names from the h5ad;
  `sidechain.data.profile` marks the read that matches.
- **Read the methods for the control definition; never infer it from the labels.** `control_label` is a list: Feng
  2026's arm is `[NonTarget, unassigned]`, 499,998 cells, not 48.
- **Check the minimum before writing "every"** (median 48, minimum 22).
- **A–F name the challenge's six anonymized contexts and nothing else:** never letter a cell line, fold, bundle or
  corpus. Name it — the fold by its directory (`loco_hek293t_ch272`), the line by its name plus the study where that
  is ambiguous (Jurkat from Nadig 2024). A two-arm comparison (`arm A` vs `arm B`) letters a contrast, not a context,
  and is fine. D/E/F were released on Oct 22, so a letter in a record written before then means A, B or C.
- **Standings numbers are generated** (`scripts/standings.py`): never hand-edit the README table,
  `site/data/submissions.json` or `site/assets/data/field.json`.

## Committing

- **Stage by explicit path.** Never `git add -A`, `.`, `-u` or `commit -a`: other sessions' uncommitted work is always
  in this tree (2026-08-22: one `add -A` swept 44 files of another session's work into a pushed commit). If
  `git status` shows changes you did not make, leave them alone and say so.
- **Never rewrite pushed history.** Commit ids are cited across the project's records; a mistake is fixed in a new
  commit.
- **Message:** one line, ``<scope>: <what the change does> (T<n>, session `<id>`)``. Never a finding, a measured number
  or score, an entry's settings, a context's identity, a competitor's method, a plan, or the name of an internal
  document.

## Env

`uv sync` installs everything incl. `dev`; Python 3.11, capped below 3.13, so the Mac and a box resolve alike. PyPI:
`arc-state`, `cell-eval2` (the 2026 scorer), `cell-eval` (2025), `cell-load`, `pdex`. The hosted lamindb
`saberhq/sidechain` backs up `derived/`, `cache/` and bundles, never `external/`: register every derived artifact in
the session that creates it (`scripts/lamin_register.py` ⇄ `lamin_pull.py`, key = path under `~/data/sidechain/`).

## Where to look, by the kind of ask

| when the ask is … | read first | then |
|---|---|---|
| code: a module, a test, a config | this file + the module you are in | — |
| score a prediction, build a bundle | `src/sidechain/eval/mirror2026.py` (docstring: `bundle` → `score`; Mac vs box) | `scripts/boxkit/README.md` |
| a challenge year's data, packaging, `vcc submit` | `challenges/<year>/CLAUDE.md` | — |
| QC on our h5ads, the controls bundle | `challenges/<year>/CLAUDE.md` · `sidechain.data.profile` · `sidechain.ingest.checks` | — |
| a generic single-cell method: QC, scVI, nf-core, instrument data | `/bio-research:single-cell-rna-qc`, `:scvi-tools`, `:nextflow-development`, `:instrument-data-to-allotrope` | — |
| a data download, disk, a new corpus, a licence | this file § Data · `configs/datasets.yaml` | — |
| the website | `site/README.md` | — |
| an agent's role | `agents/<role>.md` (researcher · verifier · analyst · critic) | its wrapper in `.claude/agents/`, which pins its model |
