# CLAUDE.md — vcc2026 (live track)

The competition we are entering. The spec landed **2026-08-20**; `config.yaml` beside this file
carries the values, this file carries what will bite you. Verbatim sources:
`private/research/reading/vcc-2026-spec.md`; the argument: `private/reports/05_vcc2026_kickoff_audit.md`.

## The shape

- **Zero-shot, multi-context.** Six anonymized cell lines: A/B/C now (validation, live board),
  D/E/F on **Oct 22** (final). **Only D/E/F decide the prizes.** Deadline Nov 5, 23:59 UTC.
- Per context you get **18,400 non-targeting control cells** (46 guides × 400, guide in
  `obs['ntc_id']`) and a list of **300 genes**. You return **400 cells per gene**, raw counts,
  on a fixed **18,533-gene** axis — 360,000 cells per file, all three contexts in one file.
- **No training data from Arc.** The corpus is the full 2025 H1 release (all 300
  perturbations: Training + Validation + Test) plus `configs/datasets.yaml`.
- Scored by **cell-eval2** (`vcc2026` preset, installed): six metrics, each scaled
  0 = context-mean baseline → 1 = split-half replicate; overall = flat mean over 6 × 3.

## Getting the data

`~/data/sidechain/vcc2026/`. Fetched with the **vcc CLI** (`uv tool install vcc-cli`; the
command is `vcc`), not through `sidechain.ingest` — the CLI resumes and crc32c-verifies, and
the bundle is gated behind an account:

```bash
# once: mint a key at virtualcellchallenge.org/app/credentials (shown once), then in YOUR terminal
printf '%s' "vcc_pat_..." | vcc login --token-stdin     # never paste the key into a chat
vcc whoami                                               # must say "ready to submit"

vcc datasets list                                        # exact bundle size
vcc datasets download controls -d ~/data/sidechain/vcc2026
unzip -o -j ~/data/sidechain/vcc2026/vcc_2026_controls.zip -d ~/data/sidechain/vcc2026
uv run python -m sidechain.data.profile --challenge-config challenges/vcc2026/config.yaml
```

The agent skill (`vcc skill install` → `~/.claude/skills/vcc/`) drives download → prep →
submit; re-run it after every `uv tool upgrade vcc-cli`.

## Things that will bite you

1. **Context labels are opaque and load-bearing.** A/B/C name held-out datasets, not cell
   lines. Never reorder, relabel or regenerate them; keep the label attached from the moment a
   control file is read. A swap scores like a bad model and nothing says "you swapped them" —
   `vcc prep` cannot detect it.
2. **D/E/F ≠ A/B/C.** The final bundle carries its own labels and its own `pert_counts.csv`;
   carrying validation labels into a final submission is rejected.
3. **The gene axis is 18,533, not H1's 18,080.** Map the H1 corpus (and every ingested line)
   onto `gene_names.csv` by symbol; ≥ 453 genes have no H1 signal. Do **not** assume a
   `var['gene_id']` Ensembl column exists — verify on the profiled bundle.
4. **Raw integer counts, not log1p.** Non-negative, whole, finite, ≤ 1,000,000 per cell;
   exactly 400 cells per perturbation; **no `non-targeting` rows**; `target_gene` holds gene
   symbols (`ADNP`, not `ADNP-1`); a `context` column; ≤ 4.75e9 stored entries (a dense matrix
   is 1.40× over by itself — store CSR without explicit zeros). The 2025 emitter's CP10k-log1p
   cells are rejected outright.
5. **Depth differs.** The truth is downsampled to a median ~20,000 UMI/cell (H1 was 44,798).
   Emit at the target context's depth.
6. **Memory.** `vcc prep` reads the whole file with `anndata.read_h5ad` (not backed) and holds
   a second copy. Context-mean density (~11,800 nnz/cell) is 4.25e9 nonzeros ≈ 48 GiB
   resident; realistic sampled cells (~6k nnz) ≈ 16 GiB plus the file copy. **A 16 GB Mac
   cannot package a submission.** Write the h5ad out of core (h5py, per context) and run
   `vcc prep` / `vcc submit` on a ≥ 64 GB machine.
7. **PDS is cosine** on pseudobulk-sum deltas with all 300 panel targets removed: invariant to
   each predicted effect's magnitude. Direction is what scores; size pays only on `nmae`/`mse`.
8. **The four DE metrics are computed on the cells you emit** (Wilcoxon vs the real controls,
   > 5 CPM in controls, BH per perturbation). The dispersion of emitted cells sets how many
   genes you "call", and `fid`/`jac` charge under-calling. Emission is a modelling choice.
9. **Two submissions per team per day** (UTC), one in flight (409 otherwise). Only submissions
   that reach scoring count. Validation and final scores are not comparable.
10. `gene_names.csv` is **headerless** — the same `pd.read_csv` off-by-one trap as 2025.

## Packaging and submitting

```bash
vcc prep pred.h5ad -g gene_names.csv --perts pert_counts.csv -o pred.vcc --dry-run   # validate
vcc prep pred.h5ad -g gene_names.csv --perts pert_counts.csv -o pred.vcc
vcc submit pred.vcc -m "rung 1' pooled delta transfer" --wait                         # prints the six
vcc status <entry-id>
```

`--perts` has no short form in `prep` (`-p` there means `--pert-col`). A `vcc sample` file is
noise by design — pipeline testing only.

## Watching the board

```bash
uv run python -m sidechain.eval.leaderboard            # prints top 20, saves JSON
```

Snapshots land in `~/data/sidechain/vcc2026/leaderboards/`. Day 0 (2026-08-20): six entries,
leader 0.134 with a cross-line pooled delta-transfer model; `mse` was 0 for everyone.

## Running the model

Not yet — the 2026 mirror (cell-eval2 + our own anchors + leave-one-context-out) is
`private/TODO.md` Now #2. No 2026 number is trusted before it exists.
