# CLAUDE.md — vcc2026 (live track)

The competition we are entering. The spec landed **2026-08-20**; `config.yaml` beside this file
carries the values, this file carries what will bite you. Verbatim sources:
`private/research/reading/vcc-2026-spec.md`; the argument: `private/reports/05_vcc2026_kickoff_audit.md`.

## The shape

- **Zero-shot, multi-context.** Six anonymized cell lines: A/B/C now (validation, live board),
  D/E/F on **Oct 22** (final). **Only D/E/F decide the prizes.** Deadline Nov 5, 23:59 UTC.
- Per context you get **18,400 non-targeting control cells** (46 guides × 400, guide in
  `obs['ntc_id']`) and a list of **300 genes to knock down**. For each knockdown you return
  **400 predicted cells, each a full 18,533-gene transcriptome**, raw counts — 360,000 cells
  per file, all three contexts in one file. Scoring is genome-wide (minus the knocked-down
  gene itself); the 300 is the number of perturbations, not the number of genes scored.
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
unzip -o ~/data/sidechain/vcc2026/vcc_2026_controls.zip -d ~/data/sidechain/vcc2026   # files sit at the zip root
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
   onto `gene_names.csv` by symbol: H1 shares 18,077, 456 genes have no H1 signal. **There is
   no `var['gene_id']` — `var` is empty, symbols only** (measured 2026-08-20). The scPerturb
   lines are pre-filtered to ~8–10k genes. `loaders.gene_index(adata, 'ensembl_gene_id')`
   will raise on these files, correctly; a symbol-keyed path is needed.
4. **The 18,533 axis is CURATED, and ~30 % of a real cell never appears on it** (measured
   2026-08-29). `gene_names.csv` is byte-identical, in order, to `context_A.h5ad`'s `var` index,
   so this is the axis itself and not a file-reading mistake. Entire families are absent: **0 of
   99 cytoplasmic ribosomal proteins (RPL/RPS), 0 of 81 mitochondrial ribosomal (MRPL/MRPS), 0
   of 21 HLA**, plus `GAPDH`, `EEF1A1`, `MALAT1`, `NEAT1`, `XIST` — while `ACTB`, `TP53`, `MYC`
   and 12 of 13 `MT-` genes stay. Summed over the 10x GRCh38-2020-A reference in five unrelated
   experiments (K562, fibroblast, Kuramochi, NK, Caco2), the off-axis genes carry **29.5–32.5 %
   of every cell's UMIs**, and they are the top-expressed genes in each one.
   Two consequences. **Normalise to the library YOU measured, never to the on-axis subtotal** —
   the off-axis share varies by cell type, so an on-axis CPM silently rescales each context by
   how much of its transcriptome the axis happens to cover. And when reading any "fraction of
   the transcriptome" claim about this challenge, remember the scored axis is a curated ~70 %
   of it; the six metrics never see the most abundant third. What Arc's rule was is not stated
   anywhere we have found — only the resulting list is, and that list is what the loaders use.
5. **Raw integer counts, not log1p.** Non-negative, whole, finite, ≤ 1,000,000 per cell;
   exactly 400 cells per perturbation; **no `non-targeting` rows**; `target_gene` holds gene
   symbols (`ADNP`, not `ADNP-1`); a `context` column; ≤ 4.75e9 stored entries (a dense matrix
   is 1.40× over by itself — store CSR without explicit zeros). The 2025 emitter's CP10k-log1p
   cells are rejected outright.
6. **Depth differs.** The truth is downsampled to a median ~20,000 UMI/cell (H1 was 44,798).
   Emit at the target context's depth.
7. **Memory.** `vcc prep` reads the whole file with `anndata.read_h5ad` (not backed) and holds
   a second copy. Cells at the controls' own density (median 6,006 detected genes) make a
   360,000-cell file of 2.16e9 nonzeros ≈ 24 GiB, ~50 GiB peak in `prep`; context-mean density
   (~11,800 nnz/cell) is 4.25e9 ≈ 48 GiB resident. **A 16 GB Mac cannot package a
   submission.** Write the h5ad out of core (h5py, per context) and run `vcc prep` /
   `vcc submit` on a 64 GB machine at minimum, 128 GB comfortably.
8. **PDS is cosine** on pseudobulk-sum deltas with all 300 panel targets removed: invariant to
   each predicted effect's magnitude. Direction is what scores; size pays only on `nmae`/`mse`.
9. **The four DE metrics are computed on the cells you emit** (Wilcoxon vs the real controls,
   > 5 CPM in controls, BH per perturbation). The dispersion of emitted cells sets how many
   genes you "call", and `fid`/`jac` charge under-calling. Emission is a modelling choice.
10. **Two submissions per team per day** (UTC), one in flight (409 otherwise). Only submissions
   that reach scoring count. Validation and final scores are not comparable.
11. **`gene_names.csv` HAS a header row this year (`gene_name`)** — the opposite of 2025. A
    `header=None` read yields 18,534 "genes" with the literal `gene_name` first. `vcc prep`
    strips known headers either way; our own readers must check the row count against the h5ad
    (`sidechain.data.profile` prints both reads and marks the right one).
12. **Target coverage per corpus is thin — but the shipped pool is not.** Of the 300 targets, H1
    covers 25, the genome-wide K562 file 272, and the four essential-gene panels (K562-essential,
    RPE1, Jurkat, HepG2) **zero**. `private/reports/06_vcc2026_controls_qc.md` §5.
    **The "28 targets are in nothing we hold" clause is pre-X-Atlas and stale (checked
    2026-09-01):** `cache/vcc2026/foldsub/{hct116,hek293t}_full_panel300.npz` each carry
    **300/300** of `pert_counts.csv`, and every build since SER-2 records `pool 300/300,
    fallback 0`. So the generic-mean-shift fallback fires on **no** target today. Do not build an
    argument on the 28 without re-measuring; the per-corpus numbers above are still correct.

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

## Building and submitting a prediction

The pipeline is `sidechain.submit` (public). It never holds the 360,000-cell matrix in memory:
blocks of 400 cells are emitted per (context, perturbation), appended to an h5ad on disk with
h5py in exactly the layout `vcc prep` produces, checked against the contract as they go, then
packed into the `.vcc` container (`tar` of `pred.h5ad.zst`). `vcc submit x.vcc` validates the
container only and the server reads the file on a 128 GB machine, so **the Mac can submit**.

```bash
# 1. per-perturbation pseudobulks of the source corpora (one streaming pass each; cached)
uv run python -m sidechain.data.stream_pseudobulk ~/data/sidechain/vcc2025/adata_{Training,Validation,Test}.h5ad \
    --label-col target_gene --control non-targeting --control-once --out ~/data/sidechain/cache/vcc2026/h1_pseudobulk.npz
uv run python -m sidechain.data.stream_pseudobulk ~/data/sidechain/external/zenodo-13350497/ReplogleWeissman2022_K562_gwps.h5ad \
    --label-col perturbation --keep ~/data/sidechain/vcc2026/pert_counts.csv --control control \
    --out ~/data/sidechain/cache/vcc2026/k562_gwps_targets_pseudobulk.npz

# 2. build (emitter x dispersion), ~15 min; writes <out>.h5ad and <out>.vcc
uv run python -m sidechain.submit.build --emitter delta-transfer --dispersion even \
    --h1-cache ~/data/sidechain/cache/vcc2026/h1_pseudobulk.npz \
    --gwps-cache ~/data/sidechain/cache/vcc2026/k562_gwps_targets_pseudobulk.npz \
    --out ~/data/sidechain/vcc2026/submissions/<name>

# 3. smoke-test the layout on a 3-perturbation panel before any full build
uv run python -m sidechain.submit.build --emitter control-null --limit-perts 3 --out .../_smoke
vcc prep .../_smoke.h5ad -g gene_names.csv --perts .../_smoke.pert_counts.csv --dry-run

# 4. submit (2 per day, UTC)
vcc submit ~/data/sidechain/vcc2026/submissions/<name>.vcc -m "<model name>" --wait
```

Two modelling choices live in the emitter (`sidechain.models.count_emitters`), not in plumbing:
**dispersion** (`even` = minimum-variance cells, which make the DE test call the whole
universe and so satisfy `fid`'s coverage term; `poisson` = realistic cells, which call only a
few hundred genes) and **shrinkage** of transferred log2FCs (gene-wise positive-part, on by
default; `--no-shrink` to compare). Score a choice locally before spending a slot — the **2026 mirror** scores a held-out line
on Arc's own 0 = mean-response → 1 = replicate scale using cell-eval2's bundle machinery:

```bash
# once per held-out line: extract a panel (controls relabelled `non-targeting` -- the competition
# rule hashes the control label, so this is what makes the bundle rule-exact; the DE backend is
# not part of the hash, the GPU box only makes it ~40x faster), then build its bundle
uv run python -m sidechain.data.stream_subset <line>.h5ad --label-col perturbation --keep panel.csv \
    --control control --relabel-control non-targeting --max-per-label 400 --max-control 10000 --out real.h5ad
uv run python -m sidechain.eval.mirror2026 bundle --real real.h5ad --out ~/data/sidechain/runs/mirror/<line> \
    --pert-col perturbation --control non-targeting
# per model: predict the held-out line from the OTHER lines' pseudobulks, score on the bundle
uv run python -m sidechain.eval.loco --real real.h5ad --pert-col perturbation --control non-targeting \
    --source <other>_all_pseudobulk.npz:control ... --bundle ~/data/sidechain/runs/mirror/<line>/bundle \
    --out ~/data/sidechain/runs/mirror/<line>/<arm> --dispersion even
```

Mirror numbers are relative (arm A vs arm B on one bundle), not forecasts of the board.

**The GPU box** (NVIDIA Brev, `scripts/brev_bootstrap.sh`): `uv sync --extra gpu` installs
cell-eval2's CUDA kernels + `gpudge` (GPU differential expression). Two traps, both paid for:
`uv run` re-syncs the env to `uv.lock` before every command, so torch must resolve from the
lock (pyproject pins the CUDA-12.6 index for Linux — the default index serves a CUDA-13 build
the 570-series driver cannot run); and a venv that has been hand-edited with `uv pip` can end
up with dist-info but no files — `rm -rf .venv && uv sync --extra gpu` is the fix, not more
`uv pip`. The login user is `shadeform`, not uid 1000.
