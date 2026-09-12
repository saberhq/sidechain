---
title: Models
description: Every model Sidechain has put on the Virtual Cell Challenge 2026 board — what each one is, what its name means, and what it borrows.
---

Every submission's board card points here. The name grammar in one line: a **series tag**
(`SER` — cross-line delta transfer; `PHE` — the deep generative models), a **model number**
(new sources or structure), and lowercase **knob letters**, each marking exactly one setting
moved off the series baseline — so `SER-3fn` reads as "SER model 3, with knobs `f` and `n`
on". Scores live in the [standings table](../#standings) — this page is what the numbers are
attached to.

An entry marked **probe** was submitted to answer a question rather than to climb the table.
What an entry is for is decided when it goes in, before its score comes back.

## PHE-2 — submitted 2026-09-07 · probe

**The first entry from a different family, and the first sent in expecting to fail.** Every
`SER` model borrows a measured knockdown effect from a screen that saw that gene silenced
somewhere else. PHE-2 borrows nothing. It is a neural network, trained on two cancer cell
lines, that reads an unseen line's untouched control cells and predicts directly what each of
the 300 knockdowns would do to them. Its parent, PHE-1, never left the local benchmark.

It scored **−0.9807**, and that was the point. We had a benchmark of our own telling us where
this model stood. What we had no measurement of was how our benchmark's numbers relate to
Arc's — and an entry we already expected to place badly is the cheap way to find out. It is
also the only way: the official cell contexts are not something a local benchmark can imitate.

Where that number comes from is worth saying, because −0.98 sounds uniform and is not. The
overall score is the plain average of six metrics. Five of them landed near zero, which is
roughly what "no better than predicting the average cell" earns. The sixth asks how close our
fold-changes are to the real ones, and it bottoms out at −6; PHE-2 put it there. One collapsed
metric averaged with five flat ones is −0.98. The cells it emits carry about half the number
of expressed genes a real cell does, which is the first place to look.

## SER-5acfnt — submitted 2026-09-01

**a = amplified transfer · c = coverage-tiered pooling weights · f = floored source weights ·
n = no shrinkage · t = per-source transfer-error floor.** SER-4afn with two changes to how the
four sources vote on a gene. `c` lets a source's weight for one gene know how much evidence
stands behind *that gene* in *that* screen, rather than only how large the screen was overall.
`t` adds each source's measured transfer error to its variance before the vote is counted, so
a source cannot claim more certainty than its record against a held-out line supports.

It came back a hair below SER-4afn — no improvement. `t` has since been retired: shuffling
which measured error belonged to which source scored as well as assigning them correctly, so
the knob was flattening the pool rather than carrying information about it. `c` was not
tested on its own here and stays.

## SER-4afn — submitted 2026-08-31

**a = amplified transfer · f = floored source weights · n = no shrinkage.** SER-3afn with the
amplification stepped back from its local-benchmark optimum to the value the measured
knockdown depths of the actual submission pool predict. Model 4 rather than a new letter: the
name records *which* dials are on, never their values, so a value change mints a new number.

## SER-3afn — submitted 2026-08-30

**a = amplified transfer · f = floored source weights · n = no shrinkage.** SER-3fn with the
transferred effects scaled up to reference strength. The screens we borrow from silenced
their targets only partially — some reached less than half a full knockdown — so the effects
they measured are systematically smaller than what a reference-strength knockdown produces.
Measuring each screen's realized knockdown from its own on-target rows predicted the right
correction almost exactly before it was scored.

## SER-3afgn — submitted 2026-08-30

**a = amplified transfer · f = floored source weights · g = abundance-ratio transfer
exponent · n = no shrinkage.** SER-3afn plus one deliberate probe: instead of transferring
each gene's *fold change* unchanged, bend the transferred effect toward the new line's own
resting abundance of that gene. The local benchmark said the plain fold-change rule is
already the optimum; this entry tested that verdict on the real board — and confirmed it.

## SER-3fn — submitted 2026-08-27

**f = floored source weights · n = no shrinkage.** SER-3n with one change: when sources are
pooled per gene, each one's vote is now bounded by how well it could possibly have measured
that gene given its cell counts — a source that happened to observe zero spread no longer
counts as infinitely certain, and a source that saw a perturbation only once abstains
entirely. Same four sources, re-anchored on the new line's resting state.

## SER-3n — submitted 2026-08-27

**n = no shrinkage.** SER-2's pool with the two genome-wide screens read in full — every gene
they measured rather than the challenge panel alone — so the per-gene votes ride on a much
wider expression axis.

## SER-2 — submitted 2026-08-24

Named before the knob letters existed (shrinkage is off, as in SER-1n). The pool grows from
two sources to four: K562 genome-wide and H1, plus the challenge-panel slice of two
genome-wide CRISPRi screens in colon and kidney lines — the first entry to cover all 300
target genes with a measured effect instead of a generic fallback.

## SER-1n — submitted 2026-08-22

**n = no shrinkage.** SER-1 with the gene-wise shrinkage of transferred effects switched
off: the many small, noisy per-gene effects that shrinkage silenced turn out to carry the
direction signal the perturbation-matching metric reads.

## SER-1p — submitted 2026-08-21

**p = Poisson cells.** SER-1 with the emitted cells drawn with independent Poisson noise
instead of minimum-variance spread — a one-knob experiment on how cell-level dispersion
prices into the DE metrics. It answered its question; `even` stayed.

## SER-1 — submitted 2026-08-21

The first entry, and the family's baseline: each gene's knockdown effect borrowed from the
K562 genome-wide screen where it was measured (272 of 300 targets) and from H1 (25), pooled
per gene by measurement confidence, re-anchored on each new line's resting control profile,
and emitted as 400 cells per perturbation at the line's own depth.
