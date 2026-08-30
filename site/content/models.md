---
title: Models
description: Every model Sidechain has put on the Virtual Cell Challenge 2026 board — what each one is, what its name means, and what it borrows.
---

Every submission's board card points here. The name grammar in one line: a **series tag**
(`SER` — cross-line delta transfer, the family every entry so far belongs to), a **model
number** (new sources or structure), and lowercase **knob letters**, each marking exactly one
setting moved off the series baseline — so `SER-3fn` reads as "SER model 3, with knobs `f`
and `n` on". Scores live in the [standings table](../#standings) — this page is what the
numbers are attached to.

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
