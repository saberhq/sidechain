---
title: Models
description: Every model Sidechain has put on the Virtual Cell Challenge 2026 board — what each one is, what its name means, and what it borrows.
---

Every submission's board card points here. The name grammar in one line: a **series tag**
(`SER` — cross-line delta transfer; `PHE` — the deep generative models), a **model number**
(new sources or structure; since SER-4afn, every new entry), and lowercase **knob letters**, each marking exactly one setting
moved off the series baseline — so `SER-3fn` reads as "SER model 3, with knobs `f` and `n`
on". Scores live in the [standings table](../#standings) and, member by member, in the table
below — the rest of this page is what the numbers are attached to.

## Every entry on one board

The leaderboard's table with Sidechain's entries alone on it, by overall score, calibration runs
included. Each of the six members shows the scaled score the board ranks on over the raw value,
coloured as the leaderboard colours them.

{{< board >}}

## SER-17aefhkrsw — submitted 2026-10-08

**a = amplified transfer · e = emission dial at λ 0.5 · f = floored source weights · h = a chosen
list of genes made callable · k = neighbour genes blended in · r = cells built from real control
cells · s = shrinkage rule moved · w = summed profile on the pooled control profile.**
SER-16aefhkrsw with a longer list: for each knockdown, the 1,500 genes whose predicted change
stands out most have their cell-to-cell variation narrowed, in place of 300.

It scored 0.1657 against SER-14aefksw's 0.1626 and SER-16aefhkrsw's 0.1423. Against
SER-16aefhkrsw the gain is in the score that counts the genes the test calls in the right
direction; the overlap between the genes called and the genes that really changed rose a little,
the test reached slightly fewer of the genes that really changed, and the other three scores
held. Against SER-14aefksw the summed-profile and expression-accuracy scores are higher and the
test reached more of the genes that really changed; the direction score is lower by about as much
as that last gain, and the overlap is a little lower.

## SER-16aefhkrsw — submitted 2026-10-08 · calibration run

**a = amplified transfer · e = emission dial at λ 0.5 · f = floored source weights · h = a chosen
list of genes made callable · k = neighbour genes blended in · r = cells built from real control
cells · s = shrinkage rule moved · w = summed profile on the pooled control profile.**
SER-15aefkrsw with one change. There the 400 cells of a knockdown kept the cell-to-cell variation
real cells have, and the differential-expression test called only a handful of genes in a typical
knockdown. Here, for each knockdown, the 300 genes whose predicted change stands out most have
that variation narrowed (`h`), so the test calls most of them; each gene's total count is
unchanged. This submission is a calibration run, to measure what a short list of called genes
scores on the challenge's cell contexts.

It scored 0.1423 against its parent SER-15aefkrsw's −0.0039 and SER-14aefksw's 0.1626. The score
that counts the genes the test calls in the right direction won back most of what SER-15aefkrsw
had lost on it and stayed below SER-14aefksw's. The overlap between the genes called and the
genes that really changed fell below both. The test reached slightly fewer of the genes that
really changed than in SER-15aefkrsw, and the other three scores held.

## SER-15aefkrsw — submitted 2026-10-07 · calibration run

**a = amplified transfer · e = emission dial at λ 0.5 · f = floored source weights · k = neighbour
genes blended in · r = cells built from real control cells · s = shrinkage rule moved · w = summed
profile on the pooled control profile.** SER-14aefksw with one change, in how each knockdown's 400
cells are written. In SER-14aefksw all 400 were drawn around one profile: the context's control
profile, moved by the predicted change. Here each one starts as a real control cell of that
context and has its counts moved to the predicted change (`r`), so the 400 keep the cell-to-cell
variation real cells have. The emission dial (`e`) stays in the name and shapes no cell here. This
submission is a calibration run, to measure how much of a gain on our own held-out tests carries
over to the challenge's cell contexts.

It scored −0.0039 against its parent SER-14aefksw's 0.1626. One of the six scores is more than
the whole fall: the one that counts the genes the differential-expression test calls in the right
direction. The other five rose or held. Locally, on four held-out test sets from three cell
lines, it had scored above SER-14aefksw's recipe on every one.

## SER-14aefksw — submitted 2026-10-03

**a = amplified transfer · e = emission dial at λ 0.5 · f = floored source weights · k = neighbour
genes blended in · s = shrinkage rule moved · w = summed profile on the pooled control profile.**
SER-12aefkw with the shrinkage rule changed. There, each borrowed gene change was shrunk by its own
measurement noise in every source cell line, and a change no larger than its noise was dropped.
Here the changes are first combined across the source cell lines, and each one is then shrunk by
how believable it is against all of that knockdown's genes together, so a weak change ends small
rather than zero.

The best entry so far — 0.1626 against its parent SER-12aefkw's 0.1438. The predicted profiles sit
closer to the control cells, and the expression-accuracy score left zero for the first time; that
score is more than the whole gain. The summed-profile score rose a little, fold-change accuracy
fell, and the differential-expression test reached slightly fewer of the genes that really changed.
Locally the rule had read as a trade between those same scores.

## SER-13aefknw — submitted 2026-10-02

**a = amplified transfer · e = emission dial at λ 0.5 · f = floored source weights · k = neighbour
genes blended in · n = no shrinkage · w = summed profile on the pooled control profile.**
SER-11abefknw with one setting moved back to the series baseline: the summed profile of each
knockdown's 400 cells carries the borrowed response at the same strength as the cells themselves
(`b` off). Sent after SER-12aefkw so that the two together read the shrinkage setting on its own.

Level with its parent — 0.1360 against SER-11abefknw's 0.1360. The summed-profile score rose a
little and the differential-expression test reached slightly fewer of the genes that really
changed. Locally the setting had read as a small gain on three cell lines.

## SER-12aefkw — submitted 2026-10-02

**a = amplified transfer · e = emission dial at λ 0.5 · f = floored source weights · k = neighbour
genes blended in · w = summed profile on the pooled control profile.** SER-11abefknw with two
settings moved back to the series baseline. The summed profile of each knockdown's 400 cells now
carries the borrowed response at the same strength as the cells themselves (`b` off), and each
borrowed gene change is shrunk toward zero by its own measurement noise, so a change no larger
than its noise is dropped (`n` off).

The best entry so far — 0.1438 against its parent SER-11abefknw's 0.1360. The summed-profile score
and fold-change accuracy rose; the differential-expression test reached fewer of the genes that
really changed. Locally the two settings had read as a small gain for one strength and a trade for
shrinkage.

## SER-11abefknw — submitted 2026-09-30

**a = amplified transfer · b = summed-profile strength set apart · e = emission dial at λ 0.5 ·
f = floored source weights · k = neighbour genes blended in · n = no shrinkage · w = summed profile
on the pooled control profile.** SER-10abefnw with the change SER-9abefkn made. Each knockdown's
borrowed response is nudged toward the average borrowed response of the 25 genes nearest to it in
the same STRING protein-interaction map, chosen from the same fixed list of 842 knockdowns, with a
slightly weaker nudge. As there, the nudge turns the part of the response specific to this knockdown
and keeps its size.

The best entry so far — 0.1360 against its parent SER-10abefnw's 0.1353. The differential-expression
test reached more of the genes that really changed; fold-change accuracy and the summed-profile score
each fell a little. Locally, on five held-out test sets from three cell lines, the gain was about ten
times as large.

## SER-10abefnw — submitted 2026-09-30

**a = amplified transfer · b = summed-profile strength set apart · e = emission dial at λ 0.5 ·
f = floored source weights · n = no shrinkage · w = summed profile on the pooled control profile.**
SER-7abefn with one change. Each knockdown's predicted cells are built on a profile of the
context's control cells. Until now that profile was the average of each control cell's own
composition, every cell counting once. The scorer, though, compares summed profiles, where a deep
cell counts more than a shallow one. Where a gene's share of a cell's counts depends on how deep
the cell is, the two profiles differ, and every prediction carried that difference. `w` builds the
summed profile of the predicted cells on the pooled control profile instead, the one the scorer
uses, and leaves each cell's own profile where it was.

The best entry at the time — 0.1353 against its parent SER-7abefn's 0.1131, nearly all of it in the
summed-profile score, which rose from 0.485 to 0.612; the cell-by-cell scores held. Locally, on
five held-out test sets from three cell lines, the gain was about the same size.

## SER-9abefkn — submitted 2026-09-29

**a = amplified transfer · b = summed-profile strength set apart · e = emission dial at λ 0.5 ·
f = floored source weights · k = neighbour genes blended in · n = no shrinkage.** SER-7abefn
with one change of the kind SER-8abefkn made, with a different map, fewer neighbours and a
stronger nudge. Each knockdown's borrowed response is nudged toward the average borrowed response
of the 25 genes nearest to it, chosen from the same fixed list of 842 knockdowns, in an embedding
of the STRING protein-interaction network: a map that places genes whose proteins work together
close to each other. As there, the nudge turns the part of the response specific to this
knockdown and keeps its size.

The best entry at the time — 0.1144 against its parent SER-7abefn's 0.1131. The summed-profile score
rose most, the differential-expression test reached more of the genes that really changed, and
fold-change accuracy fell by about half of what those two gained. Locally, on the five held-out
test sets from three cell lines its settings were picked on, the gain was about six times as
large.

## SER-8abefkn — submitted 2026-09-29 · calibration run

**a = amplified transfer · b = summed-profile strength set apart · e = emission dial at λ 0.5 ·
f = floored source weights · k = neighbour genes blended in · n = no shrinkage.** SER-7abefn
with one change. Each knockdown's borrowed response is nudged toward the average borrowed
response of the 50 genes nearest to it in Tahoe-x1's gene embedding, a map of genes learned
from single-cell data, chosen from a fixed list of 842 knockdowns. The nudge turns the part of
the response specific to this knockdown a little, and keeps its size. Sent to measure how much
of a gain on our own held-out tests carries over to the challenge's cell contexts.

It scored level with SER-7abefn, 0.1132 against 0.1131. The summed-profile score rose, and two
of the cell-by-cell scores together fell by most of that gain. On five held-out test sets from
three cell lines, the one that fell most had risen on every set.

## SER-7abefn — submitted 2026-09-17

**a = amplified transfer · b = summed-profile strength set apart · e = emission dial at λ 0.5 ·
f = floored source weights · n = no shrinkage.** SER-6aefn with one change. The scorer reads
the 400 predicted cells of a knockdown twice: once added up into one profile, once cell by
cell. Until now both readings carried the borrowed response at one strength, 1.35. `b` lets
the summed profile carry it at 1.5 while each cell still carries it at 1.35, by moving counts
between deeper and shallower cells without changing any cell's depth.

The best entry at the time — 0.1131 against SER-6aefn's 0.1091. The summed-profile score rose and
the cell-by-cell scores held, the pattern the local sweep showed on six held-out test sets
from three cell lines before it was submitted. The gain on the board is smaller than it was
locally.

## SER-6aefn — submitted 2026-09-12

**a = amplified transfer · e = emission dial at λ 0.5 · f = floored source weights · n = no
shrinkage.** SER-4afn with one change, and the first new knob since the source weights: `e`
sets how much cell-to-cell noise the 400 predicted cells carry. At λ 0.5 they are drawn with
half the spread that Poisson counting would give, instead of every cell sitting at the same
depth. That lets the differential-expression test call the right genes a little more often,
at a small cost in fold-change accuracy.

It scored 0.1091 against SER-4afn's 0.1078, and the gain landed where the local
sweep said it would: more of the genes that really changed are reached, with slightly worse
fold-change accuracy paying for it. The dial was swept on six held-out cell lines before it
was submitted, and it moved the same way on all six.

## PHE-2 — submitted 2026-09-07 · calibration run

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
