---
title: Models
description: Every model Sidechain has put on the Virtual Cell Challenge 2026 board — the idea each one tested, what it scored, and what its name means.
---

All but one come from the SER series, which predicts what silencing a gene (a knockdown) does to a cell line it has never
seen by borrowing what silencing the same gene did in large public screens of other cell lines, the sources. The one
exception, PHE-2, is a neural network that predicts the knockdowns directly. Every entry so far was scored on the
challenge's three validation cell lines; the three lines that decide the final ranking arrive on October 22.

A name reads in three parts. In `SER-28abefhkrswx`, `SER` is the series, `28` the model number, and each lowercase letter
a knob moved off the series' plain recipe, ten of them here. A letter keeps one meaning in every series, and it says that a
knob is on, not where it is set: setting the same knob differently takes a new model number.

Each section says what its model set out to test and how it scored, and leaves out where its knobs were set: how strong,
how many, how far. What each knob does is explained once, at the end of the page, under [The knobs](#the-knobs), and what
each of the board's six scores reads is under [The six scores](#the-six-scores). "Our own held-out tests" are public
screens of other cell lines, kept from the model and scored the way the board scores. The story behind these models, the reasoning
at each fork and what did not work, is told in [Knobs N' Knockdowns](/posts/knobs-n-knockdowns/).

## Every entry on one board

The leaderboard's table with only Sidechain's entries on it, by overall score, calibration runs included. Each cell holds
the scaled score the board ranks on over the raw value, coloured as the leaderboard colours it, and a model's name opens
its section below.

Wondering where SER-22 to SER-26 went? The validation board closes on Oct 22 and we had more models lined up than days
left to score them, so we skipped ahead to SER-27abefhkrswx; the missing numbers are the models that never made it to the
start line.

{{< board >}}

## SER-29abefhkrswx — submitted 2026-10-21

SER-27abefhkrswx with two changes and no new knob. The pool of sources grows from four public screens to nine. The five
added are a screen in CD4 T cells, a second screen in H1 stem cells, a pooled screen across stem-cell lines, a screen in a
lung cancer line (Calu-3) and one in glioblastoma lines. And the strength of the summed profile (`b`) is no longer one
setting shared by every line: each line gets its own, kept short of the point past which more strength costs distance
(mse). The list of callable genes (`h`) follows the pool: it is chosen again on the larger pool's predictions, with the
knob itself unchanged.

It scored 0.2226 against SER-27abefhkrswx's 0.1875, and all six scores rose. Distance (mse) rose most, from 0.212 to
0.299; telling knockdowns apart (pds) went from 0.729 to 0.757, and size of change (nmae) from 0.094 to 0.126. Direction
(fid), reach and overlap (jac) also rose: from −0.013 to 0.012, from 0.105 to 0.139, and from −0.002 to 0.002.

## SER-28abefhkrswx — submitted 2026-10-20

SER-27abefhkrswx with one change: which genes go on its list of callable genes (`h`). For the knockdowns whose predicted
change is largest, the list now takes the genes predicted to move most; for the rest it keeps, as before, the genes whose
predicted move the scorer's test can see most clearly.

It scored 0.1875, level with SER-27abefhkrswx. Direction (fid) and size of change (nmae) rose a little, overlap (jac) fell
by about as much, and the other three scores held.

## SER-27abefhkrswx — submitted 2026-10-20

SER-21aefhkrswx with `b` back on, so the summed profile of a knockdown's cells carries the borrowed response more strongly
than each single cell does: the idea SER-7abefn first tried, now on top of everything added since. It went after the
distance score (mse), which had barely moved since SER-14aefksw.

It scored 0.1875 against SER-21aefhkrswx's 0.1757. Nearly all of the gain is in distance (mse), which rose from 0.143 to
0.212; the other five scores held.

## SER-21aefhkrswx — submitted 2026-10-19

SER-19aefhkrsw with one new knob, `x`. Untreated cells of one line already differ from each other along a few main
directions, such as how deeply each cell was sequenced and where it is in the cell cycle, and many knockdowns push cells
along those same directions. Here the directions, read from each line's own control cells, are taken out of the predicted
summed profile of most knockdowns, all but those with the largest predicted change; the single cells are left as they
were.

It scored 0.1757 against SER-19aefhkrsw's 0.1684. Telling knockdowns apart (pds) rose from 0.678 to 0.728, distance (mse)
fell a little, and the four test-based scores held.

## SER-20aefhkrsw — submitted 2026-10-19 · calibration run

SER-19aefhkrsw with its list of callable genes (`h`) taken to the far end: every gene that can be made callable in a
knockdown goes on the list. A calibration run, to see on the board what the longest possible list scores beside the
shorter ones SER-16aefhkrsw to SER-19aefhkrsw tried.

It scored 0.1673 against SER-19aefhkrsw's 0.1684. Reach fell from 0.104 to 0.088, overlap (jac) rose a little, and the
other four scores held.

## SER-19aefhkrsw — submitted 2026-10-16

SER-17aefhkrsw with a longer list of callable genes (`h`): in each knockdown, more of the genes whose predicted change
stands out most are made callable.

It scored 0.1684 against SER-17aefhkrsw's 0.1657. Direction (fid) and overlap (jac) rose, reach fell a little, and the
other three scores held.

## SER-18aefksvw — submitted 2026-10-16 · calibration run

SER-14aefksw with one new knob, `v`. When the effects borrowed from the sources are pooled, a source's vote counts for
more where its measurement is less noisy. Until here that noise was read from each knockdown's own cells; here it is read
from one curve per source, fitted over all of its knockdowns, of how a gene's cell-to-cell spread falls as its expression
rises. The same curve feeds the shrinkage rule (`s`). A calibration run, to see how a change of this kind carries over to
the challenge's cell lines.

It scored 0.1612 against SER-14aefksw's 0.1626. Telling knockdowns apart (pds) fell; distance (mse) and reach rose; size
of change (nmae) and direction (fid) fell a little, and overlap (jac) held.

## SER-17aefhkrsw — submitted 2026-10-08

SER-16aefhkrsw with a longer list of callable genes (`h`): the second point on the question of how long that list should
be.

It scored 0.1657 against SER-16aefhkrsw's 0.1423 and SER-14aefksw's 0.1626. Against SER-16aefhkrsw the gain is in
direction (fid); overlap (jac) rose a little and reach fell a little. Against SER-14aefksw, telling knockdowns apart
(pds), distance (mse) and reach are higher, and direction (fid) is lower by about as much as reach gained.

## SER-16aefhkrsw — submitted 2026-10-08 · calibration run

SER-15aefkrsw with one new knob, `h`. In SER-15aefkrsw the predicted cells kept all the cell-to-cell variety real cells
have, and that variety hid most predicted changes from the scorer's test: it called only a handful of genes changed in a
typical knockdown. Here, for a short list of genes in each knockdown, those whose predicted change stands out most, the
spread across cells is narrowed so that the test can call them; each gene's total count is unchanged. A calibration run,
to see what a short list scores on the challenge's cell lines.

It scored 0.1423 against SER-15aefkrsw's −0.0039 and SER-14aefksw's 0.1626. Direction (fid) won back most of what
SER-15aefkrsw had lost on it and stayed below SER-14aefksw's; overlap (jac) fell below both, and reach fell a little.

## SER-15aefkrsw — submitted 2026-10-07 · calibration run

SER-14aefksw with one new knob, `r`, in how each knockdown's 400 cells are written. In SER-14aefksw all 400 were drawn
around one profile: the line's control profile, moved by the predicted change. Here each one starts as a real control
cell of that line and has its counts moved to the predicted change, so the 400 keep the variety real cells have. The
emission dial (`e`) stays in the name and shapes no cell here. A calibration run, to see how much of a gain on our own
held-out tests carries over to the challenge's cell lines.

It scored −0.0039 against SER-14aefksw's 0.1626. One score, direction (fid), fell by more than the whole drop; the other
five rose or held. On four of our own held-out tests, from three cell lines, it had scored above SER-14aefksw's recipe on
every one.

## SER-14aefksw — submitted 2026-10-03

SER-12aefkw with a new shrinkage rule (`s`). Borrowed gene changes are noisy, so each one is pulled toward zero by how far
it can be trusted. Until here each change was shrunk by its own measurement noise in every source, and a change no bigger
than its noise was dropped. Here the changes are first combined across the sources, and each one is then shrunk by how
believable it looks next to all of that knockdown's other genes, so a weak change ends small rather than zero.

It scored 0.1626 against SER-12aefkw's 0.1438. Distance (mse) moved off zero for the first time, by more than the whole
gain; telling knockdowns apart (pds) rose a little, size of change (nmae) fell, and reach fell a little. On our own
held-out tests the rule had read as a trade between those same scores.

## SER-13aefknw — submitted 2026-10-02

SER-11abefknw with `b` turned off, so the summed profile carries the borrowed response at the same strength as each cell.
Sent beside SER-12aefkw, which also turned `n` off, so that the two together read the shrinkage setting on its own.

It scored 0.1360, level with SER-11abefknw. Telling knockdowns apart (pds) rose a little and reach fell a little. On our
own held-out tests the change had read as a small gain on three cell lines.

## SER-12aefkw — submitted 2026-10-02

SER-11abefknw with two knobs turned off. Without `b`, the summed profile carries the borrowed response at the same
strength as each cell; without `n`, shrinkage is back on, so each borrowed gene change is pulled toward zero by its own
measurement noise, and a change no bigger than its noise is dropped.

It scored 0.1438 against SER-11abefknw's 0.1360. Telling knockdowns apart (pds) and size of change (nmae) rose; reach
fell. On our own held-out tests the two changes had read as a small gain for the first and a trade for the second.

## SER-11abefknw — submitted 2026-09-30

SER-10abefnw with the neighbour knob (`k`) SER-9abefkn tried: each knockdown's borrowed response is nudged toward the
average response of the genes nearest to it in the STRING protein-interaction network, a map that places genes whose
proteins work together close to each other. The nudge turns the part of the response specific to this knockdown and
keeps its size.

It scored 0.1360 against SER-10abefnw's 0.1353. Reach rose; size of change (nmae) and telling knockdowns apart (pds) each
fell a little. On our own held-out tests, from three cell lines, the gain had been about ten times as large.

## SER-10abefnw — submitted 2026-09-30

SER-7abefn with one new knob, `w`. Each knockdown's predicted cells are built on a profile of the line's control cells.
Until here that profile was the average of each control cell's own make-up, every cell counting once. The scorer, though,
adds cells up before it compares them, so a deeply sequenced cell counts for more than a shallow one. Where a gene's share
of a cell depends on how deep the cell is, the two profiles differ, and every prediction carried that difference. `w`
builds the summed profile of the predicted cells on the pooled profile the scorer uses, and leaves each single cell where
it was.

It scored 0.1353 against SER-7abefn's 0.1131, nearly all of it in telling knockdowns apart (pds), which rose from 0.485 to
0.612; the four test-based scores held. On our own held-out tests the gain had been about the same size.

## SER-9abefkn — submitted 2026-09-29

SER-7abefn with the neighbour knob (`k`) as SER-8abefkn tried it, on a different gene map: the STRING protein-interaction
network, which places genes whose proteins work together close to each other. Each knockdown's borrowed response is
nudged toward the average response of its nearest genes on that map; the nudge turns the part of the response specific to
this knockdown and keeps its size.

It scored 0.1144 against SER-7abefn's 0.1131. Telling knockdowns apart (pds) rose most, reach rose, and size of change
(nmae) fell by about half of what those two gained. On the held-out tests its settings were picked on, the gain had been
about six times as large.

## SER-8abefkn — submitted 2026-09-29 · calibration run

SER-7abefn with one new knob, `k`. Each knockdown's borrowed response is nudged toward the average response of the genes
nearest to it in Tahoe-x1's gene embedding, a map of genes learned from single-cell data. The nudge turns the part of the
response specific to this knockdown a little, and keeps its size. A calibration run, to see how much of a gain on our own
held-out tests carries over to the challenge's cell lines.

It scored 0.1132, level with SER-7abefn's 0.1131. Telling knockdowns apart (pds) rose, and two of the test-based scores
together fell by most of that gain. On our own held-out tests, the one that fell most had risen on every one.

## SER-7abefn — submitted 2026-09-17

SER-6aefn with one new knob, `b`. The scorer reads a knockdown's 400 predicted cells twice: once added up into one summed
profile, and once cell by cell. Until here both readings carried the borrowed response at the same strength. `b` lets the
summed profile carry it more strongly than each single cell, by moving counts between deeper and shallower cells without
changing any cell's depth.

It scored 0.1131 against SER-6aefn's 0.1091. Telling knockdowns apart (pds) rose and the four test-based scores held, the
pattern our own held-out tests on three cell lines had shown; the gain on the board is smaller.

## SER-6aefn — submitted 2026-09-12

SER-4afn with one new knob, `e`, the emission dial: how much cell-to-cell noise the 400 predicted cells carry. Until here
every predicted cell sat evenly at the line's depth, with as little noise as possible; `e` sets the noise somewhere
between that and the full noise of counting molecules at random. That lets the scorer's test call the right genes a
little more often, at a small cost in size of change.

It scored 0.1091 against SER-4afn's 0.1078, and the gain landed where our own held-out tests said it would: reach rose,
and size of change (nmae) paid a little for it. The dial moved the same way on all six held-out cell lines it was tried
on.

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

SER-4afn with two new knobs on how the sources vote on a gene. `c` lets a source's weight for one gene know how much
evidence stands behind that gene in that screen, not only how large the screen was overall. `t` adds each source's
measured error on a held-out line to its uncertainty before the votes are counted, so a source cannot claim more
certainty than its record supports.

It scored 0.1075, a hair below SER-4afn's 0.1078. `t` has since been retired: shuffling which measured error belonged to
which source scored as well as assigning them correctly, so the knob was flattening the vote rather than informing it.
`c` was not tested on its own here.

## SER-4afn — submitted 2026-08-31

SER-3afn with the amplification (`a`) set again: stepped back from the strength our own held-out tests liked best to the
strength that the measured knockdown depth of the screens we borrow from points to. A new setting of a knob takes a new
model number, so this is model 4.

It scored 0.1078 against SER-3afn's 0.1071. Size of change (nmae) rose and telling knockdowns apart (pds) fell.

## SER-3afn — submitted 2026-08-30

SER-3fn with one new knob, `a`: the borrowed effects scaled up toward the size of a full knockdown. The screens we borrow
from silenced their genes only partly, so the effects they measured are smaller than what a complete knockdown produces.
Measuring how far each screen really silenced its genes, from its own data, predicted the right correction almost exactly
before it was scored.

It scored 0.1071 against SER-3fn's 0.1034. Telling knockdowns apart (pds) and reach rose; size of change (nmae) fell.

## SER-3afgn — submitted 2026-08-30

SER-3afn plus one deliberate probe, `g`: instead of carrying each gene's fold change over unchanged, bend the borrowed
effect toward how abundant that gene is at rest in the new line. Our own held-out tests said the plain fold change was
already the best rule; this entry tested that verdict on the real board, and it held.

It scored 0.0992 against SER-3afn's 0.1071.

## SER-3fn — submitted 2026-08-27

SER-3n with one new knob, `f`. When the sources are pooled per gene, each one's vote is now capped by how well it could
have measured that gene with the cells it had: a source that happened to see no spread no longer counts as infinitely
certain, and a source that saw a knockdown only once sits it out.

It scored 0.1034 against SER-3n's 0.0963. Size of change (nmae) and telling knockdowns apart (pds) rose; reach fell.

## SER-3n — submitted 2026-08-27

SER-2's four sources with the two genome-wide screens read in full, every gene they measured rather than the challenge's
panel alone, so each gene's vote rests on a much wider view of expression.

It scored 0.0963 against SER-2's 0.0937.

## SER-2 — submitted 2026-08-24

Named before the knob letters existed; shrinkage is off, as in SER-1n. The sources grow from two to four: the K562
genome-wide screen and H1, plus the challenge-panel part of two genome-wide CRISPRi screens in colon and kidney lines. It
is the first entry to cover all 300 target genes with a measured effect instead of a generic fallback.

It scored 0.0937 against SER-1n's 0.0822.

## SER-1n — submitted 2026-08-22

SER-1 with the shrinkage of borrowed effects switched off (`n`): the many small, noisy per-gene effects that shrinkage
silenced turn out to carry the direction signal that telling knockdowns apart reads.

It scored 0.0822 against SER-1's 0.0788.

## SER-1p — submitted 2026-08-21

SER-1 with its predicted cells carrying the full noise of random counting (`p`, Poisson cells) instead of sitting evenly at
the line's depth: a one-knob test of how cell-to-cell noise prices into the test-based scores. Evenly spread cells won,
and stayed.

It scored 0.0730 against SER-1's 0.0788.

## SER-1 — submitted 2026-08-21

The first entry, and the series' plain recipe. Each target gene's knockdown effect is borrowed from the screen where it was
measured, the K562 genome-wide screen (272 of the 300 targets) or H1 (25), pooled per gene by how confidently each source
measured it, set on each new line's own resting control profile, and written out as 400 cells per knockdown at the line's
own sequencing depth.

It scored 0.0788.

## The knobs

Each lowercase letter in a name is one knob moved off the series' plain recipe, with the same meaning in every series. The
entry that first carried it is in brackets.

- **`a` · amplified transfer.** The screens we borrow from silenced their genes only partly, so the borrowed effects are
  scaled up toward the size of a full knockdown. (SER-3afn)
- **`b` · summed-profile strength set apart.** The scorer reads a knockdown's cells twice, added up into one summed profile
  and one by one; this knob lets the summed profile carry the borrowed response at a different strength from the single cells.
  (SER-7abefn)
- **`c` · coverage-tiered weights.** A source's vote on a gene also counts how much evidence stands behind that gene in that
  screen. (SER-5acfnt)
- **`e` · emission dial.** How much cell-to-cell noise the predicted cells carry, set between evenly spread cells and the
  full noise of random counting. (SER-6aefn)
- **`f` · floored source weights.** A source's vote on a gene is capped by how well its cells could have measured that gene.
  (SER-3fn)
- **`g` · abundance-ratio transfer.** The borrowed change is bent toward how abundant the gene is at rest in the new line;
  tested once, and the plain fold change held. (SER-3afgn)
- **`h` · a chosen list of callable genes.** For a list of genes in each knockdown, the spread across cells is narrowed so
  the scorer's test can call them changed, with each gene's total count unchanged. Which genes, and how many, is what later
  entries move. (SER-16aefhkrsw)
- **`k` · neighbour genes blended in.** Each knockdown's borrowed response is nudged toward the average response of the
  genes near it on a gene map, which turns its direction and keeps its size. (SER-8abefkn)
- **`n` · no shrinkage.** The borrowed changes are used as measured, with no pull toward zero. (SER-1n)
- **`p` · Poisson cells.** The predicted cells carry the full noise of random counting, in place of evenly spread cells.
  (SER-1p)
- **`r` · cells built from real control cells.** Each predicted cell starts as a real untreated cell of the line, with its
  counts moved to the prediction. (SER-15aefkrsw)
- **`s` · a different shrinkage rule.** The sources are combined first, and each change is then shrunk by how believable it
  looks next to all of that knockdown's genes. (SER-14aefksw)
- **`t` · transfer-error floor.** A source's measured error on a held-out line is added to its uncertainty before it votes.
  Retired after one entry. (SER-5acfnt)
- **`v` · a fitted noise curve.** A source's noise is read from one curve fitted over all of its knockdowns, not from each
  knockdown's own cells. (SER-18aefksvw)
- **`w` · the scorer's control profile.** The summed profile of the predicted cells is built on the pooled control profile
  the scorer itself uses. (SER-10abefnw)
- **`x` · control-cell directions taken out.** The main directions along which a line's own untreated cells differ are taken
  out of the summed profile of most knockdowns. (SER-21aefhkrswx)

## The six scores

The board scores an entry on six things, each scaled so that 0 is what predicting a line's average knockdown earns and 1
is what one half of the real experiment earns predicting the other half; the overall score is their plain average over
the three cell lines. The last four read one test, which compares a knockdown's predicted cells with the line's control
cells gene by gene and calls a gene changed when the difference is clear.

- **pds · telling knockdowns apart.** Is each predicted summed profile closer to its own real knockdown than to the
  others?
- **mse · distance.** How far the predicted summed profile sits from the real one, after the scatter expected from
  sampling a few hundred cells is taken off.
- **jac · overlap.** How well the genes the test calls changed match the genes that really changed.
- **nmae · size of change.** On the genes that really changed, how close the predicted fold changes are to the real ones.
- **fid · direction.** Of the genes the test calls changed, the share moving the right way, discounted when too few are
  called.
- **reach · reach.** Going down the genes that really changed in our order of confidence, how far before the predicted
  directions start going wrong.
