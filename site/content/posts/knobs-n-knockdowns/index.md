---
title: "Knobs N' Knockdowns"
date: 2026-10-21
draft: false
pinned: true
description: "How Sidechain's entries to Arc's Virtual Cell Challenge 2026 were built one setting at a time: what each setting does to a predicted cell, why it was tried, and what the leaderboard said back."
short: "knobs n' knockdowns"
image: cover.jpg
caption: "November 8, 2025"
image_alt: "Leaf shadows on a sunlit orange surface"
images: [social.jpg]
---

During the validation phase of the Virtual Cell Challenge, the two months in which Arc scores entries on its first three cell lines, we sent 30 entries to the leaderboard. Six of them were calibration runs, entries sent to test a new method on the official board, and nearly all of them changed a single thing from an earlier entry, so that the leaderboard would read as a lab notebook rather than a scoreboard. Our first entry, SER-1, scored 0.0788, and our best so far, SER-29abefhkrswx, scored 0.2226. On this scale, 0 is what you get by predicting a cell line's average knockdown response for every knockdown, and 1 is how well one half of the real experiment predicts the other half.

Looking back, the biggest swings in our score did not come from predicting the gene changes better. They came from two steps that sit after the prediction. The first is how we write the predicted change into cells: keeping the prediction the same, switching from synthetic cells to real control cells dropped our score from 0.1626 to below zero, and making the right genes detectable in those real cells brought it back above where it started. The second is the reference the prediction is compared against: building our pseudobulk on the scorer's own control profile added 0.022 without changing a single predicted fold change. Most of this post is about how we learned that, one knob at a time. The exception came at the very end, when our last entry widened its pool from four screens to nine and every metric rose.

## The recipe

Every model in our SER series rests on one idea: never predict a cell's expression from scratch. Predict the change a knockdown makes, and apply it to real control cells of the new line, because "the control cell already knows everything hard about being that line." Arc gives us 18,400 untouched control cells for each challenge line, and they carry the line's whole identity. What they cannot show is what silencing one gene does, and that is all the model has to supply. We do it in four steps:

- **Borrow.** For each of the 300 target genes, take the log2 fold change measured in every CRISPRi screen that silenced it: Arc's 2025 data in H1 embryonic stem cells, the genome-wide K562 screen of Replogle and colleagues (2022), and Xaira's X-Atlas/Orion screens in HCT116 and HEK293T, joined in our last entry by five more. A coming post on our corpus tells how we built that shelf.
- **Pool.** Average the sources gene by gene, weighting each one by how precise its measurement is.
- **Adjust.** Resize the pooled change, or nudge it.
- **Emit.** Write each knockdown as 400 cells of integer counts, built on the line's own control cells.

Each lowercase letter in a model's name marks one knob moved away from the plain recipe, so SER-1 is the recipe at its plainest and SER-27abefhkrswx has ten knobs moved. SER stands for serine, whose side chain transfers a group ([why Sidechain](/posts/what-sidechain-is/#why-sidechain)), and every entry has a short section on the [models page](/models/).

{{< svg src="fig-recipe.svg" caption="The SER recipe in four steps, with every letter a model's name can carry placed on the step it changes. The value each knob takes lives in the build record, never in the name." >}}

## How the score reads a prediction

For each of 300 knockdowns, in each of three cell lines that Arc keeps anonymous as contexts A, B and C, we submit 400 cells. Each cell is a vector of raw counts over 18,533 genes. Arc scores the file with cell-eval2, its public scoring package, as the mean of six metrics that read the cells in two different ways.

Two metrics read the pseudobulk, a knockdown's 400 cells summed into one profile. `pds` asks whether the predicted change is closer, by cosine, to its own real knockdown than to the other 299. `mse` measures how far the predicted pseudobulk sits from the real one, after allowing for the noise that sampling a few hundred cells brings.

The other four read a differential-expression table. For every gene, a Wilcoxon rank-sum test compares a knockdown's predicted cells with the line's control cells and calls the gene changed when the difference is significant. The same test runs on the real knocked-down cells, and four metrics compare the two tables. `nmae` scores the size of our fold changes on the genes that really changed. `fid` scores the direction of the genes we call and penalises calling too few. `reach` asks how far down the real list our ranking stays right, and `jac` measures the overlap of the two called sets. Each metric is then scaled per line, from the baseline of 0 to the replicate's 1.

This matters because a submission is not a table of fold changes but cells, and we construct every count in them. That is two decisions for each gene in each knockdown: its total count across the 400 cells, which sets its mean and so its fold change and pseudobulk value, and how that total is spread over the cells, which the Wilcoxon test reads as well as the mean. Two predictions with the same mean can be called changed or not by the spread alone. August worked on the first decision; from September on, more and more of the gains came from the second.

## Scoring at home

Arc scores only its three challenge lines and accepts two entries a day, so we built a local copy of the scoring that we call the mirror. It takes a cell line for which we do hold a CRISPRi screen, hides that line's knockdowns and keeps only its control cells. We then predict the hidden knockdowns from the other screens, exactly as we would for a new line, and score the prediction with cell-eval2. Holding out each of HCT116, HEK293T and K562 in turn gives several test sets.

The mirror is good at one job: comparing two versions of our recipe, such as the same model with and without one knob, and telling which is better. It is poor at another: forecasting the score the leaderboard will give. The scale's 0 and 1 are computed from each line's own data, so the same model lands on a different number on every line, and a gain measured on our lines rarely arrives on the leaderboard at the same size. Much of this post is about that gap.

{{< svg src="fig-overall.svg" caption="The overall score of our 24 normal entries in the order they went up, from SER-1's 0.0788 to the latest, in carmine. No line joins the points, because each entry asks a different question; the six calibration runs are discussed in the text." >}}

## August: getting the change right

SER-1, sent the day after the challenge opened, was the plainest recipe we could defend. It borrowed each target's change from the K562 screen, which had silenced 272 of the 300 targets, and from Arc's H1 data. It shrank each change toward zero wherever it was no bigger than its own noise, and it spread each gene's counts over the cells as evenly as whole numbers allow. It scored 0.0788. Switching the shrinkage off then won, because the many small, noisy changes it had been setting to zero carried the direction `pds` reads.

SER-2 added X-Atlas to the pool, and with it every one of the 300 targets had a change measured in a real knockdown somewhere. It scored 0.0937, but the gain did not come where we expected. `pds` stayed flat, and the rise came from the four test-based metrics. The mirror had promised more because its held-out line was K562, whose own screen leaves the pool when it is held out, so there X-Atlas filled a hole in coverage; on the challenge lines it mostly added second and third votes. The lesson: hold coverage fixed when you price a new source.

The next fix came from Claude, the AI model that works this project with me, reading the X-Atlas ingest. The pool weights each source by one over its variance, and a source that had seen a knockdown in a single cell, or a gene that no cell expressed, reported a variance of exactly zero. Such a source gets a near-infinite weight, and 11.2 percent of the variances in H1's table were exactly zero. The letter `f` puts a floor of counting noise under every variance, and SER-3fn scored 0.1034.

Then the size of the change. The screens we borrow from silenced their targets only partly, some to less than half of a full knockdown, so the effects they measured are smaller than what a complete knockdown produces. Claude measured how far the targeted genes themselves fell in each screen, and the letter `a` scales the pooled change up to match; we call that multiplier the strength. SER-3afn scored 0.1071.

SER-5acfnt moved two letters at once. One of them added each source's transfer error, how badly it had predicted a held-out line, to its variance, and the entry landed just below its parent. Giving each source another source's error scored just as well on the mirror, which gave us the rule we would carve over the door: a knob is real only if it beats a scrambled copy of itself.

## September: the cells and the anchor

In September the question moved from what to predict to how to write it into cells. SER-6aefn put a dial on the cells' noise, between perfectly even cells and the random scatter of Poisson sampling. SER-7abefn used the fact that the scorer reads the same cells twice, once summed and once one by one. The letter `b` lets the pseudobulk carry the borrowed change at a higher strength than each single cell does. It works by moving counts between deeper and shallower cells, which changes the sum without changing any cell's depth. SER-7abefn scored 0.1131; remember `b`, because it comes back.

The letter `k` nudges each knockdown's predicted change toward those of its nearest neighbours on a gene map, on the idea that genes that behave alike can lend each other a little. With neighbours from STRING, a database of protein interactions, SER-9abefkn gained a little, inside the recipe's run-to-run noise and about a sixth of what the mirror had promised.

That gap between local and leaderboard gains had followed us since SER-2, and late in September Claude found part of the reason. Every predicted pseudobulk is compared with a control profile. The scorer builds that profile by pooling the counts of all the control cells, so a deeply sequenced cell counts for more than a shallow one. We had built ours by averaging each control cell's composition, so that every cell counted once. Wherever a gene's share of a cell's counts depends on sequencing depth, the two profiles differ, and every SER prediction up to then carried that difference.

The letter `w` builds our pseudobulk on the scorer's own control profile. SER-10abefnw, which is SER-7abefn with `w` and nothing else, gained 0.022. It was the first time a gain measured on the mirror arrived on the leaderboard at full size, and the rented GPU machine that built it cost about two dollars. When Claude re-ran 22 earlier measurements on the scorer's profile, about half of the knobs changed their verdict.

## October: shrinkage, then the shape of a cell

Shrinkage returned in October. Strength is a volume knob, one multiplier on every gene; shrinkage is an equaliser that pulls each gene toward zero by how little it stands out from its own noise, so the pattern changes. SER-12aefkw, which turned the plain shrinkage back on and dropped `b`, scored 0.1438. SER-14aefksw then replaced the rule (`s`). It pools the sources first and then shrinks each pooled change by how believable it is next to all of that knockdown's genes, so a weak change ends small rather than at zero. It scored 0.1626, and `mse`, which had been at zero for every entry until then, finally moved.

Then the second decision, the shape of the cells. Every SER entry up to SER-14aefksw had drawn its cells around one averaged profile, and drawn cells are tidier than real ones: their depths and counts vary less from cell to cell. SER-15aefkrsw, a calibration run, instead built each of the 400 cells from a real control cell of the line and moved its counts to the predicted change (`r`). On the mirror it had beaten its parent on four test sets out of four. On the leaderboard it scored −0.0039, and almost all of the fall was in `fid`. Real-shaped cells are so like control cells that the test called only a handful of genes per knockdown, and `fid` penalises every real change we fail to call. The predicted change was in the cells, but the test could no longer see it.

The letter `h` is the answer. For a chosen list of genes in each knockdown, it narrows the spread of counts across the cells until the test can call them, and it leaves each gene's total untouched, so the pseudobulk barely moves. With a short list, SER-16aefhkrsw, another calibration run, recovered to 0.1423. A longer list took SER-17aefhkrsw to 0.1657, and a longer one still took SER-19aefhkrsw to 0.1684. Putting every gene that could be made callable on the list, in the calibration run SER-20aefhkrsw, gave a little back.

{{< svg src="fig-callset.svg" caption="Raw fid, reach and jac on the leaderboard along the list of callable genes: SER-14aefksw on drawn cells (the dashed line), then the real-cell entries with no list, a short list, two longer ones and every callable gene, with SER-19aefhkrsw in carmine. SER-15, SER-16 and SER-20 are calibration runs, drawn because this figure is about them. Raw values, not the scaled ones the leaderboard ranks on." >}}

The last two knobs of the validation phase both work on the pseudobulk alone, and together they added 0.019 to the overall score. The first, `x`, starts from the control cells. Untreated cells of one line already differ from each other along a few main directions: how deeply each cell was sequenced, where it sits in the cell cycle, how stressed it is. Many knockdowns that slow growth push cells along those same directions. That part of a predicted change is shared by many knockdowns, so it says little about which gene was silenced. And `pds` ranks each prediction against all the others, so anything the predictions share blurs the ranking. `x` reads the main directions of variation from the line's own control cells and removes them from the predicted pseudobulk of most knockdowns. The knockdowns with the largest predicted change keep theirs, since for them a shift along those directions can be real signal. The single cells are left alone, so the four test-based metrics do not move. SER-21aefhkrswx scored 0.1757, with `pds` up from 0.678 to 0.728.

The second was `b`, back from September. Its small September gain did not survive the anchor fix, and on drawn cells it could not pay where it should have, on `mse`. That metric forgives the sampling noise in our predicted cells, but only up to how much our predictions differ from one another, and a stronger response in the pseudobulk widens that difference, so more of the noise is forgiven. To give the sum a stronger response than each cell, `b` leans a gene's counts toward the deeper cells. Our drawn cells varied too little in depth, so the lean had to be steep, and it disturbed the shape the test reads. The tidy cells also carried little sampling noise to forgive. Real control cells span the real range of depths and carry real noise, so the same lean is gentle and there is more to forgive. SER-27abefhkrswx put `b` back on top of everything since, and `mse` rose from 0.143 to 0.212, which took the overall score to 0.1875.

The validation phase ended with an entry that turned no new knob. SER-29abefhkrswx widened what we borrow, from four screens to nine. The five new ones are a screen in CD4 T cells, a second screen in H1 stem cells, a pooled screen across stem-cell lines, and screens in a lung cancer line and in glioblastoma lines; the coming post on our corpus tells their story. The same entry gave each line its own strength for `b` instead of one shared setting. It scored 0.2226, up from 0.1875, and all six metrics rose.

## What did not ship

Much of what we tried never reached the leaderboard, or reached it once and stopped there. Most of these ideas were stopped by a test written before the run, and each one's file keeps a note of what would reopen it.

- **Single-cell foundation models.** We read ten pretrained models closely, and none can be asked what silencing a given gene does: none takes a perturbation as input over an open list of genes, and the published checkpoints of STATE, Arc's own perturbation model, know a fixed list of 2,024 perturbed genes that holds none of the 272 targets we checked. What these models do ship is an embedding for every gene, so we asked a narrower question: do genes that sit close together in a model's embedding respond alike when silenced? Each table was tested against the same table with its genes shuffled, and against ESM2, a protein language model, as the bar to beat. Some tables carried real signal and beat their own scramble, but none beat ESM2 consistently across pairs of cell lines, and line-specific tables never beat a single static one. As the gene map for the neighbour knob, Tahoe-x1's table (SER-8abefkn, a calibration run) scored level with its parent, and STRING's network did a little better. We closed the class in good company: "none of the deep learning models was able to consistently outperform the mean prediction or the linear model" (Ahlmann-Eltze and colleagues, *Nature Methods*, 2025).
- **A neural network of our own.** PHE-2, a calibration run, is the one entry that predicts each knockdown with a network instead of borrowing it: it reads a line's control cells and writes the knocked-down cells directly. It scored −0.9807, and its cells carried about half the expressed genes a real cell does. As a second arm beside SER it did no better, whether blended in or chosen target by target. A perfect chooser would have gained a lot, but nothing we could compute in advance told us which arm to trust for which gene.
- **Smarter trust in the sources.** Six ideas tried to weigh the sources better than one over the variance does, and none earned its place against that plain weight:
  - adding each source's transfer error to its variance;
  - counting the cells behind each gene in each source;
  - smoothing each source's variance along one curve over expression, which went up once as the calibration run SER-18aefksvw and scored a little under its parent;
  - shrinking only the small screens with many cells per target;
  - a weight for the sources that reach a target through the very same guide construct, as K562 and X-Atlas do for all 272 of their shared targets;
  - weighting each source's cells by how strongly its guide took.
- **Bending the change to the new line.** The letter `g` bent each borrowed change toward how abundant the gene is in the new line's control cells. SER-3afgn lost to the plain fold change on the leaderboard, as the mirror had said it would.

## What we learned

**One knob per entry turns a leaderboard into an experiment log.** Only a single change against a named parent makes the difference between two scores mean something. The two entries where we moved two letters at once are exactly the two where we needed a follow-up to learn which letter did what.

**The mirror ranks; it does not measure.** Orderings usually travel, and sizes rarely do. The STRING nudge arrived at a sixth of its local size. Shrinkage went from a trade-off at home to a clear gain on the leaderboard. SER-15aefkrsw beat its parent on every local test set and then fell to −0.0039. The anchor fix was the first gain to arrive whole, because it changed how every prediction, local and official, was built.

**Shape is not size.** SER-15aefkrsw, SER-16aefhkrsw and SER-17aefhkrsw predict the same change and score from −0.0039 to 0.1657. A scorer that calls genes with a rank test reads the cells, not the profile behind them, so how the cells are written is a modelling decision as much as the change they carry. The scorer's code is public, and several of these lessons were already in it.

**The leaderboard was the only window on A, B and C, and I used too little of it.** The rules allowed two entries a day, about 120 in the validation phase, and we sent 30. Our own test lines could rank two versions of a model, but only the leaderboard could say how a change behaves on the challenge's own lines. The biggest surprises in this post, the real-cell crash and the flat end of the gene list, came from the leaderboard alone. Every unused slot was a reading of those lines I chose not to take. If I started again, I would send far more of the variants we only tested at home, and treat each one as a measurement rather than a bid for a better rank.

## Where it stands

Our best entry, SER-29abefhkrswx, scores 0.2226 overall. Here are its six metrics on the same scale, where 0 is the score of predicting each line's average knockdown response and 1 is the score of one half of the real data predicting the other half:

| metric | what it asks | SER-29abefhkrswx |
|---|---|---|
| `pds` | Is each predicted change closest to its own knockdown? | 0.757 |
| `mse` | How far is the predicted pseudobulk from the real one? | 0.299 |
| `reach` | How far down the real list does our ranking stay right? | 0.139 |
| `nmae` | How close are our fold changes on the genes that really changed? | 0.126 |
| `fid` | Do the genes we call move in the right direction? | 0.012 |
| `jac` | Do we call the same genes as the real data? | 0.002 |

On `pds` we have covered three quarters of the distance from the baseline to a replicate, so we mostly tell the knockdowns apart. Everything that depends on which genes the test calls is still close to the baseline.

{{< svg src="fig-members.svg" caption="Each of the six scaled metrics for our 24 normal entries, one panel per metric on its own axis, the latest in carmine; mse sits at zero until SER-14aefksw. The calibration runs are left off: SER-15aefkrsw's fid of −1.138 would flatten every other point." >}}

Every entry was built by a Claude session and sent only on my go, which a coming post on how we run the project describes, and the live table is the [standings](/#standings). On October 22 Arc releases the three cell lines that decide the prizes, and there is no leaderboard for final entries until the winners are announced. The record of the final phase starts after the November 5 deadline, and it will get a post of its own.
