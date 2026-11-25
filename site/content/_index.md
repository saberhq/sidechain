---
title: Sidechain
description: Predicting how a cell's transcriptome shifts when a gene is silenced — a solo entry to the Virtual Cell Challenge 2026, built in the open, ranked 44th of 807 teams on the final leaderboard.
# The line under the wordmark (Saber, 2026-10-23). Markdown; `description` above stays the search and link-preview text.
# The picture a link to this page shows (layouts/_partials/share.html); without these two lines, the share card.
images: [desk.jpg]
images_alt: "A desk at night: a lit lamp, an alarm clock, an Arsenal cup of pens and a laptop on a stand, below a yellow sheet that reads “Oct 16 → 22, last week of validation phase”."
tagline: "This is a fun project where I'm the only human in the loop of a multi-agent research workflow powered by Claude — a solo-team entry to [VCC26](https://virtualcellchallenge.org/) that built robust models to predict how a cell's transcriptome shifts when a gene is silenced. This website documents our journey in public: the [models page](/models/) says what each of our submissions tried, and the [blog posts](/posts/) tell the story behind them."
# "How it works": the recipe's lede (Markdown) and steps (layouts/home.html, layouts/home.llms.txt).
recipe:
  lede: "Our two series of models, SER and CYS, share one recipe: take the expression change a gene's knockdown caused in public screens, and build the predicted cells on the new cell line's own control cells. It runs in four steps, and [Knobs N' Knockdowns](/posts/knobs-n-knockdowns/) tells how the recipe took shape through the validation round, one setting at a time."
  steps:
    - name: Borrow
      text: "For each target gene, take the change measured in the public screens that silenced it."
    - name: Pool
      text: "Average the screens gene by gene, weighting each one by how precise its measurement is."
    - name: Adjust
      text: "Resize the pooled change, or nudge it toward the changes of genes that behave alike."
    - name: Emit
      text: "Write each knockdown as 400 cells of integer counts, built on the line's own control cells. In our last two entries, CYS-5 and CYS-6, a small model, learned on those screens, sets how far and which way each gene moves in those cells."
---
