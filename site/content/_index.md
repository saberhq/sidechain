---
title: Sidechain
description: Predicting how a cell's transcriptome shifts when a gene is silenced — a solo entry to the Virtual Cell Challenge 2026, built in the open.
topics:
  - Perturb-seq
  - zero-shot cell lines
  - human + Claude Code
topicAccent: Virtual Cell Challenge 2026
series:
  - code: GLY
    chain: none — the simplest residue
    names: nulls and baselines
  - code: ALA
    chain: a single methyl
    names: a single statistical shift
  - code: SER
    chain: a hydroxyl — small, reactive, transfers a group
    names: cross-line delta transfer (today's models)
  - code: CYS
    chain: forms bridges between chains
    names: context-aware models
  - code: HIS / LYS / ARG
    chain: long and charged — act at a distance
    names: graph and prior heads
  - code: PHE / TYR / TRP
    chain: the aromatic heavyweights
    names: deep generative models
  - code: PRO
    chain: bends the backbone
    names: fusion
---

**What it is.** Given expression data for genes that *have* been silenced, predict what happens when you silence a gene never seen silenced — in cell lines the model has never seen, given only their resting state. Sidechain is a solo entry that works as a small research group: Saber at the bench, Claude Code at the keyboard, and a ladder of models climbed from the simplest baseline up, keeping only what the metric pays for.

**How it's shared.** Code, configs and tests are on [GitHub](https://github.com/saberhq/sidechain). Progress notes go out on [LinkedIn](https://www.linkedin.com/in/saberhq); the longer write-ups live here, and dashboards of the data and the models will join them as they are built.
