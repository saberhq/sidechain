<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/wordmark-dark.svg">
  <img alt="Sidechain — Virtual Cell Challenge 2026" src="assets/wordmark-light.svg">
</picture>

*Predicting how a cell's transcriptome shifts when a gene is silenced — a solo entry to the
[Virtual Cell Challenge 2026](https://virtualcellchallenge.org/), built in the open.*

Created by [Saber Hafezqorani](https://saberhq.com)

---

**What it is.** Sidechain is a multi-agentic research workflow that tries to decode one of biology's hardest problems – understanding how a cell's biology changes when a gene is silenced. The idea is to climb a ladder of models from the simplest
baseline up and keeping only what the metric pays for.

**How it's shared.** Code, configs and tests land here, and I'll share the journey with the broader scPerturbSeq research community through [LinkedIn](https://www.linkedin.com/in/saberhq) and [blog](https://saberhq.com/blog/) posts. Let's catch up along the way.

## Where we stand

Every submission, as Arc scored it. The [live board](https://virtualcellchallenge.org/leaderboard)
shows only a team's latest entry, so the rank is the one each entry held when it was scored.

| date (UTC) | submission | overall | rank when scored |
|---|---|---|---|
| 2026-08-21 | `sidechain r1-delta-even v1` | 0.0788 | #2 of 41 |
| 2026-08-21 | `Sidechain SER-1p` | 0.0730 | — (a probe) |
| 2026-08-22 | `Sidechain SER-1n` | 0.0822 | #14 of 107 |

## Why "Sidechain", and how the models are named

A *side chain* is the part of an amino acid that makes it different from the other nineteen.
Our models are named the same way: each **series** is an amino acid whose side chain matches
the model's character, and a number counts entries within it — `GLY` (no side chain: nulls and
baselines), `ALA` (a single statistical shift), `SER` (transfers a group: cross-line delta
transfer), `CYS` (bridges chains: context-aware models), `HIS/LYS/ARG` (long-range charged:
graph and prior heads), `PHE/TYR/TRP` (the aromatic heavyweights: deep generative models),
`PRO` (bends the backbone: fusion). A letter suffix marks a one-knob variant — `SER-1p` is
`SER-1` with Poisson cells. For the chemistry behind the pun, Compound Interest's
[20 common amino acids](https://www.compoundchem.com/2014/09/16/aminoacids/) poster.

