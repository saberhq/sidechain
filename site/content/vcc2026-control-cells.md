---
title: Control cells
description: Quality checks on the untreated control cells Arc released for each of the six cell lines in the Virtual Cell Challenge 2026.
layout: controls
date: 2026-10-23
# a card in the landing page's rail (layouts/_partials/page-card.html)
rail: true
card_title: VCC 2026 control cell populations
badge: interactive
cta: Explore →
---

The challenge asks every team to predict what silencing a gene does to six cell lines, named only by letters: A, B and
C were the validation round, and D, E and F decide the final ranking. For each line Arc released 18,400 control cells,
cells that carry a guide designed to silence nothing (46 such guides, 400 cells each). They are the only cells of those
lines anyone gets, so every prediction starts from them. This page runs the quality checks a single-cell study opens
with on those controls.

**Maps and QC.** Each dot is one cell, and cells with similar expression sit together. *Validation*, *Test* and *All
six* put those contexts on one set of axes, so contexts separate by identity and by assay; a single letter shows that
context on its own map, which is where what varies inside one context shows. *Regressed* first removes each cell's
depth and its two cell-cycle scores from every variable gene, so what still forms an island is where a clone, a
sub-population or a misbehaving guide would show. *Violin* shows the same cells as one QC measure, one violin per
context, cluster or phase. A card in the rail shows its context in every figure. Cluster numbers are local to each map.
In the cluster table, a clone would show as a cluster drawn from a few guides (the largest guide's share far above 1/46,
about 2.2 %), and a group of damaged or shallow cells as a cluster spread over every guide with a fraction of the
context's UMIs.

The measures behind the colours:

- **UMIs**: molecules counted per cell, which is sequencing depth. Arc states a median near 20,000 for its lines.
- **Genes**: genes detected per cell. It rises with depth; at equal depth, fewer genes means a library made of few
  transcripts.
- **Mito UMI %**: the share of a cell's UMIs on mitochondrial genes. In droplet data a high share marks a stressed or
  broken cell. These cells were fixed and read with probes (10x Flex), the share sits far lower, and the usual 10–20 %
  cut-off does not apply.
- **Top-50 UMI %**: the share held by the cell's own 50 most-counted genes, high in a cell whose library is made of few
  transcripts.
- **Complexity**: log10 genes over log10 UMIs, how quickly new genes appear as depth grows.
- **S score** and **G2/M score**: how strongly a cell expresses the DNA-replication and the mitosis genes of Tirosh et
  al. (2016), against its own context. **Phase** is the call the two scores make, G1, S or G2/M. The call compares
  cells within a context, so a context where nearly every cell is cycling gets few G1 calls: read the scores rather
  than the labels.
- **IFN score**: the interferon response, an antiviral programme (ISG15, the IFIT genes, MX1 and others).
- **Stress score**: immediate-early and heat-shock genes (FOS, JUN, EGR1 and others) a cell switches on within minutes
  of handling.

**Guides and clones.** If one of the 46 non-targeting guides still moves expression (an off-target site), or a context
holds a clone that grew out, its cells would sit apart from the rest. Each dot is one guide: how far its 400 cells'
centre sits from the context's centre, as measured (x) and regressed (y), in standard deviations of 500 shuffles of the
guide labels, so 0 is what chance gives. Dots in ink pass q < 0.05 after the Benjamini–Hochberg correction over the 46
guides. The bars are each guide's median UMIs: a guide whose cells were all shallower would show there first.

**Highest-expressed genes.** The 20 genes holding the most UMIs, as each cell's share (box 25–75 %, whiskers 5–95 %),
and every gene's mean count against the share of cells detecting it, with the 2,000 variable genes the context's own
map uses in its colour.
