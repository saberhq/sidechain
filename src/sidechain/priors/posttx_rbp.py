"""Post-transcriptional layer: RBP -> target binding edges (Saber's edge).

CLIP-based RBP binding sites filtered to 3'UTRs (POSTAR3, ENCODE eCLIP), with
ATtRACT/oRNAment motif predictions as a fallback where CLIP is unavailable.
Directed edges onto the bound target gene.
"""
from __future__ import annotations
from sidechain.priors.base import PriorSource, PriorArtifact


class RBPBindingSource(PriorSource):
    def fetch(self) -> None:
        # TODO(dev): download CLIP peaks; filter region_filter=='3utr'; cache.
        raise NotImplementedError

    def build(self) -> PriorArtifact:
        # TODO(dev): (rbp_gene, target_gene, n_sites/clip_score) -> directed edges.
        raise NotImplementedError
