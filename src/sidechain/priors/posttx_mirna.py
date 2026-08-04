"""Post-transcriptional layer: miRNA -> target repression edges (Saber's edge).

Primary source miRBind2 (sequence-only binding + repression score); TargetScan/
miRDB as toggleable alternates. Directed edges onto the target gene, weighted by
repression strength — refines the *magnitude* of downstream deltas the trans graph
predicts (mRNA stability via 3'UTR).
"""
from __future__ import annotations
from sidechain.priors.base import PriorSource, PriorArtifact


class MiRNATargetSource(PriorSource):
    def fetch(self) -> None:
        # TODO(dev): run/download miRBind2 predictions (or parse TargetScan); cache.
        raise NotImplementedError

    def build(self) -> PriorArtifact:
        # TODO(dev): (miRNA, target_gene, repression_score) -> directed edges.
        # Aggregate multiple miRNAs per target as needed (sum/max of scores).
        raise NotImplementedError
