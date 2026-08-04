"""Cis layer: per-gene sequence embeddings (the modern ntEmbd).

Embeds each gene's 3'UTR (fallback: full cDNA) with a nucleotide language model
(Nucleotide Transformer). Long UTRs are chunked and pooled. Frozen node features;
`precompute: true` means run once on a rented GPU and cache vectors in lamindb, then
everything downstream runs on the Mac against the cache.
"""
from __future__ import annotations
from sidechain.priors.base import PriorSource, PriorArtifact


class CisSequenceSource(PriorSource):
    def fetch(self) -> None:
        # TODO(dev): pull 3'UTR sequences (GENCODE/Ensembl) for master gene space; cache.
        raise NotImplementedError

    def build(self) -> PriorArtifact:
        # TODO(dev): tokenize -> NT forward pass -> pool chunks -> (n_genes, embed_dim).
        # Genes without a UTR: zero rows (preserve order). Cache the matrix to lamindb.
        raise NotImplementedError
