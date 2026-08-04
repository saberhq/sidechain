"""Trans layer: transcriptional regulation (TF -> target) and functional edges.

Sources handled by this loader: DoRothEA/TRRUST (directed, signed, via decoupler/
OmniPath), STRING (undirected functional/PPI), and — reused for now — ENCODE
chromatin tracks. Emits directed/undirected COO edges aligned to gene_index.
"""
from __future__ import annotations
from sidechain.priors.base import PriorSource, PriorArtifact


class TransGRNSource(PriorSource):
    def fetch(self) -> None:
        # TODO(dev): decoupler.get_dorothea() / OmniPath / STRING download; cache to lamindb.
        raise NotImplementedError

    def build(self) -> PriorArtifact:
        # TODO(dev): map (source_gene, target_gene[, sign]) -> edge_index (+ edge_attr).
        # Respect spec: directed, signed, confidence_levels / score_threshold.
        raise NotImplementedError
