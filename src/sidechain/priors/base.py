"""The extensibility contract. Every biological prior implements PriorSource and
returns a PriorArtifact aligned to the master gene space. New biology plugs in
here WITHOUT touching the model or the graph builder.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np

Layer = Literal["trans", "cis", "posttx", "epigenomic"]
Kind = Literal["node_feature", "edge"]


@dataclass
class PriorArtifact:
    """Normalized output every source returns, aligned to gene_index.

    node_feature:  features (n_genes, dim). Genes without data -> zero rows.
    edge:          edge_index (2, n_edges) COO + optional edge_attr (n_edges, n_attr).
                   NEVER a dense n_genes x n_genes matrix.
    """
    kind: Kind
    relation: str
    layer: Layer
    features: Optional[np.ndarray] = None
    edge_index: Optional[np.ndarray] = None
    edge_attr: Optional[np.ndarray] = None
    directed: bool = True
    meta: dict = field(default_factory=dict)


class PriorSource(ABC):
    """Subclass this, implement fetch() + build(). That's the whole extension point."""

    def __init__(self, spec: dict, gene_index: dict[str, int]):
        self.spec = spec
        self.gene_index = gene_index          # ensembl_gene_id -> position
        self.name = spec.get("name", self.__class__.__name__)
        self.layer: Layer = spec.get("layer")
        self.relation = spec.get("relation", "")

    @abstractmethod
    def fetch(self) -> None:
        """Download / load the raw source and cache it (lamindb). Idempotent."""

    @abstractmethod
    def build(self) -> PriorArtifact:
        """Return a PriorArtifact aligned to self.gene_index. Preserve gene order;
        map unknown genes to nothing (no edge) or zero rows (node_feature)."""

    # -- helpers shared by all sources --
    def enabled(self) -> bool:
        return bool(self.spec.get("enabled", True))

    def to_positions(self, gene_ids) -> np.ndarray:
        """Map gene IDs -> integer positions, dropping any not in the master space."""
        idx = self.gene_index
        return np.array([idx[g] for g in gene_ids if g in idx], dtype=np.int64)
