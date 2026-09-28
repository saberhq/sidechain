"""The extensibility contract. Every biological prior implements PriorSource and
returns a PriorArtifact aligned to the master gene space. New biology plugs in
here WITHOUT touching the model or the graph builder.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Layer = Literal["trans", "cis", "posttx", "epigenomic"]
Kind = Literal["node_feature", "edge"]


@dataclass
class PriorArtifact:
    """Normalized output every source returns, aligned to gene_index.

    node_feature:  features (n_genes, dim). Genes without data -> zero rows.
    edge:          edge_index (2, n_edges) COO + optional edge_attr (n_edges, n_attr).
                   NEVER a dense n_genes x n_genes matrix.

    **Bipartite edges** (`src_names` set): the edge's SOURCE nodes are not genes on the
    master axis -- a miRNA family (TargetScan, T102), an RBP that is also a gene but is
    scored as a regulator -- so row 0 of `edge_index` indexes `src_names` and only row 1
    indexes `gene_index`. `build_checked` checks each row against its own vocabulary. A
    consumer that wants gene-gene edges ("share a regulator") derives them from this form
    rather than the source emitting a dense family x gene block.
    """
    kind: Kind
    relation: str
    layer: Layer
    features: np.ndarray | None = None
    edge_index: np.ndarray | None = None
    edge_attr: np.ndarray | None = None
    directed: bool = True
    meta: dict = field(default_factory=dict)
    src_names: list[str] | None = None


class PriorSource(ABC):
    """Subclass this, implement fetch() + build(). That's the whole extension point."""

    def __init__(self, spec: dict, gene_index: dict[str, int]):
        self.spec = spec
        self.gene_index = gene_index          # ensembl_gene_id -> position
        self.name = spec.get("name", self.__class__.__name__)
        self.layer: Layer = spec.get("layer")
        self.relation = spec.get("relation", "")

        # `kind` is declared in YAML, not inferred from build(): the registry has
        # to know the graph shape before building anything, and building
        # nt_utr_embed means a GPU pass over every 3'UTR.
        kind = spec.get("kind")
        if kind not in ("node_feature", "edge"):
            raise ValueError(
                f"Source {self.name!r} must declare kind: node_feature | edge in "
                f"configs/data_sources.yaml (got {kind!r})."
            )
        self.kind: Kind = kind

    @property
    def edge_type_key(self) -> str:
        """Unique edge-type key. `relation` alone collides -- mirbind2/targetscan
        both declare `mirna_target`, postar3/encode_eclip both `rbp_binding` --
        and the registry explicitly invites enabling those pairs together."""
        return f"{self.layer}:{self.relation}:{self.name}"

    @abstractmethod
    def fetch(self) -> None:
        """Download / load the raw source and cache it (lamindb). Idempotent."""

    @abstractmethod
    def build(self) -> PriorArtifact:
        """Return a PriorArtifact aligned to self.gene_index. Preserve gene order;
        map unknown genes to nothing (no edge) or zero rows (node_feature)."""

    def build_checked(self) -> PriorArtifact:
        """build() plus the contract checks: declared kind matches what came back,
        edges are COO and in range, features are one row per gene."""
        art = self.build()
        if art.kind != self.kind:
            raise ValueError(
                f"{self.name}: declared kind={self.kind!r} but build() returned "
                f"kind={art.kind!r}. Fix the YAML or the loader."
            )
        n_genes = len(self.gene_index)
        if art.kind == "edge":
            ei = art.edge_index
            if ei is None:
                raise ValueError(f"{self.name}: kind='edge' but edge_index is None.")
            if ei.ndim != 2 or ei.shape[0] != 2:
                raise ValueError(
                    f"{self.name}: edge_index must be (2, n_edges) COO, got {ei.shape}."
                )
            if not np.issubdtype(ei.dtype, np.integer):
                raise ValueError(f"{self.name}: edge_index must be integer, got {ei.dtype}.")
            if art.src_names is not None:
                n_src = len(art.src_names)
                if ei.size and (ei[0].min() < 0 or ei[0].max() >= n_src):
                    raise ValueError(
                        f"{self.name}: bipartite edge_index row 0 out of range for "
                        f"{n_src} source nodes (min={ei[0].min()}, max={ei[0].max()})."
                    )
                if ei.size and (ei[1].min() < 0 or ei[1].max() >= n_genes):
                    raise ValueError(
                        f"{self.name}: bipartite edge_index row 1 out of range for "
                        f"{n_genes} genes (min={ei[1].min()}, max={ei[1].max()})."
                    )
            elif ei.size and (ei.min() < 0 or ei.max() >= n_genes):
                raise ValueError(
                    f"{self.name}: edge_index out of range for {n_genes} genes "
                    f"(min={ei.min()}, max={ei.max()})."
                )
            if art.edge_attr is not None and art.edge_attr.shape[0] != ei.shape[1]:
                raise ValueError(
                    f"{self.name}: edge_attr has {art.edge_attr.shape[0]} rows but "
                    f"there are {ei.shape[1]} edges."
                )
        else:
            if art.features is None:
                raise ValueError(f"{self.name}: kind='node_feature' but features is None.")
            if art.features.shape[0] != n_genes:
                raise ValueError(
                    f"{self.name}: features must have one row per gene "
                    f"({art.features.shape[0]} != {n_genes})."
                )
        return art

    # -- helpers shared by all sources --
    def enabled(self) -> bool:
        return bool(self.spec.get("enabled", True))

    def to_positions(self, gene_ids) -> np.ndarray:
        """Map gene IDs -> integer positions, dropping any not in the master space.

        Use this ONLY for a standalone list of genes (e.g. which nodes a
        node_feature source covers). For edges, use `to_edge_index`: calling this
        separately on sources and destinations drops unknown IDs independently
        per call, which silently mis-pairs the survivors.
        """
        idx = self.gene_index
        return np.array([idx[g] for g in gene_ids if g in idx], dtype=np.int64)

    def to_edge_index(
        self,
        srcs,
        dsts,
        edge_attr: np.ndarray | None = None,
        *,
        drop_self_loops: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Build a (2, n_edges) COO edge_index from paired gene-ID sequences.

        An edge survives only if BOTH endpoints are in the master gene space, and
        the same mask is applied to `edge_attr` -- so rows stay aligned to the
        edges they describe. This is the pair-aware counterpart to
        `to_positions`, and every edge source must use it.

        Returns (edge_index, edge_attr). Never allocates a dense gene x gene
        matrix.
        """
        srcs = list(srcs)
        dsts = list(dsts)
        if len(srcs) != len(dsts):
            raise ValueError(
                f"{self.name}: srcs and dsts must be the same length "
                f"({len(srcs)} != {len(dsts)})"
            )

        attr = None if edge_attr is None else np.asarray(edge_attr)
        if attr is not None and attr.shape[0] != len(srcs):
            raise ValueError(
                f"{self.name}: edge_attr has {attr.shape[0]} rows but there are "
                f"{len(srcs)} candidate edges."
            )

        idx = self.gene_index
        keep = np.fromiter(
            ((str(s) in idx and str(d) in idx) for s, d in zip(srcs, dsts)),
            dtype=bool,
            count=len(srcs),
        )
        s_pos = np.array([idx[str(s)] for s, k in zip(srcs, keep) if k], dtype=np.int64)
        d_pos = np.array([idx[str(d)] for d, k in zip(dsts, keep) if k], dtype=np.int64)

        if drop_self_loops and s_pos.size:
            not_loop = s_pos != d_pos
            s_pos, d_pos = s_pos[not_loop], d_pos[not_loop]
            if attr is not None:
                attr = attr[keep][not_loop]
        elif attr is not None:
            attr = attr[keep]

        edge_index = np.vstack([s_pos, d_pos]).astype(np.int64) if s_pos.size else np.zeros(
            (2, 0), dtype=np.int64
        )
        return edge_index, attr

    def to_bipartite_edge_index(
        self,
        srcs,
        dsts,
        src_index: dict[str, int],
        edge_attr: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """(2, n_edges) COO whose row 0 indexes `src_index` and row 1 the master gene space.

        The pair-aware rule of `to_edge_index` for a source vocabulary that is NOT the
        gene axis (a miRNA family has no position in `gene_index`): ONE mask keeps an
        edge only when its source is in `src_index` and its destination in `gene_index`,
        and the same mask is applied to `edge_attr`. Never two `to_positions` calls.
        """
        srcs = list(srcs)
        dsts = list(dsts)
        if len(srcs) != len(dsts):
            raise ValueError(
                f"{self.name}: srcs and dsts must be the same length "
                f"({len(srcs)} != {len(dsts)})"
            )
        attr = None if edge_attr is None else np.asarray(edge_attr)
        if attr is not None and attr.shape[0] != len(srcs):
            raise ValueError(
                f"{self.name}: edge_attr has {attr.shape[0]} rows but there are "
                f"{len(srcs)} candidate edges."
            )
        idx = self.gene_index
        keep = np.fromiter(
            ((str(s) in src_index and str(d) in idx) for s, d in zip(srcs, dsts)),
            dtype=bool,
            count=len(srcs),
        )
        s_pos = np.array([src_index[str(s)] for s, k in zip(srcs, keep) if k], dtype=np.int64)
        d_pos = np.array([idx[str(d)] for d, k in zip(dsts, keep) if k], dtype=np.int64)
        if attr is not None:
            attr = attr[keep]
        edge_index = np.vstack([s_pos, d_pos]).astype(np.int64) if s_pos.size else np.zeros(
            (2, 0), dtype=np.int64
        )
        return edge_index, attr
