"""The prior contract: a `PriorSource` subclass implements `fetch()` and `build()`, returning a `PriorArtifact`.

    class MySource(PriorSource):
        def fetch(self): ...                       # download or load the raw source, idempotently
        def build(self):
            ei, attr = self.to_edge_index(srcs, dsts, weights)
            return PriorArtifact(kind="edge", relation=self.relation, layer=self.layer, edge_index=ei, edge_attr=attr)
    art = MySource(spec, gene_index).build_checked()

Reads the source's registry block (`spec`) and the gene index; writes nothing. Notes: private/code-notes/src/sidechain/priors/base.py.md
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Layer = Literal["trans", "cis", "posttx", "epigenomic", "phenotype"]
Kind = Literal["node_feature", "edge"]


@dataclass
class PriorArtifact:
    """Normalized output every source returns, aligned to gene_index.

    node_feature: features (n_genes, dim), zero rows for genes without data. edge: edge_index (2, n_edges) COO and
    an optional edge_attr (n_edges, n_attr), never a dense gene x gene matrix. With `src_names` set the edges are
    bipartite: row 0 indexes `src_names` (source nodes off the gene axis), row 1 `gene_index`.
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
        self.gene_index = gene_index          # gene ID -> position
        self.name = spec.get("name", self.__class__.__name__)
        self.layer: Layer = spec.get("layer")
        self.relation = spec.get("relation", "")

        # `kind` is declared in YAML, not inferred from build(): the registry needs the shape before any build.
        kind = spec.get("kind")
        if kind not in ("node_feature", "edge"):
            raise ValueError(
                f"Source {self.name!r} must declare kind: node_feature | edge in "
                f"configs/data_sources.yaml (got {kind!r})."
            )
        self.kind: Kind = kind

    @property
    def edge_type_key(self) -> str:
        """Unique edge-type key, `layer:relation:name` (two sources may declare one relation)."""
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
        """Map a standalone gene-ID list to positions, dropping unknown IDs; edges use `to_edge_index`."""
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
        """(2, n_edges) COO from paired gene IDs; returns (edge_index, edge_attr). Every edge source uses it.

        One mask keeps an edge when both ends are in the gene index, and filters `edge_attr` alike.
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
        """(2, n_edges) COO whose row 0 indexes `src_index` and row 1 the gene index.

        One mask keeps an edge when its source is in `src_index` and its destination in `gene_index`, and filters
        `edge_attr` alike.
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
