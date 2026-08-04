"""Assemble ONE multi-relational sparse graph + node-feature block from all enabled
sources. Each 'edge' source contributes a distinct relation (edge type); each
'node_feature' source contributes a concatenated feature block. Add/remove
sources freely -- this function does not know or care which biology is present.

Returns a torch_geometric HeteroData when PyG is importable, else a plain dict
with the same content.

Edge types are keyed on `layer:relation:name`, NOT on `relation`. Keying on
relation alone silently drops sources: mirbind2 and targetscan both declare
`mirna_target`, postar3 and encode_eclip both declare `rbp_binding`, and
data_sources.yaml explicitly invites enabling each pair together to ensemble
them. The final edge-type count is asserted against the number of enabled edge
sources so a future collision fails loudly instead of quietly halving the graph.
"""
from __future__ import annotations

import logging

import numpy as np

from sidechain.priors.base import PriorArtifact, PriorSource

logger = logging.getLogger(__name__)


def build_prior_graph(sources: list[PriorSource], n_genes: int, *, to_pyg: bool = True):
    node_blocks: list[np.ndarray] = []
    node_source_names: list[str] = []
    edges: dict[str, PriorArtifact] = {}

    n_edge_sources = 0
    for src in sources:
        art = src.build_checked()
        if art.kind == "node_feature":
            node_blocks.append(art.features)
            node_source_names.append(src.name)
        else:
            n_edge_sources += 1
            key = src.edge_type_key
            if key in edges:
                raise ValueError(
                    f"Duplicate edge-type key {key!r}. Two sources share "
                    f"layer+relation+name; rename one in configs/data_sources.yaml."
                )
            edges[key] = art

    if len(edges) != n_edge_sources:
        raise AssertionError(
            f"Edge-type count {len(edges)} != enabled edge sources {n_edge_sources}. "
            f"A source was dropped by a key collision."
        )

    node_features = (
        np.concatenate(node_blocks, axis=1) if node_blocks else np.zeros((n_genes, 0))
    )
    if node_blocks and node_features.shape[0] != n_genes:
        raise ValueError(
            f"node_features has {node_features.shape[0]} rows, expected {n_genes}."
        )

    logger.info(
        "prior graph: %d genes | %d edge types | %d node-feature dims from %s",
        n_genes,
        len(edges),
        node_features.shape[1],
        node_source_names or "no sources",
    )
    for key, art in edges.items():
        logger.info("  %-40s %d edges", key, art.edge_index.shape[1])

    plain = {
        "node_features": node_features,
        "node_sources": node_source_names,
        "edges": edges,
        "n_genes": n_genes,
    }
    if not to_pyg:
        return plain

    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError:
        logger.info("torch_geometric unavailable; returning plain dict.")
        return plain

    data = HeteroData()
    data["gene"].x = torch.from_numpy(np.ascontiguousarray(node_features)).float()
    data["gene"].num_nodes = n_genes
    for key, art in edges.items():
        # The relation slot of the PyG triple carries the disambiguated key, so
        # two sources of the same biological relation stay separate edge types.
        rel = ("gene", key, "gene")
        data[rel].edge_index = torch.from_numpy(np.ascontiguousarray(art.edge_index)).long()
        if art.edge_attr is not None:
            data[rel].edge_attr = torch.from_numpy(
                np.ascontiguousarray(art.edge_attr)
            ).float()
    return data
