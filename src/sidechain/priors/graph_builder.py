"""Assemble ONE multi-relational sparse graph + node-feature block from all enabled
sources. Each 'edge' source contributes a distinct relation (edge type); each
'node_feature' source contributes a concatenated feature block. Add/remove sources
freely — this function does not know or care which biology is present.

Returns a torch_geometric HeteroData (or a plain dict if PyG unavailable).
"""
from __future__ import annotations
import numpy as np

from sidechain.priors.base import PriorSource, PriorArtifact


def build_prior_graph(sources: list[PriorSource], n_genes: int):
    node_blocks: list[np.ndarray] = []
    edges: dict[str, PriorArtifact] = {}

    for src in sources:
        art = src.build()
        if art.kind == "node_feature":
            node_blocks.append(art.features)
        else:  # edge
            key = f"{art.layer}:{art.relation}:{src.name}"
            edges[key] = art

    node_features = (
        np.concatenate(node_blocks, axis=1) if node_blocks else np.zeros((n_genes, 0))
    )

    # TODO(dev): pack into torch_geometric.data.HeteroData:
    #   data['gene'].x = torch.tensor(node_features)
    #   for key, art in edges.items():
    #       data['gene', art.relation, 'gene'].edge_index = torch.tensor(art.edge_index)
    #       ... edge_attr, per relation
    return {"node_features": node_features, "edges": edges}
