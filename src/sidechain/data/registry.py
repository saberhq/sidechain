"""Load configs/data_sources.yaml and instantiate the enabled PriorSource objects.

Adding a source is a YAML edit; this resolves `loader:` names to classes. The model
never imports a specific source -- it consumes whatever the registry yields.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from sidechain.priors.base import PriorSource
from sidechain.priors.cis_sequence import CisSequenceSource
from sidechain.priors.posttx_mirna import MiRNATargetSource
from sidechain.priors.posttx_rbp import RBPBindingSource
from sidechain.priors.trans_grn import TransGRNSource
from sidechain.utils.paths import resolve_config

logger = logging.getLogger(__name__)

# name in YAML -> class. Register a new loader here (one line).
LOADERS: dict[str, type[PriorSource]] = {
    "TransGRNSource": TransGRNSource,
    "CisSequenceSource": CisSequenceSource,
    "MiRNATargetSource": MiRNATargetSource,
    "RBPBindingSource": RBPBindingSource,
}


@dataclass(frozen=True)
class MasterGeneSpace:
    """The declared ID space every source must align to."""

    reference: str
    version: str
    id_type: str


@dataclass(frozen=True)
class Registry:
    """Everything the YAML declares, resolved."""

    sources: list[PriorSource]
    master_gene_space: MasterGeneSpace

    @property
    def edge_sources(self) -> list[PriorSource]:
        return [s for s in self.sources if s.kind == "edge"]

    @property
    def node_feature_sources(self) -> list[PriorSource]:
        return [s for s in self.sources if s.kind == "node_feature"]

    def describe(self) -> str:
        """Graph shape WITHOUT building anything -- possible only because `kind`
        is declared in YAML rather than discovered by calling build()."""
        mgs = self.master_gene_space
        lines = [
            f"master gene space: {mgs.reference} {mgs.version} ({mgs.id_type})",
            (
                f"enabled sources: {len(self.sources)} "
                f"({len(self.edge_sources)} edge, "
                f"{len(self.node_feature_sources)} node_feature)"
            ),
        ]
        for s in self.sources:
            tag = s.edge_type_key if s.kind == "edge" else f"{s.layer}:{s.relation}"
            lines.append(f"  {s.kind:13s} {tag}")
        return "\n".join(lines)


def load_registry(
    config_path: str | Path,
    gene_index: dict[str, int],
    *,
    id_type: str | None = None,
) -> Registry:
    """Return the enabled PriorSource objects plus the declared master gene space.

    `id_type` is the ID space the caller actually built `gene_index` in. When
    given it must match the registry's declared `master_gene_space.id_type` --
    otherwise every source silently resolves zero genes.
    """
    cfg = yaml.safe_load(resolve_config(config_path).read_text())

    mgs_raw = cfg.get("master_gene_space")
    if not mgs_raw:
        raise KeyError(
            "configs/data_sources.yaml has no master_gene_space block; every source "
            "aligns to it, so it is required."
        )
    mgs = MasterGeneSpace(
        reference=mgs_raw.get("reference", "unknown"),
        version=str(mgs_raw.get("version", "unknown")),
        id_type=mgs_raw["id_type"],
    )

    if id_type is not None and id_type != mgs.id_type:
        raise ValueError(
            f"gene_index was built in {id_type!r} but the registry declares "
            f"master_gene_space.id_type={mgs.id_type!r}. Every source would resolve "
            f"zero genes. Rebuild the index in {mgs.id_type!r} or update the YAML."
        )

    out: list[PriorSource] = []
    for spec in cfg.get("sources", []):
        # Skip disabled sources BEFORE resolving the loader, so a shelved block
        # may name a loader that isn't written yet (`enabled: false` is the
        # documented way to sketch a source without implementing it).
        if not spec.get("enabled", True):
            continue
        loader_name = spec.get("loader")
        cls = LOADERS.get(loader_name)
        if cls is None:
            raise KeyError(
                f"Unknown loader '{loader_name}' for source '{spec.get('name')}'. "
                f"Register it in registry.LOADERS."
            )
        out.append(cls(spec=spec, gene_index=gene_index))

    registry = Registry(sources=out, master_gene_space=mgs)
    logger.info("Loaded registry:\n%s", registry.describe())
    return registry
