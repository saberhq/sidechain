"""Load configs/data_sources.yaml and instantiate the enabled PriorSource objects.

Adding a source is a YAML edit; this resolves `loader:` names to classes. The model
never imports a specific source — it consumes whatever the registry yields.
"""
from __future__ import annotations
from pathlib import Path
import yaml

from sidechain.priors.base import PriorSource
from sidechain.priors.trans_grn import TransGRNSource
from sidechain.priors.cis_sequence import CisSequenceSource
from sidechain.priors.posttx_mirna import MiRNATargetSource
from sidechain.priors.posttx_rbp import RBPBindingSource

# name in YAML -> class. Register a new loader here (one line).
LOADERS: dict[str, type[PriorSource]] = {
    "TransGRNSource": TransGRNSource,
    "CisSequenceSource": CisSequenceSource,
    "MiRNATargetSource": MiRNATargetSource,
    "RBPBindingSource": RBPBindingSource,
}


def load_registry(config_path: str | Path, gene_index: dict[str, int]) -> list[PriorSource]:
    """Return instantiated, ENABLED PriorSource objects from the YAML registry."""
    cfg = yaml.safe_load(Path(config_path).read_text())
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
            raise KeyError(f"Unknown loader '{loader_name}' for source '{spec.get('name')}'. "
                           f"Register it in registry.LOADERS.")
        out.append(cls(spec=spec, gene_index=gene_index))
    return out
