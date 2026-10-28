"""External dataset ingestion: the metadata gate (`provenance`), the ingest checks (`checks`) and the contract.

Notes: private/code-notes/src/sidechain/ingest/__init__.py.md
"""
from sidechain.ingest.checks import (
    RAW_COUNTS,
    TRANSFORMED,
    control_mask,
    counts_state,
    require_raw_counts,
    to_cp10k,
)
from sidechain.ingest.contract import (
    HarmonizedDataset,
    Provenance,
    harmonize,
    qc_report,
)
from sidechain.ingest.provenance import (
    GateError,
    HostRecord,
    RemoteFile,
    gate,
    probe_zenodo,
    to_provenance,
    write_provenance,
)

__all__ = [
    "RAW_COUNTS",
    "TRANSFORMED",
    "GateError",
    "HarmonizedDataset",
    "HostRecord",
    "Provenance",
    "RemoteFile",
    "control_mask",
    "counts_state",
    "gate",
    "harmonize",
    "probe_zenodo",
    "qc_report",
    "require_raw_counts",
    "to_cp10k",
    "to_provenance",
    "write_provenance",
]
