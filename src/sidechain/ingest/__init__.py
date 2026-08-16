"""External dataset ingestion.

One shape enters Sidechain (`HarmonizedDataset`), and everything upstream of it
lives here: the metadata gate that decides whether a corpus may be fetched at
all, the checks that refuse two specific ways of being confidently wrong, and
the adapters that map a source onto the contract.
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
