"""Post-transcriptional layer: RBP -> target binding edges (Saber's edge), and the RBP
knockdown responses that test them across lines.

Two registry blocks, two classes, one fetch:

* ``RBPBindingSource`` -- CLIP-based RBP binding sites (``encode_eclip``: ENCODE eCLIP
  narrowPeak files per line; ``postar3``; ``attract`` as a motif fallback), to become
  directed edges onto the bound target gene once filtered to 3'UTRs (``region_filter``).
* ``RBPKnockdownSource`` -- ``encode_rbp_knockdown``: the same RBP knocked down in K562
  and in HepG2 (ENCODE shRNA / CRISPR RNA-seq, RSEM gene quantifications against a paired
  non-targeting control). A measurement corpus, not a model input: T102 reads it to ask
  whether a receiving line's own site set predicts where a knockdown's response fails to
  carry between lines (private ``plans/2026-10-16-t102-expressed-factor-test.md``).

``fetch()`` on either is ``posttx_mirna.fetch_gated_files`` on the block, the shared loop
every file-hosted prior uses (probe -> gate -> PROVENANCE.json -> bytes, ADR 0003). Both
ENCODE blocks list their files in a checked-in manifest (``files_manifest``, hundreds of
per-file urls generated from the ENCODE API), each with the md5 the API states, so the
gate verifies them without ``allow_missing_checksum``. ``build()`` is not written for
either: the first use is a reading, and an edge layer or a feature comes after it.
"""
from __future__ import annotations

from pathlib import Path

from sidechain.priors.base import PriorArtifact, PriorSource
from sidechain.priors.posttx_mirna import DATA_ROOT, REGISTRY, fetch_gated_files


class _GatedFilesSource(PriorSource):
    """A source whose raw bytes come through the gate from its own registry block."""

    def __init__(self, spec: dict, gene_index: dict[str, int], root: Path = DATA_ROOT,
                 registry: str | Path = REGISTRY):
        super().__init__(spec, gene_index)
        self.root = Path(root).expanduser()
        self.registry = registry

    @property
    def dest(self) -> Path:
        return self.root / self.spec["dest"]

    def fetch(self, *, refresh: bool = False, progress: bool = False) -> Path:
        """Gate, record provenance, then download the block's files. Idempotent."""
        return fetch_gated_files(self.spec, self.root, refresh=refresh, progress=progress,
                                 registry=self.registry)


class RBPBindingSource(_GatedFilesSource):
    def build(self) -> PriorArtifact:
        # (rbp_gene, target_gene, n_sites / clip_score) -> directed edges, 3'UTR-filtered:
        # written when a reading earns it (T102 direction 4b; the peaks are per line).
        raise NotImplementedError(f"{self.name}: build() is not written; fetch() brings the peaks through the gate")


class RBPKnockdownSource(_GatedFilesSource):
    def build(self) -> PriorArtifact:
        # A per-gene feature from knockdown responses is not what this block is for yet;
        # the per-(target, line) fold changes are derived by T102's reading scripts.
        raise NotImplementedError(f"{self.name}: build() is not written; fetch() brings the quantifications through the gate")
