"""Post-transcriptional layer: a gene's polyadenylation layout as a per-gene node feature.

The third posttx table beside ``posttx_mirna`` (the 3'UTR grammar) and ``posttx_stability``
(how long the transcript lasts). This one reads WHERE the transcript can end: how far the
representative transcript's last splice acceptor sits from the first supported poly(A) site,
how many supported sites the gene's 3' window holds, how short and how long its 3'UTR can be,
and how much of the pooled signal the proximal site takes. The acceptor-to-proximal-site
distance is the sequence cue Borzoi Prime found strongest for differential polyadenylation
(``research/reading/linder-2025-borzoi-prime.md`` measurement 6, private); the plan that
specifies every column is ``plans/2026-10-15-t102-polyasite.md`` (private).

**Two registry blocks, one class**, as ``MRNAStabilitySource`` does it:

* ``polyasite_2`` -- the PolyASite 2.0 human atlas TSV (GRCh38, Ensembl 96; 569,005 clusters
  from 221 libraries), ``route: stream`` because its terms are stated nowhere (``UNSTATED``,
  recorded with both encumbrance flags). Its ``route`` decides how the bytes are read: streamed,
  ``fetch()`` runs the gate, writes PROVENANCE.json and ``gene_table()`` reads the TSV once over
  the network through ``posttx_mirna.stream_gated_file``, keeping only the per-gene parquet;
  downloaded (ADR 0003 as amended 2026-10-15: a cited source with unstated terms may be held),
  the file lands under ``dest`` and is read from there. The LINEAGE entry says which.
* ``gencode_v47_basic`` -- the GENCODE v47 basic annotation GTF, downloaded through the same
  gate (``fetch_gated_files``) and checked against the release's published md5 after landing,
  because ``probe_https`` cannot read a sidecar. ``transcript_table()`` parses it once into one
  row per basic transcript: its last exon, its stop codon, its tags.

**What the per-gene build does** (``gene_table``). One representative transcript per gene, in
this order, recorded in ``pa_tx_source``: TargetScan's representative (``Gene_info``, a GENCODE
v19 id looked up by stem in v47 and accepted when the gene stem matches), else ``MANE_Select``,
else ``Ensembl_canonical``, else the longest-CDS protein-coding transcript, else the longest
transcript. A strand-aware window runs from the representative's last splice acceptor (A) to the
farthest 3' end among the gene's transcripts that share that acceptor, plus ``DOWNSTREAM_NT``
(1,000 nt, PolyASite's own DS class); it starts after the stop codon when the stop codon lies in
that exon. A cluster belongs to the gene when its representative site falls in the window on the
same strand -- never by the atlas's own gene column, which is Ensembl 96's call and is kept only
for the annotation-agreement check. The default support filter keeps clusters with at least
``MIN_PROTOCOLS`` protocols and at least ``MIN_TPM_FRAC`` of the window's summed mean TPM; the
unfiltered counts are written beside the filtered ones (``_all``).

**Coordinates.** The atlas's ``rep`` column is read as a 1-based genomic position, GENCODE's
GTF is 1-based inclusive; a one-base disagreement is below the resolution of any column here
(distances in the hundreds to thousands of nt) and is recorded in LINEAGE.json rather than
argued. The atlas spells chromosomes without ``chr`` and the mitochondrion ``MT``; the GTF
spells ``chr10`` and ``chrM``. Positions are handled in TRANSCRIPT orientation (negated on the
minus strand), so every rule is written once.

**Per-line usage columns are written and used nowhere** (Saber, 2026-10-15): the TSV's per-library
columns give the proximal site's usage in the atlas's unperturbed HeLa, HEK293 and K562 libraries;
they are pooled-atlas measurements of three named lines, not features of a context, and they
wait on the calibration call. Which libraries went into each is in LINEAGE.json.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sidechain.priors.base import PriorArtifact, PriorSource
from sidechain.priors.posttx_mirna import (
    DATA_ROOT,
    HUMAN,
    REGISTRY,
    MiRNATargetSource,
    fetch_gated_files,
    sha256_of,
    spec_from_registry,
    stream_gated_file,
)

#: The registry blocks this loader serves, and the files each one names.
POLYASITE_BLOCK = "polyasite_2"
GENCODE_BLOCK = "gencode_v47_basic"
TARGETSCAN_BLOCK = "targetscan"

ATLAS_FILE = "atlas.clusters.2.0.GRCh38.96.tsv.gz"
GTF_FILE = "gencode.v47.basic.annotation.gtf.gz"
GENE_INFO_FILE = "Gene_info.txt.zip"

#: GENCODE's own MD5SUMS entry for the basic GTF (release 47, read 2026-10-15). The https probe
#: reads no sidecar, so the fetch checks the landed file against this and refuses on a mismatch.
GENCODE_MD5 = "823c2d44a2268d2c79a11986ffa29028"

#: The window and the support filter (plans/2026-10-15-t102-polyasite.md § 3, private).
DOWNSTREAM_NT = 1000          # PolyASite's DS class: within 1,000 nt downstream of a terminal exon
MIN_PROTOCOLS = 2             # a cluster supported by fewer protocols is not read
MIN_TPM_FRAC = 0.05           # ... nor one under this share of the window's summed mean TPM
CANONICAL_PAS = ("AATAAA", "ATTAAA")   # the two canonical hexamers, as the atlas spells them (DNA)

#: The thirteen fixed columns of the "Atlas with Samples" TSV, in order; every later column is one
#: library, headed SAMPLE_ID|PROTOCOL|SOURCE|TITLE|TREATMENT.
ATLAS_COLUMNS = ("chrom", "chromStart", "chromEnd", "name", "score", "strand", "rep",
                 "frac_samples", "nr_prots", "annotation", "gene_name", "gene_id", "repSite_signals")

#: Which of the atlas's libraries count as a named line's UNPERTURBED measurement: the SOURCE
#: field names the line, and the TREATMENT field is empty, none, NA, or a control. The match is
#: written into LINEAGE.json library by library so the grouping is auditable, not asserted.
ATLAS_LINES: dict[str, tuple[str, ...]] = {
    "hela": ("hela",),
    "hek293": ("hek293", "293t"),
    "k562": ("k562",),
}
_UNPERTURBED_TREATMENT = re.compile(r"^(?:|none|na|n/a|wt|wild ?type)$|(?:^|[^a-z])(?:ctrl|control|mock)(?:$|[^a-z])")

#: One row per basic transcript (`transcript_table`).
TRANSCRIPT_COLUMNS = {
    "transcript_id": "Ensembl transcript id, version stripped",
    "transcript_version": "the version the GTF carries (int)",
    "gene_id": "Ensembl gene id, version stripped",
    "symbol": "GENCODE's gene_name (2024-era symbols)",
    "gene_type": "GENCODE gene_type",
    "transcript_type": "GENCODE transcript_type",
    "chrom": "chromosome as the GTF spells it (chr1 ... chrX, chrY, chrM)",
    "strand": "+ or -",
    "tx_start": "transcript start (1-based, inclusive, genomic)",
    "tx_end": "transcript end (1-based, inclusive, genomic)",
    "gene_start": "the gene feature's start (genomic), for the intronic window",
    "gene_end": "the gene feature's end (genomic)",
    "n_exons": "exons on the transcript",
    "exonic_len": "sum of exon widths",
    "last_exon_start": "genomic start of the 3'-most exon",
    "last_exon_end": "genomic end of the 3'-most exon",
    "acceptor": "the 5' end of the 3'-most exon in TRANSCRIPT orientation (genomic position; "
                "NaN on a single-exon transcript, which has no last splice acceptor)",
    "stop_3p": "the 3'-most base of the stop codon (genomic position; NaN without a CDS)",
    "stop_in_last_exon": "the stop codon lies in the 3'-most exon",
    "exonic_stop_to_acceptor": "exonic bases after the stop codon and before the last exon "
                               "(0 when the stop is in the last exon or there is none)",
    "cds_len": "sum of CDS widths (0 without a CDS)",
    "utr3_len": "exonic bases after the stop codon to the annotated transcript end (NaN without a CDS)",
    "mane_select": "tag MANE_Select",
    "ensembl_canonical": "tag Ensembl_canonical",
}

#: One row per gene (`gene_table`).
GENE_COLUMNS = {
    "gene_id": "Ensembl gene id, version stripped (GENCODE v47)",
    "symbol": "GENCODE v47 gene_name",
    "transcript_id": "the representative transcript the window is built on",
    "pa_tx_source": "which rule chose it: targetscan | mane_select | ensembl_canonical | "
                    "longest_cds | longest",
    "pa_tx_version_changed": "TargetScan's v19 transcript version differs from v47's (the exon "
                             "ends may have moved); False when the rule was not targetscan",
    "pa_single_exon": "the representative has one exon (no last splice acceptor)",
    "pa_te_len": "length of the representative's last exon, nt",
    "pa_coding": "the representative carries a CDS with an annotated stop codon (a cds_end_NF transcript reads as non-coding here)",
    "pa_window_len": "length of the cluster window, nt",
    "pa_window_tpm_sum": "summed mean TPM of every cluster in the window (the filter's denominator)",
    "pa_n_sites_all": "clusters in the window, unfiltered",
    "pa_n_sites": "clusters in the window passing the support filter",
    "pa_has_apa": "pa_n_sites >= 2",
    "pa_acc_to_prox_nt": "representative site of the proximal (most 5') passing cluster minus the "
                         "last splice acceptor, nt (the Borzoi Prime feature); NaN for a "
                         "single-exon gene or no passing cluster",
    "pa_acc_to_prox_nt_all": "the same without the support filter",
    "pa_log10_acc_to_prox": "log10(pa_acc_to_prox_nt + 1)",
    "pa_stop_to_prox_nt": "3'UTR length to the proximal passing site, nt (NaN without a CDS)",
    "pa_stop_to_dist_nt": "3'UTR length to the distal (most 3') passing site, nt",
    "pa_span_nt": "distal minus proximal site, nt",
    "pa_prox_usage": "the proximal cluster's mean TPM over the sum of the passing clusters' mean "
                     "TPM (pooled over the atlas's 221 libraries: HeLa- and HEK-heavy)",
    "pa_prox_has_canonical_pas": "AATAAA or ATTAAA among the proximal cluster's signals in -60..+10",
    "pa_prox_frac_samples": "share of the atlas's samples supporting the proximal cluster",
    "pa_prox_nr_prots": "protocols supporting the proximal cluster",
    "pa_n_intronic": "same-strand IN clusters passing the protocol filter inside the gene span "
                     "and outside the window",
    "pa_window_te_ds_share": "share of the window's unfiltered clusters the atlas itself labels "
                             "TE or DS (the v47-versus-Ensembl-96 agreement)",
}


def _code_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:                                          # noqa: BLE001
        return "unknown"


def _squash(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def atlas_chrom(chrom: str) -> str:
    """The atlas's chromosome spelling as the GTF spells it (10 -> chr10, MT -> chrM)."""
    c = str(chrom).strip()
    if c.startswith("chr"):
        return c
    return "chrM" if c in ("MT", "M") else f"chr{c}"


def classify_library(header: str) -> tuple[str | None, str, str]:
    """A per-library TSV column header -> (line or None, source, treatment).

    The header is ``SAMPLE_ID|PROTOCOL|SOURCE|TITLE|TREATMENT``. A library counts for a line
    when SOURCE carries one of the line's tokens and TREATMENT is empty, none, NA or a control
    (``_UNPERTURBED_TREATMENT``); a knockdown, a transfection or a mutant counts for no line.
    """
    parts = header.split("|")
    if len(parts) < 5:
        return None, header, ""
    source, treatment = parts[2].strip(), parts[4].strip()
    squashed = _squash(source)
    line = next((ln for ln, toks in ATLAS_LINES.items() if any(t in squashed for t in toks)), None)
    if line is None or not _UNPERTURBED_TREATMENT.search(treatment.strip().lower()):
        return None, source, treatment
    return line, source, treatment


_ATTR = re.compile(r'(\S+)\s+"([^"]*)"')


def parse_gtf_attributes(text: str) -> tuple[dict[str, str], set[str]]:
    """GTF column 9 -> (first value per key, the set of `tag` values)."""
    attrs: dict[str, str] = {}
    tags: set[str] = set()
    for key, value in _ATTR.findall(text):
        if key == "tag":
            tags.add(value)
        else:
            attrs.setdefault(key, value)
    return attrs, tags


def _strip(s: str) -> str:
    return s.split(".")[0]


def _version(s: str) -> int:
    tail = s.rsplit(".", 1)
    return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else -1


class PolyASiteSource(PriorSource):
    """PolyASite 2.0 on GENCODE v47 as a per-gene node feature.

    Constructed like every PriorSource -- ``(spec, gene_index)`` -- but the table entry points
    need no gene index: ``fetch()``, ``transcript_table()`` and ``gene_table()`` work on the
    spec alone, so ``gene_index`` may be an empty dict for those. ``build()`` needs it.
    ``registry`` names the prior registry the sibling block is looked up in.
    """

    def __init__(self, spec: dict, gene_index: dict[str, int], root: Path = DATA_ROOT,
                 registry: str | Path = REGISTRY):
        super().__init__(spec, gene_index)
        self.root = Path(root).expanduser()
        self.registry = registry

    # ------------------------------------------------------------------ fetch --

    @property
    def dest(self) -> Path:
        return self.root / self.spec["dest"]

    @property
    def derived(self) -> Path:
        return self.root / self.spec["derived"]

    def _for(self, name: str) -> PolyASiteSource:
        if self.name == name:
            return self
        return PolyASiteSource(spec_from_registry(name, self.registry), self.gene_index,
                               root=self.root, registry=self.registry)

    def _targetscan(self) -> MiRNATargetSource:
        return MiRNATargetSource(spec_from_registry(TARGETSCAN_BLOCK, self.registry),
                                 self.gene_index, root=self.root)

    def fetch(self, *, refresh: bool = False, progress: bool = False) -> Path:
        """Gate and record provenance for THIS block; download only what may land.

        The atlas block follows its ``route``: as a stream the gate writes PROVENANCE.json into
        ``dest`` and nothing else, and the bytes are read later by ``gene_table``; as a download
        (ADR 0003 as amended 2026-10-15: a cited source with unstated terms may be held) the file
        lands under ``dest`` like any other. The GENCODE block is a download
        through ``fetch_gated_files``, then the landed GTF is checked against the release's
        published md5 (``GENCODE_MD5``); a mismatch is a refusal that keeps the file for a
        human to look at.
        """
        if self.name == POLYASITE_BLOCK and _is_stream(self.spec):
            return stream_gated_file(self.spec, self.root, refresh=refresh, progress=progress,
                                     registry=self.registry).dest
        dest = fetch_gated_files(self.spec, self.root, refresh=refresh, progress=progress,
                                 registry=self.registry)
        if self.name == GENCODE_BLOCK:
            path = dest / GTF_FILE
            got = _md5_of(path)
            if got != GENCODE_MD5:
                raise RuntimeError(f"{GTF_FILE}: md5 {got} is not the release's published "
                                   f"{GENCODE_MD5}; the file is kept at {path} for a look, and "
                                   "nothing is built from it")
        return dest

    # ----------------------------------------------------------------- lineage --

    def _file_input(self, dest: Path, name: str) -> dict:
        prov = dest / "PROVENANCE.json"
        selected = json.loads(prov.read_text()).get("selected", []) if prov.exists() else []
        f = next((x for x in selected if x.get("name") == name), {})
        return {"dest": str(dest), "bytes": f.get("size_bytes"), "host_evidence": f.get("checksum"),
                "sha256": sha256_of(dest / name)}

    def _record_lineage(self, key: str, *, what: str, inputs: dict, code: str, counts: dict,
                        out_path: Path, columns: dict | None = None, note: str = "",
                        extra: dict | None = None) -> None:
        """Merge one entry into the derived directory's LINEAGE.json (never overwrite others)."""
        path = self.derived / "LINEAGE.json"
        lineage = json.loads(path.read_text()) if path.exists() else {"schema_version": 2, "entries": {}}
        lineage.setdefault("entries", {})
        prov = self.dest / "PROVENANCE.json"
        flags = json.loads(prov.read_text()).get("license_flags") if prov.exists() else None
        entry = {
            "dataset": self.name, "what": what, "derives_from": str(prov),
            "inputs": inputs, "code": code, "code_sha": _code_sha(),
            "built": datetime.now(UTC).isoformat(timespec="seconds"),
            "sha256": sha256_of(out_path), "bytes": out_path.stat().st_size,
            "counts": counts, "licence": self.spec.get("license"), "license_flags": flags,
            "note": note,
        }
        if columns:
            entry["columns"] = columns
        if extra:
            entry.update(extra)
        lineage["entries"][key] = entry
        path.write_text(json.dumps(lineage, indent=2) + "\n")

    # ------------------------------------------------------------ transcripts --

    def _iter_gtf(self) -> Iterator[tuple[str, str, int, int, str, str]]:
        """(chrom, feature, start, end, strand, attributes) for every non-comment GTF row."""
        path = self._for(GENCODE_BLOCK).dest / GTF_FILE
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                p = line.rstrip("\n").split("\t")
                if len(p) < 9:
                    continue
                yield p[0], p[2], int(p[3]), int(p[4]), p[6], p[8]

    def transcript_table(self, *, rebuild: bool = False) -> pd.DataFrame:
        """One row per basic transcript: structure, last exon, stop codon, tags.

        Cached at ``<gencode derived>/transcript_structure.parquet``. Read line by line from
        the gzipped GTF; exon, CDS and stop_codon rows are gathered per transcript and reduced
        once the file is through. The stop codon's 3'-most base is taken over its rows (a stop
        codon split across two exons has two), in transcript orientation.
        """
        src = self._for(GENCODE_BLOCK)
        if src is not self:
            return src.transcript_table(rebuild=rebuild)
        out_path = self.derived / "transcript_structure.parquet"
        if out_path.exists() and not rebuild:
            return pd.read_parquet(out_path)
        self.derived.mkdir(parents=True, exist_ok=True)

        genes: dict[str, tuple[int, int]] = {}
        tx: dict[str, dict] = {}
        exons: dict[str, list[tuple[int, int]]] = {}
        cds: dict[str, list[tuple[int, int]]] = {}
        stops: dict[str, list[tuple[int, int]]] = {}
        n_rows = 0
        for chrom, feature, start, end, strand, attr_text in self._iter_gtf():
            n_rows += 1
            if feature == "gene":
                attrs, _ = parse_gtf_attributes(attr_text)
                genes[_strip(attrs["gene_id"])] = (start, end)
                continue
            if feature not in ("transcript", "exon", "CDS", "stop_codon"):
                continue
            attrs, tags = parse_gtf_attributes(attr_text)
            tid_full = attrs.get("transcript_id", "")
            tid = _strip(tid_full)
            if feature == "transcript":
                tx[tid] = {
                    "transcript_id": tid, "transcript_version": _version(tid_full),
                    "gene_id": _strip(attrs.get("gene_id", "")), "symbol": attrs.get("gene_name", ""),
                    "gene_type": attrs.get("gene_type", ""),
                    "transcript_type": attrs.get("transcript_type", ""),
                    "chrom": chrom, "strand": strand, "tx_start": start, "tx_end": end,
                    "mane_select": "MANE_Select" in tags,
                    "ensembl_canonical": "Ensembl_canonical" in tags,
                }
            elif feature == "exon":
                exons.setdefault(tid, []).append((start, end))
            elif feature == "CDS":
                cds.setdefault(tid, []).append((start, end))
            else:
                stops.setdefault(tid, []).append((start, end))

        rows = []
        for tid, t in tx.items():
            ex = sorted(exons.get(tid, []))
            if not ex:
                continue
            minus = t["strand"] == "-"
            # transcript orientation: on the minus strand the 3'-most exon is the lowest
            last = ex[0] if minus else ex[-1]
            acceptor = (last[1] if minus else last[0]) if len(ex) > 1 else np.nan
            st = stops.get(tid)
            cd = cds.get(tid, [])
            cds_len = int(sum(e - s + 1 for s, e in cd))
            if st:
                stop_3p = min(s for s, _ in st) if minus else max(e for _, e in st)
                in_last = last[0] <= stop_3p <= last[1]
                # exonic bases strictly after the stop codon in the exons BEFORE the last one, in
                # transcript orientation; zero by construction when the stop is in the last exon
                pre = 0
                for s, e in ex:
                    if (s, e) == last:
                        continue
                    lo, hi = (s, min(e, stop_3p - 1)) if minus else (max(s, stop_3p + 1), e)
                    if hi >= lo:
                        pre += hi - lo + 1
                # the annotated 3'UTR: what is left of the last exon after the stop, plus `pre`
                utr3 = ((stop_3p - last[0]) if minus else (last[1] - stop_3p)) if in_last \
                    else pre + (last[1] - last[0] + 1)
            else:
                stop_3p, in_last, pre, utr3 = np.nan, False, 0, np.nan
            gs, ge = genes.get(t["gene_id"], (t["tx_start"], t["tx_end"]))
            rows.append({**t, "gene_start": gs, "gene_end": ge, "n_exons": len(ex),
                         "exonic_len": int(sum(e - s + 1 for s, e in ex)),
                         "last_exon_start": last[0], "last_exon_end": last[1],
                         "acceptor": acceptor, "stop_3p": stop_3p, "stop_in_last_exon": bool(in_last),
                         "exonic_stop_to_acceptor": int(pre), "cds_len": cds_len, "utr3_len": utr3})
        if not rows:
            raise ValueError(f"{GTF_FILE}: no transcript with exons was read")
        table = pd.DataFrame(rows)[list(TRANSCRIPT_COLUMNS)]
        table = table.sort_values(["gene_id", "transcript_id"]).reset_index(drop=True)
        table.to_parquet(out_path, index=False)

        self._record_lineage(
            "gencode-v47/transcript_structure",
            what="one row per GENCODE v47 basic transcript: structure, last exon, stop codon, "
                 "MANE_Select and Ensembl_canonical tags (T102, T120)",
            inputs={GTF_FILE: {**self._file_input(self.dest, GTF_FILE), "md5": _md5_of(self.dest / GTF_FILE),
                               "md5_published": GENCODE_MD5}},
            code="src/sidechain/priors/posttx_polya.py::PolyASiteSource.transcript_table",
            counts={"gtf_rows": n_rows, "genes": len(genes), "transcripts_out": len(table),
                    "transcripts_with_cds": int((table["cds_len"] > 0).sum()),
                    "transcripts_single_exon": int((table["n_exons"] == 1).sum()),
                    "mane_select": int(table["mane_select"].sum()),
                    "ensembl_canonical": int(table["ensembl_canonical"].sum())},
            out_path=out_path, columns=TRANSCRIPT_COLUMNS,
            extra={"coordinates": "1-based inclusive genomic, as the GTF; `acceptor` and `stop_3p` "
                                  "are genomic positions read in transcript orientation"})
        return table

    # ------------------------------------------------------------------ genes --

    def _targetscan_representative(self) -> pd.DataFrame:
        """TargetScan's representative transcript per gene WITH its version: stem, version, gene stem."""
        ts = self._targetscan()
        info = ts._read_zipped_table(GENE_INFO_FILE, dtype=str)
        info.columns = [c.strip() for c in info.columns]
        col = {c.lower(): c for c in info.columns}
        rep = next((c for c in info.columns if c.lower().startswith("representative")), None)
        info = info[(info[col["species id"]].astype(int) == HUMAN)
                    & info[rep].astype(str).str.strip().isin({"1", "yes", "Yes", "TRUE", "True"})]
        tids = info[col["transcript id"]].astype(str).str.strip()
        return pd.DataFrame({"transcript_id": tids.map(_strip), "ts_version": tids.map(_version),
                             "gene_id": info[col["gene id"]].astype(str).map(_strip)}
                            ).drop_duplicates("gene_id").reset_index(drop=True)

    def representatives(self, tx: pd.DataFrame) -> pd.DataFrame:
        """One representative transcript per v47 gene, by the ordered rules (`pa_tx_source`)."""
        by_tid = tx.set_index("transcript_id")
        chosen: dict[str, tuple[str, str, bool]] = {}
        ts = self._targetscan_representative()
        for r in ts.itertuples(index=False):
            if r.transcript_id in by_tid.index and by_tid.at[r.transcript_id, "gene_id"] == r.gene_id:
                changed = bool(r.ts_version >= 0
                               and int(by_tid.at[r.transcript_id, "transcript_version"]) != int(r.ts_version))
                chosen[r.gene_id] = (r.transcript_id, "targetscan", changed)
        for gid, g in tx.groupby("gene_id", sort=False):
            if gid in chosen:
                continue
            g = g.sort_values("transcript_id")
            m = g[g["mane_select"]]
            if len(m):
                chosen[gid] = (m.iloc[0]["transcript_id"], "mane_select", False)
                continue
            c = g[g["ensembl_canonical"]]
            if len(c):
                chosen[gid] = (c.iloc[0]["transcript_id"], "ensembl_canonical", False)
                continue
            pc = g[(g["transcript_type"] == "protein_coding") & (g["cds_len"] > 0)]
            if len(pc):
                chosen[gid] = (pc.sort_values("cds_len", ascending=False).iloc[0]["transcript_id"],
                               "longest_cds", False)
                continue
            chosen[gid] = (g.sort_values("exonic_len", ascending=False).iloc[0]["transcript_id"],
                           "longest", False)
        return pd.DataFrame([{"gene_id": gid, "transcript_id": t, "pa_tx_source": src,
                              "pa_tx_version_changed": ch} for gid, (t, src, ch) in chosen.items()])

    def _read_atlas(self, *, progress: bool = False) -> tuple[pd.DataFrame, dict]:
        """The atlas TSV read once into a small frame: the fixed columns plus, per named line,
        the summed TPM over that line's unperturbed libraries.

        Over the network (`stream_gated_file`) when the block says ``route: stream``, from the
        file the gate landed under ``dest`` otherwise; either way the same reader shape, and
        the LINEAGE entry says which (``streamed``). Returns ``(clusters, info)``;
        ``info["stream"]`` is the reader (its sha256 is set once the read is through),
        ``info["libraries"]`` the per-library line assignment
        and ``info["line_columns"]`` the libraries each line's sum was taken over. Column lists
        rather than row dicts: 569k rows of dicts is half a gigabyte for nothing.
        """
        src = self._for(POLYASITE_BLOCK)
        if _is_stream(src.spec):
            stream = stream_gated_file(src.spec, src.root, name=ATLAS_FILE, progress=progress,
                                       registry=src.registry)
        else:
            stream = _LocalFile(src.dest, ATLAS_FILE, src._file_input(src.dest, ATLAS_FILE))
        lines = stream.lines()
        header = next(lines).split("\t")
        if tuple(header[:len(ATLAS_COLUMNS)]) != ATLAS_COLUMNS:
            raise ValueError(f"{ATLAS_FILE}: unexpected fixed columns {header[:len(ATLAS_COLUMNS)]}; "
                             f"expected {ATLAS_COLUMNS}")
        libraries: dict[str, dict] = {}
        line_cols: dict[str, list[int]] = {ln: [] for ln in ATLAS_LINES}
        for j, h in enumerate(header[len(ATLAS_COLUMNS):], start=len(ATLAS_COLUMNS)):
            line, source, treatment = classify_library(h)
            libraries[h] = {"line": line, "source": source, "treatment": treatment}
            if line:
                line_cols[line].append(j)
        cols: dict[str, list] = {k: [] for k in ("chrom", "strand", "rep", "tpm", "frac_samples",
                                                  "nr_prots", "annotation", "gene_ids", "signals")}
        for ln in ATLAS_LINES:
            cols[f"tpm_{ln}"] = []
        for line in lines:
            if not line:
                continue
            p = line.split("\t")
            cols["chrom"].append(atlas_chrom(p[0]))
            cols["strand"].append(p[5])
            cols["rep"].append(int(p[6]))
            cols["tpm"].append(float(p[4]))
            cols["frac_samples"].append(float(p[7]))
            cols["nr_prots"].append(int(p[8]))
            cols["annotation"].append(p[9])
            cols["gene_ids"].append(p[11])
            cols["signals"].append(p[12])
            for ln, js in line_cols.items():
                cols[f"tpm_{ln}"].append(sum(float(p[j]) for j in js) if js else 0.0)
        clusters = pd.DataFrame(cols)
        info = {"stream": stream, "libraries": libraries,
                "line_columns": {ln: [header[j] for j in js] for ln, js in line_cols.items()},
                "n_libraries": len(header) - len(ATLAS_COLUMNS)}
        return clusters, info

    def gene_table(self, *, rebuild: bool = False, progress: bool = False) -> pd.DataFrame:
        """One row per v47 gene: the polyadenylation layout (`GENE_COLUMNS`).

        Cached at ``<atlas derived>/pas_gene_features.parquet``. The atlas is streamed once;
        the clusters are kept in memory as a small frame (569k rows, a dozen columns), the
        transcript table gives the windows, and each gene's clusters are found by a sorted
        search per (chromosome, strand). Nothing of the atlas is written.
        """
        src = self._for(POLYASITE_BLOCK)
        if src is not self:
            return src.gene_table(rebuild=rebuild, progress=progress)
        out_path = self.derived / "pas_gene_features.parquet"
        if out_path.exists() and not rebuild:
            return pd.read_parquet(out_path)
        self.derived.mkdir(parents=True, exist_ok=True)

        tx = self.transcript_table()
        reps = self.representatives(tx)
        clusters, info = self._read_atlas(progress=progress)
        stream = info["stream"]
        if clusters.empty:
            raise ValueError(f"{ATLAS_FILE}: no cluster rows")
        clusters["canonical_pas"] = clusters["signals"].map(
            lambda sig: any(h in str(sig) for h in CANONICAL_PAS))
        clusters["te_ds"] = clusters["annotation"].isin(["TE", "DS"])
        # transcript-orientation position: negated on the minus strand, so "downstream" is "larger"
        clusters["pos"] = np.where(clusters["strand"] == "-", -clusters["rep"], clusters["rep"])
        groups = {k: g.sort_values("pos").reset_index(drop=True)
                  for k, g in clusters.groupby(["chrom", "strand"], sort=False)}
        line_names = list(ATLAS_LINES)

        by_tid = tx.set_index("transcript_id")
        gene_tx = {gid: g for gid, g in tx.groupby("gene_id", sort=False)}
        out_rows = []
        n_windows_empty = 0
        for r in reps.itertuples(index=False):
            t = by_tid.loc[r.transcript_id]
            minus = t["strand"] == "-"
            sign = -1 if minus else 1
            single = int(t["n_exons"]) == 1
            # a CDS whose 3' end is not annotated (cds_end_NF) has no stop codon: no 3'UTR start
            coding = bool(t["cds_len"] > 0) and bool(np.isfinite(float(t["stop_3p"])))
            acceptor = None if single else float(t["acceptor"])
            stop = float(t["stop_3p"]) if coding else None
            # the window start, in transcript orientation
            if single:
                start = sign * (stop + sign) if coding else sign * (t["tx_start"] if not minus else t["tx_end"])
            else:
                start = sign * acceptor
                if coding and bool(t["stop_in_last_exon"]):
                    start = max(start, sign * stop + 1)
            # the window end: the farthest 3' end among the gene's transcripts sharing the acceptor
            g = gene_tx[r.gene_id]
            same = g[g["n_exons"] == 1] if single else g[g["acceptor"] == acceptor]
            ends = (sign * same["tx_end"]) if not minus else (sign * same["tx_start"])
            end = int(max(ends.max(), sign * (t["tx_start"] if minus else t["tx_end"]))) + DOWNSTREAM_NT
            key = (t["chrom"], t["strand"])
            grp = groups.get(key)
            win = grp.iloc[grp["pos"].searchsorted(start, "left"):grp["pos"].searchsorted(end, "right")] \
                if grp is not None else None
            row = {"gene_id": r.gene_id, "symbol": t["symbol"], "transcript_id": r.transcript_id,
                   "pa_tx_source": r.pa_tx_source, "pa_tx_version_changed": bool(r.pa_tx_version_changed),
                   "pa_single_exon": single, "pa_te_len": int(t["last_exon_end"] - t["last_exon_start"] + 1),
                   "pa_coding": coding, "pa_window_len": int(end - start + 1)}
            if win is None or win.empty:
                n_windows_empty += 1
                row.update({"pa_window_tpm_sum": 0.0, "pa_n_sites_all": 0, "pa_n_sites": 0, "pa_has_apa": False,
                            "pa_acc_to_prox_nt": np.nan, "pa_acc_to_prox_nt_all": np.nan,
                            "pa_log10_acc_to_prox": np.nan, "pa_stop_to_prox_nt": np.nan,
                            "pa_stop_to_dist_nt": np.nan, "pa_span_nt": np.nan, "pa_prox_usage": np.nan,
                            "pa_prox_has_canonical_pas": False, "pa_prox_frac_samples": np.nan,
                            "pa_prox_nr_prots": np.nan, "pa_window_te_ds_share": np.nan})
                for ln in line_names:
                    row[f"pa_prox_usage_{ln}"] = np.nan
                    row[f"pa_window_tpm_{ln}"] = 0.0
            else:
                total = float(win["tpm"].sum())
                keep = win[(win["nr_prots"] >= MIN_PROTOCOLS) & (win["tpm"] >= MIN_TPM_FRAC * total)]
                prox_all = int(win.iloc[0]["pos"])
                row.update({"pa_window_tpm_sum": total, "pa_n_sites_all": len(win),
                            "pa_n_sites": len(keep), "pa_has_apa": bool(len(keep) >= 2),
                            "pa_acc_to_prox_nt_all": (prox_all - sign * acceptor) if acceptor is not None else np.nan,
                            "pa_window_te_ds_share": float(win["te_ds"].mean())})
                if keep.empty:
                    row.update({"pa_acc_to_prox_nt": np.nan, "pa_log10_acc_to_prox": np.nan,
                                "pa_stop_to_prox_nt": np.nan, "pa_stop_to_dist_nt": np.nan,
                                "pa_span_nt": np.nan, "pa_prox_usage": np.nan,
                                "pa_prox_has_canonical_pas": False, "pa_prox_frac_samples": np.nan,
                                "pa_prox_nr_prots": np.nan})
                    for ln in line_names:
                        row[f"pa_prox_usage_{ln}"] = np.nan
                        row[f"pa_window_tpm_{ln}"] = float(win[f"tpm_{ln}"].sum())
                else:
                    prox, dist = keep.iloc[0], keep.iloc[-1]
                    p_pos, d_pos = int(prox["pos"]), int(dist["pos"])
                    acc_to_prox = (p_pos - sign * acceptor) if acceptor is not None else np.nan

                    utr = {"coding": coding, "stop_in_last": bool(t["stop_in_last_exon"]) or single,
                           "stop": stop, "sign": sign, "acceptor": acceptor,
                           "pre": int(t["exonic_stop_to_acceptor"])}
                    kept_total = float(keep["tpm"].sum())
                    row.update({"pa_acc_to_prox_nt": acc_to_prox,
                                "pa_log10_acc_to_prox": np.log10(acc_to_prox + 1) if np.isfinite(acc_to_prox) and acc_to_prox >= 0 else np.nan,
                                "pa_stop_to_prox_nt": _utr_len_to(p_pos, **utr), "pa_stop_to_dist_nt": _utr_len_to(d_pos, **utr),
                                "pa_span_nt": float(d_pos - p_pos),
                                "pa_prox_usage": float(prox["tpm"] / kept_total) if kept_total > 0 else np.nan,
                                "pa_prox_has_canonical_pas": bool(prox["canonical_pas"]),
                                "pa_prox_frac_samples": float(prox["frac_samples"]),
                                "pa_prox_nr_prots": int(prox["nr_prots"])})
                    for ln in line_names:
                        s = float(keep[f"tpm_{ln}"].sum())
                        row[f"pa_prox_usage_{ln}"] = float(prox[f"tpm_{ln}"] / s) if s > 0 else np.nan
                        row[f"pa_window_tpm_{ln}"] = float(win[f"tpm_{ln}"].sum())
            # intronic: same-strand IN clusters inside the gene span and outside the window
            if grp is not None:
                g_lo, g_hi = sorted((sign * int(t["gene_start"]), sign * int(t["gene_end"])))
                span = grp.iloc[grp["pos"].searchsorted(g_lo, "left"):grp["pos"].searchsorted(g_hi, "right")]
                intr = span[(span["annotation"] == "IN") & (span["nr_prots"] >= MIN_PROTOCOLS)
                            & ((span["pos"] < start) | (span["pos"] > end))]
                row["pa_n_intronic"] = len(intr)
            else:
                row["pa_n_intronic"] = 0
            out_rows.append(row)

        line_cols = [f"pa_prox_usage_{ln}" for ln in line_names] + [f"pa_window_tpm_{ln}" for ln in line_names]
        table = pd.DataFrame(out_rows)[list(GENE_COLUMNS) + line_cols]
        table = table.sort_values("gene_id").reset_index(drop=True)
        table.to_parquet(out_path, index=False)

        gencode = self._for(GENCODE_BLOCK)
        self._record_lineage(
            "polyasite-2.0/pas_gene_features",
            what="one row per GENCODE v47 gene: acceptor-to-proximal-site distance, supported "
                 "poly(A) sites in the 3' window, short and long 3'UTR lengths, proximal usage "
                 "and signal, intronic sites, from PolyASite 2.0 streamed once (T102, T120)",
            inputs={ATLAS_FILE: stream.lineage_input(),
                    GTF_FILE: self._file_input(gencode.dest, GTF_FILE),
                    GENE_INFO_FILE: self._file_input(self._targetscan().dest, GENE_INFO_FILE)},
            code="src/sidechain/priors/posttx_polya.py::PolyASiteSource.gene_table",
            counts={"clusters": len(clusters), "libraries": info["n_libraries"],
                    "genes_out": len(table), "windows_empty": n_windows_empty,
                    "genes_with_passing_site": int(table["pa_n_sites"].gt(0).sum()),
                    "genes_with_apa": int(table["pa_has_apa"].sum()),
                    "genes_single_exon": int(table["pa_single_exon"].sum()),
                    "genes_acc_to_prox": int(table["pa_acc_to_prox_nt"].notna().sum()),
                    "tx_source": table["pa_tx_source"].value_counts().to_dict(),
                    "tx_version_changed": int(table["pa_tx_version_changed"].sum()),
                    "line_libraries": {ln: len(cols) for ln, cols in info["line_columns"].items()}},
            out_path=out_path,
            columns={**GENE_COLUMNS,
                     **{f"pa_prox_usage_{ln}": f"the proximal passing cluster's TPM over the passing clusters' "
                                               f"TPM in the atlas's unperturbed {ln} libraries; written, used "
                                               "nowhere (Saber, 2026-10-15)" for ln in line_names},
                     **{f"pa_window_tpm_{ln}": f"summed TPM of the window's clusters in those {ln} libraries"
                        for ln in line_names}},
            extra={"filter": {"min_protocols": MIN_PROTOCOLS, "min_tpm_frac_of_window": MIN_TPM_FRAC,
                              "downstream_nt": DOWNSTREAM_NT},
                   "window": "from the representative's last splice acceptor (after the stop codon when it "
                             "lies in that exon) to the farthest 3' end among the gene's transcripts sharing "
                             "that acceptor, plus downstream_nt; same chromosome and strand; a cluster is "
                             "in the window by its `rep` site",
                   "coordinates": "atlas `rep` read as 1-based genomic; GTF 1-based inclusive; a one-base "
                                  "disagreement is below the resolution of every column",
                   "representative_rules": ["targetscan (Gene_info, by stem, gene stem must match)",
                                            "mane_select", "ensembl_canonical",
                                            "longest_cds (protein_coding)", "longest"],
                   "line_columns": info["line_columns"],
                   "libraries": info["libraries"]})
        return table

    # ------------------------------------------------------------ the feature --

    def build(self) -> PriorArtifact:
        """Three per-gene columns aligned to the master gene space, with a `has_value` mask.

        ``features[:, 0]`` is ``pa_log10_acc_to_prox``, ``[:, 1]`` ``pa_n_sites``, ``[:, 2]``
        ``pa_prox_usage``. A gene with no row, or NaN in a column, is 0.0 there and False in
        the mask -- and 0 sites is a measurement for a gene the atlas covers, so read the mask.
        """
        table = self._for(POLYASITE_BLOCK).gene_table()
        names = ["pa_log10_acc_to_prox", "pa_n_sites", "pa_prox_usage"]
        n_genes = len(self.gene_index)
        features = np.zeros((n_genes, len(names)), dtype=np.float32)
        has_value = np.zeros((n_genes, len(names)), dtype=bool)
        ids = table["gene_id"].astype(str).to_numpy()
        for j, col in enumerate(names):
            values = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
            for gene, value in zip(ids, values):
                position = self.gene_index.get(gene)
                if position is None or not np.isfinite(value):
                    continue
                features[position, j] = np.float32(value)
                has_value[position, j] = True
        return PriorArtifact(
            kind="node_feature", relation=self.relation, layer=self.layer, features=features,
            meta={"feature_names": names, "has_value": has_value,
                  "sources": {n: POLYASITE_BLOCK for n in names},
                  "candidate_genes": len(table),
                  "genes_with_value": {n: int(has_value[:, i].sum()) for i, n in enumerate(names)},
                  "genes_with_any_value": int(has_value.any(axis=1).sum()),
                  "zero_is_a_real_value": True,
                  "licence": self._for(POLYASITE_BLOCK).spec.get("license")},
        )


def _is_stream(spec: dict) -> bool:
    from sidechain.ingest.fetch import block_route
    from sidechain.ingest.provenance import STREAM
    return block_route(spec) == STREAM


class _LocalFile:
    """The on-disk twin of `posttx_mirna.StreamedFile`: the same ``lines()`` and
    ``lineage_input()`` over a gzipped file the gate landed under ``dest``."""

    def __init__(self, dest: Path, name: str, file_input: dict):
        self.dest, self.name, self._input = Path(dest), name, file_input
        self.sha256: str | None = None

    def lines(self, encoding: str = "utf-8") -> Iterator[str]:
        with gzip.open(self.dest / self.name, "rt", encoding=encoding) as fh:
            for line in fh:
                yield line.rstrip("\n").rstrip("\r")
        self.sha256 = self._input["sha256"]

    def lineage_input(self) -> dict:
        return {**self._input, "streamed": False}


def _utr_len_to(pos: int, *, coding: bool, stop_in_last: bool, stop: float | None, sign: int,
                acceptor: float | None, pre: int) -> float:
    """3'UTR length from the stop codon to a site at `pos` (transcript orientation), in nt.

    With the stop in the last exon (or a single exon) it is the straight distance; with the stop
    in an earlier exon it is the exonic bases after the stop before the acceptor (`pre`), plus
    the distance from the acceptor to the site, plus one for the acceptor base itself.
    """
    if not coding:
        return np.nan
    if stop_in_last:
        return float(pos - sign * stop)
    return float(pre + (pos - sign * acceptor) + 1)


def _md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with Path(path).open("rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


__all__ = [
    "DOWNSTREAM_NT",
    "GENCODE_MD5",
    "GENE_COLUMNS",
    "MIN_PROTOCOLS",
    "MIN_TPM_FRAC",
    "TRANSCRIPT_COLUMNS",
    "PolyASiteSource",
    "atlas_chrom",
    "classify_library",
    "parse_gtf_attributes",
]
