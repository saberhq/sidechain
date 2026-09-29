"""Post-transcriptional layer: measured mRNA stability as a per-gene node feature.

The sibling of ``posttx_mirna``. That module reads the 3'UTR *grammar* a gene carries;
this one reads how long the transcript actually lasts, from two published measurements
that are universal per gene rather than per context -- which is exactly the kind of
covariate T102's decomposition needs, because a feature that varies by cell line cannot
explain the gene-consistent part of the truth.

**What is built (T102, step 3).** Three tables, each cached as a parquet under its
block's ``derived/`` with an entry in the LINEAGE.json beside it:

* ``halflife_table()`` -- one row per human gene from Agarwal & Kelley 2022 (Genome
  Biology 23:245, doi:10.1186/s13059-022-02811-x) Additional file 3, sheet ``human``:
  the published ``half-life (PC1)`` consensus, plus the mean and count of each cell
  line's sample columns. PC1 summarises 54 samples across nine lines and the authors
  report no detectable cell-type signal in it; the per-line means are here so that
  claim can be *tested* on HEK293 and K562 before the consensus is trusted as universal.
* ``codon_table()`` -- one row per representative transcript, from the human ORFs of
  TargetScan 8.0's ``ORF_Sequences.txt`` scored against Wu et al. 2019 (eLife 8:e45396)
  Figure 1 source data 2: ORF length, GC3, codon counts, and the gene's mean codon
  stability coefficient (CSC) under each of Wu's six measurement columns.
  ``csc_endo_mean`` averages the three actinomycin-D endogenous-decay columns, which
  are the three that agree (Spearman 0.91-0.95 pairwise; across all six it falls to
  0.56, and only 34 of 61 codons keep their sign).
* ``decay_table()`` -- the per-gene decay rates of Wu's Figure 1 source data 1, raw.

``build()`` is the node feature the registry consumes: two columns,
``[halflife_pc1, csc_endo_mean]``, aligned to the master gene space.

**Name the lines by Wu's Methods, never by its Key Resources Table.** That table gives
K562 the ATCC number of hTERT RPE-1 and RPE the number of K-562; the Methods settle it
by growth medium. So the lines are ``293T``, ``HeLa``, ``RPE`` and ``K562``
(``research/reading/posttx-verify-08-wu2019-csc-crossline.md``, private).

**Two registry blocks, one class.** ``halflife_consensus`` and ``codon_optimality_wu2019``
are separate ``configs/data_sources.yaml`` blocks because they are separate publishers
with separate terms, separate destinations and separate budgets -- but one loader, because
the feature they build is one vector. An instance knows which block it is; an entry point
belonging to the other block delegates to a sibling built from the registry (``_for``),
so ``halflife_table()``, ``codon_table()`` and ``build()`` all work from either.

**The fetch.** ``fetch()`` is ``posttx_mirna.fetch_gated_files``, shared with
``MiRNATargetSource``: probe -> gate -> PROVENANCE.json before any byte lands (ADR 0003).
Both hosts are plain file servers that publish no checksum, so the block carries
``allow_missing_checksum`` and the sha256 of what landed is recorded in LINEAGE.json.
``codon_table()`` additionally reads two files this block does NOT fetch --
``ORF_Sequences.txt.zip`` and ``Gene_info.txt.zip`` from the ``targetscan`` block's dest.
They are already in that block's file list; claiming them here too would give one set of
bytes two PROVENANCE.json records.
"""
from __future__ import annotations

import io
import itertools
import json
import re
import subprocess
from collections import Counter
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
)

#: The registry blocks this loader serves, and the files each one names.
HALFLIFE_BLOCK = "halflife_consensus"
CODON_BLOCK = "codon_optimality_wu2019"
TARGETSCAN_BLOCK = "targetscan"

HALFLIFE_FILE = "13059_2022_2811_MOESM3_ESM.xlsx"
CSC_FILE = "elife-45396-fig1-data2-v2.csv"
DECAY_FILE = "elife-45396-fig1-data1-v2.xlsx"
ORF_FILE = "ORF_Sequences.txt.zip"
GENE_INFO_FILE = "Gene_info.txt.zip"

#: Sheet and header of Table S2. Row 1 of the `human` sheet is a note ("The half-life
#: (PC1) is a summary of all of the datasets combined, excluding data from Gejman et
#: al."), so the header is row 2 -- but `_header_row` FINDS it by looking for the key
#: column rather than trusting that, because a republished file that drops the note
#: would otherwise be read one row off with no error at all.
HALFLIFE_SHEET = "human"
HALFLIFE_KEY_COLUMN = "ensembl gene id"
HALFLIFE_HEADER_SCAN = 8

#: Cell lines the 54 sample columns of Table S2 are grouped into, and how a column name
#: is matched to one. `tokens` are matched against the column name split on every
#: non-alphanumeric run (``Bazzini_ActD_HEK293_1`` -> ``BAZZINI ACTD HEK293 1``);
#: `substrings` are matched against the squashed name only when no token matched, and
#: deliberately EXCLUDE the short ambiguous forms -- an exact token ``H1`` names the
#: embryonic stem line, the substring ``H1`` also sits inside ``H1975``. Every column's
#: assignment and the rule that made it are written into LINEAGE.json, so the grouping
#: is auditable rather than asserted. HEK293T is folded into HEK293: the paper's samples
#: do not separate them and the challenge has no use for the distinction.
CELL_LINES: dict[str, dict[str, tuple[str, ...]]] = {
    "hek293": {"tokens": ("HEK293T", "HEK293", "HEK293FT", "293T", "293"),
               "substrings": ("HEK293T", "HEK293")},
    "k562": {"tokens": ("K562",), "substrings": ("K562",)},
    "hela": {"tokens": ("HELA", "HELAS3"), "substrings": ("HELAS3", "HELA")},
    "hepg2": {"tokens": ("HEPG2",), "substrings": ("HEPG2",)},
    "rpe": {"tokens": ("RPE", "RPE1", "HTERTRPE1"), "substrings": ("HTERTRPE1", "RPE1", "RPE")},
    "h1esc": {"tokens": ("H1ESC", "H1HESC", "H1ES", "HESC", "H1"),
              "substrings": ("H1HESC", "H1ESC", "HESC")},
    "mcf7": {"tokens": ("MCF7",), "substrings": ("MCF7",)},
    "a549": {"tokens": ("A549",), "substrings": ("A549",)},
    "lcl": {"tokens": ("LCL", "LCLS", "GM12878"), "substrings": ("GM12878", "LCL")},
}

#: The 64 codons, the three stops, and the 61 SENSE codons a CSC table scores. The mean
#: CSC of a gene is taken over its sense codons only: a stop codon has no CSC and the
#: terminal one is not a coding choice.
BASES = "ACGT"
CODONS: tuple[str, ...] = tuple(a + b + c for a in BASES for b in BASES for c in BASES)
STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})
SENSE_CODONS: tuple[str, ...] = tuple(c for c in CODONS if c not in STOP_CODONS)

#: Wu's six CSC columns, as Figure 1 source data 2 spells them, and the three that were
#: measured the same way (actinomycin D, endogenous transcripts). `csc_endo_mean` is the
#: mean of those three: they agree at Spearman 0.91-0.95, while the full six fall to 0.56
#: -- and the low pairs are METHODS, not lines, so averaging all six would average a
#: measurement artefact into the feature.
CSC_COLUMNS: tuple[str, ...] = ("293T_endo", "HeLa_endo", "RPE_endo",
                                "293T_ORFome", "K562_ORFome", "K562_SLAM")
CSC_ENDOGENOUS: tuple[str, ...] = ("293T_endo", "HeLa_endo", "RPE_endo")

#: The sheets of Figure 1 source data 1 that `decay_table` reads, and the line and method
#: each one is, by the paper's Methods. The two ORFome sheets are SKIPPED: they are keyed
#: by a 24-nt library barcode plus four spike-ins with no gene id, so they need the TRC3
#: ORF barcode map before a gene can be named at all.
DECAY_SHEETS: dict[str, tuple[str, str]] = {
    "293T-endogenous": ("293T", "actinomycin D, endogenous transcripts"),
    "HeLa-endogenous": ("HeLa", "actinomycin D, endogenous transcripts"),
    "RPE-endogenous": ("RPE", "actinomycin D, endogenous transcripts"),
    "k562-SLAM-seq": ("K562", "SLAM-seq, 4sU metabolic labelling"),
}
DECAY_SHEETS_SKIPPED: dict[str, str] = {
    "293T-ORFome": "keyed by a 24-nt ORFome barcode, not a gene id",
    "k562-ORFome": "keyed by a 24-nt ORFome barcode, not a gene id",
}

#: Gap and padding characters stripped from an aligned ORF before it is read as codons.
#: TargetScan's sequence files come from the 84-way alignment, so `-` is the gap; the
#: others are here because a reader that silently keeps one would shift every codon
#: downstream of it and the frame check would then fail for a reason nobody could see.
GAP_CHARACTERS = "-.~*_ \t\r\n"
_UNGAP = str.maketrans("", "", GAP_CHARACTERS)

#: The per-gene columns `halflife_table` always produces (the per-line ones are named
#: from the grouping, so they are not fixed).
HALFLIFE_COLUMNS = {
    "gene_id": "Ensembl gene id, version stripped",
    "symbol": "the `Gene name` column of Table S2 (2022-era symbols)",
    "halflife_pc1": "the published `half-life (PC1)` consensus, as published: PC1 of the "
                    "filtered, transformed, imputed and quantile-normalised matrix of 54 "
                    "human samples, excluding the Gejman LCL data. NOT z-scored -- the "
                    "paper's figures z-score it, this column does not",
}

#: The per-gene columns `codon_table` always produces, besides the 64 `codon_<XXX>`
#: counts and the six `csc_<column>` means.
CODON_COLUMNS = {
    "transcript_id": "TargetScan's representative transcript for the gene, version stripped",
    "gene_id": "Ensembl gene id, version stripped",
    "symbol": "TargetScan's gene symbol (GENCODE-era, ~2018)",
    "orf_len": "CDS length in nt: the ungapped row read in frame from its first base, up to and "
               "including the first in-frame stop (whole codons of the row when there is none)",
    "raw_len": "the ungapped row's length -- TargetScan's ORF rows run a few nt past the stop",
    "trailing_nt": "raw_len - orf_len: what the row carries after the CDS",
    "n_codons": "codons in the CDS, the stop included",
    "n_sense_codons": "codons that are one of the 61 sense codons (no stop, no ambiguity)",
    "n_stops_raw": "in-frame stop codons over the whole raw row (the CDS's own is one of them)",
    "n_ambiguous_codons": "codons carrying a base outside ACGT",
    "has_in_frame_stop": "a TAA, TAG or TGA occurs in frame; the CDS ends at the first one",
    "len_multiple_of_3": "raw_len is a whole number of codons (descriptive; the CDS never needs it)",
    "starts_atg": "the ORF begins ATG",
    "gc3": "fraction of SENSE codons whose third position is G or C",
    "csc_endo_mean": "mean of csc_293T_endo, csc_HeLa_endo and csc_RPE_endo -- the three "
                     "actinomycin-D endogenous columns, the ones that agree",
}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _squash(name: str) -> str:
    """A column or sheet name reduced to what a match should not care about."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def ungap_orf(seq: str) -> str:
    """An aligned ORF row as plain DNA: gaps removed, uppercased, U read as T.

    U appears because some sequence tables are published on the RNA alphabet; reading
    codons on one alphabet and the CSC table on the other would score nothing at all.
    """
    return seq.translate(_UNGAP).upper().replace("U", "T")


def group_sample_columns(columns: list[str]) -> tuple[dict[str, list[str]], dict[str, str], list[str]]:
    """Group Table S2's per-sample columns by cell line.

    Returns ``(line -> columns, column -> how it matched, unassigned columns)``. The
    grouping is returned rather than applied so `halflife_table` can write it into
    LINEAGE.json: which sample went into which line's mean is the one fact a reader of
    ``halflife_hek293_mean`` needs and cannot recover from the parquet.

    Two rules, in order. A column whose name contains an exact TOKEN of a line
    (``Bazzini_ActD_HEK293_1`` -> ``HEK293``) is assigned first. Anything still
    unmatched is tried against the `substrings` of every line, longest first, in case a
    name glues the line to its neighbour. A column matching neither is left unassigned
    and named in the lineage -- never folded into a line on a guess.
    """
    by_line: dict[str, list[str]] = {line: [] for line in CELL_LINES}
    how: dict[str, str] = {}
    unassigned: list[str] = []

    # longest first so HELAS3 is tried before HELA and H1ESC before HESC
    substrings = sorted(
        ((pat, line) for line, spec in CELL_LINES.items() for pat in spec["substrings"]),
        key=lambda pl: -len(pl[0]),
    )
    for col in columns:
        upper = str(col).upper()
        tokens = {t for t in re.split(r"[^A-Z0-9]+", upper) if t}
        hit = next((line for line, spec in CELL_LINES.items()
                    if tokens & set(spec["tokens"])), None)
        if hit is not None:
            by_line[hit].append(col)
            how[col] = "token"
            continue
        squashed = re.sub(r"[^A-Z0-9]", "", upper)
        hit = next((line for pat, line in substrings if pat in squashed), None)
        if hit is not None:
            by_line[hit].append(col)
            how[col] = "substring"
        else:
            unassigned.append(col)
    return {line: cols for line, cols in by_line.items() if cols}, how, unassigned


def codon_features(seq: str, csc: pd.DataFrame) -> dict:
    """CDS length, frame checks, codon counts, GC3 and one mean CSC per CSC column.

    `seq` is already ungapped (`ungap_orf`); `csc` is indexed by codon with one column
    per measurement. **TargetScan's ORF rows run past the stop codon** -- measured on the
    release 2026-09-28: ARF5's row is 557 nt, its CDS 543 nt ending TAA at codon 181, then
    14 nt of 3'UTR; read to the row's end, 98.8 % of genes looked out of frame and 74 % had
    "internal" stops. So the row is read in frame from its first base (19,307 of 19,440
    start ATG) and the CDS ends at the FIRST in-frame stop, inclusive; `trailing_nt` records
    what came after. The mean is over the CDS's SENSE codons, weighted by how often each
    one occurs -- the CSC of an average codon in this ORF, which is what Wu's per-gene
    analysis uses. A codon whose CSC is missing drops out of both the numerator and the
    denominator rather than being read as zero.

    Nothing here rejects a gene: `starts_atg`, `has_in_frame_stop` and `trailing_nt` are
    reported so a consumer can filter on its own bar, and `codon_table` counts them.
    """
    raw = [seq[i:i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    stop_at = next((k for k, c in enumerate(raw) if c in STOP_CODONS), None)
    codons = raw[:stop_at + 1] if stop_at is not None else raw
    n = 3 * len(codons)
    counts = Counter(codons)
    out: dict = {
        "orf_len": n,
        "raw_len": len(seq),
        "trailing_nt": len(seq) - n,
        "n_codons": len(codons),
        "len_multiple_of_3": bool(len(seq) > 0 and len(seq) % 3 == 0),
        "starts_atg": bool(seq[:3] == "ATG"),
        "has_in_frame_stop": stop_at is not None,
        "n_stops_raw": int(sum(1 for c in raw if c in STOP_CODONS)),
    }
    weights = np.array([counts.get(c, 0) for c in SENSE_CODONS], dtype=float)
    n_sense = int(weights.sum())
    n_stop = int(sum(counts.get(c, 0) for c in STOP_CODONS))
    out["n_sense_codons"] = n_sense
    out["n_ambiguous_codons"] = int(len(codons) - n_sense - n_stop)
    for c in CODONS:
        out[f"codon_{c}"] = int(counts.get(c, 0))
    if n_sense:
        out["gc3"] = float(sum(counts.get(c, 0) for c in SENSE_CODONS if c[2] in "GC") / n_sense)
    else:
        out["gc3"] = np.nan
    for col in csc.columns:
        values = csc[col].reindex(list(SENSE_CODONS)).to_numpy(dtype=float)
        ok = np.isfinite(values) & (weights > 0)
        total = weights[ok].sum()
        out[f"csc_{col}"] = float((weights[ok] * values[ok]).sum() / total) if total else np.nan
    endo = [out[f"csc_{c}"] for c in CSC_ENDOGENOUS if f"csc_{c}" in out]
    finite = [v for v in endo if np.isfinite(v)]
    out["csc_endo_mean"] = float(np.mean(finite)) if finite else np.nan
    return out


class MRNAStabilitySource(PriorSource):
    """Measured mRNA half-life and codon optimality as a per-gene node feature.

    Constructed like every PriorSource -- ``(spec, gene_index)`` -- but the table entry
    points need no gene index: ``fetch()``, ``halflife_table()``, ``codon_table()`` and
    ``decay_table()`` work on the spec alone, so ``gene_index`` may be an empty dict for
    those. ``build()`` needs it (Ensembl gene id -> position).

    ``registry`` names the prior registry the sibling block is looked up in, so a test can
    point both blocks at a fixture without reaching into the checked-in config.
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

    def fetch(self, *, refresh: bool = False, progress: bool = False) -> Path:
        """Gate, record provenance, then download the block's files. Idempotent.

        The loop is `posttx_mirna.fetch_gated_files`, shared with `MiRNATargetSource`;
        the order it enforces (probe -> gate -> PROVENANCE.json -> bytes) is that
        function's docstring. It fetches THIS block only: ``codon_table`` also needs
        TargetScan's ORF and gene-info files, which the ``targetscan`` block fetches.
        """
        return fetch_gated_files(self.spec, self.root, refresh=refresh, progress=progress,
                                 registry=self.registry)

    def _for(self, name: str) -> MRNAStabilitySource:
        """This source if it is the named block, otherwise a sibling built from the registry."""
        if self.name == name:
            return self
        return MRNAStabilitySource(spec_from_registry(name, self.registry), self.gene_index,
                                   root=self.root, registry=self.registry)

    def _targetscan(self) -> MiRNATargetSource:
        """The TargetScan source, for its representative transcripts and its ORF sequences.

        Built from the registry rather than reimplemented: ``_representative`` already
        encodes which transcript a gene's sites are counted on, and the codon score has
        to be on that same isoform or the two posttx tables describe different molecules.
        """
        return MiRNATargetSource(spec_from_registry(TARGETSCAN_BLOCK, self.registry),
                                 self.gene_index, root=self.root)

    # ----------------------------------------------------------------- lineage --

    def _code_sha(self) -> str:
        try:
            return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                  text=True, check=True, cwd=Path(__file__).parent).stdout.strip()
        except Exception:                                          # noqa: BLE001
            return "unknown"

    def _inputs(self, entries: list[tuple[Path, str]]) -> dict:
        """sha256 and host evidence for each (dest, filename).

        Takes the destination per file rather than assuming ``self.dest``: the codon
        score reads two publishers' bytes -- eLife's CSC table from this block and
        TargetScan's ORFs from another -- and a lineage entry that named only one of them
        would let the other change without anybody noticing.
        """
        cache: dict[Path, dict] = {}
        out: dict = {}
        for dest, name in entries:
            if dest not in cache:
                prov = dest / "PROVENANCE.json"
                cache[dest] = json.loads(prov.read_text()) if prov.exists() else {"selected": []}
            by_name = {f["name"]: f for f in cache[dest].get("selected", [])}
            f = by_name.get(name, {})
            out[name] = {"dest": str(dest), "bytes": f.get("size_bytes"),
                         "host_evidence": f.get("checksum"), "sha256": sha256_of(dest / name)}
        return out

    def _record_lineage(self, key: str, *, what: str, inputs: list[tuple[Path, str]], code: str,
                        counts: dict, out_path: Path, columns: dict | None = None,
                        note: str = "", extra: dict | None = None) -> None:
        """Merge one entry into the derived directory's LINEAGE.json (never overwrite others)."""
        path = self.derived / "LINEAGE.json"
        lineage = json.loads(path.read_text()) if path.exists() else {"schema_version": 2, "entries": {}}
        lineage.setdefault("entries", {})
        entry = {
            "dataset": self.name, "what": what,
            "derives_from": str(self.dest / "PROVENANCE.json"),
            "inputs": self._inputs(inputs), "code": code, "code_sha": self._code_sha(),
            "built": datetime.now(UTC).isoformat(timespec="seconds"),
            "sha256": sha256_of(out_path), "bytes": out_path.stat().st_size,
            "counts": counts, "licence": self.spec.get("license"), "note": note,
        }
        if columns:
            entry["columns"] = columns
        if extra:
            entry.update(extra)
        lineage["entries"][key] = entry
        path.write_text(json.dumps(lineage, indent=2) + "\n")

    # ------------------------------------------------------------- half-life --

    def _halflife_header_row(self, path: Path) -> tuple[int, str]:
        """Which row of the `human` sheet is the header, and what the rows above it say.

        The published file puts a note on row 1 and the header on row 2. That is read
        rather than assumed: passing ``header=1`` to a republished file that dropped the
        note would take the first GENE as the header and lose it, silently and with the
        right-looking number of columns.
        """
        probe = pd.read_excel(path, sheet_name=HALFLIFE_SHEET, header=None,
                              nrows=HALFLIFE_HEADER_SCAN, dtype=str)
        for i in range(len(probe)):
            values = [_squash(v) for v in probe.iloc[i].tolist() if pd.notna(v)]
            if _squash(HALFLIFE_KEY_COLUMN) in values:
                above = " | ".join(str(v).strip() for v in probe.iloc[:i].to_numpy().ravel()
                                   if pd.notna(v))
                return i, above
        raise ValueError(
            f"{HALFLIFE_FILE}: no row in the first {HALFLIFE_HEADER_SCAN} of sheet "
            f"{HALFLIFE_SHEET!r} carries a {HALFLIFE_KEY_COLUMN!r} column; the file's shape "
            "has changed and reading it on a guessed header row would be worse than stopping")

    def halflife_table(self, *, rebuild: bool = False, register_note: str | None = None
                       ) -> pd.DataFrame:
        """One row per human gene: the consensus half-life, and a mean per cell line.

        Cached at ``<derived>/halflife.parquet`` with a LINEAGE.json entry beside it that
        names the PROVENANCE.json, the input file with its sha256, the header row that was
        found, and the full sample-column grouping. ``rebuild=True`` ignores the cache.

        **What the consensus column is.** ``half-life (PC1)`` is the first principal
        component of the filtered, transformed, imputed and quantile-normalised matrix of
        54 human samples -- a measurement summary, not a model prediction. The sheet
        carries no Saluki column. It is recorded as published and NOT z-scored; the
        paper's figures z-score it, and doing that here would silently change the units
        of anything that compares two builds.

        **Why the per-line means exist.** The consensus is only useful to T102 if
        half-life really is gene-intrinsic. The 54 columns name their line, so grouping
        them gives a HEK293 mean over 14 samples and a K562 mean over 8 -- two of our own
        lines, from one file, on one axis. Whether those agree is a measurement, and this
        table is what makes it one.
        """
        src = self._for(HALFLIFE_BLOCK)
        if src is not self:
            return src.halflife_table(rebuild=rebuild, register_note=register_note)

        out_path = self.derived / "halflife.parquet"
        if out_path.exists() and not rebuild:
            return pd.read_parquet(out_path)
        self.derived.mkdir(parents=True, exist_ok=True)

        path = self.dest / HALFLIFE_FILE
        header_row, note_above = self._halflife_header_row(path)
        raw = pd.read_excel(path, sheet_name=HALFLIFE_SHEET, header=header_row)
        raw.columns = [str(c).strip() for c in raw.columns]
        by_squash = {_squash(c): c for c in raw.columns}

        c_gene = by_squash.get(_squash(HALFLIFE_KEY_COLUMN))
        c_symbol = by_squash.get("genename") or by_squash.get("genesymbol")
        c_pc1 = next((c for c in raw.columns if "pc1" in _squash(c)), None)
        missing = [n for n, c in (("Ensembl Gene Id", c_gene), ("Gene name", c_symbol),
                                  ("half-life (PC1)", c_pc1)) if c is None]
        if missing:
            raise ValueError(f"{HALFLIFE_FILE}: sheet {HALFLIFE_SHEET!r} is missing columns "
                             f"{missing}; have {list(raw.columns)[:10]}")

        sample_cols = [c for c in raw.columns if c not in {c_gene, c_symbol, c_pc1}
                       and not str(c).startswith("Unnamed:")]
        by_line, how, unassigned = group_sample_columns(sample_cols)

        table = pd.DataFrame({
            "gene_id": raw[c_gene].astype(str).str.split(".").str[0].str.strip(),
            "symbol": raw[c_symbol].astype(str).str.strip(),
            "halflife_pc1": _num(raw[c_pc1]),
        })
        for line, cols in by_line.items():
            block = raw[cols].apply(_num)
            table[f"halflife_{line}_mean"] = block.mean(axis=1, skipna=True)
            # the per-GENE count of samples that mean was taken over, which is not the
            # number of columns in the group: the matrix is imputed but not complete
            table[f"halflife_{line}_n"] = block.notna().sum(axis=1).astype(int)

        table = table[table["gene_id"].str.startswith("ENSG")].reset_index(drop=True)
        duplicates = int(table["gene_id"].duplicated().sum())
        table = table.drop_duplicates("gene_id").reset_index(drop=True)
        table.to_parquet(out_path, index=False)

        self._record_lineage(
            "halflife-agarwal-kelley-2022/halflife",
            what="one row per human gene: the published half-life (PC1) consensus of Agarwal "
                 "& Kelley 2022 Table S2, plus the mean and per-gene sample count of each "
                 "cell line's columns (T102)",
            inputs=[(self.dest, HALFLIFE_FILE)],
            code="src/sidechain/priors/posttx_stability.py::MRNAStabilitySource.halflife_table",
            counts={
                "sheet_rows": int(len(raw)), "header_row_index": header_row,
                "sample_columns": len(sample_cols),
                "sample_columns_assigned": len(sample_cols) - len(unassigned),
                "sample_columns_unassigned": len(unassigned),
                "lines": len(by_line), "duplicate_gene_ids_dropped": duplicates,
                "genes_out": int(len(table)),
                "genes_with_pc1": int(table["halflife_pc1"].notna().sum()),
            },
            out_path=out_path,
            columns={**HALFLIFE_COLUMNS,
                     **{f"halflife_{line}_mean": f"mean of the {len(cols)} {line} sample columns"
                        for line, cols in by_line.items()},
                     **{f"halflife_{line}_n": f"how many of the {len(cols)} {line} columns this "
                                              "gene has a value in" for line, cols in by_line.items()}},
            note=register_note or "",
            extra={"header_rows_above": note_above,
                   "sample_column_grouping": {line: sorted(cols) for line, cols in by_line.items()},
                   "sample_column_match_rule": how,
                   "sample_columns_unassigned": sorted(unassigned)})
        return table

    # ---------------------------------------------------------- codon score --

    def csc_table(self) -> tuple[pd.DataFrame, dict]:
        """Wu's per-codon CSC table, indexed by codon. Returns (table, resolved column map).

        The codon column is found by CONTENT -- the column whose values are three-letter
        ACGT strings -- because its header is the one field of the file nothing has
        measured, while the six CSC columns are resolved by name from `CSC_COLUMNS` and a
        missing one raises. The output columns are the canonical spellings, so a change in
        the file's punctuation cannot rename a parquet column under a consumer.
        """
        raw = pd.read_csv(self.dest / CSC_FILE)
        raw.columns = [str(c).strip() for c in raw.columns]

        codon_col = None
        for c in raw.columns:
            values = raw[c].astype(str).str.strip().str.upper().str.replace("U", "T", regex=False)
            if values.str.fullmatch(f"[{BASES}]{{3}}").mean() > 0.9:
                codon_col = c
                break
        if codon_col is None:
            raise ValueError(f"{CSC_FILE}: no column holds three-letter codons; "
                             f"have {list(raw.columns)}")

        by_squash = {_squash(c): c for c in raw.columns}
        resolved = {want: by_squash.get(_squash(want)) for want in CSC_COLUMNS}
        missing = [w for w, c in resolved.items() if c is None]
        if missing:
            raise ValueError(f"{CSC_FILE}: missing CSC column(s) {missing}; "
                             f"have {list(raw.columns)}")

        codons = (raw[codon_col].astype(str).str.strip().str.upper()
                  .str.replace("U", "T", regex=False))
        table = pd.DataFrame({want: _num(raw[resolved[want]]) for want in CSC_COLUMNS})
        table.index = codons
        table = table[~table.index.duplicated()]
        return table, {"codon_column": codon_col, **{w: resolved[w] for w in CSC_COLUMNS}}

    def _iter_human_orf_sequences(self, ts: MiRNATargetSource) -> Iterator[tuple[str, str, str, str]]:
        """(transcript_id, gene_id, symbol, ungapped ORF) for every human row of the ORF file.

        Line by line, never pandas: the file is the ORF half of the same 84-way alignment
        ``UTR_Sequences.txt`` comes from (4.6 GB of text), and the human rows are the only ones
        read past the species field.

        **The release's layout, read on the box 2026-09-28: NO header row and three columns** --
        versioned transcript id, species id, gapped lower-case ORF
        (``ENST00000000233.5  9606  atgggcctcacc...``). The gene id and symbol are not in the file,
        so they come from ``Gene_info`` through the transcript id. A file that does open on a
        header is read by column name instead, every spelling TargetScan uses accepted, and an
        unrecognised header raises with the header printed.
        """
        rep = ts._representative().drop_duplicates("transcript_id").set_index("transcript_id")
        z, member = ts._zip_member(ORF_FILE)       # one reader, one zip layout: reuse it
        with z, z.open(member) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8")
            first_line = text.readline().rstrip("\n")
            head = [c.strip() for c in first_line.split("\t")]
            headerless = (len(head) == 3 and head[0].upper().startswith(("ENST", "NM_", "NR_"))
                          and head[1].isdigit())
            if headerless:
                i_tid, i_sp, i_seq, i_gid, i_sym = 0, 1, 2, None, None
                pending = [first_line]
            else:
                low = [c.lower() for c in head]

                def first(*names: str) -> int:
                    for n in names:
                        if n in low:
                            return low.index(n)
                    raise ValueError(f"{ORF_FILE}: no column among {names}; header is {head}")

                i_tid = first("transcript id", "refseq id", "transcript")
                i_gid = first("gene id")
                i_sym = first("gene symbol")
                i_sp = first("species id", "gene tax id")
                i_seq = first("orf sequence", "orf seq", "sequence", "orf")
                pending = []
            width = max(i for i in (i_tid, i_gid, i_sym, i_sp, i_seq) if i is not None)
            for line in itertools.chain(pending, text):
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= width or parts[i_sp].strip() != str(HUMAN):
                    continue
                tid = parts[i_tid].split(".")[0].strip()
                if headerless:
                    if tid not in rep.index:
                        gid, sym = "", ""
                    else:
                        gid, sym = str(rep.at[tid, "gene_id"]), str(rep.at[tid, "symbol"])
                else:
                    gid, sym = parts[i_gid].split(".")[0].strip(), parts[i_sym].strip()
                yield tid, gid, sym, ungap_orf(parts[i_seq])

    def codon_table(self, *, rebuild: bool = False) -> pd.DataFrame:
        """One row per representative transcript: ORF composition and mean CSC per column.

        Cached at ``<derived>/codon_score.parquet``. The ORFs come from TargetScan 8.0's
        ``ORF_Sequences.txt`` restricted to human (9606) rows on the gene's REPRESENTATIVE
        transcript -- the same isoform ``posttx_mirna`` counts miRNA sites on, so the two
        posttx tables describe one molecule. The scores come from Wu's per-codon table.

        **The checks, and why nothing is dropped for failing one.** An ORF is expected to
        be a whole number of codons and to start ATG, and to carry no stop before its
        last codon. All three are counted into LINEAGE.json and written per gene, and a
        gene that fails one still gets a score: the mean CSC of 400 codons is not made
        wrong by a truncated 3' end, and dropping genes on an annotation bar would take
        the decision away from whoever reads the parquet.
        """
        src = self._for(CODON_BLOCK)
        if src is not self:
            return src.codon_table(rebuild=rebuild)

        out_path = self.derived / "codon_score.parquet"
        if out_path.exists() and not rebuild:
            return pd.read_parquet(out_path)
        self.derived.mkdir(parents=True, exist_ok=True)

        csc, resolved = self.csc_table()
        ts = self._targetscan()
        rep = ts._representative()          # the isoform contract, not a copy of it
        rep_ids = set(rep["transcript_id"])

        rows: list[dict] = []
        seen: set[str] = set()
        n_human = 0
        for tid, gid, symbol, seq in self._iter_human_orf_sequences(ts):
            n_human += 1
            if tid not in rep_ids or tid in seen:
                continue
            seen.add(tid)
            rows.append({"transcript_id": tid, "gene_id": gid, "symbol": symbol,
                         **codon_features(seq, csc)})
        if not rows:
            raise ValueError(f"{ORF_FILE}: no human representative-transcript rows")

        table = pd.DataFrame(rows).sort_values("transcript_id").reset_index(drop=True)
        lead = [c for c in CODON_COLUMNS if c in table.columns]
        table = table[lead + [c for c in table.columns if c not in lead]]
        table.to_parquet(out_path, index=False)

        self._record_lineage(
            "codon-optimality-wu2019/codon_score",
            what="one row per TargetScan representative transcript: ORF length, GC3, codon "
                 "counts, and the gene's mean codon stability coefficient under each of Wu "
                 "et al. 2019's six measurement columns (T102)",
            inputs=[(self.dest, CSC_FILE), (ts.dest, ORF_FILE), (ts.dest, GENE_INFO_FILE)],
            code="src/sidechain/priors/posttx_stability.py::MRNAStabilitySource.codon_table",
            counts={
                "orf_rows_human": n_human, "representative_transcripts": int(len(rep)),
                "genes_out": int(len(table)),
                "codons_scored": int(len(csc)),
                "genes_raw_length_multiple_of_3": int(table["len_multiple_of_3"].sum()),
                "genes_starting_atg": int(table["starts_atg"].sum()),
                "genes_with_in_frame_stop": int(table["has_in_frame_stop"].sum()),
                "trailing_nt_median": float(table["trailing_nt"].median()),
                "genes_trailing_over_60nt": int((table["trailing_nt"] > 60).sum()),
                "genes_with_ambiguous_codons": int((table["n_ambiguous_codons"] > 0).sum()),
                "genes_with_csc_endo_mean": int(table["csc_endo_mean"].notna().sum()),
            },
            out_path=out_path, columns=CODON_COLUMNS,
            extra={"csc_columns_resolved": resolved,
                   "csc_endogenous_columns": list(CSC_ENDOGENOUS),
                   "gap_characters_removed": GAP_CHARACTERS,
                   "isoform": "TargetScan's representative transcript per gene (Gene_info), the "
                              "same one posttx_mirna counts sites on",
                   "line_names": "Wu's lines by the paper's Methods -- 293T, HeLa, RPE, K562. "
                                 "Its Key Resources Table crosses the ATCC numbers of K562 and "
                                 "RPE and must not be cited for them."})
        return table

    # ---------------------------------------------------------- decay rates --

    def decay_table(self, *, rebuild: bool = False) -> pd.DataFrame:
        """Wu's per-gene decay rates, one column per Ensembl-keyed sheet, RAW.

        Cached at ``<derived>/decay_rates.parquet``. Four of the six sheets of Figure 1
        source data 1 are read -- ``293T-endogenous``, ``HeLa-endogenous``,
        ``RPE-endogenous`` and ``k562-SLAM-seq``. The two ORFome sheets are skipped
        because they are keyed by a 24-nt library barcode with no gene id, so naming a
        gene in them needs the TRC3 ORF barcode map we do not hold.

        **Nothing is standardised here, on purpose.** The sign convention differs between
        sheets: the Verifier measured a median of -0.12 on the SLAM-seq sheet against
        +0.012 on the 293T endogenous one. Flipping or z-scoring a sheet to make them
        agree would bake one reading of that difference into the parquet; instead each
        sheet's median, sign and range go into LINEAGE.json and a cross-sheet consumer
        standardises per sheet, deliberately.
        """
        src = self._for(CODON_BLOCK)
        if src is not self:
            return src.decay_table(rebuild=rebuild)

        out_path = self.derived / "decay_rates.parquet"
        if out_path.exists() and not rebuild:
            return pd.read_parquet(out_path)
        self.derived.mkdir(parents=True, exist_ok=True)

        book = pd.ExcelFile(self.dest / DECAY_FILE)
        available = {_squash(s): s for s in book.sheet_names}
        missing = [s for s in DECAY_SHEETS if _squash(s) not in available]
        if missing:
            raise ValueError(f"{DECAY_FILE}: missing sheet(s) {missing}; "
                             f"have {book.sheet_names}")

        table: pd.DataFrame | None = None
        stats: dict[str, dict] = {}
        for sheet, (line, method) in DECAY_SHEETS.items():
            raw = book.parse(available[_squash(sheet)])
            raw.columns = [str(c).strip() for c in raw.columns]
            gene_col = next(
                (c for c in raw.columns
                 if raw[c].astype(str).str.strip().str.startswith("ENSG").mean() > 0.5), None)
            decay_col = next((c for c in raw.columns if "decayrate" in _squash(c)), None)
            if gene_col is None or decay_col is None:
                raise ValueError(f"{DECAY_FILE}: sheet {sheet!r} has no "
                                 f"{'Ensembl gene id' if gene_col is None else 'decay_rate'} "
                                 f"column; have {list(raw.columns)}")
            name = f"decay_rate_{re.sub(r'[^0-9A-Za-z]+', '_', sheet)}"
            part = pd.DataFrame({
                "gene_id": raw[gene_col].astype(str).str.split(".").str[0].str.strip(),
                name: _num(raw[decay_col]),
            })
            part = part[part["gene_id"].str.startswith("ENSG")]
            duplicates = int(part["gene_id"].duplicated().sum())
            part = part.drop_duplicates("gene_id")
            values = part[name].dropna()
            stats[sheet] = {
                "column": name, "line": line, "method": method,
                "source_gene_column": gene_col, "source_decay_column": decay_col,
                "rows": int(len(raw)), "genes": int(len(part)),
                "duplicate_gene_ids_dropped": duplicates,
                "median": float(values.median()) if len(values) else None,
                "median_sign": ("positive" if values.median() > 0 else
                                "negative" if values.median() < 0 else "zero") if len(values) else None,
                "min": float(values.min()) if len(values) else None,
                "max": float(values.max()) if len(values) else None,
            }
            table = part if table is None else table.merge(part, on="gene_id", how="outer")

        table = table.sort_values("gene_id").reset_index(drop=True)
        table.to_parquet(out_path, index=False)

        self._record_lineage(
            "codon-optimality-wu2019/decay_rates",
            what="per-gene mRNA decay rates from the four Ensembl-keyed sheets of Wu et al. "
                 "2019 Figure 1 source data 1, raw and unstandardised (T102)",
            inputs=[(self.dest, DECAY_FILE)],
            code="src/sidechain/priors/posttx_stability.py::MRNAStabilitySource.decay_table",
            counts={"sheets_read": len(DECAY_SHEETS), "sheets_skipped": len(DECAY_SHEETS_SKIPPED),
                    "genes_out": int(len(table)),
                    **{f"genes_{v['column']}": v["genes"] for v in stats.values()}},
            out_path=out_path,
            extra={"sheets": stats, "sheets_skipped": DECAY_SHEETS_SKIPPED,
                   "sign_convention": "raw as published. The medians differ in SIGN between "
                                      "sheets, so a cross-sheet comparison must standardise per "
                                      "sheet first; this table deliberately does not.",
                   "line_names": "by Wu's Methods -- 293T, HeLa, RPE, K562 -- never by its Key "
                                 "Resources Table, which crosses the ATCC numbers of K562 and RPE."})
        return table

    # ------------------------------------------------------------ the feature --

    def build(self) -> PriorArtifact:
        """Two measured per-gene columns, aligned to the master gene space.

        ``features[:, 0]`` is ``halflife_pc1`` from the half-life block, ``features[:, 1]``
        ``csc_endo_mean`` from the codon block. Works from either block's spec: whichever
        one this source is, the other is built from the registry (`_for`).

        **Zero is a real value here, so read the mask.** A gene with no row in a table
        gets 0.0 in that column, exactly as the PriorArtifact contract requires -- but a
        CSC mean of 0.0 means "an average codon in this ORF is neutral", which is a
        measurement, not an absence. ``meta["has_value"]`` is a (n_genes, 2) boolean array
        saying which entries are measurements; a consumer that ignores it is training on
        18,533 genes of which several thousand are silently zero.

        The values are paired to positions one gene at a time rather than through
        `to_positions`, for the reason that method's docstring gives: two independent
        lookups drop different genes and mis-pair the survivors.
        """
        halflife = self._for(HALFLIFE_BLOCK).halflife_table()
        codon = self._for(CODON_BLOCK).codon_table()

        n_genes = len(self.gene_index)
        names = ["halflife_pc1", "csc_endo_mean"]
        features = np.zeros((n_genes, len(names)), dtype=np.float32)
        has_value = np.zeros((n_genes, len(names)), dtype=bool)
        candidates = []
        for column, (table, col) in enumerate(((halflife, "halflife_pc1"),
                                               (codon, "csc_endo_mean"))):
            ids = table["gene_id"].astype(str).to_numpy()
            values = _num(table[col]).to_numpy(dtype=float)
            candidates.append(int(len(ids)))
            for gene, value in zip(ids, values):
                position = self.gene_index.get(gene)
                if position is None or not np.isfinite(value):
                    continue
                features[position, column] = np.float32(value)
                has_value[position, column] = True

        return PriorArtifact(
            kind="node_feature", relation=self.relation, layer=self.layer, features=features,
            meta={"feature_names": names, "has_value": has_value,
                  "relations": {"halflife_pc1": "mrna_halflife", "csc_endo_mean": "codon_optimality"},
                  "sources": {"halflife_pc1": HALFLIFE_BLOCK, "csc_endo_mean": CODON_BLOCK},
                  "candidate_genes": dict(zip(names, candidates)),
                  "genes_with_value": {n: int(has_value[:, i].sum()) for i, n in enumerate(names)},
                  "genes_with_any_value": int(has_value.any(axis=1).sum()),
                  "zero_is_a_real_value": True},
        )
