"""Post-transcriptional layer: miRNA -> target repression edges (Saber's edge).

Primary source miRBind2 (sequence-only binding + repression score); TargetScan/
miRDB as toggleable alternates. Directed edges onto the target gene, weighted by
repression strength — refines the *magnitude* of downstream deltas the trans graph
predicts (mRNA stability via 3'UTR).

**What is built (T102).** Four tables from TargetScan Human 8.0, every one cached as a
parquet under ``derived/targetscan-vert_80/`` with an entry in the LINEAGE.json beside it:

* ``utr_load_table()`` (2026-09-25) -- one row per gene: representative-transcript 3'UTR
  length, conserved-site counts, summed context++. The per-gene *load* T102 first read.
* ``site_table()`` (2026-09-28) -- one row per SITE of a conserved miRNA family on the
  representative transcript, with its 1-based position on the ungapped human 3'UTR, seed
  type, PCT, whether it is a default (conserved) prediction, and the site's weighted
  context++ where a member miRNA's row joins. The positions are what Saber's NAR 2016
  cooperation / competition analysis needs (``research/reading/hafezqorani-2016-rbp-mirna-ptr.md`` §4).
* ``edge_table()`` (2026-09-28) -- one row per (miRNA family, gene): the EDGE form of the
  layer, site counts by type, cumulative weighted context++, aggregate PCT. ``scope="all"``
  reads the all-predictions file (nonconserved families too). ``build()`` turns the default
  scope into a bipartite ``PriorArtifact`` (family -> gene) through
  ``PriorSource.to_bipartite_edge_index`` -- one mask for both endpoints, never a dense block.
* ``utr_sequence_table()`` (2026-09-28) -- one row per representative transcript from the
  human rows of ``UTR_Sequences.txt``: length, base and dinucleotide composition (the NAR
  2016 paper's strongest half-life feature class), consensus-motif counts for the RBPs that
  paper leaned on (``MOTIFS``), and the cooperation / competition counts against the miRNA
  sites: sites with an RBP motif within 200 nt, sites overlapped by one. Motif scans are the
  sequence-only stand-in for the paper's PFMs until a motif catalogue is ingested.

``utr_features_table()`` joins the load and the sequence tables into the widened per-gene
table. Every table is keyed by Ensembl gene id (version stripped) and by TargetScan's 2018
symbol; our 2026 axes spell 2024 symbols, so the join goes through the corpora's own gene
tables (``scripts/t102_utr_load.py::symbol_to_ensg``).

**The fetch.** ``fetch()`` brings the block's files through the ADR 0003 gate
(``sidechain.ingest.fetch.run_gate``: probe, gate, PROVENANCE.json before any byte lands),
then downloads each in-process with a resumable ``.part``. The registry block lists the
whole Release 8.0 download since 2026-09-28 (3.0 GB, 19 files, Saber's ask); every table
above filters the all-species text to human (9606) on read, in chunks, so no file is ever
held whole -- the UTR sequence file alone is over 4 GB of text.

Why a fetch here rather than a ``datasets.yaml`` block: ``data_sources.yaml`` is the prior
registry and the ``targetscan`` block already lived there. The block carries the same
``host / record / files / dest / budget_gb / license`` fields a corpus block does, and
``fetch()`` hands it to ``run_gate`` so a prior enters under exactly the rules a corpus does.
The host publishes no checksum (``probe_https``), so the sha256 of what landed is recorded in
LINEAGE.json.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import urllib.request
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sidechain.priors.base import PriorArtifact, PriorSource
from sidechain.utils.paths import resolve_config

DATA_ROOT = Path.home() / "data" / "sidechain"
REGISTRY = "configs/data_sources.yaml"
HUMAN = 9606
USER_AGENT = "sidechain-ingest/0.1 (+https://github.com/saberhq/sidechain)"

#: The per-gene columns `utr_load_table` produces, and what each one is.
UTR_LOAD_COLUMNS = {
    "symbol": "TargetScan's gene symbol (GENCODE-era, ~2018)",
    "gene_id": "Ensembl gene id, version stripped",
    "transcript_id": "the representative transcript the sites are counted on",
    "utr_len": "3'UTR length in nt: sum of the transcript's 3'UTR exon widths, end - start + 1 "
               "per exon (hg19 GFF, 1-based inclusive; one short per exon before 2026-09-28)",
    "n_cons_sites": "conserved sites of conserved miRNA families, all types, summed over families",
    "n_cons_8mer": "of those, 8mer sites",
    "n_cons_7mer": "of those, 7mer-m8 plus 7mer-1a sites",
    "n_families": "conserved miRNA families with at least one conserved site",
    "n_noncons_sites": "nonconserved sites of conserved families that ALSO have a conserved site on "
                       "this gene: the default-predictions file has no row for a (transcript, "
                       "family) pair without one, so this is about 6 % of such sites; the full "
                       "count is `n_noncons_sites_all` in utr_features (from sites.parquet)",
    "context_score": "sum over families of TargetScan's cumulative weighted context++ score "
                     "(negative; more negative = stronger predicted repression)",
    "aggregate_pct_max": "the largest aggregate PCT over the gene's families",
    "sites_per_kb": "n_cons_sites per kb of 3'UTR",
}

#: One row per miRNA site (`site_table`).
SITE_COLUMNS = {
    "family": "TargetScan miRNA family name (the `miR Family` column), e.g. miR-23-3p",
    "gene_id": "Ensembl gene id, version stripped",
    "symbol": "TargetScan's gene symbol",
    "transcript_id": "the transcript the site sits on, version stripped",
    "utr_start": "start of the site on the ungapped human 3'UTR, as the family files print it "
                 "(1-based; `check_site_coordinates` verifies the convention against the sequence)",
    "utr_end": "end of the site, inclusive in the same convention",
    "msa_start": "start in the 84-way alignment (gapped) coordinates",
    "msa_end": "end in the alignment coordinates",
    "site_type": "seed match as TargetScan spells it: 8mer, 7mer-m8, 7mer-a1",
    "pct": "probability of conserved targeting; NaN where the file says NULL",
    "conserved_site": "True when the site is a default prediction (Predicted_Targets_Info: a "
                      "conserved site of a conserved family); False for the nonconserved sites "
                      "of conserved families that Conserved_Family_Info adds",
    "context_pp": "weighted context++ of the site, the most negative over the family's member "
                  "miRNAs in Conserved_Site_Context_Scores, joined at the same human UTR "
                  "coordinates; NaN where no member row joined",
    "representative": "the transcript is the gene's representative one (Gene_info)",
}

#: One row per (miRNA family, gene) (`edge_table`).
EDGE_COLUMNS = {
    "family": "TargetScan miRNA family NAME (miR_Family_Info, e.g. miR-23-3p); the seed itself "
              "where no human family row carries that seed",
    "seed_m8": "the family's seed+m8 (nt 2-8), which is how the Summary_Counts files key a family",
    "gene_id": "Ensembl gene id, version stripped",
    "symbol": "TargetScan's gene symbol",
    "transcript_id": "the representative transcript the counts are on",
    "family_conserved": "True for a conserved family (the default-predictions file); False for a "
                        "nonconserved family, present only under scope='all'",
    "n_cons_sites": "conserved sites, all types",
    "n_cons_8mer": "conserved 8mer sites",
    "n_cons_7mer_m8": "conserved 7mer-m8 sites",
    "n_cons_7mer_1a": "conserved 7mer-1a sites",
    "n_noncons_sites": "nonconserved sites, all types",
    "n_6mer": "6mer sites",
    "context_score_total": "TargetScan's total context++ score for the pair",
    "context_score_weighted": "cumulative weighted context++ (the repression prediction, log2, negative)",
    "aggregate_pct": "aggregate PCT of the pair; NaN where NULL",
    "representative_mirna": "the member miRNA TargetScan names for the family",
}

#: Consensus motifs for the RBPs the NAR 2016 paper leaned on, on the U alphabet of
#: TargetScan's UTR sequences. name -> (regex, what it stands in for, where the consensus
#: comes from). These are sequence-only stand-ins for the paper's RNAcompete / RBPDB PFMs;
#: a proper catalogue (CisBP-RNA, RBNS) is the posttx survey's question (2026-09-28).
#: Fixed-length motifs are matched with a lookahead so overlapping copies count (the class II
#: ARE is overlapping AUUUA repeats); the tract motifs (poly-U, CA repeats) count maximal runs.
MOTIFS: dict[str, tuple[str, str, str]] = {
    "pum": ("(?=UGUA[ACU]AUA)", "Pumilio response element (PUM1/PUM2)",
            "Kedde 2010 Nat Cell Biol; the NAR 2016 PUM1(2) site set"),
    "are_auuua": ("(?=AUUUA)", "AU-rich element pentamer (ZFP36, AUF1, KHSRP, TIA1)", "Chen & Shyu 1995"),
    "are_heptamer": ("(?=UAUUUAU)", "the ARE heptamer core (UAUUUAU)", "Zubiaga 1995"),
    "polyu": ("U{5,}", "U-rich tract (hnRNP C, HuR, TIA1)", "Konig 2010; Mukherjee 2011"),
    "msi": ("(?=[GA]U{1,3}AGU)", "Musashi element (MSI1/MSI2)", "Zearfoss 2014"),
    "qki": ("(?=ACUAA[CU])", "QKI response element core", "Galarneau & Richard 2005"),
    "gre": ("(?=UGU[UG]UGU)", "GU-rich element (CELF1)", "Vlasova 2008"),
    "ca_repeat": ("(?:CA){4,}", "CA repeats (hnRNP L)", "Hui 2005"),
}
COOP_WINDOW_NT = 200          # the NAR 2016 paper's co-occurrence distance
NUCLEOTIDES = "ACGU"
DINUCLEOTIDES = [a + b for a in NUCLEOTIDES for b in NUCLEOTIDES]
_RC = str.maketrans("ACGU", "UGCA")


def revcomp_rna(seq: str) -> str:
    return seq.translate(_RC)[::-1]


def spec_from_registry(name: str = "targetscan", registry: str | Path = REGISTRY) -> dict:
    """The named block of the prior registry, enabled or not.

    `load_registry` skips a disabled block on purpose (a shelved source may name a loader
    that is not written). This reads the block regardless, because fetching a prior's
    raw table and building an edge layer from it are different decisions: the
    ``targetscan`` block stays ``enabled: false`` while its tables are in use.
    """
    cfg = yaml.safe_load(resolve_config(registry).read_text())
    for spec in cfg.get("sources", []):
        if spec.get("name") == name:
            return spec
    raise KeyError(f"no source named {name!r} in {registry}")


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


def fetch_gated_files(spec: dict, root: Path, *, refresh: bool = False, progress: bool = False,
                      registry: str | Path = REGISTRY) -> Path:
    """Bring one prior-registry block's files onto disk. Idempotent, resumable.

    Order: ``run_gate`` (probe -> gate -> PROVENANCE.json) exactly as
    ``sidechain.ingest.fetch`` runs it for a corpus, then each selected file that is not
    already on disk at the probed size is downloaded. A ``.part`` left by an interrupted
    run is resumed with a Range request (the hosts we use honour them); a host that
    ignores the range restarts the file rather than appending twice, and a short file is
    deleted rather than kept. Returns the destination directory. ``refresh=True`` accepts
    a changed block or upstream state and rewrites PROVENANCE.json -- it is what widening
    a file list needs.

    Module-level rather than a method because every prior block that carries
    ``host / record / files / dest`` fetches identically: ``MiRNATargetSource`` (TargetScan)
    and ``MRNAStabilitySource`` (``posttx_stability``, the Springer and eLife blocks) both
    call it, and a second copy of this loop is how the two would drift.
    """
    from sidechain.ingest.fetch import run_gate

    name = spec.get("name", "<unnamed>")
    if "host" not in spec:
        raise ValueError(f"{name}: the registry block declares no host/record/files, "
                         "so there is nothing to fetch through the gate")
    record, selected, dest = run_gate(spec, root, refresh=refresh, config=registry)
    for f in selected:
        target = dest / f.name
        if target.exists() and target.stat().st_size == f.size_bytes:
            continue
        tmp = target.with_suffix(target.suffix + ".part")
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(f.url, headers=headers)
        with urllib.request.urlopen(req, timeout=300) as resp:
            if have and resp.status != 206:
                # the host ignored the range: start over rather than append twice
                have = 0
                tmp.unlink()
            with tmp.open("ab" if have else "wb") as out:
                done = have
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if progress and done % (64 << 20) < (1 << 20):
                        print(f"  {f.name}: {done / 1e6:.0f} / {f.size_bytes / 1e6:.0f} MB",
                              flush=True)
        got = tmp.stat().st_size
        if got != f.size_bytes:
            tmp.unlink()
            raise RuntimeError(f"{f.name}: downloaded {got} bytes, the probe said "
                               f"{f.size_bytes}; refusing to keep a short file")
        tmp.replace(target)
    return dest


def _strip_version(s: pd.Series) -> pd.Series:
    return s.astype(str).str.split(".").str[0].str.strip()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


class MiRNATargetSource(PriorSource):
    """TargetScan (and, unbuilt, miRBind2) as a prior source.

    Constructed like every PriorSource -- ``(spec, gene_index)`` -- but the table entry
    points need no gene index: ``fetch()``, ``utr_load_table()``, ``site_table()``,
    ``edge_table()`` and ``utr_sequence_table()`` work on the spec alone, so ``gene_index``
    may be an empty dict for those. ``build()`` needs it (Ensembl gene id -> position).
    """

    def __init__(self, spec: dict, gene_index: dict[str, int], root: Path = DATA_ROOT):
        super().__init__(spec, gene_index)
        self.root = Path(root).expanduser()

    # ------------------------------------------------------------------ fetch --

    @property
    def dest(self) -> Path:
        return self.root / self.spec["dest"]

    @property
    def derived(self) -> Path:
        return self.root / self.spec["derived"]

    def fetch(self, *, refresh: bool = False, progress: bool = False) -> Path:
        """Gate, record provenance, then download the block's files. Idempotent, resumable.

        The loop itself is `fetch_gated_files`, shared with `MRNAStabilitySource`; the
        order it enforces (probe -> gate -> PROVENANCE.json -> bytes) is that function's
        docstring.
        """
        return fetch_gated_files(self.spec, self.root, refresh=refresh, progress=progress)

    # ---------------------------------------------------------------- readers --

    def _zip_member(self, name: str) -> tuple[zipfile.ZipFile, str]:
        z = zipfile.ZipFile(self.dest / name)
        members = [m for m in z.namelist() if not m.endswith("/")]
        if len(members) != 1:
            z.close()
            raise ValueError(f"{name}: expected one member, found {members}")
        return z, members[0]

    def _read_zipped_table(self, name: str, *, skip_prefixes: tuple[str, ...] = (),
                           **read_kwargs) -> pd.DataFrame:
        """The one text file inside `name` (TargetScan zips one table per zip), whole.

        `skip_prefixes` drops leading lines that are not rows -- the hg19 GFF opens with a
        UCSC `browser` line and a `track` line, which are neither a header nor a comment.
        Only for the small files; the big ones go through `_iter_human_chunks`.
        """
        z, member = self._zip_member(name)
        with z, z.open(member) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8")
            if skip_prefixes:
                text = io.StringIO("".join(
                    line for line in text if not line.startswith(skip_prefixes)))
            return pd.read_csv(text, sep="\t", **read_kwargs)

    def _iter_chunks(self, name: str, species_col: str, *, usecols: list[str] | None = None,
                     chunksize: int = 250_000) -> Iterator[pd.DataFrame]:
        """An all-species table one chunk at a time, as strings, columns stripped.

        Every large TargetScan table lists all 84 species; the human share is a few per
        cent. Reading in chunks with `usecols` keeps the peak at a few hundred MB whatever
        the file is (the nonconserved site table is 3.8 GB of text).
        """
        z, member = self._zip_member(name)
        with z, z.open(member) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8")
            header = list(pd.read_csv(io.StringIO(text.readline()), sep="\t", nrows=0).columns)
            cols = {c.strip(): c for c in header}
            if species_col not in cols:
                raise ValueError(f"{name}: no column {species_col!r}; have {list(cols)}")
            keep = None if usecols is None else sorted({cols[c] for c in usecols} | {cols[species_col]},
                                                       key=header.index)
            reader = pd.read_csv(text, sep="\t", header=None, names=header, dtype=str,
                                 usecols=keep, chunksize=chunksize, na_filter=False)
            for chunk in reader:
                chunk.columns = [c.strip() for c in chunk.columns]
                yield chunk

    def _human_table(self, name: str, species_col: str, *, usecols: list[str] | None = None,
                     chunksize: int = 250_000) -> tuple[pd.DataFrame, int]:
        """All human (9606) rows of a table, plus how many rows of ALL species were read."""
        parts, n = [], 0
        for chunk in self._iter_chunks(name, species_col, usecols=usecols, chunksize=chunksize):
            n += len(chunk)
            sub = chunk[_num(chunk[species_col]) == HUMAN]
            if len(sub):
                parts.append(sub.reset_index(drop=True))
        if not parts:
            raise ValueError(f"{name}: no human ({HUMAN}) rows")
        return pd.concat(parts, ignore_index=True), n

    def _iter_human_utr_sequences(self) -> Iterator[tuple[str, str, str, str]]:
        """(transcript_id, gene_id, symbol, ungapped sequence) for every human row.

        Line by line, never pandas: the file is over 4 GB of gapped alignment text and the
        human rows are the only ones read past the species field.
        """
        z, member = self._zip_member("UTR_Sequences.txt.zip")
        with z, z.open(member) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8")
            header = [c.strip() for c in text.readline().rstrip("\n").split("\t")]
            low = [c.lower() for c in header]
            try:
                i_tid = next(i for i, c in enumerate(low) if c in ("refseq id", "transcript id"))
                i_gid = low.index("gene id")
                i_sym = low.index("gene symbol")
                i_sp = low.index("species id")
                i_seq = low.index("utr sequence")
            except (StopIteration, ValueError) as exc:
                raise ValueError(f"UTR_Sequences.txt: unexpected columns {header}") from exc
            for line in text:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= i_seq or parts[i_sp].strip() != str(HUMAN):
                    continue
                seq = parts[i_seq].replace("-", "").upper().replace("T", "U")
                yield (parts[i_tid].split(".")[0].strip(), parts[i_gid].split(".")[0].strip(),
                       parts[i_sym].strip(), seq)

    def _representative(self) -> pd.DataFrame:
        """transcript_id, gene_id, symbol of every human representative transcript."""
        info = self._read_zipped_table("Gene_info.txt.zip", dtype=str)
        info.columns = [c.strip() for c in info.columns]
        col = {c.lower(): c for c in info.columns}
        species = col.get("species id")
        rep = next((c for c in info.columns if c.lower().startswith("representative")), None)
        if species is None or rep is None:
            raise ValueError(f"Gene_info.txt: unexpected columns {list(info.columns)}")
        info = info[info[species].astype(int) == HUMAN]
        info = info[info[rep].astype(str).str.strip().isin({"1", "yes", "Yes", "TRUE", "True"})]
        return pd.DataFrame({
            "transcript_id": _strip_version(info[col["transcript id"]]),
            "gene_id": _strip_version(info[col["gene id"]]),
            "symbol": info[col["gene symbol"]].str.strip(),
        }).drop_duplicates("transcript_id").reset_index(drop=True)

    def family_table(self) -> pd.DataFrame:
        """Human miRNA families: family, seed_m8, mirbase_id, conservation class."""
        fam = self._read_zipped_table("miR_Family_Info.txt.zip", dtype=str)
        fam.columns = [c.strip() for c in fam.columns]
        col = {c.lower(): c for c in fam.columns}
        need = ["mir family", "seed+m8", "species id", "mirbase id"]
        missing = [k for k in need if k not in col]
        if missing:
            raise ValueError(f"miR_Family_Info.txt: missing columns {missing}; have {list(fam.columns)}")
        fam = fam[_num(fam[col["species id"]]) == HUMAN]
        out = pd.DataFrame({
            "family": fam[col["mir family"]].str.strip(),
            "seed_m8": fam[col["seed+m8"]].str.strip().str.upper().str.replace("T", "U"),
            "mirbase_id": fam[col["mirbase id"]].str.strip(),
        })
        cons = col.get("family conservation?")
        out["family_conservation"] = _num(fam[cons]).to_numpy() if cons else np.nan
        return out.reset_index(drop=True)

    # ----------------------------------------------------------------- lineage --

    def _code_sha(self) -> str:
        try:
            return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                  text=True, check=True, cwd=Path(__file__).parent).stdout.strip()
        except Exception:                                          # noqa: BLE001
            return "unknown"

    def _inputs(self, names: list[str]) -> dict:
        prov = json.loads((self.dest / "PROVENANCE.json").read_text())
        by_name = {f["name"]: f for f in prov["selected"]}
        out = {}
        for n in names:
            f = by_name.get(n, {})
            out[n] = {"bytes": f.get("size_bytes"), "host_evidence": f.get("checksum"),
                      "sha256": sha256_of(self.dest / n)}
        return out

    def _record_lineage(self, key: str, *, what: str, inputs: list[str], code: str,
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

    # ------------------------------------------------------------- the table --

    def utr_load_table(self, *, rebuild: bool = False, register_note: str | None = None
                       ) -> pd.DataFrame:
        """One row per human gene: 3'UTR length, conserved-site counts, summed context++.

        Cached at ``<derived>/utr_load.parquet`` with a LINEAGE.json entry beside it that
        names the PROVENANCE.json, the three input files with their sha256, the code, and
        the row counts at each step. ``rebuild=True`` ignores the cache.

        **Which transcript.** TargetScan counts sites on one *representative* transcript
        per gene (its Gene_info flag), and the hg19 GFF carries the 3'UTR exons of those
        same transcripts, so length and sites are read off one isoform by construction.
        That is the annotation-fixed choice the idea file prices in: our corpora carry
        gene-level counts only, so which isoform a cell actually made is not recoverable.

        **Which sites.** The *default predictions* file lists conserved sites of conserved
        miRNA families per (transcript, family) with counts by site type; it also carries
        the nonconserved sites of those families. ``n_cons_sites`` sums the conserved
        ones over families; ``context_score`` sums the cumulative weighted context++
        score, TargetScan's predicted repression on the log2 scale (negative). A gene
        with no row in that file has zero conserved sites, not a missing value -- it is
        in Gene_info and the GFF, so its UTR length is known and its load is 0.
        """
        out_path = self.derived / "utr_load.parquet"
        if out_path.exists() and not rebuild:
            return pd.read_parquet(out_path)
        self.derived.mkdir(parents=True, exist_ok=True)

        genes = self._representative()
        n_info = int(len(self._read_zipped_table("Gene_info.txt.zip", dtype=str)))

        # 3'UTR length: the GFF's coordinates are 1-based INCLUSIVE, as a GFF's are, so an
        # exon's width is end - start + 1. The publisher's score column equals end - start
        # (one short per exon) and the first build of 2026-09-25 took it as the width; the
        # 2026-09-28 review measured the ungapped UTR sequence one nucleotide longer than
        # that per exon on 19,426 of 19,426 genes, which settles it. The agreement count
        # of the score column with end - start is still recorded, as evidence of the file.
        gff = self._read_zipped_table("TSHuman_7_hg19_3UTRs.gff.zip", header=None,
                                      comment="#", dtype=str,
                                      skip_prefixes=("browser", "track"))
        if gff.shape[1] < 9:
            raise ValueError(f"3'UTR GFF: expected 9 columns, got {gff.shape[1]}")
        attr = gff.iloc[:, 8].astype(str)
        tid = attr.str.extract(r"(ENST\d+)")[0]
        if tid.isna().all():
            raise ValueError("3'UTR GFF: no ENST ids in the attribute column")
        start = gff.iloc[:, 3].astype(int)
        end = gff.iloc[:, 4].astype(int)
        score = pd.to_numeric(gff.iloc[:, 5], errors="coerce")
        agree = np.isfinite(score) & (score == (end - start))
        width = (end - start + 1).to_numpy().astype(np.int64)
        utr = pd.DataFrame({"transcript_id": tid, "width": width}).dropna()
        utr_len = utr.groupby("transcript_id")["width"].sum().rename("utr_len")

        counts, n_counts = self._summary_counts("Summary_Counts.default_predictions.txt.zip")
        # a family with sites on the representative transcript counts once, whatever its name
        g = counts.groupby("transcript_id")
        load = pd.DataFrame({
            "n_cons_sites": g["n_cons_sites"].sum(),
            "n_cons_8mer": g["n_cons_8mer"].sum(),
            "n_cons_7mer": (g["n_cons_7mer_m8"].sum() + g["n_cons_7mer_1a"].sum()),
            "n_families": g["n_cons_sites"].apply(lambda s: int((s > 0).sum())),
            "n_noncons_sites": g["n_noncons_sites"].sum(),
            "context_score": g["context_score_weighted"].sum(min_count=1),
            "aggregate_pct_max": g["aggregate_pct"].max(),
        })

        table = (genes.merge(utr_len, left_on="transcript_id", right_index=True, how="left")
                      .merge(load, left_on="transcript_id", right_index=True, how="left"))
        for c in ("n_cons_sites", "n_cons_8mer", "n_cons_7mer", "n_families", "n_noncons_sites"):
            table[c] = table[c].fillna(0).astype(int)
        table["context_score"] = table["context_score"].fillna(0.0)
        n_no_utr = int(table["utr_len"].isna().sum())
        table = table.dropna(subset=["utr_len"]).copy()
        table["utr_len"] = table["utr_len"].astype(int)
        table["sites_per_kb"] = table["n_cons_sites"] / (table["utr_len"] / 1000.0)
        # One row per SYMBOL: a symbol carried by two representative transcripts (two
        # Ensembl genes sharing a name) keeps the longer UTR's row, and the choice is
        # counted rather than silent.
        dup_symbols = int(table["symbol"].duplicated().sum())
        table = (table.sort_values(["symbol", "utr_len"], ascending=[True, False])
                      .drop_duplicates("symbol").reset_index(drop=True))
        table = table[list(UTR_LOAD_COLUMNS)]
        table.to_parquet(out_path, index=False)
        self._record_lineage(
            "targetscan-vert_80/utr_load",
            what="one row per human gene: representative-transcript 3'UTR length, conserved "
                 "miRNA site counts, summed context++ score (T102)",
            inputs=["Gene_info.txt.zip", "TSHuman_7_hg19_3UTRs.gff.zip",
                    "Summary_Counts.default_predictions.txt.zip"],
            code="src/sidechain/priors/posttx_mirna.py::MiRNATargetSource.utr_load_table",
            counts={
                "gene_info_rows": n_info, "representative_transcripts": int(len(genes)),
                "gff_rows": int(len(gff)), "gff_transcripts": int(utr_len.size),
                "gff_score_equals_width_on_rows": int(agree.sum()),
                "summary_counts_rows": n_counts, "summary_counts_human_rows": int(len(counts)),
                "transcripts_with_sites": int(len(load)),
                "dropped_no_utr_in_gff": n_no_utr, "symbols_deduplicated": dup_symbols,
                "genes_out": int(len(table)),
            },
            out_path=out_path, columns=UTR_LOAD_COLUMNS, note=register_note or "")
        return table

    # ------------------------------------------------------------ edge form --

    def _summary_counts(self, name: str) -> tuple[pd.DataFrame, int]:
        """Human rows of a Summary_Counts file, typed, one row per (transcript, family)."""
        # NB: the `miRNA family` column of the Summary_Counts files holds the SEED+m8
        # (GCAGCAU), not the family name the site files use (miR-103-3p/107); `edge_table`
        # maps it through miR_Family_Info.
        need = {"transcript id": "transcript_id", "gene symbol": "symbol", "mirna family": "seed_m8",
                "total num conserved sites": "n_cons_sites",
                "number of conserved 8mer sites": "n_cons_8mer",
                "number of conserved 7mer-m8 sites": "n_cons_7mer_m8",
                "number of conserved 7mer-1a sites": "n_cons_7mer_1a",
                "total num nonconserved sites": "n_noncons_sites",
                "number of 6mer sites": "n_6mer", "representative mirna": "representative_mirna",
                "total context++ score": "context_score_total",
                "cumulative weighted context++ score": "context_score_weighted",
                "aggregate pct": "aggregate_pct"}
        z, member = self._zip_member(name)
        with z, z.open(member) as fh:
            header = [c.strip() for c in io.TextIOWrapper(fh, encoding="utf-8").readline().rstrip("\n").split("\t")]
        low = {c.lower(): c for c in header}
        missing = [k for k in need if k not in low]
        if missing:
            raise ValueError(f"{name}: missing columns {missing}; have {header}")
        usecols = [low[k] for k in need]
        raw, n_all = self._human_table(name, "Species ID", usecols=usecols)
        df = pd.DataFrame({v: raw[low[k]] for k, v in need.items()})
        df["transcript_id"] = _strip_version(df["transcript_id"])
        df["symbol"] = df["symbol"].str.strip()
        df["seed_m8"] = df["seed_m8"].str.strip().str.upper().str.replace("T", "U")
        for c in ("n_cons_sites", "n_cons_8mer", "n_cons_7mer_m8", "n_cons_7mer_1a",
                  "n_noncons_sites", "n_6mer"):
            df[c] = _num(df[c]).fillna(0).astype(int)
        for c in ("context_score_total", "context_score_weighted", "aggregate_pct"):
            df[c] = _num(df[c])
        return df, n_all

    def edge_table(self, *, scope: str = "default", rebuild: bool = False) -> pd.DataFrame:
        """One row per (miRNA family, gene): the EDGE form of the layer.

        ``scope="default"`` reads the default-predictions summary (conserved families; the
        table T102's load came from) and marks every row ``family_conserved=True``.
        ``scope="all"`` streams the 2.4 GB all-predictions summary, which adds the
        nonconserved families (``family_conserved=False`` where the family is not in the
        default file). Both keep the representative transcript only, so a (family, gene)
        pair appears once. Cached at ``<derived>/edges_<scope>.parquet``.
        """
        if scope not in ("default", "all"):
            raise ValueError(f"scope must be 'default' or 'all', got {scope!r}")
        out_path = self.derived / f"edges_{scope}.parquet"
        if out_path.exists() and not rebuild:
            return pd.read_parquet(out_path)
        self.derived.mkdir(parents=True, exist_ok=True)
        rep = self._representative()
        name = ("Summary_Counts.default_predictions.txt.zip" if scope == "default"
                else "Summary_Counts.all_predictions.txt.zip")
        df, n_all = self._summary_counts(name)
        n_human = len(df)
        df = df.merge(rep[["transcript_id", "gene_id"]], on="transcript_id", how="inner")
        n_rep = len(df)
        # seed -> family name through the human family rows; a seed with no human family
        # row keeps the seed as its name and is counted
        fam = self.family_table().drop_duplicates("seed_m8")
        seed2name = dict(zip(fam["seed_m8"], fam["family"]))
        df["family"] = df["seed_m8"].map(seed2name)
        unmapped = int(df["family"].isna().sum())
        df["family"] = df["family"].fillna(df["seed_m8"])
        if scope == "default":
            df["family_conserved"] = True
        else:
            default, _ = self._summary_counts("Summary_Counts.default_predictions.txt.zip")
            df["family_conserved"] = df["seed_m8"].isin(set(default["seed_m8"]))
        dup = int(df.duplicated(["family", "gene_id"]).sum())
        df = df.drop_duplicates(["family", "gene_id"]).reset_index(drop=True)
        df = df[list(EDGE_COLUMNS)]
        df.to_parquet(out_path, index=False)
        self._record_lineage(
            f"targetscan-vert_80/edges_{scope}",
            what=f"one row per (miRNA family, gene) on the representative transcript, scope {scope} (T102)",
            inputs=["Gene_info.txt.zip", "miR_Family_Info.txt.zip", name]
                   + (["Summary_Counts.default_predictions.txt.zip"] if scope == "all" else []),
            code="src/sidechain/priors/posttx_mirna.py::MiRNATargetSource.edge_table",
            counts={"rows_all_species": n_all, "rows_human": n_human,
                    "rows_on_representative_transcript": n_rep, "pairs_deduplicated": dup,
                    "seeds_without_a_human_family_row": unmapped,
                    "edges_out": int(len(df)), "families": int(df["family"].nunique()),
                    "genes": int(df["gene_id"].nunique()),
                    "families_conserved": int(df.loc[df["family_conserved"], "family"].nunique())},
            out_path=out_path, columns=EDGE_COLUMNS)
        return df

    def build(self) -> PriorArtifact:
        """The default-scope edge table as a bipartite artifact: miRNA family -> gene.

        Row 0 of ``edge_index`` indexes ``src_names`` (the families, sorted); row 1 the
        master gene space (Ensembl gene ids). ``edge_attr`` columns: conserved site count,
        cumulative weighted context++ (negative = stronger repression), aggregate PCT (0
        where NULL). One mask keeps an edge only when both endpoints resolve.
        """
        edges = self.edge_table(scope="default")
        families = sorted(edges["family"].unique())
        src_index = {f: i for i, f in enumerate(families)}
        attr = np.column_stack([
            edges["n_cons_sites"].to_numpy(dtype=np.float32),
            edges["context_score_weighted"].fillna(0.0).to_numpy(dtype=np.float32),
            edges["aggregate_pct"].fillna(0.0).to_numpy(dtype=np.float32),
        ])
        ei, attr = self.to_bipartite_edge_index(edges["family"], edges["gene_id"], src_index, attr)
        return PriorArtifact(
            kind="edge", relation=self.relation, layer=self.layer, edge_index=ei, edge_attr=attr,
            directed=True, src_names=families,
            meta={"edge_attr_names": ["n_cons_sites", "context_score_weighted", "aggregate_pct"],
                  "src_entity": self.spec.get("entity_src", "mirna_family"),
                  "candidate_edges": int(len(edges)), "kept_edges": int(ei.shape[1])},
        )

    # ---------------------------------------------------------------- sites --

    def _family_sites(self, name: str) -> tuple[pd.DataFrame, int]:
        """Human rows of a *_Family_Info / Predicted_Targets_Info file, typed."""
        need = {"mir family": "family", "gene id": "gene_id", "gene symbol": "symbol",
                "transcript id": "transcript_id", "utr start": "utr_start", "utr end": "utr_end",
                "msa start": "msa_start", "msa end": "msa_end", "seed match": "site_type", "pct": "pct"}
        z, member = self._zip_member(name)
        with z, z.open(member) as fh:
            header = [c.strip() for c in io.TextIOWrapper(fh, encoding="utf-8").readline().rstrip("\n").split("\t")]
        low = {c.lower(): c for c in header}
        missing = [k for k in need if k not in low]
        if missing:
            raise ValueError(f"{name}: missing columns {missing}; have {header}")
        raw, n_all = self._human_table(name, "Species ID", usecols=[low[k] for k in need])
        df = pd.DataFrame({v: raw[low[k]] for k, v in need.items()})
        df["family"] = df["family"].str.strip()
        df["gene_id"] = _strip_version(df["gene_id"])
        df["transcript_id"] = _strip_version(df["transcript_id"])
        df["symbol"] = df["symbol"].str.strip()
        for c in ("utr_start", "utr_end", "msa_start", "msa_end"):
            df[c] = _num(df[c]).astype("Int64")
        df["site_type"] = df["site_type"].str.strip()
        df["pct"] = _num(df["pct"])
        return df, n_all

    def site_table(self, *, rebuild: bool = False) -> pd.DataFrame:
        """One row per miRNA site of a conserved family, with its position on the 3'UTR.

        Sources: ``Conserved_Family_Info`` (conserved AND nonconserved sites of conserved
        families, with positions and PCT) flagged by membership in
        ``Predicted_Targets_Info.default_predictions`` (``conserved_site``). The weighted
        context++ of a site is attached from ``Conserved_Site_Context_Scores``, whose rows
        are per member miRNA (mapped to the family through miR_Family_Info) at the same UTR
        coordinates as the family files for human rows: the join key is (transcript,
        family, start, end) and the most negative member score is kept. The join rate is
        recorded in LINEAGE.json; an unjoined site keeps NaN. Cached at
        ``<derived>/sites.parquet``.
        """
        out_path = self.derived / "sites.parquet"
        if out_path.exists() and not rebuild:
            return pd.read_parquet(out_path)
        self.derived.mkdir(parents=True, exist_ok=True)
        rep = self._representative()
        allsites, n_fam_all = self._family_sites("Conserved_Family_Info.txt.zip")
        default, n_def_all = self._family_sites("Predicted_Targets_Info.default_predictions.txt.zip")
        key = ["transcript_id", "family", "utr_start", "utr_end"]
        n_before = len(allsites)
        allsites = allsites.drop_duplicates(key).reset_index(drop=True)
        dkeys = set(map(tuple, default[key].astype(str).to_numpy()))
        allsites["conserved_site"] = [tuple(r) in dkeys for r in allsites[key].astype(str).to_numpy()]
        # a default prediction absent from the family file (should not happen) is appended
        # rather than lost, and counted
        akeys = set(map(tuple, allsites[key].astype(str).to_numpy()))
        extra = default[[tuple(r) not in akeys for r in default[key].astype(str).to_numpy()]].copy()
        if len(extra):
            extra["conserved_site"] = True
            allsites = pd.concat([allsites, extra], ignore_index=True)
        allsites["representative"] = allsites["transcript_id"].isin(set(rep["transcript_id"]))
        # context++ per site, from the member-miRNA table via the family map
        ctx, n_ctx_all, join_rate = self._site_context_scores(allsites)
        allsites["context_pp"] = ctx
        allsites = allsites[list(SITE_COLUMNS)].sort_values(
            ["transcript_id", "utr_start", "family"]).reset_index(drop=True)
        allsites.to_parquet(out_path, index=False)
        self._record_lineage(
            "targetscan-vert_80/sites",
            what="one row per miRNA site of a conserved family with its 3'UTR position, seed "
                 "type, PCT, default-prediction flag and weighted context++ (T102)",
            inputs=["Conserved_Family_Info.txt.zip", "Predicted_Targets_Info.default_predictions.txt.zip",
                    "Conserved_Site_Context_Scores.txt.zip", "miR_Family_Info.txt.zip", "Gene_info.txt.zip"],
            code="src/sidechain/priors/posttx_mirna.py::MiRNATargetSource.site_table",
            counts={"family_info_rows_all_species": n_fam_all, "family_info_rows_human": n_before,
                    "duplicates_dropped": int(n_before - len(allsites) + len(extra)),
                    "default_rows_all_species": n_def_all, "default_rows_human": int(len(default)),
                    "default_sites_missing_from_family_file": int(len(extra)),
                    "sites_out": int(len(allsites)),
                    "sites_conserved": int(allsites["conserved_site"].sum()),
                    "sites_on_representative": int(allsites["representative"].sum()),
                    "context_score_rows_all_species": n_ctx_all,
                    "context_join_rate_on_conserved_sites": join_rate},
            out_path=out_path, columns=SITE_COLUMNS)
        return allsites

    def _site_context_scores(self, sites: pd.DataFrame) -> tuple[np.ndarray, int, float]:
        """Weighted context++ per site from Conserved_Site_Context_Scores, joined per family.

        Returns (per-site values aligned to `sites`, rows read, join rate over conserved sites).
        """
        name = "Conserved_Site_Context_Scores.txt.zip"
        need = {"transcript id": "transcript_id", "mirna": "mirna", "utr_start": "cs_start",
                "utr end": "cs_end", "weighted context++ score": "wctx"}
        z, member = self._zip_member(name)
        with z, z.open(member) as fh:
            header = [c.strip() for c in io.TextIOWrapper(fh, encoding="utf-8").readline().rstrip("\n").split("\t")]
        low = {c.lower(): c for c in header}
        missing = [k for k in need if k not in low]
        if missing:
            raise ValueError(f"{name}: missing columns {missing}; have {header}")
        raw, n_all = self._human_table(name, "Gene Tax ID", usecols=[low[k] for k in need])
        cs = pd.DataFrame({v: raw[low[k]] for k, v in need.items()})
        cs["transcript_id"] = _strip_version(cs["transcript_id"])
        cs["cs_start"] = _num(cs["cs_start"]).astype("Int64")
        cs["cs_end"] = _num(cs["cs_end"]).astype("Int64")
        cs["wctx"] = _num(cs["wctx"])
        fam = self.family_table()
        mir2fam = dict(zip(fam["mirbase_id"], fam["family"]))
        cs["family"] = cs["mirna"].str.strip().map(mir2fam)
        cs = cs.dropna(subset=["family", "wctx"])
        # The HUMAN rows of the context-score file carry the same UTR coordinates as the
        # family files (A1BG miR-23-3p: 143-150 in both). Other species' rows sit at that
        # species' own position (the macaque row of the same site reads 142-149), which is
        # what a first read of the file's head mistook for a global off-by-one; the join
        # rate recorded in LINEAGE.json is the check (0.0 under the shifted join, 2026-09-28).
        cs["utr_start"] = cs["cs_start"]
        cs["utr_end"] = cs["cs_end"]
        best = (cs.groupby(["transcript_id", "family", "utr_start", "utr_end"])["wctx"].min()
                  .rename("context_pp").reset_index())
        merged = sites[["transcript_id", "family", "utr_start", "utr_end"]].merge(
            best, on=["transcript_id", "family", "utr_start", "utr_end"], how="left")
        vals = merged["context_pp"].to_numpy(dtype=np.float64)
        cons = sites["conserved_site"].to_numpy() if "conserved_site" in sites else np.ones(len(sites), bool)
        rate = float(np.isfinite(vals[cons]).mean()) if cons.any() else float("nan")
        return vals, n_all, rate

    def check_site_coordinates(self, sites: pd.DataFrame, seqs: dict[str, str],
                               *, n: int = 5000, seed: int = 0) -> dict:
        """Which coordinate convention puts a site on its seed match: 1-based inclusive or
        0-based. Samples `n` conserved sites on transcripts we hold a sequence for and
        compares the sequence at the site with the reverse complement of the family's
        seed (8mer: seed+m8 complement followed by A; 7mer-m8: the complement alone;
        7mer-a1: the complement of nt 2-7 followed by A). Returns the match rate under
        each convention and the one that wins; a site_table consumer reads positions in
        the winning convention.
        """
        fam = self.family_table().drop_duplicates("family").set_index("family")["seed_m8"]
        cand = sites[sites["conserved_site"] & sites["transcript_id"].isin(seqs) & sites["family"].isin(fam.index)]
        if len(cand) == 0:
            return {"n": 0, "one_based": float("nan"), "zero_based": float("nan"), "convention": "unknown"}
        rng = np.random.default_rng(seed)
        take = cand.iloc[rng.choice(len(cand), size=min(n, len(cand)), replace=False)]
        hits = {"one_based": 0, "zero_based": 0}
        checked = 0
        for r in take.itertuples(index=False):
            seed_m8 = fam[r.family]
            if len(seed_m8) != 7:
                continue
            comp7 = revcomp_rna(seed_m8)          # pairs nt 2-8
            if r.site_type == "8mer":
                expect = comp7 + "A"
            elif r.site_type == "7mer-m8":
                expect = comp7
            elif r.site_type == "7mer-a1":
                expect = revcomp_rna(seed_m8[:6]) + "A"   # nt 2-7 plus the A opposite nt 1
            else:
                continue
            seq = seqs[r.transcript_id]
            s, e = int(r.utr_start), int(r.utr_end)
            checked += 1
            if seq[s - 1:e] == expect:
                hits["one_based"] += 1
            if seq[s:e + 1] == expect:
                hits["zero_based"] += 1
        if checked == 0:
            return {"n": 0, "one_based": float("nan"), "zero_based": float("nan"), "convention": "unknown"}
        one, zero = hits["one_based"] / checked, hits["zero_based"] / checked
        conv = "one_based" if one >= zero else "zero_based"
        if max(one, zero) < 0.9:
            conv = "unknown"
        return {"n": checked, "one_based": one, "zero_based": zero, "convention": conv}

    # ------------------------------------------------------ sequence features --

    @staticmethod
    def sequence_features(seq: str) -> dict:
        """Composition and consensus-motif counts of one ungapped RNA sequence."""
        n = len(seq)
        out = {"seq_len": n}
        if n == 0:
            out.update({f"frac_{b}": np.nan for b in NUCLEOTIDES})
            out["au_content"] = np.nan
            out.update({f"dn_{d}": np.nan for d in DINUCLEOTIDES})
            out.update({f"m_{m}": 0 for m in MOTIFS})
            return out
        for b in NUCLEOTIDES:
            out[f"frac_{b}"] = seq.count(b) / n
        out["au_content"] = out["frac_A"] + out["frac_U"]
        if n >= 2:
            counts = {d: 0 for d in DINUCLEOTIDES}
            for i in range(n - 1):
                d = seq[i:i + 2]
                if d in counts:
                    counts[d] += 1
            tot = max(sum(counts.values()), 1)
            for d in DINUCLEOTIDES:
                out[f"dn_{d}"] = counts[d] / tot
        else:
            for d in DINUCLEOTIDES:
                out[f"dn_{d}"] = np.nan
        for m, (pat, _, _) in MOTIFS.items():
            out[f"m_{m}"] = len(re.findall(pat, seq))
        return out

    @staticmethod
    def motif_intervals(seq: str) -> dict[str, list[tuple[int, int]]]:
        """1-based inclusive [start, end] of every match, per motif; overlapping copies of a
        fixed-length motif each count (lookahead), a tract is its maximal run."""
        out: dict[str, list[tuple[int, int]]] = {}
        for m, (pat, _, _) in MOTIFS.items():
            ivs = []
            for mt in re.finditer(pat, seq):
                if mt.end() > mt.start():                       # a tract: the run itself
                    ivs.append((mt.start() + 1, mt.end()))
                else:                                           # a lookahead: re-match the body
                    body = re.match(pat[3:-1], seq[mt.start():])
                    ivs.append((mt.start() + 1, mt.start() + body.end()))
            out[m] = ivs
        return out

    @staticmethod
    def cooperation_counts(site_iv: list[tuple[int, int]], motifs: dict[str, list[tuple[int, int]]],
                           utr_len: int, *, window: int = COOP_WINDOW_NT) -> dict:
        """The NAR 2016 co-occurrence and competition counts for one transcript.

        `site_iv` are the conserved miRNA sites as 1-based inclusive intervals. Per site:
        `within` = an RBP motif whose interval lies within `window` nt of the site (gap
        between the intervals at most `window`; overlap counts as distance 0); `overlap` =
        an RBP motif sharing at least one nucleotide with the site (the paper's competition
        rule: every site of the factor overlapped, here per site). Position features:
        sites in the first and last 15 % of the UTR (Grimson 2007: sites near the UTR ends
        are more effective) and in the last 500 nt.
        """
        out = {"n_cons_sites_seq": len(site_iv)}
        any_iv = [iv for ivs in motifs.values() for iv in ivs]

        def gap(a, b):
            return max(0, max(a[0] - b[1], b[0] - a[1]))

        def n_near(ivs):
            return sum(1 for s in site_iv if any(gap(s, m) <= window for m in ivs))

        def n_over(ivs):
            return sum(1 for s in site_iv if any(gap(s, m) == 0 for m in ivs))

        out["n_sites_rbp_within200"] = n_near(any_iv)
        out["n_sites_rbp_overlap"] = n_over(any_iv)
        out["frac_sites_rbp_overlap"] = (out["n_sites_rbp_overlap"] / len(site_iv)) if site_iv else np.nan
        for m in ("pum", "are_auuua", "polyu", "msi"):
            out[f"n_sites_{m}_within200"] = n_near(motifs.get(m, []))
        if utr_len > 0 and site_iv:
            starts = np.array([s for s, _ in site_iv], dtype=float)
            out["n_sites_first15pct"] = int((starts <= 0.15 * utr_len).sum())
            out["n_sites_last15pct"] = int((starts >= 0.85 * utr_len).sum())
            out["n_sites_last500nt"] = int((starts >= utr_len - 500).sum())
        else:
            out["n_sites_first15pct"] = out["n_sites_last15pct"] = out["n_sites_last500nt"] = 0
        return out

    def utr_sequence_table(self, *, rebuild: bool = False, sites: pd.DataFrame | None = None
                           ) -> pd.DataFrame:
        """One row per representative transcript: composition, motif counts, cooperation counts.

        Streams the human rows of ``UTR_Sequences.txt`` (over 4 GB of gapped text, read line
        by line), ungaps each sequence, computes `sequence_features`, and -- with the
        miRNA sites of `site_table` on the same transcript -- `cooperation_counts`. The
        coordinate convention of the sites is verified against the sequences first
        (`check_site_coordinates`) and recorded; positions are read in the convention
        that matched. Cached at ``<derived>/utr_seq_features.parquet``.
        """
        out_path = self.derived / "utr_seq_features.parquet"
        if out_path.exists() and not rebuild:
            return pd.read_parquet(out_path)
        self.derived.mkdir(parents=True, exist_ok=True)
        rep = self._representative()
        rep_ids = set(rep["transcript_id"])
        if sites is None:
            sites = self.site_table()
        seqs: dict[str, str] = {}
        meta: dict[str, tuple[str, str]] = {}
        n_rows = n_rep = 0
        for tid, gid, sym, seq in self._iter_human_utr_sequences():
            n_rows += 1
            if tid not in rep_ids or tid in seqs:
                continue
            n_rep += 1
            seqs[tid] = seq
            meta[tid] = (gid, sym)
        if not seqs:
            raise ValueError("UTR_Sequences.txt: no human representative-transcript rows")
        conv = self.check_site_coordinates(sites, seqs)
        offset = 0 if conv["convention"] in ("one_based", "unknown") else 1
        cons = sites[sites["conserved_site"] & sites["transcript_id"].isin(seqs)]
        by_tid: dict[str, list[tuple[int, int]]] = {}
        for tid, s, e in zip(cons["transcript_id"], cons["utr_start"], cons["utr_end"]):
            by_tid.setdefault(tid, []).append((int(s) + offset, int(e) + offset))
        rows = []
        for tid, seq in seqs.items():
            gid, sym = meta[tid]
            row = {"transcript_id": tid, "gene_id": gid, "symbol": sym}
            row.update(self.sequence_features(seq))
            row.update(self.cooperation_counts(by_tid.get(tid, []), self.motif_intervals(seq), len(seq)))
            rows.append(row)
        table = pd.DataFrame(rows).sort_values("transcript_id").reset_index(drop=True)
        table.to_parquet(out_path, index=False)
        self._record_lineage(
            "targetscan-vert_80/utr_seq_features",
            what="one row per representative transcript: 3'UTR length, base and dinucleotide "
                 "composition, consensus RBP motif counts, and the NAR 2016 co-occurrence / "
                 "competition counts against the conserved miRNA sites (T102)",
            inputs=["UTR_Sequences.txt.zip", "Gene_info.txt.zip"],
            code="src/sidechain/priors/posttx_mirna.py::MiRNATargetSource.utr_sequence_table",
            counts={"utr_rows_human": n_rows, "representative_transcripts_with_sequence": n_rep,
                    "rows_out": int(len(table)), "transcripts_with_conserved_sites": int(len(by_tid))},
            out_path=out_path,
            extra={"site_coordinate_check": conv, "coordinate_offset_applied": offset,
                   "motifs": {m: {"regex": p, "stands_in_for": w, "source": s}
                              for m, (p, w, s) in MOTIFS.items()},
                   "cooperation_window_nt": COOP_WINDOW_NT})
        return table

    def utr_features_table(self, *, rebuild: bool = False) -> pd.DataFrame:
        """The widened per-gene table: `utr_load_table` joined with `utr_sequence_table` on the
        representative transcript, plus `n_noncons_sites_all`, the full count of nonconserved
        sites of conserved families on that transcript from `site_table` (the load table's
        `n_noncons_sites` only counts families that also have a conserved site). Cached at
        ``<derived>/utr_features.parquet``."""
        out_path = self.derived / "utr_features.parquet"
        if out_path.exists() and not rebuild:
            return pd.read_parquet(out_path)
        load = self.utr_load_table()
        seq = self.utr_sequence_table().drop(columns=["gene_id", "symbol"])
        sites = self.site_table()
        noncons = (sites[~sites["conserved_site"]].groupby("transcript_id").size()
                   .rename("n_noncons_sites_all").reset_index())
        table = (load.merge(seq, on="transcript_id", how="left")
                     .merge(noncons, on="transcript_id", how="left"))
        table["n_noncons_sites_all"] = table["n_noncons_sites_all"].fillna(0).astype(int)
        table.to_parquet(out_path, index=False)
        self._record_lineage(
            "targetscan-vert_80/utr_features",
            what="utr_load joined with utr_seq_features on the representative transcript, plus "
                 "the full nonconserved-site count from sites.parquet (T102)",
            inputs=["Gene_info.txt.zip", "UTR_Sequences.txt.zip", "Conserved_Family_Info.txt.zip"],
            code="src/sidechain/priors/posttx_mirna.py::MiRNATargetSource.utr_features_table",
            counts={"genes": int(len(table)), "genes_with_sequence": int(table["seq_len"].notna().sum()),
                    "genes_with_nonconserved_sites": int((table["n_noncons_sites_all"] > 0).sum())},
            out_path=out_path)
        return table
