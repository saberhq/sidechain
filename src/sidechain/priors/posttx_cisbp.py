"""CisBP-RNA 2.0 RBP motifs scanned over TargetScan's representative 3'UTRs, as per-gene site
counts per RNA-binding protein (T120, phase 2).

    .venv/bin/python scripts/t120_build_cisbp.py            # fetch through the gate, build, print the counts

Reads the human CisBP-RNA archive the gate lands under ``external/cisbp-rna-2.0/`` (the block
``cisbp_rna_2`` in ``configs/data_sources.yaml``: ``RBP_Information_all_motifs.txt`` and the
``pwms_all_motifs/*.txt`` position frequency matrices) and TargetScan's human representative
3'UTR sequences (``UTR_Sequences.txt.zip`` and ``Gene_info.txt.zip`` of the ``targetscan``
block, the same transcript per gene as every ``ts_*`` column). Writes, under
``derived/cisbp-rna-2.0/``: ``rbp_motifs.parquet`` (one row per RBP x motif with a usable
matrix, its status, source, length, information content and hit rates), ``pfms.parquet``
(the matrices, long), ``rbp_site_counts.parquet`` (one row per representative transcript,
one count column per RBP) and a LINEAGE.json entry per table.

**The scan.** Each matrix becomes a log-odds matrix against the 0-order base composition of all
scanned 3'UTRs (a pseudocount of ``PSEUDOCOUNT`` on each probability); a window is a hit when its
score is at least ``REL_SCORE_MIN`` of the way from the matrix's worst to its best score (JASPAR's
relative score); a window never crosses a sequence boundary and any base outside ACGU fails it.
An RBP's count is the number of window starts where ANY of its matrices hits, so overlapping
copies count once per position, as the ``ts_m_*`` regexes do. Each matrix is also scanned over a
per-sequence base shuffle of the same 3'UTRs and its hits per kb there are written beside the real
rate, which is how permissive its threshold is, and each gene's direct total is counted once more on
its own shuffled 3'UTR (``rb_sites_direct_null``: what length and composition alone give). A scan at
another threshold writes ``rbp_site_counts_r<NNN>.parquet`` and ``rbp_motifs_r<NNN>.parquet`` beside
the default's and leaves them untouched. Direct (D) and inferred (I) motifs are scanned
alike and the status is in the dictionary; a zero-byte matrix file (42 of 586 in build 2.00, mostly
RBPDB's) is skipped and counted.

The idea and the results: private ``research/ideas/ptr-feature-table.md`` § Outcome; the task:
private ``tasks/T120.md``.
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sidechain.priors.base import PriorArtifact, PriorSource
from sidechain.priors.posttx_mirna import (
    DATA_ROOT,
    REGISTRY,
    MiRNATargetSource,
    fetch_gated_files,
    sha256_of,
    spec_from_registry,
)

CISBP_BLOCK = "cisbp_rna_2"
TARGETSCAN_BLOCK = "targetscan"
#: The name the gate gives the host-built human archive (the host names it by its timestamp).
ARCHIVE_FILE = "cisbp-rna_2.00_Homo_sapiens_RBP_Information_PFMs.zip"
RBP_TABLE = "RBP_Information_all_motifs.txt"
PFM_DIR = "pwms_all_motifs"

MOTIF_FILE = "rbp_motifs.parquet"
PFM_FILE = "pfms.parquet"
GENE_FILE = "rbp_site_counts.parquet"

ALPHABET = "ACGU"
PSEUDOCOUNT = 0.01            # added to each PFM probability before the log-odds (then renormalised)
REL_SCORE_MIN = 0.85          # a window is a hit at this fraction of the matrix's worst-to-best range
NULL_SEED = 0                 # the per-sequence base shuffle behind the null hit rate
_NO_BASE = np.float32(-1e6)   # the score of a window holding a base outside ACGU

#: The per-gene columns beside the per-RBP counts, and what each is.
SUMMARY_COLUMNS = {
    "symbol": "TargetScan's gene symbol (GENCODE-era, ~2018)",
    "gene_id": "Ensembl gene id, version stripped",
    "transcript_id": "the representative transcript the 3'UTR was scanned on",
    "seq_len": "ungapped 3'UTR sequence length, nt",
    "rb_sites_direct": "window starts hit by any directly measured (D) RBP motif, summed over RBPs",
    "rb_n_rbps_direct": "RBPs with a direct motif that hit this 3'UTR at least once",
    "rb_sites_inferred": "the same sum over RBPs whose only motifs are inferred (I)",
    "rb_sites_direct_null": "rb_sites_direct counted on the same 3'UTR with its bases shuffled (seed NULL_SEED): "
                            "what length and composition alone give this gene",
}


def _code_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:                                          # noqa: BLE001
        return "unknown"


# ------------------------------------------------------------------- matrices --


def parse_pfm(text: str) -> np.ndarray:
    """A CisBP-RNA matrix file (``Pos A C G U``, one row per position) -> (L, 4) probabilities.

    Rows are renormalised to sum to one; an empty file, a bad header or a row that is not four
    numbers is a ValueError (the caller decides whether that skips the motif).
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty matrix file")
    header = lines[0].split("\t")
    if [h.strip().upper() for h in header[:5]] != ["POS", "A", "C", "G", "U"]:
        raise ValueError(f"unexpected matrix header {header!r}")
    rows = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) < 5:
            raise ValueError(f"short matrix row {ln!r}")
        rows.append([float(x) for x in parts[1:5]])
    pfm = np.asarray(rows, dtype=np.float64)
    if pfm.ndim != 2 or pfm.shape[0] == 0 or (pfm < 0).any():
        raise ValueError("matrix has no rows or a negative entry")
    sums = pfm.sum(axis=1, keepdims=True)
    if (sums <= 0).any():
        raise ValueError("a matrix row sums to zero")
    return pfm / sums


def information_content(pfm: np.ndarray) -> float:
    """Bits over the matrix, 2 bits per fully determined position (uniform background)."""
    p = np.clip(pfm, 1e-12, None)
    return float(np.sum(2.0 + (p * np.log2(p)).sum(axis=1)))


def log_odds(pfm: np.ndarray, background: np.ndarray, pseudocount: float = PSEUDOCOUNT) -> np.ndarray:
    """(L, 5) log2 odds against the background; column 4 (a base outside ACGU) fails the window."""
    p = (pfm + pseudocount) / (1.0 + 4.0 * pseudocount)
    bg = np.asarray(background, dtype=np.float64)
    w = np.log2(p) - np.log2(bg)[None, :]
    out = np.full((pfm.shape[0], 5), _NO_BASE, dtype=np.float32)
    out[:, :4] = w.astype(np.float32)
    return out


def relative_threshold(w: np.ndarray, rel: float = REL_SCORE_MIN) -> float:
    """The absolute score at ``rel`` of the way from the matrix's worst to its best window score."""
    lo = float(w[:, :4].min(axis=1).sum())
    hi = float(w[:, :4].max(axis=1).sum())
    return lo + rel * (hi - lo)


# ------------------------------------------------------------------ sequences --


def encode(seqs: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All sequences end to end: base codes (A0 C1 G2 U3, 4 otherwise), the sequence index of
    every position, and how many bases of its sequence remain from each position on."""
    table = np.full(256, 4, dtype=np.int8)
    for i, b in enumerate(ALPHABET):
        table[ord(b)] = i
    table[ord("T")] = 3
    lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
    joined = "".join(seqs).upper().encode("ascii", errors="replace")
    code = table[np.frombuffer(joined, dtype=np.uint8)]
    seq_index = np.repeat(np.arange(len(seqs), dtype=np.int32), lens)
    starts = np.repeat(np.cumsum(lens) - lens, lens)
    remaining = (np.repeat(np.cumsum(lens), lens) - np.arange(code.size)).astype(np.int32)
    del starts
    return code, seq_index, remaining


def shuffle_within(seqs: list[str], seed: int = NULL_SEED) -> list[str]:
    """Each sequence's bases permuted among themselves (the 0-order null of the same composition)."""
    rng = np.random.default_rng(seed)
    out = []
    for s in seqs:
        a = np.frombuffer(s.encode("ascii", errors="replace"), dtype=np.uint8)
        out.append(rng.permutation(a).tobytes().decode("ascii"))
    return out


def background_of(code: np.ndarray) -> np.ndarray:
    """0-order ACGU frequencies of the scanned bases (bases outside ACGU ignored)."""
    counts = np.bincount(code[code < 4].astype(np.int64), minlength=4)[:4].astype(np.float64)
    if counts.sum() == 0:
        raise ValueError("no ACGU bases to take a background from")
    return counts / counts.sum()


def scan_hits(code: np.ndarray, remaining: np.ndarray, w: np.ndarray, threshold: float) -> np.ndarray:
    """Window start positions (into ``code``) whose score reaches ``threshold`` and that lie
    inside one sequence."""
    length = w.shape[0]
    m = code.size - length + 1
    if m <= 0:
        return np.zeros(0, dtype=np.int64)
    score = w[0][code[:m]].astype(np.float32, copy=True)
    for k in range(1, length):
        score += w[k][code[k:k + m]]
    hit = (score >= np.float32(threshold)) & (remaining[:m] >= length)
    return np.flatnonzero(hit).astype(np.int64)


# ------------------------------------------------------------------- the source --


class CisBPRNASource(PriorSource):
    """CisBP-RNA 2.0 motif sites per RBP, on TargetScan's representative 3'UTRs.

    Constructed like every PriorSource -- ``(spec, gene_index)`` -- but ``fetch()``,
    ``motif_table()`` and ``gene_table()`` need no gene index (an empty dict will do);
    ``build()`` needs it. ``registry`` names the prior registry the TargetScan block is
    looked up in.
    """

    def __init__(self, spec: dict, gene_index: dict[str, int], root: Path = DATA_ROOT,
                 registry: str | Path = REGISTRY):
        super().__init__(spec, gene_index)
        self.root = Path(root).expanduser()
        self.registry = registry

    @property
    def dest(self) -> Path:
        return self.root / self.spec["dest"]

    @property
    def derived(self) -> Path:
        return self.root / self.spec["derived"]

    def _targetscan(self) -> MiRNATargetSource:
        return MiRNATargetSource(spec_from_registry(TARGETSCAN_BLOCK, self.registry),
                                 self.gene_index, root=self.root)

    def fetch(self, *, refresh: bool = False, progress: bool = False) -> Path:
        """The human archive through the gate: a form POST names it, the gate lands it."""
        return fetch_gated_files(self.spec, self.root, refresh=refresh, progress=progress,
                                 registry=self.registry)

    # ----------------------------------------------------------------- lineage --

    def _archive_inputs(self) -> dict:
        path = self.dest / ARCHIVE_FILE
        prov = self.dest / "PROVENANCE.json"
        selected = json.loads(prov.read_text()).get("selected", []) if prov.exists() else []
        f = next((x for x in selected if x.get("name") == ARCHIVE_FILE), {})
        # the zip's own sha256 changes with every generation (a directory entry's time stamp);
        # the content digest below does not, so both are recorded
        h = __import__("hashlib").sha256()
        with zipfile.ZipFile(path) as z:
            for info in sorted(z.infolist(), key=lambda i: i.filename):
                if info.is_dir():
                    continue
                h.update(info.filename.encode())
                h.update(z.read(info))
        return {ARCHIVE_FILE: {"dest": str(self.dest), "bytes": f.get("size_bytes"),
                               "host_evidence": f.get("checksum"), "sha256": sha256_of(path),
                               "content_sha256": h.hexdigest()}}

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

    # ------------------------------------------------------------------ motifs --

    def _read_archive(self) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, str]]:
        """The RBP x motif table, the usable matrices by motif id, and why the rest were skipped."""
        with zipfile.ZipFile(self.dest / ARCHIVE_FILE) as z:
            names = z.namelist()
            table_member = next((n for n in names if n.endswith("/" + RBP_TABLE) or n == RBP_TABLE), None)
            if table_member is None:
                raise ValueError(f"{ARCHIVE_FILE}: no {RBP_TABLE} inside")
            text = z.read(table_member).decode("utf-8")
            table = pd.read_csv(io.StringIO(text), sep="\t", dtype=str, keep_default_na=False)
            pfms: dict[str, np.ndarray] = {}
            skipped: dict[str, str] = {}
            pat = re.compile(rf"(?:^|/){PFM_DIR}/([^/]+)\.txt$")
            for n in names:
                m = pat.search(n)
                if not m:
                    continue
                motif_id = m.group(1)
                raw = z.read(n)
                if not raw.strip():
                    skipped[motif_id] = "empty file"
                    continue
                try:
                    pfms[motif_id] = parse_pfm(raw.decode("utf-8"))
                except ValueError as exc:
                    skipped[motif_id] = str(exc)
        # the archive's table repeats the DBID header: the first is the RBP's Ensembl id, the
        # second the motif's own identifier at its source
        cols = list(table.columns)
        seen = 0
        for i, c in enumerate(cols):
            if c == "DBID" or c.startswith("DBID."):
                cols[i] = "rbp_ensembl_id" if seen == 0 else "motif_source_id"
                seen += 1
        table.columns = cols
        return table, pfms, skipped

    def motif_table(self, *, rebuild: bool = False) -> pd.DataFrame:
        """One row per (RBP, motif) with a usable matrix. Cached at ``<derived>/rbp_motifs.parquet``;
        the matrices beside it in ``pfms.parquet`` (motif_id, pos, A, C, G, U)."""
        out_path = self.derived / MOTIF_FILE
        if out_path.exists() and not rebuild:
            return pd.read_parquet(out_path)
        self.derived.mkdir(parents=True, exist_ok=True)
        table, pfms, skipped = self._read_archive()
        has = table["Motif_ID"].isin(pfms)
        rows = table[has].copy()
        rows["pfm_len"] = rows["Motif_ID"].map(lambda m: int(pfms[m].shape[0]))
        rows["ic_bits"] = rows["Motif_ID"].map(lambda m: information_content(pfms[m]))
        keep = ["RBP_ID", "RBP_Name", "rbp_ensembl_id", "RBP_Status", "Family_Name", "RBDs", "Cutoff",
                "Motif_ID", "Motif_Type", "MSource_Identifier", "MSource_Type", "MSource_Year", "PMID",
                "motif_source_id", "pfm_len", "ic_bits"]
        keep = [c for c in keep if c in rows.columns]
        motifs = rows[keep].rename(columns={
            "RBP_ID": "rbp_id", "RBP_Name": "rbp", "RBP_Status": "status", "Family_Name": "family",
            "RBDs": "rbds", "Cutoff": "cutoff", "Motif_ID": "motif_id", "Motif_Type": "motif_type",
            "MSource_Identifier": "source", "MSource_Type": "source_type", "MSource_Year": "source_year",
            "PMID": "pmid"}).drop_duplicates(["rbp_id", "motif_id"]).reset_index(drop=True)
        long = pd.concat([pd.DataFrame({"motif_id": m, "pos": np.arange(1, p.shape[0] + 1),
                                        "A": p[:, 0], "C": p[:, 1], "G": p[:, 2], "U": p[:, 3]})
                          for m, p in sorted(pfms.items())], ignore_index=True)
        long.to_parquet(self.derived / PFM_FILE, index=False)
        motifs.to_parquet(out_path, index=False)
        status_counts = table.drop_duplicates("RBP_ID")["RBP_Status"].value_counts().to_dict()
        listed_no_file = sorted(set(table.loc[table["Motif_ID"] != ".", "Motif_ID"]) - set(pfms) - set(skipped))
        self._record_lineage(
            "cisbp-rna-2.0/rbp_motifs",
            what="CisBP-RNA 2.00 human RBP x motif rows with a usable position frequency matrix, "
                 "the matrices beside it (T120 phase 2)",
            inputs=self._archive_inputs(),
            code="src/sidechain/priors/posttx_cisbp.py::CisBPRNASource.motif_table",
            counts={"archive_rows": int(len(table)), "rbps_in_archive": int(table["RBP_ID"].nunique()),
                    "rbp_status_in_archive": {str(k): int(v) for k, v in status_counts.items()},
                    "matrix_files": int(len(pfms) + len(skipped)), "usable_matrices": int(len(pfms)),
                    "skipped_matrices": int(len(skipped)),
                    "motifs_listed_without_a_file": int(len(listed_no_file)),
                    "rows_out": int(len(motifs)), "rbps_with_a_usable_motif": int(motifs["rbp_id"].nunique()),
                    "rbps_direct": int(motifs.loc[motifs["status"] == "D", "rbp_id"].nunique()),
                    "rbps_inferred_only": int(motifs.loc[motifs["status"] == "I", "rbp_id"].nunique())},
            out_path=out_path,
            extra={"pfm_file": str(self.derived / PFM_FILE), "pfm_sha256": sha256_of(self.derived / PFM_FILE),
                   "skipped": skipped, "motifs_listed_without_a_file": listed_no_file},
        )
        return motifs

    def pfms(self) -> dict[str, np.ndarray]:
        long = pd.read_parquet(self.derived / PFM_FILE)
        return {m: g.sort_values("pos")[["A", "C", "G", "U"]].to_numpy(dtype=np.float64)
                for m, g in long.groupby("motif_id")}

    # ------------------------------------------------------------------- genes --

    def _sequences(self) -> pd.DataFrame:
        """transcript_id, gene_id, symbol, seq for every human representative transcript."""
        ts = self._targetscan()
        rep_ids = set(ts._representative()["transcript_id"])
        rows, seen = [], set()
        for tid, gid, sym, seq in ts._iter_human_utr_sequences():
            if tid in rep_ids and tid not in seen:
                seen.add(tid)
                rows.append((tid, gid, sym, seq))
        if not rows:
            raise ValueError("UTR_Sequences.txt: no human representative-transcript rows")
        return pd.DataFrame(rows, columns=["transcript_id", "gene_id", "symbol", "seq"])

    def gene_table(self, *, rebuild: bool = False, rel_score_min: float = REL_SCORE_MIN,
                   progress: bool = False) -> pd.DataFrame:
        """One row per representative transcript, one site count per RBP. Cached at
        ``<derived>/rbp_site_counts.parquet``; the per-motif hit rates go into the motif table."""
        default = abs(rel_score_min - REL_SCORE_MIN) < 1e-9
        tag = "" if default else f"_r{int(round(rel_score_min * 100)):03d}"
        out_path = self.derived / (GENE_FILE if default else GENE_FILE.replace(".parquet", f"{tag}.parquet"))
        motif_out = self.derived / (MOTIF_FILE if default else MOTIF_FILE.replace(".parquet", f"{tag}.parquet"))
        if out_path.exists() and not rebuild:
            return pd.read_parquet(out_path)
        motifs = self.motif_table(rebuild=rebuild and default)
        pfms = self.pfms()
        seqs = self._sequences()
        sequences = seqs["seq"].tolist()
        code, seq_index, remaining = encode(sequences)
        bg = background_of(code)
        null_code, _, null_remaining = encode(shuffle_within(sequences))
        total_kb = float(code.size) / 1000.0
        weights = {m: log_odds(p, bg) for m, p in pfms.items()}
        thresholds = {m: relative_threshold(w, rel_score_min) for m, w in weights.items()}
        used = set(motifs["motif_id"])
        hits: dict[str, np.ndarray] = {}
        null_hits: dict[str, np.ndarray] = {}
        real_rate: dict[str, float] = {}
        null_rate: dict[str, float] = {}
        todo = sorted(m for m in weights if m in used)
        for i, m in enumerate(todo):
            w = weights[m]
            hits[m] = scan_hits(code, remaining, w, thresholds[m])
            null_hits[m] = scan_hits(null_code, null_remaining, w, thresholds[m])
            real_rate[m] = len(hits[m]) / total_kb
            null_rate[m] = len(null_hits[m]) / total_kb
            if progress and i % 50 == 0:
                print(f"  scanned {i + 1} / {len(todo)} matrices", flush=True)
        motifs = motifs.copy()
        motifs["threshold"] = motifs["motif_id"].map(thresholds)
        motifs["hits_per_kb"] = motifs["motif_id"].map(real_rate)
        motifs["null_hits_per_kb"] = motifs["motif_id"].map(null_rate)
        motifs.to_parquet(motif_out, index=False)

        n_seq = len(sequences)
        table = seqs[["symbol", "gene_id", "transcript_id"]].copy()
        table["seq_len"] = seqs["seq"].str.len().to_numpy()
        columns: dict[str, str] = dict(SUMMARY_COLUMNS)
        direct_total = np.zeros(n_seq, dtype=np.int64)
        direct_null = np.zeros(n_seq, dtype=np.int64)
        direct_n = np.zeros(n_seq, dtype=np.int64)
        inferred_total = np.zeros(n_seq, dtype=np.int64)
        counts: dict[str, np.ndarray] = {}
        for rbp_id, g in motifs.groupby("rbp_id", sort=True):
            name = str(g["rbp"].iloc[0])
            status = "D" if (g["status"] == "D").any() else str(g["status"].iloc[0])
            pos = np.unique(np.concatenate([hits[m] for m in g["motif_id"]]))
            c = np.bincount(seq_index[pos], minlength=n_seq).astype(np.int64)
            npos = np.unique(np.concatenate([null_hits[m] for m in g["motif_id"]]))
            cn = np.bincount(seq_index[npos], minlength=n_seq).astype(np.int64)
            col = f"rb_{name}"
            if col in counts:                       # two RBP ids under one name: keep the union
                counts[col] = counts[col] + c
            else:
                counts[col] = c
            kinds = ", ".join(sorted(set(g["motif_type"])))
            columns[col] = (f"{name} ({status}; Ensembl {g['rbp_ensembl_id'].iloc[0]}; {len(g)} matrices: {kinds}) "
                            f"window starts in the representative 3'UTR at relative score >= {rel_score_min}, "
                            f"hit by any of its matrices")
            if status == "D":
                direct_total += c
                direct_null += cn
                direct_n += (c > 0)
            else:
                inferred_total += c
        for col in sorted(counts):
            table[col] = counts[col]
        table["rb_sites_direct"] = direct_total
        table["rb_n_rbps_direct"] = direct_n
        table["rb_sites_inferred"] = inferred_total
        table["rb_sites_direct_null"] = direct_null
        table = table.sort_values("transcript_id").reset_index(drop=True)
        table.to_parquet(out_path, index=False)
        ts = self._targetscan()
        self._record_lineage(
            "cisbp-rna-2.0/rbp_site_counts" + tag,
            what="one row per TargetScan representative transcript: CisBP-RNA 2.00 motif sites per RBP "
                 "in its 3'UTR at a relative-score threshold, plus the direct and inferred totals (T120 phase 2)",
            inputs={**self._archive_inputs(),
                    "targetscan": {"dest": str(ts.dest),
                                   "files": {n: sha256_of(ts.dest / n) for n in ("UTR_Sequences.txt.zip", "Gene_info.txt.zip")
                                             if (ts.dest / n).exists()}},
                    "rbp_motifs": {"path": str(motif_out)}},
            code="src/sidechain/priors/posttx_cisbp.py::CisBPRNASource.gene_table",
            counts={"transcripts": int(n_seq), "bases_scanned": int(code.size),
                    "matrices_scanned": int(len(todo)), "rbp_columns": int(len(counts)),
                    "rbps_direct": int((motifs.groupby("rbp_id")["status"].apply(lambda s: (s == "D").any())).sum()),
                    "transcripts_with_a_direct_site": int((direct_total > 0).sum()),
                    "median_direct_sites": float(np.median(direct_total))},
            out_path=out_path,
            columns=columns,
            extra={"scan": {"rel_score_min": rel_score_min, "pseudocount": PSEUDOCOUNT, "gene_file": str(out_path),
                            "motif_file": str(motif_out),
                            "background_acgu": [float(x) for x in bg], "null_seed": NULL_SEED,
                            "null": "each 3'UTR's bases permuted among themselves; hits per kb per matrix "
                                    "in rbp_motifs.parquet",
                            "window_rule": "a window never crosses a sequence boundary; a base outside ACGU fails it; "
                                           "overlapping hits count once per start position"}},
        )
        return table

    def build(self) -> PriorArtifact:
        """The direct-RBP count columns aligned to the master gene space, with a `has_value` mask."""
        table = self.gene_table()
        motifs = self.motif_table()
        direct = sorted({f"rb_{r}" for r in motifs.loc[motifs["status"] == "D", "rbp"]} & set(table.columns))
        n_genes = len(self.gene_index)
        features = np.zeros((n_genes, len(direct)), dtype=np.float32)
        has_value = np.zeros((n_genes, len(direct)), dtype=bool)
        ids = table["gene_id"].astype(str).to_numpy()
        values = table[direct].to_numpy(dtype=np.float32)
        for gene, row in zip(ids, values):
            position = self.gene_index.get(gene)
            if position is None:
                continue
            features[position] = row
            has_value[position] = True
        return PriorArtifact(
            kind="node_feature", relation=self.relation, layer=self.layer, features=features,
            meta={"feature_names": direct, "has_value": has_value,
                  "sources": {n: CISBP_BLOCK for n in direct},
                  "candidate_genes": int(len(table)),
                  "genes_with_any_value": int(has_value.any(axis=1).sum()),
                  "zero_is_a_real_value": True,
                  "licence": self.spec.get("license")},
        )
