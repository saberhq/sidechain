"""RNAplfold accessibility of TargetScan's representative 3'UTRs: per gene, around its conserved
miRNA sites and over its consensus RBP motifs (T120, phase 2).

    .venv/bin/python scripts/t120_build_access.py --workers 8     # about ten minutes on the Mac

Reads TargetScan's human representative 3'UTR sequences (``UTR_Sequences.txt.zip`` and ``Gene_info.txt.zip``
of the ``targetscan`` block, through ``MiRNATargetSource``), the conserved miRNA sites of
``derived/targetscan-vert_80/sites.parquet`` (1-based inclusive on the ungapped UTR; the convention is
verified in that directory's LINEAGE), and the eight consensus RBP motifs of ``posttx_mirna.MOTIFS``.
Writes ``derived/rnaplfold-access/utr_accessibility.parquet`` and a LINEAGE.json entry.

**Accessibility** is ViennaRNA's RNAplfold (``RNA.pfl_fold_up``): for every position the probability,
under the thermodynamic ensemble of local structures, that the ``u`` nucleotides ending there are all
unpaired, with base pairs allowed within ``SPAN`` nt inside sliding windows of ``WINDOW`` nt. A
nucleotide in a stem reads near 0, one in a loop near 1. The per-gene columns are the mean per-nucleotide
accessibility of the 3'UTR, the share of open nucleotides, the mean 8-nt accessibility (the baseline a
site is read against), the accessibility of each conserved miRNA site's own span and of its flanks, and
the per-nucleotide accessibility over each consensus motif's matches. All universal: the sequence is the
same in every line.

The idea and the results: private ``research/ideas/ptr-feature-table.md`` § Outcome; the task: private
``tasks/T120.md``.
"""
from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from sidechain.priors.posttx_mirna import (
    DATA_ROOT,
    MOTIFS,
    REGISTRY,
    MiRNATargetSource,
    sha256_of,
    spec_from_registry,
)

TARGETSCAN_BLOCK = "targetscan"
TS_DERIVED = "derived/targetscan-vert_80"
SITES_FILE = "sites.parquet"
DERIVED = "derived/rnaplfold-access"
TABLE_FILE = "utr_accessibility.parquet"
LINEAGE_KEY = "rnaplfold-access/utr_accessibility"

WINDOW = 80          # RNAplfold -W: the sliding window a local structure is folded in
SPAN = 40            # RNAplfold -L: the longest base-pair span inside it
ULEN = 8             # the longest unpaired stretch asked for (a miRNA site is 7 or 8 nt)
FLANK = 30           # nt on each side of a site read as its context
OPEN = 0.5           # a nucleotide with P(unpaired) at or above this is "open"

#: The per-gene columns and what each is.
COLUMNS = {
    "symbol": "TargetScan's gene symbol (GENCODE-era, ~2018)",
    "gene_id": "Ensembl gene id, version stripped",
    "transcript_id": "the representative transcript the 3'UTR was folded on",
    "seq_len": "ungapped 3'UTR sequence length, nt",
    "acc_mean_u1": "mean per-nucleotide unpaired probability over the 3'UTR (RNAplfold W80 L40)",
    "acc_frac_open_u1": f"share of nucleotides with unpaired probability >= {OPEN}",
    "acc_mean_u8": "mean probability that an 8-nt window is entirely unpaired, over all windows: the baseline a site reads against",
    "acc_mirna_n_sites": "conserved miRNA sites on the representative 3'UTR (TargetScan default predictions, all types)",
    "acc_mirna_site_mean": "mean over those sites of the probability that the site's own span (7 or 8 nt) is unpaired",
    "acc_mirna_site_max": "the most accessible site's span probability",
    "acc_mirna_site_excess": "acc_mirna_site_mean minus acc_mean_u8: how much more open the sites are than the 3'UTR's typical 8-nt window",
    "acc_mirna_flank_mean": f"mean per-nucleotide unpaired probability over the {FLANK} nt on each side of a site (the site excluded), averaged over sites",
    "acc_motif_rbp_mean": "mean per-nucleotide unpaired probability over every match of the eight consensus RBP motifs",
}
for _m, (_pat, _what, _src) in MOTIFS.items():
    COLUMNS[f"acc_motif_{_m}_mean"] = f"mean per-nucleotide unpaired probability over matches of the {_what} motif; NaN with no match"


def _code_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:                                          # noqa: BLE001
        return "unknown"


def unpaired_probabilities(seq: str, *, ulen: int = ULEN, window: int = WINDOW, span: int = SPAN) -> np.ndarray:
    """(n + 1, ulen + 1) array: ``[i, u]`` is the probability that the ``u`` nucleotides ending at 1-based
    position ``i`` are all unpaired; row 0 and column 0 are unused (0)."""
    import RNA

    up = RNA.pfl_fold_up(seq, ulen, window, span)
    n = len(seq)
    out = np.zeros((n + 1, ulen + 1), dtype=np.float64)
    for i in range(1, n + 1):
        row = up[i]
        k = min(len(row), ulen + 1)
        out[i, :k] = row[:k]
    return out


def accessibility_row(seq: str, sites: list[tuple[int, int]], *, ulen: int = ULEN, window: int = WINDOW,
                      span: int = SPAN, flank: int = FLANK, open_at: float = OPEN) -> dict:
    """The per-gene columns for one 3'UTR; ``sites`` are 1-based inclusive (start, end) spans."""
    n = len(seq)
    up = unpaired_probabilities(seq, ulen=ulen, window=window, span=span)
    u1 = up[1:, 1]
    row = {"seq_len": n, "acc_mean_u1": float(u1.mean()) if n else np.nan,
           "acc_frac_open_u1": float((u1 >= open_at).mean()) if n else np.nan,
           "acc_mean_u8": float(up[ulen:, ulen].mean()) if n >= ulen else np.nan}
    site_p, flank_p = [], []
    for s, e in sites:
        if s < 1 or e > n or e < s:
            continue
        span_len = e - s + 1
        if span_len <= ulen:
            site_p.append(float(up[e, span_len]))
        lo, hi = max(1, s - flank), min(n, e + flank)
        ctx = np.concatenate([u1[lo - 1:s - 1], u1[e:hi]])
        if ctx.size:
            flank_p.append(float(ctx.mean()))
    row["acc_mirna_n_sites"] = len(site_p)
    row["acc_mirna_site_mean"] = float(np.mean(site_p)) if site_p else np.nan
    row["acc_mirna_site_max"] = float(np.max(site_p)) if site_p else np.nan
    row["acc_mirna_site_excess"] = (row["acc_mirna_site_mean"] - row["acc_mean_u8"]) if site_p else np.nan
    row["acc_mirna_flank_mean"] = float(np.mean(flank_p)) if flank_p else np.nan
    all_pos: list[np.ndarray] = []
    for m, ivs in MiRNATargetSource.motif_intervals(seq).items():
        pos = np.concatenate([np.arange(s - 1, e) for s, e in ivs]) if ivs else np.zeros(0, dtype=int)
        row[f"acc_motif_{m}_mean"] = float(u1[pos].mean()) if pos.size else np.nan
        if pos.size:
            all_pos.append(pos)
    pooled = np.unique(np.concatenate(all_pos)) if all_pos else np.zeros(0, dtype=int)
    row["acc_motif_rbp_mean"] = float(u1[pooled].mean()) if pooled.size else np.nan
    return row


def _work(args: tuple) -> dict:
    tid, gid, sym, seq, sites = args
    return {"transcript_id": tid, "gene_id": gid, "symbol": sym, **accessibility_row(seq, sites)}


def conserved_sites(root: Path) -> dict[str, list[tuple[int, int]]]:
    """Conserved miRNA sites on representative transcripts, by transcript id, 1-based inclusive."""
    s = pd.read_parquet(root / TS_DERIVED / SITES_FILE,
                        columns=["transcript_id", "utr_start", "utr_end", "conserved_site", "representative"])
    s = s[s["conserved_site"] & s["representative"]].dropna(subset=["utr_start", "utr_end"])
    out: dict[str, list[tuple[int, int]]] = {}
    for tid, a, b in zip(s["transcript_id"], s["utr_start"].astype(int), s["utr_end"].astype(int)):
        out.setdefault(str(tid), []).append((a, b))
    return out


def build_accessibility(root: Path = DATA_ROOT, *, workers: int = 1, registry: str | Path = REGISTRY,
                        sequences: pd.DataFrame | None = None,
                        sites: dict[str, list[tuple[int, int]]] | None = None,
                        progress: bool = False) -> tuple[pd.DataFrame, dict]:
    """The table and its LINEAGE entry (not written). ``sequences`` (transcript_id, gene_id, symbol, seq)
    and ``sites`` default to TargetScan's; a test passes its own."""
    root = Path(root).expanduser()
    ts = MiRNATargetSource(spec_from_registry(TARGETSCAN_BLOCK, registry), {}, root=root)
    if sequences is None:
        rep_ids = set(ts._representative()["transcript_id"])
        rows, seen = [], set()
        for tid, gid, sym, seq in ts._iter_human_utr_sequences():
            if tid in rep_ids and tid not in seen:
                seen.add(tid)
                rows.append((tid, gid, sym, seq))
        sequences = pd.DataFrame(rows, columns=["transcript_id", "gene_id", "symbol", "seq"])
    if sites is None:
        sites = conserved_sites(root)
    jobs = [(t, g, s, q, sites.get(t, [])) for t, g, s, q in
            zip(sequences["transcript_id"], sequences["gene_id"], sequences["symbol"], sequences["seq"])]
    if workers > 1:
        with Pool(workers) as pool:
            out = []
            for i, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=16)):
                out.append(r)
                if progress and i % 2000 == 0:
                    print(f"  folded {i + 1} / {len(jobs)}", flush=True)
    else:
        out = [_work(j) for j in jobs]
    table = pd.DataFrame(out)
    table = table[list(COLUMNS)].sort_values("transcript_id").reset_index(drop=True)
    n_sites = int(sum(len(v) for v in sites.values()))
    entry = {
        "dataset": TARGETSCAN_BLOCK,
        "what": "RNAplfold accessibility of each representative 3'UTR: per-nucleotide and 8-nt unpaired "
                "probabilities, around the conserved miRNA sites and over the consensus RBP motifs (T120 phase 2)",
        "derives_from": str(ts.dest / "PROVENANCE.json"),
        "inputs": {"targetscan": {"dest": str(ts.dest),
                                  "files": {n: sha256_of(ts.dest / n) for n in ("UTR_Sequences.txt.zip", "Gene_info.txt.zip")
                                            if (ts.dest / n).exists()}},
                   "sites": {"path": str(root / TS_DERIVED / SITES_FILE),
                             "sha256": sha256_of(root / TS_DERIVED / SITES_FILE) if (root / TS_DERIVED / SITES_FILE).exists() else None,
                             "coordinates": "1-based inclusive on the ungapped UTR (targetscan-vert_80/utr_seq_features LINEAGE)"}},
        "code": "src/sidechain/priors/posttx_access.py::build_accessibility", "code_sha": _code_sha(),
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "params": {"tool": "ViennaRNA RNAplfold (RNA.pfl_fold_up)", "window": WINDOW, "span": SPAN, "ulen": ULEN,
                   "flank": FLANK, "open_at": OPEN, "workers": workers},
        "counts": {"transcripts": int(len(table)), "bases_folded": int(table["seq_len"].sum()),
                   "conserved_sites_on_representatives": n_sites,
                   "transcripts_with_a_site": int((table["acc_mirna_n_sites"] > 0).sum()),
                   "median_acc_mean_u1": float(table["acc_mean_u1"].median()),
                   "median_site_excess": float(table["acc_mirna_site_excess"].median())},
        "columns": COLUMNS, "licence": ts.spec.get("license"),
    }
    return table, entry


def write_accessibility(root: Path = DATA_ROOT, **kw) -> Path:
    root = Path(root).expanduser()
    out = root / DERIVED
    out.mkdir(parents=True, exist_ok=True)
    table, entry = build_accessibility(root, **kw)
    table.to_parquet(out / TABLE_FILE, index=False)
    ts_prov = Path(entry["derives_from"])
    flags = json.loads(ts_prov.read_text()).get("license_flags") if ts_prov.exists() else None
    entry = {**entry, "sha256": sha256_of(out / TABLE_FILE), "bytes": (out / TABLE_FILE).stat().st_size,
             "license_flags": flags}
    path = out / "LINEAGE.json"
    lineage = json.loads(path.read_text()) if path.exists() else {"schema_version": 2, "entries": {}}
    lineage.setdefault("entries", {})[LINEAGE_KEY] = entry
    path.write_text(json.dumps(lineage, indent=2) + "\n")
    return out
