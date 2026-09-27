"""Post-transcriptional layer: miRNA -> target repression edges (Saber's edge).

Primary source miRBind2 (sequence-only binding + repression score); TargetScan/
miRDB as toggleable alternates. Directed edges onto the target gene, weighted by
repression strength — refines the *magnitude* of downstream deltas the trans graph
predicts (mRNA stability via 3'UTR).

**What is built (T102, 2026-09-25) and what is not.** The EDGE form -- miRNA-family ->
target-gene edges for a graph head -- is still unbuilt: ``build()`` raises. What exists is
the first thing the layer was ever asked for in 2026, a *per-gene 3'UTR regulatory load*:
how many conserved miRNA sites a gene's representative 3'UTR carries, how long that UTR
is, and TargetScan's own summed repression score. ``fetch()`` brings TargetScan Human 8.0's
three small tables through the ADR 0003 gate, and ``utr_load_table()`` turns them into one
row per gene, cached as a parquet under ``derived/`` with a LINEAGE.json beside it. The
question it serves -- does that load predict how badly a gene's knockdown response
transfers across cell lines? -- is asked in ``sidechain.eval.per_gene_transfer`` and
answered in ``private/research/ideas/utr-layer-context-invariance.md``.

Why a fetch here rather than a ``datasets.yaml`` block: ``data_sources.yaml`` is the prior
registry and the ``targetscan`` block already lived there. The block now carries the same
``host / record / files / dest / budget_gb / license`` fields a corpus block does, and
``fetch()`` hands it to ``sidechain.ingest.fetch.run_gate`` -- probe, gate, PROVENANCE.json
before any byte lands -- so a prior enters under exactly the rules a corpus does. The
download itself is in-process (three files, 11.6 MB) rather than the curl plan the corpus
entry point prints, and the sha256 of what landed is recorded in LINEAGE.json because the
host publishes no checksum (``probe_https``).
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import urllib.request
import zipfile
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
    "utr_len": "3'UTR length in nt: sum of the transcript's 3'UTR exon widths (hg19 GFF)",
    "n_cons_sites": "conserved sites of conserved miRNA families, all types, summed over families",
    "n_cons_8mer": "of those, 8mer sites",
    "n_cons_7mer": "of those, 7mer-m8 plus 7mer-1a sites",
    "n_families": "conserved miRNA families with at least one conserved site",
    "n_noncons_sites": "nonconserved sites of conserved families (the default file carries them too)",
    "context_score": "sum over families of TargetScan's cumulative weighted context++ score "
                     "(negative; more negative = stronger predicted repression)",
    "aggregate_pct_max": "the largest aggregate PCT over the gene's families",
    "sites_per_kb": "n_cons_sites per kb of 3'UTR",
}


def spec_from_registry(name: str = "targetscan", registry: str | Path = REGISTRY) -> dict:
    """The named block of the prior registry, enabled or not.

    `load_registry` skips a disabled block on purpose (a shelved source may name a loader
    that is not written). This reads the block regardless, because fetching a prior's
    raw table and building an edge layer from it are different decisions: the
    ``targetscan`` block stays ``enabled: false`` while its table is in use.
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


class MiRNATargetSource(PriorSource):
    """TargetScan (and, unbuilt, miRBind2) as a prior source.

    Constructed like every PriorSource -- ``(spec, gene_index)`` -- but the two entry
    points T102 uses need no gene index: ``fetch()`` and ``utr_load_table()`` work on the
    spec alone, so ``gene_index`` may be an empty dict for those.
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

    def fetch(self, *, refresh: bool = False) -> Path:
        """Gate, record provenance, then download the block's files. Idempotent.

        Order: ``run_gate`` (probe -> gate -> PROVENANCE.json) exactly as
        ``sidechain.ingest.fetch`` runs it for a corpus, then each selected file that is
        not already on disk is downloaded and its size checked against the probe's
        Content-Length. Returns the destination directory.
        """
        from sidechain.ingest.fetch import run_gate

        if "host" not in self.spec:
            raise ValueError(f"{self.name}: the registry block declares no host/record/files, "
                             "so there is nothing to fetch through the gate")
        record, selected, dest = run_gate(self.spec, self.root, refresh=refresh, config=REGISTRY)
        for f in selected:
            target = dest / f.name
            if target.exists() and target.stat().st_size == f.size_bytes:
                continue
            req = urllib.request.Request(f.url, headers={"User-Agent": USER_AGENT})
            tmp = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(req, timeout=300) as resp, tmp.open("wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            got = tmp.stat().st_size
            if got != f.size_bytes:
                tmp.unlink()
                raise RuntimeError(f"{f.name}: downloaded {got} bytes, the probe said "
                                   f"{f.size_bytes}; refusing to keep a short file")
            tmp.replace(target)
        return dest

    # ------------------------------------------------------------- the table --

    def _read_zipped_table(self, name: str, *, skip_prefixes: tuple[str, ...] = (),
                           **read_kwargs) -> pd.DataFrame:
        """The one text file inside `name` (TargetScan zips one table per zip).

        `skip_prefixes` drops leading lines that are not rows -- the hg19 GFF opens with a
        UCSC `browser` line and a `track` line, which are neither a header nor a comment.
        """
        with zipfile.ZipFile(self.dest / name) as z:
            members = [m for m in z.namelist() if not m.endswith("/")]
            if len(members) != 1:
                raise ValueError(f"{name}: expected one member, found {members}")
            with z.open(members[0]) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8")
                if skip_prefixes:
                    text = io.StringIO("".join(
                        line for line in text if not line.startswith(skip_prefixes)))
                return pd.read_csv(text, sep="\t", **read_kwargs)

    def utr_load_table(self, *, rebuild: bool = False, register_note: str | None = None
                       ) -> pd.DataFrame:
        """One row per human gene: 3'UTR length, conserved-site counts, summed context++.

        Cached at ``<derived>/utr_load.parquet`` with a ``LINEAGE.json`` beside it that
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

        info = self._read_zipped_table("Gene_info.txt.zip", dtype=str)
        info.columns = [c.strip() for c in info.columns]
        col = {c.lower(): c for c in info.columns}
        species = col.get("species id")
        rep = next((c for c in info.columns if c.lower().startswith("representative")), None)
        if species is None or rep is None:
            raise ValueError(f"Gene_info.txt: unexpected columns {list(info.columns)}")
        n_info = len(info)
        info = info[info[species].astype(int) == HUMAN]
        n_human = len(info)
        info = info[info[rep].astype(str).str.strip().isin({"1", "yes", "Yes", "TRUE", "True"})]
        genes = pd.DataFrame({
            "transcript_id": info[col["transcript id"]].str.split(".").str[0].str.strip(),
            "gene_id": info[col["gene id"]].str.split(".").str[0].str.strip(),
            "symbol": info[col["gene symbol"]].str.strip(),
        }).drop_duplicates("transcript_id")

        # 3'UTR length: the GFF's score column is the exon width (the download page says
        # so, and on 42,426 of 42,427 rows it equals end - start: the coordinates are
        # BED-style half-open despite the .gff name, so end - start + 1 would overcount
        # every exon by one). A transcript's 3'UTR is the sum of its exon rows. The
        # publisher's width is used where present, end - start where it is not.
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
        width_from_coords = (end - start)
        score = pd.to_numeric(gff.iloc[:, 5], errors="coerce")
        agree = np.isfinite(score) & (score == width_from_coords)
        width = np.where(np.isfinite(score), score, width_from_coords).astype(np.int64)
        utr = pd.DataFrame({"transcript_id": tid, "width": width}).dropna()
        utr_len = utr.groupby("transcript_id")["width"].sum().rename("utr_len")

        counts = self._read_zipped_table("Summary_Counts.default_predictions.txt.zip", dtype=str)
        counts.columns = [c.strip() for c in counts.columns]
        cc = {c.lower(): c for c in counts.columns}
        need = ["transcript id", "species id", "total num conserved sites",
                "number of conserved 8mer sites", "number of conserved 7mer-m8 sites",
                "number of conserved 7mer-1a sites", "total num nonconserved sites",
                "cumulative weighted context++ score", "aggregate pct"]
        missing = [k for k in need if k not in cc]
        if missing:
            raise ValueError(f"Summary_Counts: missing columns {missing}; have {list(counts.columns)}")
        n_counts = len(counts)
        counts = counts[counts[cc["species id"]].astype(int) == HUMAN].copy()
        counts["transcript_id"] = counts[cc["transcript id"]].str.split(".").str[0].str.strip()
        num = lambda k: pd.to_numeric(counts[cc[k]], errors="coerce").fillna(0.0)  # noqa: E731
        counts["_cons"] = num("total num conserved sites")
        counts["_8mer"] = num("number of conserved 8mer sites")
        counts["_7mer"] = num("number of conserved 7mer-m8 sites") + num("number of conserved 7mer-1a sites")
        counts["_noncons"] = num("total num nonconserved sites")
        counts["_ctx"] = pd.to_numeric(counts[cc["cumulative weighted context++ score"]], errors="coerce")
        counts["_pct"] = pd.to_numeric(counts[cc["aggregate pct"]], errors="coerce")
        g = counts.groupby("transcript_id")
        load = pd.DataFrame({
            "n_cons_sites": g["_cons"].sum(),
            "n_cons_8mer": g["_8mer"].sum(),
            "n_cons_7mer": g["_7mer"].sum(),
            "n_families": g["_cons"].apply(lambda s: int((s > 0).sum())),
            "n_noncons_sites": g["_noncons"].sum(),
            "context_score": g["_ctx"].sum(min_count=1),
            "aggregate_pct_max": g["_pct"].max(),
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

        prov = json.loads((self.dest / "PROVENANCE.json").read_text())
        inputs = {f["name"]: {"bytes": f["size_bytes"], "host_evidence": f["checksum"],
                              "sha256": sha256_of(self.dest / f["name"])}
                  for f in prov["selected"]}
        try:
            code_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                      text=True, check=True, cwd=Path(__file__).parent).stdout.strip()
        except Exception:                                          # noqa: BLE001
            code_sha = "unknown"
        lineage = {
            "schema_version": 2,
            "entries": {
                "targetscan-vert_80/utr_load": {
                    "dataset": self.name,
                    "what": "one row per human gene: representative-transcript 3'UTR length, "
                            "conserved miRNA site counts, summed context++ score (T102)",
                    "derives_from": str(self.dest / "PROVENANCE.json"),
                    "inputs": inputs,
                    "code": "src/sidechain/priors/posttx_mirna.py::MiRNATargetSource.utr_load_table",
                    "code_sha": code_sha,
                    "built": datetime.now(UTC).isoformat(timespec="seconds"),
                    "sha256": sha256_of(out_path),
                    "bytes": out_path.stat().st_size,
                    "columns": UTR_LOAD_COLUMNS,
                    "counts": {
                        "gene_info_rows": n_info, "gene_info_human_rows": n_human,
                        "representative_transcripts": int(len(genes)),
                        "gff_rows": int(len(gff)), "gff_transcripts": int(utr_len.size),
                        "gff_score_equals_width_on_rows": int(agree.sum()),
                        "summary_counts_rows": n_counts, "summary_counts_human_rows": int(len(counts)),
                        "transcripts_with_sites": int(len(load)),
                        "dropped_no_utr_in_gff": n_no_utr,
                        "symbols_deduplicated": dup_symbols,
                        "genes_out": int(len(table)),
                    },
                    "licence": self.spec.get("license"),
                    "note": register_note or "",
                },
            },
        }
        (self.derived / "LINEAGE.json").write_text(json.dumps(lineage, indent=2) + "\n")
        return table

    # ------------------------------------------------------------------ build --

    def build(self) -> PriorArtifact:
        # The EDGE form (miRNA family -> target gene, weighted by context++ score) is
        # not built: nothing in the 2026 shipping path consumes an edge list, and
        # T102's per-gene load is read straight from `utr_load_table()`.
        raise NotImplementedError(
            f"{self.name}: the miRNA->target edge layer is unbuilt; the per-gene 3'UTR "
            "load is MiRNATargetSource.utr_load_table() (T102)")
