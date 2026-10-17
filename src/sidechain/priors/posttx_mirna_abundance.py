"""Measured per-line miRNA abundance, and each gene's TargetScan site load weighted by the line's own
miRNA repertoire (T120, phase 3).

    .venv/bin/python scripts/t120_build_mirna_abundance.py           # fetch through the gate, build the three sources

Reads, under ``~/data/sidechain/``, the three blocks of ``configs/data_sources.yaml`` this loader
serves (``MiRNAAbundanceSource``): ``ccle_mirna_2019`` (the NanoString panel, ``CCLE_miRNA_20181103.gct``),
``encode_mirna_seq`` (fourteen STAR per-gene count files, GENCODE V29 miRNA genes) and
``mirnatissueatlas_2025`` (the harmonised small-RNA-seq h5ad, reads per million mapped); plus
TargetScan's ``miR_Family_Info.txt.zip`` (``external/targetscan-vert_80/``), its per-gene x family
site table ``derived/targetscan-vert_80/edges_default.parquet`` and, for ENCODE's gene ids, GENCODE
v47's annotation (``external/gencode-v47/``). Writes, under each block's ``derived/``,
``mirna_family_abundance.parquet`` (one row per line x TargetScan family: the summed abundance of the
family's mapped miRNAs), ``mirna_load.parquet`` (one row per gene: the per-line columns and the
cross-line mean and spread) and a LINEAGE.json entry with the column dictionary.

**What a column is.** TargetScan counts, per gene and miRNA family, the conserved 3'UTR sites of the
broadly conserved families (``n_cons_sites`` in ``edges_default``); that count knows nothing about
which miRNAs a given line expresses. Here each site is weighted by the line's own abundance of the
family: ``mirna_load_log_<line>`` is the sum over the gene's families of sites x log2(1 + abundance),
``mirna_sites_expressed_<line>`` counts the sites of families above the line's median family
abundance, ``mirna_top_family_log_<line>`` is the most abundant family among the gene's sites. The
cross-line mean and standard deviation of the load (``mirna_load_log_mean``, ``_sd``) are universal
columns a per-line read can nest on. Each source keeps its own scale (nSolver counts, CPM, rpmm), so a
column compares lines within a source, never across sources.

**Names onto families** (``family_of``). TargetScan names a mature miRNA ``hsa-let-7a-5p``; the
NanoString panel names most probes without the arm (``hsa-let-7a``) and ENCODE's counts are per
precursor GENE (``MIRLET7A1``), which is translated to the mature stem. The arm is then chosen by one
rule: an exact match; else the one arm that exists; else, of two arms, the one whose family TargetScan
calls conserved; else unmatched, counted in LINEAGE and dropped. A combined probe (``hsa-miR-106a+hsa-
miR-17``) maps only when every part lands on the same family.

The idea and the results: private ``research/ideas/ptr-feature-table.md`` § Outcome; the task:
private ``tasks/T120.md``.
"""
from __future__ import annotations

import gzip
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
    fetch_gated_files,
    sha256_of,
    spec_from_registry,
)

CCLE_BLOCK = "ccle_mirna_2019"
ENCODE_BLOCK = "encode_mirna_seq"
ATLAS_BLOCK = "mirnatissueatlas_2025"
BLOCKS: tuple[str, ...] = (CCLE_BLOCK, ENCODE_BLOCK, ATLAS_BLOCK)

FAMILY_FILE = "external/targetscan-vert_80/miR_Family_Info.txt.zip"
EDGES_FILE = "derived/targetscan-vert_80/edges_default.parquet"
GENCODE_GTF = "external/gencode-v47/gencode.v47.basic.annotation.gtf.gz"
HUMAN_TAXON = "9606"

CCLE_FILE = "CCLE_miRNA_20181103.gct"
#: our spelling -> the panel's column
CCLE_LINES: dict[str, str] = {
    "k562": "K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE", "hct116": "HCT116_LARGE_INTESTINE",
    "hepg2": "HEPG2_LIVER", "jurkat": "JURKAT_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE",
    "cal33": "CAL33_UPPER_AERODIGESTIVE_TRACT",
}
#: ENCODE's line names (the manifest's note) -> our spelling
ENCODE_LINES: dict[str, str] = {"H1": "h1", "HCT116": "hct116", "HepG2": "hepg2", "K562": "k562"}
ATLAS_H5AD = "all_human_miRNA_rpmm_harmonized_meta_data.h5ad"
ATLAS_LABEL_COLUMN = "Tissue"
ATLAS_BIOTYPE_COLUMN = "Biotype"
ATLAS_BIOTYPE = "cell_line"
#: the atlas's labels -> our spelling (HEK293T is folded into HEK293 there)
ATLAS_LINES: dict[str, str] = {"hela": "HeLa", "hek293": "HEK293", "k562": "K562", "hct116": "HCT116",
                               "hepg2": "HepG2", "jurkat": "Jurkat"}

ABUNDANCE_FILE = "mirna_family_abundance.parquet"
LOAD_FILE = "mirna_load.parquet"
UNIVERSAL_COLUMNS = ("mirna_load_log_mean", "mirna_load_log_sd")


# ------------------------------------------------------------- names -> families --

def read_family_table(root: Path) -> pd.DataFrame:
    """TargetScan's human mature miRNAs: MiRBase ID, family, conserved flag (``Family Conservation?``
    of 1 or 2 means a broadly or conserved family; -1 and 0 are poorly conserved)."""
    z = zipfile.ZipFile(Path(root) / FAMILY_FILE)
    d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), sep="\t", dtype=str)
    d = d[d["Species ID"] == HUMAN_TAXON].rename(columns={"miR family": "family", "MiRBase ID": "mirbase_id",
                                                           "Family Conservation?": "conservation"})
    d["conserved"] = pd.to_numeric(d["conservation"], errors="coerce").fillna(-1) >= 1
    return d[["mirbase_id", "family", "conserved"]].drop_duplicates("mirbase_id").reset_index(drop=True)


def family_of(name: str, fam: pd.DataFrame, *, _index: dict | None = None) -> tuple[str | None, str]:
    """One miRBase-style name onto a TargetScan family: (family, how). ``how`` is ``exact``, ``one_arm``,
    ``conserved_arm``, ``combined`` or the reason it failed (``ambiguous_arms``, ``combined_disagree``,
    ``none``)."""
    idx = _index if _index is not None else _family_index(fam)
    by_id, conserved = idx
    n = str(name).strip()
    if "+" in n:
        parts = [family_of(p, fam, _index=idx) for p in n.split("+")]
        fams = {f for f, _ in parts if f is not None}
        if len(fams) == 1 and all(f is not None for f, _ in parts):
            return fams.pop(), "combined"
        return None, "combined_disagree" if fams else "none"
    if n in by_id:
        return by_id[n], "exact"
    arms = [a for a in (n + "-5p", n + "-3p") if a in by_id]
    if len(arms) == 1:
        return by_id[arms[0]], "one_arm"
    if len(arms) == 2:
        cons = [a for a in arms if conserved.get(a, False)]
        if len(cons) == 1:
            return by_id[cons[0]], "conserved_arm"
        return None, "ambiguous_arms"
    return None, "none"


def _family_index(fam: pd.DataFrame) -> tuple[dict[str, str], dict[str, bool]]:
    return (dict(zip(fam["mirbase_id"], fam["family"])), dict(zip(fam["mirbase_id"], fam["conserved"])))


GENE_NAME = re.compile(r"^MIR(LET)?(\d+)([A-Z]?)(?:-?\d+)?$")


def mature_stem(gene_name: str) -> str | None:
    """A GENCODE miRNA gene name (``MIRLET7A1``, ``MIR21``, ``MIR9-1``) as the mature stem TargetScan
    names (``hsa-let-7a``, ``hsa-miR-21``, ``hsa-miR-9``); None when the name is not of that shape."""
    m = GENE_NAME.match(str(gene_name).strip())
    if not m:
        return None
    let, num, letter = m.groups()
    return f"hsa-{'let' if let else 'miR'}-{num}{letter.lower()}"


def map_names(names: pd.Series, fam: pd.DataFrame) -> tuple[pd.Series, dict[str, int]]:
    idx = _family_index(fam)
    out = []
    counts: dict[str, int] = {}
    for n in names.astype(str):
        f, how = family_of(n, fam, _index=idx)
        out.append(f)
        counts[how] = counts.get(how, 0) + 1
    return pd.Series(out, index=names.index, dtype=object), counts


def gencode_mirna_names(root: Path) -> dict[str, str]:
    """Unversioned Ensembl gene id -> gene name for GENCODE v47's miRNA genes."""
    out: dict[str, str] = {}
    with gzip.open(Path(root) / GENCODE_GTF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 9)
            if f[2] != "gene" or 'gene_type "miRNA"' not in f[8]:
                continue
            gid = re.search(r'gene_id "([^"]+)"', f[8]).group(1).split(".")[0]
            out[gid] = re.search(r'gene_name "([^"]+)"', f[8]).group(1)
    return out


# ------------------------------------------------------------------ the load --

def gene_loads(abund: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per gene, per line: the site load weighted by the line's family abundance.

    ``abund`` has columns line, family, abundance (summed over the family's mapped miRNAs); ``edges``
    has gene_id, symbol, family, n_cons_sites. A family with no abundance row in a line contributes
    nothing to that line's columns and is counted. The expression floor per line is the median
    abundance over the families that have at least one site in ``edges``.
    """
    sites = edges.groupby(["gene_id", "symbol", "family"], as_index=False)["n_cons_sites"].sum()
    sites = sites[sites["n_cons_sites"] > 0]
    fam_with_sites = set(sites["family"])
    lines = sorted(abund["line"].unique())
    genes = sites[["gene_id", "symbol"]].drop_duplicates("gene_id").sort_values("gene_id").reset_index(drop=True)
    out = genes.copy()
    columns: dict[str, str] = {}
    counts: dict = {"lines": lines, "families_with_sites": len(fam_with_sites), "per_line": {}}
    load_cols = []
    for line in lines:
        a = abund[abund["line"] == line].set_index("family")["abundance"]
        a = a[a.index.isin(fam_with_sites)]
        floor = float(a.median()) if len(a) else float("nan")
        joined = sites.merge(a.rename("abundance"), left_on="family", right_index=True, how="inner")
        joined["log_a"] = np.log2(1.0 + joined["abundance"].clip(lower=0))
        joined["w"] = joined["n_cons_sites"] * joined["log_a"]
        joined["expressed"] = joined["n_cons_sites"] * (joined["abundance"] >= floor)
        g = joined.groupby("gene_id").agg(load=("w", "sum"), expressed=("expressed", "sum"), top=("log_a", "max"))
        col_load, col_exp, col_top = (f"mirna_load_log_{line}", f"mirna_sites_expressed_{line}", f"mirna_top_family_log_{line}")
        out[col_load] = out["gene_id"].map(g["load"])
        out[col_exp] = out["gene_id"].map(g["expressed"])
        out[col_top] = out["gene_id"].map(g["top"])
        load_cols.append(col_load)
        columns[col_load] = (f"sum over the gene's TargetScan families of conserved 3'UTR sites x log2(1 + the family's "
                             f"abundance in {line}); a gene whose families are all unmeasured is NaN")
        columns[col_exp] = (f"conserved sites of families at or above {line}'s median family abundance ({floor:.3g} on "
                            f"the source's scale)")
        columns[col_top] = f"log2(1 + abundance) of the most abundant family with a site in the gene, in {line}"
        counts["per_line"][line] = {"families_measured_with_sites": int(len(a)), "floor": floor,
                                    "genes_with_load": int(out[col_load].notna().sum()),
                                    "load_median": float(out[col_load].median())}
    out["mirna_load_log_mean"] = out[load_cols].mean(axis=1)
    out["mirna_load_log_sd"] = out[load_cols].std(axis=1, ddof=1) if len(load_cols) > 1 else np.nan
    columns["mirna_load_log_mean"] = f"mean of mirna_load_log over the source's {len(lines)} lines (universal)"
    columns["mirna_load_log_sd"] = f"standard deviation of mirna_load_log over the source's {len(lines)} lines (universal)"
    # how different the lines' repertoires are, on the families with sites
    wide = abund[abund["family"].isin(fam_with_sites)].pivot_table(index="family", columns="line", values="abundance", aggfunc="sum")
    counts["line_family_spearman"] = {f"{a}~{b}": float(wide[a].corr(wide[b], method="spearman"))
                                      for i, a in enumerate(lines) for b in lines[i + 1:]} if len(lines) > 1 else {}
    counts["genes"] = int(len(out))
    return out, columns, counts


class MiRNAAbundanceSource(PriorSource):
    """One source of per-line miRNA abundance as per-gene site loads; the parser follows the block."""

    def __init__(self, spec: dict, gene_index: dict[str, int], root: Path = DATA_ROOT,
                 registry: str | Path = REGISTRY):
        super().__init__(spec, gene_index)
        if self.name not in BLOCKS:
            raise ValueError(f"{self.name!r} is not a block this loader serves; one of {BLOCKS}")
        self.root = Path(root).expanduser()
        self.registry = registry

    @property
    def dest(self) -> Path:
        return self.root / self.spec["dest"]

    @property
    def derived(self) -> Path:
        return self.root / self.spec["derived"]

    @property
    def lineage_key(self) -> str:
        return f"{Path(self.spec['derived']).name}/mirna_load"

    def fetch(self, *, refresh: bool = False, progress: bool = False) -> Path:
        return fetch_gated_files(self.spec, self.root, refresh=refresh, progress=progress, registry=self.registry)

    # ----------------------------------------------------------------- lineage --

    def _code_sha(self) -> str:
        try:
            return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                  text=True, check=True, cwd=Path(__file__).parent).stdout.strip()
        except Exception:                                          # noqa: BLE001
            return "unknown"

    def _inputs(self, names: list[str], extra: list[Path]) -> dict:
        prov = self.dest / "PROVENANCE.json"
        by_name = {f["name"]: f for f in (json.loads(prov.read_text()) if prov.exists() else {}).get("selected", [])}
        out = {}
        for name in names:
            f = by_name.get(name, {})
            out[name] = {"dest": str(self.dest), "bytes": f.get("size_bytes"),
                         "host_evidence": f.get("checksum"), "sha256": sha256_of(self.dest / name)}
        for path in extra:
            out[str(path.relative_to(self.root))] = {"sha256": sha256_of(path), "bytes": path.stat().st_size}
        return out

    def _record_lineage(self, key: str, *, what: str, inputs: dict, code: str, counts: dict, out_path: Path,
                        columns: dict | None = None, note: str = "") -> None:
        path = self.derived / "LINEAGE.json"
        lineage = json.loads(path.read_text()) if path.exists() else {"schema_version": 2, "entries": {}}
        lineage.setdefault("entries", {})
        entry = {"dataset": self.name, "what": what, "derives_from": str(self.dest / "PROVENANCE.json"),
                 "inputs": inputs, "code": code, "code_sha": self._code_sha(),
                 "built": datetime.now(UTC).isoformat(timespec="seconds"),
                 "sha256": sha256_of(out_path), "bytes": out_path.stat().st_size,
                 "counts": counts, "licence": self.spec.get("license"), "note": note}
        if columns:
            entry["columns"] = columns
        lineage["entries"][key] = entry
        path.write_text(json.dumps(lineage, indent=2) + "\n")

    # -------------------------------------------------------- per-line abundance --

    def abundance_table(self, *, rebuild: bool = False) -> pd.DataFrame:
        """Rows (line, family, abundance, n_mirnas): the family's summed abundance per line, cached at
        ``<derived>/mirna_family_abundance.parquet``."""
        out_path = self.derived / ABUNDANCE_FILE
        if out_path.exists() and not rebuild:
            return pd.read_parquet(out_path)
        self.derived.mkdir(parents=True, exist_ok=True)
        fam = read_family_table(self.root)
        builder = {CCLE_BLOCK: self._ccle, ENCODE_BLOCK: self._encode, ATLAS_BLOCK: self._atlas}[self.name]
        per_mirna, meta = builder()                     # columns: line, name, value (+ anything)
        families, how = map_names(per_mirna["name"], fam)
        per_mirna = per_mirna.assign(family=families.to_numpy())
        mapped = per_mirna[per_mirna["family"].notna()]
        table = (mapped.groupby(["line", "family"], as_index=False)
                 .agg(abundance=("value", "sum"), n_mirnas=("name", "nunique")))
        table = table.sort_values(["line", "family"]).reset_index(drop=True)
        table.to_parquet(out_path, index=False)
        lines = sorted(table["line"].unique())
        meta["counts"].update({
            "names_in": int(per_mirna["name"].nunique()), "name_mapping": how,
            "families_measured": int(table["family"].nunique()), "lines": lines,
            "share_of_signal_mapped": {l: float(mapped.loc[mapped["line"] == l, "value"].sum()
                                                / max(per_mirna.loc[per_mirna["line"] == l, "value"].sum(), 1e-9))
                                       for l in lines},
        })
        self._record_lineage(f"{Path(self.spec['derived']).name}/mirna_family_abundance", out_path=out_path,
                             what=meta["what"], inputs=meta["inputs"], code=meta["code"], counts=meta["counts"],
                             note=meta.get("note", ""))
        return table

    def _ccle(self) -> tuple[pd.DataFrame, dict]:
        path = self.dest / CCLE_FILE
        with path.open() as fh:
            version = fh.readline().strip()
            dims = fh.readline().split()
        raw = pd.read_csv(path, sep="\t", skiprows=2)
        missing = [c for c in CCLE_LINES.values() if c not in raw.columns]
        if missing:
            raise ValueError(f"{CCLE_FILE}: lines missing from the panel: {missing}")
        rows = []
        for line, col in CCLE_LINES.items():
            rows.append(pd.DataFrame({"line": line, "name": raw["Description"].astype(str).str.strip(),
                                      "probe": raw["Name"].astype(str), "value": pd.to_numeric(raw[col], errors="coerce")}))
        per = pd.concat(rows, ignore_index=True)
        meta = {"what": "CCLE 2019 NanoString miRNA abundance per TargetScan family for five of our lines "
                        "(nSolver-normalised counts summed over the family's mapped probes; T120 phase 3)",
                "inputs": self._inputs([CCLE_FILE], [self.root / FAMILY_FILE]),
                "code": "src/sidechain/priors/posttx_mirna_abundance.py::MiRNAAbundanceSource._ccle",
                "counts": {"gct_version": version, "gct_dims": dims, "probes": int(len(raw)),
                           "panel_lines": int(raw.shape[1] - 2), "lines_taken": dict(CCLE_LINES)},
                "note": "the probe named without an arm is the panel's design; the arm rule is family_of's"}
        return per, meta

    def _encode_files(self) -> list[tuple[str, str, str]]:
        """(file name, line, note) for every manifest entry, the line read off the note."""
        from sidechain.ingest.fetch import block_files
        out = []
        for f in block_files(self.spec):
            note = str(f.get("note", ""))
            line = next((ours for theirs, ours in ENCODE_LINES.items() if f" {theirs} " in f" {note} "), None)
            if line is None:
                raise ValueError(f"{self.name}: cannot read the line off the manifest note {note!r}")
            out.append((f["name"], line, note))
        return out

    def _encode(self) -> tuple[pd.DataFrame, dict]:
        files = self._encode_files()
        names = gencode_mirna_names(self.root)
        per_file = []
        for name, line, note in files:
            d = pd.read_csv(self.dest / name, sep="\t", header=None, names=["gene_id", "unstranded", "sense", "antisense"])
            d = d[d["gene_id"].astype(str).str.startswith("ENSG")].copy()
            d["gid"] = d["gene_id"].str.split(".").str[0]
            total = float(d["unstranded"].sum())
            d["cpm"] = d["unstranded"] / total * 1e6 if total > 0 else np.nan
            d["gene_name"] = d["gid"].map(names)
            d["file"] = name
            d["line"] = line
            per_file.append(d)
        allf = pd.concat(per_file, ignore_index=True)
        unnamed = int(allf["gene_name"].isna().groupby(allf["file"]).sum().iloc[0]) if len(allf) else 0
        allf["stem"] = allf["gene_name"].map(lambda g: mature_stem(g) if isinstance(g, str) else None)
        # replicate agreement per line, on log CPM, before averaging
        rep_rho: dict[str, float] = {}
        for line, g in allf.groupby("line"):
            w = g.pivot_table(index="gid", columns="file", values="cpm")
            if w.shape[1] >= 2:
                lw = np.log2(1 + w)
                rhos = [lw.iloc[:, i].corr(lw.iloc[:, j], method="spearman") for i in range(w.shape[1]) for j in range(i + 1, w.shape[1])]
                rep_rho[line] = float(np.median(rhos))
        per_gene = (allf[allf["stem"].notna()].groupby(["line", "gid", "stem"], as_index=False)["cpm"].mean())
        per = per_gene.rename(columns={"stem": "name", "cpm": "value"})[["line", "name", "value", "gid"]]
        meta = {"what": "ENCODE microRNA-seq abundance per TargetScan family for H1, HCT116, HepG2 and K562 (STAR per-gene "
                        "counts as CPM, averaged over a line's replicate files, summed over the family's mapped genes; T120 phase 3)",
                "inputs": self._inputs([n for n, _, _ in files], [self.root / FAMILY_FILE, self.root / GENCODE_GTF]),
                "code": "src/sidechain/priors/posttx_mirna_abundance.py::MiRNAAbundanceSource._encode",
                "counts": {"files": [{"file": n, "line": l, "note": t} for n, l, t in files],
                           "mirna_gene_rows_per_file": int(len(per_file[0])) if per_file else 0,
                           "gene_ids_without_v47_name": unnamed,
                           "gene_names_without_mature_stem": int(allf.loc[allf["gene_name"].notna() & allf["stem"].isna(), "gid"].nunique()),
                           "replicate_spearman_log_cpm_median": rep_rho},
                "note": "per precursor gene, not per mature arm; the arm is chosen by family_of's rule on the stem"}
        return per, meta

    def _atlas(self) -> tuple[pd.DataFrame, dict]:
        import anndata as ad
        a = ad.read_h5ad(self.dest / ATLAS_H5AD, backed="r")
        obs = a.obs
        for col in (ATLAS_LABEL_COLUMN, ATLAS_BIOTYPE_COLUMN):
            if col not in obs.columns:
                raise ValueError(f"{ATLAS_H5AD}: obs has no column {col!r}; have {list(obs.columns)[:20]}")
        keep = obs[ATLAS_LABEL_COLUMN].astype(str).isin(set(ATLAS_LINES.values())) & (obs[ATLAS_BIOTYPE_COLUMN].astype(str) == ATLAS_BIOTYPE)
        sub = a[np.flatnonzero(keep.to_numpy())].to_memory()
        x = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
        label_to_line = {v: k for k, v in ATLAS_LINES.items()}
        labels = sub.obs[ATLAS_LABEL_COLUMN].astype(str).map(label_to_line).to_numpy()
        var_names = pd.Index(sub.var_names.astype(str))
        rows = []
        samples: dict[str, int] = {}
        for line in sorted(set(labels)):
            m = labels == line
            samples[line] = int(m.sum())
            med = np.median(x[m], axis=0)
            rows.append(pd.DataFrame({"line": line, "name": var_names, "value": med}))
        per = pd.concat(rows, ignore_index=True)
        missing = [k for k in ATLAS_LINES if k not in samples]
        meta = {"what": "miRNATissueAtlas 2025 miRNA abundance per TargetScan family for six lines (the MEDIAN rpmm over "
                        "a line's cell-line samples, summed over the family's mapped mature miRNAs; T120 phase 3)",
                "inputs": self._inputs([ATLAS_H5AD], [self.root / FAMILY_FILE]),
                "code": "src/sidechain/priors/posttx_mirna_abundance.py::MiRNAAbundanceSource._atlas",
                "counts": {"samples_per_line": samples, "lines_without_samples": missing, "mirnas": int(len(var_names)),
                           "obs_total": int(len(obs)), "label_column": ATLAS_LABEL_COLUMN, "biotype_kept": ATLAS_BIOTYPE},
                "note": "pooled public runs incl. perturbed ones; the median is the repertoire, not a baseline"}
        return per, meta

    # ------------------------------------------------------------------ genes --

    def gene_table(self, *, rebuild: bool = False) -> pd.DataFrame:
        """One row per gene with a TargetScan conserved site: the per-line loads and the universal
        mean and spread; cached at ``<derived>/mirna_load.parquet``."""
        out_path = self.derived / LOAD_FILE
        if out_path.exists() and not rebuild:
            return pd.read_parquet(out_path)
        abund = self.abundance_table(rebuild=rebuild)
        edges = pd.read_parquet(self.root / EDGES_FILE, columns=["gene_id", "symbol", "family", "n_cons_sites"])
        edges["gene_id"] = edges["gene_id"].astype(str).str.split(".").str[0]
        table, columns, counts = gene_loads(abund, edges)
        table.to_parquet(out_path, index=False)
        self._record_lineage(self.lineage_key, out_path=out_path,
                             what=f"per-gene TargetScan conserved-site load weighted by each line's own miRNA family abundance "
                                  f"({self.name}), with the cross-line mean and spread (T120 phase 3)",
                             inputs={**self._inputs([], [self.root / EDGES_FILE]),
                                     ABUNDANCE_FILE: {"sha256": sha256_of(self.derived / ABUNDANCE_FILE)}},
                             code="src/sidechain/priors/posttx_mirna_abundance.py::gene_loads", counts=counts, columns=columns,
                             note="conserved sites of broadly conserved families only (edges_default); a column compares "
                                  "lines within this source, never across sources")
        return table

    def build(self) -> PriorArtifact:
        """The universal mean load aligned to the gene space, behind a mask."""
        table = self.gene_table()
        n = len(self.gene_index)
        features = np.zeros((n, 1), dtype=np.float32)
        has_value = np.zeros((n, 1), dtype=bool)
        for gene, value in zip(table["gene_id"].astype(str), pd.to_numeric(table["mirna_load_log_mean"], errors="coerce")):
            pos = self.gene_index.get(gene)
            if pos is None or not np.isfinite(value):
                continue
            features[pos, 0] = np.float32(value)
            has_value[pos, 0] = True
        return PriorArtifact(kind="node_feature", relation=self.relation, layer=self.layer, features=features,
                             meta={"feature_names": ["mirna_load_log_mean"], "has_value": has_value,
                                   "candidates": int(len(table)), "placed": int(has_value.sum())})


def abundance_sources(root: Path = DATA_ROOT, registry: str | Path = REGISTRY) -> list[MiRNAAbundanceSource]:
    return [MiRNAAbundanceSource(spec_from_registry(b, registry), {}, root=root, registry=registry) for b in BLOCKS]
