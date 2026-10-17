"""Measured per-line mRNA stability as one per-gene table per line (T120, phase 3).

    .venv/bin/python scripts/t120_build_line_stability.py           # fetch through the gate, build the four tables

Reads, under ``~/data/sidechain/``, the four blocks of ``configs/data_sources.yaml`` this loader
serves (``LineStabilitySource``): ``jurkat_decay_blackinton2016`` (GEO GSE71382's processed
sheet), ``hct116_actd_pasterczyk2024`` (GSE234877's nine HCT116 RSEM files),
``k562_halflife_schofield2018`` (Nature Methods Supplementary Table 2) and
``hek293_halflife_schueler2014`` (GSE49831's two half-life tables); plus, for the two
symbol-keyed sources, ``vcc2026/gene_names.csv``, the T102 bridge
``runs/probes/t102_utr_load/bridge/symbol_to_ensg.json`` and HGNC's previous-symbol table
``external/hgnc-custom-uniprot-20261005/hgnc_custom_uniprot.tsv``. Writes, under each block's
``derived/``, ``stability.parquet`` (one row per gene: ``gene_id``, ``symbol``, the line's
columns) and a LINEAGE.json entry with the column dictionary; the view (``ptr_table``) joins
them under scope ``line:<name>``.

**What each table is.** Jurkat: the resting decay rate of Blackinton & Keene's 4sU course
(``Avg Opt DR U``, negative by the paper's convention, more negative is faster decay), the
four activated-hour rates beside it, a half-life in hours derived from the resting rate, and
HuR's iCLIP target call and binding scores. HCT116: no rate is published, so one is fitted here
from the nine TPM values (three replicates at 0, 2 and 4 h of actinomycin D): ``log2(TPM +
PSEUDO)``, each sample centred on its median over the expressed genes, then least squares on
hours. The centring sets the level: TPM is a share of the library, and when the unstable half of
the transcriptome decays the stable half's share rises, so an uncentred slope reads every gene
as more stable than it is; the centred slope is a rate RELATIVE to the median gene, negative for
a gene that decays faster than the median, positive for one that decays slower (it shifts every
gene by one constant, so the two slopes rank genes identically). K562: Schofield's
TimeLapse-seq half-lives (two replicates and their mean, hours) with the fraction new after the
4 h label. HEK293 and MCF7: Schueler's 4sU half-lives (two replicates, minutes) and their mean.

**Gene keys.** Jurkat and HCT116 carry Ensembl ids and are joined on them. Schofield's and
Schueler's tables carry symbols only, in the dialect of their year; ``resolve_symbols`` maps each
onto the 2026 axis (an exact match first, then HGNC: a previous symbol whose one approved
successor is on the axis) and fills ``gene_id`` from the T102 bridge, recording how many went
each way. A symbol neither rule places keeps its source spelling and no id, so the view leaves
it NaN rather than guessing.

The idea and the results: private ``research/ideas/ptr-feature-table.md`` § Outcome; the task:
private ``tasks/T120.md``.
"""
from __future__ import annotations

import json
import re
import subprocess
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

JURKAT_BLOCK = "jurkat_decay_blackinton2016"
HCT116_BLOCK = "hct116_actd_pasterczyk2024"
K562_BLOCK = "k562_halflife_schofield2018"
HEK293_BLOCK = "hek293_halflife_schueler2014"
BLOCKS: tuple[str, ...] = (JURKAT_BLOCK, HCT116_BLOCK, K562_BLOCK, HEK293_BLOCK)

TABLE_FILE = "stability.parquet"
AXIS_FILE = "vcc2026/gene_names.csv"
BRIDGE_FILE = "runs/probes/t102_utr_load/bridge/symbol_to_ensg.json"
HGNC_FILE = "external/hgnc-custom-uniprot-20261005/hgnc_custom_uniprot.tsv"

JURKAT_FILE = "GSE71382_ProcessedData.xlsx"
#: source column -> (table column, description). The activated-hour rates are kept because the
#: change from rest is itself a Jurkat-specific fact; the T/IP means, their SDs and the T scores
#: are not (they are the inputs the rates were fitted from).
JURKAT_COLUMNS: dict[str, tuple[str, str]] = {
    "Avg Opt DR U": ("decay_rate_jurkat_rest",
                     "Blackinton & Keene 2016 resting (unactivated) mRNA decay rate in Jurkat, per hour, negative "
                     "by the paper's convention (ln(1-.266*(net txn/total))); more negative is faster decay"),
    "StDev Opt DR U": ("decay_rate_jurkat_rest_sd", "SD of the resting decay rate over three replicates"),
    "Avg Opt DR 1": ("decay_rate_jurkat_act1h", "decay rate 1 h after PMA+PHA activation (same convention)"),
    "Avg Opt DR 2": ("decay_rate_jurkat_act2h", "decay rate 2 h after activation"),
    "Avg Opt DR 3": ("decay_rate_jurkat_act3h", "decay rate 3 h after activation"),
    "Avg Opt DR 4": ("decay_rate_jurkat_act4h", "decay rate 4 h after activation"),
    "Avg Tot RPKM": ("tot_rpkm_jurkat", "mean total-mRNA RPKM over the course (the sheet's Avg Tot RPKM)"),
    "iCLIP Target": ("hur_iclip_target_jurkat", "HuR (ELAVL1) iCLIP target in Jurkat (the sheet's yes/no as 1/0)"),
    "HuR LOD 0hr": ("hur_lod_jurkat_0h", "HuR iCLIP binding log-odds at rest"),
    "HuR LOD 4hr": ("hur_lod_jurkat_4h", "HuR iCLIP binding log-odds at 4 h of activation"),
}
JURKAT_ID_COLUMN = "EnsemblGene"
JURKAT_SYMBOL_COLUMN = "Gene Symbol"

#: HCT116: how a sample's hour and replicate are read off its file name, the expression floor
#: that decides which genes get a rate, and the pseudocount under the log.
HCT116_NAME = re.compile(r"HCT116(?:ActD)?(\d+)hr(\d+)_rsem", re.IGNORECASE)
HCT116_TPM_MIN = 1.0
HCT116_PSEUDO = 0.1

K562_FILE = "41592_2018_BFnmeth4582_MOESM5_ESM.xlsx"
K562_SHEET = "Table S2_K562"
K562_COLUMNS: dict[str, tuple[str, str]] = {
    "mean_half_life": ("halflife_k562_timelapse_mean_h",
                       "Schofield 2018 TimeLapse-seq half-life in K562, mean of two replicates, hours (the sheet "
                       "states no unit; the 4 h label implies hours)"),
    "half_life_rep1": ("halflife_k562_timelapse_rep1_h", "replicate 1 half-life, hours"),
    "half_life_rep2": ("halflife_k562_timelapse_rep2_h", "replicate 2 half-life, hours"),
    "theta_n1": ("frac_new_4h_k562_timelapse_rep1", "fraction of the transcript made during the 4 h s4U label, replicate 1"),
    "theta_n2": ("frac_new_4h_k562_timelapse_rep2", "fraction new after 4 h, replicate 2"),
}
K562_SYMBOL_COLUMN = "transcript"

#: Schueler: one file per line, each an R write.table (a quoted two-name header, then symbol, rep1, rep2).
SCHUELER_FILES: dict[str, str] = {"hek293": "GSE49831_HEK293_halflives.txt.gz",
                                  "mcf7": "GSE49831_MCF7_halflives.txt.gz"}

#: The column `build()` aligns to the gene space, per block.
PRIMARY_COLUMN: dict[str, str] = {
    JURKAT_BLOCK: "halflife_jurkat_rest_h",
    HCT116_BLOCK: "actd_slope_hct116",
    K562_BLOCK: "halflife_k562_timelapse_mean_h",
    HEK293_BLOCK: "halflife_hek293_schueler_mean_min",
}


# ------------------------------------------------------------------ gene keys --

def read_axis(root: Path) -> list[str]:
    return pd.read_csv(Path(root) / AXIS_FILE)["gene_name"].astype(str).tolist()


def read_bridge(root: Path) -> dict[str, str]:
    """Axis symbol -> unversioned Ensembl gene id (the T102 bridge)."""
    return {k: str(v).split(".")[0] for k, v in json.loads((Path(root) / BRIDGE_FILE).read_text()).items()}


def hgnc_previous(path: Path) -> dict[str, set[str]]:
    """Previous symbol -> the approved symbols that list it (HGNC's custom download)."""
    hg = pd.read_csv(path, sep="\t", dtype=str, usecols=["Approved symbol", "Previous symbols"])
    out: dict[str, set[str]] = {}
    for sym, prev in zip(hg["Approved symbol"], hg["Previous symbols"].fillna("")):
        for p in (x.strip() for x in prev.split(",")):
            if p:
                out.setdefault(p, set()).add(sym)
    return out


def resolve_symbols(symbols: pd.Series, axis: list[str], bridge: dict[str, str],
                    previous: dict[str, set[str]]) -> tuple[pd.DataFrame, dict]:
    """Each source symbol onto the 2026 axis: exact first, then a previous symbol with exactly ONE
    approved successor in HGNC, that successor on the axis (a retired symbol two genes once
    carried is left unmatched rather than guessed, as ``gene_aliases`` does). Returns (symbol,
    gene_id, how) rows and the counts."""
    on_axis = set(axis)
    sym_out, id_out, how_out = [], [], []
    counts = {"exact": 0, "hgnc_previous": 0, "none": 0}
    for s in symbols.astype(str).str.strip():
        if s in on_axis:
            target, how = s, "exact"
        else:
            succ = previous.get(s, set())
            one = next(iter(succ)) if len(succ) == 1 else None
            target, how = (one, "hgnc_previous") if one is not None and one in on_axis else (s, "none")
        counts[how] += 1
        sym_out.append(target)
        id_out.append(bridge.get(target) if how != "none" else None)
        how_out.append(how)
    return pd.DataFrame({"symbol": sym_out, "gene_id": id_out, "symbol_match": how_out}), counts


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ------------------------------------------------------------------- the fit --

def actd_fit(log_tpm: np.ndarray, hours: np.ndarray) -> dict[str, np.ndarray]:
    """Per-gene least squares of log2 TPM on hours over every sample. ``log_tpm`` is
    (genes, samples), already centred if centring is wanted. Returns the slope, its standard
    error (residual, n-2 degrees of freedom) and the fitted 4 h minus 0 h change."""
    t = np.asarray(hours, dtype=float)
    tc = t - t.mean()
    sxx = float((tc ** 2).sum())
    y = np.asarray(log_tpm, dtype=float)
    yc = y - y.mean(axis=1, keepdims=True)
    slope = (yc * tc[None, :]).sum(axis=1) / sxx
    resid = yc - slope[:, None] * tc[None, :]
    dof = max(len(t) - 2, 1)
    se = np.sqrt((resid ** 2).sum(axis=1) / dof / sxx)
    return {"slope": slope, "se": se}


class LineStabilitySource(PriorSource):
    """One measured line's stability table as a per-gene node feature.

    Constructed like every PriorSource -- ``(spec, gene_index)``; ``stability_table`` needs no
    gene index, ``build`` does. Which parser runs follows the block's name (``BLOCKS``).
    """

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
        return f"{Path(self.spec['derived']).name}/stability"

    def fetch(self, *, refresh: bool = False, progress: bool = False) -> Path:
        """Gate, record provenance, then download the block's files (`fetch_gated_files`)."""
        return fetch_gated_files(self.spec, self.root, refresh=refresh, progress=progress,
                                 registry=self.registry)

    # ----------------------------------------------------------------- lineage --

    def _code_sha(self) -> str:
        try:
            return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                  text=True, check=True, cwd=Path(__file__).parent).stdout.strip()
        except Exception:                                          # noqa: BLE001
            return "unknown"

    def _inputs(self, names: list[str], extra: list[Path] = ()) -> dict:
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

    def _record_lineage(self, *, what: str, inputs: dict, code: str, counts: dict, out_path: Path,
                        columns: dict, note: str = "", extra: dict | None = None) -> None:
        path = self.derived / "LINEAGE.json"
        lineage = json.loads(path.read_text()) if path.exists() else {"schema_version": 2, "entries": {}}
        lineage.setdefault("entries", {})
        entry = {
            "dataset": self.name, "what": what, "derives_from": str(self.dest / "PROVENANCE.json"),
            "inputs": inputs, "code": code, "code_sha": self._code_sha(),
            "built": datetime.now(UTC).isoformat(timespec="seconds"),
            "sha256": sha256_of(out_path), "bytes": out_path.stat().st_size,
            "counts": counts, "licence": self.spec.get("license"), "columns": columns, "note": note,
        }
        if extra:
            entry.update(extra)
        lineage["entries"][self.lineage_key] = entry
        path.write_text(json.dumps(lineage, indent=2) + "\n")

    # ------------------------------------------------------------------ tables --

    def stability_table(self, *, rebuild: bool = False) -> pd.DataFrame:
        """This block's per-gene table, cached at ``<derived>/stability.parquet``."""
        out_path = self.derived / TABLE_FILE
        if out_path.exists() and not rebuild:
            return pd.read_parquet(out_path)
        self.derived.mkdir(parents=True, exist_ok=True)
        builder = {JURKAT_BLOCK: self._jurkat, HCT116_BLOCK: self._hct116,
                   K562_BLOCK: self._k562, HEK293_BLOCK: self._hek293}[self.name]
        table, meta = builder()
        table = table.reset_index(drop=True)
        table.to_parquet(out_path, index=False)
        self._record_lineage(out_path=out_path, code=f"src/sidechain/priors/posttx_line_stability.py::"
                             f"LineStabilitySource._{meta.pop('builder')}", **meta)
        return table

    def _symbol_resolution(self, symbols: pd.Series) -> tuple[pd.DataFrame, dict, list[Path]]:
        axis = read_axis(self.root)
        bridge = read_bridge(self.root)
        hgnc = self.root / HGNC_FILE
        previous = hgnc_previous(hgnc)
        keys, counts = resolve_symbols(symbols, axis, bridge, previous)
        counts["with_gene_id"] = int(keys["gene_id"].notna().sum())
        return keys, counts, [self.root / AXIS_FILE, self.root / BRIDGE_FILE, hgnc]

    def _jurkat(self) -> tuple[pd.DataFrame, dict]:
        raw = pd.read_excel(self.dest / JURKAT_FILE)
        raw.columns = [str(c).strip() for c in raw.columns]
        missing = [c for c in (JURKAT_ID_COLUMN, JURKAT_SYMBOL_COLUMN, *JURKAT_COLUMNS) if c not in raw.columns]
        if missing:
            raise ValueError(f"{JURKAT_FILE}: missing column(s) {missing}; have {list(raw.columns)}")
        table = pd.DataFrame({"gene_id": raw[JURKAT_ID_COLUMN].astype(str).str.strip().str.split(".").str[0],
                              "symbol": raw[JURKAT_SYMBOL_COLUMN].astype(str).str.strip()})
        columns: dict[str, str] = {}
        for src, (name, desc) in JURKAT_COLUMNS.items():
            if name == "hur_iclip_target_jurkat":
                table[name] = raw[src].astype(str).str.strip().str.lower().map({"yes": 1.0, "no": 0.0})
            else:
                table[name] = _num(raw[src]).astype(float)
            columns[name] = desc
        rate = table["decay_rate_jurkat_rest"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            hl = np.where(rate < 0, np.log(2.0) / -rate, np.nan)
        table["halflife_jurkat_rest_h"] = hl
        columns["halflife_jurkat_rest_h"] = ("ln(2) / -decay_rate_jurkat_rest, hours; NaN where the resting rate is "
                                             "missing or not negative (derived here, not in the sheet)")
        table = table[table["gene_id"].str.startswith("ENSG")]
        dups = int(table["gene_id"].duplicated().sum())
        table = table.drop_duplicates("gene_id")
        rest = table["decay_rate_jurkat_rest"].dropna()
        meta = {
            "builder": "jurkat",
            "what": "Blackinton & Keene 2016 Jurkat mRNA decay rates (4sU course, three replicates), the resting "
                    "rate first, with the activated-hour rates, a derived resting half-life and the HuR iCLIP "
                    "columns; Ensembl-keyed, raw as published (T120, phase 3)",
            "inputs": self._inputs([JURKAT_FILE]),
            "counts": {"rows_in": int(len(raw)), "genes": int(len(table)), "duplicate_gene_ids_dropped": dups,
                       "genes_with_resting_rate": int(rest.notna().sum()),
                       "resting_rate_median": float(rest.median()), "resting_rate_min": float(rest.min()),
                       "resting_rate_max": float(rest.max()),
                       "genes_with_positive_resting_rate": int((rest > 0).sum()),
                       "halflife_rest_h_median": float(table["halflife_jurkat_rest_h"].median()),
                       "hur_iclip_targets": int((table["hur_iclip_target_jurkat"] == 1).sum())},
            "columns": columns,
            "note": "the resting column is the steady-state baseline; the activated columns are a PMA+PHA "
                    "response and read as Jurkat biology, not as a resting rate (private "
                    "research/reading/posttx-verify-29)",
        }
        return table, meta

    def _hct116_samples(self) -> list[tuple[str, int, int]]:
        """(file name, hour, replicate) for every HCT116 file the block lists."""
        out = []
        for f in self.spec.get("files", []):
            m = HCT116_NAME.search(f["name"])
            if m:
                out.append((f["name"], int(m.group(1)), int(m.group(2))))
        if len(out) < 6 or len({h for _, h, _ in out}) < 3:
            raise ValueError(f"{self.name}: the block lists {len(out)} HCT116 files over "
                             f"{sorted({h for _, h, _ in out})} hours; a chase needs three time points")
        return sorted(out, key=lambda x: (x[1], x[2]))

    def _hct116(self) -> tuple[pd.DataFrame, dict]:
        samples = self._hct116_samples()
        tpm: pd.DataFrame | None = None
        for name, hour, rep in samples:
            d = pd.read_csv(self.dest / name, sep="\t", usecols=["gene_id", "TPM"])
            col = f"h{hour}_r{rep}"
            part = pd.DataFrame({"raw_id": d["gene_id"].astype(str), col: _num(d["TPM"]).astype(float)})
            tpm = part if tpm is None else tpm.merge(part, on="raw_id", how="outer")
        assert tpm is not None
        ids = tpm["raw_id"].str.split("_", n=1)
        tpm["gene_id"] = ids.str[0].str.split(".").str[0]
        tpm["symbol"] = ids.str[1].fillna("")
        tpm = tpm[tpm["gene_id"].str.startswith("ENSG")]
        dups = int(tpm["gene_id"].duplicated().sum())
        tpm = tpm.drop_duplicates("gene_id").reset_index(drop=True)
        cols = [f"h{h}_r{r}" for _, h, r in samples]
        hours = np.array([h for _, h, _ in samples], dtype=float)
        x = tpm[cols].to_numpy(dtype=float)
        t0 = hours == 0
        tpm0 = x[:, t0].mean(axis=1)
        expressed = tpm0 >= HCT116_TPM_MIN
        y = np.log2(x + HCT116_PSEUDO)
        offsets = np.nanmedian(y[expressed], axis=0)              # each sample's median expressed gene
        yc = y - offsets[None, :]
        fit_c = actd_fit(yc, hours)
        fit_raw = actd_fit(y, hours)
        t4 = hours == hours.max()
        lfc4 = yc[:, t4].mean(axis=1) - yc[:, t0].mean(axis=1)
        nan = np.full(len(tpm), np.nan)
        table = pd.DataFrame({
            "gene_id": tpm["gene_id"], "symbol": tpm["symbol"],
            "actd_slope_hct116": np.where(expressed, fit_c["slope"], nan),
            "actd_slope_hct116_se": np.where(expressed, fit_c["se"], nan),
            "actd_slope_raw_hct116": np.where(expressed, fit_raw["slope"], nan),
            "actd_lfc4h_hct116": np.where(expressed, lfc4, nan),
            "tpm0_hct116": tpm0,
        })
        # replicate agreement of the centred slope: fitted on one replicate series against the other two
        def _rep_slope(reps: set[int]) -> np.ndarray:
            keep = np.array([r in reps for _, _, r in samples])
            return actd_fit(yc[:, keep], hours[keep])["slope"]
        s1 = _rep_slope({1})
        s23 = _rep_slope({2, 3})
        ok = expressed & np.isfinite(s1) & np.isfinite(s23)
        rep_rho = float(pd.Series(s1[ok]).corr(pd.Series(s23[ok]), method="spearman")) if ok.sum() > 50 else float("nan")
        sl = table["actd_slope_hct116"].dropna()
        columns = {
            "actd_slope_hct116": (f"least-squares slope of log2(TPM + {HCT116_PSEUDO}) on hours of actinomycin D over "
                                  f"the nine HCT116 samples (0, 2, 4 h x 3), each sample centred on its median gene "
                                  f"with mean 0 h TPM >= {HCT116_TPM_MIN}: a decay rate RELATIVE to the median gene, "
                                  f"log2 per hour; negative decays faster than the median gene, positive slower; NaN "
                                  f"below the expression floor (fitted here; the paper publishes no rate)"),
            "actd_slope_hct116_se": "standard error of that slope from the residuals (7 degrees of freedom)",
            "actd_slope_raw_hct116": "the same slope without the per-sample centring (log2 TPM per hour): the centred "
                                     "slope plus one constant, so the two rank genes identically",
            "actd_lfc4h_hct116": "centred mean log2 TPM at 4 h minus at 0 h (the two-point form of the slope)",
            "tpm0_hct116": "mean TPM over the three 0 h samples, every gene",
        }
        meta = {
            "builder": "hct116",
            "what": "a per-gene mRNA decay rate in HCT116 fitted from Pasterczyk 2024's actinomycin D chase (GSE234877, "
                    "nine RSEM TPM tables), relative to the median expressed gene; Ensembl-keyed (T120, phase 3)",
            "inputs": self._inputs([n for n, _, _ in samples]),
            "counts": {"samples": [{"file": n, "hour": h, "replicate": r} for n, h, r in samples],
                       "rows_in": int(len(ids)), "genes": int(len(table)), "duplicate_gene_ids_dropped": dups,
                       "tpm_floor_at_0h": HCT116_TPM_MIN, "pseudocount": HCT116_PSEUDO,
                       "genes_expressed": int(expressed.sum()),
                       "sample_median_log2_tpm_expressed": {c: float(o) for c, o in zip(cols, offsets)},
                       "slope_median": float(sl.median()), "slope_sd": float(sl.std()),
                       "slope_q05": float(sl.quantile(0.05)), "slope_q95": float(sl.quantile(0.95)),
                       "replicate_agreement_spearman_rep1_vs_rep23": rep_rho,
                       "raw_slope_median": float(table["actd_slope_raw_hct116"].median())},
            "columns": columns,
            "note": "two points after the drug (2 h, 4 h), so the fit has little leverage on fast decay; the "
                    "centring subtracts one offset per sample, so it shifts every gene's slope by the same constant "
                    "(the slope of the offsets on hours, sample_median_log2_tpm_expressed) and the centred and raw "
                    "columns rank genes identically -- the centred level is the one that reads as a rate relative to "
                    "the median gene, the raw one carries the chase's re-normalisation of a share measure (TPM)",
        }
        return table, meta

    def _k562(self) -> tuple[pd.DataFrame, dict]:
        book = pd.ExcelFile(self.dest / K562_FILE)
        if K562_SHEET not in book.sheet_names:
            raise ValueError(f"{K562_FILE}: no sheet {K562_SHEET!r}; have {book.sheet_names}")
        raw = book.parse(K562_SHEET)
        raw.columns = [str(c).strip() for c in raw.columns]
        missing = [c for c in (K562_SYMBOL_COLUMN, *K562_COLUMNS) if c not in raw.columns]
        if missing:
            raise ValueError(f"{K562_FILE}: sheet {K562_SHEET!r} missing column(s) {missing}; have {list(raw.columns)}")
        keys, how, extra = self._symbol_resolution(raw[K562_SYMBOL_COLUMN])
        table = keys.copy()
        table.insert(0, "symbol_source", raw[K562_SYMBOL_COLUMN].astype(str).str.strip().to_numpy())
        columns: dict[str, str] = {}
        for src, (name, desc) in K562_COLUMNS.items():
            table[name] = _num(raw[src]).astype(float)
            columns[name] = desc
        dups = int(table["symbol"].duplicated().sum())
        table = table.drop_duplicates("symbol")
        hl = table["halflife_k562_timelapse_mean_h"].dropna()
        r1, r2 = table["halflife_k562_timelapse_rep1_h"], table["halflife_k562_timelapse_rep2_h"]
        meta = {
            "builder": "k562",
            "what": "Schofield 2018 TimeLapse-seq half-lives in K562 (Supplementary Table 2, sheet Table S2_K562), "
                    "symbol-keyed and bridged onto the 2026 axis (T120, phase 3)",
            "inputs": self._inputs([K562_FILE], extra),
            "counts": {"rows_in": int(len(raw)), "genes": int(len(table)), "duplicate_symbols_dropped": dups,
                       "symbol_resolution": how, "halflife_mean_h_median": float(hl.median()),
                       "halflife_mean_h_min": float(hl.min()), "halflife_mean_h_max": float(hl.max()),
                       "replicate_spearman": float(r1.corr(r2, method="spearman"))},
            "columns": columns,
            "note": "two of Agarwal & Kelley's eight K562 samples (Simon_4sU_K562_1/2) are this measurement, so "
                    "the consensus's K562 mean is not independent of this column; symbols resolved by exact match "
                    "then HGNC previous symbols (symbol_match says which)",
        }
        return table, meta

    def _read_schueler(self, path: Path, line: str) -> pd.DataFrame:
        """An R write.table: a quoted two-name header (the row names have no column), then rows of
        symbol, rep1, rep2. The header is read on its own and checked, because a whitespace split
        cannot be trusted with the spaces inside its quoted names."""
        import gzip
        with gzip.open(path, "rt") as fh:
            header = fh.readline()
        names = re.findall(r'"([^"]*)"', header)
        want = (f"{line.upper()}_RNA half-life (min) rep1", f"{line.upper()}_RNA half-life (min) rep2")
        if tuple(names) != want:
            raise ValueError(f"{path.name}: header names {names} are not {list(want)}; the file's shape has changed")
        d = pd.read_csv(path, sep=r"\s+", header=None, skiprows=1, names=["symbol", "rep1", "rep2"],
                        quotechar='"', engine="python")
        d["symbol"] = d["symbol"].astype(str).str.strip().str.strip('"')
        d = d.set_index("symbol")
        return d.apply(_num)

    def _hek293(self) -> tuple[pd.DataFrame, dict]:
        parts: dict[str, pd.DataFrame] = {line: self._read_schueler(self.dest / f, line) for line, f in SCHUELER_FILES.items()}
        symbols = pd.Series(sorted(set().union(*(set(d.index) for d in parts.values()))))
        keys, how, extra = self._symbol_resolution(symbols)
        table = keys.copy()
        table.insert(0, "symbol_source", symbols.to_numpy())
        columns: dict[str, str] = {}
        counts: dict = {"symbol_resolution": how, "lines": {}}
        for line, d in parts.items():
            d = d.reindex(symbols.to_numpy())
            for rep in ("rep1", "rep2"):
                name = f"halflife_{line}_schueler_{rep}_min"
                table[name] = d[rep].to_numpy(dtype=float)
                columns[name] = (f"Schueler 2014 4sU mRNA half-life in {line.upper()}, replicate {rep[-1]}, minutes "
                                 f"(GSE49831; computed as in Schwanhaeusser 2011)")
            mean = f"halflife_{line}_schueler_mean_min"
            table[mean] = table[[f"halflife_{line}_schueler_rep1_min", f"halflife_{line}_schueler_rep2_min"]].mean(axis=1)
            columns[mean] = f"mean of the two {line.upper()} replicates, minutes (derived here)"
            counts["lines"][line] = {"genes": int(len(d.dropna(how="all"))),
                                     "median_min": float(d.mean(axis=1).median()),
                                     "replicate_spearman": float(d["rep1"].corr(d["rep2"], method="spearman"))}
        dups = int(table["symbol"].duplicated().sum())
        table = table.drop_duplicates("symbol")
        counts.update({"rows_in": {line: int(len(d)) for line, d in parts.items()}, "genes": int(len(table)),
                       "duplicate_symbols_dropped": dups})
        meta = {
            "builder": "hek293",
            "what": "Schueler 2014 4sU half-lives in HEK293 and MCF7 (GSE49831, two replicates each, minutes), "
                    "symbol-keyed and bridged onto the 2026 axis (T120, phase 3)",
            "inputs": self._inputs(list(SCHUELER_FILES.values()), extra),
            "counts": counts, "columns": columns,
            "note": "inside Agarwal & Kelley's consensus already (Dieterich_4sU_HEK293_1/2, two of fourteen HEK293 "
                    "samples; Dieterich_4sU_MCF7_1/2, the whole MCF7 group), so neither per-line mean there is "
                    "independent of these columns; HEK293, not HEK293T",
        }
        return table, meta

    # ------------------------------------------------------------------- build --

    def build(self) -> PriorArtifact:
        """One column -- the block's primary stability measure -- aligned to the gene space, with a
        ``has_value`` mask (a gene with no row is 0.0, which is not a measurement)."""
        table = self.stability_table()
        col = PRIMARY_COLUMN[self.name]
        n = len(self.gene_index)
        features = np.zeros((n, 1), dtype=np.float32)
        has_value = np.zeros((n, 1), dtype=bool)
        ids = table["gene_id"].astype(str).to_numpy()
        values = _num(table[col]).to_numpy(dtype=float)
        for gene, value in zip(ids, values):
            pos = self.gene_index.get(gene)
            if pos is None or not np.isfinite(value):
                continue
            features[pos, 0] = np.float32(value)
            has_value[pos, 0] = True
        return PriorArtifact(kind="node_feature", relation=self.relation, layer=self.layer, features=features,
                             meta={"feature_names": [col], "has_value": has_value,
                                   "candidates": int(len(ids)), "placed": int(has_value.sum())})


def line_sources(root: Path = DATA_ROOT, registry: str | Path = REGISTRY) -> list[LineStabilitySource]:
    """The four loaders, from the registry, in the plan's order."""
    return [LineStabilitySource(spec_from_registry(b, registry), {}, root=root, registry=registry) for b in BLOCKS]
