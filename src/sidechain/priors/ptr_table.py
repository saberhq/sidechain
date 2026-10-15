"""One wide per-gene table of post-transcriptional (PTR) features on the 2026 axis (T120).

A VIEW, never a merge at source: every source keeps its own ``derived/<source>/`` parquet,
LINEAGE.json and licence flags, and this module joins them on the gene axis and writes, beside
the wide table, a column dictionary that says for every column where it came from, whether it is
universal (the same in every cell line) or line-specific (one named line's measurement), what it
is, which licence it carries and how many genes it covers. A reader that wants only the
columns a given licence allows filters the dictionary; a source with unstated terms (PolyASite,
``UNSTATED``, both encumbrance flags) sits in the table with its flags intact and cannot leak
into a column that says otherwise. Rebuilding the view is minutes.

**The join.** The axis is the 18,533 symbols of ``vcc2026/gene_names.csv``; each symbol meets a
source's row through its Ensembl id first (the T102 bridge, ``symbol_to_ensg``) and through the
source's own symbol column when the id finds nothing -- our axes spell 2024 symbols, TargetScan
2018's, Agarwal 2022's. How many genes each source matched by id, by symbol and not at all is
recorded in LINEAGE.json.

**Scope.** Each column is ``universal`` or ``line:<name>``, where the name is the measured line
as the source names it, lowercased and brought to one spelling per line: ``hek293`` is Agarwal &
Kelley's HEK293 group (their samples do not separate HEK293 from HEK293T), ``hek293t`` is Wu's
293T, ``rpe1`` is hTERT RPE-1, ``h1`` is H1 hESC. A per-line column is NaN wherever the line has
no measurement for the gene; a coverage column (``_n``, ``_window_tpm_``) sits beside each family
where the source provides one.

**Phase 1** (``plans/2026-10-15-t102-ptr-feature-table.md``, private) is what ``SOURCES`` lists:
TargetScan's widened 3'UTR table, the half-life consensus with its per-line means, the codon
score with its per-line CSC columns, Wu's per-line decay rates, and PolyASite 2.0's layout. A new
source is one more ``SourceSpec``; the dictionary, the lineage and the coverage counts follow.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sidechain.priors.posttx_mirna import (
    DATA_ROOT,
    REGISTRY,
    sha256_of,
    spec_from_registry,
)

#: Where the view lands under the data root.
DERIVED = "derived/ptr-feature-table"
TABLE_FILE = "ptr_features.parquet"
DICT_FILE = "ptr_columns.parquet"
AXIS_FILE = "vcc2026/gene_names.csv"
BRIDGE_FILE = "runs/probes/t102_utr_load/bridge/symbol_to_ensg.json"

#: One spelling per measured line. Keys are the tokens the sources use (lowercased); a token not
#: listed is kept as it is.
LINE_SPELLING = {
    "293t": "hek293t", "hek293t": "hek293t", "hek293": "hek293",
    "hela": "hela", "k562": "k562", "hepg2": "hepg2", "rpe": "rpe1", "rpe1": "rpe1",
    "h1esc": "h1", "h1": "h1", "mcf7": "mcf7", "a549": "a549", "lcl": "lcl",
    "hct116": "hct116", "jurkat": "jurkat", "cal33": "cal33",
}


@dataclass(frozen=True)
class SourceSpec:
    """One per-source table and how its columns enter the view.

    ``scope`` maps a column-name regex to ``"universal"`` or ``"line:{0}"`` (group 1 of the regex
    is the line token); the first matching pattern wins and an unmatched numeric column is
    universal. ``drop`` names columns that never enter (ids, free text). ``describe`` overrides
    or adds a description for a column; the rest come from the source's LINEAGE entry.
    """
    key: str                    # the prefix in the wide table
    block: str                  # the registry block (licence, PROVENANCE)
    derived: str                # the derived directory under the data root
    table: str                  # the parquet in it
    lineage_key: str            # its entry in that directory's LINEAGE.json
    id_col: str = "gene_id"
    symbol_col: str | None = "symbol"
    scope: tuple[tuple[str, str], ...] = ()
    drop: tuple[str, ...] = ()
    describe: dict[str, str] = field(default_factory=dict)
    note: str = ""


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        key="ts", block="targetscan", derived="derived/targetscan-vert_80", table="utr_features.parquet",
        lineage_key="targetscan-vert_80/utr_features", drop=("transcript_id",),
        describe={"utr_len": "3'UTR length of the representative transcript, nt (hg19 GFF exon widths)",
                  "seq_len": "ungapped 3'UTR sequence length, nt",
                  "n_noncons_sites_all": "nonconserved sites of conserved families, all, from sites.parquet"},
        note="universal 3'UTR grammar: TargetScan 8.0 load, composition, consensus RBP motifs, "
             "NAR 2016 cooperation and competition counts (one representative transcript per gene)",
    ),
    SourceSpec(
        key="hl", block="halflife_consensus", derived="derived/halflife-agarwal-kelley-2022",
        table="halflife.parquet", lineage_key="halflife-agarwal-kelley-2022/halflife",
        scope=(("^halflife_pc1$", "universal"), (r"^halflife_([a-z0-9]+)_(?:mean|n)$", "line:{0}")),
        note="measured mRNA half-life (Agarwal & Kelley 2022 Table S2): the PC1 consensus is universal, "
             "each line's sample mean and per-gene sample count are line-specific",
    ),
    SourceSpec(
        key="cs", block="codon_optimality_wu2019", derived="derived/codon-optimality-wu2019",
        table="codon_score.parquet", lineage_key="codon-optimality-wu2019/codon_score", drop=("transcript_id",),
        scope=(("^csc_endo_mean$", "universal"), (r"^csc_([0-9A-Za-z]+)_", "line:{0}")),
        describe={**{f"csc_{c}": f"mean CSC of the ORF's sense codons under Wu's {c} column"
                     for c in ("293T_endo", "HeLa_endo", "RPE_endo", "293T_ORFome", "K562_ORFome", "K562_SLAM")}},
        note="codon optimality (Wu 2019 CSC x TargetScan's representative ORF): csc_endo_mean, ORF length, "
             "GC3 and the 64 codon counts are universal; a csc_<line>_<method> column is that line's CSC "
             "table applied to the universal ORF, so it is line-specific through the table, not the gene",
    ),
    SourceSpec(
        key="dr", block="codon_optimality_wu2019", derived="derived/codon-optimality-wu2019",
        table="decay_rates.parquet", lineage_key="codon-optimality-wu2019/decay_rates", symbol_col=None,
        scope=((r"^decay_rate_([0-9A-Za-z]+)_", "line:{0}"),),
        describe={"decay_rate_293T_endogenous": "Wu 2019 per-gene decay rate, 293T, actinomycin D (raw sign)",
                  "decay_rate_HeLa_endogenous": "Wu 2019 per-gene decay rate, HeLa, actinomycin D (raw sign)",
                  "decay_rate_RPE_endogenous": "Wu 2019 per-gene decay rate, RPE, actinomycin D (raw sign)",
                  "decay_rate_k562_SLAM_seq": "Wu 2019 per-gene decay rate, K562, SLAM-seq (raw sign; "
                                              "the median's sign differs from the ActD sheets)"},
        note="measured per-line decay rates, raw as published; standardise per sheet before comparing lines",
    ),
    SourceSpec(
        key="pa", block="polyasite_2", derived="derived/polyasite-2.0", table="pas_gene_features.parquet",
        lineage_key="polyasite-2.0/pas_gene_features", drop=("transcript_id", "pa_tx_source"),
        scope=((r"^pa_(?:prox_usage|window_tpm)_(hela|hek293|k562)$", "line:{0}"),),
        note="polyadenylation layout from PolyASite 2.0 on GENCODE v47 (UNSTATED terms, streamed; both "
             "encumbrance flags): universal except the per-line usage columns, which are written and used nowhere",
    ),
)


def _code_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:                                          # noqa: BLE001
        return "unknown"


def line_token(raw: str) -> str:
    return LINE_SPELLING.get(raw.lower(), raw.lower())


def column_scope(spec: SourceSpec, column: str) -> str:
    for pattern, scope in spec.scope:
        m = re.match(pattern, column)
        if m:
            return scope.format(*[line_token(g) for g in m.groups()]) if "{" in scope else scope
    return "universal"


def wide_name(spec: SourceSpec, column: str) -> str:
    return column if column.startswith(spec.key + "_") else f"{spec.key}_{column}"


def read_axis(root: Path) -> list[str]:
    """The 2026 gene axis, from the challenge's own gene_names.csv (it has a header)."""
    return pd.read_csv(root / AXIS_FILE)["gene_name"].astype(str).tolist()


def read_bridge(root: Path) -> dict[str, str]:
    return {k: v.split(".")[0] for k, v in json.loads((root / BRIDGE_FILE).read_text()).items()}


def source_licence(spec: SourceSpec, root: Path, registry: str | Path) -> dict:
    """The block's licence and flags, from the external PROVENANCE.json the gate wrote; the
    registry's declared licence when the provenance is not on this machine."""
    block = spec_from_registry(spec.block, registry)
    prov = root / block["dest"] / "PROVENANCE.json"
    if prov.exists():
        p = json.loads(prov.read_text())
        return {"licence": p.get("record", {}).get("license", block.get("license")),
                "noncommercial": bool(p.get("license_flags", {}).get("noncommercial", False)),
                "redistribution_encumbered": bool(p.get("license_flags", {}).get("redistribution_encumbered", False)),
                "provenance": str(prov), "route": p.get("route")}
    from sidechain.ingest.provenance import NONCOMMERCIAL, REDISTRIBUTION_ENCUMBERED
    lic = block.get("license")
    return {"licence": lic, "noncommercial": lic in NONCOMMERCIAL,
            "redistribution_encumbered": lic in REDISTRIBUTION_ENCUMBERED, "provenance": None,
            "route": block.get("route", "download")}


def join_source(spec: SourceSpec, table: pd.DataFrame, axis: list[str], bridge: dict[str, str]
                ) -> tuple[pd.DataFrame, dict]:
    """The source's numeric columns on the axis: Ensembl id first, then the source's symbol."""
    by_id = table.drop_duplicates(spec.id_col).set_index(spec.id_col)
    by_sym = (table.drop_duplicates(spec.symbol_col).set_index(spec.symbol_col)
              if spec.symbol_col and spec.symbol_col in table.columns else None)
    cols = [c for c in table.columns
            if c not in {spec.id_col, spec.symbol_col, *spec.drop}
            and (pd.api.types.is_numeric_dtype(table[c]) or pd.api.types.is_bool_dtype(table[c]))]
    out = np.full((len(axis), len(cols)), np.nan)
    how = {"id": 0, "symbol": 0, "none": 0}
    id_rows = by_id[cols].apply(pd.to_numeric, errors="coerce")
    sym_rows = by_sym[cols].apply(pd.to_numeric, errors="coerce") if by_sym is not None else None
    for i, g in enumerate(axis):
        e = bridge.get(g)
        if e is not None and e in id_rows.index:
            out[i] = id_rows.loc[e].to_numpy(dtype=float)
            how["id"] += 1
        elif sym_rows is not None and g in sym_rows.index:
            out[i] = sym_rows.loc[g].to_numpy(dtype=float)
            how["symbol"] += 1
        else:
            how["none"] += 1
    return pd.DataFrame(out, columns=[wide_name(spec, c) for c in cols]), {"matched": how, "columns": cols}


def build_ptr_table(root: Path = DATA_ROOT, *, sources: tuple[SourceSpec, ...] = SOURCES,
                    registry: str | Path = REGISTRY, axis: list[str] | None = None,
                    bridge: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The wide table, the column dictionary and the lineage entry, for the axis given or read."""
    root = Path(root).expanduser()
    axis = read_axis(root) if axis is None else list(axis)
    bridge = read_bridge(root) if bridge is None else bridge
    wide = pd.DataFrame({"symbol": axis, "gene_id": [bridge.get(g) for g in axis]})
    rows: list[dict] = []
    inputs: dict = {}
    per_source: dict = {}
    for spec in sources:
        path = root / spec.derived / spec.table
        table = pd.read_parquet(path)
        lineage_path = root / spec.derived / "LINEAGE.json"
        entry = (json.loads(lineage_path.read_text()).get("entries", {}).get(spec.lineage_key, {})
                 if lineage_path.exists() else {})
        descriptions = {**entry.get("columns", {}), **spec.describe}
        lic = source_licence(spec, root, registry)
        joined, info = join_source(spec, table, axis, bridge)
        if set(joined.columns) & set(wide.columns):
            raise ValueError(f"{spec.key}: duplicate wide columns {sorted(set(joined.columns) & set(wide.columns))}")
        wide = pd.concat([wide, joined], axis=1)
        for raw, name in zip(info["columns"], joined.columns):
            scope = column_scope(spec, raw)
            rows.append({"column": name, "source": spec.block, "key": spec.key, "table": f"{spec.derived}/{spec.table}",
                         "source_column": raw, "scope": scope, "universal": scope == "universal",
                         "line": scope.split(":", 1)[1] if scope.startswith("line:") else "",
                         "description": descriptions.get(raw, ""), "licence": lic["licence"],
                         "noncommercial": lic["noncommercial"],
                         "redistribution_encumbered": lic["redistribution_encumbered"],
                         "genes_covered": int(joined[name].notna().sum())})
        inputs[f"{spec.derived}/{spec.table}"] = {"sha256": sha256_of(path), "bytes": path.stat().st_size,
                                                   "lineage_entry": spec.lineage_key,
                                                   "source_sha256": entry.get("sha256"),
                                                   "source_built": entry.get("built"), **lic}
        per_source[spec.key] = {"block": spec.block, "rows_in": len(table), "columns": len(info["columns"]),
                                "matched": info["matched"], "note": spec.note}
    dictionary = pd.DataFrame(rows)
    lineage_entry = {
        "dataset": "ptr-feature-table", "what": "one row per 2026-axis gene, every built post-transcriptional "
                                               "feature as a column; a view over the per-source tables, with a "
                                               "column dictionary beside it (T120, phase 1)",
        "inputs": inputs, "axis": {"file": AXIS_FILE, "genes": len(axis),
                                   "bridged_to_ensembl": int(wide["gene_id"].notna().sum())},
        "bridge": {"file": BRIDGE_FILE, "sha256": sha256_of(root / BRIDGE_FILE) if (root / BRIDGE_FILE).exists() else None},
        "code": "src/sidechain/priors/ptr_table.py::build_ptr_table", "code_sha": _code_sha(),
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": per_source,
        "counts": {"genes": len(axis), "columns": len(dictionary),
                   "columns_universal": int(dictionary["universal"].sum()) if len(dictionary) else 0,
                   "columns_line": int((~dictionary["universal"]).sum()) if len(dictionary) else 0,
                   "lines": sorted(dictionary.loc[dictionary["line"] != "", "line"].unique().tolist()) if len(dictionary) else [],
                   "genes_with_any_value": int(wide.drop(columns=["symbol", "gene_id"]).notna().any(axis=1).sum()),
                   "genes_with_any_universal": int(wide[[r["column"] for r in rows if r["universal"]]].notna().any(axis=1).sum()) if rows else 0},
        "licence_rule": "every column carries its source's licence and flags; a column from an UNSTATED or "
                        "NonCommercial source keeps those flags here, and anything published from this table "
                        "must be filtered on them first",
    }
    return wide, dictionary, lineage_entry


def write_ptr_table(root: Path = DATA_ROOT, **kw) -> Path:
    """Build and write the view, the dictionary (parquet and json) and the LINEAGE entry."""
    root = Path(root).expanduser()
    out = root / DERIVED
    out.mkdir(parents=True, exist_ok=True)
    wide, dictionary, entry = build_ptr_table(root, **kw)
    wide.to_parquet(out / TABLE_FILE, index=False)
    dictionary.to_parquet(out / DICT_FILE, index=False)
    (out / DICT_FILE.replace(".parquet", ".json")).write_text(
        json.dumps(dictionary.to_dict(orient="records"), indent=1) + "\n")
    entry = {**entry, "sha256": sha256_of(out / TABLE_FILE), "bytes": (out / TABLE_FILE).stat().st_size,
             "dictionary_sha256": sha256_of(out / DICT_FILE)}
    path = out / "LINEAGE.json"
    lineage = json.loads(path.read_text()) if path.exists() else {"schema_version": 2, "entries": {}}
    lineage.setdefault("entries", {})["ptr-feature-table/ptr_features"] = entry
    path.write_text(json.dumps(lineage, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Build the wide PTR feature table (T120)")
    ap.add_argument("--root", type=Path, default=DATA_ROOT)
    args = ap.parse_args(argv)
    out = write_ptr_table(args.root)
    entry = json.loads((out / "LINEAGE.json").read_text())["entries"]["ptr-feature-table/ptr_features"]
    print(json.dumps({"counts": entry["counts"], "sources": {k: v["matched"] for k, v in entry["sources"].items()}}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
