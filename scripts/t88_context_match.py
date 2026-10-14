#!/usr/bin/env python
"""T88 -- re-run the named-line context match with new rows in the panel.

    .venv/bin/python scripts/t88_context_match.py \
        --panel ~/data/sidechain/derived/scbasecount/named_line_panel.npz \
        --out ~/data/sidechain/runs/context_match/t88_named_lines_20261013/v1_reproduction
    .venv/bin/python scripts/t88_context_match.py \
        --panel ~/data/sidechain/derived/scbasecount/named_line_panel.npz \
        --panel ~/data/sidechain/derived/scbasecount/named_line_panel_t88_additions.npz \
        --merged-out ~/data/sidechain/derived/scbasecount/named_line_panel_v2.npz \
        --out ~/data/sidechain/runs/context_match/t88_named_lines_20261013/v2

The instrument is the one `private/research/ideas/scbasecount-context-matching.md` validated on
2026-08-29 (T34): a context's control cells, summarised as the mean of per-cell CPM over the 18,533
challenge genes, log1p, cosine against every named cell line in a panel streamed from scBaseCount
with every experiment of a screen we pool excluded. The reading rule came from two disjoint panels:
a top-1-to-top-2 gap of about 0.02-0.05 means the line is in the panel; about 0.00x means it is not.
The 2026-08-29 script was not kept, so this one re-derives its numbers on the 25-row panel first
(the `v1_reproduction` run) before anything new is read off it.

What it reads, per query (contexts A/B/C; our own control pseudobulks for K562, Jurkat, HepG2, RPE1,
H1, HCT116 and HEK293T as positive controls):

* the ranking of every panel row by cosine, **plain** (both sides restricted to the genes the query
  measures, no rescaling) and **renormalised** (both sides rescaled to a million on that shared
  support before log1p -- the form that removes the gene-support asymmetry of our ~8,000-gene
  scPerturb files; it changed no ranking in T34);
* the top-1-to-top-2 gap, and the gap from the top row to the best row of a DIFFERENT line (two
  spellings of one line are one line, so a HeLa / "HeLa cells" pair is not a gap);
* the depth confound: Spearman between a panel row's mean UMI per cell and its cosine to the query,
  and a **depth-robust** ranking with the 2,000 genes whose level tracks row depth most (|Spearman|
  across panel rows) left out -- T34's own check, which widened A's gap rather than closing it;
* row ablations for the contexts (`--ablate-rows`): the ranking with one named row removed, so a
  call that rests on a single row shows it (a line present under two spellings should survive
  losing either one);
* the held-out-spelling control: every row with a same-line sibling, used as a query against the
  other rows -- does its sibling rank first? (free, uses no data of ours, cannot be a self-match);
* lineage-marker ranks for the new rows, and a small gene table (CPM) for the contexts beside the
  candidate rows, so a reader can see what the cosine saw.

Writes `results.json`, `rankings.csv`, `spelling_control.csv`, `markers.csv`, `gene_table.csv` under
`--out`, and with `--merged-out` the concatenated panel as a `ContextProfiles` npz with a lineage
file naming its parts (the rows are kept as streamed; nothing is re-normalised in the artifact).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

try:  # anndata >= 0.11
    from anndata.io import read_elem
except ImportError:  # pragma: no cover - older anndata
    from anndata.experimental import read_elem

from sidechain.data.stream_observational import ContextProfiles
from sidechain.data.stream_pseudobulk import PseudobulkSums

DATA = Path("~/data/sidechain").expanduser()

#: Our own control pseudobulks: name -> (file, control label, the panel line it must find).
LINE_QUERIES: dict[str, tuple[Path, str, str]] = {
    "k562": (DATA / "cache/vcc2026/k562_gwps_union_pseudobulk.npz", "non-targeting", "K562"),
    "jurkat": (DATA / "cache/vcc2026/jurkat_all_pseudobulk.npz", "control", "Jurkat"),
    "hepg2": (DATA / "cache/vcc2026/hepg2_all_pseudobulk.npz", "control", "HepG2"),
    "rpe1": (DATA / "cache/vcc2026/rpe1_all_pseudobulk.npz", "control", "RPE-1"),
    "h1": (DATA / "cache/vcc2026/h1_pseudobulk.npz", "non-targeting", "H1"),
    "hct116": (DATA / "cache/vcc2026/foldsub/hct116_full_panel300.npz", "Non-Targeting", "HCT116"),
    "hek293t": (DATA / "cache/vcc2026/foldsub/hek293t_full_panel300.npz", "Non-Targeting", "HEK293"),
}

#: Which panel rows are spellings or derivatives of ONE line. Rows not listed are their own line.
#: HEK293T, HEK293FT and HEK293 are kept as one family on purpose: the question the panel answers
#: is lineage, and the three are derivatives of one embryonic-kidney line.
LINE_OF: dict[str, str] = {
    "MDA-MB-231": "MDA-MB-231", "MDAMB231": "MDA-MB-231",
    "Kuramochi": "Kuramochi", "KURAMOCHI": "Kuramochi",
    "T-47D": "T-47D", "T47D": "T-47D",
    "MCF-7": "MCF-7", "MCF7": "MCF-7",
    "HeLa": "HeLa", "HeLa cells": "HeLa", "HeLa S3": "HeLa",
    "Fadu cell line": "FaDu", "TPF-resistant Fadu": "FaDu", "TPF-resistant Fadu cell line": "FaDu",
    "hTERT RPE-1": "RPE-1", "RPE-1": "RPE-1", "RPE1-hTert": "RPE-1",
    "HEK293T": "HEK293", "HEK293FT": "HEK293", "HEK293": "HEK293",
}

#: Textbook lineage markers for the rows T88 adds (the T34 panel's own marker check covered its 25).
MARKERS: dict[str, list[str]] = {
    "squamous (FaDu, KYSE30, cSCC4)": ["KRT5", "KRT14", "KRT6A", "TP63", "SFN"],
    "intestinal (HCT116, HT-29, Caco-2)": ["CDX2", "VIL1", "CDH17", "LGALS4"],
    "prostate (LNCaP)": ["KLK3", "AR", "NKX3-1"],
    "melanoma (A375)": ["PMEL", "MLANA", "MITF", "SOX10"],
    "ovarian / endometrial (OVCAR3, Ishikawa)": ["PAX8", "WT1", "MUC16"],
    "lung adenocarcinoma (Calu-3)": ["NKX2-1", "MUC5AC", "SFTPB"],
}
MARKER_ROWS: dict[str, list[str]] = {
    "squamous (FaDu, KYSE30, cSCC4)": ["Fadu cell line", "TPF-resistant Fadu", "TPF-resistant Fadu cell line", "KYSE30", "cSCC4"],
    "intestinal (HCT116, HT-29, Caco-2)": ["HCT116", "HT-29", "Caco-2"],
    "prostate (LNCaP)": ["LNCaP"],
    "melanoma (A375)": ["A375"],
    "ovarian / endometrial (OVCAR3, Ishikawa)": ["OVCAR3", "Ishikawa"],
    "lung adenocarcinoma (Calu-3)": ["Calu-3"],
}

#: The genes a reader wants to see beside the contexts: T lineage, mesenchymal / epithelial, the
#: p53 axis (the 2026-10-09 marker read), squamous, proliferation.
GENE_TABLE = ["CD3E", "PTPRC", "VIM", "KRT18", "KRT8", "KRT7", "EPCAM", "CDH1", "CDKN2A", "TP53",
              "CDKN1A", "KRT5", "KRT14", "TP63", "SOX2", "MKI67"]
GENE_TABLE_ROWS = ["HeLa", "HeLa cells", "HeLa S3", "Fadu cell line", "KYSE30", "cSCC4", "Jurkat",
                   "K562", "HCT116", "HEK293T", "HEK293FT", "MDAMB231", "MDA-MB-231", "OVCAR5", "U2OS"]


# ------------------------------------------------------------------ helpers --


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_panel(paths: list[Path]) -> tuple[ContextProfiles, list[dict]]:
    parts = [ContextProfiles.load(p) for p in paths]
    genes = parts[0].genes
    for p, path in zip(parts, paths):
        if not np.array_equal(p.genes, genes):
            raise SystemExit(f"{path}: gene axis differs from {paths[0]}; the parts must share one axis")
    labels = [c for p in parts for c in p.contexts]
    if len(set(labels)) != len(labels):
        dup = sorted({c for c in labels if labels.count(c) > 1})
        raise SystemExit(f"duplicate panel rows across parts: {dup}")
    cat = lambda k: np.concatenate([getattr(p, k) for p in parts], axis=0)
    merged = ContextProfiles(
        contexts=labels, genes=genes, count_sum=cat("count_sum"), cpm_sum=cat("cpm_sum"),
        cpm_sq_sum=cat("cpm_sq_sum"), detect_count=cat("detect_count"), n_cells=cat("n_cells"),
        libsize_sum=cat("libsize_sum"), n_experiments=cat("n_experiments"),
        sources=sorted({s for p in parts for s in p.sources}),
    )
    record = [{"path": str(p), "sha256": _sha256(p), "rows": len(part.contexts),
               "cells": int(part.n_cells.sum())} for p, part in zip(paths, parts)]
    return merged, record


def context_query(path: Path, axis: np.ndarray) -> dict:
    """Mean of per-cell CPM over a context's control cells, on the panel's axis."""
    with h5py.File(path) as h:
        X = h["X"]
        shape = tuple(X.attrs["shape"])
        M = sp.csr_matrix((X["data"][:], X["indices"][:], X["indptr"][:]), shape=shape)
        var = read_elem(h["var"]).index.to_numpy().astype(str)
    if not np.array_equal(var, axis):
        raise SystemExit(f"{path}: var axis is not the panel's axis")
    lib = np.asarray(M.sum(axis=1)).ravel()
    cpm = sp.diags(1e6 / np.maximum(lib, 1.0)) @ M
    mean_cpm = np.asarray(cpm.mean(axis=0)).ravel()
    pooled_cpm = np.asarray(M.sum(axis=0)).ravel() / max(lib.sum(), 1.0) * 1e6
    return {"mean_cpm": mean_cpm, "pooled_cpm": pooled_cpm, "support": np.ones(len(axis), bool),
            "n_cells": int(M.shape[0]), "median_umi": float(np.median(lib)),
            "genes_measured": int(len(var))}


def line_query(path: Path, label: str, axis: np.ndarray) -> dict:
    """The control row of one of our pseudobulk files, mapped onto the panel's axis by symbol."""
    ps = PseudobulkSums.load(path)
    i = ps.labels.index(label)
    pos = {g: k for k, g in enumerate(axis)}
    idx = np.array([pos.get(str(g), -1) for g in ps.genes])
    keep = idx >= 0
    v = np.zeros(len(axis))
    v[idx[keep]] = ps.mean_cpm()[i][keep]
    support = np.zeros(len(axis), bool)
    support[idx[keep]] = True
    return {"mean_cpm": v, "pooled_cpm": None, "support": support, "n_cells": int(ps.n_cells[i]),
            "median_umi": float(ps.libsize_sum[i] / max(ps.n_cells[i], 1)),
            "genes_measured": int(len(ps.genes)), "genes_on_axis": int(keep.sum())}


def cosine_rows(q: np.ndarray, P: np.ndarray) -> np.ndarray:
    u = np.log1p(q)
    V = np.log1p(P)
    return (V @ u) / (np.linalg.norm(V, axis=1) * np.linalg.norm(u) + 1e-300)


def rank_query(q: dict, panel_mean: np.ndarray, labels: list[str], depth: np.ndarray,
               exclude_row: str | None = None) -> dict:
    """Plain and renormalised rankings of every panel row for one query vector."""
    S = q["support"]
    rows = [k for k, c in enumerate(labels) if c != exclude_row]
    P = panel_mean[rows][:, S]
    x = q["mean_cpm"][S]
    plain = cosine_rows(x, P)
    Pn = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-300) * 1e6
    xn = x / max(x.sum(), 1e-300) * 1e6
    renorm = cosine_rows(xn, Pn)
    out = {}
    for name, cos in (("plain", plain), ("renorm", renorm)):
        order = np.argsort(-cos)
        ranked = [(labels[rows[j]], float(cos[j])) for j in order]
        top_line = LINE_OF.get(ranked[0][0], ranked[0][0])
        other = next((c for lab, c in ranked[1:] if LINE_OF.get(lab, lab) != top_line), None)
        rho, p = spearmanr(depth[rows], cos)
        out[name] = {
            "ranking": ranked,
            "top": ranked[0][0], "top_line": top_line, "cos": ranked[0][1],
            "gap_top2": ranked[0][1] - ranked[1][1],
            "gap_other_line": (ranked[0][1] - other) if other is not None else None,
            "depth_spearman": {"rho": float(rho), "p": float(p)},
        }
    return out


def line_rank(ranking: list[tuple[str, float]], line: str) -> int | None:
    for r, (lab, _) in enumerate(ranking, start=1):
        if LINE_OF.get(lab, lab) == line:
            return r
    return None


# --------------------------------------------------------------------- main --


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", type=Path, action="append", required=True,
                    help="a ContextProfiles npz; repeat to concatenate parts (one shared gene axis)")
    ap.add_argument("--merged-out", type=Path, help="write the concatenated panel here (+ .lineage.json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--contexts", default="A,B,C")
    ap.add_argument("--lines", default=",".join(LINE_QUERIES),
                    help="which of our own control pseudobulks to run as positive controls")
    ap.add_argument("--ablate-rows", default="",
                    help="comma-separated panel rows; each context is re-ranked with each one removed")
    ap.add_argument("--depth-genes", type=int, default=2000,
                    help="how many depth-tracking genes the depth-robust ranking leaves out")
    args = ap.parse_args(argv)

    t0 = time.time()
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    panel, parts = load_panel([p.expanduser() for p in args.panel])
    labels = panel.contexts
    axis = panel.genes
    mean = panel.mean_cpm()
    depth = panel.libsize_sum / np.maximum(panel.n_cells, 1)
    print(f"panel: {len(labels)} rows, {len(axis):,} genes, {int(panel.n_cells.sum()):,} cells "
          f"from {len(parts)} part(s)")

    # the genes whose level across panel rows tracks row depth most: left out of the depth-robust read
    L = np.log1p(mean)
    rho_depth = np.array([abs(spearmanr(depth, L[:, j]).statistic) if L[:, j].std() > 0 else 0.0
                          for j in range(L.shape[1])])
    depth_mask = np.ones(len(axis), bool)
    depth_mask[np.argsort(-np.nan_to_num(rho_depth))[: args.depth_genes]] = False
    ablate = [x.strip() for x in args.ablate_rows.split(",") if x.strip()]
    for a in ablate:
        if a not in labels:
            raise SystemExit(f"--ablate-rows names {a!r}, which is not a panel row")

    results: dict = {
        "task": "T88", "session": "8e4f9c3c", "date": time.strftime("%Y-%m-%d"),
        "depth_robust": {"genes_left_out": int((~depth_mask).sum()),
                         "min_abs_rho_left_out": float(np.sort(np.nan_to_num(rho_depth))[::-1][args.depth_genes - 1])},
        "code_sha": _git_sha(), "panel_parts": parts,
        "panel_rows": [{"row": c, "line": LINE_OF.get(c, c), "n_cells": int(panel.n_cells[k]),
                        "n_experiments": int(panel.n_experiments[k]), "mean_umi_per_cell": float(depth[k])}
                       for k, c in enumerate(labels)],
        "queries": {},
    }
    long_rows = []

    # -- the contexts
    for c in [x.strip() for x in args.contexts.split(",") if x.strip()]:
        q = context_query(DATA / f"vcc2026/context_{c}.h5ad", axis)
        r = rank_query(q, mean, labels, depth)
        # pooled-CPM variant, for the record: the same instrument on the other pseudobulk convention
        qp = dict(q, mean_cpm=q["pooled_cpm"])
        rp = rank_query(qp, mean, labels, depth)["plain"]
        qd = dict(q, support=q["support"] & depth_mask)
        rd = rank_query(qd, mean, labels, depth)["plain"]
        abl = {}
        for a in ablate:
            ra = rank_query(q, mean, labels, depth, exclude_row=a)["plain"]
            abl[a] = {"top": ra["top"], "cos": ra["cos"], "gap_top2": ra["gap_top2"],
                      "gap_other_line": ra["gap_other_line"], "top5": ra["ranking"][:5]}
        results["queries"][f"context_{c}"] = {
            "kind": "challenge context", "n_cells": q["n_cells"], "median_umi_per_cell": q["median_umi"],
            "genes_measured": q["genes_measured"],
            **{k: {kk: vv for kk, vv in v.items() if kk != "ranking"} for k, v in r.items()},
            "pooled_cpm_plain": {"top": rp["top"], "cos": rp["cos"], "gap_top2": rp["gap_top2"]},
            "depth_robust_plain": {"top": rd["top"], "cos": rd["cos"], "gap_top2": rd["gap_top2"],
                                   "gap_other_line": rd["gap_other_line"], "top5": rd["ranking"][:5]},
            "ablations": abl,
            "top5_plain": r["plain"]["ranking"][:5], "top5_renorm": r["renorm"]["ranking"][:5],
        }
        for a, ra in abl.items():
            print(f"    without {a!r}: top {ra['top']} {ra['cos']:.4f} gap {ra['gap_top2']:.4f} (other line {ra['gap_other_line']:.4f})")
        print(f"    depth-robust: top {rd['top']} {rd['cos']:.4f} gap {rd['gap_top2']:.4f}")
        for name in ("plain", "renorm"):
            for rank, (lab, cos) in enumerate(r[name]["ranking"], start=1):
                long_rows.append({"query": f"context_{c}", "variant": name, "rank": rank, "row": lab,
                                  "line": LINE_OF.get(lab, lab), "cos": cos})
        print(f"context {c}: plain top {r['plain']['top']} {r['plain']['cos']:.4f} gap {r['plain']['gap_top2']:.4f} "
              f"(other line {r['plain']['gap_other_line']:.4f}) | renorm top {r['renorm']['top']} "
              f"{r['renorm']['cos']:.4f} gap {r['renorm']['gap_top2']:.4f}")

    # -- our own lines as positive controls
    for name in [x.strip() for x in args.lines.split(",") if x.strip()]:
        path, label, expect = LINE_QUERIES[name]
        if not path.exists():
            print(f"  {name}: {path} missing, skipped"); continue
        q = line_query(path, label, axis)
        r = rank_query(q, mean, labels, depth)
        present = any(LINE_OF.get(c, c) == expect for c in labels)
        entry = {
            "kind": "positive control", "expected_line": expect, "expected_line_in_panel": present,
            "n_cells": q["n_cells"], "median_umi_per_cell": q["median_umi"],
            "genes_measured": q["genes_measured"], "genes_on_axis": q["genes_on_axis"],
            **{k: {kk: vv for kk, vv in v.items() if kk != "ranking"} for k, v in r.items()},
            "rank_of_expected_plain": line_rank(r["plain"]["ranking"], expect),
            "rank_of_expected_renorm": line_rank(r["renorm"]["ranking"], expect),
            "top5_plain": r["plain"]["ranking"][:5], "top5_renorm": r["renorm"]["ranking"][:5],
        }
        results["queries"][f"our_{name}"] = entry
        for v in ("plain", "renorm"):
            for rank, (lab, cos) in enumerate(r[v]["ranking"], start=1):
                long_rows.append({"query": f"our_{name}", "variant": v, "rank": rank, "row": lab,
                                  "line": LINE_OF.get(lab, lab), "cos": cos})
        flag = "" if not present else (" OK" if entry["rank_of_expected_plain"] == 1 else " MISS")
        print(f"our {name:8s}: plain top {r['plain']['top']:14s} {r['plain']['cos']:.4f} gap {r['plain']['gap_top2']:.4f} "
              f"| expected {expect} rank {entry['rank_of_expected_plain']}"
              f"{'' if present else ' (not in panel)'}{flag} | renorm top {r['renorm']['top']} gap {r['renorm']['gap_top2']:.4f}")

    pd.DataFrame(long_rows).to_csv(out / "rankings.csv", index=False)

    # -- held-out-spelling control: a row against the other rows
    spell = []
    for k, c in enumerate(labels):
        line = LINE_OF.get(c, c)
        sibs = [x for x in labels if x != c and LINE_OF.get(x, x) == line]
        if not sibs:
            continue
        q = {"mean_cpm": mean[k], "support": np.ones(len(axis), bool)}
        r = rank_query(q, mean, labels, depth, exclude_row=c)["plain"]
        ranked = r["ranking"]
        best_sib = min(line_rank([(lab, s) for lab, s in ranked], line) or 10**6, 10**6)
        top_other = next(((lab, s) for lab, s in ranked if LINE_OF.get(lab, lab) != line), (None, None))
        spell.append({"row": c, "line": line, "siblings": "|".join(sibs), "best_sibling_rank": best_sib,
                      "best_sibling_cos": next(s for lab, s in ranked if LINE_OF.get(lab, lab) == line),
                      "top_row": ranked[0][0], "top_cos": ranked[0][1],
                      "top_other_line_row": top_other[0], "top_other_line_cos": top_other[1]})
    spell_df = pd.DataFrame(spell)
    spell_df.to_csv(out / "spelling_control.csv", index=False)
    if len(spell_df):
        ok = int((spell_df.best_sibling_rank == 1).sum())
        print(f"held-out-spelling control: {ok} of {len(spell_df)} rows put a same-line sibling first")
        results["spelling_control"] = {"rows": len(spell_df), "sibling_first": ok,
                                       "misses": spell_df[spell_df.best_sibling_rank != 1][["row", "best_sibling_rank", "top_row"]].to_dict("records")}

    # -- lineage markers for the new rows
    gpos = {g: k for k, g in enumerate(axis)}
    mk = []
    for group, genes in MARKERS.items():
        rows = [r for r in MARKER_ROWS[group] if r in labels]
        for g in genes:
            if g not in gpos:
                continue
            col = mean[:, gpos[g]]
            order = np.argsort(-col)
            rank_of = {labels[j]: r for r, j in enumerate(order, start=1)}
            for r in rows:
                mk.append({"group": group, "row": r, "gene": g, "rank_of_row": rank_of[r],
                           "n_rows": len(labels), "cpm": float(col[labels.index(r)]),
                           "top_row": labels[order[0]], "top_cpm": float(col[order[0]])})
    mk_df = pd.DataFrame(mk)
    mk_df.to_csv(out / "markers.csv", index=False)
    if len(mk_df):
        top3 = int((mk_df.rank_of_row <= 3).sum())
        print(f"markers: {top3} of {len(mk_df)} (row, gene) pairs put the row in the top 3 of {len(labels)}")
        results["markers"] = {"pairs": len(mk_df), "row_in_top3": top3}

    # -- the gene table: contexts beside the candidate rows
    tbl = {}
    for c in [x.strip() for x in args.contexts.split(",") if x.strip()]:
        q = context_query(DATA / f"vcc2026/context_{c}.h5ad", axis)
        tbl[f"context_{c}"] = {g: float(q["mean_cpm"][gpos[g]]) for g in GENE_TABLE if g in gpos}
    for r in GENE_TABLE_ROWS:
        if r in labels:
            tbl[r] = {g: float(mean[labels.index(r), gpos[g]]) for g in GENE_TABLE if g in gpos}
    gt = pd.DataFrame(tbl).T
    gt.index.name = "profile"
    gt.round(1).to_csv(out / "gene_table.csv")

    # -- the merged panel as an artifact
    if args.merged_out:
        mo = args.merged_out.expanduser()
        panel.save(mo)
        lineage = {
            "produced_by": "scripts/t88_context_match.py (concatenation of streamed parts)",
            "code_sha": _git_sha(), "task": "T88", "session": "8e4f9c3c",
            "parts": parts, "rows": len(labels), "genes": int(len(axis)),
            "cells": int(panel.n_cells.sum()), "note": "rows are kept exactly as each part streamed them; "
            "the parts' own .lineage.json files carry the filter, the exclusion list and the experiments",
        }
        mo.with_name(mo.name + ".lineage.json").write_text(json.dumps(lineage, indent=1) + "\n")
        results["merged_panel"] = {"path": str(mo), "sha256": _sha256(mo), "rows": len(labels)}
        print(f"merged panel -> {mo}")

    results["seconds"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    print(f"-> {out / 'results.json'} ({results['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
