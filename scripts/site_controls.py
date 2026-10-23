"""Write the data behind the site's control-cells page (site/content/controls.md) from the board's `challenge` figure stage.

usage (cwd ~/code/sidechain): .venv/bin/python scripts/site_controls.py [--src DIR] [--out DIR] [--check]
Reads  ~/data/sidechain/derived/figures/challenge/ (summary, cells_<X>, guides_<X>, genes_<X>, joint, gene_index, manifest).
Writes site/static/data/controls/: index.json, cells_<X>.json, emb_<X>.json, guides_<X>.json, genes_<X>.json,
       joint.json, genes.json. Every field is copied by name; anything not named here never reaches the site, and a
       string outside the page's vocabulary (context letters, metric names, guide ids, gene symbols, colours) stops
       the build. --check re-reads an existing output directory with the same vocabulary test and writes nothing.
Pointer: private/tasks/T107.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np

from sidechain.viz import palette
from sidechain.viz.qc import compact

REPO = Path(__file__).resolve().parents[1]
SRC = Path("~/data/sidechain/derived/figures/challenge").expanduser()
OUT = REPO / "site" / "static" / "data" / "controls"

CONTEXTS = ("A", "B", "C", "D", "E", "F")
ROUND = {"validation": ("A", "B", "C"), "final": ("D", "E", "F")}
PHASES = ("G1", "S", "G2M")
INT_METRICS = {"total_umi", "n_genes"}
SCORE_METRICS = {"S_score", "G2M_score", "ifn_score", "stress_score"}
METRIC_ORDER = ("total_umi", "n_genes", "pct_mt", "pct_ribo", "pct_top50", "complexity",
                "S_score", "G2M_score", "ifn_score", "stress_score")
SUMMARY_FIELDS = ("n_cells", "n_guides", "median", "phase_share", "n_clusters", "n_clusters_resid")
GUIDE_FIELDS = ("guide", "dist_z", "dist_q", "dist_z_resid", "dist_q_resid", "chi2_q_resid",
                "top_cluster_share_resid", "top_cluster_expected_resid", "median_umi", "median_pct_mt", "share_G2M")
EMBEDDINGS = ("umap", "pca", "umap_resid", "pca_resid")
GENE_FIELDS = ("mean_count", "detect", "mean_cp10k")

GUIDE_ID = re.compile(r"^non-targeting-\d+$")
OK_PATTERNS = (re.compile(r"^#[0-9a-fA-F]{6}$"), re.compile(r"^\d{4}-\d{2}-\d{2}(T[\d:+-]+)?$"),
               re.compile(r"^[0-9a-f]{7,12}$"), re.compile(r"^\d+$"), GUIDE_ID)
FIXED_WORDS = {*CONTEXTS, *PHASES, *METRIC_ORDER, "validation", "test", "final", "circle", "diamond", "light", "dark",
               "neg", "pos", "mid", "lfc"}


def fixed(a, decimals: int) -> list:
    """A float array rounded to fixed decimals (scores and coordinates sit near zero, where significant figures bloat)."""
    a = np.round(np.asarray(a, dtype=np.float64), decimals)
    return [None if not np.isfinite(v) else (int(v) if float(v).is_integer() else float(v)) for v in a.tolist()]


def metric_values(key: str, v) -> list:
    if key in INT_METRICS:
        return [None if x is None else int(round(x)) for x in v]
    if key in SCORE_METRICS:
        return fixed(v, 3)
    return compact(np.asarray(v, dtype=np.float64), 3 if key != "complexity" else 4)


def codes(block: dict) -> dict:
    return {"levels": [str(x) for x in block["levels"]], "codes": [int(x) for x in block["codes"]]}


def load(src: Path, name: str):
    return json.loads((src / name).read_text())


def build(src: Path) -> dict[str, object]:
    """Every output file as {name: object}, built field by field from the stage under `src`."""
    summ, man = load(src, "summary.json"), load(src, "manifest.json")
    ctxs = [c for c in CONTEXTS if c in summ["contexts"]]
    absent = set().union(*(summ["contexts"][c].get("metrics_absent") or [] for c in ctxs))
    metrics = [m for m in METRIC_ORDER if m in summ["metrics"] and m not in absent]
    genes = [str(g) for g in load(src, "gene_index.json")["genes"]]
    out: dict[str, object] = {"genes.json": {"genes": genes}}
    contexts = {}
    for c in ctxs:
        s = summ["contexts"][c]
        row = {k: s[k] for k in SUMMARY_FIELDS}
        row["median"] = {m: s["median"][m] for m in metrics}
        row["phase_share"] = {p: s["phase_share"].get(p, 0.0) for p in PHASES}
        row["round"] = palette.CONTEXTS[c]["round"]
        contexts[c] = row
        cells = load(src, f"cells_{c}.json")
        cat = cells["cat"]
        out[f"cells_{c}.json"] = {"context": c, "n": int(cells["n"]),
                                  "metrics": {m: metric_values(m, cells["metrics"][m]) for m in metrics},
                                  "cat": {k: codes(cat[k]) for k in ("guide", "phase", "cluster", "cluster_resid")}}
        emb = cells["emb"]
        out[f"emb_{c}.json"] = {"context": c, **{k: [fixed(col, 2) for col in emb[k]] for k in EMBEDDINGS},
                                "pca_var": compact(np.asarray(cells["pca_var"]), 4),
                                "pca_var_resid": compact(np.asarray(cells["pca_var_resid"]), 4)}
        g = load(src, f"guides_{c}.json")["rows"]
        out[f"guides_{c}.json"] = {"context": c, "n_perm": 500, "rows": {
            k: ([str(x) for x in g[k]] if k == "guide" else compact(np.asarray(g[k], dtype=np.float64), 4))
            for k in GUIDE_FIELDS}}
        gn = load(src, f"genes_{c}.json")
        out[f"genes_{c}.json"] = {"context": c, **{k: compact(np.asarray(gn[k], dtype=np.float64), 3) for k in GENE_FIELDS},
                                  "hvg": [bool(x) for x in gn["hvg"]],
                                  "top20_share": [{"gene": str(r["gene"]), "q": compact(np.asarray(r["q"]), 3)}
                                                  for r in gn["top20_share"]]}
    j = load(src, "joint.json")
    if [str(x) for x in j["contexts"]] != ctxs:
        raise SystemExit(f"joint.json holds {j['contexts']}, the per-context files {ctxs}: rebuild the stage")
    out["joint.json"] = {"contexts": ctxs, "ctx_codes": [int(x) for x in j["ctx_codes"]],
                         "umap": [fixed(col, 2) for col in j["umap"]], "pca": [fixed(col, 1) for col in j["pca"]],
                         "pca_var": compact(np.asarray(j["pca_var"]), 4), "cluster": [int(x) for x in j["cluster"]]}
    pal = palette.PALETTE
    out["index.json"] = {
        "built": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage_built": str(man.get("built", ""))[:10], "stage_commit": str(man.get("commit", "")),
        "contexts": contexts, "metrics": {m: summ["metrics"][m] for m in metrics},
        "targets": {"validation": [str(x) for x in summ["targets"].get("validation", [])],
                    "final": [str(x) for x in summ["targets"].get("final", [])]},
        "palette": {"contexts": {c: {k: pal["contexts"][c][k] for k in ("light", "dark", "marker", "round")} for c in ctxs},
                    "sequential": list(pal["sequential"]), "diverging": {"lfc": pal["diverging"]["lfc"]},
                    "phase": pal["phase"], "clusters": list(pal["clusters"])},
    }
    return out


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)


def vocabulary(files: dict[str, object]) -> set[str]:
    """The strings the page may carry: fixed words, the stage's metric labels, field names, gene symbols, guide ids."""
    idx = files["index.json"]
    words = set(FIXED_WORDS) | set(idx["metrics"].values())
    words |= {"genes", "context", "n", "metrics", "cat", "levels", "codes", "guide", "phase", "cluster", "cluster_resid",
              "rows", "n_perm", "hvg", "top20_share", "gene", "q", "pca_var", "pca_var_resid", "ctx_codes", "built",
              "stage_built", "stage_commit", "contexts", "targets", "palette", "sequential", "diverging", "clusters",
              "marker", "round", "median", "phase_share", *SUMMARY_FIELDS, *GUIDE_FIELDS, *EMBEDDINGS, *GENE_FIELDS}
    words |= set(files["genes.json"]["genes"])
    return words


def check(files: dict[str, object]) -> list[str]:
    """Every string that is neither in the page's vocabulary nor one of its patterns, as `file: value`."""
    ok = vocabulary(files)
    bad = []
    for name, obj in files.items():
        for s in strings(obj):
            if s not in ok and not any(p.match(s) for p in OK_PATTERNS):
                bad.append(f"{name}: {s!r}")
    return bad


def write(files: dict[str, object], out: Path) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, obj in sorted(files.items()):
        text = json.dumps(obj, separators=(",", ":"), allow_nan=False)
        tmp = out / (name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(out / name)
        sizes[name] = len(text)
    return sizes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="test an existing output directory; write nothing")
    a = ap.parse_args(argv)
    if a.check:
        files = {p.name: json.loads(p.read_text()) for p in sorted(a.out.glob("*.json"))}
    else:
        files = build(a.src)
    bad = check(files)
    if bad:
        print(f"{len(bad)} string(s) outside the page's vocabulary; nothing written:", *bad[:20], sep="\n  ")
        return 1
    if a.check:
        print(f"{len(files)} files in {a.out}: every string is in the page's vocabulary")
        return 0
    sizes = write(files, a.out)
    for name, n in sizes.items():
        print(f"{name:18s} {n / 1e6:6.2f} MB")
    print(f"total {sum(sizes.values()) / 1e6:.1f} MB -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
