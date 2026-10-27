"""A PIE predictions parquet's `lfc_pred` as the per-target log2FC map, in place of the pooled delta.

    --prediction-source {ser,pie} --pie-parquet PATH --pie-context [CTX=]NAME      (submit.build, eval.loco)

Reads one context's rows of the parquet (gene order in its `pie` schema-metadata key); writes nothing.
Notes: private/code-notes/src/sidechain/submit/pie_source.py.md
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FLOOR = -10.0          # a value at or below this is set to 0
CLIP = 5.0
SOURCES = ("ser", "pie")


def add_prediction_source_args(ap, *, twin: str, per_context: bool) -> None:
    ap.add_argument("--prediction-source", choices=list(SOURCES), default="ser",
                    help="where each target's log2FC map comes from before alpha. 'ser' (the default) is "
                         "the pooled delta; 'pie' is PIE's lfc_pred from --pie-parquet (values at or below "
                         f"{FLOOR:g} set to 0, clipped to +-{CLIP:g}, unscored genes 0), and every later step "
                         f"is unchanged (sidechain.submit.pie_source; the same flag on {twin})")
    ap.add_argument("--pie-parquet", type=Path, default=None, metavar="PATH",
                    help="with --prediction-source pie: PIE's predictions parquet")
    if per_context:
        ap.add_argument("--pie-context", action="append", default=[], metavar="[CTX=]NAME",
                        help="with --prediction-source pie: the parquet's context rows a challenge context reads, "
                             "as CTX=NAME (repeatable); a bare NAME is every context's; a context named in "
                             "neither reads the rows of its own name")
    else:
        ap.add_argument("--pie-context", default=None, metavar="NAME",
                        help="with --prediction-source pie: the parquet's context rows this fold reads")


def check_prediction_source_args(ap, args, *, per_context: bool) -> None:
    pie = args.prediction_source == "pie"
    given = args.pie_parquet is not None or bool(args.pie_context)
    if not pie and given:
        ap.error("--pie-parquet and --pie-context act only with --prediction-source pie")
    if not pie:
        return
    if args.pie_parquet is None:
        ap.error("--prediction-source pie needs --pie-parquet")
    if not Path(args.pie_parquet).expanduser().exists():
        ap.error(f"--pie-parquet: {args.pie_parquet} does not exist")
    if not per_context and not args.pie_context:
        ap.error("--prediction-source pie needs --pie-context NAME")


def parse_pie_contexts(specs: list[str], contexts: list[str]) -> dict[str, str]:
    """{challenge context: the parquet's context name} from `[CTX=]NAME` specs (build's `--pie-context`)."""
    out = {c: c for c in contexts}
    bare = [s for s in specs if "=" not in s]
    if len(bare) > 1:
        raise SystemExit(f"--pie-context: more than one bare NAME ({', '.join(bare)}); a bare name is every context's")
    if bare:
        out = {c: bare[0] for c in contexts}
    seen = set()
    for s in specs:
        if "=" not in s:
            continue
        ctx, name = (x.strip() for x in s.split("=", 1))
        if ctx not in out:
            raise SystemExit(f"--pie-context {s!r}: {ctx!r} is not a context of this build ({', '.join(contexts)})")
        if ctx in seen or not name:
            raise SystemExit(f"--pie-context {s!r}: a context named twice, or no name")
        seen.add(ctx)
        out[ctx] = name
    return out


def read_pie_rows(path: Path, context: str) -> tuple[list[str], list[str], np.ndarray, dict]:
    """One context's rows of a PIE parquet: perturbations, PIE's genes, `lfc_pred` [P, G], and the file's record."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    path = Path(path).expanduser()
    meta = pq.read_schema(path).metadata or {}
    if b"pie" not in meta:
        raise SystemExit(f"--pie-parquet {path}: no 'pie' key in the schema metadata (the gene order lives there)")
    gene_sets = json.loads(meta[b"pie"])["genes"]
    t = pq.read_table(path, columns=["context", "perturbation", "dataset", "lfc_pred"])
    names = sorted(set(t.column("context").to_pylist()))
    t = t.filter(pc.equal(t.column("context"), context))
    if t.num_rows == 0:
        raise SystemExit(f"--pie-parquet {path.name}: no rows for context {context!r} (it has {', '.join(names)})")
    datasets = sorted(set(t.column("dataset").to_pylist()))
    if len(datasets) != 1:
        raise SystemExit(f"--pie-parquet {path.name}: context {context!r} spans datasets {datasets}; one gene "
                         "order per read")
    dataset = datasets[0]
    if dataset not in gene_sets:
        raise SystemExit(f"--pie-parquet {path.name}: no gene order for dataset {dataset!r} in the metadata")
    genes = [str(g) for g in gene_sets[dataset]]
    perts = [str(p) for p in t.column("perturbation").to_pylist()]
    dup = sorted({p for p in perts if perts.count(p) > 1})
    if dup:
        raise SystemExit(f"--pie-parquet {path.name}: context {context!r} lists {len(dup)} perturbation(s) twice, "
                         f"e.g. {dup[:3]}")
    lfc = np.stack([np.asarray(v, dtype=np.float64) for v in t.column("lfc_pred").to_pylist()])
    if lfc.shape[1] != len(genes):
        raise SystemExit(f"--pie-parquet {path.name}: lfc_pred has {lfc.shape[1]} values a row, the gene order "
                         f"{len(genes)}")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    record = {"parquet": str(path), "sha256": h.hexdigest(), "context": context, "dataset": dataset,
              "rows": len(perts), "pie_genes": len(genes)}
    return perts, genes, lfc, record


def to_log2fc(lfc: np.ndarray) -> tuple[np.ndarray, dict]:
    """PIE's `lfc_pred` as a log2FC: non-finite and at-or-below-`FLOOR` values to 0, then clipped to +-`CLIP`."""
    x = np.array(lfc, dtype=np.float64, copy=True)
    bad = ~np.isfinite(x)
    x[bad] = 0.0
    floored = x <= FLOOR
    x[floored] = 0.0
    clipped = np.abs(x) > CLIP
    np.clip(x, -CLIP, CLIP, out=x)
    return x, {"floor": FLOOR, "clip": CLIP, "values_non_finite": int(bad.sum()),
               "values_floored": int(floored.sum()), "values_clipped": int(clipped.sum())}


def pie_shift_map(path: Path, context: str, axis, perturbations=None) -> tuple[dict, dict]:
    """{perturbation: log2FC vector on `axis`} from one context's rows of a PIE parquet, and its record.

    `perturbations` (the build's or fold's list) limits the map and counts what it misses; None keeps every row.
    """
    perts, genes, lfc, record = read_pie_rows(path, context)
    axis = np.asarray(axis, dtype=str)
    vals, conv = to_log2fc(lfc)
    pos = {g: i for i, g in enumerate(genes)}
    idx = np.array([pos.get(g, -1) for g in axis], dtype=np.int64)
    hit = idx >= 0
    wanted = None if perturbations is None else [str(p) for p in perturbations]
    keep = set(perts) if wanted is None else set(perts) & set(wanted)
    out = {}
    for i, p in enumerate(perts):
        if p not in keep:
            continue
        v = np.zeros(len(axis), dtype=np.float64)
        v[hit] = vals[i, idx[hit]]
        out[p] = v
    record = {**record, **conv,
              "axis_genes": int(len(axis)), "axis_genes_scored": int(hit.sum()),
              "axis_genes_unscored": int((~hit).sum()),
              "pie_genes_off_axis": int(len(genes) - len(set(genes) & set(axis.tolist()))),
              "perturbations_mapped": len(out)}
    if wanted is not None:
        missing = [p for p in wanted if p not in out]
        record |= {"perturbations_asked": len(wanted), "perturbations_missing": len(missing),
                   "missing": missing[:50], "rows_not_asked": len(set(perts) - set(wanted))}
    return out, record
