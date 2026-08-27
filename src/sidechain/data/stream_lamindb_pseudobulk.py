"""Stream a LaminDB-hosted h5ad into a per-label pseudobulk, without landing it.

The third streaming reader, after local h5ad (`stream_pseudobulk`) and HF
parquet (`stream_parquet_pseudobulk`), and it emits the same `PseudobulkSums`
object -- so a lamin-hosted corpus becomes a new `--source` on the existing
`sidechain.eval.loco` / `sidechain.submit.build` command lines with no change
to the model, the emitter, or the scorer.

Built for `laminlabs/pertdata`'s curated scperturb tree, whose uniform layout is
`scperturb/<dataset>/{X.h5ad, obs.parquet, var.parquet}`:

  * `Artifact.open()` on the X.h5ad returns a lazy accessor whose `.storage` is
    an open h5py.File over S3 -- the SAME handle type the local streamer uses,
    so the accumulator (`stream_pseudobulk_file`) is shared, not re-implemented.
  * the harmonized perturbation label (`pert_target`) lives in the obs.parquet
    SIDECAR, not in X.h5ad's own obs. Sidecar-to-matrix row alignment is
    ASSERTED over every row before any label is trusted (`--skip-align-check`
    exists for a dataset whose X.h5ad carries no cell index at all, and prints
    loudly). A misjoin here would relabel cells silently, which is the same
    failure `configs/datasets.yaml` documents for Feng's three barcode spellings.
  * `--gene-col symbol` re-keys the gene axis from the var INDEX (ensembl ids in
    pertdata) to a var column, because the pooling remaps sources onto the
    challenge axis by SYMBOL. Empty entries become "" and never match; the meta
    line counts them and the duplicates rather than hiding the loss.

    uv run python -m sidechain.data.stream_lamindb_pseudobulk \
        --instance laminlabs/pertdata --key scperturb/frangieh21 \
        --label-col pert_target --gene-col symbol \
        --out ~/data/sidechain/derived/lamin-pertdata/frangieh21_all_pseudobulk.npz

Writes `<out>` plus `<out stem>.lineage.json` beside it: which instance and
artifact versions (uid, hash, size) produced the aggregate, at what resolution,
by which code. That file is what answers "what is this npz?" without re-reading
gigabytes over the network (ADR 0003's LINEAGE shape).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sidechain.data.stream_pseudobulk import stream_pseudobulk_file

try:  # anndata >= 0.11
    from anndata.io import read_elem
except ImportError:  # pragma: no cover - older anndata
    from anndata.experimental import read_elem


def _artifact(db, key: str):
    # `.defer("extra_data")` because an instance whose SQLite snapshot predates
    # that column (altoslabs/perturbench) crashes ANY full-model fetch with
    # `no such column`; deferring it keeps `.open()` working everywhere.
    art = db.Artifact.filter(key=key, is_latest=True).defer("extra_data").first()
    if art is None:
        raise SystemExit(f"no latest artifact with key {key!r} in this instance")
    return art


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:  # noqa: BLE001 -- lineage should record "unknown", not crash the stream
        return "unknown"


DROP = "__drop__"  # sentinel label for rows excluded from every pseudobulk


def _derive_labels(
    obs: pd.DataFrame,
    label_col: str,
    *,
    multi_label: bool = False,
    control_rule: tuple[str, tuple[str, ...], str] | None = None,
    row_filters: dict[str, str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Per-cell pseudobulk labels from the sidecar obs, with the three shapes
    the curated pertdata corpora actually come in.

    * `multi_label` -- `pert_target_multi` holds a STRINGIFIED list
      (`"['VAC14']"`, `"['CEBPE' 'RUNX1T1']"`, `"None"`). A row with exactly one
      token becomes that token; combinatorial and empty rows are dropped, not
      guessed at.
    * `control_rule` = (col, prefixes, out_label) -- control cells are found by
      their GUIDE NAME (`pert_name` startswith `non-targeting`/`NO_SITE`/...),
      because the harmonized `pert_target` for a control cell is NaN, which is
      also what an unassigned cell reads. The rule is applied AFTER the base
      label so it overrides NaN with the declared control label; NaN rows the
      rule does not claim are dropped rather than pooled into the control arm --
      that distinction is the Feng lesson in the other direction (there the
      paper POOLS no-guide into controls; here nothing says these curations do).
    * `row_filters` = {col: value} -- keep only rows where obs[col] == value.
      Frangieh's three condition arms (Control / Co-culture / IFN-γ) are the
      case: pooling them would blend a stimulation into every knockout delta.

    Returns `(labels, stats)`; dropped rows carry the DROP sentinel, which the
    caller hands to the accumulator as a skip label.
    """
    n = len(obs)
    if multi_label:
        raw = obs[label_col].astype(str)
        def one_token(s: str) -> str:
            if s in ("None", "nan", ""):
                return DROP
            toks = re.findall(r"'([^']+)'", s)
            if not toks:
                return s  # already a plain label
            return toks[0] if len(toks) == 1 else DROP
        labels = raw.map(one_token).to_numpy()
    else:
        labels = obs[label_col].astype(str).to_numpy()
        labels = np.where(np.isin(labels, ("nan", "None", "")), DROP, labels)
    stats: dict = {"rows_total": int(n)}
    if control_rule:
        col, prefixes, out_label = control_rule
        if col not in obs.columns:
            raise SystemExit(f"control rule column {col!r} not in obs; columns: {list(obs.columns)[:20]}")
        mask = obs[col].astype(str).str.startswith(tuple(prefixes)).to_numpy()
        labels = np.where(mask, out_label, labels)
        stats["control_cells"] = int(mask.sum())
        if not mask.any():
            raise SystemExit(f"control rule matched 0 rows: {col} startswith {prefixes}")
    for col, value in (row_filters or {}).items():
        if col not in obs.columns:
            raise SystemExit(f"row filter column {col!r} not in obs; columns: {list(obs.columns)[:20]}")
        keep_mask = (obs[col].astype(str) == value).to_numpy()
        labels = np.where(keep_mask, labels, DROP)
        stats[f"rows_kept_{col}={value}"] = int(keep_mask.sum())
    stats["rows_dropped"] = int((labels == DROP).sum())
    return labels, stats


def stream_lamindb_pseudobulk(
    instance: str,
    key_prefix: str,
    label_col: str,
    *,
    keep: set[str] | None = None,
    gene_col: str | None = None,
    block_rows: int | None = None,
    align_check: bool = True,
    progress: bool = True,
    multi_label: bool = False,
    control_rule: tuple[str, tuple[str, ...], str] | None = None,
    row_filters: dict[str, str] | None = None,
    drop_labels: set[str] | None = None,
    x_path: str = "X",
):
    """One remote pass over the corpus at `key_prefix`.

    Two layouts, told apart by whether `key_prefix` ends in `.h5ad`:

      * a pertdata-style DIRECTORY (`scperturb/frangieh21`) -- labels come from
        the `obs.parquet` sidecar, the matrix from `<prefix>/X.h5ad`, and the
        sidecar's row alignment is asserted against the h5ad's own obs index.
      * a single H5AD (`mcfaline23_gxe_processed.h5ad`, perturbench) -- labels
        come from the file's EMBEDDED obs; there is no sidecar to misalign.

    `x_path` picks the matrix within the file: perturbench's processed h5ads
    hold a NORMALISED X and keep raw counts in `layers/counts` -- streaming
    their X would put log-normalised values through arithmetic that assumes
    raw counts, silently.

    Returns `(PseudobulkSums, lineage_dict)`.
    """
    import lamindb as ln

    db = ln.DB(instance)
    embedded = key_prefix.endswith(".h5ad")
    obs_art = None if embedded else _artifact(db, f"{key_prefix}/obs.parquet")
    x_art = _artifact(db, key_prefix if embedded else f"{key_prefix}/X.h5ad")

    acc = x_art.open()
    f = acc.storage  # open h5py.File over the object store

    if embedded:
        obs = read_elem(f["obs"])
    else:
        # Not a context manager: `.open()` on parquet returns a pyarrow dataset directly.
        obs = obs_art.open().to_table().to_pandas()
    if label_col not in obs.columns:
        raise SystemExit(f"{label_col!r} not in obs; columns: {list(obs.columns)[:20]}")
    labels_all, label_stats = _derive_labels(
        obs, label_col, multi_label=multi_label, control_rule=control_rule,
        row_filters=row_filters)
    if drop_labels:
        labels_all = np.where(np.isin(labels_all, sorted(drop_labels)), DROP, labels_all)
        label_stats["drop_labels"] = sorted(drop_labels)

    xnode = f[x_path]
    n_rows = int(xnode.attrs["shape"][0]) if "shape" in getattr(xnode, "attrs", {}) \
        else int(xnode.shape[0])
    if len(labels_all) != n_rows:
        raise SystemExit(f"obs has {len(labels_all)} rows, {x_path} has {n_rows} -- not the same cells")
    if embedded:
        pass  # obs and X live in one file; there is no join to get wrong
    elif align_check:
        idx_attr = f["obs"].attrs.get("_index", "_index")
        h5_index = np.asarray(read_elem(f["obs"][idx_attr])).astype(str)
        sidecar_index = obs.index.astype(str).to_numpy()
        n_mismatch = int((h5_index != sidecar_index).sum())
        if n_mismatch:
            raise SystemExit(
                f"obs.parquet index disagrees with X.h5ad obs index on {n_mismatch} of "
                f"{n_rows} rows -- refusing to relabel cells across a misjoin")
    else:
        print("ALIGN CHECK SKIPPED -- labels are trusted on row order alone", flush=True)

    genes = None
    gene_meta: dict = {}
    if gene_col:
        var = read_elem(f["var"])
        if gene_col not in var.columns:
            raise SystemExit(f"--gene-col {gene_col!r} not in var; columns: {list(var.columns)}")
        sym = var[gene_col].astype(str).replace({"nan": "", "None": ""}).to_numpy()
        gene_meta = {"gene_col": gene_col, "genes_total": int(len(sym)),
                     "genes_empty_symbol": int((sym == "").sum()),
                     "genes_duplicated_symbol": int(pd.Index(sym[sym != ""]).duplicated().sum())}
        genes = sym

    pb = stream_pseudobulk_file(
        f, label_col, keep, block_rows=block_rows, progress=progress,
        labels_all=labels_all, genes=genes, skip_labels={DROP}, x_path=x_path,
        source=f"lamin://{instance}/{key_prefix}@{x_art.uid}")

    lineage = {
        "produced": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": {"module": "sidechain.data.stream_lamindb_pseudobulk", "git_sha": _git_sha()},
        "instance": instance,
        "artifacts": {
            key_prefix if embedded else "X.h5ad":
                {"uid": x_art.uid, "hash": x_art.hash, "size": int(x_art.size or 0)},
            **({} if embedded else {"obs.parquet": {
                "uid": obs_art.uid, "hash": obs_art.hash, "size": int(obs_art.size or 0)}}),
        },
        "x_path": x_path,
        "label_col": label_col,
        "multi_label": multi_label,
        "control_rule": list(control_rule) if control_rule else None,
        "row_filters": row_filters or None,
        "label_stats": label_stats,
        "resolution": "per-perturbation (labels x genes); no batch split",
        "cells_total": n_rows,
        "cells_kept": int(pb.n_cells.sum()),
        "labels_kept": len(pb.labels),
        **gene_meta,
    }
    return pb, lineage


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", required=True, help="e.g. laminlabs/pertdata")
    ap.add_argument("--key", required=True, help="key prefix, e.g. scperturb/frangieh21")
    ap.add_argument("--label-col", default="pert_target")
    ap.add_argument("--keep", help="CSV of labels to keep (column target_gene or first column); default all")
    ap.add_argument("--control", help="control label, always kept when --keep is given")
    ap.add_argument("--gene-col", default="symbol",
                    help="var column to use as the gene axis ('' keeps the var index; "
                         "pertdata indexes var by ensembl_id and the pooling matches by symbol)")
    ap.add_argument("--block-rows", type=int)
    ap.add_argument("--skip-align-check", action="store_true",
                    help="do NOT verify obs.parquet row order against X.h5ad's obs index")
    ap.add_argument("--multi-label", action="store_true",
                    help="label col holds a stringified list (pert_target_multi); keep "
                         "single-token rows only, drop combinatorial and empty ones")
    ap.add_argument("--control-rule", metavar="COL~PREFIX[,PREFIX...]=LABEL",
                    help="rows whose COL starts with any PREFIX become control cells "
                         "labelled LABEL, e.g. pert_name~non-targeting,NO_SITE=control")
    ap.add_argument("--row-filter", action="append", default=[], metavar="COL=VALUE",
                    help="keep only rows where obs COL == VALUE (repeatable); e.g. "
                         "perturbation_2=Control to take one condition arm of a "
                         "multi-arm design")
    ap.add_argument("--drop-label", action="append", default=[], metavar="LABEL",
                    help="exclude this perturbation label entirely (repeatable); e.g. "
                         "RcontrolSEL, mcfaline23's unflagged random-control pseudo-target")
    ap.add_argument("--x-path", default="X",
                    help="matrix to stream within the h5ad (default X; layers/counts for "
                         "perturbench's processed files, whose X is log-normalised)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    control_rule = None
    if args.control_rule:
        head, _, out_label = args.control_rule.rpartition("=")
        col, _, prefixes = head.partition("~")
        if not (col and prefixes and out_label):
            ap.error("--control-rule must look like COL~PREFIX[,PREFIX...]=LABEL")
        control_rule = (col, tuple(prefixes.split(",")), out_label)
    row_filters = {}
    for spec in args.row_filter:
        col, _, value = spec.partition("=")
        if not (col and value):
            ap.error("--row-filter must look like COL=VALUE")
        row_filters[col] = value

    keep = None
    if args.keep:
        df = pd.read_csv(args.keep)
        col = "target_gene" if "target_gene" in df.columns else df.columns[0]
        keep = set(df[col].astype(str))
        if args.control:
            keep |= {args.control}
        if control_rule:
            keep |= {control_rule[2]}

    t0 = time.time()
    pb, lineage = stream_lamindb_pseudobulk(
        args.instance, args.key, args.label_col, keep=keep,
        gene_col=args.gene_col or None, block_rows=args.block_rows,
        align_check=not args.skip_align_check, multi_label=args.multi_label,
        control_rule=control_rule, row_filters=row_filters,
        drop_labels=set(args.drop_label) or None, x_path=args.x_path)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    pb.save(out)
    lineage["out"] = str(out)
    lineage["seconds"] = round(time.time() - t0)
    lpath = out.with_suffix("").with_suffix(".lineage.json") if out.suffix == ".npz" \
        else out.with_name(out.name + ".lineage.json")
    lpath.write_text(json.dumps(lineage, indent=1) + "\n")
    print(json.dumps(lineage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
