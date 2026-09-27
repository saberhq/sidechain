"""Stream a published per-(target, condition) DE table out of a remote AnnData.

The second table-type source after Feng, and the first that ships a real standard
error. GWCD4i (Zhu, Dann ... Marson, Cell 2026; genome-scale CRISPRi in primary
human CD4+ T cells) publishes `GWCD4i.DE_stats.h5ad`: one row per (knocked-down
gene, culture condition) -- Rest, Stim8hr, Stim48hr -- one column per measured
gene, and layers `log_fc`, `lfcSE`, `adj_p_value`, `baseMean`, ... It is 16.79 GB
and it never lands. Its layers are contiguous and uncompressed, so the rows we want
are plain HTTP range reads: all 291 challenge targets' `log_fc` is 88.6 MB of it
(measured 2026-09-21, `runs/gwcd4i_range_read_2026-09-21/`).

TWO ARTIFACTS, ON PURPOSE
-------------------------
1. `DEConditions` -- the rows we read, per condition, (C, T, G): effect, SE, adjusted
   p, plus every per-row quality flag the publisher ships. This is the network
   cost, paid once. It keeps the conditions SEPARATE because how to combine them
   is a modelling choice with a measurable answer, and baking one choice into the
   only copy would make the others cost another pull.
2. `LfcTable` views -- one per (conditions, correlation, quality gate), rebuilt from
   (1) in seconds. This is what `--lfc-source` consumes.

THE VARIANCE IS THE WHOLE GAME
------------------------------
A table-type source's variance BYPASSES the pool's Poisson floor, so an
over-confident one wins every gene it touches. Feng's variance, derived by
inverting p-values, measured 9-17x over-confident against held-out pseudobulk
(2026-08-31). GWCD4i ships `lfcSE`, so `var = lfcSE^2`
with no inversion; its median is 0.018, between our HEK293T (0.014) and HCT116
(0.027) sources, so it is not claiming absurd certainty on scale. Two things can
still make it wrong, and both are parameters here rather than decisions:

* **Combining conditions.** The three conditions are the same four donors and the
  same guide library in different cultures, so their errors are positively
  correlated. An inverse-variance mean that assumes independence reports
  var = 1 / sum(1/v_c), up to 3x too small -- the Feng failure by another door.
  `rho` states the assumed error correlation; the combined variance is exact for it.
  It is MEASURED for GWCD4i, not guessed: donor-split deviations correlate across
  conditions at median r = 0.10 (40 targets, 360 comparisons, 2026-09-25, from the
  publisher's by_donors fits); it was not measured separately at responding genes.

      w_c = (1/v_c) / sum_d (1/v_d)
      var = (1 - rho) * sum_c w_c^2 v_c  +  rho * (sum_c w_c sqrt(v_c))^2

  rho = 0 is independent IVW; rho = 1 gives no reduction for equal variances (and
  slightly MORE than the best single arm for unequal ones -- conservative). A single-condition
  view (`conditions=["Rest"]`) sidesteps the question.
* **Rows whose knockdown did not take.** 37.6 % of all 33,983 published rows have
  `ontarget_significant = False` (21.8 % of the rows kept for our 615 fold and
  challenge targets, which lean toward expressed genes). Such a row is evidence about a guide, not about the
  gene: it votes "little change" on a gene whose knockdown elsewhere does change
  things. `gate` names the flags that make a row ABSTAIN (variance inf, weight 0),
  never vote toward zero -- the same reasoning Feng's saturated rows abstain on.

Which setting is honest is measured, not argued: fit each view against held-out
pseudobulk truth with `sidechain.eval.transfer_tau2` and read its calibration.

    uv run python -m sidechain.data.stream_de_h5ad pull --keep targets.json \\
        --out ~/data/sidechain/derived/gwcd4i/gwcd4i_de_conditions.npz
    uv run python -m sidechain.data.stream_de_h5ad table \\
        --de ~/data/sidechain/derived/gwcd4i/gwcd4i_de_conditions.npz \\
        --conditions Rest --gate knockdown --out .../gwcd4i_rest_kd.npz
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sidechain.data.lfc_table import LfcTable

GWCD4I_URL = ("https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/"
              "marson2025_data/GWCD4i.DE_stats.h5ad")

# The quality gates a view may apply. A row that FAILS any listed check abstains.
# Each entry maps a gate name to (obs column, the value that means "fails").
GATES: dict[str, tuple[tuple[str, bool], ...]] = {
    # nothing abstains: every published row votes
    "none": (),
    # the knockdown must have taken, must not hit a distal gene, and the target must
    # be expressed enough in CD4 cells for a knockdown to mean anything
    "knockdown": (("ontarget_significant", False),
                  ("distal_offtarget_flag", True),
                  ("low_target_gex", True)),
    # ...and the row must rest on two guides. A single-guide row has no guide
    # replication at all, and guide heterogeneity is a large error lfcSE
    # omits: the publisher's guide-split fits put it at 2.46x in variance at
    # responding genes (measured 2026-09-25 from GWCD4i.DE_stats.by_guide.h5mu).
    "knockdown_2guide": (("ontarget_significant", False),
                         ("distal_offtarget_flag", True),
                         ("low_target_gex", True),
                         ("single_guide_estimate", True)),
}
# obs columns read and kept, whether or not a gate uses them
FLAG_COLS = ("ontarget_significant", "distal_offtarget_flag", "low_target_gex",
             "single_guide_estimate")
NUM_COLS = ("n_cells_target", "n_guides", "ontarget_effect_size")


# ------------------------------------------------------------------ the container

@dataclass
class DEConditions:
    """What we read, per condition. NaN wherever a (condition, target) row is absent."""

    targets: list[str]                 # (T,) sorted
    conditions: list[str]              # (C,)
    genes: np.ndarray                  # (G,) readout symbols
    lfc: np.ndarray                    # (C, T, G) float32
    se: np.ndarray                     # (C, T, G) float32
    padj: np.ndarray                   # (C, T, G) float32
    present: np.ndarray                # (C, T) bool -- the publisher has this row
    flags: dict[str, np.ndarray]       # name -> (C, T) bool
    nums: dict[str, np.ndarray]        # name -> (C, T) float64
    source: str = ""
    notes: dict = field(default_factory=dict)

    def usable(self, gate: str) -> np.ndarray:
        """(C, T): present, and passing every check `gate` names."""
        if gate not in GATES:
            raise KeyError(f"unknown gate {gate!r}; have {sorted(GATES)}")
        ok = self.present.copy()
        for col, fails_when in GATES[gate]:
            if col not in self.flags:
                raise KeyError(f"gate {gate!r} needs flag {col!r}, which was not read")
            ok &= self.flags[col] != fails_when
        return ok

    def save(self, path: str | Path) -> None:
        arrays = {
            "targets": np.asarray(self.targets, dtype=object),
            "conditions": np.asarray(self.conditions, dtype=object),
            "genes": np.asarray(self.genes, dtype=object),
            "lfc": self.lfc, "se": self.se, "padj": self.padj, "present": self.present,
            "source": np.asarray([self.source], dtype=object),
            "notes": np.asarray([json.dumps(self.notes)], dtype=object),
        }
        for k, v in self.flags.items():
            arrays[f"flag__{k}"] = v
        for k, v in self.nums.items():
            arrays[f"num__{k}"] = v
        np.savez_compressed(Path(path).expanduser(), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> DEConditions:
        z = np.load(Path(path).expanduser(), allow_pickle=True)
        return cls(
            targets=[str(x) for x in z["targets"]],
            conditions=[str(x) for x in z["conditions"]],
            genes=z["genes"].astype(str),
            lfc=z["lfc"], se=z["se"], padj=z["padj"], present=z["present"],
            flags={k[6:]: z[k] for k in z.files if k.startswith("flag__")},
            nums={k[5:]: z[k] for k in z.files if k.startswith("num__")},
            source=str(z["source"][0]), notes=json.loads(str(z["notes"][0])),
        )


# ------------------------------------------------------------------ reading

def _obs_col(h, col: str) -> np.ndarray:
    """One obs column from an AnnData h5: a categorical group, or a plain dataset."""
    node = h[f"obs/{col}"]
    if hasattr(node, "keys") and "categories" in node and "codes" in node:
        cats = node["categories"]
        cats = np.asarray(cats.asstr()[:] if cats.dtype.kind in "OS" else cats[:])
        codes = node["codes"][:]
        out = np.empty(codes.shape, dtype=object)
        valid = codes >= 0
        out[valid] = cats[codes[valid]]
        out[~valid] = None
        return out
    if node.dtype.kind in "OS":
        return np.asarray(node.asstr()[:], dtype=object)
    return node[:]


def _runs(rows: np.ndarray) -> list[tuple[int, int]]:
    """Coalesce sorted row indices into [start, stop) runs -- one range read each."""
    if rows.size == 0:
        return []
    out, start, prev = [], int(rows[0]), int(rows[0])
    for r in rows[1:]:
        r = int(r)
        if r == prev + 1:
            prev = r
        else:
            out.append((start, prev + 1))
            start = prev = r
    out.append((start, prev + 1))
    return out


def read_de_h5ad(h, *, keep: set[str], target_col: str = "target_contrast_gene_name",
                 condition_col: str = "culture_condition", gene_col: str = "gene_name",
                 effect_layer: str = "log_fc", se_layer: str = "lfcSE",
                 padj_layer: str = "adj_p_value", source: str = "") -> DEConditions:
    """Read the rows whose target is in `keep` out of an open AnnData h5 file.

    `h` is an `h5py.File` -- over fsspec for the remote object, or a local file in
    tests. Only the selected rows of the three layers are read, as contiguous runs.
    """
    tg = _obs_col(h, target_col).astype(str)
    cond = _obs_col(h, condition_col).astype(str)
    gnode = h[f"var/{gene_col}"]
    genes = (_obs_col_var(h, gene_col) if hasattr(gnode, "keys")
             else np.asarray(gnode.asstr()[:] if gnode.dtype.kind in "OS" else gnode[:]))
    genes = np.asarray(genes).astype(str)

    sel = np.isin(tg, np.asarray(sorted(keep), dtype=object).astype(str))
    rows = np.flatnonzero(sel)
    targets = sorted(set(tg[rows]))
    conditions = sorted(set(cond[rows]))
    ti = {t: i for i, t in enumerate(targets)}
    ci = {c: i for i, c in enumerate(conditions)}
    C, T, G = len(conditions), len(targets), genes.size

    lfc = np.full((C, T, G), np.nan, dtype=np.float32)
    se = np.full((C, T, G), np.nan, dtype=np.float32)
    padj = np.full((C, T, G), np.nan, dtype=np.float32)
    present = np.zeros((C, T), dtype=bool)

    # Checked BEFORE any layer is read: on the remote object the reads are the cost, and
    # a second row for a slot would silently overwrite the first.
    for r in rows:
        if present[ci[cond[r]], ti[tg[r]]]:
            raise ValueError(f"two rows for ({tg[r]}, {cond[r]}); the table is not one "
                             "row per (target, condition) as this reader assumes")
        present[ci[cond[r]], ti[tg[r]]] = True
    if len(set(genes)) != genes.size:
        dup = sorted({g for g in genes if (genes == g).sum() > 1})[:5]
        raise ValueError(f"duplicate gene names on the readout axis, e.g. {dup}; a name "
                         "must identify one column")

    t0 = time.time()
    runs = _runs(rows)
    for layer, dest in ((effect_layer, lfc), (se_layer, se), (padj_layer, padj)):
        ds = h[f"layers/{layer}"]
        if ds.shape[1] != G:
            raise ValueError(f"layer {layer} has {ds.shape[1]} columns, var has {G}")
        for a, b in runs:
            block = ds[a:b, :]
            for k, r in enumerate(range(a, b)):
                dest[ci[cond[r]], ti[tg[r]]] = block[k]

    def per_row(values, dtype, fill):
        out = np.full((C, T), fill, dtype=dtype)
        for r in rows:
            out[ci[cond[r]], ti[tg[r]]] = values[r]
        return out

    flags, nums = {}, {}
    obs_cols = set(h["obs"].keys())
    for col in FLAG_COLS:
        if col in obs_cols:
            flags[col] = per_row(_as_bool(_obs_col(h, col), col), bool, False)
    for col in NUM_COLS:
        if col in obs_cols:
            v = np.asarray(_obs_col(h, col), dtype=np.float64)
            nums[col] = per_row(v, np.float64, np.nan)

    return DEConditions(
        targets=targets, conditions=conditions, genes=genes, lfc=lfc, se=se, padj=padj,
        present=present, flags=flags, nums=nums, source=source,
        notes={"rows_read": int(rows.size), "range_runs": len(runs),
               "seconds": round(time.time() - t0, 1),
               "keep_requested": len(keep), "keep_found": T,
               "keep_absent": sorted(set(keep) - set(targets))[:50]},
    )


def _obs_col_var(h, col: str) -> np.ndarray:
    """A categorical column under var/ (same encoding as obs).

    A code of -1 is a missing value. Indexing categories with it silently returns the LAST
    category, which would give an unnamed gene someone else's symbol -- so it is refused.
    """
    node = h[f"var/{col}"]
    cats = node["categories"]
    cats = np.asarray(cats.asstr()[:] if cats.dtype.kind in "OS" else cats[:])
    codes = node["codes"][:]
    if (codes < 0).any():
        raise ValueError(f"var/{col} has {int((codes < 0).sum())} missing value(s); "
                         "a gene with no name cannot be placed on an axis")
    return cats[codes]


def _as_bool(values, col: str) -> np.ndarray:
    """A flag column as booleans, refusing anything that only LOOKS boolean.

    `astype(bool)` on the strings "True"/"False" makes every one True -- a gate built on it
    would pass or fail every row alike and say nothing. Real bools pass through; a
    categorical of exactly those spellings is mapped; anything else is an error.
    """
    arr = np.asarray(values)
    if arr.dtype == bool:
        return arr
    if arr.dtype.kind in "OSU":
        vals = {str(x) for x in arr}
        if vals <= {"True", "False", "true", "false"}:
            return np.array([str(x).lower() == "true" for x in arr], dtype=bool)
    raise ValueError(f"flag {col!r} is {arr.dtype}, not boolean; refusing to guess its meaning")


# ------------------------------------------------------------------ combining

def combine(de: DEConditions, *, conditions: list[str] | None = None,
            rho: float | None = None, gate: str = "knockdown") -> LfcTable:
    """One LfcTable view: pick conditions, combine them, apply a quality gate.

    `conditions=None` means all of them. `rho` is the assumed correlation of the
    conditions' errors (see the module docstring). It has NO DEFAULT when more than one
    condition is combined: rho = 0 is the textbook independence assumption and also the
    over-confident one here, so a caller must say which they mean rather than inherit it.
    A single-condition view needs none. A (target, gene) with no usable condition
    abstains (variance inf) -- it does not vote zero.
    """
    names = de.conditions if conditions is None else list(conditions)
    if rho is None:
        if len(names) > 1:
            raise ValueError(f"combining {len(names)} conditions needs an explicit rho: "
                             "0 assumes independent errors, which these are not")
        rho = 0.0
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho is a correlation in [0, 1], got {rho}")
    missing = [c for c in names if c not in de.conditions]
    if missing:
        raise KeyError(f"conditions {missing} not in {de.conditions}")
    idx = [de.conditions.index(c) for c in names]

    usable_rows = de.usable(gate)[idx]                        # (c, T)
    x = de.lfc[idx].astype(np.float64)                        # (c, T, G)
    v = de.se[idx].astype(np.float64) ** 2
    ok = (usable_rows[:, :, None] & np.isfinite(x) & np.isfinite(v) & (v > 0))

    inv = np.where(ok, 1.0 / np.where(ok, v, 1.0), 0.0)       # 1/v, 0 where unusable
    s_inv = inv.sum(axis=0)                                   # (T, G)
    any_ok = s_inv > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(any_ok[None], inv / np.where(any_ok, s_inv, 1.0)[None], 0.0)
    xs = np.where(ok, x, 0.0)
    fc = (w * xs).sum(axis=0)

    vs = np.where(ok, v, 0.0)
    indep = (w * w * vs).sum(axis=0)                          # = 1 / sum(1/v)
    corr = ((w * np.sqrt(vs)).sum(axis=0)) ** 2
    var = (1.0 - rho) * indep + rho * corr
    var = np.where(any_ok & (var > 0), var, np.inf)
    fc = np.where(any_ok, fc, 0.0)

    view = f"{'+'.join(names)}|rho={rho:g}|gate={gate}"
    return LfcTable(
        labels=list(de.targets), genes=np.asarray(de.genes), lfc=fc, var=var,
        source=f"{de.source} [{view}]", context="cd4_tcell",
        notes={"conditions": names, "rho": rho, "gate": gate,
               "targets_with_any_usable_row": int(usable_rows.any(axis=0).sum()),
               "targets_abstaining_entirely": int((~any_ok).all(axis=1).sum())},
    )


# ------------------------------------------------------------------ CLI

def _read_keep(path: Path) -> set[str]:
    path = Path(path).expanduser()
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        if isinstance(d, dict):
            for k in ("union_covered_any", "union_targets", "targets"):
                if k in d:
                    return {str(x) for x in d[k]}
            raise KeyError(f"{path}: no target list under a known key")
        return {str(x) for x in d}
    import csv
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    return {r[0].strip() for r in rows[1:] if r} - {"non-targeting", "control", ""}


def _git_sha(*modules) -> str:
    """HEAD's short SHA, suffixed `-dirty` when any of these modules differs from it.

    The suffix is the point. A lineage record that names a commit the running code is
    not in -- a module run before it was committed -- claims a reproducibility it does
    not have. The first GWCD4i pull (2026-09-25) did exactly that. Pass every module whose
    code shaped the artifact; this file is always included.
    """
    import subprocess
    files = [Path(__file__).resolve()] + [Path(m).resolve() for m in modules]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=files[0].parent, text=True).strip()
        for f in files:
            dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", f.name],
                                   cwd=f.parent).returncode != 0
            tracked = subprocess.run(["git", "ls-files", "--error-unmatch", f.name],
                                     cwd=f.parent, capture_output=True).returncode == 0
            if dirty or not tracked:
                return sha + "-dirty"
        return sha
    except Exception:                          # pragma: no cover - lineage only
        return "unknown"


# The registry spec keys this reader needs, and the read_de_h5ad argument each feeds.
SPEC_TO_READER = {"target_col": "target_col", "condition_col": "condition_col",
                  "gene_symbol_col": "gene_col", "effect_col": "effect_layer",
                  "se_col": "se_layer", "pvalue_col": "padj_layer"}


def reader_kwargs(spec: dict) -> dict:
    """The block's spec as `read_de_h5ad` keyword arguments -- one source of truth.

    Until 2026-09-26 the reader used its own defaults and never looked at the block, so the
    two could drift apart with nothing failing. Every key is required: a missing one is a
    block that does not describe this file, and guessing it is how a column gets misread.
    """
    missing = [k for k in SPEC_TO_READER if not spec.get(k)]
    if missing:
        raise KeyError(f"the registry spec lacks {missing}; this reader needs every one of "
                       f"{sorted(SPEC_TO_READER)}")
    return {arg: spec[key] for key, arg in SPEC_TO_READER.items()}


def _head(url: str) -> dict:
    """Content-Length and ETag of the remote object, as S3 reports them right now."""
    import urllib.request
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return {"size_bytes": int(resp.headers.get("Content-Length", -1)),
                "etag": (resp.headers.get("ETag") or "").strip('"')}


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).expanduser().open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions() -> dict:
    import h5py
    out = {"numpy": np.__version__, "h5py": h5py.__version__}
    try:
        import fsspec
        out["fsspec"] = fsspec.__version__
    except ImportError:                        # pragma: no cover
        pass
    return out


def resolve_pull(dataset: str, config: str, root: Path, head=_head) -> dict:
    """Everything a pull needs from the registry and the gate, or a refusal.

    The gate is ENFORCED here, not trusted: no PROVENANCE.json means the gate never ran,
    and a remote object whose size or ETag differs from the recorded one means the file
    changed upstream since it did -- both stop the pull. The URL comes from PROVENANCE,
    never from a default, so what is read is exactly what was admitted.
    """
    from sidechain.data.stream_parquet_pseudobulk import _dataset_block
    from sidechain.ingest.provenance import read_provenance

    block = _dataset_block(dataset, config)
    entry = next((f for f in block["files"] if f.get("kind") == "lfc_table"), None)
    if entry is None:
        raise SystemExit(f"{dataset}: no kind: lfc_table file in the block")
    dest = Path(root) / block["dest"]
    prov = read_provenance(dest)
    if prov is None:
        raise SystemExit(f"no PROVENANCE.json at {dest}: run "
                         f"`python -m sidechain.ingest.fetch --dataset {dataset}` first")
    sel = next((f for f in prov["selected"] if f["name"] == entry["name"]), None)
    if sel is None:
        raise SystemExit(f"{entry['name']} is not in the gate's selection at {dest}")
    now = head(sel["url"])
    recorded_etag = (sel.get("checksum") or "").split(":", 1)[-1]
    if now["size_bytes"] != sel["size_bytes"] or (recorded_etag and now["etag"] != recorded_etag):
        raise SystemExit(f"{sel['url']} changed upstream since the gate ran "
                         f"(size {sel['size_bytes']} -> {now['size_bytes']}, etag "
                         f"{recorded_etag} -> {now['etag']}): re-run the gate before pulling")
    return {"block": block, "entry": entry, "provenance": prov, "selected": sel,
            "provenance_path": str(dest / "PROVENANCE.json"), "head": now,
            "reader_kwargs": reader_kwargs(entry.get("spec") or {})}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pull", help="range-read the kept rows into a DEConditions cache")
    p.add_argument("--dataset", required=True,
                   help="the configs/datasets.yaml block; its gate must have run (PROVENANCE.json)")
    p.add_argument("--config", default="configs/datasets.yaml")
    p.add_argument("--root", type=Path, default=Path.home() / "data" / "sidechain")
    p.add_argument("--keep", type=Path, required=True,
                   help="targets to keep: a JSON list/dict or a CSV whose first column lists them")
    p.add_argument("--out", type=Path, required=True)
    # 64 KiB, not megabytes. fsspec's readahead fetches `block_size` PAST every request,
    # and the rows we want are ~600 scattered runs of ~3 rows (246 KB) each -- an 8 MiB
    # block would pull ~30x the bytes asked for. 64 KiB measured 1.26x overhead on the
    # real file (runs/gwcd4i_range_read_2026-09-21/, 2026-09-21).
    p.add_argument("--block-size", type=int, default=64 << 10)

    t = sub.add_parser("table", help="build an LfcTable view from a DEConditions cache")
    t.add_argument("--de", type=Path, required=True)
    t.add_argument("--conditions", default="all",
                   help="comma list, e.g. Rest or Rest,Stim8hr; 'all' for every condition")
    t.add_argument("--rho", type=float, default=None,
                   help="error correlation between conditions; required when combining several")
    t.add_argument("--gate", default="knockdown", choices=sorted(GATES))
    t.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    # The code is loaded NOW. Asking git at the END of a pull names whatever HEAD has become
    # by then -- a commit made during the run -- not the code that ran (2026-09-26: a pull
    # launched at 4712738 recorded 522caed, committed while it was reading).
    from sidechain.data import lfc_table as _lfc
    code_sha = _git_sha(_lfc.__file__)

    if args.cmd == "pull":
        import fsspec
        import h5py
        r = resolve_pull(args.dataset, args.config, args.root)
        url = r["selected"]["url"]
        keep = _read_keep(args.keep)
        t0 = time.time()
        with fsspec.open(url, block_size=args.block_size, cache_type="readahead") as fo, \
                h5py.File(fo, "r") as h:
            de = read_de_h5ad(h, keep=keep, source=r["entry"]["name"], **r["reader_kwargs"])
        out = args.out.expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        de.save(out)
        lineage = {
            "produced": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "code": {"module": "sidechain.data.stream_de_h5ad", "git_sha": code_sha,
                     "versions": _versions(), "block_size": args.block_size},
            "dataset": r["block"]["name"], "host": r["block"]["host"],
            "record": r["block"]["record"], "license": r["block"].get("license"),
            "provenance": r["provenance_path"],
            "source": {"url": url, "file": r["entry"]["name"],
                       "size_bytes_at_read": r["head"]["size_bytes"],
                       "etag_at_read": r["head"]["etag"]},
            "reader_spec": r["reader_kwargs"],
            "keep_file": str(args.keep), "keep_sha256": _sha256(args.keep),
            "out": str(out), "seconds": round(time.time() - t0, 1),
            "shape": {"conditions": de.conditions, "targets": len(de.targets),
                      "genes": int(de.genes.size)},
            "rows_present_per_condition": {c: int(de.present[i].sum())
                                           for i, c in enumerate(de.conditions)},
            **de.notes,
        }
        out.with_suffix(".lineage.json").write_text(json.dumps(lineage, indent=1))
        print(json.dumps(lineage, indent=1))
        return 0

    de = DEConditions.load(args.de)
    conds = None if args.conditions == "all" else [c.strip() for c in args.conditions.split(",")]
    tab = combine(de, conditions=conds, rho=args.rho, gate=args.gate)
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.save(out)
    out.with_suffix(".lineage.json").write_text(json.dumps({
        "produced": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code": {"module": "sidechain.data.stream_de_h5ad table",
                 "git_sha": code_sha, "versions": _versions()},
        "from": str(Path(args.de).expanduser()), "from_sha256": _sha256(args.de),
        "view": tab.source, **tab.notes}, indent=1, default=str))
    usable = tab.n_usable
    print(json.dumps({"out": str(out), "view": tab.source, **tab.notes,
                      "genes_with_finite_weight": {
                          "min": int(usable.min()), "median": int(np.median(usable)),
                          "max": int(usable.max())},
                      "median_var_finite": float(np.median(tab.var[np.isfinite(tab.var)]))},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
