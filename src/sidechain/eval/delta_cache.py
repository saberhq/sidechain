"""A per-fold cache of pooled deltas for `sidechain.eval.loco`: a pre-pass pools each label once, arms read it.

    python -m sidechain.eval.loco ... --delta-cache DIR --delta-cache-build    # pool every label, write DIR/<key>/
    python -m sidechain.eval.loco ... --delta-cache DIR                        # an arm reads DIR/<key>/

The key hashes each source's sha256 and control label (then each `--lfc-source`), the gene axis, the pooling knobs,
the adaptive rule's constants, the code of each `CODE_MODULES` module (its syntax tree: text edits leave the key),
Python, numpy, BLAS, thread settings and the machine; the fold's
labels are not in it. A read under another key, or of a label the cache lacks, is refused; only the plain pool is
cached. Writes deltas.npy and meta.json under DIR/<key>/. Notes: private/code-notes/src/sidechain/eval/delta_cache.py.md
"""
from __future__ import annotations

import ast
import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import sys
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
# every module a pooled delta's arithmetic passes through: the pool and the garrote, the
# adaptive fit, the per-source log2FC and the axis remap, and the reader of the source files
CODE_MODULES = ("sidechain.submit.build", "sidechain.models.adaptive_shrink",
                "sidechain.models.count_emitters", "sidechain.data.stream_pseudobulk",
                # the variance behind the pooling weight, and the dispersion math its fits carry
                "sidechain.submit.variance_model", "sidechain.data.dispersion")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).expanduser().open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


# the third field of a published-contrast source's id: no control label can be spelled this way
LFC_TABLE = "<lfc-table>"
# read only for a pool that holds a published-contrast source
LFC_MODULE = "sidechain.data.lfc_table"


def source_ids(source_specs: Sequence[str], lfc_paths: Sequence[str] = ()) -> list[list[str]]:
    """`--source NPZ:CONTROL` specs -> [[stem, file sha256, control label], ...], in order, then each
    `--lfc-source NPZ` as [stem, file sha256, LFC_TABLE] (the order the pool holds them in)."""
    out = []
    for spec in source_specs:
        path, _, ctrl = spec.rpartition(":")
        out.append([Path(path).expanduser().stem, file_sha256(path), ctrl or "control"])
    for path in lfc_paths:
        out.append([Path(path).expanduser().stem, file_sha256(path), LFC_TABLE])
    return out


TEXT_KEYWORDS = ("help", "description", "epilog")


class _CodeOnly(ast.NodeTransformer):
    """Drop docstrings and blank help-like keyword text, leaving what the code computes."""

    def _drop_doc(self, node):
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
        return self.generic_visit(node)

    visit_Module = visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _drop_doc

    def visit_keyword(self, node):
        if node.arg in TEXT_KEYWORDS:
            node.value = ast.Constant(value="")
            return node
        return self.generic_visit(node)


def code_sha256(path: str | Path) -> str:
    """sha256 of a module's syntax tree without docstrings, comments, help text or line numbers."""
    tree = _CodeOnly().visit(ast.parse(Path(path).expanduser().read_text(encoding="utf-8")))
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def code_digests(lfc: bool = False) -> dict:
    """`code_sha256` of each `CODE_MODULES` module (and the LfcTable reader's, with `lfc`)."""
    import importlib

    mods = CODE_MODULES + ((LFC_MODULE,) if lfc else ())
    return {m: code_sha256(importlib.import_module(m).__file__) for m in mods}


def machine() -> dict:
    """Where the fits ran: platform, CPU, BLAS and thread pools."""
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    blas = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
    try:
        import threadpoolctl
        pools = [{k: p.get(k) for k in ("internal_api", "version", "num_threads", "architecture")}
                 for p in threadpoolctl.threadpool_info()]
    except ImportError:
        pools = None
    return {"platform": sys.platform, "arch": platform.machine(), "cpu": cpu,
            "cpu_count": os.cpu_count(), "node": platform.node(),
            "blas": {k: blas.get(k) for k in ("name", "version")}, "threadpools": pools}


def key_fields(sources: Sequence[Sequence[str]], axis: np.ndarray, pooling: dict) -> dict:
    """Everything a pooled delta depends on, as one JSON-able dict."""
    from sidechain.models import adaptive_shrink

    return {
        "sources": [list(s) for s in sources],
        "axis_sha256": hashlib.sha256("\n".join(map(str, axis)).encode()).hexdigest(),
        "n_genes": len(axis),
        "pooling": pooling,
        "adaptive": {"max_cycles": adaptive_shrink.MAX_CYCLES,
                     "tol_per_gene": adaptive_shrink.TOL_PER_GENE,
                     "calm_cycles": adaptive_shrink.CALM_CYCLES,
                     "min_genes": adaptive_shrink.MIN_GENES},
        "code": (code_digests(lfc=True) if any(len(s) > 2 and s[2] == LFC_TABLE for s in sources)
                 else code_digests()),
        "python": ".".join(platform.python_version_tuple()[:2]),
        "numpy": np.__version__,
        "threads": {v: os.environ.get(v) for v in THREAD_VARS},
        "machine": machine(),
    }


def key_of(fields: dict) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


class DeltaCache:
    """A built cache, opened for reading: `get(label, stats)` is `pooled_delta` for that label."""

    def __init__(self, root: str | Path, fields: dict):
        self.root = Path(root).expanduser()
        self.key = key_of(fields)
        self.dir = self.root / self.key
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise SystemExit(
                f"--delta-cache: no cache for this fold and these knobs under {self.root} (key "
                f"{self.key[:16]}). Build it first with --delta-cache-build, with the same "
                f"sources, knobs and thread settings; a cache is never filled by an arm.")
        meta = json.loads(meta_path.read_text())
        if meta["fields"] != json.loads(json.dumps(fields)):
            raise SystemExit(f"--delta-cache: {meta_path} was written for other inputs than "
                             f"its key says; rebuild it")
        self.row = meta["row"]
        self.stats = meta["stats"]
        self.deltas = np.load(self.dir / "deltas.npy", mmap_mode="r")
        if self.deltas.shape[1] != fields["n_genes"]:
            raise SystemExit(f"--delta-cache: {self.dir} holds {self.deltas.shape[1]} genes, "
                             f"the fold has {fields['n_genes']}")
        self.hits = 0

    def get(self, label: str, stats: dict | None = None) -> np.ndarray | None:
        if label not in self.row:
            raise SystemExit(f"--delta-cache: {label!r} is not in the cache at {self.dir}: it "
                             f"was built for other labels (another pool file, or another fold "
                             f"file with these sources). Rebuild it with this arm's "
                             f"--neighbour-pool.")
        self.hits += 1
        if stats is not None:
            for k, v in self.stats[label].items():
                stats[k] = stats.get(k, 0) + v
        i = self.row[label]
        return None if i < 0 else np.array(self.deltas[i])

    def record(self) -> dict:
        return {"dir": str(self.dir), "key": self.key, "labels": len(self.row), "hits": self.hits}


_WORK: dict = {}


def _one(label: str):
    stats: dict = {}
    d = _WORK["fn"](label, stats)
    return label, d, stats


def build_cache(root: str | Path, fields: dict, labels: Sequence[str],
                delta_fn: Callable[[str, dict], np.ndarray | None], *, jobs: int = 1) -> dict:
    """Pool every label once and write the cache; `delta_fn(label, stats)` is the caller's
    `pooled_delta`. Returns what was written. An existing cache under the same key is left as
    it is when it holds every label, and refused otherwise."""
    root = Path(root).expanduser()
    key = key_of(fields)
    final = root / key
    labels = list(dict.fromkeys(labels))
    if (final / "meta.json").exists():
        have = json.loads((final / "meta.json").read_text())["row"]
        missing = [lab for lab in labels if lab not in have]
        if missing:
            raise SystemExit(f"--delta-cache-build: {final} exists and lacks {len(missing)} of "
                             f"these labels (e.g. {missing[:3]}). One cache serves every fold "
                             f"file that shares its sources and axis: build the fold with the "
                             f"most labels first, or give this fold its own DIR")
        return {"dir": str(final), "key": key, "labels": len(have), "built": False}
    tmp = root / f"{key}.building.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True)
    n_genes = fields["n_genes"]
    block = np.lib.format.open_memmap(tmp / "deltas.npy", mode="w+", dtype=np.float64,
                                      shape=(len(labels), n_genes))
    row, stats, n = {}, {}, 0
    _WORK["fn"] = delta_fn
    if jobs > 1:
        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            results = list(pool.imap(_one, labels, chunksize=1))
    else:
        results = [_one(lab) for lab in labels]
    _WORK.clear()
    for label, d, st in results:
        bad = {k: v for k, v in st.items() if not isinstance(v, int)}
        if bad:
            raise SystemExit(f"--delta-cache-build: {label!r} wrote a counter the cache cannot "
                             f"replay ({sorted(bad)}): only the plain pool is cached")
        stats[label] = st
        if d is None:
            row[label] = -1
            continue
        d = np.asarray(d)
        if d.dtype != np.float64 or d.shape != (n_genes,):
            raise SystemExit(f"--delta-cache-build: {label!r} pooled to {d.dtype}{d.shape}, "
                             f"expected float64({n_genes},)")
        block[n] = d
        row[label] = n
        n += 1
    block.flush()
    del block
    if n < len(labels):                       # uncovered labels hold no row: cut the file
        full = np.load(tmp / "deltas.npy", mmap_mode="r")
        np.save(tmp / "deltas.cut.npy", np.asarray(full[:n]))
        del full
        os.replace(tmp / "deltas.cut.npy", tmp / "deltas.npy")
    (tmp / "meta.json").write_text(json.dumps(
        {"fields": fields, "row": row, "stats": stats, "covered": n, "jobs": jobs}) + "\n")
    try:
        os.rename(tmp, final)
    except OSError as e:                      # another writer got there first
        have = json.loads((final / "meta.json").read_text())["row"] if (
            final / "meta.json").exists() else {}
        if all(lab in have for lab in labels):
            shutil.rmtree(tmp)
            return {"dir": str(final), "key": key, "labels": len(have), "built": False}
        raise SystemExit(f"--delta-cache-build: {final} appeared while this build ran and "
                         f"lacks some of its labels; this build is left at {tmp}") from e
    return {"dir": str(final), "key": key, "labels": len(labels), "covered": n, "built": True}
