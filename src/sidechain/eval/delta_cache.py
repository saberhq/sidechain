"""A per-fold cache of pooled deltas for `sidechain.eval.loco` (T103 round three).

Every arm of one fold pools the same labels from the same sources with the same knobs: the
fold's targets, and the neighbour pool's members. Under the adaptive shrinkage rule each of those
is a fit of a few seconds, repeated in every arm (and twice inside one arm for a target that is
also a pool member), which made an arm three times as long as a garrote arm. The cache holds
each label's pooled delta once per fold.

- **One pre-pass writes it, arms only read it.** `loco --delta-cache DIR --delta-cache-build`
  pools every label (optionally over several processes), writes into a temporary directory and
  renames it into place. A second writer finds the directory there and stops.
- **The key is everything the delta depends on**: each source file's sha256 and control label,
  in order; the gene axis; every pooling knob; the adaptive rule's constants; the source files
  that hold the arithmetic (`CODE_MODULES`); numpy's version, the BLAS it links, the thread
  settings and the machine. A read under any other key finds no cache and is refused -- there
  is no fallback to pooling, because an arm that silently pooled would look like a cached arm
  and cost three times as much. The machine is in the key because the adaptive fit agrees
  across machines only to about 1e-4: a cache is read where it was built and nowhere else.
- **The fold's labels are not in the key**, so two fold files that share sources and axis (a
  fold and the panel fold cut from it) share one cache: build the larger first, and the
  smaller's build finds every label there. A build that finds a cache lacking some of its
  labels is refused rather than extended.
- **The Python argument trusts its caller**: `delta_cache["sources"]` is what the key reads,
  and `main` derives it from the same `--source` specs the sources are loaded from.
- **The counters are replayed.** `pooled_delta` counts into `stats`; each label's counts are
  stored and added back on a read, so a cached arm's record equals an uncached one's.
- **Only the plain pool is cached**: gamma 1, no similarity weight, no coverage tiers, no
  per-source override, no published-contrast source. Anything else is refused.

Off (no `--delta-cache`) this module is never imported into the path. `submit.build` has no
such flag: a submission is one build and gains nothing from it.
"""
from __future__ import annotations

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
                # T98: the variance behind the pooling weight, and the dispersion math its fits carry
                "sidechain.submit.variance_model", "sidechain.data.dispersion")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).expanduser().open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def source_ids(source_specs: Sequence[str]) -> list[list[str]]:
    """`--source NPZ:CONTROL` specs -> [[stem, file sha256, control label], ...], in order."""
    out = []
    for spec in source_specs:
        path, _, ctrl = spec.rpartition(":")
        out.append([Path(path).expanduser().stem, file_sha256(path), ctrl or "control"])
    return out


def code_digests() -> dict:
    """sha256 of each `CODE_MODULES` file as it is on disk."""
    import importlib

    return {m: file_sha256(importlib.import_module(m).__file__) for m in CODE_MODULES}


def machine() -> dict:
    """Where the fits ran: the adaptive rule is bit-reproducible on one machine only."""
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
        "code": code_digests(),
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
