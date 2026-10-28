"""A step map on the per-cell channel's |log2FC|: each change takes its bin's size, sign kept, times an optional scale.

    python -m sidechain.eval.loco ... --cell-lfc-map MAP.json      # also sidechain.submit.build, two-channel only
    MAP.json = {"lower": [0, ...], "size": [...], "scale": {"gene": {symbol: x}, "target": {knockdown: x} or x,
                "clip": [lo, hi], "gene_default": x, "pair": "PAIRS.parquet"}}    # or "contexts": {"D": scale, ...}

`lower` holds the bins' increasing lower edges from 0 (the last bin is open), `size` each bin's emitted |log2FC|; 0
stays 0 and the target gene is pinned after the map. The pair parquet has `target`, `feature`, `factor`. Reads the map
and its tables; the run's record (`cell_lfc_map`) carries the file's sha256 and bins.
Notes: private/code-notes/src/sidechain/submit/cell_map.py.md
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_pair(path: Path, where: str) -> dict:
    """A pair table: {knockdown: (gene names, factors)}, refused unless every factor is finite and >= 0."""
    if not path.exists():
        raise SystemExit(f"--cell-lfc-map: {where}'s pair table {path} does not exist")
    df = pd.read_parquet(path)
    if not {"target", "feature", "factor"} <= set(df.columns):
        raise SystemExit(f"--cell-lfc-map: {where}'s pair table needs columns target, feature, factor")
    f = df["factor"].to_numpy(dtype=np.float64)
    if not np.isfinite(f).all() or (f < 0).any():
        raise SystemExit(f"--cell-lfc-map: every factor in {where}'s pair table must be finite and >= 0")
    if df.duplicated(["target", "feature"]).any():
        raise SystemExit(f"--cell-lfc-map: {where}'s pair table repeats a (target, feature) pair")
    out = {}
    for t, g in df.groupby("target", sort=False):
        out[str(t)] = (g["feature"].astype(str).to_numpy(), g["factor"].to_numpy(dtype=np.float64))
    return out


def _check_scale(spec, where: str) -> dict:
    if not isinstance(spec, dict) or not ({"gene", "target", "pair"} & set(spec)):
        raise SystemExit(f"--cell-lfc-map: {where} needs `gene`, `target` and/or `pair`")
    unknown = set(spec) - {"gene", "target", "clip", "gene_default", "pair"}
    if unknown:
        raise SystemExit(f"--cell-lfc-map: {where} has unknown keys {sorted(unknown)}")
    vals = list(spec.get("gene", {}).values()) + [spec.get("gene_default", 1.0)]
    t = spec.get("target", 1.0)
    vals += list(t.values()) if isinstance(t, dict) else [t]
    v = np.asarray(vals, dtype=np.float64)
    if not np.isfinite(v).all() or (v < 0).any():
        raise SystemExit(f"--cell-lfc-map: every factor in {where} must be finite and >= 0")
    if "clip" in spec:
        c = spec["clip"]
        if not (isinstance(c, list) and len(c) == 2 and 0 < float(c[0]) <= float(c[1]) and np.isfinite(c[1])):
            raise SystemExit(f"--cell-lfc-map: {where}'s `clip` must be [lo, hi] with 0 < lo <= hi")
    return spec


class CellMap:
    def __init__(self, lower, size, *, path: str | None = None, sha256: str | None = None, scale: dict | None = None,
                 contexts: dict | None = None, base: Path | None = None):
        lower = np.asarray(lower, dtype=np.float64)
        size = np.asarray(size, dtype=np.float64)
        if lower.ndim != 1 or lower.shape != size.shape or len(lower) == 0:
            raise SystemExit("--cell-lfc-map: `lower` and `size` must be lists of the same, non-zero length")
        if lower[0] != 0.0 or not np.isfinite(lower).all() or (np.diff(lower) <= 0).any():
            raise SystemExit("--cell-lfc-map: `lower` must start at 0 and increase strictly, all finite")
        if not np.isfinite(size).all() or (size < 0).any():
            raise SystemExit("--cell-lfc-map: every `size` must be finite and >= 0")
        if scale is not None and contexts is not None:
            raise SystemExit("--cell-lfc-map: give `scale` or `contexts`, not both")
        self.lower, self.size, self.path, self.sha256 = lower, size, path, sha256
        self.scale = None if scale is None else _check_scale(scale, "`scale`")
        self.contexts = (None if contexts is None else
                         {str(c): _check_scale(s, f"context {c}'s scale") for c, s in contexts.items()})
        self.base = Path(base) if base is not None else (Path(path).parent if path else Path("."))
        self._pairs, self._pair_sha = {}, {}           # pair path -> table, sha256; read once, before any work
        for where, spec in ([("`scale`", self.scale)] if self.scale is not None else []) + \
                [(f"context {c}'s scale", s) for c, s in (self.contexts or {}).items()]:
            if "pair" in spec and spec["pair"] not in self._pairs:
                pp = Path(spec["pair"]).expanduser()
                pp = pp if pp.is_absolute() else self.base / pp
                self._pairs[spec["pair"]] = _load_pair(pp, where)
                self._pair_sha[spec["pair"]] = hashlib.sha256(pp.read_bytes()).hexdigest()
        self._g = self._t = self._clip = self._p = None          # set by bind

    @classmethod
    def load(cls, path) -> "CellMap":
        path = Path(path).expanduser()
        if not path.exists():
            raise SystemExit(f"--cell-lfc-map: {path} does not exist")
        raw = path.read_bytes()
        try:
            spec = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--cell-lfc-map: {path} is not JSON ({e})") from None
        if not isinstance(spec, dict) or "lower" not in spec or "size" not in spec:
            raise SystemExit(f"--cell-lfc-map: {path} needs `lower` and `size`")
        return cls(spec["lower"], spec["size"], path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                   scale=spec.get("scale"), contexts=spec.get("contexts"), base=path.parent)

    @property
    def scaled(self) -> bool:
        return self.scale is not None or self.contexts is not None

    def context_key(self, context: str | None):
        """What a context's mapping depends on: contexts with the same key can share one pooled map."""
        if self.contexts is None:
            return None
        if context not in self.contexts:
            raise SystemExit(f"--cell-lfc-map: no scale for context {context!r} (the map has {sorted(self.contexts)})")
        return json.dumps(self.contexts[context], sort_keys=True)

    def bind(self, genes, targets, context: str | None = None) -> "CellMap":
        """A copy ready to apply on this gene axis and panel (the map itself when unscaled)."""
        if not self.scaled:
            return self
        if self.contexts is not None:
            if context is None:
                raise SystemExit("--cell-lfc-map: this map holds one scale per context; name the context")
            self.context_key(context)
            spec = self.contexts[context]
        else:
            spec = self.scale
        genes = np.asarray(genes).astype(str)
        table = spec.get("gene", {})
        g = np.array([float(table.get(x, spec.get("gene_default", 1.0))) for x in genes])
        if table and sum(x in table for x in genes) < 0.5 * len(genes):
            raise SystemExit(f"--cell-lfc-map: the gene table matches {sum(x in table for x in genes)} of the "
                             f"{len(genes)} axis genes (another gene space?)")
        t = spec.get("target", 1.0)
        if isinstance(t, dict):
            missing = sorted(set(map(str, targets)) - set(t))
            if missing:
                raise SystemExit(f"--cell-lfc-map: no target factor for {len(missing)} knockdowns "
                                 f"({', '.join(missing[:5])}...)")
            t = {k: float(v) for k, v in t.items()}
        out = CellMap.__new__(CellMap)
        out.__dict__.update(self.__dict__)
        out._g, out._t, out._p = g, t, None
        if "pair" in spec:
            gpos = {x: i for i, x in enumerate(genes)}
            out._p = {}
            for k in map(str, targets):
                v = np.ones(len(genes))
                if k in self._pairs[spec["pair"]]:
                    names, fac = self._pairs[spec["pair"]][k]
                    hit = np.array([x in gpos for x in names], dtype=bool)
                    v[[gpos[x] for x in names[hit]]] = fac[hit]
                out._p[k] = v
            tab = self._pairs[spec["pair"]]
            n_all = sum(len(tab[k][0]) for k in map(str, targets) if k in tab)
            n_hit = sum(sum(x in gpos for x in tab[k][0]) for k in map(str, targets) if k in tab)
            if n_all and n_hit < 0.5 * n_all:
                raise SystemExit(f"--cell-lfc-map: the pair table matches {n_hit} of its {n_all} pairs on this axis "
                                 f"(another gene space?)")
        out._clip = tuple(map(float, spec["clip"])) if "clip" in spec else None
        out._bound_context = context
        return out

    def apply(self, d: np.ndarray, target: str | None = None) -> np.ndarray:
        """The mapped vector (a new array); NaN stays NaN, 0 stays 0."""
        d = np.asarray(d, dtype=np.float64)
        a = np.abs(d)
        b = np.searchsorted(self.lower, np.where(np.isfinite(a), a, 0.0), side="right") - 1
        s = self.size[b]
        if self.scaled:
            if self._g is None:
                raise SystemExit("--cell-lfc-map: a scaled map must be bound to the axis and panel first")
            if len(self._g) != len(d):
                raise SystemExit("--cell-lfc-map: the vector is not on the axis the map was bound to")
            tf = self._t if not isinstance(self._t, dict) else self._t[str(target)]
            f = self._g * tf
            if self._p is not None:
                if str(target) not in self._p:
                    raise SystemExit(f"--cell-lfc-map: knockdown {target!r} is not in the panel the map was bound to")
                f = f * self._p[str(target)]
            if self._clip is not None:
                f = np.clip(f, *self._clip)
            s = s * f
        return np.where(np.isfinite(d), np.sign(d) * s, d)

    def record(self) -> dict:
        rec = {"path": self.path, "sha256": self.sha256, "lower": self.lower.tolist(), "size": self.size.tolist()}
        if self.scaled:
            def summ(spec):
                t = spec.get("target", 1.0)
                out = {"genes": len(spec.get("gene", {})), "gene_default": spec.get("gene_default", 1.0),
                       "targets": len(t) if isinstance(t, dict) else float(t), "clip": spec.get("clip")}
                if "pair" in spec:
                    tab = self._pairs[spec["pair"]]
                    out["pair"] = {"path": spec["pair"], "sha256": self._pair_sha[spec["pair"]],
                                   "targets": len(tab), "pairs": int(sum(len(v[0]) for v in tab.values()))}
                return out
            rec["scale"] = (summ(self.scale) if self.scale is not None else
                            {c: summ(s) for c, s in self.contexts.items()})
        return rec
