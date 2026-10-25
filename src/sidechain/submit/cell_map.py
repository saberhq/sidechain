"""A step map on the per-cell channel's |log2FC|: each gene's per-cell change, after the strength, is replaced by
the map's size for the bin its size falls in, its sign kept, optionally times a per-gene and per-target scale.

`--cell-lfc-map PATH` on `sidechain.submit.build` and `sidechain.eval.loco`: a JSON file with `lower` (the bins'
lower edges on |log2FC|, strictly increasing from 0) and `size` (the |log2FC| emitted for a change in that bin,
finite, >= 0), one entry per bin; the last bin is open above. A change of exactly 0 stays 0, and the knockdown's
own gene is pinned after the map as before. It acts on the per-cell channel only, so a build needs the two-channel
emission (`--alpha-bulk`, or eval.loco's pooled anchor); the summed-profile channel is untouched. The record
(`cell_lfc_map` in loco's summary and in build's `<stem>.dual.json`) carries the file's sha256 and its bins.

An optional `scale` multiplies each mapped size: `{"gene": {symbol: factor}, "target": {knockdown: factor} or one
number, "clip": [lo, hi], "gene_default": factor}` -- the product of the gene's and the knockdown's factors, clipped
to [lo, hi] when `clip` is given (a gene the table lacks takes `gene_default`, 1.0 unless set; every knockdown of the
panel must have a target factor when `target` is a table). Or `contexts`, one such scale per challenge context
(`{"D": {...}, "E": {...}}`) for a build over several, the bins shared. A scaled map is bound to the gene axis and
the panel before use (`bind`). Task: private/tasks/T134.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _check_scale(spec, where: str) -> dict:
    if not isinstance(spec, dict) or not ({"gene", "target"} & set(spec)):
        raise SystemExit(f"--cell-lfc-map: {where} needs `gene` and/or `target`")
    unknown = set(spec) - {"gene", "target", "clip", "gene_default"}
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
                 contexts: dict | None = None):
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
        self._g = self._t = self._clip = None          # set by bind

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
                   scale=spec.get("scale"), contexts=spec.get("contexts"))

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
        out = CellMap(self.lower, self.size, path=self.path, sha256=self.sha256, scale=self.scale,
                      contexts=self.contexts)
        out._g, out._t = g, t
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
            if self._clip is not None:
                f = np.clip(f, *self._clip)
            s = s * f
        return np.where(np.isfinite(d), np.sign(d) * s, d)

    def record(self) -> dict:
        rec = {"path": self.path, "sha256": self.sha256, "lower": self.lower.tolist(), "size": self.size.tolist()}
        if self.scaled:
            def summ(spec):
                t = spec.get("target", 1.0)
                return {"genes": len(spec.get("gene", {})), "gene_default": spec.get("gene_default", 1.0),
                        "targets": len(t) if isinstance(t, dict) else float(t), "clip": spec.get("clip")}
            rec["scale"] = (summ(self.scale) if self.scale is not None else
                            {c: summ(s) for c, s in self.contexts.items()})
        return rec
