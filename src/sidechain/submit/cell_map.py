"""A step map on the per-cell channel's |log2FC|: each gene's per-cell change, after the strength, is replaced by
the map's size for the bin its size falls in, its sign kept.

`--cell-lfc-map PATH` on `sidechain.submit.build` and `sidechain.eval.loco`: a JSON file with `lower` (the bins'
lower edges on |log2FC|, strictly increasing from 0) and `size` (the |log2FC| emitted for a change in that bin,
finite, >= 0), one entry per bin; the last bin is open above. A change of exactly 0 stays 0, and the knockdown's
own gene is pinned after the map as before. It acts on the per-cell channel only, so a build needs the two-channel
emission (`--alpha-bulk`, or eval.loco's pooled anchor); the summed-profile channel is untouched. The record
(`cell_lfc_map` in loco's summary and in build's `<stem>.dual.json`) carries the file's sha256 and its bins.
Task: private/tasks/T134.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class CellMap:
    def __init__(self, lower, size, *, path: str | None = None, sha256: str | None = None):
        lower = np.asarray(lower, dtype=np.float64)
        size = np.asarray(size, dtype=np.float64)
        if lower.ndim != 1 or lower.shape != size.shape or len(lower) == 0:
            raise SystemExit("--cell-lfc-map: `lower` and `size` must be lists of the same, non-zero length")
        if lower[0] != 0.0 or not np.isfinite(lower).all() or (np.diff(lower) <= 0).any():
            raise SystemExit("--cell-lfc-map: `lower` must start at 0 and increase strictly, all finite")
        if not np.isfinite(size).all() or (size < 0).any():
            raise SystemExit("--cell-lfc-map: every `size` must be finite and >= 0")
        self.lower, self.size, self.path, self.sha256 = lower, size, path, sha256

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
        return cls(spec["lower"], spec["size"], path=str(path), sha256=hashlib.sha256(raw).hexdigest())

    def apply(self, d: np.ndarray) -> np.ndarray:
        """The mapped vector (a new array); NaN stays NaN, 0 stays 0."""
        d = np.asarray(d, dtype=np.float64)
        a = np.abs(d)
        b = np.searchsorted(self.lower, np.where(np.isfinite(a), a, 0.0), side="right") - 1
        return np.where(np.isfinite(d), np.sign(d) * self.size[b], d)

    def record(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "lower": self.lower.tolist(), "size": self.size.tolist()}
