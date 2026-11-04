"""Emit a context's blocks over worker processes, each block equal to the serial emitter's.

    with EmitPool(em, 8, on_fail="fallback", shape=True) as pool:
        for block in pool.blocks(calls): ...     # calls: ("emit", n, d) or ("dual", n, d_cell, d_bulk, kw)

Reads and writes nothing. Notes: private/code-notes/src/sidechain/models/emit_pool.py.md
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp

import numpy as np

ATTRS = ("last_dual", "last_dual_reason", "last_dual_keep", "last_dual_clamp", "last_sharpened",
         "last_shaped", "last_shape_held", "last_shape_unmet")
COUNTERS = ("dual_fallbacks", "dual_fallbacks_anchor", "dual_fallbacks_shrink")

_W: dict = {}


def _one(task):
    k, state, n, d_cell, d_bulk, kw = task
    em = _W["em"]
    before = {c: getattr(em, c, 0) for c in COUNTERS}
    em.rng.bit_generator.state = state
    block = em.emit_dual(n, d_cell, d_bulk, **_W["kw"], **kw)
    return (k, block, {a: getattr(em, a, None) for a in ATTRS},
            {c: getattr(em, c, 0) - before[c] for c in COUNTERS}, em.rng.bit_generator.state)


def _digest(calls) -> str:
    h = hashlib.blake2b(digest_size=16)
    for c in calls:
        h.update(f"{c[0]}:{c[1]}:".encode())
        d = c[2]
        h.update(b"-" if d is None else np.ascontiguousarray(d, dtype=np.float64).tobytes())
    return h.hexdigest()


class EmitPool:
    """`emit_dual` calls run in `workers` forked processes; the parent advances the emitter's stream in call order."""

    def __init__(self, em, workers: int, **dual_kw):
        if workers < 2:
            raise ValueError(f"EmitPool needs 2 or more workers, got {workers}")
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("EmitPool needs the fork start method")
        self.em, self.workers, self.kw = em, int(workers), dual_kw
        self._walk: tuple[str, list] | None = None
        self._pool = None

    def __enter__(self):
        _W.update(em=self.em, kw=self.kw)
        try:
            self._pool = mp.get_context("fork").Pool(self.workers)
        finally:
            _W.clear()
        return self

    def __exit__(self, *exc):
        self._pool.terminate()
        self._pool.join()
        return False

    def walk(self, em, calls) -> list[dict]:
        """The stream state before each call and after the last, from `em`'s current state."""
        start = em.rng.bit_generator.state
        key = json.dumps(start, sort_keys=True) + _digest(calls)
        if self._walk is not None and self._walk[0] == key:
            return self._walk[1]
        states = []
        for c in calls:
            states.append(em.rng.bit_generator.state)
            em.emit(c[1], c[2])
            if c[0] == "dual":
                em.rng.integers(0, 2**32 - 1)
        states.append(em.rng.bit_generator.state)
        em.rng.bit_generator.state = start
        self._walk = (key, states)
        return states

    def blocks(self, calls, em=None):
        """Yield each call's block in order; `em` (default the pool's) carries the attributes and counters after each."""
        em = self.em if em is None else em
        calls = list(calls)
        states = self.walk(em, calls)
        tasks = ((k, states[k], c[1], c[2], c[3], c[4]) for k, c in enumerate(calls) if c[0] == "dual")
        results = self._pool.imap(_one, tasks)
        for k, c in enumerate(calls):
            if c[0] == "emit":
                em.rng.bit_generator.state = states[k]
                block = em.emit(c[1], c[2])
                end = em.rng.bit_generator.state
            else:
                kk, block, attrs, deltas, end = next(results)
                if kk != k:
                    raise RuntimeError(f"EmitPool: block {kk} returned for call {k}")
                for a, v in attrs.items():
                    setattr(em, a, v)
                for name, v in deltas.items():
                    if v:
                        setattr(em, name, getattr(em, name, 0) + v)
            if end != states[k + 1]:
                raise RuntimeError(f"EmitPool: call {k}'s stream ended off the serial walk")
            yield block
        em.rng.bit_generator.state = states[-1]
