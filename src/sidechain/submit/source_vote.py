"""Per-source vote weights per context, and per-source strengths, for the pooled delta.

    --source-weight NAME=CTX:W   --source-weight-table PARQUET   --source-scale NAME=S   (submit.build, eval.loco)

Reads the weight table; the callers write `<stem>.source_weight.json` and `<stem>.source_scale.json`.
Notes: private/code-notes/src/sidechain/submit/source_vote.py.md
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd


def _number(text: str, flag: str, part: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise SystemExit(f"{flag}: {part!r}: {text!r} is not a number") from None
    if not math.isfinite(v) or v < 0:
        raise SystemExit(f"{flag}: {part!r}: a weight is finite and >= 0, got {v}")
    return v


def parse_source_weights(specs: list[str] | None) -> dict[tuple[str, str], float]:
    """``['src_x=A:1.5']`` -> ``{('src_x', 'A'): 1.5}``; None/[] -> ``{}``."""
    out: dict[tuple[str, str], float] = {}
    for part in specs or []:
        name, _, rest = part.partition("=")
        ctx, _, value = rest.rpartition(":")
        if not name or not ctx or not value:
            raise SystemExit(f"--source-weight: {part!r} is not NAME=CTX:W, e.g. 'src_x=A:1.5'")
        key = (name, ctx)
        if key in out:
            raise SystemExit(f"--source-weight: {name!r} in context {ctx!r} given twice")
        out[key] = _number(value, "--source-weight", part)
    return out


def read_source_weight_table(path: Path | None) -> pd.DataFrame | None:
    """`source`, `context`, `target` (None where a row applies to every knockdown) and `weight`."""
    if path is None:
        return None
    path = Path(path).expanduser()
    if not path.exists():
        raise SystemExit(f"--source-weight-table: {path} does not exist")
    tab = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype=str)
    missing = [c for c in ("source", "context", "weight") if c not in tab.columns]
    if missing:
        raise SystemExit(f"--source-weight-table: {path} lacks the column(s) {missing}; it needs `source`, "
                         "`context`, `weight` and optionally `target`")
    tab = tab[["source", "context", *(["target"] if "target" in tab.columns else []), "weight"]].copy()
    if "target" not in tab.columns:
        tab["target"] = None
    for c in ("source", "context"):
        tab[c] = tab[c].astype(str).str.strip()
    t = tab["target"].astype(object)
    tab["target"] = t.where(t.notna(), None).map(lambda v: None if v is None or str(v).strip() == "" else str(v))
    w = pd.to_numeric(tab["weight"], errors="coerce")
    if not w.map(lambda v: math.isfinite(v) and v >= 0).all():
        raise SystemExit(f"--source-weight-table: {path} carries a weight that is not finite and >= 0")
    tab["weight"] = w.astype(float)
    key = tab["source"] + "\x00" + tab["context"] + "\x00" + tab["target"].fillna("\x01")
    if key.duplicated().any():
        r = tab[key.duplicated()].iloc[0]
        raise SystemExit(f"--source-weight-table: {path} names ({r['source']}, {r['context']}, "
                         f"{r['target'] or 'every target'}) more than once")
    has_none = tab["target"].isna()
    both = set(zip(tab.loc[has_none, "source"], tab.loc[has_none, "context"])) & set(
        zip(tab.loc[~has_none, "source"], tab.loc[~has_none, "context"]))
    if both:
        s, c = sorted(both)[0]
        raise SystemExit(f"--source-weight-table: {path} gives ({s}, {c}) a row for every target AND rows for "
                         "single targets; give one or the other")
    tab.attrs["path"] = str(path)
    tab.attrs["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return tab


class SourceVote:
    """One context's vote multipliers: `m(name, target)` = the flag's W times the table's weight (1 unnamed)."""

    def __init__(self, context: str, flag: dict[str, float], every: dict[str, float],
                 per_target: dict[tuple[str, str], float]):
        self.context, self.flag, self.every, self.per_target = context, flag, every, per_target

    def m(self, name: str | None, target: str) -> float:
        if name is None:
            return 1.0
        t = self.per_target.get((name, target))
        return self.flag.get(name, 1.0) * (t if t is not None else self.every.get(name, 1.0))

    def key(self) -> tuple | None:
        """Equal for two contexts that vote alike (None for the plain vote), so they share one pool."""
        if self.is_identity:
            return None
        return (tuple(sorted(self.flag.items())), tuple(sorted(self.every.items())),
                tuple(sorted(self.per_target.items())))

    @property
    def is_identity(self) -> bool:
        return all(v == 1.0 for d in (self.flag, self.every, self.per_target) for v in d.values())

    def record(self) -> dict:
        names = sorted(set(self.flag) | set(self.every) | {n for n, _ in self.per_target})
        rec = {}
        for n in names:
            pt = [v for (s, _), v in self.per_target.items() if s == n]
            rec[n] = {"flag": self.flag.get(n, 1.0), "table_every_target": self.every.get(n),
                      "table_targets": len(pt),
                      **({"table_target_weight": {"min": min(pt), "median": float(pd.Series(pt).median()),
                                                  "max": max(pt), "zero": sum(v == 0 for v in pt)}} if pt else {})}
        return rec


def source_votes(flags: dict[tuple[str, str], float], table: pd.DataFrame | None, contexts: list[str],
                 source_names: list[str | None]) -> tuple[dict[str, SourceVote] | None, dict | None]:
    """Each context's `SourceVote`, and the record; (None, None) when nothing is given."""
    if not flags and table is None:
        return None, None
    have = [n for n in source_names if n is not None]
    named = {n for n, _ in flags} | (set(table["source"]) if table is not None else set())
    missing = sorted(named - set(have))
    if missing:
        raise SystemExit(f"--source-weight names {missing} match no source; have {sorted(have)}")
    twice = sorted(n for n in named if have.count(n) > 1)
    if twice:
        raise SystemExit(f"--source-weight names {twice} match more than one source: two files share the stem")
    named_ctx = {c for _, c in flags} | (set(table["context"]) if table is not None else set())
    if not named_ctx & set(contexts):
        raise SystemExit(f"--source-weight: no flag or table row names a context of this build "
                         f"({', '.join(contexts)}); they name {', '.join(sorted(named_ctx))}")
    votes = {}
    for c in contexts:
        flag = {n: w for (n, cc), w in flags.items() if cc == c}
        every, per_target = {}, {}
        if table is not None:
            rows = table[table["context"] == c]
            for s, t, w in zip(rows["source"], rows["target"], rows["weight"]):
                if t is None:
                    every[s] = float(w)
                else:
                    per_target[(s, t)] = float(w)
        votes[c] = SourceVote(c, flag, every, per_target)
    record = {"contexts": {c: votes[c].record() for c in contexts},
              "contexts_not_in_build": sorted(named_ctx - set(contexts)),
              "flags": {f"{n}={c}": w for (n, c), w in flags.items()}}
    if table is not None:
        record["table"] = {"path": table.attrs.get("path"), "sha256": table.attrs.get("sha256"),
                           "rows": int(len(table))}
    return votes, record


def vote_groups(ctxs, votes: dict[str, SourceVote] | None) -> list[tuple[str, ...]]:
    """`ctxs` (one pool's contexts) split by their votes, in order; one group when they vote alike."""
    if votes is None:
        return [tuple(ctxs)]
    groups: dict[tuple, list[str]] = {}
    for c in ctxs:
        groups.setdefault(votes[c].key(), []).append(c)
    return [tuple(g) for g in groups.values()]


def parse_source_scales(specs: list[str] | None, source_names: list[str | None]) -> dict[str, float]:
    """``['src_x=0.8']`` -> ``{'src_x': 0.8}``; S finite and > 0; the name must match exactly one source."""
    out: dict[str, float] = {}
    have = [n for n in source_names if n is not None]
    for part in specs or []:
        name, _, value = part.partition("=")
        if not name or not value:
            raise SystemExit(f"--source-scale: {part!r} is not NAME=S, e.g. 'src_x=0.8'")
        try:
            s = float(value)
        except ValueError:
            raise SystemExit(f"--source-scale: {value!r} is not a number") from None
        if not math.isfinite(s) or s <= 0:
            raise SystemExit(f"--source-scale: {part!r}: a strength is finite and > 0 (a vote of 0 is "
                             "--source-weight's)")
        if name in out:
            raise SystemExit(f"--source-scale: {name!r} given twice")
        if have.count(name) != 1:
            raise SystemExit(f"--source-scale: {name!r} matches {have.count(name)} sources; have {sorted(have)}")
        out[name] = s
    return out


def pool_kwargs(votes: dict[str, SourceVote] | None, context: str | None, scales: dict[str, float]) -> dict:
    """`pooled_delta`'s two keyword arguments for one context; {} when both are neutral."""
    kw = {}
    if votes is not None and context is not None and not votes[context].is_identity:
        kw["source_vote"] = votes[context]
    if scales and any(s != 1.0 for s in scales.values()):
        kw["source_scale"] = scales
    return kw
