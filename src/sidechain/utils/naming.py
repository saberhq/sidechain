"""The model-name grammar, syntax only: the short name, the board name and the build stem.

    parse_short("SER-2n")                                   # ('SER', 2, 'n')
    short_from_stem("ser-2n_example_v1")                    # 'SER-2n'
    check_out_leaf(out.name, context="mirror2026.score")    # SystemExit on a malformed model-like name

Short name `TAG-<number><letters>` (a `SERIES` tag; letters unique and alphabetical); board name `Sidechain ` plus
it; build stem the lowercased short name, `_<slug>`, `_v<k>`. A label with no series tag is not a model name.
Reads and writes nothing. Notes: private/code-notes/src/sidechain/utils/naming.py.md
"""
from __future__ import annotations

import re

BOARD_PREFIX = "Sidechain "
SERIES = ("GLY", "ALA", "SER", "CYS", "HIS", "LYS", "ARG", "PHE", "TYR", "TRP", "PRO")

SHORT_RE = re.compile(rf"^({'|'.join(SERIES)})-(\d+)([a-z]*)$")
STEM_RE = re.compile(rf"^({'|'.join(s.lower() for s in SERIES)})-(\d+)([a-z]*)(?:_[a-z0-9]+(?:_[a-z0-9]+)*_v\d+)?$")
# A string "claims" to be a model name if it opens with a series tag (any case) followed
# by a number, with or without the dash. `serine_test` does not claim; `ser2_x` does.
CLAIMS_RE = re.compile(rf"^(?:{'|'.join(SERIES)})[-_]?\d", re.IGNORECASE)


def parse_short(name: str) -> tuple[str, int, str] | None:
    """``'SER-2n'`` -> ``('SER', 2, 'n')``, or None if it is not a valid short name."""
    m = SHORT_RE.match(name)
    if not m or not _letters_ok(m.group(3)):
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def _letters_ok(letters: str) -> bool:
    return list(letters) == sorted(set(letters))


def problems_short(name: str) -> list[str]:
    """Why ``name`` is not a valid short name (empty list = valid)."""
    m = SHORT_RE.match(name)
    if not m:
        got = re.match(r"^([A-Za-z]{3})[-_]?(\d+)([A-Za-z]*)$", name)
        if got and got.group(1).upper() in SERIES:
            fixed = f"{got.group(1).upper()}-{got.group(2)}{got.group(3).lower()}"
            return [f"'{name}' is malformed; the grammar is TAG-<number><letters>, e.g. '{fixed}'"]
        return [f"'{name}' does not match TAG-<number><letters> with TAG one of {', '.join(SERIES)}"]
    if not _letters_ok(m.group(3)):
        return [(f"'{name}': knob letters must be unique and alphabetical "
                 f"('{''.join(sorted(set(m.group(3))))}')")]
    return []


def problems_stem(stem: str) -> list[str]:
    """Why ``stem`` is not a valid build stem / run-directory name (empty list = valid)."""
    m = STEM_RE.match(stem)
    if not m:
        if stem != stem.lower():
            return [f"'{stem}': stems are lowercase (the short name uppercases, the stem never does)"]
        return [(f"'{stem}' does not match <tag>-<number><letters>[_<slug>_v<k>], "
                 "e.g. 'ser-2n_example_v1'")]
    if not _letters_ok(m.group(3)):
        return [(f"'{stem}': knob letters must be unique and alphabetical "
                 f"('{''.join(sorted(set(m.group(3))))}')")]
    return []


def short_from_stem(stem: str) -> str | None:
    """``'ser-2n_example_v1'`` -> ``'SER-2n'``, or None."""
    m = STEM_RE.match(stem)
    if not m or not _letters_ok(m.group(3)):
        return None
    return f"{m.group(1).upper()}-{m.group(2)}{m.group(3)}"


def check_out_leaf(leaf: str, *, context: str, require_slug: bool = False) -> None:
    """Refuse an output name that claims a series tag but breaks the grammar; other names pass.

    ``require_slug`` also demands the full ``_<slug>_v<k>`` tail.
    """
    if not CLAIMS_RE.match(leaf):
        return
    probs = problems_stem(leaf)
    if not probs and require_slug and "_" not in leaf:
        probs = [(f"'{leaf}': a build stem needs the full form <short>_<slug>_v<k>, "
                  f"e.g. '{leaf}_delta_even_v1'")]
    if probs:
        raise SystemExit(f"{context}: output name looks like a model name but breaks the "
                         f"naming grammar (ADR 0005) -- " + "; ".join(probs))
