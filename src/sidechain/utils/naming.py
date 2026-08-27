"""The model-name grammar, syntax only: SER-2n / ser-2n_delta4_even_noshrink_v1.

One grammar, three surfaces:

- **short name** ``SER-2n`` — series tag (three uppercase letters), dash, model number,
  then zero or more lowercase *knob letters* in alphabetical order. A new number is a new
  model; a letter marks one knob moved off the series baseline, and a model that inherits
  a parent's knob carries the letter from birth.
- **board name** ``Sidechain SER-2n`` — the short name with the team prefix, as typed at
  submit time. The board never renames, so this string is permanent the moment it is sent.
- **build stem** ``ser-2n_delta4_even_noshrink_v1`` — the lowercased short name, an
  underscore, a description slug, and a ``_v<k>`` tail. File stems, submission records and
  mirror run directories use this form (a bare lowercased short name is also fine for a
  run directory).

What the letters *mean* is registered in the naming decision record (ADR 0005, private
repo), which is also where the series list below is mirrored; this module owns only the
syntax so the build, the local mirror and the leaderboard tooling refuse a malformed name
before anything is scored under it. Freeform labels that never mention a series tag
(mirror ablation arms like ``h1_xatlas``) are not model names and pass untouched.
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
                 "e.g. 'ser-2n_delta4_even_noshrink_v1'")]
    if not _letters_ok(m.group(3)):
        return [(f"'{stem}': knob letters must be unique and alphabetical "
                 f"('{''.join(sorted(set(m.group(3))))}')")]
    return []


def short_from_stem(stem: str) -> str | None:
    """``'ser-2n_delta4_even_noshrink_v1'`` -> ``'SER-2n'``, or None."""
    m = STEM_RE.match(stem)
    if not m or not _letters_ok(m.group(3)):
        return None
    return f"{m.group(1).upper()}-{m.group(2)}{m.group(3)}"


def check_out_leaf(leaf: str, *, context: str, require_slug: bool = False) -> None:
    """Refuse an output name that claims a series tag but breaks the grammar.

    Called by the submission build and the local mirror on the ``--out`` leaf, so a typo
    (``ser2``, ``SER-2N``, letters out of order) dies before hours of compute are scored
    under it. A leaf that never mentions a series tag is a freeform label and passes.
    ``require_slug`` additionally demands the full ``_<slug>_v<k>`` tail (build stems name
    files that live for good; run directories may use the bare short name).
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
