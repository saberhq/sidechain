"""Path resolution that does not depend on the process CWD.

    resolve_config("configs/eval.yaml")     # as given, then against the CWD, then against the repo root

Reads and writes nothing. Notes: private/code-notes/src/sidechain/utils/paths.py.md
"""
from __future__ import annotations

from pathlib import Path

_MARKERS = ("pyproject.toml", ".git")


def find_repo_root(start: Path | None = None) -> Path | None:
    """Walk up from `start` (default: this module) looking for a repo marker."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if any((parent / m).exists() for m in _MARKERS):
            return parent
    return None


def resolve_config(path: str | Path) -> Path:
    """Resolve a config path against, in order: as-given, the CWD, the repo root.

    Raises FileNotFoundError naming every location tried.
    """
    p = Path(path).expanduser()
    tried: list[Path] = []

    if p.is_absolute():
        if p.exists():
            return p
        tried.append(p)
    else:
        for base in (Path.cwd(), find_repo_root()):
            if base is None:
                continue
            cand = (base / p).resolve()
            tried.append(cand)
            if cand.exists():
                return cand

    raise FileNotFoundError(
        f"Config {path!r} not found. Tried: " + ", ".join(str(t) for t in tried)
    )
