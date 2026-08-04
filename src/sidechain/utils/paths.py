"""Path resolution that does not depend on the process CWD.

`score(config="configs/eval.yaml")` resolving against the CWD means an installed
package invoked from anywhere but the repo root gets FileNotFoundError. Everything
that reads a config goes through `resolve_config` instead.
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

    Raises FileNotFoundError naming every location tried, so a misconfigured
    deployment says where it looked instead of just which file it wanted.
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
