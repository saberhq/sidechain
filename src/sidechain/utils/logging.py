"""lamindb run logging — config, seed, metrics, git SHA per run. Reproducibility is
the moat when you're solo.

Non-fatal by contract: a scoring run's validity comes from the mirror, so the logger
must never be the thing that fails it (the same philosophy `code_sha` was written
under — lineage must never be the thing that fails a 4h stream). With no lamin
instance configured, `log_run` degrades to one warning per process and returns.

lamindb is imported lazily so `log_run(...)`-free paths (and `--no-log-run` runs)
never pay for, or depend on, the import.
"""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

_WARNED = False


def code_sha() -> str:
    """The commit the working tree is at, or `dirty:<sha>` / `unknown`.

    Lifted from `data.stream_parquet_pseudobulk` (which now imports it from here)
    so run logging and stream lineage stamp the same value.
    """
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True, cwd=Path(__file__).resolve().parent).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True,
                               cwd=Path(__file__).resolve().parent).stdout.strip()
        return f"dirty:{sha}" if dirty else sha
    except Exception:  # noqa: BLE001 - lineage must never be the thing that fails a 4h stream
        return "unknown"


def log_run(config: dict, metrics: dict, artifacts: list[str] | None = None) -> None:
    """Record one scored run in lamindb: params (config + git SHA), metrics, artifacts.

    Never raises. Any failure — lamindb not importable, no instance connected,
    network down, a non-serialisable param — is reduced to a single
    RuntimeWarning per process, because the run being logged is already done and
    its numbers already live in the caller's own summary/report JSON.
    """
    global _WARNED
    try:
        import lamindb as ln

        ln.track(params={"config": config, "metrics": metrics, "code_sha": code_sha()})
        for path in artifacts or []:
            p = Path(path).expanduser()
            ln.Artifact(str(p), key=f"runs/{p.name}").save()
        ln.finish()
    except Exception as exc:  # noqa: BLE001 - see the module docstring: non-fatal by contract
        if not _WARNED:
            warnings.warn(f"log_run: lamindb logging skipped ({exc!r})", RuntimeWarning,
                          stacklevel=2)
            _WARNED = True
