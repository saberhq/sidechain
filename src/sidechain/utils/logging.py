"""lamindb run logging: config, metrics, git SHA and artifacts per scored run, never raising.

    log_run({"entry": "loco", ...}, metrics, artifacts=[str(out / "summary.json")])

Any failure (no lamindb, no instance, no network) becomes one RuntimeWarning per process; lamindb is imported
lazily. `SIDECHAIN_LAMIN_INSTANCE` picks the instance, the empty string skips connecting. Reads the files it is
given; writes a lamindb run, its artifacts and the refreshed registry export. Notes: private/code-notes/src/sidechain/utils/logging.py.md
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

from .lamin import DEFAULT_INSTANCE, artifact_key, export_registries, instance

__all__ = ["DEFAULT_INSTANCE", "code_sha", "log_run"]

_WARNED = False

# lamindb's `Run.cli_args` is a 1024-character column, filled by `ln.track()` with " ".join(sys.argv[1:]).
CLI_ARGS_MAX = 1024



def code_sha() -> str:
    """The commit the working tree is at, or `dirty:<sha>` / `unknown`."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True, cwd=Path(__file__).resolve().parent).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True,
                               cwd=Path(__file__).resolve().parent).stdout.strip()
        return f"dirty:{sha}" if dirty else sha
    except Exception:  # noqa: BLE001 - lineage never fails a run
        return "unknown"


def _key(p: Path) -> str:
    """The artifact key for a run output: its path under the data root, else `runs/<basename>`."""
    try:
        return artifact_key(p)
    except ValueError:
        return f"runs/{p.name}"


def log_run(config: dict, metrics: dict, artifacts: list[str] | None = None) -> None:
    """Record one scored run in lamindb: params (config + git SHA), metrics, artifacts (files only). Never raises."""
    global _WARNED
    saved = 0
    try:
        import lamindb as ln

        if inst := instance():
            # An unauthenticated machine fails here into the one-warning except below.
            ln.connect(inst)
        params = {"config": config, "metrics": metrics, "code_sha": code_sha()}
        argv = sys.argv
        cli = " ".join(argv[1:])
        if len(cli) > CLI_ARGS_MAX:
            # the whole call goes into params, the column gets a marked head of it
            params["argv"] = list(argv)
            mark = " ...[truncated; the whole call is params['argv']]"
            sys.argv = [argv[0], cli[: CLI_ARGS_MAX - len(mark)] + mark]
        try:
            ln.track(params=params)
        finally:
            sys.argv = argv
        for path in artifacts or []:
            p = Path(path).expanduser()
            if p.is_dir():
                # A directory is registered by `scripts/lamin_register.py`, never here.
                warnings.warn(f"log_run: refusing to register directory {p} (pass files)",
                              RuntimeWarning, stacklevel=2)
                continue
            ln.Artifact(str(p), key=_key(p)).save()
            saved += 1
        ln.finish()
    except Exception as exc:  # noqa: BLE001 - see the module docstring: non-fatal by contract
        if not _WARNED:
            warnings.warn(f"log_run: lamindb logging skipped ({exc!r})", RuntimeWarning,
                          stacklevel=2)
            _WARNED = True
        return
    if saved:
        _refresh_catalogue()


def _refresh_catalogue() -> None:
    """Refresh the registry export after an upload, as `lamin_register.py` does; a failure only warns."""
    try:
        export_registries()
    except Exception as exc:  # noqa: BLE001 - see the docstring
        warnings.warn(f"log_run: run logged, but the exit catalogue was not refreshed ({exc!r}); "
                      "re-run scripts/lamin_export.py", RuntimeWarning, stacklevel=3)
