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
import sys
import warnings
from pathlib import Path

from .lamin import DEFAULT_INSTANCE, artifact_key, export_registries, instance

__all__ = ["DEFAULT_INSTANCE", "code_sha", "log_run"]

_WARNED = False

# lamindb's `Run.cli_args` is a 1024-character column, filled by `ln.track()` with
# " ".join(sys.argv[1:]). An `eval.loco` arm on a box (full source paths, neighbour flags)
# runs past it, and the save fails -- every round-two T103 arm went unlogged that way.
CLI_ARGS_MAX = 1024

# Where run logs land. `ln.connect()` is PROCESS-LOCAL (verified 2026-08-27: a
# fresh process still sees none/none), so connecting here never changes machine
# state for the other sessions in this checkout -- which is why this is wired
# per-call rather than via `lamin connect` on the machine. Override with
# SIDECHAIN_LAMIN_INSTANCE; set it to the empty string to skip connecting and
# fall back to whatever default instance the process already has (usually none,
# which degrades to the one-warning skip below). `utils.lamin.instance()` owns
# that reading now, and DEFAULT_INSTANCE is re-exported above so callers and
# tests that import it from here keep working.


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


def _key(p: Path) -> str:
    """The artifact key for a run output: its path under the data root (ADR 0007 §1).

    The old rule was `runs/<basename>`, and on 2026-08-28 five unrelated scored runs
    all wrote `summary.json`, so the instance folded them into five *versions* of one
    `runs/summary.json` -- four scored runs silently demoted to history of a fifth.
    Keying by the full path keeps distinct runs distinct, and re-scoring the same fold
    to the same directory still versions, which is what versioning is for.

    Outputs written outside `~/data/sidechain/` keep the old flat key: a run is never
    worth failing over a filing question.
    """
    try:
        return artifact_key(p)
    except ValueError:
        return f"runs/{p.name}"


def log_run(config: dict, metrics: dict, artifacts: list[str] | None = None) -> None:
    """Record one scored run in lamindb: params (config + git SHA), metrics, artifacts.

    Never raises. Any failure — lamindb not importable, no instance connected,
    network down, a non-serialisable param — is reduced to a single
    RuntimeWarning per process, because the run being logged is already done and
    its numbers already live in the caller's own summary/report JSON.
    """
    global _WARNED
    saved = 0
    try:
        import lamindb as ln

        if inst := instance():
            # Unauthenticated machines (a fresh Brev box) fail here and land in
            # the except below -- the run still completes, one warning.
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
                # A run OUTDIR is not a run artifact. `local_mirror` used to pass one,
                # and a mirror outdir is ~21 GB per line -- a scored run would have
                # silently uploaded it as a folder artifact, and folder artifacts
                # OVERWRITE their own previous version's bytes in S3 (all versions share
                # uid[:16], lamindb/core/storage/paths.py:39-43). Registering a
                # directory is `scripts/lamin_register.py`'s job: deliberate, loud, and
                # never a side effect of scoring.
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
    """Refresh the exit catalogue after an upload (ADR 0007 §7b), as `lamin_register.py` does.

    The catalogue is the only thing that maps lamindb's uid-named S3 objects back to our
    paths, so it must never be behind the bucket. Until 2026-09-25 only the register script
    refreshed it, and 34 artifacts saved here since 2026-09-19 were missing from it. About
    12 s against our instance. Non-fatal, like the rest of `log_run`: the run already
    succeeded, and a stale catalogue is fixed by `scripts/lamin_export.py`.
    """
    try:
        export_registries()
    except Exception as exc:  # noqa: BLE001 - see the docstring
        warnings.warn(f"log_run: run logged, but the exit catalogue was not refreshed ({exc!r}); "
                      "re-run scripts/lamin_export.py", RuntimeWarning, stacklevel=3)
