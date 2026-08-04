"""lamindb run logging — config, seed, metrics, git SHA per run. Reproducibility is
the moat when you're solo.
"""
from __future__ import annotations


def log_run(config: dict, metrics: dict, artifacts: list[str] | None = None) -> None:
    # TODO(dev): ln.track() the run; register artifacts; store metrics.
    raise NotImplementedError
