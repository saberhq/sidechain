"""Local mirror of Arc's cell-eval — the highest-leverage piece. Nothing is trusted
until it clears this. Emits the 3 challenge metrics (DES/PDS/MAE) + STATE's extras
(7 total, the Generalist-prize set) and runs the anti-hacking guardrails.
"""
from __future__ import annotations
from pathlib import Path
import yaml


def score(pred_h5ad: str, truth_h5ad: str, config: str = "configs/eval.yaml") -> dict:
    """Run cell-eval on a prediction. Returns {metric: value} incl. average-rank.

    TODO(dev): call cell_eval (ArcInstitute/cell-eval) + `cell-eval score` against
    the baseline agg_results; add variance-inflation + public-split-only checks.
    """
    cfg = yaml.safe_load(Path(config).read_text())
    raise NotImplementedError("wire to cell-eval")


if __name__ == "__main__":  # `python -m sidechain.eval.local_mirror --help`
    import typer
    typer.run(score)
