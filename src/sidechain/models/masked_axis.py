"""Train `state tx` on a shared gene axis while each corpus is scored only on the genes it measured.

`sidechain.data.union_axis` puts every corpus on one wide axis, which leaves each corpus's rows
full of **structural zeros** -- genes that corpus never measured. Fit as written, those zeros are
a lie: the model would learn that K562-gwps silences the 10,854 genes Replogle's pipeline simply
did not quantify. This module is the other half: it tells the training loss to ignore them.

**Why per-set masking is cheap.** `cell_load`'s sampler builds every cell *sentence* from a
single h5 file (`samplers.py::_create_sentences` walks one subset at a time), so all 512 cells
in a set come from one corpus and share one mask. The mask is therefore a `(B, G)` row per
batch, not a `(B, S, G)` tensor, and it is applied once per set.

**What masking does to each loss.**

* `energy` / `sinkhorn` (`geomloss.SamplesLoss`, what PHE-1 used) read only pairwise Euclidean
  distances. Zeroing a coordinate in **both** `pred` and `target` removes it from every distance,
  so the masked loss is *exactly* the loss computed on the sub-axis. Tested.
* `mse` is an average over coordinates, so zeroing dilutes rather than removes. That path is
  computed directly instead: squared error summed over measured genes, divided by how many there
  were. Also exact.

**The scale knob, and why it is not free.** A set from a 7,679-gene corpus and a set from an
18,106-gene one now produce losses of different size -- for a distributional loss roughly in the
ratio sqrt(d), since a Euclidean norm over d iid-ish coordinates grows like sqrt(d). Left alone,
the five narrow corpora contribute the smallest gradients, which is the intersection's problem
sneaking back in through the loss instead of the axis. `mask_scaling` multiplies each set's loss
by `(G/measured)**0.5` (`auto`, `sqrt`), by `G/measured` (`linear`), or not at all (`none`).
`sqrt` is the first-order correction and a **heuristic** -- gene coordinates are not iid -- so it
is a declared knob to sweep, not a fact. It has never been priced against the local mirror.

Usage -- our flags first, then `state`'s own command line unchanged after `--`::

    uv run python -m sidechain.models.masked_axis \
        --axis-manifest ~/data/sidechain/derived/tx-train-union/axis_manifest.json \
        --mask-scaling sqrt -- \
        tx train data.kwargs.toml_config_path=.../union.toml training.max_steps=6000 ...

The TOML's ``[datasets]`` keys MUST be the manifest's source keys: that string is how a set finds
its mask, and a key the manifest does not know raises rather than training unmasked.

Masking is a training-time device. `state tx infer` emits the full axis, which is what the
submission and the local mirror want; nothing here changes inference.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sidechain.data.union_axis import load_manifest

SCALINGS = ("auto", "sqrt", "linear", "none")


# ------------------------------------------------------------------ pure functions


def masked_pair(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Zero the unmeasured coordinates of both tensors.

    `pred`/`target` are `(B, S, G)`; `mask` is `(B, G)` or `(G,)`, True where measured.
    Zeroing both sides leaves every pairwise Euclidean distance equal to the distance over
    the measured coordinates alone, and stops the masked outputs receiving any gradient.
    """
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(pred.shape[0], -1)
    if mask.shape != (pred.shape[0], pred.shape[-1]):
        raise ValueError(f"mask {tuple(mask.shape)} does not match pred {tuple(pred.shape)}")
    m = mask.unsqueeze(1).to(pred.dtype)
    return pred * m, target * m


def mask_scale(mask: torch.Tensor, mode: str, *, dtype=torch.float32) -> torch.Tensor:
    """Per-set multiplier that puts sets measured on different numbers of genes on one scale."""
    if mode not in SCALINGS:
        raise ValueError(f"mask_scaling must be one of {SCALINGS}, not {mode!r}")
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    n_total = mask.shape[-1]
    n_measured = mask.sum(-1).to(dtype).clamp_min(1.0)
    if mode == "none":
        return torch.ones_like(n_measured)
    ratio = n_total / n_measured
    return ratio if mode == "linear" else ratio.sqrt()


def masked_mse_per_set(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error per set over the measured genes only -- exact, not diluted."""
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(pred.shape[0], -1)
    m = mask.unsqueeze(1).to(pred.dtype)  # (B, 1, G), broadcast over the set
    err2 = ((pred - target) ** 2 * m).sum(dim=(1, 2))
    # cells in the set x genes it measured -- the (B, 1, G) mask sums to the gene count alone.
    denom = (m.sum(dim=(1, 2)) * pred.shape[1]).clamp_min(1.0)
    return err2 / denom


# --------------------------------------------------------------------- mask lookup


class SetMaskIndex:
    """`dataset_name` -> the row of the axis that corpus measured."""

    def __init__(self, genes: Sequence[str], measured: dict[str, np.ndarray]):
        self.genes = list(genes)
        self.n_genes = len(self.genes)
        self._masks: dict[str, torch.Tensor] = {}
        for key, idx in measured.items():
            m = torch.zeros(self.n_genes, dtype=torch.bool)
            m[torch.as_tensor(np.asarray(idx, dtype=np.int64))] = True
            self._masks[key] = m

    @classmethod
    def from_manifest(cls, path: str | Path) -> SetMaskIndex:
        genes, measured, _ = load_manifest(path)
        return cls(genes, measured)

    def check_axis(self, gene_names: Sequence[str] | None) -> None:
        """Refuse a run whose data is not on the manifest's axis."""
        if gene_names is None:
            return
        got = list(gene_names)
        if len(got) != self.n_genes:
            raise ValueError(
                f"the run's gene axis has {len(got):,} genes, the manifest {self.n_genes:,}. "
                "Project the h5ads with `python -m sidechain.data.union_axis project` first."
            )
        if got != self.genes:
            bad = next(i for i, (a, b) in enumerate(zip(got, self.genes)) if a != b)
            raise ValueError(
                f"the run's gene axis disagrees with the manifest at position {bad} "
                f"({got[bad]!r} vs {self.genes[bad]!r}); same length, different order or content."
            )

    def set_names(self, batch: dict, n_sets: int, key: str = "dataset_name") -> list[str]:
        """One source name per set, checked rather than assumed to be constant within a set."""
        names = batch.get(key)
        if names is None:
            raise KeyError(
                f"batch has no {key!r}; cell_load puts 'dataset_name' in every sample, so either "
                "the loader changed or --mask-key names a column that is not in additional_obs."
            )
        if isinstance(names, str):
            names = [names]
        names = [str(n) for n in names]
        if len(names) % n_sets:
            raise ValueError(f"{len(names)} {key} values do not divide into {n_sets} sets")
        span = len(names) // n_sets
        out = []
        for i in range(n_sets):
            chunk = names[i * span : (i + 1) * span]
            if len(set(chunk)) != 1:
                raise ValueError(
                    f"cell set {i} mixes sources {sorted(set(chunk))}. The mask is per corpus, so "
                    "a mixed set would be masked with one corpus's axis and scored on another's."
                )
            out.append(chunk[0])
        return out

    def rows(self, names: Sequence[str], device=None) -> torch.Tensor:
        """Stack one mask row per set; an unknown source raises, never trains unmasked."""
        missing = sorted({n for n in names if n not in self._masks})
        if missing:
            raise KeyError(
                f"no axis mask for source(s) {missing}; the manifest knows "
                f"{sorted(self._masks)}. The TOML [datasets] keys must be the manifest's keys."
            )
        out = torch.stack([self._masks[n] for n in names])
        return out.to(device) if device is not None else out


# ------------------------------------------------------- the model, and the install


_CONFIG: dict = {"manifest": None, "scaling": "auto", "key": "dataset_name"}


def _build_masked_class():
    """Defined lazily so importing this module does not drag in lightning + geomloss."""
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    class MaskedStateTransitionPerturbationModel(StateTransitionPerturbationModel):
        """STATE's transition model with the loss restricted to each corpus's measured genes."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            cfg = dict(_CONFIG)
            self._mask_index = SetMaskIndex.from_manifest(cfg["manifest"])
            self._mask_index.check_axis(self.hparams.get("gene_names"))
            self._mask_scaling = cfg["scaling"]
            self._mask_key = cfg["key"]
            self._current_mask: torch.Tensor | None = None
            # Into hparams so the checkpoint records that it was trained masked, and with what.
            self.hparams["sidechain_axis_manifest"] = str(cfg["manifest"])
            self.hparams["sidechain_mask_scaling"] = cfg["scaling"]
            self.hparams["sidechain_mask_key"] = cfg["key"]

        # -- mask plumbing ------------------------------------------------------

        @contextmanager
        def _masked(self, batch: dict, padded: bool):
            n_cells = batch["pert_cell_emb"].shape[0]
            n_sets = max(1, n_cells // self.cell_sentence_len) if padded else 1
            names = self._mask_index.set_names(batch, n_sets, self._mask_key)
            self._current_mask = self._mask_index.rows(names, device=batch["pert_cell_emb"].device)
            try:
                yield
            finally:
                self._current_mask = None

        def _compute_distribution_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            mask = self._current_mask
            if mask is None:
                return super()._compute_distribution_loss(pred, target)
            if isinstance(self.loss_fn, nn.MSELoss):
                return masked_mse_per_set(pred, target, mask)
            p, t = masked_pair(pred, target, mask)
            per_set = super()._compute_distribution_loss(p, t)
            scale = mask_scale(mask, self._mask_scaling, dtype=per_set.dtype).to(per_set.device)
            return per_set * scale.reshape(per_set.shape)

        # -- the three steps that reach the loss ---------------------------------

        def training_step(self, batch, batch_idx, padded=True):
            with self._masked(batch, padded):
                return super().training_step(batch, batch_idx, padded=padded)

        def validation_step(self, batch, batch_idx):
            with self._masked(batch, padded=True):
                return super().validation_step(batch, batch_idx)

        def test_step(self, batch, batch_idx):
            with self._masked(batch, padded=False):
                return super().test_step(batch, batch_idx)

    return MaskedStateTransitionPerturbationModel


def install(manifest: str | Path, *, scaling: str = "auto", key: str = "dataset_name") -> type:
    """Make `state tx train` build the masked model instead of the stock one.

    `get_lightning_module` imports `StateTransitionPerturbationModel` from its own module at
    call time, so rebinding the name there is enough -- every constructor argument state
    computes is passed through untouched, which is the point: no kwargs are re-derived here.
    """
    if scaling not in SCALINGS:
        raise ValueError(f"--mask-scaling must be one of {SCALINGS}, not {scaling!r}")
    manifest = Path(manifest).expanduser()
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    SetMaskIndex.from_manifest(manifest)  # fail here, not eight hours into a fit
    _CONFIG.update(manifest=manifest, scaling=scaling, key=key)

    import state.tx.models.state_transition as st

    masked = _build_masked_class()
    st.StateTransitionPerturbationModel = masked
    return masked


# ---------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=True
    )
    ap.add_argument("--axis-manifest", required=True, type=Path)
    ap.add_argument("--mask-scaling", choices=SCALINGS, default="auto")
    ap.add_argument("--mask-key", default="dataset_name")
    ap.add_argument("state_args", nargs=argparse.REMAINDER,
                    help="everything after `--`, passed to the `state` CLI unchanged")
    args = ap.parse_args(argv)

    rest = list(args.state_args)
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        raise SystemExit("nothing after `--`; expected e.g. `-- tx train data.kwargs...=...`")

    install(args.axis_manifest, scaling=args.mask_scaling, key=args.mask_key)
    print(f"masked axis: {args.axis_manifest} | scaling {args.mask_scaling} | key {args.mask_key}")

    from state.__main__ import main as state_main

    sys.argv = ["state", *rest]
    return state_main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
