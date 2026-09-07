"""Put `state tx`'s two encoder outputs on the same scale before they are added.

STATE's transition model adds the perturbation encoding to the basal-cell encoding in hidden
space and hands the sum to the transformer (`state_transition.py`, `forward`)::

    combined_input = self.encode_perturbation(pert) + self.encode_basal_expression(basal)

With `n_encoder_layers: 1` both encoders are a bare `nn.Linear` (`build_mlp` returns a single
layer), and nothing normalises either side. The two addends then arrive at wildly different
sizes, because their *inputs* are wildly different sizes: a perturbation one-hot has L2 norm 1,
a raw-count basal cell has norm ~659. Measured on the trained PHE-1 checkpoint, the spread of
the perturbation encoding across 831 targets was **1/227** of the spread of the basal encoding
across cells -- the perturbation was a rounding error on the transformer's input, and the model
learned one average perturbation direction and scaled it (effective rank 1.31 against real
data's 32.97).

**Rescaling the input cannot fix this on its own.** A bare `nn.Linear` is scale-equivariant:
shrink its input by 9x and it can learn weights 9x larger and land in the same place. Training
did exactly that -- `basal_encoder`'s first layer grew to 2.27x its initialisation while
`pert_encoder`'s grew to 1.32x, i.e. the fit pushed the imbalance *wider*. The ratio has to be
fixed where the addition happens, which is what this module does.

**What it does.** It subclasses STATE's transition model and wraps each encoder's output in a
`LayerNorm`, then rebinds the name `state.tx.models.state_transition.StateTransitionPerturbationModel`
so `state tx train` builds the subclass. Both paths default to normalised because normalising
only one moves the imbalance rather than removing it -- measured on 512 real HCT116 control
cells through untrained encoders at PHE-1's dimensions (38,584 genes, hidden 768, 831 targets):

===============================  ===========================  ==========================
input / normalisation            pert spread / basal spread    comment
===============================  ===========================  ==========================
raw counts, neither (PHE-1)      1/48 at init, 1/227 trained   what produced rank 1.31
raw counts, basal only           1/17                          better, still lopsided
shifted log, neither             1/6.6                         the log alone buys a lot
shifted log, basal only          1/25                          **worse** than no norm
shifted log, both                1/0.70                        parity, either direction
===============================  ===========================  ==========================

The fourth row is the one worth staring at: a `LayerNorm` fixes its output's norm at
`sqrt(hidden_dim)` ~ 27.7 whatever went in, so on a *log-transformed* input -- whose encoding
is already only ~7 long -- normalising the basal path alone inflates it. "LayerNorm the big
side" is the right instinct for raw counts and the wrong one once the counts are logged. Both
sides, or neither.

Parity is a starting point, not a claim: `LayerNorm`'s elementwise affine is a learned per-dimension
gain on each path, so gradient descent can still choose the mixture. The difference from the
stock model is that it now chooses from 1:1 rather than from 227:1, and that the basal path's
gradient no longer arrives ~659x larger than the perturbation path's under one shared learning rate.

Nothing else is touched. In particular the residual path is unaffected: with
`predict_residual: true` and `output_space: all`, STATE adds the **raw gene-space** basal vector
back after the transformer (`out_pred = project_out(res_pred) + basal`), so per-cell depth is
never carried by the encoding we normalise.

Usage -- our flags first, then `state`'s own command line unchanged after `--`::

    uv run python -m sidechain.models.balanced_encoders \\
        --basal-norm layernorm --pert-norm layernorm -- \\
        tx train data.kwargs.toml_config_path=.../fold.toml training.max_steps=2000 ...

The two settings are written into `hparams` as `sidechain_basal_norm` / `sidechain_pert_norm`,
so a checkpoint says how it was built and `load_from_checkpoint` reconstructs it correctly
whatever `install()` was last called with. **Reading such a checkpoint needs the class installed
first** -- without it `load_state_dict` refuses the extra keys, loudly. So inference goes through
this wrapper too (`... balanced_encoders -- tx infer --model-dir ...`; `_infer.py` imports the
class inside `run_tx_infer`, so the rebind reaches it), and `scripts/diagnose_tx_arm.py` takes a
`--balanced-encoders` flag for the same reason.

Composes with `sidechain.models.masked_axis`: each `install()` subclasses whatever is bound at
the time, so calling both -- in either order -- yields one class with both behaviours.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from torch import nn

NORMS = ("layernorm", "none")

_CONFIG: dict = {"basal": "layernorm", "pert": "layernorm"}


def make_norm(kind: str, hidden_dim: int) -> nn.Module:
    """The module put on one encoder's output. `none` is `nn.Identity`, not a skipped branch.

    Keeping an Identity in the graph means the two configurations differ only in what the
    module does, never in whether the attribute exists -- so a state_dict from one loads
    into the other, and `model.basal_norm` is always something you can print.
    """
    if kind not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, not {kind!r}")
    return nn.LayerNorm(hidden_dim) if kind == "layernorm" else nn.Identity()


def _build_balanced_class():
    """Defined lazily so importing this module does not drag in lightning + geomloss."""
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    class BalancedStateTransitionPerturbationModel(StateTransitionPerturbationModel):
        """STATE's transition model with each encoder output normalised before the add."""

        sidechain_balanced_encoders = True

        def __init__(self, *args, **kwargs):
            # The checkpoint's own record wins over whatever install() was last called with,
            # so load_from_checkpoint rebuilds the architecture that was trained.
            basal = str(kwargs.get("sidechain_basal_norm", _CONFIG["basal"]))
            pert = str(kwargs.get("sidechain_pert_norm", _CONFIG["pert"]))
            super().__init__(*args, **kwargs)
            self.basal_norm = make_norm(basal, self.hidden_dim)
            self.pert_norm = make_norm(pert, self.hidden_dim)
            self.hparams["sidechain_basal_norm"] = basal
            self.hparams["sidechain_pert_norm"] = pert

        def encode_basal_expression(self, expr):
            return self.basal_norm(super().encode_basal_expression(expr))

        def encode_perturbation(self, pert):
            return self.pert_norm(super().encode_perturbation(pert))

    return BalancedStateTransitionPerturbationModel


def install(*, basal: str = "layernorm", pert: str = "layernorm") -> type:
    """Make `state tx train` build the balanced model instead of the stock one.

    `get_lightning_module` imports `StateTransitionPerturbationModel` from its own module at
    call time, so rebinding the name there is enough -- every constructor argument state
    computes is passed through untouched.

    Installing twice updates the settings rather than wrapping a wrapper, which would stack
    two LayerNorms per path and silently change the architecture.
    """
    for name, kind in (("--basal-norm", basal), ("--pert-norm", pert)):
        if kind not in NORMS:
            raise ValueError(f"{name} must be one of {NORMS}, not {kind!r}")
    _CONFIG.update(basal=basal, pert=pert)

    import state.tx.models.state_transition as st

    current = st.StateTransitionPerturbationModel
    if getattr(current, "sidechain_balanced_encoders", False):
        return current
    balanced = _build_balanced_class()
    st.StateTransitionPerturbationModel = balanced
    return balanced


# ---------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=True
    )
    ap.add_argument("--basal-norm", choices=NORMS, default="layernorm")
    ap.add_argument("--pert-norm", choices=NORMS, default="layernorm")
    ap.add_argument("state_args", nargs=argparse.REMAINDER,
                    help="everything after `--`, passed to the `state` CLI unchanged")
    args = ap.parse_args(argv)

    rest = list(args.state_args)
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        raise SystemExit("nothing after `--`; expected e.g. `-- tx train data.kwargs...=...`")

    install(basal=args.basal_norm, pert=args.pert_norm)
    print(f"balanced encoders: basal {args.basal_norm} | pert {args.pert_norm}")

    from state.__main__ import main as state_main

    sys.argv = ["state", *rest]
    return state_main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
