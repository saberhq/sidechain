"""Contract tests for balancing STATE's two encoder outputs.

The claim the whole change rests on is `test_the_norms_bring_the_two_encodings_to_one_scale`:
with a raw-count basal cell and a one-hot perturbation, the stock model adds two vectors that
differ by two orders of magnitude, and after the install they are comparable. Everything else
here is about not breaking the model in the process -- same forward shape, same numbers when
both norms are off, and a checkpoint that says how it was built.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from sidechain.models.balanced_encoders import NORMS, install, make_norm

SET_LEN = 4
GENES = [f"G{i}" for i in range(12)]
PERT_DIM = 5


def _tiny_model(cls, *, gene_names=GENES, pert_dim=PERT_DIM, **extra):
    """STATE's real transition model at toy dimensions, so the wiring is the real wiring."""
    G, H = len(gene_names), 8
    kwargs = {
        "input_dim": G, "gene_dim": G, "hvg_dim": G, "output_dim": G, "pert_dim": pert_dim,
        "batch_dim": None, "cell_set_len": SET_LEN, "hidden_dim": H, "loss": "energy",
        "distributional_loss": "energy", "embed_key": None, "output_space": "all",
        "gene_names": list(gene_names), "batch_size": 2, "control_pert": "control",
        "predict_residual": True, "softplus": True, "n_encoder_layers": 1,
        "n_decoder_layers": 1, "lr": 1e-4, "basal_mapping_strategy": "random",
        "transformer_backbone_key": "llama",
        "transformer_backbone_kwargs": {
            "bidirectional_attention": True, "max_position_embeddings": SET_LEN,
            "hidden_size": H, "intermediate_size": 2 * H, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 4,
            "use_cache": False, "attention_dropout": 0.0, "hidden_dropout": 0.0,
            "layer_norm_eps": 1e-6, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
            "tie_word_embeddings": False, "rotary_dim": 0, "use_rotary_embeddings": False,
        },
    }
    kwargs.update(extra)
    return cls(**kwargs)


@pytest.fixture
def state_module():
    """`state.tx.models.state_transition`, with its class restored after every test."""
    st = pytest.importorskip("state.tx.models.state_transition")
    original = st.StateTransitionPerturbationModel
    try:
        yield st, original
    finally:
        st.StateTransitionPerturbationModel = original


def _batch(n_sets, G, pert_dim=PERT_DIM):
    torch.manual_seed(3)
    n = n_sets * SET_LEN
    return {
        "pert_cell_emb": torch.rand(n, G),
        "ctrl_cell_emb": torch.rand(n, G),
        "pert_emb": torch.eye(pert_dim)[torch.zeros(n, dtype=torch.long)],
    }


# ------------------------------------------------------------------------ the norm factory


def test_make_norm_builds_a_layernorm_over_the_hidden_dim():
    norm = make_norm("layernorm", 16)
    assert isinstance(norm, nn.LayerNorm)
    assert norm.normalized_shape == (16,)


def test_make_norm_none_is_an_identity_not_a_missing_module():
    """`none` still occupies the attribute, so one state_dict loads into the other."""
    assert isinstance(make_norm("none", 16), nn.Identity)
    assert list(make_norm("none", 16).parameters()) == []


def test_make_norm_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="must be one of"):
        make_norm("batchnorm", 8)


# ----------------------------------------------------------------------------- the install


def test_install_rebinds_the_class_state_builds(state_module):
    st, original = state_module
    balanced = install()
    assert st.StateTransitionPerturbationModel is balanced
    assert issubclass(balanced, original)


def test_install_twice_does_not_stack_two_norms_per_path(state_module):
    """Wrapping a wrapper would double every LayerNorm and change the architecture silently."""
    st, _ = state_module
    first = install()
    second = install(basal="none", pert="none")
    assert second is first
    assert st.StateTransitionPerturbationModel is first


def test_install_rejects_an_unknown_norm(state_module):
    with pytest.raises(ValueError, match="--basal-norm"):
        install(basal="whatever")
    with pytest.raises(ValueError, match="--pert-norm"):
        install(pert="whatever")


def test_install_composes_with_the_masked_axis_install(state_module, tmp_path):
    """Both rebind the same name; each subclasses whatever is bound, so order does not matter."""
    from sidechain.data.union_axis import build_axis, write_manifest
    from sidechain.models.masked_axis import install as install_masked

    st, original = state_module
    manifest = write_manifest(build_axis({"a": GENES[:6]}, mode="union", anchor=GENES),
                              tmp_path / "m.json")
    install()
    combined = install_masked(manifest, scaling="sqrt")
    assert st.StateTransitionPerturbationModel is combined
    assert issubclass(combined, original)
    assert getattr(combined, "sidechain_balanced_encoders", False)
    assert hasattr(combined, "_compute_distribution_loss")


# ------------------------------------------------- the real STATE class, wired up


def test_the_model_carries_a_norm_on_each_encoder(state_module):
    model = _tiny_model(install())
    assert isinstance(model.basal_norm, nn.LayerNorm)
    assert isinstance(model.pert_norm, nn.LayerNorm)


def test_none_on_both_paths_reproduces_the_stock_encodings(state_module):
    """The escape hatch has to be exact, or `--basal-norm none` is a third architecture."""
    _st, original = state_module
    torch.manual_seed(0)
    stock = _tiny_model(original)
    torch.manual_seed(0)
    balanced = _tiny_model(install(basal="none", pert="none"))
    expr = torch.rand(3, len(GENES))
    pert = torch.eye(PERT_DIM)
    assert torch.allclose(stock.encode_basal_expression(expr),
                          balanced.encode_basal_expression(expr))
    assert torch.allclose(stock.encode_perturbation(pert), balanced.encode_perturbation(pert))


def test_the_norms_bring_the_two_encodings_to_one_scale(state_module):
    """The finding this module exists for, at toy scale.

    A raw-count basal cell is orders of magnitude longer than a perturbation one-hot, and the
    two encodings are *added*, so the stock model hands the transformer a sum in which the
    perturbation is a rounding error. Measured as the diagnostic measures it: the spread of the
    perturbation encoding across targets, over the spread of the basal encoding across cells.
    """
    _st, original = state_module

    def ratio(model):
        torch.manual_seed(1)
        counts = torch.poisson(torch.full((64, len(GENES)), 200.0))  # depth ~2,400 per cell
        with torch.no_grad():
            basal = model.encode_basal_expression(counts)
            pert = model.encode_perturbation(torch.eye(PERT_DIM))
        spread = lambda a: (a - a.mean(0)).norm(dim=1).median().item()
        return spread(pert) / spread(basal)

    torch.manual_seed(0)
    stock = ratio(_tiny_model(original))
    torch.manual_seed(0)
    balanced = ratio(_tiny_model(install()))
    # Read in orders of magnitude, because the exact ratio is a function of the toy dimensions
    # and the real one is measured on real cells. What must hold at any scale: the stock model
    # is off by more than an order of magnitude, the balanced one is within one.
    assert abs(np.log10(stock)) > 1.0, f"the imbalance this fixes is absent from the fixture: {stock}"
    assert abs(np.log10(balanced)) < 1.5, f"still lopsided after the install: {balanced}"
    assert balanced > 20 * stock


def test_the_forward_still_runs_and_keeps_its_shape(state_module):
    model = _tiny_model(install())
    batch = _batch(2, len(GENES))
    out = model.forward(batch)
    assert out.shape == (2 * SET_LEN, len(GENES))
    assert torch.isfinite(out).all()


def test_a_full_training_step_runs_and_is_finite(state_module):
    model = _tiny_model(install())
    assert torch.isfinite(model.training_step(_batch(2, len(GENES)), 0))


def test_the_norm_parameters_get_gradient(state_module):
    """The learned gain is the whole point: training must be able to re-choose the mixture."""
    model = _tiny_model(install())
    model.training_step(_batch(2, len(GENES)), 0).backward()
    assert model.basal_norm.weight.grad.abs().max() > 0
    assert model.pert_norm.weight.grad.abs().max() > 0


def test_the_checkpoint_records_how_it_was_built(state_module):
    model = _tiny_model(install(basal="layernorm", pert="none"))
    assert model.hparams["sidechain_basal_norm"] == "layernorm"
    assert model.hparams["sidechain_pert_norm"] == "none"


def test_the_checkpoints_own_record_beats_the_current_install(state_module):
    """`load_from_checkpoint` passes hparams back as kwargs, so a reload must not inherit
    whatever `install()` happened to be called with in the reading process."""
    balanced = install(basal="layernorm", pert="layernorm")
    model = _tiny_model(balanced, sidechain_basal_norm="none", sidechain_pert_norm="none")
    assert isinstance(model.basal_norm, nn.Identity)
    assert isinstance(model.pert_norm, nn.Identity)
    assert model.hparams["sidechain_pert_norm"] == "none"


def test_a_checkpoint_round_trips_through_load_from_checkpoint(state_module, tmp_path):
    """The path `state tx infer` and the diagnostic both take.

    Lightning rebuilds the model from the checkpoint's `hyper_parameters`, so the norms must
    come back the way they were trained even though `install()` here asks for something else.
    """
    balanced = install(basal="layernorm", pert="none")
    # `decoder_cfg` because PHE-1's own hparams carry one and STATE's `on_load_checkpoint`
    # rebuilds the decoder from it; a checkpoint without one cannot be reloaded at all.
    model = _tiny_model(balanced, decoder_cfg={
        "latent_dim": len(GENES), "gene_dim": len(GENES), "hidden_dims": [4],
        "dropout": 0.1, "residual_decoder": False})
    path = tmp_path / "c.ckpt"
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": "2.0.0"}, path)

    install(basal="none", pert="layernorm")  # the reader asks for the opposite
    # weights_only=False for the same reason `state`'s own `_predict.py:307` passes it: the
    # hparams carry an `nn.Module` loss, which the PyTorch 2.6 default refuses to unpickle.
    back = balanced.load_from_checkpoint(path, map_location="cpu", weights_only=False)
    assert isinstance(back.basal_norm, nn.LayerNorm)
    assert isinstance(back.pert_norm, nn.Identity)
    expr = torch.rand(3, len(GENES))
    assert torch.allclose(back.encode_basal_expression(expr),
                          model.encode_basal_expression(expr))


def test_a_balanced_state_dict_does_not_load_into_the_stock_class(state_module):
    """Why `diagnose_tx_arm.py --balanced-encoders` exists: the failure is loud, not silent."""
    _st, original = state_module
    torch.manual_seed(0)
    balanced = _tiny_model(install())
    torch.manual_seed(0)
    stock = _tiny_model(original)
    with pytest.raises(RuntimeError, match="basal_norm"):
        stock.load_state_dict(balanced.state_dict())


def test_norms_is_the_vocabulary_the_cli_offers():
    assert set(NORMS) == {"layernorm", "none"}


def test_a_layernorm_output_has_the_length_the_docstring_claims():
    """`sqrt(hidden_dim)` at init -- the number the module's comparison table is read against."""
    h = 768
    out = make_norm("layernorm", h)(torch.randn(32, h) * 100 + 7)
    assert np.allclose(out.norm(dim=1).detach().numpy(), np.sqrt(h), rtol=1e-3)
