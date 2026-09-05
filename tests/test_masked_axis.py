"""Contract tests for masked-axis training.

The claim the whole design rests on is the first test here: for a distributional loss, zeroing
a coordinate in BOTH prediction and truth is the same as never having had that coordinate. If
that stops being true the union axis quietly starts fitting structural zeros.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sidechain.data.union_axis import build_axis, write_manifest
from sidechain.models.masked_axis import (
    SetMaskIndex,
    install,
    mask_scale,
    masked_mse_per_set,
    masked_pair,
)

GENES = ["A", "B", "C", "D", "E"]


def _index():
    return SetMaskIndex(GENES, {"narrow": np.array([1, 2]), "wide": np.array([0, 1, 2, 3])})


# ------------------------------------------------------------- the equivalence claim


@pytest.mark.parametrize("kind", ["energy", "sinkhorn"])
def test_masking_equals_training_on_the_sub_axis(kind):
    """Zeroed coordinates drop out of every pairwise distance, so the loss is unchanged."""
    geomloss = pytest.importorskip("geomloss")
    torch.manual_seed(0)
    B, S, G = 2, 16, 8
    keep = torch.tensor([True, True, False, True, False, False, True, False])
    pred, target = torch.rand(B, S, G), torch.rand(B, S, G)
    loss = geomloss.SamplesLoss(loss=kind, blur=0.05)

    p, t = masked_pair(pred, target, keep)
    full = loss(p, t)
    sub = loss(pred[..., keep].contiguous(), target[..., keep].contiguous())
    assert torch.allclose(full, sub, atol=1e-5), (full, sub)


def test_masked_coordinates_get_no_gradient():
    pred = torch.rand(1, 4, 5, requires_grad=True)
    target = torch.rand(1, 4, 5)
    mask = torch.tensor([[True, False, True, False, True]])
    p, t = masked_pair(pred, target, mask)
    ((p - t) ** 2).sum().backward()
    assert pred.grad[..., [1, 3]].abs().max() == 0.0
    assert pred.grad[..., [0, 2, 4]].abs().max() > 0.0


def test_masked_pair_rejects_a_mask_of_the_wrong_width():
    with pytest.raises(ValueError, match="does not match"):
        masked_pair(torch.rand(2, 3, 5), torch.rand(2, 3, 5), torch.ones(2, 4, dtype=torch.bool))


# ------------------------------------------------------------------------ mse is exact


def test_masked_mse_matches_mse_on_the_sub_axis():
    torch.manual_seed(1)
    pred, target = torch.rand(3, 7, 6), torch.rand(3, 7, 6)
    mask = torch.zeros(3, 6, dtype=torch.bool)
    mask[0, :2] = True
    mask[1, :5] = True
    mask[2, :] = True
    got = masked_mse_per_set(pred, target, mask)
    for i in range(3):
        k = mask[i]
        want = ((pred[i][:, k] - target[i][:, k]) ** 2).mean()
        assert torch.allclose(got[i], want, atol=1e-6)


def test_masked_mse_is_not_diluted_by_the_structural_zeros():
    """The stock nn.MSELoss on a padded axis divides by G; this divides by what was measured."""
    pred = torch.ones(1, 2, 10)
    target = torch.zeros(1, 2, 10)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, :2] = True
    assert masked_mse_per_set(pred, target, mask).item() == pytest.approx(1.0)
    p, t = masked_pair(pred, target, mask)
    assert torch.nn.functional.mse_loss(p, t).item() == pytest.approx(0.2)  # the diluted number


# ------------------------------------------------------------------------- the scale


def test_scale_modes():
    mask = torch.zeros(2, 100, dtype=torch.bool)
    mask[0, :25] = True
    mask[1, :100] = True
    assert torch.allclose(mask_scale(mask, "none"), torch.tensor([1.0, 1.0]))
    assert torch.allclose(mask_scale(mask, "linear"), torch.tensor([4.0, 1.0]))
    assert torch.allclose(mask_scale(mask, "sqrt"), torch.tensor([2.0, 1.0]))
    assert torch.allclose(mask_scale(mask, "auto"), mask_scale(mask, "sqrt"))


def test_scale_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mask_scaling"):
        mask_scale(torch.ones(1, 4, dtype=torch.bool), "biggest")


# ---------------------------------------------------------------------- mask lookup


def test_rows_are_the_manifest_masks():
    idx = _index()
    rows = idx.rows(["narrow", "wide"])
    assert rows.shape == (2, 5)
    assert list(rows[0]) == [False, True, True, False, False]
    assert list(rows[1]) == [True, True, True, True, False]


def test_an_unknown_source_raises_rather_than_training_unmasked():
    with pytest.raises(KeyError, match="no axis mask"):
        _index().rows(["narrow", "somebody_elses_key"])


def test_set_names_takes_one_name_per_set():
    batch = {"dataset_name": ["narrow"] * 4 + ["wide"] * 4}
    assert _index().set_names(batch, n_sets=2) == ["narrow", "wide"]


def test_a_set_that_mixes_sources_raises():
    batch = {"dataset_name": ["narrow", "wide", "wide", "wide"]}
    with pytest.raises(ValueError, match="mixes sources"):
        _index().set_names(batch, n_sets=2)


def test_a_batch_without_the_key_raises():
    with pytest.raises(KeyError, match="dataset_name"):
        _index().set_names({"cell_type": ["x"]}, n_sets=1)


def test_check_axis_refuses_a_run_on_a_different_axis():
    idx = _index()
    idx.check_axis(GENES)  # the happy path is silent
    with pytest.raises(ValueError, match="manifest"):
        idx.check_axis(GENES[:-1])
    with pytest.raises(ValueError, match="disagrees"):
        idx.check_axis(["A", "B", "C", "E", "D"])


def test_index_from_manifest_matches_the_plan(tmp_path):
    plan = build_axis({"a": ["A", "B"], "b": ["B", "C"]}, mode="union", anchor=GENES)
    path = write_manifest(plan, tmp_path / "m.json")
    idx = SetMaskIndex.from_manifest(path)
    assert idx.genes == list(plan.genes)
    assert list(idx.rows(["a"])[0]) == list(plan.mask("a"))


# ----------------------------------------------------------------------- the install


def test_install_rebinds_the_class_state_builds(tmp_path):
    st = pytest.importorskip("state.tx.models.state_transition")
    plan = build_axis({"a": ["A", "B"]}, mode="union", anchor=GENES)
    manifest = write_manifest(plan, tmp_path / "m.json")
    original = st.StateTransitionPerturbationModel
    try:
        masked = install(manifest, scaling="sqrt")
        assert st.StateTransitionPerturbationModel is masked
        assert issubclass(masked, original)
    finally:
        st.StateTransitionPerturbationModel = original


def test_install_rejects_an_unknown_scaling(tmp_path):
    plan = build_axis({"a": ["A", "B"]}, mode="union", anchor=GENES)
    manifest = write_manifest(plan, tmp_path / "m.json")
    with pytest.raises(ValueError, match="mask-scaling"):
        install(manifest, scaling="whatever")


def test_install_fails_now_rather_than_hours_into_a_fit(tmp_path):
    with pytest.raises(FileNotFoundError):
        install(tmp_path / "nope.json")


# ------------------------------------------------- the real STATE class, wired up

GENES12 = [f"G{i}" for i in range(12)]
NARROW = GENES12[:6]  # a corpus that measured half the axis
WIDE = GENES12[2:]
SET_LEN = 4


def _tiny_masked_model(masked, gene_names, *, pert_dim=3):
    """STATE's real transition model at toy dimensions, so the wiring is the real wiring."""
    G, H = len(gene_names), 8
    return masked(
        input_dim=G, gene_dim=G, hvg_dim=G, output_dim=G, pert_dim=pert_dim, batch_dim=None,
        cell_set_len=SET_LEN, hidden_dim=H, loss="energy", distributional_loss="energy",
        embed_key=None, output_space="all", gene_names=list(gene_names), batch_size=2,
        control_pert="control", predict_residual=True, softplus=True,
        n_encoder_layers=1, n_decoder_layers=1, lr=1e-4, basal_mapping_strategy="random",
        transformer_backbone_key="llama",
        transformer_backbone_kwargs={
            "bidirectional_attention": True, "max_position_embeddings": SET_LEN,
            "hidden_size": H, "intermediate_size": 2 * H, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 4,
            "use_cache": False, "attention_dropout": 0.0, "hidden_dropout": 0.0,
            "layer_norm_eps": 1e-6, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
            "tie_word_embeddings": False, "rotary_dim": 0, "use_rotary_embeddings": False,
        },
    )


@pytest.fixture
def masked_state(tmp_path):
    """(model, plan) on a 12-gene axis, with state's class rebound and restored after."""
    st = pytest.importorskip("state.tx.models.state_transition")
    plan = build_axis({"narrow": NARROW, "wide": WIDE}, mode="union", anchor=GENES12)
    manifest = write_manifest(plan, tmp_path / "m.json")
    original = st.StateTransitionPerturbationModel
    try:
        masked = install(manifest, scaling="sqrt")
        yield _tiny_masked_model(masked, GENES12), plan, masked
    finally:
        st.StateTransitionPerturbationModel = original


def _batch(n_sets, G, pert_dim=3):
    torch.manual_seed(3)
    n = n_sets * SET_LEN
    return {
        "pert_cell_emb": torch.rand(n, G),
        "ctrl_cell_emb": torch.rand(n, G),
        "pert_emb": torch.eye(pert_dim)[torch.zeros(n, dtype=torch.long)],
        "dataset_name": ["narrow"] * SET_LEN + ["wide"] * SET_LEN,
    }


def test_the_real_model_takes_its_mask_from_the_batch(masked_state):
    model, plan, _ = masked_state
    batch = _batch(2, len(GENES12))
    assert model._current_mask is None
    with model._masked(batch, padded=True):
        assert model._current_mask.shape == (2, len(GENES12))
        assert list(model._current_mask[0]) == list(plan.mask("narrow"))
        assert list(model._current_mask[1]) == list(plan.mask("wide"))
    assert model._current_mask is None  # cleared, so a later step cannot inherit it


def test_no_loss_gradient_reaches_a_gene_the_corpus_never_measured(masked_state):
    """Through the real model's forward and the real energy loss, per set.

    The invariant is on the model's OUTPUT, not on any inner weight: `final_down_then_up`
    mixes every gene into every other, so the shared weights legitimately get gradient from
    the measured genes. What must be zero is the signal entering at a masked coordinate.
    """
    model, plan, _ = masked_state
    G = len(GENES12)
    batch = _batch(2, G)
    pred = model.forward(batch).reshape(-1, SET_LEN, G).detach().requires_grad_(True)
    target = batch["pert_cell_emb"].reshape(-1, SET_LEN, G)
    with model._masked(batch, padded=True):
        model._compute_distribution_loss(pred, target).sum().backward()
    for i, name in enumerate(["narrow", "wide"]):
        mask = torch.as_tensor(plan.mask(name).copy())
        g = pred.grad[i].abs().sum(0)
        assert g[~mask].max() == 0.0, f"{name}: gradient on an unmeasured gene"
        assert g[mask].max() > 0.0, f"{name}: no gradient on the measured genes either"


def test_without_the_mask_the_structural_zeros_are_fitted(masked_state):
    """The bug this exists to prevent, made visible: unmasked, the padding pulls on the model."""
    model, plan, _ = masked_state
    G = len(GENES12)
    batch = _batch(2, G)
    pred = model.forward(batch).reshape(-1, SET_LEN, G).detach().requires_grad_(True)
    target = batch["pert_cell_emb"].reshape(-1, SET_LEN, G)
    model._compute_distribution_loss(pred, target).sum().backward()  # no mask set
    mask = torch.as_tensor(plan.mask("narrow").copy())
    assert pred.grad[0][:, ~mask].abs().max() > 0.0


def test_a_full_training_step_runs_and_is_finite(masked_state):
    model, _plan, _ = masked_state
    loss = model.training_step(_batch(2, len(GENES12)), 0)
    assert torch.isfinite(loss)


def test_the_checkpoint_records_that_it_was_trained_masked(masked_state):
    model, _plan, _ = masked_state
    assert model.hparams["sidechain_mask_scaling"] == "sqrt"
    assert model.hparams["sidechain_mask_key"] == "dataset_name"
    assert model.hparams["sidechain_axis_manifest"].endswith("m.json")


def test_a_model_built_on_a_different_axis_is_refused(masked_state):
    _model, _plan, masked = masked_state
    with pytest.raises(ValueError, match="manifest"):
        _tiny_masked_model(masked, GENES12[:4])
