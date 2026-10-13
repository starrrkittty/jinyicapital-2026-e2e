"""Cross-part fusion checks at the two seams of the A -> B -> C chain.

Seam 1 (A -> B) is the token boundary exposed by ``TemporalScoreModel.encode/temporal``.
Seam 2 (B -> C) is the objective boundary the trainer mounts. The temporal-model cases
need the pinned Qlib, so they skip when it is absent instead of passing silently.
"""

import json
import math

import numpy as np
import pytest
import torch

from jyc_e2e.losses import SCORE_UNITS, PredictionLossConfig
from ptop.backbones import TransformerBackbone
from ptop.objectives import (NORMALIZED_SCORE_UNITS, RegressionObjective,
                             build_prediction_objective, prediction_target_scale)
from ptop.schema import FEATURES, ContractError


def test_objectives_declare_distinct_score_units():
    """The two tails exist; each must say which convention it expects."""
    assert build_prediction_objective(
        PredictionLossConfig(), target_scale=0.01).configuration()["score_units"] == SCORE_UNITS
    assert RegressionObjective().configuration()["score_units"] == NORMALIZED_SCORE_UNITS
    assert SCORE_UNITS != NORMALIZED_SCORE_UNITS


def test_objective_declares_json_configuration():
    """B's checkpoint/run_config path serialises the objective config with allow_nan=False."""
    objective = build_prediction_objective(
        PredictionLossConfig(regression="huber", rank_weight=0.1), target_scale=0.01)
    config = objective.configuration()
    roundtrip = json.loads(json.dumps(config, ensure_ascii=False, allow_nan=False))
    assert roundtrip["name"] == "PredictionObjective"
    assert roundtrip["target_scale"] == 0.01
    assert roundtrip["rank_weight"] == 0.1


def test_objective_accepts_trainer_positional_arguments():
    """Trainer calls objective(scores, target, ids, flags) positionally and needs a grad."""
    objective = build_prediction_objective(
        PredictionLossConfig(regression="mse", target_scale=0.01))
    scores = torch.tensor([0.005, -0.005], requires_grad=True)
    target = torch.tensor([0.01, -0.01])
    ids = torch.tensor([0, 0])
    flags = torch.tensor([True, True])
    loss, report = objective(scores, target, ids, flags)
    loss.backward()
    assert torch.isfinite(loss)
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    assert report["valid_samples"] == 2


def test_raw_and_normalised_conventions_agree_after_head_rescaling():
    """The two conventions differ only by how the head is parameterised, not by the loss."""
    scale = 0.01
    target = torch.tensor([0.02, -0.01, 0.0])
    ids = torch.tensor([0, 0, 0])
    flags = torch.full((3,), True)
    raw = torch.tensor([0.03, -0.02, 0.01])
    raw_loss, _ = build_prediction_objective(PredictionLossConfig(
        regression="huber", reduction="sample", target_scale=scale))(raw, target, ids, flags)
    normalised_loss, _ = RegressionObjective(target_scale=scale)(raw / scale, target, ids, flags)
    assert math.isclose(float(raw_loss), float(normalised_loss), rel_tol=1e-5, abs_tol=1e-9)


def test_target_scale_matches_trainer_calibration():
    targets = np.array([0.01, -0.02, 0.03])
    expected = float(np.array([0.01, -0.02, 0.03], dtype=np.float64).std(ddof=0))
    assert math.isclose(prediction_target_scale(targets), expected, rel_tol=1e-12)
    valid = np.array([True, False, True])
    expected_masked = float(np.array([0.01, 0.03], dtype=np.float64).std(ddof=0))
    assert math.isclose(prediction_target_scale(targets, valid), expected_masked, rel_tol=1e-12)


def test_target_scale_rejects_unusable_labels():
    with pytest.raises(ContractError, match="at least two"):
        prediction_target_scale(np.array([0.01]))
    with pytest.raises(ContractError):
        prediction_target_scale(np.array([0.01, 0.02]), np.array([True, True, False]))
    with pytest.raises(ContractError, match="zero or nonfinite"):
        prediction_target_scale(np.array([0.02, 0.02]))


def test_build_prediction_objective_validates_arguments():
    with pytest.raises(ContractError, match="PredictionLossConfig"):
        build_prediction_objective(object())
    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(ContractError, match="finite and positive"):
            build_prediction_objective(PredictionLossConfig(), target_scale=bad)


def test_transformer_prefix_does_not_see_later_history():
    """Guard the single causal mask: a future token must not change earlier outputs."""
    torch.manual_seed(0)
    backbone = TransformerBackbone(8, 16, 2, 0.0, 4, 4, True).eval()
    tokens = torch.randn(1, 8, 8)
    valid = torch.ones(1, 8, dtype=torch.bool)
    with torch.no_grad():
        base, _, _ = backbone(tokens, valid)
        changed = tokens.clone()
        changed[0, -1, :] = 7.5
        updated, _, _ = backbone(changed, valid)
    torch.testing.assert_close(base[0, :-1], updated[0, :-1])
    assert not torch.allclose(base[0, -1], updated[0, -1])


def test_transformer_causal_flag_can_be_disabled():
    torch.manual_seed(0)
    causal = TransformerBackbone(8, 16, 2, 0.0, 4, 4, True)
    bidirectional = TransformerBackbone(8, 16, 2, 0.0, 4, 4, False)
    bidirectional.load_state_dict(causal.state_dict())
    bidirectional.eval()
    causal.eval()
    tokens = torch.randn(1, 8, 8)
    valid = torch.ones(1, 8, dtype=torch.bool)
    with torch.no_grad():
        changed = tokens.clone()
        changed[0, -1, :] = 7.5
        causal_base, _, _ = causal(tokens, valid)
        causal_updated, _, _ = causal(changed, valid)
        bidi_base, _, _ = bidirectional(tokens, valid)
        bidi_updated, _, _ = bidirectional(changed, valid)
    assert torch.allclose(causal_base[0, :-1], causal_updated[0, :-1])
    assert not torch.allclose(bidi_base[0, :-1], bidi_updated[0, :-1])


def test_encode_then_temporal_equals_forward():
    pytest.importorskip("qlib")
    from ptop.temporal import ModelSpec, create_score_model

    torch.manual_seed(0)
    model = create_score_model(ModelSpec(name="B-GRU", encoder="Raw", width=16, layers=1)).eval()
    observed = torch.rand(3, 12, len(FEATURES)) > 0.1
    x = torch.randn(3, 12, len(FEATURES)).masked_fill(~observed, 0)
    step_valid = torch.ones(3, 12, dtype=torch.bool)
    with torch.no_grad():
        direct = model(x, observed, step_valid)
        tokens, token_valid = model.encode(x, observed, step_valid)
        staged = model.temporal(tokens, token_valid)
    assert tokens.shape[:2] == (3, 12)
    assert token_valid.dtype == torch.bool and token_valid.shape == (3, 12)
    torch.testing.assert_close(direct, staged)


def test_encode_rejects_non_window_input():
    pytest.importorskip("qlib")
    from ptop.temporal import ModelSpec, create_score_model

    model = create_score_model(ModelSpec(name="B-GRU", encoder="Raw", width=16, layers=1))
    with pytest.raises(ContractError, match="\\[B,L,F\\]"):
        model.encode(torch.randn(4, len(FEATURES)), torch.ones(4, len(FEATURES), dtype=torch.bool),
                     torch.ones(4, dtype=torch.bool))
