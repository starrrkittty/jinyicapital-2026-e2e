"""Trainer objective contracts, normalized units, and opt-in configuration."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from jyc_e2e.losses import PredictionLossConfig, PredictionObjective
from ptop.objectives import RegressionObjective
from ptop.schema import ContractError
from ptop.training_config import TrainingConfig


def test_default_objective_keeps_huber_value_and_gradient():
    scores = torch.tensor([-0.7, 0.2, 1.4], requires_grad=True)
    target = torch.tensor([-0.02, 0.01, 0.04])
    times = torch.zeros(3, dtype=torch.int64)
    loss, report = RegressionObjective(target_scale=0.02)(scores, target, times, torch.ones(3, dtype=torch.bool))
    expected = F.huber_loss(scores, target / 0.02)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(loss, scores)[0], torch.autograd.grad(expected, scores)[0])
    assert report["rank"] == 0 and report["ic"] == 0
    assert not RegressionObjective().requires_full_decision


def test_rank_parameters_preserve_raw_gap_units_and_are_serialized():
    settings = dict(target_scale=0.02, rank_weight=0.3, rank_temperature=0.7,
                    rank_min_target_gap=0.009, rank_max_pairs_per_time=99,
                    rank_weighting="target_gap", rank_gap_cap=1.5)
    objective = RegressionObjective(**settings)
    q = torch.tensor([-0.5, 0.7, 0.1, 1.2], requires_grad=True)
    target = torch.tensor([-0.02, -0.015, 0.01, 0.05])
    times = torch.zeros(4, dtype=torch.int64)
    flags = torch.ones(4, dtype=torch.bool)
    actual, _ = objective(q, target, times, flags)
    raw_objective = PredictionObjective(PredictionLossConfig(
        target_scale=0.02, reduction="sample", rank_weight=0.3, rank_temperature=0.7,
        min_target_gap=0.009, max_pairs_per_time=99, rank_weighting="target_gap", rank_gap_cap=1.5))
    expected, _ = raw_objective(q * 0.02, target, times)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, q)[0], torch.autograd.grad(expected, q)[0])
    assert all(objective.configuration()[name] == value for name, value in settings.items())


def test_ic_objective_masks_missing_labels_and_keeps_constant_score_gradient():
    q = torch.zeros(4, requires_grad=True)
    y = torch.tensor([-0.02, 0.0, 0.02, float("nan")])
    times = torch.zeros(4, dtype=torch.int64)
    objective = RegressionObjective(target_scale=0.02, ic_weight=0.2, ic_epsilon=0.01)
    loss, report = objective(q, y, times, torch.tensor([True, True, True, False]))
    assert objective.requires_full_decision
    assert report["ic"] == pytest.approx(1.0) and report["ic_times"] == 1
    assert report["count"] == 3
    loss.backward()
    assert torch.isfinite(q.grad).all() and q.grad[:3].abs().sum() > 0 and q.grad[3] == 0
    assert objective.configuration()["ic_epsilon"] == 0.01


def test_all_invalid_objective_has_differentiable_zero():
    q = torch.randn(3, requires_grad=True)
    loss, report = RegressionObjective(ic_weight=0.2)(
        q, torch.full((3,), float("nan")), torch.zeros(3, dtype=torch.int64), torch.zeros(3, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0 and report["count"] == 0 and report["ic_times"] == 0
    assert torch.equal(q.grad, torch.zeros_like(q))


@pytest.mark.parametrize("settings", [
    {"objective_scope": "batch"}, {"ic_weight": 0.1},
    {"objective_scope": "decision", "ic_weight": -1},
    {"ic_epsilon": 0}, {"ic_epsilon": True}, {"ic_epsilon": float("nan")},
    {"rank_temperature": 0}, {"rank_temperature": True},
    {"rank_min_target_gap": -1}, {"rank_max_pairs_per_time": 1.5},
    {"rank_max_pairs_per_time": True}, {"rank_weighting": "tail"}, {"rank_gap_cap": 0},
])
def test_training_config_rejects_invalid_objective_settings(settings):
    with pytest.raises(ContractError):
        TrainingConfig(**settings)


def test_training_config_round_trip_and_old_json_defaults():
    default = TrainingConfig.from_dict({"epochs": 2})
    assert default.objective_scope == "microbatch" and default.ic_weight == 0 and default.rank_weight == 0
    full = replace(default, objective_scope="decision", ic_weight=0.2, rank_weight=0.1,
                   rank_temperature=0.5, rank_min_target_gap=0.001,
                   rank_max_pairs_per_time=512, rank_weighting="target_gap", rank_gap_cap=2.0)
    assert TrainingConfig.from_dict(full.to_dict()) == full


@pytest.mark.parametrize("settings", [{"ic_weight": -1}, {"ic_epsilon": 0},
                                     {"rank_temperature": 0}, {"rank_weighting": "bad"},
                                     {"rank_gap_cap": float("inf")}, {"target_scale": True}])
def test_objective_rejects_invalid_parameters(settings):
    with pytest.raises(ContractError):
        RegressionObjective(**settings)
