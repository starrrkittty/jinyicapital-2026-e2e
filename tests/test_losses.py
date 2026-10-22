import math

import pytest
import torch
from torch.nn import functional as F

from jyc_e2e.losses import PredictionLossConfig, PredictionObjective, pairwise_rank_loss, pearson_ic_loss


def test_pairwise_default_and_weighted_hand_calculation():
    scores = torch.tensor([0.2, -0.3, 0.8], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0., 0.5, 3.], dtype=torch.float64)
    times = torch.zeros(3, dtype=torch.long)
    pairs = [(0, 1), (0, 2), (1, 2)]
    terms = torch.stack([F.softplus((scores[i] - scores[j]) / 2 / .7) for i, j in pairs])
    weights = torch.tensor([.25, 1., 1.], dtype=torch.float64)
    uniform = pairwise_rank_loss(scores, target, times, target_scale=2, temperature=.7)
    weighted = pairwise_rank_loss(scores, target, times, target_scale=2, temperature=.7,
                                  rank_weighting="target_gap", rank_gap_cap=1.)
    torch.testing.assert_close(uniform, terms.mean())
    torch.testing.assert_close(weighted, (weights * terms).sum() / weights.sum())
    assert not torch.isclose(uniform, weighted)
    assert torch.isfinite(torch.autograd.grad(weighted, scores)[0]).all()


def test_weighted_scale_conversion_and_tiny_weights():
    scores = torch.tensor([.1, -.2, .4], dtype=torch.float64)
    target = torch.tensor([0., .5, 2.], dtype=torch.float64)
    times = torch.zeros(3, dtype=torch.long)
    first = pairwise_rank_loss(scores, target, times, rank_weighting="target_gap")
    scaled = pairwise_rank_loss(scores * 10, target * 10, times, target_scale=10,
                                rank_weighting="target_gap")
    torch.testing.assert_close(first, scaled)
    tiny = pairwise_rank_loss(scores, target, times, target_scale=1e300,
                              rank_weighting="target_gap")
    assert torch.isfinite(tiny) and tiny > 0


def test_pairwise_gradcheck_and_temperature():
    scores = torch.tensor([.1, -.2, .4], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0., .5, 2.], dtype=torch.float64)
    times = torch.zeros(3, dtype=torch.long)
    assert torch.autograd.gradcheck(
        lambda x: pairwise_rank_loss(x, target, times, rank_weighting="target_gap"), (scores,))
    low = pairwise_rank_loss(target, target, times, temperature=.2)
    high = pairwise_rank_loss(target, target, times, temperature=2.)
    assert low < high


def test_prediction_objective_defaults_and_weighted_forward():
    scores = torch.tensor([.1, -.2, .4], requires_grad=True)
    target = torch.tensor([0., .5, 2.])
    times = torch.zeros(3, dtype=torch.long)
    baseline, info = PredictionObjective()(scores, target, times)
    torch.testing.assert_close(baseline, F.huber_loss(scores, target))
    assert info["rank_loss"] == 0
    cfg = PredictionLossConfig(rank_weight=.3, rank_weighting="target_gap", rank_gap_cap=.7)
    actual, info = PredictionObjective(cfg)(scores, target, times)
    expected = baseline + .3 * pairwise_rank_loss(scores, target, times,
                                                  rank_weighting="target_gap", rank_gap_cap=.7)
    torch.testing.assert_close(actual, expected)
    assert info["valid_pairs"] == 3 and info["rank_times"] == 1


def test_weighted_subnormal_gap_does_not_underflow_normalization():
    scores = torch.tensor([0., 1., 2.], requires_grad=True)
    target = torch.tensor([0., 1e-40, 2e-40])
    loss = pairwise_rank_loss(scores, target, torch.zeros(3, dtype=torch.long),
                             target_scale=1e20, rank_weighting="target_gap")
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert torch.isfinite(scores.grad).all()


@pytest.mark.parametrize("weighting", ["uniform", "target_gap"])
def test_pairwise_mask_ties_gap_and_empty_pairs(weighting):
    scores = torch.tensor([0., 1., 9.], requires_grad=True)
    target = torch.tensor([0., .5, float("nan")])
    times = torch.zeros(3, dtype=torch.long)
    mask = torch.tensor([True, True, False])
    loss = pairwise_rank_loss(scores, target, times, target_valid=mask,
                             min_target_gap=.5, rank_weighting=weighting)
    assert loss == 0
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.zeros(3))
    assert pairwise_rank_loss(scores, torch.zeros(3), times, rank_weighting=weighting) == 0


def test_pairwise_equal_time_weighting_and_sample_reproducibility(monkeypatch):
    scores = torch.arange(5, dtype=torch.float64)
    target = torch.tensor([1., 0., 0., 2., 1.], dtype=torch.float64)
    times = torch.tensor([1, 1, 2, 2, 2])
    kwargs = {"rank_weighting": "target_gap", "rank_gap_cap": .8}
    actual = pairwise_rank_loss(scores, target, times, **kwargs)
    expected = (pairwise_rank_loss(scores[:2], target[:2], times[:2], **kwargs)
                + pairwise_rank_loss(scores[2:], target[2:], times[2:], **kwargs)) / 2
    torch.testing.assert_close(actual, expected)
    def no_quadratic_allocation(*args, **kwargs):
        raise AssertionError("large sections must not allocate all pairs")
    monkeypatch.setattr(torch, "triu_indices", no_quadratic_allocation)
    scores = torch.linspace(0, 1, 10000)
    target = -scores
    times = torch.zeros(10000, dtype=torch.long)
    first = pairwise_rank_loss(scores, target, times, max_pairs_per_time=23,
                              generator=torch.Generator().manual_seed(19), **kwargs)
    second = pairwise_rank_loss(scores, target, times, max_pairs_per_time=23,
                               generator=torch.Generator().manual_seed(19), **kwargs)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [{"rank_weighting": "unknown"}, {"rank_gap_cap": 0},
                                     {"rank_gap_cap": float("inf")}, {"rank_gap_cap": True}])
def test_pairwise_configuration_validation(kwargs):
    with pytest.raises(ValueError):
        PredictionLossConfig(**kwargs)


def test_pearson_formula_direction_shift_and_equal_time_weights():
    scores = torch.tensor([0., 2., 1., 2., -1., 1., 0.], dtype=torch.float64)
    target = torch.tensor([1., 4., 2., 3., 2., 1., 0.], dtype=torch.float64)
    times = torch.tensor([1, 1, 1, 2, 2, 2, 2])
    epsilon = .2
    expected = []
    for mask in (times == 1, times == 2):
        x = scores[mask] - scores[mask].mean()
        y = target[mask] - target[mask].mean()
        expected.append(1 - (x*y).mean() / math.sqrt(float((x*x).mean()) + epsilon**2)
                        / math.sqrt(float((y*y).mean()) + epsilon**2))
    loss, info = pearson_ic_loss(scores, target, times, epsilon=epsilon)
    torch.testing.assert_close(loss, torch.stack(expected).mean())
    assert info == {"ic_times": 2, "ic_skipped_times": 0}
    shifted, _ = pearson_ic_loss(scores + 7, target - 8, times, epsilon=epsilon)
    reversed_loss, _ = pearson_ic_loss(-scores, target, times, epsilon=epsilon)
    torch.testing.assert_close(loss, shifted)
    torch.testing.assert_close(loss + reversed_loss, torch.tensor(2., dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_pearson_constant_scores_restore_with_finite_gradients(dtype):
    scores = torch.ones(3, dtype=dtype, requires_grad=True)
    loss, info = pearson_ic_loss(scores, torch.tensor([-1., 0., 1.], dtype=dtype),
                                  torch.zeros(3, dtype=torch.long), epsilon=.01)
    assert loss == 1 and info["ic_times"] == 1
    assert loss.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    loss.backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.abs().sum() > 0
    assert scores.grad[0] > 0 and scores.grad[-1] < 0


def test_pearson_mask_constant_targets_and_small_groups():
    scores = torch.arange(9., requires_grad=True)
    target = torch.tensor([0., 1., 2., 3., 3., 3., 1., 2., float("nan")])
    times = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 1])
    loss, info = pearson_ic_loss(scores, target, times, target_valid=torch.arange(9) < 8)
    assert info == {"ic_times": 1, "ic_skipped_times": 2}
    loss.backward()
    torch.testing.assert_close(scores.grad[3:], torch.zeros(6))
    empty, info = pearson_ic_loss(scores, torch.ones(9), times)
    assert empty == 0 and info == {"ic_times": 0, "ic_skipped_times": 3}
    assert torch.isfinite(torch.autograd.grad(empty, scores)[0]).all()


def test_pearson_gradcheck_and_reordering():
    scores = torch.tensor([.3, -.5, .8, 1.2], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([.1, .7, -.3, .2], dtype=torch.float64)
    times = torch.zeros(4, dtype=torch.long)
    assert torch.autograd.gradcheck(lambda x: pearson_ic_loss(x, target, times)[0], (scores,))
    perm = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(pearson_ic_loss(scores, target, times)[0],
                               pearson_ic_loss(scores[perm], target[perm], times[perm])[0])


@pytest.mark.parametrize("epsilon", [0, -1, True, float("nan"), float("inf")])
def test_pearson_epsilon_validation(epsilon):
    with pytest.raises(ValueError):
        pearson_ic_loss(torch.ones(3), torch.arange(3.), torch.zeros(3, dtype=torch.long), epsilon=epsilon)


def test_pearson_all_invalid_retains_input_contract():
    with pytest.raises(ValueError, match="at least one valid"):
        pearson_ic_loss(torch.ones(3), torch.ones(3), torch.zeros(3, dtype=torch.long),
                        target_valid=torch.zeros(3, dtype=torch.bool))


def test_pearson_rejects_overflow_when_finite_variance_and_epsilon_are_added():
    with pytest.raises(ValueError, match="nonfinite"):
        pearson_ic_loss(torch.tensor([-1e19, 0., 1e19]), torch.tensor([-1., 0., 1.]),
                        torch.zeros(3, dtype=torch.long), epsilon=1.73e19)


def test_pearson_rejects_overflowing_centered_moments():
    scores = torch.tensor([1e30, 0., -1e30], requires_grad=True)
    with pytest.raises(ValueError, match="nonfinite"):
        pearson_ic_loss(scores, torch.tensor([1., 0., -1.]), torch.zeros(3, dtype=torch.long))


@pytest.mark.parametrize("epsilon", [1e-30, 1e30, 1e300])
def test_pearson_rejects_unrepresentable_epsilon_square(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        pearson_ic_loss(torch.tensor([0., 1., 2.]), torch.tensor([2., 1., 0.]),
                        torch.zeros(3, dtype=torch.long), epsilon=epsilon)


def test_pearson_decimal_constant_target_skipped_before_centering():
    scores = torch.arange(1000., requires_grad=True)
    target = torch.full((1000,), .1)
    loss, info = pearson_ic_loss(scores, target, torch.zeros(1000, dtype=torch.long))
    assert loss == 0 and info == {"ic_times": 0, "ic_skipped_times": 1}
