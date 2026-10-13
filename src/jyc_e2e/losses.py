"""Stage-one objectives; model predictions and labels use raw return units."""

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PredictionLossConfig:
    regression: Literal["mse", "huber"] = "huber"
    reduction: Literal["time", "sample"] = "time"
    target_scale: float = 1.0
    huber_delta: float = 1.0
    rank_weight: float = 0.0
    rank_temperature: float = 1.0
    min_target_gap: float = 0.0
    max_pairs_per_time: int = 8192

    def __post_init__(self) -> None:
        if self.regression not in {"mse", "huber"}:
            raise ValueError("regression must be mse or huber")
        if self.reduction not in {"time", "sample"}:
            raise ValueError("reduction must be time or sample")
        for name in ("target_scale", "huber_delta", "rank_temperature"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("rank_weight", "min_target_gap"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (isinstance(self.max_pairs_per_time, bool)
                or not isinstance(self.max_pairs_per_time, int)
                or self.max_pairs_per_time < 1):
            raise ValueError("max_pairs_per_time must be a positive integer")


def _valid_inputs(scores, target, times, target_valid):
    if scores.ndim != 1 or scores.shape != target.shape or scores.shape != times.shape:
        raise ValueError("scores, target, and times must be equal-length vectors")
    if not scores.is_floating_point() or not target.is_floating_point():
        raise TypeError("scores and target must be floating-point tensors")
    if times.dtype != torch.int64:
        raise TypeError("times must contain int64 decision identifiers")
    if scores.device != target.device or scores.device != times.device:
        raise ValueError("scores, target, and times must be on the same device")
    if not torch.isfinite(scores).all():
        raise ValueError("all model scores must be finite")
    if target_valid is None:
        target_valid = torch.ones_like(times, dtype=torch.bool)
    if (target_valid.shape != scores.shape or target_valid.dtype != torch.bool
            or target_valid.device != scores.device):
        raise ValueError("target_valid must be a same-device boolean vector")
    if not target_valid.any():
        raise ValueError("batch must contain at least one valid target")
    if not torch.isfinite(target[target_valid]).all():
        raise ValueError("valid targets must be finite; mask missing labels explicitly")
    # Loss arithmetic retains at least float32 precision under mixed precision.
    dtype = torch.promote_types(scores.dtype, target.dtype)
    if dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    return scores[target_valid].to(dtype), target[target_valid].to(dtype), times[target_valid]


def _rank_terms(scores, target, times, config, generator):
    losses = []
    counts = {"rank_times": 0, "considered_pairs": 0, "valid_pairs": 0,
              "singleton_times": 0, "no_pair_times": 0}
    for time in torch.unique(times):
        indices = torch.nonzero(times == time, as_tuple=True)[0]
        size = indices.numel()
        if size < 2:
            counts["singleton_times"] += 1
            continue
        if size * (size - 1) // 2 <= config.max_pairs_per_time:
            left, right = torch.triu_indices(size, size, offset=1, device=scores.device)
        else:
            # Uniform distinct-index pairs with replacement, with O(pair cap) memory.
            left = torch.randint(size, (config.max_pairs_per_time,),
                                 device=scores.device, generator=generator)
            right = torch.randint(size - 1, (config.max_pairs_per_time,),
                                  device=scores.device, generator=generator)
            right = right + (right >= left).to(right.dtype)
        gap = target[indices[left]] - target[indices[right]]
        counts["considered_pairs"] += gap.numel()
        usable = gap.abs() > config.min_target_gap
        left, right, gap = left[usable], right[usable], gap[usable]
        counts["valid_pairs"] += gap.numel()
        if not gap.numel():
            counts["no_pair_times"] += 1
            continue
        difference = (scores[indices[left]] - scores[indices[right]]) / config.target_scale
        losses.append(F.softplus(-gap.sign() * difference / config.rank_temperature).mean())
    counts["rank_times"] = len(losses)
    loss = torch.stack(losses).mean() if losses else scores[:0].sum()
    return loss, counts


class PredictionObjective(nn.Module):
    """Return loss and detached logs; scales are fixed, never fitted in forward.

    target_scale and min_target_gap use raw return units; huber_delta and
    rank_temperature use scaled units. All valid times have equal rank weight.
    """

    def __init__(self, config: PredictionLossConfig | None = None):
        super().__init__()
        self.config = config if config is not None else PredictionLossConfig()

    def forward(
        self, scores: torch.Tensor, target: torch.Tensor, times: torch.Tensor, *,
        target_valid: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
        raw_count = scores.numel()
        scores, target, times = _valid_inputs(scores, target, times, target_valid)
        cfg = self.config
        scaled_scores, scaled_target = scores / cfg.target_scale, target / cfg.target_scale
        if cfg.regression == "mse":
            per_sample = F.mse_loss(scaled_scores, scaled_target, reduction="none")
        else:
            per_sample = F.huber_loss(scaled_scores, scaled_target,
                                      delta=cfg.huber_delta, reduction="none")
        unique_times, inverse = torch.unique(times, return_inverse=True)
        if cfg.reduction == "sample":
            regression = per_sample.mean()
        else:
            sums = per_sample.new_zeros(unique_times.numel()).scatter_add(0, inverse, per_sample)
            counts = torch.bincount(inverse, minlength=unique_times.numel())
            regression = (sums / counts).mean()
        rank = scores[:0].sum()
        rank_info = {"rank_times": 0, "considered_pairs": 0, "valid_pairs": 0,
                     "singleton_times": 0, "no_pair_times": 0}
        if cfg.rank_weight:
            rank, rank_info = _rank_terms(scores, target, times, cfg, generator)
        weighted_rank = cfg.rank_weight * rank
        total = regression + weighted_rank
        if not torch.isfinite(total):
            raise ValueError("nonfinite loss; check return scale and score magnitudes")
        return total, {
            "loss": total.detach(), "regression_loss": regression.detach(),
            "rank_loss": rank.detach(), "weighted_rank_loss": weighted_rank.detach(),
            "valid_samples": scores.numel(), "masked_samples": raw_count - scores.numel(),
            "valid_times": unique_times.numel(), **rank_info,
        }


def pairwise_rank_loss(
    scores: torch.Tensor, target: torch.Tensor, times: torch.Tensor, *,
    min_target_gap: float = 0.0, max_pairs_per_time: int = 8192,
    target_scale: float = 1.0, temperature: float = 1.0,
    target_valid: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Equal-time rank loss; large groups use bounded sampling with replacement."""
    cfg = PredictionLossConfig(min_target_gap=min_target_gap,
                               max_pairs_per_time=max_pairs_per_time,
                               target_scale=target_scale, rank_temperature=temperature)
    scores, target, times = _valid_inputs(scores, target, times, target_valid)
    loss, _ = _rank_terms(scores, target, times, cfg, generator)
    return loss


def score_loss(
    scores: torch.Tensor, target: torch.Tensor, times: torch.Tensor, *,
    rank_weight: float = 0.0, huber_delta: float = 1.0,
) -> torch.Tensor:
    """Compatibility wrapper: raw-unit Huber, sample mean, optional rank loss."""
    cfg = PredictionLossConfig(reduction="sample", rank_weight=rank_weight,
                               huber_delta=huber_delta)
    loss, _ = PredictionObjective(cfg)(scores, target, times)
    return loss
