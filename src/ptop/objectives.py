"""A default C-compatible objective; callers can supply their own callable."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from jyc_e2e.losses import pairwise_rank_loss, pearson_ic_loss
from .schema import ContractError


class RegressionObjective(nn.Module):
    """Normalized-score Huber with optional pairwise and full-decision Pearson."""

    def __init__(self, *, target_scale=1.0, huber_delta=1.0, rank_weight=0.0,
                 rank_temperature=1.0, rank_min_target_gap=0.0, rank_max_pairs_per_time=8192,
                 rank_weighting="uniform", rank_gap_cap=1.0, ic_weight=0.0, ic_epsilon=1e-4):
        super().__init__()
        positive = dict(target_scale=target_scale, huber_delta=huber_delta,
                        rank_temperature=rank_temperature, rank_gap_cap=rank_gap_cap, ic_epsilon=ic_epsilon)
        nonnegative = dict(rank_weight=rank_weight, rank_min_target_gap=rank_min_target_gap, ic_weight=ic_weight)
        for name, value in {**positive, **nonnegative}.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ContractError(f"{name} must be finite numeric")
            if value < 0 or (name in positive and value == 0):
                raise ContractError(f"invalid objective parameter: {name}")
            setattr(self, name, float(value))
        if type(rank_max_pairs_per_time) is not int or rank_max_pairs_per_time < 1:
            raise ContractError("rank_max_pairs_per_time must be a positive integer")
        if rank_weighting not in {"uniform", "target_gap"}:
            raise ContractError("rank_weighting must be uniform or target_gap")
        self.rank_max_pairs_per_time = rank_max_pairs_per_time
        self.rank_weighting = rank_weighting

    @property
    def requires_full_decision(self) -> bool:
        return self.ic_weight > 0

    def configuration(self) -> dict:
        return {"name": "HuberWithinTimeRank", "target_scale": self.target_scale,
                "huber_delta": self.huber_delta, "rank_weight": self.rank_weight,
                "rank_temperature": self.rank_temperature, "rank_min_target_gap": self.rank_min_target_gap,
                "rank_max_pairs_per_time": self.rank_max_pairs_per_time, "rank_weighting": self.rank_weighting,
                "rank_gap_cap": self.rank_gap_cap, "ic_weight": self.ic_weight, "ic_epsilon": self.ic_epsilon}

    def forward(self, scores, target, decision_id, target_valid):
        if scores.ndim != 1 or scores.shape != target.shape or scores.shape != decision_id.shape:
            raise ContractError("scores, target and decision IDs must be equal-length vectors")
        if target_valid.shape != scores.shape or target_valid.dtype != torch.bool:
            raise ContractError("target_valid must be a boolean vector")
        selected = target_valid
        ic_info = {"ic_times": 0, "ic_skipped_times": 0}
        if not selected.any():
            return scores.sum() * 0.0, {"huber": 0.0, "rank": 0.0, "ic": 0.0, "count": 0, **ic_info}
        scores, target, decision_id = scores[selected].float(), target[selected].float(), decision_id[selected]
        if not torch.isfinite(scores).all() or not torch.isfinite(target).all():
            raise ContractError("valid score/target is nonfinite")
        scaled = target / self.target_scale
        if not torch.isfinite(scaled).all():
            raise ContractError("scaled target is nonfinite; check target_scale")
        huber = F.huber_loss(scores, scaled, delta=self.huber_delta)
        rank = pairwise_rank_loss(
            scores, scaled, decision_id, temperature=self.rank_temperature,
            min_target_gap=self.rank_min_target_gap / self.target_scale,
            max_pairs_per_time=self.rank_max_pairs_per_time,
            rank_weighting=self.rank_weighting, rank_gap_cap=self.rank_gap_cap,
        ) if self.rank_weight else huber * 0.0
        ic = huber * 0.0
        if self.ic_weight:
            ic, ic_info = pearson_ic_loss(scores, scaled, decision_id, epsilon=self.ic_epsilon)
        return huber + self.rank_weight * rank + self.ic_weight * ic, {
            "huber": float(huber.detach().cpu()), "rank": float(rank.detach().cpu()),
            "ic": float(ic.detach().cpu()), "count": len(scores), **ic_info,
        }
