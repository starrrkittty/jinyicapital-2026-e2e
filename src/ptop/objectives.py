"""A default C-compatible objective; callers can supply their own callable."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from jyc_e2e.losses import pairwise_rank_loss
from .schema import ContractError


class RegressionObjective(nn.Module):
    """Huber on a frozen training-label scale, with optional within-time ranking."""

    def __init__(self, *, target_scale=1.0, huber_delta=1.0, rank_weight=0.0):
        super().__init__()
        if not math.isfinite(target_scale) or target_scale <= 0:
            raise ContractError("target scale must be finite and positive")
        if not math.isfinite(huber_delta) or huber_delta <= 0 or not math.isfinite(rank_weight) or rank_weight < 0:
            raise ContractError("invalid Huber delta or ranking weight")
        self.target_scale = float(target_scale)
        self.huber_delta = float(huber_delta)
        self.rank_weight = float(rank_weight)

    def configuration(self) -> dict:
        return {"name": "HuberWithinTimeRank", "target_scale": self.target_scale,
                "huber_delta": self.huber_delta, "rank_weight": self.rank_weight}

    def forward(self, scores, target, decision_id, target_valid):
        if scores.ndim != 1 or scores.shape != target.shape or scores.shape != decision_id.shape:
            raise ContractError("scores, target and decision IDs must be equal-length vectors")
        if target_valid.shape != scores.shape or target_valid.dtype != torch.bool:
            raise ContractError("target_valid must be a boolean vector")
        selected = target_valid
        if not selected.any():
            return scores.sum() * 0.0, {"huber": 0.0, "rank": 0.0, "count": 0}
        scores, target, decision_id = scores[selected].float(), target[selected].float(), decision_id[selected]
        if not torch.isfinite(scores).all() or not torch.isfinite(target).all():
            raise ContractError("valid score/target is nonfinite")
        scaled = target / self.target_scale
        huber = F.huber_loss(scores, scaled, delta=self.huber_delta)
        rank = pairwise_rank_loss(scores, scaled, decision_id) if self.rank_weight else huber * 0.0
        return huber + self.rank_weight * rank, {
            "huber": float(huber.detach().cpu()), "rank": float(rank.detach().cpu()), "count": len(scores),
        }
