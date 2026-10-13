"""C-compatible objectives; callers can also supply their own callable."""

from dataclasses import replace
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from jyc_e2e.losses import (SCORE_UNITS, PredictionLossConfig, PredictionObjective,
                            pairwise_rank_loss)
from .schema import ContractError

NORMALIZED_SCORE_UNITS = "training-target-scale normalized score"


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
        return {"name": "HuberWithinTimeRank", "score_units": NORMALIZED_SCORE_UNITS,
                "target_scale": self.target_scale,
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


def prediction_target_scale(targets, target_valid=None) -> float:
    """Fix the training-label scale once, outside every forward pass.

    Mirrors the trainer's own calibration (std with ddof=0 over valid finite labels)
    so an objective built here and ``Trainer.target_scale`` agree by construction.
    """
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 1:
        raise ContractError("targets must be a 1-D array of labels")
    if target_valid is not None:
        mask = np.asarray(target_valid, dtype=bool)
        if mask.shape != values.shape:
            raise ContractError("target_valid must align with targets")
        values = values[mask]
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ContractError("need at least two finite training labels to fix the scale")
    scale = float(values.std(ddof=0))
    if not math.isfinite(scale) or scale <= 0:
        raise ContractError("training-label scale is zero or nonfinite")
    return scale


def build_prediction_objective(config: PredictionLossConfig | None = None, *,
                               target_scale: float | None = None) -> PredictionObjective:
    """Mount the stage-one C objective as a trainer-ready objective.

    The result satisfies B's trainer contract: it exposes ``configuration()`` and accepts
    ``(scores, target, decision_id, target_valid)`` positionally, so it drops in exactly
    where ``RegressionObjective`` does. One difference matters: this objective expects the
    head to emit raw return units (``SCORE_UNITS``) while ``RegressionObjective`` expects
    an already-normalised ``y / target_scale``. Fix the convention per run and never mix
    them across a train/inference pair.

    ``target_scale`` should come from ``prediction_target_scale`` on the training fold.
    It is frozen here and is never re-fitted in forward, matching
    docs/stage_one_losses.md. Pass ``None`` to keep the config's own value.
    """
    if config is not None and not isinstance(config, PredictionLossConfig):
        raise ContractError("build_prediction_objective needs a PredictionLossConfig")
    config = PredictionLossConfig() if config is None else config
    if target_scale is not None:
        if isinstance(target_scale, bool) or not isinstance(target_scale, (int, float)):
            raise ContractError("target_scale must be a number")
        if not math.isfinite(target_scale) or target_scale <= 0:
            raise ContractError("target_scale must be finite and positive")
        config = replace(config, target_scale=float(target_scale))
    return PredictionObjective(config)

