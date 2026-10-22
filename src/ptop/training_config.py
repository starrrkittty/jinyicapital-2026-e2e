"""B's reproducible optimizer, validation and resource settings."""

from dataclasses import asdict, dataclass
import math

from .schema import ContractError


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    clip_norm: float = 1.0
    patience: int = 3
    min_delta: float = 0.0
    selection_metric: str = "ic_mean"
    mode: str = "max"
    seed: int = 7
    precision: str = "fp32"
    deterministic: bool = True
    microbatch_size: int = 256
    objective_scope: str = "microbatch"
    target_scale: float | None = None
    rank_weight: float = 0.0
    rank_temperature: float = 1.0
    rank_min_target_gap: float = 0.0
    rank_max_pairs_per_time: int = 8192
    rank_weighting: str = "uniform"
    rank_gap_cap: float = 1.0
    ic_weight: float = 0.0
    ic_epsilon: float = 1e-4
    huber_delta: float = 1.0
    log_every: int = 10
    amp_initial_scale: float = 1024.0

    def __post_init__(self):
        for name in ("epochs", "patience", "microbatch_size", "log_every", "rank_max_pairs_per_time"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ContractError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ContractError("seed must be an integer in [0, 2**32)")
        for name in ("learning_rate", "min_learning_rate", "weight_decay", "clip_norm", "min_delta", "rank_weight", "huber_delta", "amp_initial_scale",
                     "rank_temperature", "rank_min_target_gap", "rank_gap_cap", "ic_weight", "ic_epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ContractError(f"{name} must be finite and nonnegative")
        if self.learning_rate <= 0 or self.clip_norm <= 0 or self.huber_delta <= 0 or self.amp_initial_scale <= 0:
            raise ContractError("learning_rate, clip_norm, huber_delta and AMP initial scale must be positive")
        if self.rank_temperature <= 0 or self.rank_gap_cap <= 0 or self.ic_epsilon <= 0:
            raise ContractError("rank_temperature, rank_gap_cap and ic_epsilon must be positive")
        if self.rank_weighting not in {"uniform", "target_gap"}:
            raise ContractError("rank_weighting must be uniform or target_gap")
        if self.objective_scope not in {"microbatch", "decision"}:
            raise ContractError("objective_scope must be microbatch or decision")
        if self.ic_weight and self.objective_scope != "decision":
            raise ContractError("IC loss requires objective_scope='decision' for full cross sections")
        if self.min_learning_rate > self.learning_rate:
            raise ContractError("min_learning_rate exceeds learning_rate")
        if self.target_scale is not None and (
            isinstance(self.target_scale, bool) or not isinstance(self.target_scale, (float, int))
            or not math.isfinite(self.target_scale) or self.target_scale <= 0
        ):
            raise ContractError("target_scale must be finite and positive")
        if self.mode not in {"min", "max"} or not isinstance(self.selection_metric, str) or not self.selection_metric:
            raise ContractError("selection requires a metric and min/max direction")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ContractError("precision must be fp32, bf16 or fp16")
        if type(self.deterministic) is not bool:
            raise ContractError("deterministic must be boolean")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, config: dict) -> "TrainingConfig":
        if not isinstance(config, dict):
            raise ContractError("training settings must be a JSON object")
        try:
            return cls(**config)
        except TypeError as error:
            raise ContractError(f"unknown or malformed training settings: {error}") from error
