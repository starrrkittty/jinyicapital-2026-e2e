"""Configurable B score models built on the authoritative A input encoders."""

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields as dataclass_fields
from typing import Any

import torch
from torch import Tensor, nn

from .backbones import GRUBackbone, SelectiveSSMBackbone, TCNBackbone, TemporalPool, TransformerBackbone
from .encoders import (ALSTMEncoder, DeepLOBEncoder, GroupedNetEncoder, InputEncoder,
                       PatchEncoder, RawEncoder, UnifiedNetEncoder, field_indices)
from .schema import BOOK12_FIELDS, FEATURES, ContractError


LEGACY_MODEL_NAMES = ("R0", "R1", "R1-Book", "Net64", "Net192", "GroupedNet", "Patch1", "Patch5", "DeepLOB")
MODERN_MODEL_NAMES = ("B-GRU", "B-Transformer", "B-TCN", "B-SelectiveSSM")
ENCODER_NAMES = ("Raw", "ALSTM", "ALSTM-Book", "Net64", "Net192", "GroupedNet", "Patch1", "Patch5", "DeepLOB")


@dataclass(frozen=True)
class ModelSpec:
    name: str = "R1"
    encoder: str = "ALSTM"
    width: int = 64
    layers: int = 2
    pooling: str = "last"
    lookback: int = 120
    dropout: float = 0.0
    heads: int = 4
    ff_multiplier: int = 4
    kernel_size: int = 3
    state_dim: int = 16
    head_width: int = 64
    causal: bool = True
    fields: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name not in LEGACY_MODEL_NAMES + MODERN_MODEL_NAMES:
            raise ContractError(f"未知模型配置 {self.name!r}")
        if not isinstance(self.encoder, str) or self.encoder not in ENCODER_NAMES:
            raise ContractError(f"未知编码器配置 {self.encoder!r}")
        for name in ("width", "layers", "lookback", "heads", "ff_multiplier", "kernel_size", "state_dim", "head_width"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ContractError(f"ModelSpec.{name} 必须为正整数")
        if type(self.dropout) not in (float, int) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ContractError("ModelSpec.dropout 必须为 [0,1) 范围内的有限数值")
        if type(self.causal) is not bool:
            raise ContractError("ModelSpec.causal 必须为 bool")
        if not isinstance(self.pooling, str) or self.pooling not in ("last", "mean", "attention"):
            raise ContractError("ModelSpec.pooling 必须为 last、mean 或 attention")
        if self.fields is not None:
            if not isinstance(self.fields, tuple) or any(not isinstance(item, str) for item in self.fields):
                raise ContractError("ModelSpec.fields 必须为字段 tuple 或 None；JSON 配置请使用 from_dict")
            field_indices(self.fields)
            if self.encoder not in ("Raw", "ALSTM"):
                raise ContractError("自定义 fields 只支持 Raw 和 ALSTM 编码器")
        if self.name == "B-Transformer" and self.width % self.heads != 0:
            raise ContractError("Transformer width 必须能被 heads 整除")
        if self.name in LEGACY_MODEL_NAMES:
            defaults = {item.name: item.default for item in dataclass_fields(self)}
            changed = [name for name, value in asdict(self).items()
                       if name not in ("name", "lookback") and value != defaults[name]]
            if changed:
                raise ContractError(f"A 固定旧模型不接受可变架构参数 {changed}；请使用 B-* 模型")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["fields"] = list(self.fields) if self.fields is not None else None
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelSpec":
        if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
            raise ContractError("模型配置必须为字符串键的对象")
        unknown = set(value).difference(item.name for item in dataclass_fields(cls))
        if unknown:
            raise ContractError(f"模型配置包含未知字段：{sorted(unknown)}")
        arguments = dict(value)
        if arguments.get("fields") is not None:
            if not isinstance(arguments["fields"], (tuple, list)):
                raise ContractError("fields 必须为字段列表或 None")
            arguments["fields"] = tuple(arguments["fields"])
        return cls(**arguments)


def create_encoder(spec: ModelSpec) -> InputEncoder:
    selected = FEATURES if spec.fields is None else spec.fields
    if spec.encoder == "Raw":
        return RawEncoder(selected)
    if spec.encoder == "ALSTM":
        return ALSTMEncoder(selected)
    if spec.encoder == "ALSTM-Book":
        return ALSTMEncoder(BOOK12_FIELDS)
    if spec.encoder in ("Net64", "Net192"):
        return UnifiedNetEncoder(output_dim=int(spec.encoder[3:]))
    if spec.encoder == "GroupedNet":
        return GroupedNetEncoder()
    if spec.encoder in ("Patch1", "Patch5"):
        return PatchEncoder(patch_size=int(spec.encoder[-1]))
    if spec.encoder == "DeepLOB":
        return DeepLOBEncoder()
    raise ContractError(f"未知编码器配置 {spec.encoder!r}")


class TemporalScoreModel(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        if spec.name not in MODERN_MODEL_NAMES:
            raise ContractError("TemporalScoreModel 需要 B-* 模型配置")
        self.spec = spec
        self.encoder = create_encoder(spec)  # Registered, trainable, saved with the whole model.
        arguments = (self.encoder.output_dim, spec.width, spec.layers, spec.dropout)
        if spec.name == "B-GRU":
            self.backbone = GRUBackbone(*arguments)
        elif spec.name == "B-Transformer":
            self.backbone = TransformerBackbone(*arguments, spec.heads, spec.ff_multiplier, spec.causal)
        elif spec.name == "B-TCN":
            self.backbone = TCNBackbone(*arguments, spec.kernel_size)
        else:
            self.backbone = SelectiveSSMBackbone(*arguments, spec.state_dim, spec.kernel_size)
        self.pool = TemporalPool(spec.width, spec.pooling)
        self.head = nn.Sequential(nn.Linear(spec.width, spec.head_width), nn.GELU(), nn.Dropout(spec.dropout),
                                  nn.Linear(spec.head_width, 1))

    def extra_repr(self) -> str:
        return f"spec={self.spec!r}"

    def forward(self, x: Tensor, observed: Tensor, step_valid: Tensor) -> Tensor:
        if x.ndim == 3 and x.shape[1] > self.spec.lookback:
            raise ContractError(f"输入窗口长度 {x.shape[1]} 超过 ModelSpec.lookback={self.spec.lookback}")
        output, valid, lengths = self.backbone(*self.encoder(x, observed, step_valid))
        score = self.head(self.pool(output, valid, lengths)).squeeze(-1)
        if not torch.isfinite(score).all():
            raise ContractError("模型 score 包含非有限数值；请检查输入缩放、参数和训练稳定性")
        return score


def create_score_model(spec: ModelSpec | Mapping[str, Any] | str = ModelSpec()) -> nn.Module:
    if isinstance(spec, str):
        spec = ModelSpec(name=spec)
    elif isinstance(spec, Mapping):
        spec = ModelSpec.from_dict(spec)
    if not isinstance(spec, ModelSpec):
        raise ContractError("create_score_model 需要 ModelSpec、模型配置对象或模型名称")
    if spec.name in LEGACY_MODEL_NAMES:
        # Avoid a circular import when bundle's new-version loader calls this factory.
        from .bundle import create_model

        return create_model(spec.name)
    return TemporalScoreModel(spec)
