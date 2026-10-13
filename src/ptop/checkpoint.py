"""Atomic, weights-only training checkpoints under A's raw25 and processor contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .bundle import (
    _create_spec_model, _model_spec_dict, _validate_contracts, _validate_state_shapes,
    _validated_model_state, dependency_versions, execution_signature,
)
from .preprocessing import FrozenPreprocessor
from .schema import FEATURES, QLIB_COMMIT, SCHEMA_VERSION, ContractError


CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class CheckpointState:
    """epoch counts completed epochs; resume starts with this epoch index."""

    epoch: int
    global_step: int
    best_metric: float | None
    bad_epochs: int
    config: dict
    preprocessor: FrozenPreprocessor
    model_spec: dict
    contracts: dict
    split: dict
    extra_state: dict


def _json_data(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ContractError(f"{name} 必须为 JSON 配置字典")
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True))
    except (ValueError, TypeError) as error:
        raise ContractError(f"{name} 必须只含有限 JSON 数值和普通配置") from error


def _safe_state(value, name: str, *, clone: bool = True):
    """Only primitives, dense tensors and their containers may enter torch.save."""
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or not torch.isfinite(value).all():
            raise ContractError(f"{name} 张量必须为有限的普通稠密张量")
        return value.detach().cpu().clone() if clone else value
    if isinstance(value, np.generic):
        return _safe_state(value.item(), name, clone=clone)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ContractError(f"{name} 不允许非有限数值")
        return value
    if isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise ContractError(f"{name} 字典键必须为字符串或整数")
        return {key: _safe_state(item, f"{name}.{key}", clone=clone) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_safe_state(item, f"{name}[{index}]", clone=clone) for index, item in enumerate(value)]
        return tuple(items) if isinstance(value, tuple) else items
    raise ContractError(f"{name} 包含不支持的状态类型 {type(value).__name__}")


def _integrity_digest(value) -> str:
    """An internal checksum detects truncation or tensor/metadata corruption."""
    digest = hashlib.sha256()

    def write(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            write(str(tensor.dtype))
            write(list(tensor.shape))
            write(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            write(len(item))
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                write(key)
                write(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"tuple" if isinstance(item, tuple) else b"list")
            write(len(item))
            for element in item:
                write(element)
        else:
            encoded = item if isinstance(item, bytes) else json.dumps(
                item, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)

    write(value)
    return digest.hexdigest()


def _class_name(value) -> str | None:
    return None if value is None else f"{type(value).__module__}.{type(value).__qualname__}"


def _optimizer_layout(model, optimizer) -> list[list[str]]:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ContractError("训练恢复必须提供 torch.optim.Optimizer")
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups, seen = [], set()
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group["params"]:
            key = id(parameter)
            if key not in names or key in seen:
                raise ContractError("优化器包含模型之外或重复的参数")
            seen.add(key)
            group_names.append(names[key])
        groups.append(group_names)
    if any(parameter.requires_grad and id(parameter) not in seen for parameter in model.parameters()):
        raise ContractError("优化器必须包含全部参与联合训练的模型参数")
    return groups


def _rng_state() -> dict:
    algorithm, keys, position, has_gauss, gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"algorithm": algorithm, "keys": torch.tensor(keys.astype(np.int64)),
                  "position": position, "has_gauss": has_gauss, "gaussian": gaussian},
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "backend": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
    }


def _validate_rng(state: dict, *, restore: bool) -> None:
    if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda", "backend"}:
        raise ContractError("checkpoint 随机状态不完整")
    try:
        random.Random().setstate(state["python"])
        torch.Generator(device="cpu").set_state(state["torch_cpu"].cpu())
    except (TypeError, ValueError, RuntimeError, AttributeError) as error:
        raise ContractError("checkpoint Python/Torch CPU 随机状态无效") from error
    numpy = state["numpy"]
    if (not isinstance(numpy, dict) or set(numpy) != {"algorithm", "keys", "position", "has_gauss", "gaussian"}
            or numpy["algorithm"] != "MT19937" or not isinstance(numpy["keys"], torch.Tensor)
            or numpy["keys"].dtype != torch.int64 or numpy["keys"].shape != (624,)
            or (numpy["keys"] < 0).any() or (numpy["keys"] > 2**32 - 1).any()
            or type(numpy["position"]) is not int or not 0 <= numpy["position"] <= 624
            or type(numpy["has_gauss"]) is not int or numpy["has_gauss"] not in (0, 1)
            or type(numpy["gaussian"]) is not float or not math.isfinite(numpy["gaussian"])):
        raise ContractError("checkpoint NumPy 随机状态无效")
    cuda = state["torch_cuda"]
    if not isinstance(cuda, list) or any(
        not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or value.numel() == 0
        for value in cuda
    ):
        raise ContractError("checkpoint CUDA 随机状态无效")
    if restore and cuda and (not torch.cuda.is_available() or torch.cuda.device_count() != len(cuda)):
        raise ContractError("恢复 CUDA 随机状态需要与 checkpoint 相同的 CUDA 设备数量")
    keys = {"deterministic_algorithms", "deterministic_warn_only", "cudnn_benchmark", "cudnn_deterministic",
            "cudnn_tf32", "matmul_tf32"}
    if (not isinstance(state["backend"], dict) or set(state["backend"]) != keys
            or any(type(value) is not bool for value in state["backend"].values())):
        raise ContractError("checkpoint 随机算法/后端配置无效")


def _restore_rng(state: dict) -> None:
    backend = state["backend"]
    torch.use_deterministic_algorithms(backend["deterministic_algorithms"],
                                       warn_only=backend["deterministic_warn_only"])
    torch.backends.cudnn.benchmark = backend["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = backend["cudnn_deterministic"]
    torch.backends.cudnn.allow_tf32 = backend["cudnn_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = backend["matmul_tf32"]
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy["algorithm"], numpy["keys"].cpu().numpy().astype(np.uint32),
                         numpy["position"], numpy["has_gauss"], numpy["gaussian"]))
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _atomic_save(path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path, model, optimizer, *, scheduler=None, scaler=None, epoch: int,
                    global_step: int, best_metric: float | None, bad_epochs: int, config: dict,
                    preprocessor: FrozenPreprocessor, model_spec, contracts: dict[str, str],
                    split: dict, extra_state: dict | None = None) -> None:
    """Save an epoch-boundary checkpoint, replacing an existing file atomically.

    config contains immutable run configuration, including scheduler horizon. Runtime-only
    early stopping of a process belongs outside it. Sampler generators, history and objective
    state may use extra_state; arbitrary Python objects and NumPy arrays are rejected.
    """
    if any(type(value) is not int or value < 0 for value in (epoch, global_step, bad_epochs)):
        raise ContractError("epoch/global_step/bad_epochs 必须为非负整数")
    if best_metric is not None and (isinstance(best_metric, bool) or not isinstance(best_metric, (int, float))
                                    or not math.isfinite(best_metric)):
        raise ContractError("best_metric 必须为有限数值或 None")
    _validate_contracts(contracts)
    config, split = _json_data(config, "config"), _json_data(split, "split")
    if not config or not split:
        raise ContractError("checkpoint 必须保存非空训练配置和数据切分")
    spec = _model_spec_dict(model_spec)
    model_state = _validated_model_state(model, model_spec=spec)
    layout = _optimizer_layout(model, optimizer)
    metadata = {
        "checkpoint_version": CHECKPOINT_VERSION, "schema_version": SCHEMA_VERSION,
        "fields": list(FEATURES), "qlib_commit": QLIB_COMMIT, "dependencies": dependency_versions(),
        "epoch": epoch, "global_step": global_step, "best_metric": best_metric, "bad_epochs": bad_epochs,
        "config": config, "model_spec": spec, "contracts": dict(contracts), "split": split,
        "preprocessor": preprocessor.state_dict(),
        "execution": execution_signature(),
        "trainable_parameters": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
    }
    payload = {
        "metadata": _json_data(metadata, "metadata"), "model": model_state,
        "optimizer": _safe_state(optimizer.state_dict(), "optimizer"),
        "optimizer_layout": layout, "optimizer_class": _class_name(optimizer),
        "scheduler": None if scheduler is None else _safe_state(scheduler.state_dict(), "scheduler"),
        "scheduler_class": _class_name(scheduler),
        "scaler": None if scaler is None else _safe_state(scaler.state_dict(), "scaler"),
        "scaler_class": _class_name(scaler), "rng": _safe_state(_rng_state(), "rng"),
        "extra": _safe_state({} if extra_state is None else extra_state, "extra_state"),
    }
    if not isinstance(payload["extra"], dict):
        raise ContractError("extra_state 必须为状态字典")
    _validate_rng(payload["rng"], restore=False)
    payload["digest"] = _integrity_digest(payload)
    _atomic_save(path, payload)


def _read_checkpoint(path, *, map_location="cpu") -> dict:
    if not isinstance(map_location, (str, torch.device)):
        raise ContractError("map_location 必须为设备名或 torch.device")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as error:
        raise ContractError("checkpoint 缺失、损坏或无法按 weights_only 安全读取") from error
    keys = {"metadata", "model", "optimizer", "optimizer_layout", "optimizer_class", "scheduler",
            "scheduler_class", "scaler", "scaler_class", "rng", "extra", "digest"}
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ContractError("checkpoint 状态字段缺失或未知")
    _safe_state({key: value for key, value in payload.items() if key != "digest"}, "checkpoint", clone=False)
    if payload["digest"] != _integrity_digest({key: value for key, value in payload.items() if key != "digest"}):
        raise ContractError("checkpoint 完整性校验失败")
    metadata = payload["metadata"]
    expected_keys = {"checkpoint_version", "schema_version", "fields", "qlib_commit", "dependencies", "epoch",
                     "global_step", "best_metric", "bad_epochs", "config", "model_spec", "contracts", "split",
                     "preprocessor", "trainable_parameters", "execution"}
    if not isinstance(metadata, dict) or set(metadata) != expected_keys:
        raise ContractError("checkpoint 元数据字段缺失或未知")
    if (type(metadata["checkpoint_version"]) is not int or metadata["checkpoint_version"] != CHECKPOINT_VERSION
            or metadata["schema_version"] != SCHEMA_VERSION or metadata["fields"] != list(FEATURES)
            or metadata["qlib_commit"] != QLIB_COMMIT):
        raise ContractError("checkpoint 版本、schema、Qlib 或字段顺序不一致")
    if metadata["dependencies"] != dependency_versions():
        raise ContractError("checkpoint 依赖版本与运行环境不一致")
    if metadata["execution"] != execution_signature():
        raise ContractError("checkpoint 推理执行源码与当前实现不一致")
    _validate_contracts(metadata["contracts"])
    if (_model_spec_dict(metadata["model_spec"]) != metadata["model_spec"]
            or not _json_data(metadata["config"], "config") or not _json_data(metadata["split"], "split")):
        raise ContractError("checkpoint 缺少完整模型配置、训练配置或数据切分")
    if any(type(metadata[name]) is not int or metadata[name] < 0 for name in ("epoch", "global_step", "bad_epochs")):
        raise ContractError("checkpoint 训练计数无效")
    metric = metadata["best_metric"]
    if metric is not None and (isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric)):
        raise ContractError("checkpoint 最佳指标无效")
    trainable = metadata["trainable_parameters"]
    if not isinstance(trainable, list) or any(not isinstance(name, str) for name in trainable) or len(set(trainable)) != len(trainable):
        raise ContractError("checkpoint 可训练参数记录无效")
    if not isinstance(payload["extra"], dict):
        raise ContractError("checkpoint extra_state 无效")
    _validate_rng(payload["rng"], restore=False)
    FrozenPreprocessor.from_state_dict(metadata["preprocessor"])
    return payload


def _validate_optimizer_state(model, optimizer, state: dict, layout: list[list[str]]) -> None:
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ContractError("checkpoint 优化器状态不完整")
    groups, moments = state["param_groups"], state["state"]
    if not isinstance(groups, list) or not isinstance(moments, dict) or len(groups) != len(layout):
        raise ContractError("checkpoint 优化器分组不一致")
    parameters = dict(model.named_parameters())
    owned = {}
    for names, group in zip(layout, groups, strict=True):
        if not isinstance(group, dict) or not isinstance(group.get("params"), list) or len(group["params"]) != len(names):
            raise ContractError("checkpoint 优化器参数数量不一致")
        for name, key in zip(names, group["params"], strict=True):
            if type(key) is not int or key in owned:
                raise ContractError("checkpoint 优化器参数索引重复或无效")
            owned[key] = parameters[name]
    if set(moments).difference(owned):
        raise ContractError("checkpoint 优化器状态含未知参数")
    for key, value in moments.items():
        if not isinstance(value, dict):
            raise ContractError("checkpoint 参数优化器状态必须为字典")
        for name, tensor in value.items():
            if isinstance(tensor, torch.Tensor) and name != "step" and tensor.shape != owned[key].shape:
                raise ContractError(f"checkpoint 优化器状态形状与参数不一致：{name}")


def load_checkpoint(path, model, optimizer, *, scheduler=None, scaler=None, expected_config: dict | None = None,
                    expected_model_spec=None, expected_contracts: dict | None = None,
                    expected_split: dict | None = None, expected_preprocessor: FrozenPreprocessor | None = None,
                    restore_rng: bool = True, map_location="cpu") -> CheckpointState:
    """Validate all contracts before restoring model/optimizer/scheduler/AMP/RNG.

    Providing expected_* binds the saved run to the current data and immutable configuration.
    No processor is fitted; the returned preprocessor is reconstructed from its frozen state.
    """
    if type(restore_rng) is not bool:
        raise ContractError("restore_rng 必须为 bool")
    payload = _read_checkpoint(path, map_location=map_location)
    metadata = payload["metadata"]
    comparisons = ((expected_config, metadata["config"], "训练配置"),
                   (expected_contracts, metadata["contracts"], "合同版本"),
                   (expected_split, metadata["split"], "数据切分"))
    for expected, actual, name in comparisons:
        if expected is not None and _json_data(expected, name) != actual:
            raise ContractError(f"checkpoint {name}与当前运行不一致")
    if expected_model_spec is not None and _model_spec_dict(expected_model_spec) != metadata["model_spec"]:
        raise ContractError("checkpoint 模型配置与当前运行不一致")
    processor = FrozenPreprocessor.from_state_dict(metadata["preprocessor"])
    if expected_preprocessor is not None and expected_preprocessor.fingerprint != processor.fingerprint:
        raise ContractError("checkpoint 冻结处理器与当前运行不一致")
    _validated_model_state(model, model_spec=metadata["model_spec"])
    _validate_state_shapes(payload["model"], model.state_dict())
    if metadata["trainable_parameters"] != [name for name, parameter in model.named_parameters() if parameter.requires_grad]:
        raise ContractError("checkpoint 可训练参数集合或顺序与当前模型不一致")
    layout = _optimizer_layout(model, optimizer)
    if payload["optimizer_class"] != _class_name(optimizer) or payload["optimizer_layout"] != layout:
        raise ContractError("checkpoint 优化器类型、参数名称或顺序与当前运行不一致")
    _validate_optimizer_state(model, optimizer, payload["optimizer"], layout)
    for name, instance in (("scheduler", scheduler), ("scaler", scaler)):
        if payload[f"{name}_class"] != _class_name(instance) or (payload[name] is None) != (instance is None):
            raise ContractError(f"checkpoint {name} 类型或启用配置与当前运行不一致")
        if instance is not None and (not isinstance(payload[name], dict)
                                     or set(payload[name]) != set(instance.state_dict())):
            raise ContractError(f"checkpoint {name} 状态字段与当前运行不一致")
    _validate_rng(payload["rng"], restore=restore_rng)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        _restore_rng(payload["rng"])
    return CheckpointState(
        metadata["epoch"], metadata["global_step"], metadata["best_metric"], metadata["bad_epochs"],
        metadata["config"], processor, metadata["model_spec"], metadata["contracts"], metadata["split"], payload["extra"],
    )


def load_inference_checkpoint(path, *, device="cpu", expected_contracts: dict | None = None):
    """Read learned weights without restoring optimizer or random state."""
    payload = _read_checkpoint(path)
    metadata = payload["metadata"]
    if expected_contracts is not None and _json_data(expected_contracts, "contracts") != metadata["contracts"]:
        raise ContractError("checkpoint 合同版本与当前推理不一致")
    with torch.random.fork_rng(devices=[]):
        model = _create_spec_model(metadata["model_spec"])
    _validate_state_shapes(payload["model"], model.state_dict())
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    processor = FrozenPreprocessor.from_state_dict(metadata["preprocessor"])
    return model, processor, dict(metadata)
