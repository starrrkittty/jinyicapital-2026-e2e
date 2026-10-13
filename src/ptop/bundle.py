import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import torch

from ._vendor.qlib import COMPONENTS_VERSION
from .encoders import GroupedNetEncoder, PatchEncoder, UnifiedNetEncoder
from .data import _validate_mapping
from .models import AttentionModel, DeepLOBModel, R0Model, R1Model
from .preprocessing import FrozenPreprocessor
from .schema import BOOK12_FIELDS, FEATURES, QLIB_COMMIT, SCHEMA_VERSION, ContractError


BUNDLE_VERSION = 2
MODEL_NAMES = ("R0", "R1", "R1-Book", "Net64", "Net192", "GroupedNet", "Patch1", "Patch5", "DeepLOB")
REQUIRED_CONTRACTS = ("source", "sampling", "universe", "visibility", "label", "training_scope")


def create_model(name: str):
    if name == "R0":
        return R0Model()
    if name == "R1":
        return R1Model()
    if name == "R1-Book":
        return R1Model(fields=BOOK12_FIELDS)
    if name in ("Net64", "Net192"):
        return AttentionModel(UnifiedNetEncoder(output_dim=int(name[3:])))
    if name == "GroupedNet":
        return AttentionModel(GroupedNetEncoder())
    if name in ("Patch1", "Patch5"):
        return AttentionModel(PatchEncoder(patch_size=int(name[-1])))
    if name == "DeepLOB":
        return DeepLOBModel()
    raise ContractError(f"未知模型配置 {name!r}")


def dependency_versions() -> dict[str, str]:
    return {name: version(name) for name in ("torch", "numpy", "pandas", "pyarrow")}


def _digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _validate_tensor_state(state: dict) -> None:
    for name, value in state.items():
        if not torch.isfinite(value).all():
            raise ContractError(f"模型状态包含非有限数值：{name}")
        if name.endswith(("running_var", "num_batches_tracked")) and (value < 0).any():
            raise ContractError(f"BatchNorm 状态必须非负：{name}")


def save_bundle(path, model, preprocessor: FrozenPreprocessor, *, model_name: str,
                window_length: int, contracts: dict[str, str], seed: int, training_record: dict,
                instrument_map: dict[int, str]) -> None:
    if type(window_length) is not int or window_length <= 0:
        raise ContractError("窗口长度必须为正整数")
    if type(seed) is not int or not isinstance(training_record, dict) or not training_record:
        raise ContractError("模型包必须包含随机种子和运行记录")
    if set(contracts) != set(REQUIRED_CONTRACTS) or any(
        not isinstance(value, str) or not value.strip() for value in contracts.values()
    ):
        raise ContractError(f"模型包必须明确提供合同版本 {REQUIRED_CONTRACTS}")
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    mapping = _validate_mapping(instrument_map)
    with torch.random.fork_rng(devices=[]):
        template = create_model(model_name)
    expected_state = template.state_dict()
    if set(state) != set(expected_state) or any(
        value.shape != expected_state[name].shape or value.dtype != expected_state[name].dtype
        for name, value in state.items()
    ):
        raise ContractError("模型权重类型、形状或名称与保存配置不一致")
    template.load_state_dict(state, strict=True)
    if type(template) is not type(model) or type(template.encoder) is not type(model.encoder):
        raise ContractError("模型类型与保存配置不一致")
    if repr(template) != repr(model):
        raise ContractError("模型模块配置与固定候选规格不一致")
    if template.encoder.fields != model.encoder.fields or template.encoder.output_dim != model.encoder.output_dim:
        raise ContractError("编码器字段或输出宽度与保存配置不一致")
    if getattr(template.encoder, "patch_size", None) != getattr(model.encoder, "patch_size", None):
        raise ContractError("patch 配置与保存配置不一致")
    _validate_tensor_state(state)
    # 导出处理器会检查其冻结资格；在创建目录前完成所有合同验证。
    processor_state = preprocessor.state_dict()
    manifest = {
        "bundle_version": BUNDLE_VERSION, "schema_version": SCHEMA_VERSION,
        "fields": list(FEATURES), "qlib_commit": QLIB_COMMIT,
        "qlib_components_version": COMPONENTS_VERSION,
        "dependencies": dependency_versions(), "model_name": model_name,
        "window_length": window_length, "contracts": contracts, "seed": seed,
        "training_record": training_record,
        "instrument_map": [[int(key), value] for key, value in sorted(mapping.items())],
    }
    json.dumps(manifest, allow_nan=False)
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    tensors = {name: {"dtype": str(tensor.dtype).removeprefix("torch."),
                       "shape": list(tensor.shape), "data": tensor.reshape(-1).tolist()}
               for name, tensor in state.items()}
    (destination / "model.json").write_text(json.dumps(tensors, allow_nan=False), encoding="utf-8")
    (destination / "preprocessor.json").write_text(
        json.dumps(processor_state, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    manifest["files"] = {name: _digest(destination / name) for name in ("model.json", "preprocessor.json")}
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def load_bundle(path):
    source = Path(path)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    required = {"bundle_version", "schema_version", "fields", "qlib_commit", "qlib_components_version", "dependencies", "model_name",
                "window_length", "contracts", "seed", "training_record", "instrument_map", "files"}
    if set(manifest) != required:
        raise ContractError("模型包配置字段缺失或未知")
    if manifest["bundle_version"] != BUNDLE_VERSION or manifest["schema_version"] != SCHEMA_VERSION:
        raise ContractError("模型包或 schema 版本不一致")
    if manifest["fields"] != list(FEATURES) or manifest["qlib_commit"] != QLIB_COMMIT:
        raise ContractError("模型包字段顺序或 Qlib 来源不一致")
    if manifest["qlib_components_version"] != COMPONENTS_VERSION:
        raise ContractError("模型包内置 Qlib 组件版本不一致")
    if manifest["dependencies"] != dependency_versions():
        raise ContractError("模型包依赖版本与运行环境不一致")
    if type(manifest["window_length"]) is not int or manifest["window_length"] <= 0:
        raise ContractError("模型包窗口配置无效")
    if set(manifest["contracts"]) != set(REQUIRED_CONTRACTS) or any(
        not isinstance(value, str) or not value.strip() for value in manifest["contracts"].values()
    ):
        raise ContractError("模型包合同版本缺失")
    if type(manifest["seed"]) is not int or not isinstance(manifest["training_record"], dict) or not manifest["training_record"]:
        raise ContractError("模型包缺少有效随机种子或运行记录")
    entries = manifest["instrument_map"]
    if not isinstance(entries, list) or any(not isinstance(entry, list) or len(entry) != 2 for entry in entries):
        raise ContractError("模型包股票映射格式无效")
    if len(dict(entries)) != len(entries):
        raise ContractError("模型包股票映射键重复")
    _validate_mapping(dict(entries))
    if set(manifest["files"]) != {"model.json", "preprocessor.json"}:
        raise ContractError("模型包文件清单不完整")
    for name, checksum in manifest["files"].items():
        if _digest(source / name) != checksum:
            raise ContractError(f"模型包文件校验失败：{name}")
    processor = FrozenPreprocessor.load(source / "preprocessor.json")
    model = create_model(manifest["model_name"])
    serialized = json.loads((source / "model.json").read_text(encoding="utf-8"))
    expected_state = model.state_dict()
    if set(serialized) != set(expected_state):
        raise ContractError("模型包权重名称与模型配置不一致")
    state = {}
    for name, expected in expected_state.items():
        metadata = serialized[name]
        if not isinstance(metadata, dict) or set(metadata) != {"dtype", "shape", "data"}:
            raise ContractError(f"张量元数据不完整：{name}")
        if not isinstance(metadata["shape"], list) or any(type(size) is not int for size in metadata["shape"]):
            raise ContractError(f"张量形状必须使用整数列表：{name}")
        if metadata["dtype"] != str(expected.dtype).removeprefix("torch.") or metadata["shape"] != list(expected.shape):
            raise ContractError(f"张量类型或形状不符合模型配置：{name}")
        values = metadata["data"]
        if not isinstance(values, list) or len(values) != expected.numel():
            raise ContractError(f"张量元素数量不符合模型配置：{name}")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ContractError(f"张量元素必须为数值：{name}")
        if not expected.is_floating_point() and any(type(value) is not int for value in values):
            raise ContractError(f"整数状态不允许截断浮点值：{name}")
        tensor = torch.tensor(values, dtype=expected.dtype).reshape(expected.shape)
        if not torch.isfinite(tensor).all():
            raise ContractError(f"模型包权重包含非有限数值：{name}")
        state[name] = tensor
    _validate_tensor_state(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, processor, manifest
