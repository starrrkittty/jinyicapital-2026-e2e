import hashlib
import json
import os
import shutil
import sys
import tempfile
from importlib.metadata import distribution, distributions, version
from pathlib import Path

import torch

from ._vendor.qlib import COMPONENTS_VERSION
from .encoders import GroupedNetEncoder, PatchEncoder, UnifiedNetEncoder
from .data import _validate_mapping
from .models import AttentionModel, DeepLOBModel, R0Model, R1Model
from .preprocessing import FrozenPreprocessor
from .schema import BOOK12_FIELDS, FEATURES, QLIB_COMMIT, SCHEMA_VERSION, ContractError


BUNDLE_VERSION = 3
MODEL_NAMES = ("R0", "R1", "R1-Book", "Net64", "Net192", "GroupedNet", "Patch1", "Patch5", "DeepLOB")
MODERN_BUNDLE_VERSION = 4
MODERN_MODEL_NAMES = ("B-GRU", "B-Transformer", "B-TCN", "B-SelectiveSSM")
ALL_MODEL_NAMES = MODEL_NAMES + MODERN_MODEL_NAMES
REQUIRED_CONTRACTS = ("source", "sampling", "universe", "visibility", "label", "training_scope")
QLIB_LICENSE_DIGEST = "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383"
EXECUTION_MODULES = ("schema.py", "data.py", "windows.py", "preprocessing.py", "encoders.py", "models.py",
                     "temporal.py", "backbones.py", "official.py", "output.py", "inference.py",
                     "_vendor/__init__.py", "_vendor/qlib/__init__.py", "_vendor/qlib/models.py",
                     "_vendor/qlib/processors.py", "_vendor/qlib/sampler.py", "_vendor/qlib/utils.py")


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
    if name in MODERN_MODEL_NAMES:
        # temporal 的旧候选工厂调用本函数；延迟导入避免循环。
        from .temporal import ModelSpec, create_score_model
        return create_score_model(ModelSpec(name=name))
    raise ContractError(f"未知模型配置 {name!r}")


def dependency_versions() -> dict[str, str]:
    return {name: version(name) for name in ("torch", "numpy", "pandas", "pyarrow")}


def _digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def execution_signature() -> dict:
    """Bind saved scores to the concrete A/B implementation, including equal-shape algorithms."""
    package = Path(__file__).parent
    try:
        files = {f"ptop/{name}": _digest(package / name) for name in EXECUTION_MODULES}
    except OSError as error:
        raise ContractError("推理执行源码缺失，无法核对可复现实现") from error
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"files": files, "fingerprint": fingerprint}


def _validate_tensor_state(state: dict) -> None:
    for name, value in state.items():
        if not torch.isfinite(value).all():
            raise ContractError(f"模型状态包含非有限数值：{name}")
        if name.endswith(("running_var", "num_batches_tracked")) and (value < 0).any():
            raise ContractError(f"BatchNorm 状态必须非负：{name}")


def _save_reference_bundle(path, model, preprocessor: FrozenPreprocessor, *, model_name: str,
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


def _load_reference_bundle(path):
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
    if type(manifest["qlib_components_version"]) is not int or manifest["qlib_components_version"] != COMPONENTS_VERSION:
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


def _json_object(path: Path) -> dict:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"JSON 字段重复：{key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=object_pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               ContractError(f"JSON 包含非有限数值：{value}")))
    except (OSError, ValueError, TypeError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError(f"模型包 JSON 缺失或无效：{path.name}") from error
    if not isinstance(value, dict):
        raise ContractError(f"模型包 JSON 必须为对象：{path.name}")
    return value


def _model_spec_dict(model_spec) -> dict:
    from .temporal import ModelSpec
    if isinstance(model_spec, ModelSpec):
        return model_spec.to_dict()
    if not isinstance(model_spec, dict):
        raise ContractError("model_spec 必须为 ModelSpec 或其完整字典")
    return ModelSpec.from_dict(model_spec).to_dict()


def _create_spec_model(model_spec):
    from .temporal import ModelSpec, create_score_model
    return create_score_model(ModelSpec.from_dict(_model_spec_dict(model_spec)))


def _validated_model_state(model, *, model_spec=None, model_name=None) -> dict:
    """先对照配置和编码器，再复制权重；检查不能只覆盖张量形状。"""
    if not isinstance(model, torch.nn.Module):
        raise ContractError("model 必须为已构造的 torch.nn.Module")
    with torch.random.fork_rng(devices=[]):
        template = _create_spec_model(model_spec) if model_spec is not None else create_model(model_name)
    if hasattr(template, "spec") and (
        not hasattr(model, "spec") or _model_spec_dict(model.spec) != _model_spec_dict(template.spec)
    ):
        raise ContractError("模型声明的规格与保存配置不一致")
    expected = template.state_dict()
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    _validate_state_shapes(state, expected)
    if type(template) is not type(model) or type(template.encoder) is not type(model.encoder):
        raise ContractError("模型类型与保存配置不一致")
    if repr(template) != repr(model):
        raise ContractError("模型模块配置与保存规格不一致")
    if template.encoder.fields != model.encoder.fields or template.encoder.output_dim != model.encoder.output_dim:
        raise ContractError("编码器字段或输出宽度与保存配置不一致")
    if getattr(template.encoder, "patch_size", None) != getattr(model.encoder, "patch_size", None):
        raise ContractError("patch 配置与保存配置不一致")
    return state


def _validate_state_shapes(state, expected) -> None:
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ContractError("模型权重名称与配置不一致")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.shape != expected[name].shape or value.dtype != expected[name].dtype):
            raise ContractError(f"模型权重类型或形状与配置不一致：{name}")
    _validate_tensor_state(state)


def _validate_contracts(contracts) -> None:
    if not isinstance(contracts, dict) or set(contracts) != set(REQUIRED_CONTRACTS) or any(
        not isinstance(value, str) or not value.strip() for value in contracts.values()
    ):
        raise ContractError(f"模型包必须明确提供合同版本 {REQUIRED_CONTRACTS}")


def _json_tensors(state: dict) -> dict:
    return {name: {"dtype": str(value.dtype).removeprefix("torch."),
                   "shape": list(value.shape), "data": value.reshape(-1).tolist()}
            for name, value in state.items()}


def _read_json_tensors(path: Path, expected: dict) -> dict:
    serialized = _json_object(path)
    if set(serialized) != set(expected):
        raise ContractError("模型包权重名称与模型配置不一致")
    state = {}
    for name, template in expected.items():
        entry = serialized[name]
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data"}:
            raise ContractError(f"张量元数据不完整：{name}")
        if (not isinstance(entry["shape"], list) or any(type(size) is not int for size in entry["shape"])
                or entry["shape"] != list(template.shape)
                or entry["dtype"] != str(template.dtype).removeprefix("torch.")):
            raise ContractError(f"张量类型或形状不符合模型配置：{name}")
        values = entry["data"]
        if (not isinstance(values, list) or len(values) != template.numel()
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values)):
            raise ContractError(f"张量元素必须是符合数量的数值：{name}")
        if not template.is_floating_point() and any(type(value) is not int for value in values):
            raise ContractError(f"整数状态不允许截断浮点值：{name}")
        try:
            state[name] = torch.tensor(values, dtype=template.dtype).reshape(template.shape)
        except (ValueError, RuntimeError, OverflowError) as error:
            raise ContractError(f"张量元素无法按配置读取：{name}") from error
    _validate_state_shapes(state, expected)
    return state


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _portable_environment() -> dict:
    packages = {}
    for item in distributions():
        name = item.metadata.get("Name")
        if isinstance(name, str) and name.strip():
            packages[name.lower().replace("_", "-")] = item.version
    return {"python_version": sys.version.split()[0], "packages": dict(sorted(packages.items())),
            "qlib_commit": QLIB_COMMIT}


def _copy_portable_source(stage: Path, manifest: dict) -> list[str]:
    files = []
    package_root = Path(__file__).parent.parent
    for package in ("ptop", "jyc_e2e"):
        directory = package_root / package
        if not directory.is_dir():
            raise ContractError(f"携带复现源码需要可信的 {package} 包")
        target_directory = stage / "source" / package
        target_directory.mkdir(parents=True)
        for original in sorted(directory.rglob("*.py")):
            relative = f"source/{package}/{original.relative_to(directory).as_posix()}"
            (stage / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, stage / relative)
            files.append(relative)
    qlib_license = Path(__file__).parent / "licenses" / "QLIB-MIT.txt"
    if not qlib_license.is_file() or _digest(qlib_license) != QLIB_LICENSE_DIGEST:
        raise ContractError("固定 Qlib 上游许可文本缺失或被修改")
    bundled_license = "source/ptop/licenses/QLIB-MIT.txt"
    (stage / bundled_license).parent.mkdir(parents=True)
    shutil.copyfile(qlib_license, stage / bundled_license)
    files.append(bundled_license)
    for name in ("LICENSE", "SOURCE.md"):
        relative = f"source/ptop/_vendor/qlib/{name}"
        shutil.copyfile(Path(__file__).parent / "_vendor" / "qlib" / name, stage / relative)
        files.append(relative)
    embedded_license = "licenses/qlib/LICENSE"
    (stage / embedded_license).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).parent / "_vendor" / "qlib" / "LICENSE", stage / embedded_license)
    files.append(embedded_license)
    environment = _portable_environment()
    _write_json(stage / "environment.json", environment)
    files.append("environment.json")
    lines = ["# Frozen environment snapshot. Install dependencies before using the trusted source directory."]
    if "+cpu" in manifest["dependencies"]["torch"]:
        lines.append("--extra-index-url https://download.pytorch.org/whl/cpu")
    for name, installed in environment["packages"].items():
        if name not in ("pyqlib", "jinyicapital-e2e", "ptop"):
            lines.append(f"{name}=={installed}")
    (stage / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files.append("requirements.txt")
    third_party = package_root.parent / "THIRD_PARTY.md"
    provenance = (third_party.read_text(encoding="utf-8") if third_party.is_file() else
                  f"# Dependency sources\n\nQlib source: https://github.com/microsoft/qlib/tree/{QLIB_COMMIT}\n")
    provenance += ("\n\n## Bundle dependency licenses\n\n"
                   "The dependency LICENSE/NOTICE files below are copied verbatim from installed distribution metadata. "
                   f"Embedded Qlib components come from https://github.com/microsoft/qlib/tree/{QLIB_COMMIT} and their MIT license is included at `{embedded_license}`. "
                   "source/ contains this repository's ptop and jyc_e2e modules, including the embedded components.\n")
    for name in manifest["dependencies"]:
        installed = distribution(name)
        license_count = 0
        for item in sorted(installed.files or [], key=str):
            if (not item.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
                    or not any(part.endswith(".dist-info") for part in item.parts)):
                continue
            marker = next(index for index, part in enumerate(item.parts) if part.endswith(".dist-info"))
            relative_parts = item.parts[marker + 1:]
            if relative_parts and relative_parts[0] == "licenses":
                relative_parts = relative_parts[1:]
            relative = "/".join(("licenses", name, *relative_parts))
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(installed.locate_file(item), target)
            files.append(relative)
            license_count += 1
            provenance += f"- {name} {manifest['dependencies'][name]}: `{relative}`\n"
        if license_count == 0:
            raise ContractError(f"无法定位 {name} 的上游许可文本；请正确安装固定依赖")
    (stage / "THIRD_PARTY.md").write_text(provenance, encoding="utf-8")
    files.append("THIRD_PARTY.md")
    return files


def _atomic_directory(path, writer) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"模型包路径已经存在：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    stage = temporary / "bundle"
    try:
        writer(stage)
        for file in stage.rglob("*"):
            if file.is_file():
                with file.open("rb") as stream:
                    os.fsync(stream.fileno())
        if destination.exists():
            raise FileExistsError(f"模型包路径已经存在：{destination}")
        os.rename(stage, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def save_bundle(path, model, preprocessor: FrozenPreprocessor, *, model_name: str | None = None,
                model_spec=None, window_length: int, contracts: dict[str, str], seed: int,
                training_record: dict, instrument_map: dict[int, str], weight_format: str = "json",
                include_source: bool | None = None) -> None:
    """保存可审计的推理包。固定候选使用 v3；ModelSpec/torch/source 使用严格 v4。

    source 只携带可信仓库代码，load_bundle 不执行包内 Python 文件。保存先验证合同，
    在同目录暂存并原子改名；已有路径不能覆盖。
    """
    if weight_format not in ("json", "torch"):
        raise ContractError("weight_format 必须为 json 或 torch")
    if include_source is not None and type(include_source) is not bool:
        raise ContractError("include_source 必须为 bool 或 None")
    if (model_spec is None and model_name in MODEL_NAMES and weight_format == "json"
            and include_source in (None, False)):
        return _atomic_directory(path, lambda stage: _save_reference_bundle(
            stage, model, preprocessor, model_name=model_name, window_length=window_length,
            contracts=contracts, seed=seed, training_record=training_record, instrument_map=instrument_map))
    from .temporal import ModelSpec
    if model_spec is None:
        if not isinstance(model_name, str):
            raise ContractError("模型包需要 model_name 或 model_spec")
        model_spec = ModelSpec(name=model_name, lookback=window_length)
    spec = _model_spec_dict(model_spec)
    if model_name is not None and model_name != spec["name"]:
        raise ContractError("model_name 与 model_spec.name 不一致")
    if type(window_length) is not int or window_length <= 0 or spec["lookback"] != window_length:
        raise ContractError("window_length 必须为正整数并与 model_spec.lookback 一致")
    if type(seed) is not int or not isinstance(training_record, dict) or not training_record:
        raise ContractError("模型包必须包含随机种子和运行记录")
    _validate_contracts(contracts)
    mapping = _validate_mapping(instrument_map)
    state = _validated_model_state(model, model_spec=spec)
    processor_state = preprocessor.state_dict()
    manifest = {
        "bundle_version": MODERN_BUNDLE_VERSION, "schema_version": SCHEMA_VERSION,
        "fields": list(FEATURES), "qlib_commit": QLIB_COMMIT, "dependencies": dependency_versions(),
        "qlib_components_version": COMPONENTS_VERSION,
        "model_name": spec["name"], "model_spec": spec, "window_length": window_length,
        "contracts": contracts, "seed": seed, "training_record": training_record,
        "instrument_map": [[int(key), value] for key, value in sorted(mapping.items())],
        "weight_format": weight_format, "source_files": [], "execution": execution_signature(),
    }
    json.dumps(manifest, allow_nan=False)
    portable = True if include_source is None else include_source

    def writer(stage):
        stage.mkdir()
        weight_name = "model.json" if weight_format == "json" else "model.pt"
        if weight_format == "json":
            _write_json(stage / weight_name, _json_tensors(state))
        else:
            torch.save(state, stage / weight_name)
        _write_json(stage / "preprocessor.json", processor_state)
        files = [weight_name, "preprocessor.json"]
        if portable:
            manifest["source_files"] = _copy_portable_source(stage, manifest)
            files.extend(manifest["source_files"])
        manifest["files"] = {name: _digest(stage / name) for name in files}
        if portable and any(manifest["files"][f"source/{name}"] != checksum
                            for name, checksum in manifest["execution"]["files"].items()):
            raise ContractError("保存过程中推理执行源码发生变化")
        _write_json(stage / "manifest.json", manifest)

    _atomic_directory(path, writer)


def load_bundle(path):
    """仅恢复已训练模型和冻结处理器；不会 fit、下载依赖或执行携带源码。"""
    source = Path(path)
    manifest = _json_object(source / "manifest.json")
    if type(manifest.get("bundle_version")) is not int:
        raise ContractError("模型包版本缺失或无效")
    if manifest["bundle_version"] == BUNDLE_VERSION:
        # 原格式和严格依赖/合同检查完整保留；创建模板不改变调用方随机状态。
        with torch.random.fork_rng(devices=[]):
            try:
                return _load_reference_bundle(source)
            except OSError as error:
                raise ContractError("模型包文件缺失或无法读取") from error
    required = {"bundle_version", "schema_version", "fields", "qlib_commit", "qlib_components_version", "dependencies", "model_name",
                "model_spec", "window_length", "contracts", "seed", "training_record", "instrument_map",
                "files", "weight_format", "source_files", "execution"}
    if set(manifest) != required or manifest["bundle_version"] != MODERN_BUNDLE_VERSION:
        raise ContractError("模型包配置字段缺失、未知或版本不支持")
    if type(manifest["qlib_components_version"]) is not int or manifest["qlib_components_version"] != COMPONENTS_VERSION:
        raise ContractError("模型包内置 Qlib 组件版本不一致")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["fields"] != list(FEATURES):
        raise ContractError("模型包 schema 或字段顺序不一致")
    if manifest["qlib_commit"] != QLIB_COMMIT or manifest["dependencies"] != dependency_versions():
        raise ContractError("模型包 Qlib 来源或依赖版本与运行环境不一致")
    if manifest["execution"] != execution_signature():
        raise ContractError("模型包推理执行源码与当前实现不一致；请从携带的可信 source 复现")
    spec = _model_spec_dict(manifest["model_spec"])
    if (manifest["model_spec"] != spec or manifest["model_name"] != spec["name"] or type(manifest["window_length"]) is not int
            or manifest["window_length"] <= 0 or manifest["window_length"] != spec["lookback"]):
        raise ContractError("模型包窗口或模型配置不一致")
    _validate_contracts(manifest["contracts"])
    if type(manifest["seed"]) is not int or not isinstance(manifest["training_record"], dict) or not manifest["training_record"]:
        raise ContractError("模型包缺少有效随机种子或运行记录")
    entries = manifest["instrument_map"]
    if (not isinstance(entries, list) or any(not isinstance(entry, list) or len(entry) != 2 for entry in entries)
            or len(dict(entries)) != len(entries)):
        raise ContractError("模型包股票映射格式无效或键重复")
    _validate_mapping(dict(entries))
    weight_format = manifest["weight_format"]
    if weight_format not in ("json", "torch"):
        raise ContractError("模型包权重格式无效")
    weight_name = "model.json" if weight_format == "json" else "model.pt"
    portable_files = manifest["source_files"]
    if (not isinstance(portable_files, list) or any(not isinstance(name, str) for name in portable_files)
            or len(set(portable_files)) != len(portable_files)):
        raise ContractError("模型包携带源码清单无效")
    embedded_files = {f"source/ptop/{name}" for name in EXECUTION_MODULES if name.startswith("_vendor/")} | {
        "source/ptop/_vendor/qlib/LICENSE", "source/ptop/_vendor/qlib/SOURCE.md",
    }
    for name in portable_files:
        parts = Path(name).parts
        code = ((len(parts) == 3 and parts[0] == "source" and parts[1] in ("ptop", "jyc_e2e") and parts[2].endswith(".py"))
                or name == "source/ptop/licenses/QLIB-MIT.txt" or name in embedded_files)
        license_file = (len(parts) >= 3 and parts[0] == "licenses" and parts[1] in manifest["dependencies"]) or name == "licenses/qlib/LICENSE"
        metadata_file = name in ("requirements.txt", "environment.json", "THIRD_PARTY.md")
        if (name != str(Path(name)) or Path(name).is_absolute() or ".." in parts
                or not (code or license_file or metadata_file)):
            raise ContractError("模型包源码路径无效")
    file_checksums = manifest["files"]
    if not isinstance(file_checksums, dict) or set(file_checksums) != {weight_name, "preprocessor.json", *portable_files}:
        raise ContractError("模型包文件清单不完整")
    if portable_files and any(file_checksums.get(f"source/{name}") != checksum
                              for name, checksum in manifest["execution"]["files"].items()):
        raise ContractError("模型包携带源码与执行指纹不一致")
    for name, checksum in file_checksums.items():
        file = source / name
        if (not isinstance(checksum, str) or len(checksum) != 64 or file.is_symlink()
                or any((source / parent).is_symlink() for parent in Path(name).parents if str(parent) != ".")):
            raise ContractError(f"模型包文件校验信息或路径无效：{name}")
        try:
            matches = _digest(file) == checksum
        except OSError as error:
            raise ContractError(f"模型包文件缺失：{name}") from error
        if not matches:
            raise ContractError(f"模型包文件校验失败：{name}")
    processor = FrozenPreprocessor.from_state_dict(_json_object(source / "preprocessor.json"))
    with torch.random.fork_rng(devices=[]):
        model = _create_spec_model(spec)
    if weight_format == "json":
        state = _read_json_tensors(source / weight_name, model.state_dict())
    else:
        try:
            state = torch.load(source / weight_name, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ContractError("模型包 torch 权重无法安全读取") from error
        _validate_state_shapes(state, model.state_dict())
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, processor, manifest
