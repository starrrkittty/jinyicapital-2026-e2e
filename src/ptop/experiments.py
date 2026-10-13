"""Reproducible, independently budgeted model comparisons on locked keys.

Raw score diagnostics are research measurements. They are not the platform's
BARRA-processed metrics, percentile ranks or five-component competition score.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from .local_training import PreparedLocalFold, _file_digest, _json_digest, _write_json
from .schema import ContractError
from .windows import WindowBatch


class ExperimentBudgetExceeded(RuntimeError):
    """A run exceeded its own declared budget; it is not a completed run."""


@dataclass(frozen=True)
class ExperimentBudget:
    epochs: int = 2
    max_wall_seconds: float = 600.0
    max_train_batches: int | None = None

    def __post_init__(self):
        if type(self.epochs) is not int or self.epochs <= 0:
            raise ContractError("实验 epochs 必须为正整数")
        if isinstance(self.max_wall_seconds, bool) or not np.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0:
            raise ContractError("每次实验 max_wall_seconds 必须为有限正数")
        if self.max_train_batches is not None and (type(self.max_train_batches) is not int or self.max_train_batches <= 0):
            raise ContractError("max_train_batches 必须为正整数或 None")


@dataclass(frozen=True)
class ExperimentPlan:
    model_specs: tuple[dict, ...]
    lookbacks: tuple[int, ...] = (60, 120)
    seeds: tuple[int, ...] = (7, 17, 29)
    training: dict | None = None
    budget: ExperimentBudget = ExperimentBudget()
    device: str = "cpu"
    contracts: dict | None = None

    def __post_init__(self):
        if not self.model_specs or any(not isinstance(spec, dict) or not isinstance(spec.get("name"), str) for spec in self.model_specs):
            raise ContractError("实验计划需要非空且含 name 的 ModelSpec 字典")
        if len({_json_digest(spec) for spec in self.model_specs}) != len(self.model_specs):
            raise ContractError("实验计划包含重复 ModelSpec")
        for name, values in (("lookbacks", self.lookbacks), ("seeds", self.seeds)):
            if not values or len(set(values)) != len(values) or any(type(value) is not int or value < (1 if name == "lookbacks" else 0) for value in values):
                raise ContractError(f"{name} 必须为无重复的非空合法整数序列")
        if any(value >= 2**32 for value in self.seeds):
            raise ContractError("seed 必须小于 2**32，与 TrainingConfig 一致")
        if self.training is not None and not isinstance(self.training, dict):
            raise ContractError("training 配置必须为字典")
        if not isinstance(self.budget, ExperimentBudget):
            raise ContractError("budget 必须为 ExperimentBudget")

    def to_dict(self) -> dict:
        return {"model_specs": list(self.model_specs), "lookbacks": list(self.lookbacks), "seeds": list(self.seeds),
                "training": self.training or {}, "budget": asdict(self.budget), "device": self.device,
                "contracts": self.contracts or {}}

    @classmethod
    def from_dict(cls, value: dict) -> ExperimentPlan:
        expected = {"model_specs", "lookbacks", "seeds", "training", "budget", "device", "contracts"}
        if not isinstance(value, dict) or set(value).difference(expected) or "model_specs" not in value:
            raise ContractError("实验计划字段缺失或未知")
        kwargs = dict(value)
        for name in ("model_specs", "lookbacks", "seeds"):
            if name in kwargs:
                kwargs[name] = tuple(kwargs[name])
        if "budget" in kwargs:
            kwargs["budget"] = ExperimentBudget(**kwargs["budget"])
        return cls(**kwargs)


@dataclass(frozen=True)
class ExperimentRequest:
    run_id: str
    model_spec: dict
    seed: int
    lookback: int
    training: dict
    budget: ExperimentBudget
    device: str
    contracts: dict


def _fingerprint_keys(values) -> str:
    return hashlib.sha256(json.dumps(sorted(str(value) for value in values), ensure_ascii=False).encode()).hexdigest()


def _target_fingerprint(frame: pd.DataFrame) -> str:
    columns = [name for name in ("sample_id", "target", "target_valid", "target_start", "target_end", "label_spec_version", "supervision_exclusion") if name in frame]
    owned = frame[columns].sort_values("sample_id", key=lambda series: series.astype(str)).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(owned, index=False).to_numpy().tobytes()).hexdigest()


def _experiment_implementation_contract() -> dict:
    from .training import _implementation_contract

    return {**_implementation_contract(), "experiments": _file_digest(Path(__file__))}


def requests_for_plan(plan: ExperimentPlan) -> tuple[ExperimentRequest, ...]:
    requests = []
    for candidate in plan.model_specs:
        for lookback in plan.lookbacks:
            for seed in plan.seeds:
                model_spec = {**candidate, "lookback": lookback}
                training = {**(plan.training or {}), "epochs": plan.budget.epochs, "seed": seed}
                identity = {"spec": model_spec, "training": training, "budget": asdict(plan.budget), "device": plan.device}
                safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in candidate["name"])
                run_id = f"{safe_name}-L{lookback}-s{seed}-{_json_digest(identity)[:12]}"
                requests.append(ExperimentRequest(run_id, model_spec, seed, lookback, training,
                                                  plan.budget, plan.device, plan.contracts or {}))
    return tuple(requests)


class _BudgetPrepared:
    def __init__(self, prepared: PreparedLocalFold, budget: ExperimentBudget):
        self.prepared, self.budget = prepared, budget
        self.deadline = time.perf_counter() + budget.max_wall_seconds
        self.training_batches = 0

    def __getattr__(self, name):
        return getattr(self.prepared, name)

    def _bounded(self, iterable, *, training: bool) -> Iterator[WindowBatch]:
        iterator = iter(iterable)
        while True:
            if time.perf_counter() > self.deadline:
                raise ExperimentBudgetExceeded("per-run wall-clock budget exceeded; only completed-epoch checkpoints are resumable")
            try:
                batch = next(iterator)
            except StopIteration:
                return
            if training:
                if self.budget.max_train_batches is not None and self.training_batches >= self.budget.max_train_batches:
                    raise ExperimentBudgetExceeded("per-run training-batch budget exceeded; no shorter model substituted")
                self.training_batches += 1
            yield batch

    def train_loader(self, epoch=0):
        return self._bounded(self.prepared.train_loader(epoch), training=True)

    def validation_loader(self):
        return self._bounded(self.prepared.validation_loader(), training=False)


def _parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    """Count trainable parameters once, under their actual registered module.

    Native A models expose rnn/att_net/fc_out while modern B models expose
    backbone/pool/head. Keep zero-parameter modules (notably R0's raw encoder)
    visible rather than inventing a learned encoder or architecture alias.
    """
    counts = {name: 0 for name, _ in model.named_children()}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = name.split(".", 1)[0] if "." in name else "__root__"
            counts[group] = counts.get(group, 0) + parameter.numel()
    return counts


def _default_runner(request: ExperimentRequest, prepared: PreparedLocalFold, output_dir: Path) -> dict:
    # Imports are deferred so preparing and inspecting data does not load the Trainer.
    from .temporal import ModelSpec, create_score_model
    from .training import Trainer, TrainingConfig, seed_everything

    training_config = TrainingConfig.from_dict(request.training)
    seed_everything(request.seed, deterministic=training_config.deterministic)
    spec = ModelSpec.from_dict(request.model_spec)
    model = create_score_model(spec)
    if tuple(model.encoder.fields) != prepared.fields:
        raise ContractError("比较候选编码器字段必须与 prepared.fields 一致")
    trainer = Trainer(model, training_config, output_dir=output_dir, model_spec=spec,
        contracts=request.contracts, instrument_map=prepared.instrument_map, fold=prepared.fold,
        device=request.device, scope="official" if prepared.scope == "official" else "diagnostic")
    result = trainer.fit(_BudgetPrepared(prepared, request.budget))
    history = [asdict(item) if is_dataclass(item) else dict(item) for item in result.history]
    candidates = [item for item in history if item.get("selection_value") is not None]
    if not candidates:
        raise ContractError("训练没有可用于比较的完整验证轮次")
    matched = [item for item in candidates if float(item["selection_value"]) == float(result.best_metric)]
    if not matched:
        raise ContractError("best_metric 与完整历史记录不一致")
    best = matched[0]
    prediction_path = output_dir / "validation" / f"epoch-{int(best['epoch']):04d}.parquet"
    if not prediction_path.is_file():
        raise ContractError(f"完整 best epoch 缺少按键验证产物：{prediction_path}")
    return {"best_metric": float(result.best_metric), "selection_metric": training_config.selection_metric,
            "metrics": best["validation"], "history": history,
            "validation_predictions": pd.read_parquet(prediction_path), "prediction_path": str(prediction_path),
            "best_checkpoint": str(result.best_checkpoint), "last_checkpoint": str(result.last_checkpoint),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "parameter_counts": _parameter_counts(model),
            "tcn_receptive_field_tokens": getattr(getattr(model, "backbone", None), "receptive_field", None)}


def run_experiments(plan: ExperimentPlan, prepared: PreparedLocalFold, *, output_dir: str | Path,
                    runner: Callable[[ExperimentRequest, PreparedLocalFold, Path], Mapping[str, Any]] | None = None,
                    resume_completed: bool = False) -> pd.DataFrame:
    """Execute every requested cell and reject changed keys/targets/scaler.

    A custom runner returns a dict containing ``validation_predictions`` (a
    DataFrame with full sample IDs, score, predictable and target_valid),
    ``metrics``, ``best_metric`` and ``selection_metric``. Failures are recorded
    and remaining independent runs continue; none is counted as completed.
    """
    if max(plan.lookbacks) > prepared.fit_window_length:
        raise ContractError("全部候选窗口须共享至少最长 lookback 拟合的冻结 N1")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    protocol = {"scope": prepared.scope, "sampling_keys_fingerprint": prepared.manifest["sampling_keys_fingerprint"],
                "preprocessor_fingerprint": prepared.processor.fingerprint,
                "train_targets_fingerprint": _target_fingerprint(prepared.train_targets),
                "validation_targets_fingerprint": _target_fingerprint(prepared.validation_targets),
                "fields": list(prepared.fields), "label_spec": vars(prepared.label_spec), "fold": prepared.manifest["fold"],
                "implementation": _experiment_implementation_contract()}
    identity = {"plan": plan.to_dict(), "protocol": protocol}
    plan_path = destination / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != identity:
        raise ContractError("现有实验目录的固定计划/数据协议不同；请使用独立目录")
    _write_json(plan_path, identity)
    _write_json(destination / "data_manifest.json", prepared.manifest)
    execute = runner or _default_runner
    expected_keys = set(prepared.validation_expected.sample_id.astype(str))
    reference_coverage = None
    records = []
    for request in requests_for_plan(plan):
        path = destination / request.run_id
        path.mkdir(exist_ok=True)
        record_path = path / "run.json"
        if record_path.exists():
            if not resume_completed:
                raise ContractError(f"实验已有产物，禁止覆盖：{request.run_id}；显式 resume_completed 可核验已完成结果")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("status") != "completed" or record.get("protocol_fingerprint") != _json_digest(protocol):
                raise ContractError(f"不能把非完成/不同协议实验当完成恢复：{request.run_id}")
            prediction_path = Path(record["prediction_path"])
            if not prediction_path.is_file() or _file_digest(prediction_path) != record["prediction_sha256"]:
                raise ContractError(f"已完成实验验证产物缺失或改变：{request.run_id}")
            coverage = record["eligible_prediction_keys_fingerprint"]
            if reference_coverage is not None and coverage != reference_coverage:
                raise ContractError("恢复实验的同键覆盖与其他候选不一致")
            reference_coverage = coverage
            records.append(record)
            continue
        request_prepared = prepared.with_window(request.lookback, seed=request.seed)
        started = time.perf_counter()
        before_profile = dict(prepared.store.profile)
        record = {"run_id": request.run_id, "model_name": request.model_spec["name"], "model_spec": request.model_spec,
                  "lookback": request.lookback, "seed": request.seed, "scope": prepared.scope,
                  "budget": asdict(request.budget), "protocol_fingerprint": _json_digest(protocol)}
        try:
            outcome = dict(execute(request, request_prepared, path))
            if time.perf_counter() - started > request.budget.max_wall_seconds:
                raise ExperimentBudgetExceeded("runner exceeded declared independent wall-clock budget")
            predictions = outcome.pop("validation_predictions", None)
            required = {"sample_id", "score", "predictable", "target_valid"}
            if not isinstance(predictions, pd.DataFrame) or not required.issubset(predictions.columns):
                raise ContractError("runner 必须返回包含完整采样键和有效性列的 validation_predictions")
            if predictions.sample_id.isna().any() or not predictions.sample_id.astype(str).is_unique or set(predictions.sample_id.astype(str)) != expected_keys:
                raise ContractError("候选验证完整采样键发生增减或重复，不能进行同键比较")
            for flag in ("predictable", "target_valid"):
                if not pd.api.types.is_bool_dtype(predictions[flag]) or predictions[flag].isna().any():
                    raise ContractError(f"验证 {flag} 必须为完整 bool")
            truth = prepared.validation_targets.set_index("sample_id").loc[predictions.sample_id, "target_valid"].to_numpy()
            if not np.array_equal(predictions.target_valid.to_numpy(), truth):
                raise ContractError("runner 改变了固定验证目标有效性")
            observed_predictability = {}
            for batch in request_prepared.validation_loader():
                observed_predictability.update(zip(batch.metadata.sample_id, batch.predictable.tolist()))
            predictable_truth = np.asarray([observed_predictability[value] for value in predictions.sample_id], dtype=bool)
            if not np.array_equal(predictions.predictable.to_numpy(), predictable_truth):
                raise ContractError("runner 的 predictable 与同字段实际窗口观测不一致")
            if not np.isfinite(predictions.loc[predictions.predictable, "score"].to_numpy(dtype=np.float64)).all():
                raise ContractError("可预测样本必须有有限 score")
            eligible = predictions.predictable & predictions.target_valid
            coverage = _fingerprint_keys(predictions.loc[eligible, "sample_id"])
            if reference_coverage is not None and coverage != reference_coverage:
                raise ContractError("候选有效预测键不同；禁止悄悄取交集后宣称公平比较")
            reference_coverage = coverage
            if prepared.processor.fingerprint != protocol["preprocessor_fingerprint"] or (
                _target_fingerprint(prepared.train_targets) != protocol["train_targets_fingerprint"] or
                _target_fingerprint(prepared.validation_targets) != protocol["validation_targets_fingerprint"]):
                raise ContractError("runner 改变了共同冻结 N1 或目标表")
            if _experiment_implementation_contract() != protocol["implementation"]:
                raise ContractError("执行源码在矩阵运行期间改变；请用当前固定实现开启独立实验目录")
            # Save the checked complete frame even for custom runners.
            prediction_path = path / "comparison_predictions.parquet"
            predictions.to_parquet(prediction_path, index=False)
            outcome.pop("prediction_path", None)
            record.update(outcome)
            record.update(status="completed", prediction_path=str(prediction_path), prediction_sha256=_file_digest(prediction_path),
                          expected_validation_count=len(expected_keys), eligible_validation_count=int(eligible.sum()),
                          eligible_prediction_keys_fingerprint=coverage)
        except ExperimentBudgetExceeded as exc:
            record.update(status="budget_exceeded", error=str(exc))
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        record.update(seconds=time.perf_counter() - started,
            source_read_seconds=prepared.store.profile["read_seconds"] - before_profile["read_seconds"],
            raw_rows_read=prepared.store.profile["rows_read"] - before_profile["rows_read"],
            peak_rss_bytes_process=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))
        _write_json(record_path, record)
        records.append(record)
        _write_report(destination, records, protocol)
    _write_report(destination, records, protocol)
    return _summary_frame(records)


def _summary_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        fields = ("run_id", "model_name", "lookback", "seed", "scope", "status", "selection_metric", "best_metric", "seconds",
                  "source_read_seconds", "raw_rows_read", "peak_rss_bytes_process", "parameter_count",
                  "expected_validation_count", "eligible_validation_count", "eligible_prediction_keys_fingerprint", "error")
        row = {name: record.get(name) for name in fields}
        for name, value in record.get("metrics", {}).items():
            if value is None or isinstance(value, (int, float, bool, np.number)):
                row[f"validation_{name}"] = value
        for name, value in record.get("parameter_counts", {}).items():
            row[f"parameters_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _write_report(destination: Path, records: list[dict], protocol: dict) -> None:
    frame = _summary_frame(records)
    frame.to_csv(destination / "comparison.csv", index=False)
    completed = frame.loc[frame.status.eq("completed")]
    notice = ("真实行情与确定性人工目标的工程诊断；以下数值不能说明金融预测效果，不能作为比赛基线成绩。"
              if protocol["scope"] != "official" else
              "显式官方标签上的本地原始 score 研究指标；未复现平台 BARRA 处理、五项百分位或比赛总分。")
    lines = [notice, "", f"完成 {len(completed)} / {len(frame)} 次计划实验。每次独立执行既定预算；失败或超限单独记录。", "",
             f"固定采样键：`{protocol['sampling_keys_fingerprint']}`", "",
             f"共同冻结 N1：`{protocol['preprocessor_fingerprint']}`", "",
             "| 主干 | L | seed | 状态 | 选择指标数值 | 秒 |", "|---|---:|---:|---|---:|---:|"]
    for row in frame.itertuples():
        metric = "—" if pd.isna(row.best_metric) else f"{row.best_metric:.6g}"
        lines.append(f"| {row.model_name} | {row.lookback} | {row.seed} | {row.status} | {metric} | {row.seconds:.3f} |")
    if not completed.empty:
        grouped = completed.groupby(["model_name", "lookback", "selection_metric"], dropna=False).best_metric.agg(["count", "mean", "std"]).reset_index()
        grouped.to_csv(destination / "seed_summary.csv", index=False)
        lines.extend(["", "跨 seed 的均值和样本标准差见 seed_summary.csv；同键覆盖验证通过才纳入汇总。",
                      "", "peak_rss_bytes_process 是整个进程的累计 RSS 峰值；逐实验 GPU 内存与批次等待见 Trainer 日志。"])
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark_windows(loader, *, model: torch.nn.Module | None = None, device: str = "cpu",
                      max_batches: int | None = None) -> dict:
    """Measure actual loader waits, tensor transfer, forward time and memory.

    No target is consulted. CPU RSS is the process-lifetime peak; CUDA peak is
    reset for this measurement and synchronization includes asynchronous work.
    """
    if max_batches is not None and (type(max_batches) is not int or max_batches <= 0):
        raise ContractError("max_batches 必须为正整数或 None")
    chosen_device = torch.device(device)
    if chosen_device.type == "cuda" and not torch.cuda.is_available():
        raise ContractError("请求 CUDA benchmark，但当前环境没有可用 CUDA")
    if chosen_device.type == "cuda":
        torch.cuda.synchronize(chosen_device)
        torch.cuda.reset_peak_memory_stats(chosen_device)
    if model is not None:
        model.to(chosen_device).eval()
    started = time.perf_counter()
    waiting = transferring = forward = 0.0
    batches = samples = predictable = window_bytes = 0
    iterator = iter(loader)
    with torch.no_grad():
        while max_batches is None or batches < max_batches:
            before = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                waiting += time.perf_counter() - before
                break
            waiting += time.perf_counter() - before
            selected = batch.predictable
            batches += 1
            samples += len(selected)
            predictable += int(selected.sum())
            window_bytes += sum(value.nbytes for value in (batch.x, batch.observed, batch.step_valid))
            if model is None or not selected.any():
                continue
            before = time.perf_counter()
            inputs = [torch.as_tensor(value[selected], device=chosen_device) for value in (batch.x, batch.observed, batch.step_valid)]
            if chosen_device.type == "cuda":
                torch.cuda.synchronize(chosen_device)
            transferring += time.perf_counter() - before
            before = time.perf_counter()
            scores = model(*inputs)
            if chosen_device.type == "cuda":
                torch.cuda.synchronize(chosen_device)
            forward += time.perf_counter() - before
            if scores.shape != (int(selected.sum()),) or not torch.isfinite(scores).all():
                raise ContractError("真实窗口 benchmark 模型必须返回有限的一维 score")
    elapsed = time.perf_counter() - started
    return {"scope": "engineering_window_throughput_not_predictive_quality", "device": str(chosen_device),
            "batches": batches, "samples": samples, "predictable_samples": predictable, "window_tensor_bytes": window_bytes,
            "wall_seconds": elapsed, "data_wait_seconds": waiting, "transfer_seconds": transferring,
            "forward_seconds": forward, "samples_per_second": samples / elapsed if elapsed else None,
            "peak_rss_bytes_process": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(chosen_device)) if chosen_device.type == "cuda" else None}
