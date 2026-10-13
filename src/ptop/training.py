"""Complete B training loop over A's WindowBatch and frozen preprocessing.

No market-label formula or platform evaluator is implemented here. C can inject
both an objective and an evaluator. Validation is assembled across microbatches
before per-decision metrics are calculated.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import resource
import tempfile
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch

from jyc_e2e.research_metrics import ic_summary, rank_ic_by_time
from .objectives import RegressionObjective
from .preprocessing import FrozenPreprocessor
from .schema import FEATURES, ContractError
from .targets import TimeFold
from .training_config import TrainingConfig
from .windows import WindowBatch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def resolve_device(value: str | torch.device) -> torch.device:
    if str(value) == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ContractError("B training currently supports cpu or cuda")
    if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
        raise ContractError(f"requested CUDA device is unavailable: {device}")
    return device


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ContractError("log diagnostics must contain scalars")
        return _jsonable(value.detach().cpu().item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def research_evaluator(predictions: pd.DataFrame) -> dict:
    """Use the existing research RankIC with full decision cross sections."""
    selected = predictions.loc[predictions.predictable & predictions.target_valid]
    ic = {}
    for decision, group in selected.groupby("decision_id", sort=False):
        # Grouping once avoids the old helper's O(number_of_times * number_of_rows) selection.
        values = rank_ic_by_time(group.score.to_numpy(), group.target.to_numpy(), np.zeros(len(group), dtype=int))
        ic[str(decision)] = values.get(0, float("nan"))
    result = ic_summary(ic)
    result.update(valid_times=sum(math.isfinite(v) for v in ic.values()),
                  invalid_times=sum(not math.isfinite(v) for v in ic.values()),
                  research_only=True, official_score=None)
    return result


def _target_table(table: pd.DataFrame, phase: str, fold: TimeFold) -> pd.DataFrame:
    required = {"sample_id", "decision_id", "instrument_id", "decision_time", "target", "target_valid", "target_start", "target_end", "label_spec_version"}
    if not isinstance(table, pd.DataFrame) or not table.columns.is_unique or not required.issubset(table):
        raise ContractError(f"{phase} targets must contain the explicit A/C label contract {sorted(required)}")
    result = table.copy()
    if result.empty or result.sample_id.isna().any() or result.sample_id.duplicated().any():
        raise ContractError(f"{phase} target sample IDs must be nonempty and unique")
    if result[["decision_id", "instrument_id", "label_spec_version"]].isna().any().any():
        raise ContractError("label identifiers and version must not be missing")
    if not pd.api.types.is_bool_dtype(result.target_valid) or result.target_valid.isna().any():
        raise ContractError("target_valid must be non-null boolean")
    for field in ("decision_time", "target_start", "target_end"):
        result[field] = pd.to_datetime(result[field], errors="raise")
        if result[field].dt.tz is not None:
            raise ContractError("target times must retain the local naive time convention")
    if result.decision_time.isna().any() or not result.decision_time.eq(result.decision_time.dt.floor("min")).all():
        raise ContractError("decision times must be complete minute timestamps")
    start, end = (fold.train_start, fold.validation_start) if phase == "train" else (fold.validation_start, fold.validation_end)
    if not (result.decision_time.ge(start) & result.decision_time.lt(end)).all():
        raise ContractError(f"{phase} target decisions lie outside the declared fold")
    if result.duplicated(["decision_time", "instrument_id"]).any():
        raise ContractError("target decision/stock keys are duplicated")
    if (result.groupby("decision_id").decision_time.nunique() != 1).any() or (result.groupby("decision_time").decision_id.nunique() != 1).any():
        raise ContractError("decision IDs must identify one complete timestamp")
    valid = result.target_valid
    values = pd.to_numeric(result.target, errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(values[valid]).all():
        raise ContractError("valid labels must be finite")
    supported = result.loc[valid]
    if supported[["target_start", "target_end"]].isna().any().any():
        raise ContractError("valid labels need explicit support timestamps")
    if (supported.target_start < supported.decision_time).any() or (supported.target_end <= supported.target_start).any():
        raise ContractError("label support must start at/after the decision and increase")
    if not supported.target_end.dt.normalize().eq(supported.decision_time.dt.normalize()).all():
        raise ContractError("intraday label support cannot cross overnight")
    crosses = valid & result.target_end.ge(end)
    prior = result.get("supervision_exclusion", pd.Series("", index=result.index)).fillna("")
    result["supervision_exclusion"] = np.select(
        [~valid, crosses], [prior.where(prior.ne(""), "invalid_target"), "target_reaches_split_boundary"], default="",
    )
    result["target_valid"] = valid & ~crosses
    result["target"] = np.where(result.target_valid, values, np.nan)
    return result


def _table_fingerprint(table: pd.DataFrame) -> str:
    columns = ["sample_id", "decision_id", "instrument_id", "decision_time", "target", "target_valid", "target_start", "target_end", "label_spec_version"]
    ordered = table[columns].sort_values("sample_id", kind="stable").reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()).hexdigest()


def _callable_contract(value) -> dict:
    if hasattr(value, "configuration"):
        result = value.configuration()
    elif inspect.isfunction(value):
        try:
            source = inspect.getsource(value)
        except (OSError, TypeError) as error:
            raise ContractError("C callable needs inspectable source or configuration()") from error
        result = {"callable": f"{value.__module__}:{value.__qualname__}",
                  "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                  "defaults": value.__defaults__, "keyword_defaults": value.__kwdefaults__,
                  "closure": [cell.cell_contents for cell in (value.__closure__ or ())]}
    else:
        raise ContractError("stateful C objective/evaluator must declare configuration()")
    try:
        return json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ContractError("C callable configuration must be finite JSON; declare its immutable settings") from error


def _implementation_contract() -> dict:
    names = ("schema", "data", "windows", "preprocessing", "encoders", "models", "backbones", "temporal",
             "training", "training_config", "objectives", "checkpoint", "local_training")
    directory = Path(__file__).parent
    result = {name: hashlib.sha256((directory / (name + ".py")).read_bytes()).hexdigest() for name in names}
    for name in ("losses", "research_metrics"):
        result["jyc_e2e/" + name] = hashlib.sha256((directory.parent / "jyc_e2e" / (name + ".py")).read_bytes()).hexdigest()
    return result


def _input_contract(prepared) -> dict:
    manifest = getattr(prepared, "manifest", None)
    if isinstance(manifest, dict) and manifest:
        return {"manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True,
                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()}
    result = {}
    for phase in ("train", "validation"):
        data = getattr(getattr(getattr(prepared, phase + "_loader"), "dataset", None), "data", None)
        if data is None:
            raise ContractError("custom prepared factories must declare a source/visibility manifest for resume")
        digest = hashlib.sha256()
        for frame in (data.values, data.observed, data.metadata):
            digest.update(json.dumps([str(name) for name in frame.columns]).encode())
            for start in range(0, len(frame), 50_000):
                digest.update(pd.util.hash_pandas_object(frame.iloc[start:start + 50_000], index=True).to_numpy().tobytes())
        result[phase + "_source_sha256"] = digest.hexdigest()
    return result


def validate_batch(batch: WindowBatch, fields: tuple[str, ...], *, length: int, scope: str) -> np.ndarray:
    if not isinstance(batch, WindowBatch):
        raise ContractError("A data loader must yield WindowBatch")
    if batch.x.ndim != 3 or batch.x.shape[1:] != (length, len(FEATURES)) or batch.x.dtype != np.float32:
        raise ContractError("WindowBatch x must match the configured float32 [B,L,25]")
    count = len(batch.x)
    if count == 0:
        raise ContractError("WindowBatch must contain at least one declared sample")
    if batch.observed.shape != batch.x.shape or batch.observed.dtype != bool:
        raise ContractError("observed must be bool [B,L,25]")
    if batch.step_valid.shape != (count, length) or batch.step_valid.dtype != bool:
        raise ContractError("step_valid must be bool [B,L]")
    if not np.isfinite(batch.x).all() or np.any(batch.x[~batch.observed] != 0):
        raise ContractError("x must be finite; masked fields must be zero")
    if np.any(batch.observed & ~batch.step_valid[..., None]) or np.any(batch.step_valid[:, :-1] & ~batch.step_valid[:, 1:]):
        raise ContractError("structural padding is left-only; real missing minutes remain valid slots")
    if batch.bar_times.shape != (count, length) or batch.bar_times.dtype != np.dtype("datetime64[ns]"):
        raise ContractError("bar_times must be datetime64[ns] [B,L]")
    if batch.row_positions.shape != (count, length) or batch.row_positions.dtype != np.int64:
        raise ContractError("row_positions must be int64 [B,L]")
    if not isinstance(batch.metadata, pd.DataFrame) or len(batch.metadata) != count:
        raise ContractError("WindowBatch metadata must preserve every sample")
    required = {"sample_id", "decision_id", "decision_time", "instrument_id"}
    if not required.issubset(batch.metadata) or batch.metadata[list(required)].isna().any().any():
        raise ContractError("window metadata is missing sample or decision keys")
    if batch.metadata.sample_id.duplicated().any():
        raise ContractError("batch sample IDs must be unique")
    if scope == "official" and "production_visibility_verified" in batch.metadata and not batch.metadata.production_visibility_verified.all():
        raise ContractError("local audit windows cannot be presented as official supervised windows")
    decisions = pd.DatetimeIndex(batch.metadata.decision_time).to_numpy(dtype="datetime64[ns]")
    if (np.any(batch.row_positions < -1) or np.any(batch.row_positions[~batch.step_valid] != -1)
            or not np.isnat(batch.bar_times[~batch.step_valid]).all()):
        raise ContractError("structural padding needs NaT times and row_positions=-1")
    if np.isnat(batch.bar_times[batch.step_valid]).any() or np.any(batch.bar_times[batch.step_valid] > np.broadcast_to(decisions[:, None], batch.step_valid.shape)[batch.step_valid]):
        raise ContractError("window contains an invalid or future bar time")
    adjacent = batch.step_valid[:, 1:] & batch.step_valid[:, :-1]
    if np.any((batch.bar_times[:, 1:] <= batch.bar_times[:, :-1])[adjacent]):
        raise ContractError("window bar times must strictly increase")
    valid_times = batch.bar_times[batch.step_valid]
    decision_grid = np.broadcast_to(decisions[:, None], batch.step_valid.shape)[batch.step_valid]
    if (not batch.step_valid[:, -1].all() or not np.array_equal(batch.bar_times[:, -1], decisions)
            or not np.array_equal(valid_times.astype("datetime64[D]"), decision_grid.astype("datetime64[D]"))
            or not np.array_equal(valid_times.astype("datetime64[m]").astype("datetime64[ns]"), valid_times)):
        raise ContractError("A windows must contain same-day minute slots ending at their decision")
    if np.any(batch.observed.any(-1) & (batch.row_positions < 0)):
        raise ContractError("observed values require a traceable source row")
    positions = [FEATURES.index(name) for name in fields]
    actual = batch.observed[..., positions].any(axis=(1, 2))
    if batch.predictable.shape != (count,) or batch.predictable.dtype != bool or not np.array_equal(actual, batch.predictable):
        raise ContractError("A loader predictable flag must use this model.encoder.fields")
    return actual


@dataclass
class TrainingResult:
    history: list[dict]
    best_checkpoint: Path
    last_checkpoint: Path
    bundle_path: Path | None
    best_metric: float
    best_epoch: int
    completed_epochs: int
    stopped_reason: str


class Trainer:
    """Optimize encoder, temporal backbone and head as one complete model."""

    def __init__(self, model, config: TrainingConfig, *, output_dir, model_spec, contracts: dict,
                 fold: TimeFold, instrument_map=None, device="cpu", objective=None,
                 evaluator: Callable[[pd.DataFrame], dict] | None = None, scope="official"):
        if not isinstance(config, TrainingConfig) or not isinstance(fold, TimeFold):
            raise ContractError("Trainer requires TrainingConfig and TimeFold")
        if scope not in {"official", "diagnostic"}:
            raise ContractError("training scope must be explicit official or diagnostic")
        if not hasattr(model, "encoder") or not hasattr(model.encoder, "fields"):
            raise ContractError("register A's input encoder as model.encoder")
        if not any(p.requires_grad for p in model.parameters()):
            raise ContractError("the complete model needs trainable parameters")
        from .bundle import REQUIRED_CONTRACTS
        if set(contracts) != set(REQUIRED_CONTRACTS) or any(not isinstance(v, str) or not v.strip() for v in contracts.values()):
            raise ContractError("all source/sampling/universe/visibility/label/training_scope contracts are required")
        if scope == "official":
            from .data import _validate_mapping
            _validate_mapping(instrument_map)
        self.model, self.config, self.model_spec = model, config, model_spec
        self.contracts, self.fold = dict(contracts), fold
        self.instrument_map, self.scope = instrument_map, scope
        self.device = resolve_device(device)
        if config.precision == "fp16" and self.device.type != "cuda":
            raise ContractError("fp16 training requires CUDA; use fp32 for CPU")
        if config.precision == "bf16" and self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ContractError("this CUDA device does not support bf16")
        self.model.to(self.device)
        self.output_dir = Path(output_dir)
        self.objective, self.evaluator = objective, evaluator or research_evaluator
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        optimized = [id(p) for group in self.optimizer.param_groups for p in group["params"]]
        trainable = [id(p) for p in self.model.parameters() if p.requires_grad]
        if not set(trainable).issubset(optimized) or len(optimized) != len(set(optimized)):
            raise ContractError("optimizer must include every model parameter exactly once")
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.min_learning_rate,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16", init_scale=config.amp_initial_scale)
        self.global_step = 0
        self._processor_fingerprint = None

    def _autocast(self):
        if self.config.precision == "fp32":
            return nullcontext()
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16
        return torch.autocast(self.device.type, dtype=dtype)

    def _log(self, value: dict):
        with (self.output_dir / "training.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False) + "\n")

    def _loader(self, prepared, phase: str, epoch: int):
        provider = getattr(prepared, phase + "_loader")
        loader = provider(epoch) if phase == "train" and callable(provider) else (provider() if callable(provider) else provider)
        if phase == "train":
            for name in ("sampler", "batch_sampler"):
                sampler = getattr(loader, name, None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                generator = getattr(sampler, "generator", None)
                if isinstance(generator, torch.Generator):
                    generator.manual_seed(self.config.seed + epoch)
        return loader

    def _align(self, batch: WindowBatch, table: pd.DataFrame) -> pd.DataFrame:
        sample_ids = batch.metadata.sample_id
        if not sample_ids.isin(table.index).all():
            raise ContractError("batch has sample IDs without explicit label validity records")
        aligned = table.loc[sample_ids].reset_index(drop=True)
        for field in ("decision_id", "instrument_id", "decision_time"):
            if not batch.metadata[field].reset_index(drop=True).eq(aligned[field]).all():
                raise ContractError(f"label alignment conflicts with sample metadata: {field}")
        return aligned

    def _loss(self, scores, target, decisions, flags=None):
        codes, _ = pd.factorize(decisions, sort=False)
        ids = torch.as_tensor(codes, dtype=torch.int64, device=self.device)
        if flags is None:
            flags = torch.ones_like(target, dtype=torch.bool)
        result = self.objective(scores, target, ids, flags)
        loss, diagnostics = result if isinstance(result, tuple) else (result, {})
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
            raise ContractError("C's objective must return one finite differentiable scalar")
        if not isinstance(diagnostics, dict):
            raise ContractError("objective diagnostics must be a dictionary")
        return loss, diagnostics

    def _selected_checkpoint(self, processor, split, best_epoch, best_metric):
        from .checkpoint import load_inference_checkpoint
        selected, selected_processor, metadata = load_inference_checkpoint(
            self.best_checkpoint, device="cpu", expected_contracts=self.contracts,
        )
        if (metadata["config"] != self.config.to_dict() or metadata["model_spec"] != self.model_spec.to_dict()
                or metadata["split"] != split or metadata["epoch"] != best_epoch
                or metadata["best_metric"] != best_metric or selected_processor.fingerprint != processor.fingerprint):
            raise ContractError("best.pt does not belong to this run's selected epoch/config/data contract")
        return selected, selected_processor

    def _phase(self, loader, table: pd.DataFrame, *, training: bool, epoch: int):
        self.model.train(training)
        count = steps = batches = skipped = 0
        amp_skipped = 0
        initial_step = self.global_step
        loss_sum = data_wait = 0.0
        frames, skip_counts = [], {}
        seen = set()
        started = time.perf_counter()
        last = started
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        for batch in loader:
            batch_start = time.perf_counter()
            data_wait += batch_start - last
            batches += 1
            predictable = validate_batch(batch, tuple(self.model.encoder.fields), length=self.model_spec.lookback, scope=self.scope)
            if any(key in seen for key in batch.metadata.sample_id):
                raise ContractError("a phase repeats sample IDs; one epoch must not silently oversample")
            seen.update(batch.metadata.sample_id)
            aligned = self._align(batch, table)
            selected = predictable & aligned.target_valid.to_numpy(dtype=bool)
            reasons = np.where(~predictable, "no_observation_in_model_fields", aligned.supervision_exclusion.to_numpy())
            skipped += int((~selected).sum())
            for reason, n in pd.Series(reasons[~selected]).value_counts().items():
                skip_counts[str(reason)] = skip_counts.get(str(reason), 0) + int(n)
            frame = batch.metadata.copy().reset_index(drop=True)
            frame["score"] = np.nan
            frame["predictable"] = predictable
            frame["target"] = aligned.target.to_numpy()
            frame["target_valid"] = aligned.target_valid.to_numpy()
            frame["exclusion_reason"] = np.where(selected, "", reasons)
            indices = np.flatnonzero(selected if training else predictable)
            if training and len(indices):
                indices = indices[torch.randperm(len(indices)).numpy()]
            for offset in range(0, len(indices), self.config.microbatch_size):
                chosen = indices[offset:offset + self.config.microbatch_size]
                inputs = [torch.as_tensor(value[chosen], device=self.device)
                          for value in (batch.x, batch.observed, batch.step_valid)]
                flags = torch.as_tensor(aligned.target_valid.to_numpy(dtype=bool)[chosen], device=self.device)
                targets = aligned.target.to_numpy(dtype=np.float32)[chosen]
                target = torch.as_tensor(np.where(flags.cpu().numpy(), targets, 0.0), device=self.device)
                if not torch.isfinite(target[flags]).all():
                    raise ContractError("target conversion to float32 overflowed")
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                # AMP is a training optimization. Select checkpoints with the
                # same FP32 forward used by the frozen inference entry point.
                with torch.set_grad_enabled(training), (self._autocast() if training else nullcontext()):
                    scores = self.model(*inputs)
                    if scores.shape != target.shape or not torch.isfinite(scores).all():
                        raise ContractError("model must return finite [B] scores")
                    if flags.any():
                        loss, diagnostics = self._loss(scores, target, aligned.decision_id.iloc[chosen].to_numpy(), flags)
                    else:
                        loss, diagnostics = scores.sum() * 0.0, {"count": 0}
                gradient_norm = None
                if training:
                    if not loss.requires_grad:
                        raise ContractError("objective detached the joint-training graph")
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    trainable = [(name, parameter) for name, parameter in self.model.named_parameters() if parameter.requires_grad]
                    missing = [name for name, parameter in trainable if parameter.grad is None]
                    if missing:
                        raise ContractError(f"joint-training gradients missing: {missing}")
                    gradients_finite = all(torch.isfinite(parameter.grad).all() for _, parameter in trainable)
                    if not gradients_finite and not self.scaler.is_enabled():
                        raise ContractError("joint-training gradients nonfinite without FP16 dynamic scaling")
                    if gradients_finite:
                        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.clip_norm, error_if_nonfinite=True,
                        ).detach().cpu())
                    else:
                        amp_skipped += 1
                    # unscale_ recorded found_inf: GradScaler skips this optimizer update
                    # and lowers its scale. Never clip or apply nonfinite gradients.
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if gradients_finite:
                        self.global_step += 1
                    if self.scaler.is_enabled() and (not math.isfinite(self.scaler.get_scale()) or self.scaler.get_scale() <= 0):
                        raise ContractError("FP16 scale is zero/nonfinite; inspect input/loss magnitude or use bf16/fp32")
                numeric_loss = float(loss.detach().float().cpu())
                frame.loc[chosen, "score"] = scores.detach().float().cpu().numpy()
                supervised_count = int(flags.sum().item())
                loss_sum += numeric_loss * supervised_count
                count += supervised_count
                steps += 1
                if training and (not gradients_finite or self.global_step == 1 or self.global_step % self.config.log_every == 0):
                    self._log({"event": "step", "epoch": epoch + 1, "global_step": self.global_step,
                               "loss": numeric_loss, "count": len(chosen), "gradient_norm": gradient_norm,
                               "learning_rate": self.optimizer.param_groups[0]["lr"], "diagnostics": diagnostics,
                               "optimizer_step_skipped": not gradients_finite, "amp_scale": self.scaler.get_scale(),
                               "data_wait_seconds": data_wait, "elapsed_seconds": time.perf_counter() - started})
            if not training:
                frames.append(frame)
            last = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if not count:
            raise ContractError("phase has no valid supervised samples; no checkpoint will be selected")
        if training and self.global_step == initial_step:
            raise ContractError("all optimizer updates skipped; lower amp_initial_scale or use bf16/fp32")
        if seen != set(table.index):
            missing = len(set(table.index) - seen)
            raise ContractError(f"phase loader did not process its complete declared sample set ({missing} missing)")
        seconds = time.perf_counter() - started
        stats = {"loss": loss_sum / count, "count": count, "skipped": skipped,
                 "skip_reasons": skip_counts, "batches": batches, "microbatches": steps,
                 "optimizer_steps": self.global_step - initial_step, "amp_skipped_steps": amp_skipped,
                 "seconds": seconds, "data_wait_seconds": data_wait, "samples_per_second": count / seconds,
                 "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
                 "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0}
        predictions = pd.concat(frames, ignore_index=True) if frames else None
        if predictions is not None:
            report = self.evaluator(predictions)
            if not isinstance(report, dict):
                raise ContractError("C's evaluator must return a metrics dictionary")
            stats.update(report)
        return stats, predictions

    def fit(self, prepared, *, resume=None, stop_after_epochs: int | None = None) -> TrainingResult:
        """Train/restart at an epoch boundary using A's prepared-fold interface.

        The loader provider may be a DataLoader or a reentrant daily partition
        factory. stop_after_epochs limits this invocation, not scheduler horizon.
        """
        from .checkpoint import load_checkpoint, save_checkpoint
        from .bundle import save_bundle

        if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
            raise ContractError("stop_after_epochs must be a positive invocation limit")
        declared_scope = getattr(prepared, "scope", None)
        if declared_scope is not None and (
            not isinstance(declared_scope, str) or
            (self.scope == "official" and declared_scope != "official") or
            (self.scope == "diagnostic" and not declared_scope.startswith("diagnostic"))
        ):
            raise ContractError("prepared data scope differs from the explicitly requested training scope")
        processor = getattr(prepared, "processor", None)
        if not isinstance(processor, FrozenPreprocessor):
            raise ContractError("A must provide a fitted and frozen FrozenPreprocessor")
        self._processor_fingerprint = processor.fingerprint
        if pd.Timestamp(processor.fit_start) < self.fold.train_start or pd.Timestamp(processor.fit_end) >= self.fold.validation_start:
            raise ContractError("normalizer fitting period must stay inside this training fold")
        train = _target_table(prepared.train_targets, "train", self.fold)
        validation = _target_table(prepared.validation_targets, "validation", self.fold)
        if set(train.sample_id).intersection(validation.sample_id):
            raise ContractError("training and validation sample sets must be disjoint")
        if train.label_spec_version.nunique() != 1 or validation.label_spec_version.nunique() != 1 or train.label_spec_version.iloc[0] != validation.label_spec_version.iloc[0]:
            raise ContractError("train/validation must share one explicit label version")
        if not train.label_spec_version.eq(self.contracts["label"]).all():
            raise ContractError("label table version differs from the persisted label contract")
        scale_values = train.loc[train.target_valid, "target"].to_numpy(dtype=np.float64)
        if self.config.target_scale is None and scale_values.size < 2:
            raise ContractError("need at least two finite training labels to calibrate/verify target scale")
        self.target_scale = self.config.target_scale if self.config.target_scale is not None else float(scale_values.std(ddof=0))
        if not math.isfinite(self.target_scale) or self.target_scale <= 0:
            raise ContractError("training-label scale is zero or nonfinite")
        if self.objective is None:
            self.objective = RegressionObjective(target_scale=self.target_scale, huber_delta=self.config.huber_delta,
                                                 rank_weight=self.config.rank_weight)
        if isinstance(self.objective, torch.nn.Module):
            if any(parameter.requires_grad for parameter in self.objective.parameters()):
                raise ContractError("trainable C loss modules must be registered inside the complete model")
            self.objective.to(self.device)
        objective_config = _callable_contract(self.objective)
        train_fingerprint, validation_fingerprint = _table_fingerprint(train), _table_fingerprint(validation)
        split = {"name": self.fold.name, "train_start": self.fold.train_start.isoformat(),
                 "validation_start": self.fold.validation_start.isoformat(), "validation_end": self.fold.validation_end.isoformat(),
                 "train_targets_fingerprint": train_fingerprint, "validation_targets_fingerprint": validation_fingerprint,
                 "scope": self.scope, "objective": objective_config,
                 "evaluator": _callable_contract(self.evaluator), "target_scale": self.target_scale,
                 "implementation": _implementation_contract(), "inputs": _input_contract(prepared),
                 "instrument_map": ([[int(key), value] for key, value in sorted(self.instrument_map.items())]
                                    if self.instrument_map is not None else None)}
        indexed_train, indexed_validation = train.set_index("sample_id", drop=False), validation.set_index("sample_id", drop=False)
        start_epoch = completed = bad_epochs = 0
        best_metric, best_epoch = None, 0
        history = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint, self.last_checkpoint = self.output_dir / "best.pt", self.output_dir / "last.pt"
        if resume is not None:
            state = load_checkpoint(
                resume, self.model, self.optimizer, scheduler=self.scheduler, scaler=self.scaler,
                expected_config=self.config.to_dict(), expected_model_spec=self.model_spec,
                expected_contracts=self.contracts, expected_split=split, expected_preprocessor=processor,
            )
            start_epoch = completed = state.epoch
            self.global_step, best_metric, bad_epochs = state.global_step, state.best_metric, state.bad_epochs
            history = state.extra_state.get("history", [])
            best_epoch = state.extra_state.get("best_epoch", 0)
            if not isinstance(history, list) or len(history) != completed or best_epoch > completed:
                raise ContractError("checkpoint history/epoch accounting is incomplete")
            if completed > self.config.epochs:
                raise ContractError("checkpoint exceeds configured scheduler horizon")
            if best_epoch and not self.best_checkpoint.exists():
                raise ContractError("resume requires the matching run's best.pt; do not copy last.pt alone")
            if best_epoch:
                self._selected_checkpoint(processor, split, best_epoch, best_metric)
        elif self.best_checkpoint.exists() or self.last_checkpoint.exists() or (self.output_dir / "history.json").exists():
            raise ContractError("output run already exists; use a new directory or explicit resume")
        write_json(self.output_dir / "run_config.json", {
            "training": self.config.to_dict(), "model": self.model_spec.to_dict(), "contracts": self.contracts,
            "split": split, "preprocessor_fingerprint": processor.fingerprint,
            "objective": objective_config, "device": str(self.device), "scope": self.scope,
            "score_units": ("training-target-scale normalized score" if isinstance(self.objective, RegressionObjective)
                            else objective_config.get("score_units", "custom objective; units declared by C")),
            "cuda_epoch_seed": "(seed + epoch_index) modulo 2**32; resets cuDNN RNN dropout state",
            "validation_precision": "fp32", "inference_precision": "fp32",
            "platform_evaluation_verified": False,
        })
        stop_reason = "epochs_completed"
        limit = min(self.config.epochs, start_epoch + stop_after_epochs) if stop_after_epochs else self.config.epochs
        if bad_epochs >= self.config.patience:
            limit, stop_reason = start_epoch, "early_stopping"
        for epoch in range(start_epoch, limit):
            epoch_start = time.perf_counter()
            if self.device.type == "cuda":
                # cuDNN caches an RNN dropout buffer outside model.state_dict and
                # the public CUDA RNG state. At an epoch boundary reseeding also
                # resets this buffer; full and resumed epochs then share a stream.
                torch.cuda.manual_seed_all((self.config.seed + epoch) % 2**32)
            train_stats, _ = self._phase(self._loader(prepared, "train", epoch), indexed_train, training=True, epoch=epoch)
            validation_stats, predictions = self._phase(
                self._loader(prepared, "validation", epoch), indexed_validation, training=False, epoch=epoch,
            )
            if processor.fingerprint != self._processor_fingerprint:
                raise ContractError("frozen A preprocessing changed during training/validation")
            selected = validation_stats.get(self.config.selection_metric)
            if isinstance(selected, bool) or not isinstance(selected, (float, int, np.number)) or not math.isfinite(float(selected)):
                raise ContractError(f"C evaluator returned no finite selection metric: {self.config.selection_metric}")
            selected = float(selected)
            improvement = (best_metric is None or
                           (selected > best_metric + self.config.min_delta if self.config.mode == "max" else selected < best_metric - self.config.min_delta))
            if improvement:
                best_metric, best_epoch, bad_epochs = selected, epoch + 1, 0
            else:
                bad_epochs += 1
            self.scheduler.step()
            completed = epoch + 1
            record = {"epoch": completed, "global_step": self.global_step,
                      "train": train_stats, "validation": validation_stats, "selection_value": selected,
                      "best_metric": best_metric, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
                      "learning_rate": self.optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - epoch_start,
                      "scope": self.scope}
            history.append(_jsonable(record))
            validation_dir = self.output_dir / "validation"
            validation_dir.mkdir(exist_ok=True)
            predictions.to_parquet(validation_dir / f"epoch-{completed:04d}.parquet", index=False)
            write_json(self.output_dir / "history.json", history)
            self._log({"event": "epoch", **record})
            extra = {"history": history, "best_epoch": best_epoch, "target_scale": self.target_scale,
                     "objective": _jsonable(objective_config), "scope": self.scope}
            save_kwargs = dict(scheduler=self.scheduler, scaler=self.scaler, epoch=completed, global_step=self.global_step,
                               best_metric=best_metric, bad_epochs=bad_epochs, config=self.config.to_dict(), preprocessor=processor,
                               model_spec=self.model_spec, contracts=self.contracts, split=split, extra_state=extra)
            if improvement:
                save_checkpoint(self.best_checkpoint, self.model, self.optimizer, **save_kwargs)
            save_checkpoint(self.last_checkpoint, self.model, self.optimizer, **save_kwargs)
            if bad_epochs >= self.config.patience:
                stop_reason = "early_stopping"
                break
        if completed < self.config.epochs and stop_reason != "early_stopping":
            stop_reason = "invocation_epoch_limit"
        if best_metric is None or not self.best_checkpoint.exists() or not self.last_checkpoint.exists():
            raise ContractError("no verified best/last checkpoint was produced")
        bundle_path = None
        if self.scope == "official":
            # Export the selected weights without replacing last-epoch model or optimizer state.
            selected_model, selected_processor = self._selected_checkpoint(processor, split, best_epoch, best_metric)
            bundle_path = self.output_dir / "bundles" / f"epoch{best_epoch:04d}"
            bundle_path.parent.mkdir(exist_ok=True)
            training_record = {"scope": self.scope, "best_epoch": best_epoch, "best_metric": best_metric,
                               "metric": self.config.selection_metric, "target_scale": self.target_scale,
                               "training": self.config.to_dict(), "objective": objective_config,
                               "validation_precision": "fp32", "inference_precision": "fp32",
                               "split": split, "research_only": True}
            if bundle_path.exists():
                from .bundle import load_bundle
                bundled, bundled_processor, manifest = load_bundle(bundle_path)
                if (bundled_processor.fingerprint != selected_processor.fingerprint or
                        manifest["contracts"] != self.contracts or
                        manifest.get("model_spec") != self.model_spec.to_dict() or
                        manifest["training_record"] != training_record or
                        any(not torch.equal(value.cpu(), bundled.state_dict()[name].cpu())
                            for name, value in selected_model.state_dict().items())):
                    raise ContractError("existing selected bundle differs from the verified best checkpoint")
            else:
                save_bundle(bundle_path, selected_model, selected_processor, model_name=self.model_spec.name,
                            model_spec=self.model_spec, window_length=self.model_spec.lookback,
                            contracts=self.contracts, seed=self.config.seed,
                            training_record=training_record, instrument_map=self.instrument_map)
        result = TrainingResult(history, self.best_checkpoint, self.last_checkpoint, bundle_path,
                                float(best_metric), best_epoch, completed, stop_reason)
        write_json(self.output_dir / "result.json", {"best_metric": best_metric, "best_epoch": best_epoch,
                   "completed_epochs": completed, "global_step": self.global_step, "stopped_reason": stop_reason,
                   "best_checkpoint": str(self.best_checkpoint), "last_checkpoint": str(self.last_checkpoint),
                   "bundle_path": str(bundle_path) if bundle_path else None, "scope": self.scope,
                   "official_label_contract_supplied": self.scope == "official", "platform_evaluation_verified": False})
        return result
