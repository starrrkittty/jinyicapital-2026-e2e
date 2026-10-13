"""Order-preserving inference with official-key and learned-coverage audits."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .bundle import load_bundle
from .data import CanonicalData, _timestamps, _validate_mapping
from .encoders import field_indices, validate_inputs
from .official import fetch_inference_inputs
from .output import audit_output
from .preprocessing import FrozenPreprocessor
from .schema import FEATURES, ContractError
from .windows import DecisionBatchSampler, WindowBatch, WindowDataset, collate_windows


SAMPLE_COLUMNS = ("sample_id", "decision_id", "instrument_id", "decision_time", "instrument")
UNPREDICTABLE_POLICIES = ("drop", "score0")


@dataclass(frozen=True)
class InferenceResult:
    output: pd.DataFrame
    records: pd.DataFrame
    excluded: pd.DataFrame
    coverage: pd.DataFrame

    @property
    def fallback(self) -> pd.DataFrame:
        return self.records.loc[self.records["prediction_status"].eq("fallback_zero")].copy().reset_index(drop=True)


@dataclass(frozen=True)
class InferencePartition:
    """A bounded full-day query; the caller supplies A's complete intraday grid."""

    start_date: str | pd.Timestamp
    end_date: str | pd.Timestamp
    history_start: str | pd.Timestamp
    trading_grid: pd.DatetimeIndex


def _expected_table(expected: pd.DataFrame) -> pd.DataFrame:
    if (not isinstance(expected, pd.DataFrame) or expected.empty or not expected.columns.is_unique
            or not set(SAMPLE_COLUMNS).issubset(expected.columns)):
        raise ContractError(f"推理必须提供完整且非空的官方 expected，包含 {list(SAMPLE_COLUMNS)}")
    table = expected.loc[:, list(SAMPLE_COLUMNS)].copy().reset_index(drop=True)
    if table.isna().any().any() or not table["sample_id"].is_unique:
        raise ContractError("官方 expected 键必须非空且 sample_id 唯一")
    if not pd.api.types.is_integer_dtype(table["instrument_id"]):
        raise ContractError("官方 expected.instrument_id 必须为整数")
    table["decision_time"] = _timestamps(table["decision_time"], "expected.decision_time")
    if (table.duplicated(["decision_time", "instrument"]).any()
            or table.duplicated(["decision_time", "instrument_id"]).any()):
        raise ContractError("官方 expected 的决策时间和股票键重复")
    if not table["instrument"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ContractError("官方 expected.instrument 必须为非空正式代码")
    if ((table.groupby("decision_time")["decision_id"].nunique() != 1).any()
            or (table.groupby("decision_id")["decision_time"].nunique() != 1).any()):
        raise ContractError("官方 expected 的 decision_id 与决策时点必须一一对应")
    if ((table.groupby("instrument_id")["instrument"].nunique() != 1).any()
            or (table.groupby("instrument")["instrument_id"].nunique() != 1).any()):
        raise ContractError("官方 expected 的股票编号和正式代码映射不一致")
    return table


def _model_fields(model) -> tuple[str, ...]:
    if not isinstance(model, torch.nn.Module) or not hasattr(model, "encoder") or not hasattr(model.encoder, "fields"):
        raise ContractError("推理模型必须注册声明 fields 的 A 编码器")
    fields = tuple(model.encoder.fields)
    field_indices(fields)
    return fields


def _validate_batch(batch: WindowBatch, expected: pd.DataFrame, fields: tuple[str, ...], *, lookback=None) -> pd.DataFrame:
    if not isinstance(batch, WindowBatch):
        raise ContractError("推理输入必须为 A WindowBatch")
    arrays = (batch.x, batch.observed, batch.step_valid, batch.predictable, batch.bar_times, batch.row_positions)
    if not all(isinstance(value, np.ndarray) for value in arrays):
        raise ContractError("WindowBatch 数值和 mask 必须使用 NumPy 数组")
    if batch.x.ndim != 3 or batch.x.dtype != np.float32 or min(batch.x.shape[:2]) <= 0:
        raise ContractError("WindowBatch.x 必须为非空 float32 [B,L,25]")
    count, length = batch.x.shape[:2]
    if lookback is not None and length != lookback:
        raise ContractError("WindowBatch 长度与训练模型 lookback 不一致")
    validate_inputs(*(torch.as_tensor(value) for value in (batch.x, batch.observed, batch.step_valid)))
    if batch.predictable.dtype != bool or batch.predictable.shape != (count,):
        raise ContractError("WindowBatch.predictable 必须为 bool [B]")
    predictable = batch.observed[..., field_indices(fields)].any(axis=(1, 2))
    if not np.array_equal(batch.predictable, predictable):
        raise ContractError("WindowBatch.predictable 与模型 encoder.fields 的真实观测不一致")
    if (batch.bar_times.shape != (count, length) or batch.bar_times.dtype != np.dtype("datetime64[ns]")
            or batch.row_positions.shape != (count, length) or batch.row_positions.dtype != np.int64):
        raise ContractError("WindowBatch 时间或原始行索引的形状/类型无效")
    if ((batch.row_positions < -1).any() or (batch.row_positions[~batch.step_valid] != -1).any()
            or not np.isnat(batch.bar_times[~batch.step_valid]).all()
            or np.isnat(batch.bar_times[batch.step_valid]).any()
            or (batch.observed.any(axis=-1) & (batch.row_positions < 0)).any()):
        raise ContractError("WindowBatch 的真实时间槽、结构填充或观测来源不一致")
    metadata = _expected_table(batch.metadata)
    if len(metadata) != count:
        raise ContractError("WindowBatch.metadata 行数与输入不一致")
    if not metadata["sample_id"].isin(expected.index).all():
        raise ContractError("WindowBatch 包含官方 expected 之外的 sample_id")
    reference = expected.loc[metadata["sample_id"], list(SAMPLE_COLUMNS)].reset_index(drop=True)
    if not metadata.equals(reference):
        raise ContractError("WindowBatch 的 sample_id、决策时点或股票代码与官方 expected 不一致")
    decisions = metadata["decision_time"].to_numpy(dtype="datetime64[ns]")
    dates = decisions.astype("datetime64[D]")
    for index in range(count):
        times = batch.bar_times[index, batch.step_valid[index]]
        if (len(times) == 0 or times[-1] != decisions[index] or (times > decisions[index]).any()
                or not (times.astype("datetime64[D]") == dates[index]).all()
                or not (times.astype("datetime64[m]").astype("datetime64[ns]") == times).all()
                or (len(times) > 1 and not (times[1:] > times[:-1]).all())):
            raise ContractError("WindowBatch 时间必须为当日已排序的可见分钟槽，并结束于决策时点")
    return metadata


def _output_from_records(records: pd.DataFrame) -> pd.DataFrame:
    result = records.loc[records["prediction_status"].ne("excluded"), ["decision_time", "instrument", "score"]].copy()
    result.columns = ["date", "instrument", "score"]
    return result.reset_index(drop=True)


def predict_windows(model, batches: Iterable[WindowBatch], expected: pd.DataFrame, *, start_date, end_date,
                    device: str | torch.device = "cpu", microbatch_size: int = 256,
                    unpredictable_policy: str = "drop") -> InferenceResult:
    """Score standardized windows and return official expected order, including skipped keys.

    Microbatches split stocks only inside inference. All supported A/B score models operate
    independently per stock in eval mode. Both policies retain an explicit status record;
    score0 never bypasses A's learned-model 60% coverage and nonempty-timepoint requirements.
    The iterable must contain each expected sample exactly once, even when no field is observed.
    """
    if type(microbatch_size) is not int or microbatch_size <= 0:
        raise ContractError("microbatch_size 必须为正整数")
    if unpredictable_policy not in UNPREDICTABLE_POLICIES:
        raise ContractError(f"unpredictable_policy 必须为 {UNPREDICTABLE_POLICIES}")
    fields = _model_fields(model)
    expected = _expected_table(expected)
    lookup = expected.set_index("sample_id", drop=False)
    model.to(device)
    modes = [(module, module.training) for module in model.modules()]
    pieces, seen = [], set()
    lookback = getattr(getattr(model, "spec", None), "lookback", None)
    try:
        model.eval()
        # A DataLoader without an explicit generator draws its worker base seed
        # even with num_workers=0; iteration must not change the caller RNG.
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            for batch in batches:
                metadata = _validate_batch(batch, lookup, fields, lookback=lookback)
                ids = metadata["sample_id"].tolist()
                if any(key in seen for key in ids):
                    raise ContractError("推理 WindowBatch 之间出现重复 sample_id")
                seen.update(ids)
                scores = np.full(len(metadata), np.nan, dtype=np.float32)
                selected = np.flatnonzero(batch.predictable)
                for start in range(0, len(selected), microbatch_size):
                    positions = selected[start:start + microbatch_size]
                    inputs = [torch.as_tensor(value[positions], device=device)
                              for value in (batch.x, batch.observed, batch.step_valid)]
                    values = model(*inputs)
                    if (not isinstance(values, torch.Tensor) or values.shape != (len(positions),)
                            or not values.is_floating_point() or not torch.isfinite(values).all()):
                        raise ContractError("模型必须返回有限的一维浮点分数 [B]")
                    predicted = values.detach().cpu().float().numpy()
                    if not np.isfinite(predicted).all():
                        raise ContractError("模型分数转换 float32 后出现非有限值")
                    scores[positions] = predicted
                status = np.full(len(metadata), "model", dtype=object)
                reason = np.full(len(metadata), "", dtype=object)
                missing = ~batch.predictable
                reason[missing] = "no_observation_in_model_fields"
                status[missing] = "excluded" if unpredictable_policy == "drop" else "fallback_zero"
                if unpredictable_policy == "score0":
                    scores[missing] = 0.0
                pieces.append(metadata.assign(score=scores, prediction_status=status, reason=reason))
    finally:
        # Preserve caller train/eval flags, including a deliberately frozen submodule.
        for module, mode in modes:
            module.training = mode
    if len(seen) != len(expected):
        missing = expected.loc[~expected["sample_id"].isin(seen), "sample_id"].tolist()
        raise ContractError(f"推理未处理完整官方 expected：缺少 {len(missing)} 个 sample_id，示例={missing[:5]}")
    records = pd.concat(pieces, ignore_index=True).set_index("sample_id", drop=False)
    records = records.loc[expected["sample_id"]].reset_index(drop=True)
    official_keys = expected.rename(columns={"decision_time": "date"})
    learned_records = records.loc[records["prediction_status"].eq("model")]
    learned_output = _output_from_records(learned_records)
    # Run the original auditor on learned scores before adding any fallback.
    coverage = audit_output(learned_output, official_keys, start_date=start_date, end_date=end_date)
    output = _output_from_records(records)
    final_coverage = audit_output(output, official_keys, start_date=start_date, end_date=end_date)
    counts = records.assign(
        excluded=records["prediction_status"].eq("excluded"),
        fallback_zero=records["prediction_status"].eq("fallback_zero"),
    ).groupby("decision_time", sort=True)[["excluded", "fallback_zero"]].sum()
    counts.index.name = "date"
    coverage = coverage.join(counts)
    coverage["output_predicted"] = final_coverage["predicted"]
    coverage["excluded_rate"] = coverage["excluded"] / coverage["expected"]
    coverage["fallback_rate"] = coverage["fallback_zero"] / coverage["expected"]
    excluded = records.loc[records["prediction_status"].eq("excluded")].copy().reset_index(drop=True)
    return InferenceResult(output, records, excluded, coverage)


# A descriptive alias for callers that already own a DataLoader.
predict_batches = predict_windows


def predict_canonical(model, processor: FrozenPreprocessor, raw: CanonicalData, expected: pd.DataFrame,
                      trading_grid: pd.DatetimeIndex, *, start_date, end_date, window_length: int = 120,
                      device: str | torch.device = "cpu", microbatch_size: int = 256,
                      unpredictable_policy: str = "drop") -> InferenceResult:
    """Reuse a frozen A processor; transform once per bounded raw partition."""
    if not isinstance(processor, FrozenPreprocessor):
        raise ContractError("推理必须使用 A FrozenPreprocessor")
    if type(window_length) is not int or window_length <= 0:
        raise ContractError("window_length 必须为正整数")
    fingerprint = processor.fingerprint
    fields = _model_fields(model)
    raw.validate()
    transformed = processor.transform(raw.values, raw.observed)
    processed = CanonicalData(transformed, raw.observed.copy(), raw.metadata, dict(raw.instrument_map))
    dataset = WindowDataset(processed, expected, trading_grid, length=window_length, fields=fields)
    loader = DataLoader(dataset, batch_sampler=DecisionBatchSampler(dataset), collate_fn=collate_windows,
                        generator=torch.Generator(device="cpu").manual_seed(0))
    result = predict_windows(model, loader, expected, start_date=start_date, end_date=end_date, device=device,
                             microbatch_size=microbatch_size, unpredictable_policy=unpredictable_policy)
    if processor.fingerprint != fingerprint:
        raise ContractError("推理过程中冻结处理器状态发生变化")
    return result


def predict_cloud(query: Callable, datasources: Mapping[str, str], model, processor: FrozenPreprocessor, *,
                  start_date, end_date, history_start, instrument_map: Mapping[int, str],
                  trading_grid: pd.DatetimeIndex, bar_end_is_available: bool,
                  device: str | torch.device = "cpu", window_length: int = 120,
                  microbatch_size: int = 256, unpredictable_policy: str = "drop",
                  table_key: str = "bar1m") -> InferenceResult:
    """A cloud input query only: no label fetch, dependency install or processor fitting."""
    raw, expected = fetch_inference_inputs(
        query, datasources, start_date=start_date, end_date=end_date, history_start=history_start,
        instrument_map=instrument_map, bar_end_is_available=bar_end_is_available, table_key=table_key,
    )
    return predict_canonical(model, processor, raw, expected, trading_grid, start_date=start_date,
                             end_date=end_date, window_length=window_length, device=device,
                             microbatch_size=microbatch_size, unpredictable_policy=unpredictable_policy)


def _load_for_inference(path, *, expected_contracts=None, instrument_map=None):
    model, processor, manifest = load_bundle(path)
    if expected_contracts is not None and expected_contracts != manifest["contracts"]:
        raise ContractError("bundle 合同版本与推理请求不一致")
    mapping = _validate_mapping(dict(manifest["instrument_map"]))
    if instrument_map is not None and _validate_mapping(instrument_map) != mapping:
        raise ContractError("推理 instrument_map 与 bundle 保存映射不一致")
    return model, processor, manifest, mapping


def predict_bundle(path, query: Callable, datasources: Mapping[str, str], *, start_date, end_date,
                   history_start, trading_grid: pd.DatetimeIndex, bar_end_is_available: bool,
                   device: str | torch.device = "cpu", microbatch_size: int = 256,
                   unpredictable_policy: str = "drop", expected_contracts=None,
                   instrument_map=None, table_key: str = "bar1m") -> InferenceResult:
    model, processor, manifest, mapping = _load_for_inference(
        path, expected_contracts=expected_contracts, instrument_map=instrument_map)
    return predict_cloud(query, datasources, model, processor, start_date=start_date, end_date=end_date,
                         history_start=history_start, instrument_map=mapping, trading_grid=trading_grid,
                         bar_end_is_available=bar_end_is_available, device=device,
                         window_length=manifest["window_length"], microbatch_size=microbatch_size,
                         unpredictable_policy=unpredictable_policy, table_key=table_key)


def predict_cloud_partitions(query: Callable, datasources: Mapping[str, str], model, processor: FrozenPreprocessor, *,
                             partitions: Iterable[InferencePartition], instrument_map: Mapping[int, str],
                             bar_end_is_available: bool, device: str | torch.device = "cpu",
                             window_length: int = 120, microbatch_size: int = 256,
                             unpredictable_policy: str = "drop", table_key: str = "bar1m") -> Iterator[InferenceResult]:
    """Yield one audited partition at a time; all partitions reuse the same frozen state.

    Full-day partitions keep A's no-cross-day history rule explicit. history_start must be
    at or before that partition's midnight, and trading_grid must include the day's history.
    Labels are never queried and no missing frequency/schema/weight is substituted.
    """
    previous_end = None
    fingerprint = processor.fingerprint
    count = 0
    for partition in partitions:
        if not isinstance(partition, InferencePartition):
            raise ContractError("推理分区必须为 InferencePartition")
        try:
            start, end, history = map(pd.Timestamp, (partition.start_date, partition.end_date, partition.history_start))
        except (ValueError, TypeError) as error:
            raise ContractError("推理分区日期无效") from error
        if (any(pd.isna(value) or value.tz is not None or value != value.normalize() for value in (start, end, history))
                or start > end or history > start or (previous_end is not None and start <= previous_end)):
            raise ContractError("推理分区须为已排序且不重叠的完整日，历史读取需覆盖分区起始日")
        if processor.fingerprint != fingerprint:
            raise ContractError("推理分区之间冻结处理器发生变化")
        result = predict_cloud(query, datasources, model, processor, start_date=start, end_date=end,
                               history_start=history, instrument_map=instrument_map,
                               trading_grid=partition.trading_grid, bar_end_is_available=bar_end_is_available,
                               device=device, window_length=window_length, microbatch_size=microbatch_size,
                               unpredictable_policy=unpredictable_policy, table_key=table_key)
        previous_end = end
        count += 1
        yield result
    if count == 0:
        raise ContractError("至少需要一个显式推理分区")


def predict_bundle_partitions(path, query: Callable, datasources: Mapping[str, str], *,
                              partitions: Iterable[InferencePartition], bar_end_is_available: bool,
                              device: str | torch.device = "cpu", microbatch_size: int = 256,
                              unpredictable_policy: str = "drop", expected_contracts=None,
                              instrument_map=None, table_key: str = "bar1m") -> Iterator[InferenceResult]:
    model, processor, manifest, mapping = _load_for_inference(
        path, expected_contracts=expected_contracts, instrument_map=instrument_map)
    yield from predict_cloud_partitions(query, datasources, model, processor, partitions=partitions,
                                       instrument_map=mapping, bar_end_is_available=bar_end_is_available,
                                       device=device, window_length=manifest["window_length"],
                                       microbatch_size=microbatch_size, unpredictable_policy=unpredictable_policy,
                                       table_key=table_key)
