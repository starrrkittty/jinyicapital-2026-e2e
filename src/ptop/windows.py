from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler

from ._vendor.qlib.sampler import TSDataSampler
from .data import CanonicalData, _timestamps
from .schema import FEATURES, ContractError


@dataclass
class WindowBatch:
    x: np.ndarray
    observed: np.ndarray
    step_valid: np.ndarray
    predictable: np.ndarray
    metadata: pd.DataFrame
    bar_times: np.ndarray
    row_positions: np.ndarray


def collate_windows(batch: list[WindowBatch]) -> WindowBatch:
    """合并 DataLoader 单样本结果；顶层定义可供 Windows spawn 序列化。"""
    if not batch or not all(isinstance(item, WindowBatch) for item in batch):
        raise ContractError("collate_windows 需要非空 WindowBatch 列表")
    columns = batch[0].metadata.columns
    length = batch[0].x.shape[1] if batch[0].x.ndim == 3 else None
    for item in batch:
        if item.x.ndim != 3 or item.x.shape[1:] != (length, len(FEATURES)) or item.x.dtype != np.float32:
            raise ContractError("合并窗口必须为相同长度的 float32 [B,L,25]")
        count = item.x.shape[0]
        if item.observed.shape != item.x.shape or item.observed.dtype != bool:
            raise ContractError("合并窗口 observed 的形状或 bool 类型不一致")
        for name in ("step_valid", "bar_times", "row_positions"):
            if getattr(item, name).shape != (count, length):
                raise ContractError(f"合并窗口 {name} 的形状不一致")
        if item.step_valid.dtype != bool or item.predictable.shape != (count,) or item.predictable.dtype != bool:
            raise ContractError("合并窗口的有效标志形状或 bool 类型不一致")
        if item.bar_times.dtype != np.dtype("datetime64[ns]") or item.row_positions.dtype != np.int64:
            raise ContractError("合并窗口的时间或源行索引类型不一致")
        if len(item.metadata) != count or not item.metadata.columns.equals(columns):
            raise ContractError("合并窗口的元数据行数或字段不一致")
    return WindowBatch(
        x=np.concatenate([item.x for item in batch], axis=0),
        observed=np.concatenate([item.observed for item in batch], axis=0),
        step_valid=np.concatenate([item.step_valid for item in batch], axis=0),
        predictable=np.concatenate([item.predictable for item in batch], axis=0),
        metadata=pd.concat([item.metadata for item in batch], ignore_index=True),
        bar_times=np.concatenate([item.bar_times for item in batch], axis=0),
        row_positions=np.concatenate([item.row_positions for item in batch], axis=0),
    )


class _WindowDatasetBase:
    def __init__(
        self,
        data: CanonicalData,
        expected: pd.DataFrame,
        trading_grid: pd.DatetimeIndex,
        *,
        length: int = 120,
        fields: Sequence[str] = FEATURES,
        bar_end_is_available: bool = False,
        local_audit: bool,
    ):
        if type(bar_end_is_available) is not bool:
            raise ContractError("bar_end_is_available 必须为 bool")
        data.validate()
        if local_audit and data.instrument_map:
            raise ContractError("本地数值审计要求未附股票代码映射的 CanonicalData")
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ContractError("窗口 length 必须为正整数")
        if not fields or len(set(fields)) != len(fields) or not set(fields).issubset(FEATURES):
            raise ContractError("模型使用字段必须为 FEATURES 的非空无重复子集")
        grid_name = "observed_grid" if local_audit else "trading_grid"
        if not isinstance(trading_grid, pd.DatetimeIndex):
            raise ContractError(f"{grid_name} 必须为显式 DatetimeIndex")
        grid = _timestamps(trading_grid, grid_name)
        if len(grid) == 0 or not grid.is_unique or not grid.is_monotonic_increasing:
            raise ContractError(f"{grid_name} 必须非空、唯一且已排序")
        table_name = "本地审计端点表" if local_audit else "应预测表"
        required = ["sample_id", "decision_id", "instrument_id", "decision_time"]
        if not expected.columns.is_unique or not set(required).issubset(expected.columns):
            raise ContractError(f"{table_name}必须包含 {required}，且字段名唯一")
        samples = expected.loc[:, required].copy().reset_index(drop=True)
        if samples.isna().any().any() or not samples["sample_id"].is_unique:
            raise ContractError(f"{table_name}标识不能缺失，sample_id 必须唯一")
        if not pd.api.types.is_integer_dtype(samples["instrument_id"].dtype):
            raise ContractError(f"{table_name} instrument_id 必须为整数")
        samples["decision_time"] = _timestamps(samples["decision_time"], "decision_time")
        if samples.duplicated(["decision_time", "instrument_id"]).any():
            raise ContractError(f"{table_name}的决策时间和股票键存在重复")
        if not samples["decision_time"].isin(grid).all():
            raise ContractError(f"决策端点必须包含在显式 {grid_name} 中")
        if (samples.groupby("decision_time")["decision_id"].nunique() != 1).any():
            raise ContractError("同一决策时点必须共用 decision_id")
        if (samples.groupby("decision_id")["decision_time"].nunique() != 1).any():
            raise ContractError("decision_id 不能跨多个决策时点")
        if local_audit:
            samples["audit_scope"] = "local_observed_grid"
            samples["production_visibility_verified"] = False
        elif data.instrument_map:
            unknown = set(samples["instrument_id"]).difference(data.instrument_map)
            if unknown:
                raise ContractError(f"应预测表出现无法映射的 instrument_id：{sorted(unknown)[:10]}")
            samples["instrument"] = samples["instrument_id"].map(data.instrument_map)
        else:
            raise ContractError("构建完整应预测表需要 CanonicalData.instrument_map")
        if data.metadata is not None and "available_at" in data.metadata:
            available_at = _timestamps(data.metadata["available_at"], "available_at", minute=False)
            visibility_basis = "provided_available_at"
            availability_assumption = "none"
        elif bar_end_is_available:
            available_at = pd.DatetimeIndex(data.values.index.get_level_values("datetime"))
            visibility_basis = "bar_end_only"
            availability_assumption = "bar_end_equals_available_at"
        else:
            raise ContractError("必须提供 available_at 或显式确认 bar_end_is_available")
        raw_times = pd.DatetimeIndex(data.values.index.get_level_values("datetime"))
        if (available_at < raw_times).any():
            raise ContractError("available_at 不能早于 Bar 结束时间")
        if local_audit:
            samples["visibility_basis"] = visibility_basis
            samples["availability_assumption"] = availability_assumption
        self.data = data
        self.expected = samples
        self.length = length
        self.fields = tuple(fields)
        self._field_positions = [FEATURES.index(name) for name in fields]
        self._values = data.values.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        self._observed = data.observed.to_numpy(dtype=bool, copy=True)
        self._available_at = available_at.to_numpy(dtype="datetime64[ns]")
        self._samplers: dict[pd.Timestamp, TSDataSampler] = {}
        self._grids: dict[pd.Timestamp, pd.DatetimeIndex] = {}
        self._locations: dict[int, tuple[pd.Timestamp, int]] = {}
        row_ids = pd.Series(np.arange(len(data.values), dtype=np.float64), index=data.values.index)
        # TSDataSampler 只保存源行位置和时间槽位置，重叠窗口在取 batch 时展开。
        for day, day_samples in samples.groupby(samples["decision_time"].dt.normalize(), sort=True):
            day_grid = grid[grid.normalize() == day]
            instruments = sorted(day_samples["instrument_id"].unique())
            index = pd.MultiIndex.from_product([day_grid, instruments], names=["datetime", "instrument"])
            owned = pd.DataFrame({"source_row": row_ids.reindex(index, fill_value=-1),
                                  "time_slot": np.repeat(np.arange(len(day_grid)), len(instruments))}, index=index)
            endpoints = pd.MultiIndex.from_frame(day_samples[["decision_time", "instrument_id"]])
            endpoints.names = index.names
            endpoint_filter = pd.Series(index.isin(endpoints), index=index)
            # 固定版本会清空传入 DataFrame 的列；owned 专供此 sampler 使用。
            sampler = TSDataSampler(owned, start=day_grid[0], end=day_grid[-1], step_len=length,
                                    fillna_type="none", dtype=np.float64, flt_data=endpoint_filter)
            indexed = pd.Series(np.arange(len(sampler)), index=sampler.get_index())
            self._samplers[day] = sampler
            self._grids[day] = day_grid
            for position, row in day_samples.iterrows():
                self._locations[position] = (day, int(indexed.loc[(row["decision_time"], row["instrument_id"])]))

    def __len__(self) -> int:
        return len(self.expected)

    def _sample_rows(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        day, sampler_index = self._locations[index]
        sampled = self._samplers[day][sampler_index]
        step_valid = np.isfinite(sampled[:, 1])
        row_positions = np.full(self.length, -1, dtype=np.int64)
        row_positions[step_valid] = sampled[step_valid, 0].astype(np.int64)
        bar_times = np.full(self.length, np.datetime64("NaT"), dtype="datetime64[ns]")
        bar_times[step_valid] = self._grids[day].to_numpy()[sampled[step_valid, 1].astype(np.int64)]
        present = row_positions >= 0
        decision = self.expected.iloc[index]["decision_time"].to_datetime64()
        visible = np.zeros(self.length, dtype=bool)
        visible[present] = self._available_at[row_positions[present]] <= decision
        row_positions[~visible] = -1
        return row_positions, step_valid, bar_times

    def get_batch(self, indices: Sequence[int]) -> WindowBatch:
        positions = np.asarray(indices)
        if positions.ndim != 1 or (positions.size and not np.issubdtype(positions.dtype, np.integer)):
            raise ContractError("batch 索引必须为一维整数序列")
        if positions.size and ((positions < 0).any() or (positions >= len(self)).any()):
            raise ContractError("batch 样本位置越界")
        positions = positions.astype(np.int64)
        shape = (len(positions), self.length)
        x = np.zeros((*shape, len(FEATURES)), dtype=np.float32)
        observed = np.zeros_like(x, dtype=bool)
        step_valid = np.zeros(shape, dtype=bool)
        bar_times = np.full(shape, np.datetime64("NaT"), dtype="datetime64[ns]")
        row_positions = np.full(shape, -1, dtype=np.int64)
        for batch_index, sample_index in enumerate(positions):
            rows, valid, times = self._sample_rows(int(sample_index))
            present = rows >= 0
            mask = self._observed[rows[present]]
            selected = np.where(mask, self._values[rows[present]], 0.0)
            if (np.abs(selected) > np.finfo(np.float32).max).any():
                raise ContractError(f"样本 {self.expected.iloc[sample_index]['sample_id']} 转换 float32 会溢出")
            x[batch_index, present] = selected.astype(np.float32)
            observed[batch_index, present] = mask
            step_valid[batch_index] = valid
            bar_times[batch_index] = times
            row_positions[batch_index] = rows
        predictable = observed[:, :, self._field_positions].any(axis=(1, 2))
        metadata = self.expected.iloc[positions].reset_index(drop=True).copy()
        metadata["unpredictable_reason"] = np.where(predictable, "", "no_observation_in_model_fields")
        return WindowBatch(x, observed, step_valid, predictable, metadata, bar_times, row_positions)

    def __getitem__(self, index: int) -> WindowBatch:
        return self.get_batch([index])


class WindowDataset(_WindowDatasetBase):
    """由完整应预测表和显式分钟网格驱动的按日惰性窗口。"""

    def __init__(
        self,
        data: CanonicalData,
        expected: pd.DataFrame,
        trading_grid: pd.DatetimeIndex,
        *,
        length: int = 120,
        fields: Sequence[str] = FEATURES,
        bar_end_is_available: bool = False,
    ):
        super().__init__(data, expected, trading_grid, length=length, fields=fields,
                         bar_end_is_available=bar_end_is_available, local_audit=False)


class LocalAuditWindowDataset(_WindowDatasetBase):
    """使用显式观察网格检查数值窗口，不提供正式股票覆盖或生产可见性证明。"""

    def __init__(
        self,
        data: CanonicalData,
        endpoints: pd.DataFrame,
        observed_grid: pd.DatetimeIndex,
        *,
        length: int = 120,
        fields: Sequence[str] = FEATURES,
        bar_end_is_available: bool = False,
    ):
        super().__init__(data, endpoints, observed_grid, length=length, fields=fields,
                         bar_end_is_available=bar_end_is_available, local_audit=True)
        self.endpoints = self.expected


class DecisionBatchSampler(Sampler[list[int]]):
    """逐批返回完整决策截面；各轮洗牌仅使用显式独立随机状态。"""

    def __init__(
        self,
        dataset: WindowDataset,
        *,
        shuffle: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ):
        if not isinstance(dataset, WindowDataset):
            raise ContractError("DecisionBatchSampler 需要 WindowDataset")
        if type(shuffle) is not bool:
            raise ContractError("shuffle 必须为 bool")
        if seed is not None and generator is not None:
            raise ContractError("seed 与 generator 只能提供一个")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ContractError("seed 必须为 [0, 2**64) 范围内的整数")
        if generator is not None and (not isinstance(generator, torch.Generator) or generator.device.type != "cpu"):
            raise ContractError("generator 必须为 CPU torch.Generator")
        if shuffle and seed is None and generator is None:
            raise ContractError("shuffle=True 必须显式提供 seed 或 generator")
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed) if seed is not None else generator
        # 保存样本位置；特征和学习后 token 均不进入 sampler 状态。
        self._groups = tuple(tuple(int(index) for index in indices)
                             for indices in dataset.expected.groupby("decision_id", sort=False).indices.values())

    def __iter__(self) -> Iterator[list[int]]:
        order = torch.randperm(len(self), generator=self.generator).tolist() if self.shuffle else range(len(self))
        for index in order:
            yield list(self._groups[index])

    def __len__(self) -> int:
        return len(self._groups)


def build_windows(data: CanonicalData, expected: pd.DataFrame, trading_grid: pd.DatetimeIndex, **kwargs) -> WindowDataset:
    return WindowDataset(data, expected, trading_grid, **kwargs)


def fit_row_union(dataset: _WindowDatasetBase, *, fit_start: str | pd.Timestamp, fit_end: str | pd.Timestamp) -> pd.MultiIndex:
    """取训练决策可见原始行并集，fit_start/end 包含首尾完整日历日。"""
    start, end = pd.Timestamp(fit_start), pd.Timestamp(fit_end)
    if pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None:
        raise ContractError("拟合日期必须为有效的无时区日期")
    if start != start.normalize() or end != end.normalize():
        raise ContractError("拟合范围使用完整日历日，请提供午夜日期")
    if start > end:
        raise ContractError("拟合开始日期晚于结束日期")
    end += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    references: set[int] = set()
    for index in range(len(dataset)):
        rows, _, _ = dataset._sample_rows(index)
        references.update(rows[rows >= 0].tolist())
    index = dataset.data.values.index.take(sorted(references))
    times = index.get_level_values("datetime")
    return index[(times >= start) & (times <= end)]
