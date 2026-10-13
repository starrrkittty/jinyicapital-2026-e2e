from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from .schema import ContractError, reject_placeholder


@dataclass(frozen=True)
class LabelSpec:
    version: str
    source: str
    target_definition: str
    residualization: str
    boundary_policy: str

    def __post_init__(self):
        for name, value in vars(self).items():
            reject_placeholder(f"标签合同 {name}", value)


def attach_targets(samples: pd.DataFrame, targets: pd.DataFrame, spec: LabelSpec) -> pd.DataFrame:
    # 精确公式尚待官方合同确认；这里仅接入合同已定义的目标表。
    sample_columns = {"sample_id", "decision_time", "instrument_id"}
    target_columns = {
        "sample_id", "target", "target_valid", "target_start", "target_end", "label_spec_version"
    }
    if not sample_columns.issubset(samples.columns):
        raise ContractError(f"样本缺少字段 {sample_columns - set(samples.columns)}")
    if not target_columns.issubset(targets.columns):
        raise ContractError(f"标签缺少字段 {target_columns - set(targets.columns)}")
    for frame in (samples, targets):
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ContractError("sample_id 必须非空且唯一")
    if (target_columns - {"sample_id"}).intersection(samples.columns):
        raise ContractError("样本已包含标签字段，禁止重复连接")
    if not is_bool_dtype(targets["target_valid"]):
        raise ContractError("target_valid 必须为 bool")
    if targets["target_valid"].isna().any():
        raise ContractError("标签有效标志不能为空")
    if not targets["label_spec_version"].eq(spec.version).all():
        raise ContractError("标签合同版本不一致")
    if not targets["sample_id"].isin(samples["sample_id"]).all():
        raise ContractError("标签包含应处理集合以外的 sample_id")
    result = samples.merge(targets[list(sorted(target_columns))], on="sample_id", how="left", sort=False,
                           validate="one_to_one")
    result["target_valid"] = result["target_valid"].astype("boolean").fillna(False).astype(bool)
    result["label_spec_version"] = result["label_spec_version"].fillna(spec.version)
    valid = result["target_valid"]
    values = pd.to_numeric(result["target"], errors="raise")
    if not np.isfinite(values[valid]).all():
        raise ContractError("有效标签必须为有限数值")
    result["target"] = values.where(valid, np.nan)
    for name in ("decision_time", "target_start", "target_end"):
        result[name] = pd.to_datetime(result[name], errors="raise")
    selected = result.loc[valid]
    if selected[["decision_time", "target_start", "target_end"]].isna().any().any():
        raise ContractError("有效标签缺少支持区间")
    if (selected["target_start"] < selected["decision_time"]).any():
        raise ContractError("目标支持区间开始时间早于决策时间")
    if (selected["target_end"] <= selected["target_start"]).any():
        raise ContractError("目标支持区间必须严格递增")
    if not selected["target_end"].dt.normalize().eq(selected["decision_time"].dt.normalize()).all():
        raise ContractError("目标支持区间不能跨夜")
    return result


@dataclass(frozen=True)
class TimeFold:
    name: str
    train_start: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp

    def __post_init__(self):
        boundaries = tuple(pd.Timestamp(value) for value in
                           (self.train_start, self.validation_start, self.validation_end))
        if any(pd.isna(value) or value.tz is not None for value in boundaries):
            raise ContractError("时间折边界必须为有效的本地无时区时间")
        if not boundaries[0] < boundaries[1] < boundaries[2]:
            raise ContractError("时间折边界必须严格递增")
        for name, value in zip(("train_start", "validation_start", "validation_end"), boundaries):
            object.__setattr__(self, name, value)

    def split(self, samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        required = {"decision_time", "target_end", "target_valid", "predictable"}
        if not required.issubset(samples.columns):
            raise ContractError(f"时间切分缺少字段 {required - set(samples.columns)}")
        for name in ("target_valid", "predictable"):
            if not is_bool_dtype(samples[name]) or samples[name].isna().any():
                raise ContractError(f"{name} 必须为非空 bool")
        time = pd.to_datetime(samples["decision_time"], errors="raise")
        end = pd.to_datetime(samples["target_end"], errors="raise")
        if time.isna().any():
            raise ContractError("决策时间不能为空")
        eligible = samples["target_valid"] & samples["predictable"]
        if end[eligible].isna().any():
            raise ContractError("训练或评价样本缺少 target_end")
        train = eligible & time.ge(self.train_start) & time.lt(self.validation_start) & end.lt(self.validation_start)
        validation = eligible & time.ge(self.validation_start) & time.lt(self.validation_end) & end.lt(self.validation_end)
        return samples.loc[train].copy(), samples.loc[validation].copy()


def make_time_folds(trading_days: pd.DatetimeIndex) -> tuple[TimeFold, ...]:
    days = pd.DatetimeIndex(trading_days)
    if days.empty or days.hasnans or days.has_duplicates or not days.is_monotonic_increasing:
        raise ContractError("交易日必须非空、唯一且递增")
    if not days.equals(days.normalize()):
        raise ContractError("交易日必须使用日初时间")
    boundaries = [int(len(days) * ratio) for ratio in (0.60, 0.75, 0.90)]
    if len(set([0, *boundaries, len(days)])) != 5:
        raise ContractError("交易日数量不足以形成三个完整时间折")
    ends = [days[boundaries[1]], days[boundaries[2]], days[-1] + pd.Timedelta(days=1)]
    return tuple(
        TimeFold(name, days[0], days[start], end)
        for name, start, end in zip(("development_1", "development_2", "locked_holdout"), boundaries, ends)
    )
