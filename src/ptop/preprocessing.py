from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ._vendor.qlib.processors import EPS, Fillna, RobustZScoreNorm, ZScoreNorm

from .schema import FEATURES, PRICE_FIELDS, QLIB_COMMIT, ContractError


def _digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _validate(values: pd.DataFrame, observed: pd.DataFrame) -> None:
    if tuple(values.columns) != FEATURES or tuple(observed.columns) != FEATURES:
        raise ContractError("预处理字段及顺序必须与 FEATURES 完全一致")
    if not isinstance(values.index, pd.MultiIndex) or values.index.names != ["datetime", "instrument"]:
        raise ContractError("预处理索引必须为 MultiIndex(datetime, instrument)")
    if not isinstance(values.index.get_level_values("datetime"), pd.DatetimeIndex):
        raise ContractError("datetime 索引必须使用 DatetimeIndex")
    if values.index.get_level_values("datetime").tz is not None:
        raise ContractError("规范时间必须使用已确认的本地无时区时间")
    if values.index.has_duplicates or values.index.to_frame(index=False).isna().any().any():
        raise ContractError("规范原始行键必须唯一且完整")
    if not values.index.equals(observed.index):
        raise ContractError("observed 的索引和顺序必须与数值完全一致")
    if not all(dtype == np.dtype("bool") for dtype in observed.dtypes):
        raise ContractError("observed 必须使用 bool 类型")
    if not all(dtype == np.dtype("float64") for dtype in values.dtypes):
        raise ContractError("拟合和预处理输入必须使用规范 float64 数值")
    raw, mask = values.to_numpy(), observed.to_numpy()
    if not np.isfinite(raw[mask]).all():
        raise ContractError("有效观测中出现非有限数值")
    if not np.isnan(raw[~mask]).all():
        raise ContractError("缺失观测的规范数值必须为 NaN")


def _feature_frame(values: pd.DataFrame) -> pd.DataFrame:
    result = values.copy(deep=True)
    result.columns = pd.MultiIndex.from_product([["feature"], values.columns])
    return result


class FrozenPreprocessor:
    def __init__(
        self,
        mode: Literal["N0", "N1", "N2"] = "N0",
        *,
        n2_base: Literal["N0", "N1"] = "N0",
    ):
        if mode not in {"N0", "N1", "N2"} or n2_base not in {"N0", "N1"}:
            raise ContractError("预处理类型必须为 N0、N1 或 N2，N2 基础类型必须为 N0 或 N1")
        self.mode = mode
        self.base = n2_base if mode == "N2" else mode
        self._processor = None
        self._fitted = False
        self._frozen = False

    def _new_processor(self):
        kwargs = dict(
            fit_start_time=self.fit_start,
            fit_end_time=self.fit_end,
            fields_group="feature",
        )
        if self.base == "N0":
            return RobustZScoreNorm(**kwargs, clip_outlier=False)
        return ZScoreNorm(**kwargs)

    def _fit_official(self, values: pd.DataFrame):
        processor = self._new_processor()
        processor.fit(_feature_frame(values).sort_index())
        return processor

    def fit(
        self,
        values: pd.DataFrame,
        observed: pd.DataFrame,
        *,
        referenced_rows: pd.MultiIndex,
        fit_start,
        fit_end,
        by_column: bool = False,
    ) -> FrozenPreprocessor:
        """使用训练窗口引用的唯一原始行；fit_start/fit_end 为包含首尾的日历日期。"""
        if self._fitted:
            raise ContractError("处理器已经拟合；新折必须创建独立处理器")
        _validate(values, observed)
        start, end = pd.Timestamp(fit_start), pd.Timestamp(fit_end)
        if pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None:
            raise ContractError("拟合日期必须为有效的无时区日期")
        if start != start.normalize() or end != end.normalize():
            raise ContractError("拟合范围使用完整日历日，请提供午夜日期")
        if start > end:
            raise ContractError("拟合开始日期晚于结束日期")
        self.fit_start = start.isoformat()
        self.fit_end = (end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)).isoformat()
        if not isinstance(referenced_rows, pd.MultiIndex) or referenced_rows.names != values.index.names:
            raise ContractError("训练引用行必须为 MultiIndex(datetime, instrument)")
        rows = referenced_rows.drop_duplicates()
        times = rows.get_level_values("datetime")
        if not isinstance(times, pd.DatetimeIndex) or times.tz is not None or times.hasnans:
            raise ContractError("训练引用时间必须为完整的无时区 DatetimeIndex")
        rows = rows[(times >= start) & (times <= pd.Timestamp(self.fit_end))].sort_values()
        if len(rows) == 0:
            raise ContractError("拟合日期内没有训练窗口引用的原始行")
        missing = rows.difference(values.index)
        if len(missing):
            raise ContractError(f"训练引用行缺少原始数据：{list(missing[:3])}")
        training = values.loc[rows].copy()
        counts = training.notna().sum()
        empty = counts[counts == 0].index.tolist()
        if empty:
            raise ContractError(f"训练范围内字段没有有效观测：{empty}")
        if by_column:
            processors = [self._fit_official(training.loc[:, [field]]) for field in FEATURES]
            processor = self._new_processor()
            processor.cols = _feature_frame(training.iloc[:0]).columns
            processor.mean_train = np.concatenate([item.mean_train for item in processors])
            processor.std_train = np.concatenate([item.std_train for item in processors])
            if self.base == "N1":
                processor.ignore = np.concatenate([item.ignore for item in processors])
        else:
            processor = self._fit_official(training)
        if self.mode == "N2":
            # 将共享组转换成官方处理器的单列输入，统计和常量语义全部由官方类处理。
            pooled = training.loc[:, list(PRICE_FIELDS)].to_numpy().reshape(-1)
            pooled_index = pd.MultiIndex.from_arrays(
                [np.repeat(training.index.get_level_values("datetime"), len(PRICE_FIELDS)), np.arange(len(pooled))],
                names=["datetime", "instrument"],
            )
            pooled_frame = pd.DataFrame({"shared_price": pooled}, index=pooled_index)
            price_processor = self._fit_official(pooled_frame)
            price_positions = [FEATURES.index(field) for field in PRICE_FIELDS]
            processor.mean_train[price_positions] = price_processor.mean_train[0]
            processor.std_train[price_positions] = price_processor.std_train[0]
            if self.base == "N1":
                processor.ignore[price_positions] = price_processor.ignore[0]
        if not np.isfinite(processor.mean_train).all() or not np.isfinite(processor.std_train).all():
            raise ContractError("官方处理器产生非有限中心或尺度")
        if (processor.std_train <= 0).any():
            raise ContractError("官方处理器产生非正尺度")
        self._processor = processor
        self.fit_row_count = len(rows)
        self.fit_rows_fingerprint = hashlib.sha256(pd.util.hash_pandas_object(rows).to_numpy().tobytes()).hexdigest()
        self.valid_counts = counts.astype(int).tolist()
        self.degenerate_fields = (
            [field for field, scale in zip(FEATURES, processor.std_train) if scale == EPS * 1.4826]
            if self.base == "N0"
            else []
        )
        self.zero_variance_fields = (
            [field for field, ignored in zip(FEATURES, processor.ignore) if ignored]
            if self.base == "N1"
            else []
        )
        self.training_constants = {field: float(training[field].dropna().iloc[0]) for field in self.zero_variance_fields}
        self._fitted = True
        self.training_diagnostics = self.diagnostics(training, observed.loc[rows])
        return self

    def _transform(self, values: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
        _validate(values, observed)
        if not self._fitted:
            raise ContractError("处理器尚未拟合或恢复状态")
        before = self._numeric_fingerprint()
        if self._frozen and before != self._expected_numeric_fingerprint:
            raise ContractError("冻结数值状态被外部修改")
        frame = _feature_frame(values).sort_index()
        transformed = Fillna(fields_group="feature", fill_value=0)(self._processor(frame))["feature"]
        transformed = transformed.reindex(values.index)
        raw = transformed.to_numpy()
        if not np.isfinite(raw).all():
            raise ContractError("预处理结果包含非有限 float64 数值")
        with np.errstate(over="ignore", invalid="ignore"):
            result = transformed.astype(np.float32)
        if not np.isfinite(result.to_numpy()).all():
            raise ContractError("预处理结果转换为 float32 后出现非有限值")
        if (result.to_numpy()[~observed.to_numpy()] != 0).any():
            raise ContractError("缺失观测经过 Fillna 后未保持零值")
        if before != self._numeric_fingerprint():
            raise ContractError("推理期间预处理状态发生变化")
        return result

    def transform(self, values: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
        if not self._frozen:
            raise ContractError("处理器尚未通过尺度审阅并冻结")
        return self._transform(values, observed)

    def diagnostics(self, values: pd.DataFrame, observed: pd.DataFrame) -> dict:
        """分组仅用于审计，任何诊断均不改变拟合状态。"""
        transformed = self._transform(values, observed)
        mask = observed.to_numpy()
        raw = values.to_numpy()
        out = transformed.to_numpy()
        quantiles = [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]

        def describe(selected: np.ndarray) -> dict:
            fields = {}
            for column, field in enumerate(FEATURES):
                valid = selected & mask[:, column]
                x, z = raw[valid, column], out[valid, column]
                count = len(x)
                fields[field] = {
                    "valid_count": count,
                    "center": float(self._processor.mean_train[column]),
                    "scale": float(self._processor.std_train[column]),
                    "zero_fraction": float(np.mean(x == 0)) if count else None,
                    "quantiles": np.quantile(x, quantiles).tolist() if count else None,
                    "minimum": float(np.min(x)) if count else None,
                    "maximum": float(np.max(x)) if count else None,
                    "transformed_minimum": float(np.min(z)) if count else None,
                    "transformed_maximum": float(np.max(z)) if count else None,
                    "float32_finite": bool(np.isfinite(z).all()),
                    "tanh_abs_ge_0_99_fraction": float(np.mean(np.abs(np.tanh(z)) >= 0.99)) if count else None,
                    "distinct_raw_count": int(np.unique(x).size),
                    "distinct_float32_count": int(np.unique(z).size),
                    "zero_variance_training": field in self.zero_variance_fields,
                    "training_constant_changed": bool(
                        count and field in self.training_constants and np.any(x != self.training_constants[field])
                    ),
                }
            return fields

        # 价格量级按股票在本次审计数据中的有效 close 中位数确定；缺失组保留。
        close = values["close"]
        price_level = close.groupby(level="instrument").median().abs()
        price_groups = pd.Series("missing", index=price_level.index)
        positive = price_level > 0
        price_groups.loc[positive] = np.floor(np.log10(price_level.loc[positive])).astype(int).map(lambda n: f"1e{n}")
        price_groups.loc[price_level == 0] = "zero"
        row_prices = values.index.get_level_values("instrument").map(price_groups).to_numpy()
        time_groups = values.index.get_level_values("datetime").strftime("%H:%M").to_numpy()
        return {
            "quantile_probabilities": quantiles,
            "tanh_diagnostic": "tanh(normalized raw value); ALSTM Linear 输出需另行检查",
            "fields": describe(np.ones(len(values), dtype=bool)),
            "by_price_magnitude": {str(group): describe(row_prices == group) for group in np.unique(row_prices)},
            "by_intraday_time": {str(group): describe(time_groups == group) for group in np.unique(time_groups)},
            "degenerate_fields": list(self.degenerate_fields),
            "zero_variance_fields": list(self.zero_variance_fields),
        }

    def freeze(self, *, scale_review: str) -> FrozenPreprocessor:
        if not self._fitted or not hasattr(self, "training_diagnostics"):
            raise ContractError("冻结前必须完成拟合与数值诊断")
        if self._frozen:
            raise ContractError("处理器已经冻结")
        if self.degenerate_fields:
            raise ContractError(f"N0 的原始 MAD 为零，不具备冻结资格：{self.degenerate_fields}")
        if not isinstance(scale_review, str) or not scale_review.strip():
            raise ContractError("冻结需要可追溯的尺度审阅记录标识")
        self.scale_review = scale_review
        self._expected_numeric_fingerprint = self._numeric_fingerprint()
        self._frozen = True
        return self

    def _numeric_state(self) -> dict:
        if not self._fitted:
            raise ContractError("处理器没有数值状态")
        return {
            "center": self._processor.mean_train.tolist(),
            "scale": self._processor.std_train.tolist(),
            "ignore": self._processor.ignore.tolist() if self.base == "N1" else None,
        }

    def _numeric_fingerprint(self) -> str:
        return _digest(self._numeric_state())

    def state_dict(self) -> dict:
        if not self._frozen:
            raise ContractError("仅可导出通过尺度审阅的冻结状态")
        if self._numeric_fingerprint() != self._expected_numeric_fingerprint:
            raise ContractError("冻结数值状态被外部修改")
        payload = {
            "state_version": 1,
            "qlib_commit": QLIB_COMMIT,
            "fields": list(FEATURES),
            "mode": self.mode,
            "base": self.base,
            "axis": "shared_price_and_per_field" if self.mode == "N2" else "per_field",
            "clip_outlier": False,
            "fit_start": self.fit_start,
            "fit_end": self.fit_end,
            "fit_row_count": self.fit_row_count,
            "fit_rows_fingerprint": self.fit_rows_fingerprint,
            "valid_counts": list(self.valid_counts),
            "training_constants": dict(self.training_constants),
            "scale_review": self.scale_review,
            "statistics": self._numeric_state(),
        }
        return {**payload, "fingerprint": _digest(payload)}

    @property
    def fingerprint(self) -> str:
        return self.state_dict()["fingerprint"]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.state_dict(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )

    @classmethod
    def from_state_dict(cls, state: dict) -> FrozenPreprocessor:
        expected_keys = {
            "state_version", "qlib_commit", "fields", "mode", "base", "axis", "clip_outlier", "fit_start", "fit_end",
            "fit_row_count", "fit_rows_fingerprint", "valid_counts", "training_constants", "scale_review", "statistics", "fingerprint",
        }
        if not isinstance(state, dict) or set(state) != expected_keys:
            raise ContractError("冻结状态字段缺失或包含未知字段")
        payload = {key: value for key, value in state.items() if key != "fingerprint"}
        if state["fingerprint"] != _digest(payload):
            raise ContractError("冻结状态指纹不匹配")
        if state["state_version"] != 1 or state["qlib_commit"] != QLIB_COMMIT or state["fields"] != list(FEATURES):
            raise ContractError("冻结状态版本、Qlib commit 或字段顺序不匹配")
        result = cls(state["mode"], n2_base=state["base"])
        expected_axis = "shared_price_and_per_field" if result.mode == "N2" else "per_field"
        if state["base"] != result.base or state["axis"] != expected_axis or state["clip_outlier"] is not False:
            raise ContractError("冻结预处理类型、统计轴或裁剪配置不匹配")
        statistics = state["statistics"]
        if not isinstance(statistics, dict) or set(statistics) != {"center", "scale", "ignore"}:
            raise ContractError("冻结数值状态不完整")
        center = np.asarray(statistics["center"], dtype=np.float64)
        scale = np.asarray(statistics["scale"], dtype=np.float64)
        if center.shape != (len(FEATURES),) or scale.shape != (len(FEATURES),):
            raise ContractError("冻结数值状态维度不匹配")
        if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ContractError("冻结中心和尺度必须有限，尺度必须为正")
        counts = state["valid_counts"]
        if (
            type(state["fit_row_count"]) is not int or state["fit_row_count"] <= 0
            or not isinstance(counts, list) or len(counts) != len(FEATURES)
            or any(type(count) is not int or count <= 0 or count > state["fit_row_count"] for count in counts)
        ):
            raise ContractError("冻结状态中的拟合行数或有效数量无效")
        if not isinstance(state["scale_review"], str) or not state["scale_review"].strip():
            raise ContractError("冻结状态缺少尺度审阅记录")
        start, end = pd.Timestamp(state["fit_start"]), pd.Timestamp(state["fit_end"])
        if pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None or start > end:
            raise ContractError("冻结拟合范围无效")
        result.fit_start, result.fit_end = state["fit_start"], state["fit_end"]
        processor = result._new_processor()
        processor.cols = pd.MultiIndex.from_product([["feature"], FEATURES])
        processor.mean_train, processor.std_train = center, scale
        if result.base == "N1":
            ignored = statistics["ignore"]
            if not isinstance(ignored, list) or len(ignored) != len(FEATURES) or any(type(x) is not bool for x in ignored):
                raise ContractError("N1 冻结状态缺少完整 ignore 数组")
            processor.ignore = np.asarray(ignored, dtype=bool)
            if (center[processor.ignore] != 0).any() or (scale[processor.ignore] != 1).any():
                raise ContractError("N1 零方差状态不符合官方处理器语义")
        elif statistics["ignore"] is not None:
            raise ContractError("N0 不包含 N1 的 ignore 状态")
        if result.base == "N0" and (scale == EPS * 1.4826).any():
            raise ContractError("N0 退化状态不具备冻结资格")
        if result.mode == "N2":
            positions = [FEATURES.index(field) for field in PRICE_FIELDS]
            if np.unique(center[positions]).size != 1 or np.unique(scale[positions]).size != 1:
                raise ContractError("N2 价格字段没有共享相同中心和尺度")
        result._processor = processor
        for attribute in ("fit_row_count", "fit_rows_fingerprint", "valid_counts", "scale_review"):
            setattr(result, attribute, state[attribute])
        result.degenerate_fields = []
        result.zero_variance_fields = (
            [field for field, ignored in zip(FEATURES, processor.ignore) if ignored] if result.base == "N1" else []
        )
        constants = state["training_constants"]
        if (
            not isinstance(constants, dict) or set(constants) != set(result.zero_variance_fields)
            or not all(isinstance(value, (int, float)) and np.isfinite(value) for value in constants.values())
        ):
            raise ContractError("冻结状态缺少训练期零方差字段的原值")
        result.training_constants = dict(constants)
        result._fitted = result._frozen = True
        result._expected_numeric_fingerprint = result._numeric_fingerprint()
        return result

    @classmethod
    def load(cls, path: str | Path) -> FrozenPreprocessor:
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))
