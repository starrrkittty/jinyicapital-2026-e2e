from collections.abc import Callable, Mapping
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .data import CanonicalData, _timestamps, _validate_mapping, normalize_bars
from .schema import FEATURES, ContractError
from .targets import LabelSpec, attach_targets

SECTION_TABLE = "cpt_jyc_2026_instruments"
LABEL_TABLE = "cpt_jyc_2026_vwap"


def _unique_keys(frame: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    if not frame.columns.is_unique or not set(keys).issubset(frame.columns):
        raise ContractError(f"{name} 缺少必要键或包含重复字段名：{keys}")
    if frame[keys].isna().any().any():
        raise ContractError(f"{name} 的键不能缺失")
    result = frame.drop_duplicates().copy()
    if result.duplicated(keys).any():
        raise ContractError(f"{name} 同一键存在冲突重复记录")
    result.attrs["duplicate_rows"] = len(frame) - len(result)
    return result


def expected_from_sections(sections: pd.DataFrame, instrument_map: Mapping[int, str]) -> pd.DataFrame:
    """直接使用模板规定的 section 表，不根据行情记录删减股票或时点。"""
    mapping = _validate_mapping(instrument_map)
    inverse = {instrument: instrument_id for instrument_id, instrument in mapping.items()}
    source = _unique_keys(sections, ["date", "instrument"], "官方截面")
    source["date"] = _timestamps(source["date"], "官方截面 date")
    if not source["instrument"].map(lambda item: isinstance(item, str) and bool(item.strip())).all():
        raise ContractError("官方截面 instrument 必须为非空字符串")
    unknown = set(source["instrument"]).difference(inverse)
    if unknown:
        raise ContractError(f"官方截面存在无法映射的 instrument：{sorted(unknown)[:10]}")
    instrument_id = source["instrument"].map(inverse).astype("int64")
    if "instrument_id" in source and not source["instrument_id"].eq(instrument_id).all():
        raise ContractError("官方截面 instrument_id 与显式映射冲突")
    source["instrument_id"] = instrument_id
    source = source.sort_values(["date", "instrument_id"], kind="stable")
    if source.duplicated(["date", "instrument_id"]).any():
        raise ContractError("官方截面规范化后存在重复键")
    result = source[["date", "instrument_id", "instrument"]].rename(columns={"date": "decision_time"}).reset_index(drop=True)
    result["decision_id"] = result["decision_time"].dt.strftime("%Y%m%dT%H%M")
    result["sample_id"] = result["decision_id"] + ":" + result["instrument_id"].astype(str)
    result = result[["sample_id", "decision_id", "instrument_id", "decision_time", "instrument"]]
    result.attrs["source"] = SECTION_TABLE
    result.attrs["duplicate_rows"] = source.attrs.get("duplicate_rows", 0)
    return result


def official_targets(
    expected: pd.DataFrame,
    labels: pd.DataFrame,
    support: pd.DataFrame,
    spec: LabelSpec,
) -> pd.DataFrame:
    """连接官方 vwap_return；支持区间和版本来自显式合同表。"""
    expected_columns = ["sample_id", "decision_time", "instrument_id", "instrument"]
    if not set(expected_columns).issubset(expected.columns):
        raise ContractError(f"应预测表必须包含 {expected_columns}")
    samples = _unique_keys(expected, ["sample_id"], "应预测表")
    if samples.duplicated(["decision_time", "instrument"]).any():
        raise ContractError("应预测表的时间和股票键重复")
    values = _unique_keys(labels, ["date", "instrument"], "官方标签")
    if "vwap_return" not in values:
        raise ContractError("官方标签缺少 vwap_return")
    if not is_numeric_dtype(values["vwap_return"].dtype) or is_bool_dtype(values["vwap_return"].dtype):
        raise ContractError("vwap_return 必须为数值，保持官方给定数值")
    values["date"] = _timestamps(values["date"], "官方标签 date")
    if values.duplicated(["date", "instrument"]).any():
        raise ContractError("官方标签规范化后存在重复键")
    support_columns = ["sample_id", "target_valid", "target_start", "target_end", "label_spec_version"]
    if not isinstance(support, pd.DataFrame) or not set(support_columns).issubset(support.columns):
        raise ContractError(f"必须提供显式目标支持区间表：{support_columns}")
    boundaries = _unique_keys(support[support_columns], ["sample_id"], "目标支持区间")
    if not is_bool_dtype(boundaries["target_valid"].dtype) or boundaries["target_valid"].isna().any():
        raise ContractError("目标支持表 target_valid 必须为非空 bool")
    if not boundaries["label_spec_version"].eq(spec.version).all():
        raise ContractError("目标支持表合同版本不一致")
    if not boundaries["sample_id"].isin(samples["sample_id"]).all():
        raise ContractError("目标支持表包含应预测集合以外的 sample_id")
    result = samples.merge(values[["date", "instrument", "vwap_return"]],
                           left_on=["decision_time", "instrument"], right_on=["date", "instrument"],
                           how="left", sort=False, validate="one_to_one")
    result = result.merge(boundaries, on="sample_id", how="left", sort=False, validate="one_to_one", indicator="support_match")
    target_values = result["vwap_return"].to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isinf(target_values).any():
        raise ContractError("应预测集合匹配的官方 vwap_return 存在无穷值")
    finite = np.isfinite(target_values)
    if (finite & result["support_match"].ne("both").to_numpy()).any():
        raise ContractError("有限官方标签缺少显式支持区间，不能推断 target_end")
    result["target_valid"] = result["target_valid"].astype("boolean").fillna(False).astype(bool) & finite
    result["target"] = result["vwap_return"].where(result["target_valid"], np.nan)
    result["label_spec_version"] = result["label_spec_version"].fillna(spec.version)
    target_columns = ["sample_id", "target", "target_valid", "target_start", "target_end", "label_spec_version"]
    checked = attach_targets(samples[expected_columns], result[target_columns], spec)
    return checked[target_columns]


def _range(start_date, end_date) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None:
        raise ContractError("查询必须提供有序且无时区的显式起止时间")
    if end == end.normalize():
        end += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    if start > end:
        raise ContractError("查询开始时间不能晚于结束时间")
    return start, end


def _query_frame(query: Callable, sql: str, filters: dict, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    result = query(sql, filters=filters).df()
    if not isinstance(result, pd.DataFrame) or "date" not in result:
        raise ContractError("DAI 查询必须返回含 date 的 DataFrame")
    times = _timestamps(result["date"], "查询结果 date")
    if not ((times >= start) & (times <= end)).all():
        raise ContractError("查询结果超出声明的时间过滤范围")
    return result


def fetch_inference_inputs(
    query: Callable,
    datasources: Mapping[str, str],
    *,
    start_date,
    end_date,
    history_start,
    instrument_map: Mapping[int, str],
    bar_end_is_available: bool = False,
    table_key: str = "bar1m",
) -> tuple[CanonicalData, pd.DataFrame]:
    """注入 dai.query 读取纯推理输入；调用者显式声明授权历史起点与频率。

    ``table_key`` 必须与训练时使用的频率一致：窗口长度按**槽数**定义，用错频率会静默改变
    模型看到的时间跨度。网格与行情都从这个键取，避免两者来自不同频率。
    """
    if type(bar_end_is_available) is not bool:
        raise ContractError("bar_end_is_available 必须为 bool")
    start, end = _range(start_date, end_date)
    history, _ = _range(history_start, end_date)
    if history > start:
        raise ContractError("history_start 不能晚于预测开始时间")
    if table_key not in datasources:
        raise ContractError(f"datasources 缺少 {table_key}（必须与训练频率一致）")
    table = datasources[table_key]
    if not isinstance(table, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", table):
        raise ContractError(f"{table_key} 必须为合法的动态表名")
    section_filter = {"date": [start.isoformat(sep=" "), end.isoformat(sep=" ")]}
    sections = _query_frame(query, f"SELECT date, instrument FROM {SECTION_TABLE} ORDER BY date, instrument",
                            section_filter, start=start, end=end)
    expected = expected_from_sections(sections, instrument_map)
    if expected.empty:
        raise ContractError("正式区间的官方应预测集合为空")
    columns = ("date", "instrument", "instrument_id", "adjust_factor") + FEATURES
    filters = {"date": [history.isoformat(sep=" "), end.isoformat(sep=" ")],
               "instrument": sorted(expected["instrument"].unique().tolist())}
    bars = _query_frame(query, f"SELECT {', '.join(columns)} FROM {table} ORDER BY date, instrument_id",
                       filters, start=history, end=end)
    if "instrument" not in bars or not bars["instrument"].isin(filters["instrument"]).all():
        raise ContractError("行情查询结果超出显式股票过滤集合")
    canonical = normalize_bars(bars, source="cloud", instrument_map=instrument_map,
                               bar_end_is_available=bar_end_is_available)
    return canonical, expected


def fetch_label_values(query: Callable, *, start_date, end_date) -> pd.DataFrame:
    """独立训练入口读取官方目标数值，不推断目标公式或支持区间。"""
    start, end = _range(start_date, end_date)
    filters = {"date": [start.isoformat(sep=" "), end.isoformat(sep=" ")]}
    return _query_frame(query, f"SELECT date, instrument, vwap_return FROM {LABEL_TABLE}", filters,
                        start=start, end=end)
