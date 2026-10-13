"""Submission entry glue for the one allowed Notebook.

The platform's judge imports ``main`` from the submitted notebook and calls it as
``main(datasources, start_date, end_date)``. It receives only logical datasource names
and the evaluation window. Everything else — the instrument mapping, the intraday
trading grid, the frozen processor and the weights — must come from the query result or
the saved bundle. This module is that glue; it deliberately contains no preprocessing of
its own, so the Notebook cannot drift from the shared code path.

Notebook body (a few lines, nothing else):

    import dai, ptop.submission as submission

    def main(datasources, start_date, end_date):
        return submission.run_submission(
            query=dai.query, datasources=datasources, bundle_path="weights/epoch0007",
            start_date=start_date, end_date=end_date,
            history_start=submission.history_start_for(start_date),
            table_key="bar30m")

``history_start`` is derived as one authorised year before ``start_date`` per the track
rules (never a hard-coded year), and ``bar_end_is_available`` stays True because the
official ``date`` column is the bar's end time (see docs/data_reference.md). ``table_key``
has no default on purpose: it must name the same frequency the bundle was trained on,
because the window length counts bars.
"""

from collections.abc import Callable, Mapping, Sequence

import pandas as pd

from .data import _validate_mapping
from .inference import InferencePartition, predict_bundle, predict_bundle_partitions
from .official import SECTION_TABLE, _query_frame, _range
from .output import audit_output
from .schema import ContractError

OUTPUT_COLUMNS = ["date", "instrument", "score"]


def history_start_for(start_date, *, years: int = 1) -> pd.Timestamp:
    """评测区间前一年：赛道规则授权的历史起点，绝不写死某个年份。

    「验证集在正式评测区间前额外提供一年历史数据」只授权 ``start_date`` 之前**一年**；
    写死更早的日期既超出授权，也会让逐日推理多读无谓的行情。
    """
    if type(years) is not int or years < 1:
        raise ContractError("历史年数必须为正整数")
    start = pd.Timestamp(start_date)
    if pd.isna(start) or start.tz is not None:
        raise ContractError("start_date 必须为无时区的显式日期")
    return start - pd.DateOffset(years=years)


def fetch_decision_days(query: Callable, *, start_date, end_date,
                        table: str = SECTION_TABLE) -> pd.DatetimeIndex:
    """正式区间内有官方应预测样本的交易日（按截面表日期去重）。"""
    if not isinstance(table, str) or not table.strip():
        raise ContractError("截面表名必须为非空字符串")
    start, end = _range(start_date, end_date)
    frame = _query_frame(query, f"SELECT DISTINCT date FROM {table} ORDER BY date",
                         {"date": [start.isoformat(sep=" "), end.isoformat(sep=" ")]}, start=start, end=end)
    days = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise")).normalize().unique().sort_values()
    if days.empty:
        raise ContractError("正式区间的官方应预测集合为空")
    if (days < start.normalize()).any() or (days > end.normalize()).any():
        raise ContractError("截面表返回的决策日超出声明区间")
    return days


def plan_day_partitions(days: Sequence, grid: pd.DatetimeIndex) -> list[InferencePartition]:
    """把正式区间内的每一天变成一个整日分区。

    窗口不跨日，所以每个分区只需要当日行情；这正是逐日推理能把一年行情拆开、把内存与
    总时长压进 3 小时的原因。所有分区复用同一份冻结处理器与权重。
    """
    if not isinstance(grid, pd.DatetimeIndex) or grid.empty:
        raise ContractError("分钟网格必须为非空 DatetimeIndex")
    partitions: list[InferencePartition] = []
    previous = None
    for raw in days:
        day = pd.Timestamp(raw)
        if pd.isna(day) or day.tz is not None or day != day.normalize():
            raise ContractError("决策日必须为无时区的完整日历日")
        if previous is not None and day <= previous:
            raise ContractError("决策日必须严格递增且不重复")
        partitions.append(InferencePartition(start_date=day, end_date=day, history_start=day, trading_grid=grid))
        previous = day
    if not partitions:
        raise ContractError("至少需要一个整日推理分区")
    return partitions


def instrument_map_from_frame(frame: pd.DataFrame) -> dict[int, str]:
    """Extract the ``instrument_id -> instrument`` bijection from a queried frame.

    Both columns appear in the bar and section tables, so the mapping is read from the
    platform instead of being hard-coded. A one-way or conflicting mapping is an error:
    silently picking one code for an id would corrupt every later join.
    """
    required = ("instrument_id", "instrument")
    if not isinstance(frame, pd.DataFrame) or not set(required).issubset(frame.columns):
        raise ContractError(f"股票映射表必须包含 {required}")
    pairs = frame.loc[:, list(required)].drop_duplicates()
    if pairs.empty or pairs.isna().any().any():
        raise ContractError("股票映射表不能为空或含缺失键")
    duplicated = pairs["instrument_id"].duplicated().any()
    if duplicated or pairs["instrument"].duplicated().any():
        raise ContractError("instrument_id 与 instrument 必须一一对应")
    mapping = dict(zip(pairs["instrument_id"].astype(int), pairs["instrument"].astype(str)))
    return _validate_mapping(mapping)


def fetch_instrument_map(query: Callable, *, start_date, end_date,
                         table: str = SECTION_TABLE) -> dict[int, str]:
    """Read the id/code bijection for a declared window; the section table is authoritative."""
    if not isinstance(table, str) or not table.strip():
        raise ContractError("股票映射表名必须为非空字符串")
    start, end = _range(start_date, end_date)
    frame = _query_frame(query,
                         f"SELECT date, instrument_id, instrument FROM {table} ORDER BY instrument_id",
                         {"date": [start.isoformat(sep=" "), end.isoformat(sep=" ")]}, start=start, end=end)
    return instrument_map_from_frame(frame)


def fetch_trading_grid(query: Callable, datasources: Mapping[str, str], *, history_start,
                       end_date, table_key: str = "bar1m") -> pd.DatetimeIndex:
    """The intraday minute grid actually present in the bar table, read not guessed.

    Windows must end exactly at a decision time and may not cross a day, so the grid has
    to be the platform's own minute slots rather than a rebuilt natural-minute range.
    """
    if table_key not in datasources:
        raise ContractError(f"datasources 缺少 {table_key}")
    table = datasources[table_key]
    if not isinstance(table, str) or not table.strip():
        raise ContractError(f"{table_key} 必须为合法的动态表名")
    start, end = _range(history_start, end_date)
    frame = _query_frame(query, f"SELECT DISTINCT date FROM {table} ORDER BY date",
                         {"date": [start.isoformat(sep=" "), end.isoformat(sep=" ")]}, start=start, end=end)
    grid = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise")).unique().sort_values()
    if grid.empty:
        raise ContractError("行情表在声明区间内没有分钟网格")
    if not grid.equals(grid.floor("min")):
        raise ContractError("分钟网格必须精确到分钟")
    return grid


def run_submission(query: Callable, datasources: Mapping[str, str], bundle_path, *,
                   start_date, end_date, history_start, bar_end_is_available: bool = True,
                   table_key: str, device: str = "cpu", microbatch_size: int = 256,
                   unpredictable_policy: str = "drop", expected_contracts=None) -> pd.DataFrame:
    """Produce the graded frame: exactly ``date``, ``instrument``, ``score``.

    ``table_key`` is deliberately **required**: the window length is a number of bars, so
    reading a different frequency than the one the bundle was trained on would silently
    change how much history the model sees. The grid and the bars are both read through
    this same key, so they can never come from different frequencies.

    Inference runs one full day at a time (``predict_bundle_partitions``): every partition
    reuses the same frozen processor and weights, reads only that day's bars, and is audited
    on its own. A bounded window therefore stays inside the 3-hour budget instead of loading
    a whole authorised year into memory at once. If the evaluation window is not day-aligned
    the day split cannot respect the exact bounds, so the single-pass path is used.

    Labels are never queried. The instrument mapping is taken from the saved bundle so it
    cannot disagree with training, and the concatenated output is re-audited against the
    official expected set (column order, minute precision, unique keys, finite scores,
    non-empty timepoints, a per-timepoint missing rate at or below 40%, and dates inside
    ``start_date``..``end_date``).
    """
    if type(bar_end_is_available) is not bool:
        raise ContractError("bar_end_is_available 必须为 bool")
    if not isinstance(table_key, str) or not table_key.strip():
        raise ContractError("table_key 必须显式声明（例如 bar1m / bar30m），且与训练频率一致")
    start, end = _range(start_date, end_date)
    history, _ = _range(history_start, end_date)
    if history > start:
        raise ContractError("history_start 不能晚于预测开始时间")
    grid = fetch_trading_grid(query, datasources, history_start=history_start, end_date=end_date,
                              table_key=table_key)
    if start == start.normalize() and end == end.normalize():
        days = fetch_decision_days(query, start_date=start_date, end_date=end_date)
        partitions = plan_day_partitions(days, grid)
        outputs, records = [], []
        for result in predict_bundle_partitions(
                bundle_path, query, datasources, partitions=partitions,
                bar_end_is_available=bar_end_is_available, device=device, microbatch_size=microbatch_size,
                unpredictable_policy=unpredictable_policy, expected_contracts=expected_contracts,
                table_key=table_key):
            outputs.append(result.output)
            records.append(result.records)
    else:
        result = predict_bundle(
            bundle_path, query, datasources, start_date=start_date, end_date=end_date,
            history_start=history_start, trading_grid=grid, bar_end_is_available=bar_end_is_available,
            device=device, microbatch_size=microbatch_size, unpredictable_policy=unpredictable_policy,
            expected_contracts=expected_contracts, table_key=table_key,
        )
        outputs, records = [result.output], [result.records]
    output = (pd.concat(outputs, ignore_index=True) if outputs else
              pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "instrument": pd.Series(dtype="str"),
                            "score": pd.Series(dtype="float32")}))
    expected = pd.concat(records, ignore_index=True)[["decision_time", "instrument"]].rename(
        columns={"decision_time": "date"})
    audit_output(output, expected, start_date=start_date, end_date=end_date)
    if list(output.columns) != OUTPUT_COLUMNS or output.empty:
        raise ContractError(f"提交输出必须恰好为 {OUTPUT_COLUMNS} 且非空")
    return output.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)
