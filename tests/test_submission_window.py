"""提交层窗口规则的单测：一年历史、决策日、逐日分区，以及 Notebook 的静态契约。

与 ``test_submission.py`` 不同，这里不依赖 Qlib —— 只覆盖无需数据引擎的纯逻辑，
因此在本机也能全量运行。
"""

import importlib

import pandas as pd
import pytest

from ptop.inference import InferencePartition
from ptop.schema import ContractError
from ptop.submission import fetch_decision_days, history_start_for, plan_day_partitions


class _Result:
    def __init__(self, frame):
        self._frame = frame

    def df(self):
        return self._frame


class _FakeQuery:
    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def __call__(self, sql, filters=None):
        self.calls.append((sql, filters))
        for key, frame in self.frames.items():
            if key in sql:
                return _Result(frame.copy())
        raise AssertionError(f"unexpected sql: {sql}")


def test_history_start_for_is_always_one_year_before_start_date():
    assert history_start_for("2025-03-01") == pd.Timestamp("2024-03-01")
    assert history_start_for(pd.Timestamp("2024-02-29")) == pd.Timestamp("2023-02-28")
    assert history_start_for("2025-01-02", years=2) == pd.Timestamp("2023-01-02")


def test_history_start_for_rejects_unusable_inputs():
    with pytest.raises(ContractError, match="正整数"):
        history_start_for("2025-01-01", years=0)
    with pytest.raises(ContractError, match="无时区"):
        history_start_for(pd.Timestamp("2025-01-01", tz="UTC"))


def test_fetch_decision_days_normalises_distinct_days_in_order():
    query = _FakeQuery({"cpt_jyc_2026_instruments": pd.DataFrame({
        "date": ["2024-01-03 10:00:00", "2024-01-02 14:00:00", "2024-01-02 10:00:00"]})})
    days = fetch_decision_days(query, start_date="2024-01-02", end_date="2024-01-03")
    assert days.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    sql, filters = query.calls[0]
    assert "DISTINCT date" in sql and "cpt_jyc_2026_instruments" in sql
    assert filters["date"][0] == "2024-01-02 00:00:00"


def test_fetch_decision_days_rejects_an_empty_window():
    empty = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]")})
    query = _FakeQuery({"cpt_jyc_2026_instruments": empty})
    with pytest.raises(ContractError, match="应预测集合为空"):
        fetch_decision_days(query, start_date="2024-01-02", end_date="2024-01-03")


def test_plan_day_partitions_reads_only_the_decision_day():
    grid = pd.DatetimeIndex(["2024-01-02 10:00", "2024-01-03 10:00"])
    days = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    partitions = plan_day_partitions(days, grid)
    assert all(isinstance(item, InferencePartition) for item in partitions)
    assert [item.start_date for item in partitions] == days
    assert [item.end_date for item in partitions] == days
    # 窗口不跨日：分区历史起点就是当日午夜，避免读入整年行情。
    assert [item.history_start for item in partitions] == days
    assert all(item.trading_grid is grid for item in partitions)


def test_plan_day_partitions_rejects_unsorted_or_non_midnight_days():
    grid = pd.DatetimeIndex(["2024-01-02 10:00"])
    with pytest.raises(ContractError, match="严格递增"):
        plan_day_partitions([pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-02")], grid)
    with pytest.raises(ContractError, match="完整日历日"):
        plan_day_partitions([pd.Timestamp("2024-01-02 10:00")], grid)
    with pytest.raises(ContractError, match="非空 DatetimeIndex"):
        plan_day_partitions([pd.Timestamp("2024-01-02")], pd.DatetimeIndex([]))


def test_notebook_declares_the_graded_main_and_dynamic_history():
    verify = importlib.import_module("submission.verify_submission")
    notebook = verify.Path(__file__).resolve().parent.parent / "submission" / "inference.ipynb"
    assert verify.notebook_defines_main(notebook)
    assert verify.notebook_main_parameters(notebook) == ["datasources", "start_date", "end_date"]
    assert "history_start_for" in verify.notebook_code_source(notebook)


def test_output_frame_check_enforces_columns_and_window(tmp_path):
    verify = importlib.import_module("submission.verify_submission")
    good = pd.DataFrame({"date": pd.to_datetime(["2024-01-02 10:00"]),
                         "instrument": ["000001.SZ"], "score": [0.01]})
    good_path = tmp_path / "good.parquet"
    good.to_parquet(good_path, index=False)
    assert verify.check_output_frame(good_path, "2024-01-02", "2024-01-03") == []

    outside = good.assign(date=pd.to_datetime(["2024-01-05 10:00"]))
    outside_path = tmp_path / "outside.parquet"
    outside.to_parquet(outside_path, index=False)
    assert any("超出" in item for item in verify.check_output_frame(outside_path, "2024-01-02", "2024-01-03"))

    wrong = good.rename(columns={"score": "value"})
    wrong_path = tmp_path / "wrong.parquet"
    wrong.to_parquet(wrong_path, index=False)
    assert any("列必须为" in item for item in verify.check_output_frame(wrong_path, None, None))
