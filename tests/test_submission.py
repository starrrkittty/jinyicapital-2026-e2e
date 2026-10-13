"""Submission glue: the mapping and minute grid the Notebook must read, not guess.

Needs the pinned Qlib because the inference path is imported; the cases skip without it.
"""

import pandas as pd
import pytest

from ptop.schema import ContractError
from ptop.submission import (OUTPUT_COLUMNS, fetch_instrument_map, fetch_trading_grid,
                             instrument_map_from_frame)

pytest.importorskip("qlib")


class _Result:
    def __init__(self, frame):
        self._frame = frame

    def df(self):
        return self._frame


class _FakeQuery:
    """Mimics ``dai.query(sql, filters=...)`` returning an object with ``.df()``."""

    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def __call__(self, sql, filters=None):
        self.calls.append((sql, filters))
        for key, frame in self.frames.items():
            if key in sql:
                return _Result(frame.copy())
        raise AssertionError(f"unexpected sql: {sql}")


def test_instrument_map_from_frame_validates_a_bijection():
    frame = pd.DataFrame({"instrument_id": [1, 2], "instrument": ["000001.SZ", "000002.SZ"]})
    assert instrument_map_from_frame(frame) == {1: "000001.SZ", 2: "000002.SZ"}
    with pytest.raises(ContractError, match="一一对应"):
        instrument_map_from_frame(pd.DataFrame({"instrument_id": [1, 1], "instrument": ["A", "B"]}))
    with pytest.raises(ContractError, match="一一对应"):
        instrument_map_from_frame(pd.DataFrame({"instrument_id": [1, 2], "instrument": ["A", "A"]}))
    with pytest.raises(ContractError, match="必须包含"):
        instrument_map_from_frame(pd.DataFrame({"instrument_id": [1]}))
    with pytest.raises(ContractError, match="为空或含缺失"):
        instrument_map_from_frame(pd.DataFrame({"instrument_id": [1], "instrument": [None]}))


def test_fetch_instrument_map_filters_by_the_declared_window():
    query = _FakeQuery({"cpt_jyc_2026_instruments": pd.DataFrame({
        "date": ["2024-01-02 10:00:00", "2024-01-02 10:00:00"],
        "instrument_id": [1, 2], "instrument": ["000001.SZ", "000002.SZ"]})})
    mapping = fetch_instrument_map(query, start_date="2024-01-02", end_date="2024-01-05")
    assert mapping == {1: "000001.SZ", 2: "000002.SZ"}
    sql, filters = query.calls[0]
    assert "cpt_jyc_2026_instruments" in sql and "date" in sql
    # _range expands a bare end date to the end of that day.
    assert filters["date"][0] == "2024-01-02 00:00:00"
    assert filters["date"][1].startswith("2024-01-05 23:59:59")


def test_fetch_trading_grid_returns_the_platform_minute_slots():
    stamps = ["2024-01-02 09:31:00", "2024-01-02 09:32:00", "2024-01-02 09:31:00"]
    query = _FakeQuery({"cpt_jyc_2026_stock_bar1m": pd.DataFrame({"date": stamps})})
    grid = fetch_trading_grid(query, {"bar1m": "cpt_jyc_2026_stock_bar1m"},
                              history_start="2024-01-02", end_date="2024-01-02")
    assert isinstance(grid, pd.DatetimeIndex)
    assert grid.tolist() == [pd.Timestamp("2024-01-02 09:31"), pd.Timestamp("2024-01-02 09:32")]


def test_fetch_trading_grid_rejects_missing_source_and_empty_grid():
    empty = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]")})
    query = _FakeQuery({"cpt_jyc_2026_stock_bar1m": empty})
    with pytest.raises(ContractError, match="datasources 缺少"):
        fetch_trading_grid(query, {}, history_start="2024-01-02", end_date="2024-01-02")
    with pytest.raises(ContractError, match="没有分钟网格"):
        fetch_trading_grid(query, {"bar1m": "cpt_jyc_2026_stock_bar1m"},
                           history_start="2024-01-02", end_date="2024-01-02")


def test_output_columns_are_the_graded_three():
    assert OUTPUT_COLUMNS == ["date", "instrument", "score"]
