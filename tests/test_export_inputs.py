"""导出脚本的契约测试：不依赖平台，用假 query 验证写出物与拒绝行为。"""

import json

import pandas as pd
import pytest

from ptop.export_inputs import EXPORTED_FILES, export_official_inputs, parse_label_spec
from ptop.local_training import _read_contract_table
from ptop.schema import ContractError
from ptop.temporal import ModelSpec

LABEL_SPEC = {
    "version": "official-vwap-30m.v1",
    "source": "cpt_jyc_2026_vwap，平台给定的未来 30 分钟 VWAP 收益列",
    "target_definition": "官方给定数值，脚本不重算公式",
    "residualization": "以官方书面口径为准",
    "boundary_policy": "目标区间由显式 support 表给出",
}

DECISIONS = ["2024-01-02 10:00:00", "2024-01-02 23:45:00", "2024-01-03 10:00:00"]
INSTRUMENTS = [("000001.SZ", 1), ("000002.SZ", 2)]


class _Frame:
    def __init__(self, frame):
        self._frame = frame

    def df(self):
        return self._frame.copy()


class _FakeQuery:
    def __init__(self, tables):
        self.tables = tables
        self.calls = []

    def __call__(self, sql, filters=None):
        self.calls.append((sql, filters))
        for name, frame in self.tables.items():
            if name in sql:
                return _Frame(frame)
        raise AssertionError(f"未预期的查询：{sql}")


def _tables():
    sections = pd.DataFrame([{"date": pd.Timestamp(day), "instrument": code, "instrument_id": ident}
                             for day in DECISIONS for code, ident in INSTRUMENTS])
    grid = pd.DataFrame({"date": pd.to_datetime(sorted({pd.Timestamp(day) for day in DECISIONS}))})
    labels = pd.DataFrame([{"date": pd.Timestamp(day), "instrument": code, "vwap_return": 0.001 * ident}
                            for day in DECISIONS for code, ident in INSTRUMENTS])
    return {"cpt_jyc_2026_instruments": sections,
            "cpt_jyc_2026_stock_bar1m": grid,
            "cpt_jyc_2026_vwap": labels}


def _export(tmp_path, **overrides):
    arguments = dict(output_dir=tmp_path / "exported", start_date="2024-01-02", end_date="2024-01-03",
                     horizon_minutes=30, label_spec=LABEL_SPEC)
    arguments.update(overrides)
    return export_official_inputs(_FakeQuery(_tables()), {"bar1m": "cpt_jyc_2026_stock_bar1m"}, **arguments)


def test_export_writes_every_declared_file(tmp_path):
    result = _export(tmp_path)
    assert sorted(path.name for path in result.files.values()) == sorted(EXPORTED_FILES)
    for path in result.files.values():
        assert path.is_file() and path.stat().st_size > 0


def test_export_bijection_is_reusable_as_a_training_input(tmp_path):
    result = _export(tmp_path)
    mapping = json.loads(result.files["instrument_map.json"].read_text(encoding="utf-8"))
    assert mapping == {"1": "000001.SZ", "2": "000002.SZ"}
    expected = pd.read_parquet(result.files["expected.parquet"])
    assert list(expected.columns) == ["sample_id", "decision_id", "instrument_id", "decision_time", "instrument"]
    assert len(expected) == len(DECISIONS) * len(INSTRUMENTS)
    assert expected["sample_id"].is_unique
    # 网格文件必须带 local_training_factory 约定的显式时间列
    grid = pd.read_parquet(result.files["grid.parquet"])
    assert "datetime" in grid
    support = pd.read_parquet(result.files["support.parquet"])
    assert set(support.columns) == {"sample_id", "target_valid", "target_start", "target_end", "label_spec_version"}
    assert result.counts["instruments"] == len(INSTRUMENTS)
    assert result.counts["expected_rows"] == len(DECISIONS) * len(INSTRUMENTS)


def test_support_marks_overnight_windows_invalid(tmp_path):
    result = _export(tmp_path)
    support = pd.read_parquet(result.files["support.parquet"]).set_index("sample_id")
    overnight = support.loc[support["target_start"].dt.strftime("%H:%M") == "23:45"]
    assert len(overnight) == len(INSTRUMENTS)
    assert not overnight["target_valid"].any(), "跨自然日的目标窗口必须标为无效（不预测隔夜收益）"
    same_day = support.loc[support["target_start"].dt.strftime("%H:%M") == "10:00"]
    assert same_day["target_valid"].all()
    assert (same_day["target_end"] - same_day["target_start"] == pd.Timedelta(minutes=30)).all()
    # targets 的 target_valid 还要与有限标签取交集
    targets = pd.read_parquet(result.files["targets.parquet"]).set_index("sample_id")
    assert not targets.loc[overnight.index, "target_valid"].any()


def test_exported_files_are_readable_by_the_training_loader(tmp_path):
    result = _export(tmp_path)
    for name in ("expected.parquet", "grid.parquet", "labels.parquet", "support.parquet", "targets.parquet"):
        frame = _read_contract_table(result.files[name], name)
        assert isinstance(frame, pd.DataFrame) and not frame.empty


def test_placeholder_label_spec_is_rejected_before_any_query(tmp_path):
    query = _FakeQuery(_tables())
    placeholder = dict(LABEL_SPEC, residualization="TBD")
    with pytest.raises(ContractError, match="占位值"):
        export_official_inputs(query, {"bar1m": "cpt_jyc_2026_stock_bar1m"}, output_dir=tmp_path / "out",
                               start_date="2024-01-02", end_date="2024-01-03",
                               horizon_minutes=30, label_spec=placeholder)
    assert query.calls == [], "占位标签合同必须在任何查询之前就被拒绝"


def test_label_spec_shape_is_enforced():
    with pytest.raises(ContractError, match="恰好包含"):
        parse_label_spec({**LABEL_SPEC, "extra": "x"})
    with pytest.raises(ContractError, match="不能为空"):
        parse_label_spec({**LABEL_SPEC, "source": "  "})


def test_non_bijective_mapping_is_rejected(tmp_path):
    tables = _tables()
    sections = tables["cpt_jyc_2026_instruments"].copy()
    sections.loc[sections["instrument"] == "000002.SZ", "instrument_id"] = 1
    tables["cpt_jyc_2026_instruments"] = sections
    with pytest.raises(ContractError, match="双射"):
        export_official_inputs(_FakeQuery(tables), {"bar1m": "cpt_jyc_2026_stock_bar1m"},
                               output_dir=tmp_path / "out", start_date="2024-01-02", end_date="2024-01-03",
                               horizon_minutes=30, label_spec=LABEL_SPEC)


def test_existing_nonempty_output_directory_is_refused(tmp_path):
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "keep.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(ContractError, match="输出目录非空"):
        _export(tmp_path, output_dir=destination)


def test_missing_datasource_key_is_reported(tmp_path):
    with pytest.raises(ContractError, match="datasources"):
        export_official_inputs(_FakeQuery(_tables()), {}, output_dir=tmp_path / "out",
                               start_date="2024-01-02", end_date="2024-01-03",
                               horizon_minutes=30, label_spec=LABEL_SPEC)


def test_exported_files_satisfy_the_official_training_factory(tmp_path):
    """导出物必须能通过正式 data_factory 的输入校验；只在读取原始行情时才允许失败。"""
    from ptop.local_training import local_training_factory
    from ptop.targets import TimeFold

    result = _export(tmp_path)
    bars = tmp_path / "empty_bars"
    bars.mkdir()
    config = {
        "scope": "official", "data_dir": str(bars), "cache_dir": str(tmp_path / "cache"),
        "instrument_map_path": str(result.files["instrument_map.json"]),
        "expected_path": str(result.files["expected.parquet"]),
        "grid_path": str(result.files["grid.parquet"]),
        "labels_path": str(result.files["labels.parquet"]),
        "support_path": str(result.files["support.parquet"]),
        "label_spec": LABEL_SPEC, "fit_window_length": 120, "seed": 7,
    }
    spec = ModelSpec.from_dict({"name": "B-GRU", "encoder": "Raw", "width": 16, "layers": 1, "lookback": 120})
    fold = TimeFold("dev", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04"))
    with pytest.raises(ContractError) as error:
        local_training_factory(config=config, model_spec=spec, fold=fold)
    message = str(error.value)
    assert "Feather" in message, f"应当只在读取原始行情时失败，实际：{message}"
