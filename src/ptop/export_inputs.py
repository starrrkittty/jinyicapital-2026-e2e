"""Export the explicit official training inputs from the platform's DAI tables.

``ptop.train`` with ``scope="official"`` needs seven declared inputs and refuses to infer
any of them from raw bars (see ``ptop.local_training.local_training_factory``).  Nothing in
the repository produced those files, so this CLI does exactly that one job, on the contest
machine where ``dai`` exists:

    instrument_map.json   {instrument_id: instrument} bijection, from the section table
    expected.parquet      the official (date, instrument) set to score, never trimmed
    grid.parquet          the platform's own bar grid, read not rebuilt
    labels.parquet        the official ``vwap_return`` values
    support.parquet       the target support interval for every expected sample
    targets.parquet       labels joined onto expected, via ``official_targets``
    manifest.json         what was queried, with which rules, and what was assumed

Two things are deliberately **not** guessed here:

* ``label_spec`` -- the label contract is supplied by the caller; obvious placeholders are
  rejected before any query runs, so an unconfirmed target definition cannot reach training.
* the support horizon -- ``--horizon-minutes`` must be stated explicitly.  A window that
  would cross the session boundary is marked invalid, which encodes the competition rule
  "不预测隔夜收益" rather than inventing a boundary.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from .official import LABEL_TABLE, SECTION_TABLE, _query_frame, _range, expected_from_sections, official_targets
from .schema import ContractError, reject_placeholder
from .submission import instrument_map_from_frame
from .targets import LabelSpec

LABEL_SPEC_KEYS = ("version", "source", "target_definition", "residualization", "boundary_policy")

EXPORTED_FILES = ("instrument_map.json", "expected.parquet", "grid.parquet", "labels.parquet",
                  "support.parquet", "targets.parquet", "manifest.json")


def parse_label_spec(payload: Mapping[str, object] | LabelSpec) -> LabelSpec:
    """Accept a ``LabelSpec`` or its JSON object, refusing unresolved placeholders."""
    if isinstance(payload, LabelSpec):
        spec = payload
    elif isinstance(payload, Mapping):
        if set(payload) != set(LABEL_SPEC_KEYS):
            raise ContractError(f"label_spec 必须恰好包含 {list(LABEL_SPEC_KEYS)}")
        spec = LabelSpec(**{key: reject_placeholder(key, payload[key]) for key in LABEL_SPEC_KEYS})
    else:
        raise ContractError("label_spec 必须为 LabelSpec 或其 JSON 对象")
    for name in LABEL_SPEC_KEYS:
        reject_placeholder(name, getattr(spec, name))
    return spec


@dataclass(frozen=True)
class ExportResult:
    output_dir: Path
    files: dict[str, Path]
    counts: dict[str, int]


def _require_table(datasources: Mapping[str, str], key: str) -> str:
    table = datasources.get(key)
    if not isinstance(table, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", table or ""):
        raise ContractError(f"datasources 缺少合法的 {key} 表名")
    return table


def export_official_inputs(
    query: Callable,
    datasources: Mapping[str, str],
    *,
    output_dir: str | Path,
    start_date,
    end_date,
    horizon_minutes: int,
    label_spec: Mapping[str, object] | LabelSpec,
    grid_key: str = "bar1m",
    section_table: str = SECTION_TABLE,
    label_table: str = LABEL_TABLE,
    write_targets: bool = True,
) -> ExportResult:
    """Write the seven declared training inputs for one explicit development range."""
    spec = parse_label_spec(label_spec)
    if type(horizon_minutes) is not int or horizon_minutes <= 0:
        raise ContractError("horizon_minutes 必须为正整数（预测窗口长度，需与官方口径一致）")
    start, end = _range(start_date, end_date)
    table = _require_table(datasources, grid_key)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ContractError(f"输出目录非空，请换新目录以免覆盖既有导出：{destination}")
    destination.mkdir(parents=True, exist_ok=True)
    window = {"date": [start.isoformat(sep=" "), end.isoformat(sep=" ")]}

    sections = _query_frame(query, f"SELECT date, instrument, instrument_id FROM {section_table} "
                                   f"ORDER BY date, instrument_id", dict(window), start=start, end=end)
    mapping_frame = sections[["instrument_id", "instrument"]].drop_duplicates()
    if mapping_frame.duplicated("instrument_id").any() or mapping_frame.duplicated("instrument").any():
        raise ContractError("截面表的 instrument_id 与 instrument 不构成双射，拒绝导出模糊映射")
    mapping = instrument_map_from_frame(mapping_frame)
    expected = expected_from_sections(sections, mapping)
    if expected.empty:
        raise ContractError("声明区间内官方应预测集合为空")

    grid_frame = _query_frame(query, f"SELECT DISTINCT date FROM {table} ORDER BY date",
                              dict(window), start=start, end=end)
    grid = pd.DatetimeIndex(pd.to_datetime(grid_frame["date"], errors="raise")).unique().sort_values()
    if grid.empty or not grid.equals(grid.floor("min")):
        raise ContractError(f"{grid_key} 未给出可用的分钟网格")
    # local_training_factory 默认按 grid_column="datetime" 读取，这里直接写成该列名，免去额外配置。
    grid_out = pd.DataFrame({"datetime": grid})

    labels = _query_frame(query, f"SELECT date, instrument, vwap_return FROM {label_table} "
                                 f"ORDER BY date, instrument", dict(window), start=start, end=end)

    support = expected[["sample_id", "decision_time"]].copy()
    support["target_start"] = support["decision_time"]
    support["target_end"] = support["decision_time"] + pd.Timedelta(minutes=horizon_minutes)
    # 不预测隔夜收益：窗口跨自然日的一律标为无效，而不是事后推断边界。
    support["target_valid"] = support["target_end"].dt.normalize().eq(support["decision_time"].dt.normalize())
    support["label_spec_version"] = spec.version
    support = support[["sample_id", "target_valid", "target_start", "target_end", "label_spec_version"]]
    support["target_valid"] = support["target_valid"].astype(bool)

    files = {}
    files["instrument_map.json"] = destination / "instrument_map.json"
    files["instrument_map.json"].write_text(
        json.dumps({str(key): value for key, value in sorted(mapping.items())}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    for name, frame in (("expected.parquet", expected), ("grid.parquet", grid_out),
                        ("labels.parquet", labels), ("support.parquet", support)):
        path = destination / name
        frame.to_parquet(path, index=False)
        files[name] = path

    targets = official_targets(expected, labels, support, spec)
    if write_targets:
        path = destination / "targets.parquet"
        targets.to_parquet(path, index=False)
        files["targets.parquet"] = path

    manifest = {
        "source": "ptop.export_inputs",
        "range": {"start_date": start.isoformat(), "end_date": end.isoformat()},
        "tables": {"grid": table, "grid_key": grid_key, "section": section_table, "label": label_table},
        "label_spec": {key: getattr(spec, key) for key in LABEL_SPEC_KEYS},
        "rules": {
            "support_horizon_minutes": horizon_minutes,
            "target_start_policy": "decision_time",
            "target_end_policy": "decision_time + horizon_minutes",
            "overnight_policy": "跨自然日的窗口标为 target_valid=False（不预测隔夜收益）",
            "universe_policy": "完全采用截面表给出的应预测集合，不按行情记录删减股票或时点",
        },
        "counts": {"instruments": len(mapping), "expected_rows": int(len(expected)),
                   "samples": int(expected["sample_id"].nunique()), "grid_rows": int(len(grid_frame)),
                   "grid_days": int(grid.normalize().nunique()), "label_rows": int(len(labels)),
                   "target_rows": int(len(targets)),
                   "target_valid_rows": int(targets["target_valid"].sum())},
        "files": sorted(files),
        "assumptions_to_confirm": [
            "预测窗口的长度与官方 vwap_return 定义完全一致（本文件按 horizon_minutes 记；请以官方书面口径复核）",
            "LabelSpec.residualization 已由官方确认，非占位值",
        ],
    }
    files["manifest.json"] = destination / "manifest.json"
    files["manifest.json"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
                                      encoding="utf-8")
    return ExportResult(output_dir=destination, files=dict(sorted(files.items())), counts=manifest["counts"])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="在竞赛机上从 DAI 平台导出 ptop.train 所需的官方训练输入。")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasources", required=True,
                        help='平台注入的表名 JSON，例如 \'{"bar1m": "cpt_jyc_2026_stock_bar1m"}\'')
    parser.add_argument("--start-date", required=True,
                        help="导出区间起点；必须覆盖训练折的 train_start")
    parser.add_argument("--end-date", required=True,
                        help="导出区间终点；必须覆盖训练折的 validation_end，否则验证集会为空")
    parser.add_argument("--horizon-minutes", type=int, default=30)
    parser.add_argument("--grid-key", default="bar1m")
    parser.add_argument("--label-spec", required=True,
                        help='标签合同 JSON，例如 \'{"version": "...", "source": "...", '
                             '"target_definition": "...", "residualization": "...", "boundary_policy": "..."}\'')
    parser.add_argument("--section-table", default=SECTION_TABLE)
    parser.add_argument("--label-table", default=LABEL_TABLE)
    arguments = parser.parse_args(argv)
    try:
        import dai  # 平台自带；本机没有，因此导入失败要给出明确提示
    except ImportError as error:
        raise SystemExit("本脚本只能在竞赛机上运行（需要平台自带的 dai 数据引擎）。") from error
    try:
        result = export_official_inputs(
            dai.query, json.loads(arguments.datasources), output_dir=arguments.output_dir,
            start_date=arguments.start_date, end_date=arguments.end_date,
            horizon_minutes=arguments.horizon_minutes, label_spec=json.loads(arguments.label_spec),
            grid_key=arguments.grid_key, section_table=arguments.section_table,
            label_table=arguments.label_table)
    except (ContractError, ValueError, OSError) as error:
        raise SystemExit(f"导出失败：{error}") from error
    print(json.dumps({"output_dir": str(result.output_dir),
                      "files": {name: str(path) for name, path in result.files.items()},
                      "counts": result.counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
