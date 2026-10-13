"""Bounded local raw-bar partitions feeding A's public window contract.

The production entry requires official sampling keys, symbol mapping and labels.
The separate diagnostic entry uses real bars with explicitly artificial targets.
Only raw daily Arrow files are cached; learned encoder outputs are never cached.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
import errno
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

from .data import CanonicalData, _timestamps, _validate_mapping, normalize_bars, normalize_local_bars
from .official import official_targets
from .schema import BOOK_ORDER_FIELDS, FEATURES, QLIB_COMMIT, ContractError
from .targets import LabelSpec, TimeFold, attach_targets
if TYPE_CHECKING:
    from .preprocessing import FrozenPreprocessor
    from .windows import WindowBatch


CACHE_VERSION = 1
DIAGNOSTIC_SCOPE = "diagnostic"
DIAGNOSTIC_DATA_SCOPE = "diagnostic_real_bars_synthetic_targets"
_SAMPLE_COLUMNS = ["sample_id", "decision_id", "instrument_id", "decision_time"]


def _json_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _days(values: Iterable) -> pd.DatetimeIndex:
    result = _timestamps(pd.Index(list(values)), "partition days")
    if result.empty or not result.equals(result.normalize()):
        raise ContractError("分区日期必须为非空的北京时间午夜日期")
    return result.unique().sort_values()


def _ids(values: Iterable[int]) -> tuple[int, ...]:
    values = list(values)
    if not values or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in values):
        raise ContractError("必须显式提供非空整数 instrument_id 集合")
    return tuple(sorted({int(x) for x in values}))


@dataclass(frozen=True)
class ReadPolicy:
    """The signed-count anomaly policy is opt-in and recorded in provenance."""

    negative_order_policy: Literal["reject", "mask_known"] = "reject"
    max_source_bytes: int = 2_000_000_000
    max_record_batch_rows: int = 262_144
    max_day_rows: int = 2_000_000

    def __post_init__(self):
        if self.negative_order_policy not in {"reject", "mask_known"}:
            raise ContractError("negative_order_policy 必须为 reject 或 mask_known")
        for name in ("max_source_bytes", "max_record_batch_rows", "max_day_rows"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ContractError(f"{name} 必须为正整数")


class LocalBarStore:
    """Stream monthly IPC once into a private cache bounded by day and batch.

    ``materialize`` declares the complete requested days and IDs.  Reads verify
    the resulting cache hashes and call A's normalizer; no symbol is inferred.
    The caller chooses a private ``cache_dir`` outside the source data directory.
    """

    def __init__(self, data_dir: str | Path, *, cache_dir: str | Path, policy: ReadPolicy | None = None):
        self.data_dir, self.cache_dir = Path(data_dir).resolve(), Path(cache_dir).resolve()
        if not self.data_dir.is_dir():
            raise ContractError(f"行情目录不存在：{self.data_dir}")
        if self.cache_dir == self.data_dir or self.data_dir in self.cache_dir.parents:
            raise ContractError("raw cache 必须写到独立私人目录，不能修改原始下载目录")
        self.policy = policy or ReadPolicy()
        self.manifest: dict | None = None
        self._scope_dir: Path | None = None
        self._verified_days: set[str] = set()
        self.anomaly_counts_by_day: dict[str, dict[str, int]] = {}
        self.availability_origin_by_day: dict[str, str] = {}
        self.profile = {"read_calls": 0, "read_seconds": 0.0, "rows_read": 0, "max_day_rows_read": 0,
                        "masked_negative_orders": {name: 0 for name in BOOK_ORDER_FIELDS}}

    def materialize(self, days: Iterable, instrument_ids: Iterable[int]) -> dict:
        selected_days, instruments = _days(days), _ids(instrument_ids)
        sources = []
        for month in selected_days.strftime("%Y%m").unique():
            paths = sorted(path for path in self.data_dir.glob(f"{month}.*.feather")
                           if re.fullmatch(r"\d{6}\.\d+\.feather", path.name))
            if not paths:
                raise ContractError(f"缺少请求月份的 Feather：{self.data_dir}/{month}.*.feather")
            for path in paths:
                size = path.stat().st_size
                if size > self.policy.max_source_bytes:
                    raise ContractError(f"月份文件超过 max_source_bytes：{path.name} ({size} bytes)")
                sources.append({"path": str(path), "size_bytes": size, "sha256": _file_digest(path)})
        identity = {"cache_version": CACHE_VERSION, "features": list(FEATURES), "source": sources,
                    "days": selected_days.strftime("%Y-%m-%d").tolist(), "instrument_ids": list(instruments)}
        scope_dir = self.cache_dir / _json_digest(identity)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not scope_dir.exists():
            staging = Path(tempfile.mkdtemp(prefix="raw-days-", dir=self.cache_dir))
            try:
                manifest = self._stream_to_days(staging, identity, selected_days, instruments)
                _write_json(staging / "manifest.json", manifest)
                # A concurrent identical run may already have completed this cache.
                try:
                    staging.rename(scope_dir)
                except OSError as exc:
                    if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not scope_dir.is_dir():
                        raise
                    shutil.rmtree(staging)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        manifest = json.loads((scope_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise ContractError("私人 raw cache 身份与请求数据不一致")
        if set(manifest.get("partitions", {})) != set(identity["days"]):
            raise ContractError("私人 raw cache 分区清单不完整")
        if self.manifest is None or self.manifest["identity"] != identity:
            self.anomaly_counts_by_day = {}
            self.availability_origin_by_day = {}
        self.manifest, self._scope_dir = manifest, scope_dir
        self._verified_days.clear()
        return manifest

    def _stream_to_days(self, staging: Path, identity: dict, days: pd.DatetimeIndex,
                        instruments: tuple[int, ...]) -> dict:
        partitions, readers_seconds = {}, 0.0
        started = time.perf_counter()
        for month in days.strftime("%Y%m").unique():
            month_days = days[days.strftime("%Y%m") == month]
            writers: dict[str, ipc.RecordBatchFileWriter] = {}
            sinks: dict[str, pa.OSFile] = {}
            counts = {day.strftime("%Y-%m-%d"): 0 for day in month_days}
            schema = None
            try:
                for source in identity["source"]:
                    path = Path(source["path"])
                    if not path.name.startswith(month):
                        continue
                    read_start = time.perf_counter()
                    with pa.memory_map(str(path), "r") as file:
                        reader = ipc.open_file(file)
                        names = reader.schema.names
                        required = ("date", "instrument_id") + FEATURES
                        missing = set(required).difference(names)
                        if missing:
                            raise ContractError(f"{path.name} 缺少 raw25 字段：{sorted(missing)}")
                        wanted = list(required) + [x for x in ("instrument", "adjust_factor", "available_at") if x in names]
                        selected_schema = pa.schema([reader.schema.field(x) for x in wanted])
                        if schema is not None and selected_schema != schema:
                            raise ContractError("月份分区 raw Arrow schema 不一致")
                        schema = selected_schema
                        if not writers:
                            for key in counts:
                                sink = pa.OSFile(str(staging / f"{key}.arrow"), "wb")
                                sinks[key], writers[key] = sink, ipc.new_file(sink, schema)
                        for index in range(reader.num_record_batches):
                            batch = reader.get_batch(index)
                            if batch.num_rows > self.policy.max_record_batch_rows:
                                raise ContractError(f"Arrow batch 超过 max_record_batch_rows：{path.name}, batch={index}")
                            table = pa.Table.from_batches([batch.select(wanted)])
                            ids = table["instrument_id"]
                            dates = table["date"]
                            if not pa.types.is_timestamp(dates.type):
                                raise ContractError("原始 date 必须声明 Arrow timestamp 单位")
                            if dates.type.tz is not None:
                                # A later confirms Asia/Shanghai timestamps; daily cache uses that calendar.
                                dates = pc.local_timestamp(pc.cast(dates, pa.timestamp("ns", "Asia/Shanghai")))
                            else:
                                dates = pc.cast(dates, pa.timestamp("ns"))
                            day_values = pc.floor_temporal(dates, unit="day")
                            selected = pc.and_(pc.is_in(ids, value_set=pa.array(instruments, type=ids.type)),
                                               pc.is_in(day_values, value_set=pa.array(month_days.to_numpy(), type=pa.timestamp("ns"))))
                            table = table.filter(selected)
                            day_values = day_values.filter(selected)
                            for day in month_days:
                                key = day.strftime("%Y-%m-%d")
                                owned = table.filter(pc.equal(day_values, pa.scalar(day.to_datetime64(), type=pa.timestamp("ns"))))
                                counts[key] += owned.num_rows
                                if counts[key] > self.policy.max_day_rows:
                                    raise ContractError(f"{key} 超过 max_day_rows；请缩小股票集或显式调整预算")
                                if owned.num_rows:
                                    writers[key].write_table(owned)
                    readers_seconds += time.perf_counter() - read_start
                    if _file_digest(path) != source["sha256"]:
                        raise ContractError(f"读取期间来源文件发生变化：{path.name}")
            finally:
                for writer in writers.values():
                    writer.close()
                for sink in sinks.values():
                    sink.close()
            for key, count in counts.items():
                file = staging / f"{key}.arrow"
                partitions[key] = {"file": file.name, "rows": count, "sha256": _file_digest(file),
                                   "size_bytes": file.stat().st_size}
        return {"identity": identity, "partitions": partitions, "cache_kind": "raw_only_no_learned_tokens",
                "materialize_seconds": time.perf_counter() - started, "source_read_seconds": readers_seconds}

    def read_day(self, day, *, instrument_map: Mapping[int, str] | None = None,
                 bar_end_is_available: bool = False) -> CanonicalData:
        key = pd.Timestamp(day).strftime("%Y-%m-%d")
        if self.manifest is None or self._scope_dir is None or key not in self.manifest["partitions"]:
            raise ContractError("read_day 前必须 materialize 请求日期和股票集合")
        started = time.perf_counter()
        entry = self.manifest["partitions"][key]
        path = self._scope_dir / entry["file"]
        if _file_digest(path) != entry["sha256"]:
            raise ContractError(f"私人 raw cache 校验失败：{key}")
        self._verified_days.add(key)
        with pa.memory_map(str(path), "r") as file:
            frame = ipc.open_file(file).read_all().to_pandas()
        if len(frame) != entry["rows"] or len(frame) > self.policy.max_day_rows:
            raise ContractError("私人 raw cache 行数与预算或清单不一致")
        counts = {}
        for name in BOOK_ORDER_FIELDS:
            invalid = frame[name].lt(0)
            count = int(invalid.sum())
            counts[name] = count
            if self.policy.negative_order_policy == "mask_known":
                self.profile["masked_negative_orders"][name] += count
                if count:
                    frame[name] = frame[name].astype(np.float64).mask(invalid)
        self.anomaly_counts_by_day[key] = counts
        normalizer = normalize_local_bars if instrument_map is None else normalize_bars
        kwargs = {"bar_end_is_available": bar_end_is_available}
        if instrument_map is not None:
            kwargs.update(source="local", instrument_map=instrument_map)
        raw = normalizer(frame, **kwargs)
        origin = ("source_available_at" if "available_at" in frame else
                  "assumed_bar_end" if bar_end_is_available else "undeclared_availability")
        if raw.metadata is not None:
            raw.metadata.attrs["availability_origin"] = origin
        self.availability_origin_by_day[key] = origin
        self.profile["read_calls"] += 1
        self.profile["read_seconds"] += time.perf_counter() - started
        self.profile["rows_read"] += len(raw.values)
        self.profile["max_day_rows_read"] = max(self.profile["max_day_rows_read"], len(raw.values))
        return raw


def split_local_targets(expected: pd.DataFrame, targets: pd.DataFrame, label_spec: LabelSpec,
                        fold: TimeFold) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep full sampling keys and mark crossing supports invalid, as in A."""
    labeled = attach_targets(expected, targets, label_spec)
    times = _timestamps(labeled["decision_time"], "decision_time")
    outputs, audits = [], []
    for name, start, end in (("train", fold.train_start, fold.validation_start),
                             ("validation", fold.validation_start, fold.validation_end)):
        table = labeled.loc[(times >= start) & (times < end)].copy()
        if table.empty:
            raise ContractError(f"{name} 区间必须有显式完整采样集合")
        original = table["target_valid"].copy()
        reaches = original & table["target_end"].ge(end)
        table["target_valid"] = original & ~reaches
        table["target"] = table["target"].where(table["target_valid"], np.nan)
        table["supervision_exclusion"] = np.select([~original, reaches],
            ["invalid_target", "target_reaches_split_boundary"], default="")
        outputs.append(table)
        audits.append(table[["sample_id", "decision_id", "decision_time", "target_valid", "supervision_exclusion"]].assign(split=name))
    return (*outputs, pd.concat(audits, ignore_index=True))


def _window_dataset(raw: CanonicalData, expected: pd.DataFrame, grid: pd.DatetimeIndex, *,
                    length: int, fields: Sequence[str], scope: str, bar_end_is_available: bool):
    from .windows import LocalAuditWindowDataset, WindowDataset

    if raw.metadata is not None and raw.metadata.attrs.get("availability_origin") == "assumed_bar_end":
        # A's normalizer materializes available_at when the caller declares bar
        # end availability. A's window constructor must see the explicit flag,
        # so it does not misreport this generated column as source-provided.
        raw = CanonicalData(raw.values, raw.observed, raw.metadata.drop(columns="available_at"),
                            dict(raw.instrument_map), raw.duplicate_rows)
    cls = WindowDataset if scope == "official" else LocalAuditWindowDataset
    return cls(raw, expected, grid, length=length, fields=fields, bar_end_is_available=bar_end_is_available)


class _ColumnReader:
    """Read a sequential reduction with a bounded buffer, not a full mmap."""

    def __init__(self, path: Path, size: int):
        self.file = path.open("rb")
        self.size, self.begin = size, -1
        self.buffer = np.empty(0, dtype=np.float64)

    def read(self, begin: int, size: int) -> np.ndarray:
        if begin < self.begin or begin + size > self.begin + len(self.buffer):
            self.file.seek(begin * 8)
            self.buffer = np.frombuffer(self.file.read(min(16384, self.size - begin) * 8), dtype=np.float64)
            self.begin = begin
        return self.buffer[begin - self.begin:begin - self.begin + size]

    def close(self) -> None:
        self.file.close()


def _native_pairwise_sum(reader: _ColumnReader, begin: int, size: int, *, squared_center: float | None = None) -> np.float64:
    """Match NumPy 2.2.6 fast-axis reduction using bounded native leaves.

    The split boundary comes from NumPy's published pairwise algorithm; each
    leaf's actual arithmetic stays in NumPy. Missing values were written as
    zero. This helper is only needed for otherwise constant fields whose tiny
    floating roundoff can change Qlib's exact ``std == 0`` ignore flag.
    """
    if size <= 128:
        values = reader.read(begin, size)
        if squared_center is not None:
            values = np.where(values != 0, (values - squared_center) ** 2, 0.0)
        return np.sum(values, dtype=np.float64)
    half = size // 2
    half -= half % 8
    return (_native_pairwise_sum(reader, begin, half, squared_center=squared_center) +
            _native_pairwise_sum(reader, begin + half, size - half, squared_center=squared_center))


def _pinned_pairwise_sum(reader: _ColumnReader, begin: int, size: int, *, squared_center: float | None = None) -> np.float64:
    # NumPy's reduction iterator calls the native pairwise kernel per buffer;
    # reproducing only the recursive kernel changes rounding after 8192 rows.
    result = np.float64(0.0)
    block_size = np.getbufsize()
    for offset in range(begin, begin + size, block_size):
        result += _native_pairwise_sum(reader, offset, min(block_size, begin + size - offset), squared_center=squared_center)
    return result


def _constant_column_statistics(store, train_expected, trading_grid, *, columns, first, count, fit_count,
                                fold, window_length, fields, instrument_map, scope, bar_end_is_available):
    """Preserve pinned Qlib roundoff/ignore even for decimal constants."""
    from .windows import fit_row_union

    output, missing_columns = {}, []
    for column in columns:
        if first[column] == 0:
            output[column] = (0.0, 0.0)
        elif count[column] == fit_count:
            view = np.broadcast_to(np.asarray(first[column], dtype=np.float64), (fit_count,))
            center = float(np.mean(view))
            variance_view = np.broadcast_to(np.asarray((first[column] - center) ** 2), (fit_count,))
            output[column] = (center, float(np.sqrt(np.mean(variance_view))))
        else:
            missing_columns.append(column)
    if not missing_columns:
        return output
    if np.__version__ != "2.2.6":
        raise ContractError("带缺失的常量字段精确 N1 归约需锁定 NumPy 2.2.6；不能静默改变 Qlib ignore 语义")
    with tempfile.TemporaryDirectory(prefix="n1-constant-columns-", dir=store.cache_dir) as directory:
        paths = {column: Path(directory) / f"{column}.f64" for column in missing_columns}
        streams = {column: path.open("wb") for column, path in paths.items()}
        fit_start = fold.train_start.normalize()
        fit_end = (fold.validation_start - pd.Timedelta(nanoseconds=1)).normalize()
        try:
            for day, samples in train_expected.groupby(train_expected.decision_time.dt.normalize(), sort=True):
                raw = store.read_day(day, instrument_map=instrument_map, bar_end_is_available=bar_end_is_available)
                windows = _window_dataset(raw, samples, trading_grid, length=window_length, fields=fields, scope=scope,
                                          bar_end_is_available=bar_end_is_available)
                references = fit_row_union(windows, fit_start=fit_start, fit_end=fit_end)
                for column in missing_columns:
                    values = raw.values.loc[references, FEATURES[column]].to_numpy(dtype=np.float64)
                    streams[column].write(np.nan_to_num(values, nan=0.0).tobytes())
        finally:
            for stream in streams.values():
                stream.close()
        for column, path in paths.items():
            if path.stat().st_size != fit_count * 8:
                raise ContractError("常量列归约引用行数不一致")
            reader = _ColumnReader(path, fit_count)
            try:
                center = float(_pinned_pairwise_sum(reader, 0, fit_count) / count[column])
                std = float(np.sqrt(_pinned_pairwise_sum(reader, 0, fit_count, squared_center=center) / count[column]))
                output[column] = (center, std)
            finally:
                reader.close()
    return output


def fit_partitioned_n1(store: LocalBarStore, train_expected: pd.DataFrame, trading_grid: pd.DatetimeIndex,
                       *, fold: TimeFold, window_length: int, fields: Sequence[str] = FEATURES,
                       instrument_map: Mapping[int, str] | None = None, scope: str = "official",
                       bar_end_is_available: bool = False, scale_review: str) -> tuple[FrozenPreprocessor, dict]:
    """Stable pooled population moments, equivalent to pinned Qlib N1.

    Each unique visible row is included once, in sorted day/key order. Missing
    labels and model-field exclusions do not select the normalization rows.
    No complete multi-year DataFrame is constructed. This is a streaming
    implementation, not a claim that A's full-table ``fit`` was called.
    """
    from .preprocessing import FrozenPreprocessor
    from .windows import fit_row_union

    if not isinstance(scale_review, str) or not scale_review.strip():
        raise ContractError("必须提供真实 scale_review，自动数值检查不能冒称人工审阅")
    count = np.zeros(len(FEATURES), dtype=np.int64)
    mean, m2 = np.zeros(len(FEATURES)), np.zeros(len(FEATURES))
    first, minima, maxima = np.full(len(FEATURES), np.nan), np.full(len(FEATURES), np.inf), np.full(len(FEATURES), -np.inf)
    row_hash, fit_count = hashlib.sha256(), 0
    fit_start, fit_end = fold.train_start.normalize(), (fold.validation_start - pd.Timedelta(nanoseconds=1)).normalize()
    for day, samples in train_expected.groupby(train_expected.decision_time.dt.normalize(), sort=True):
        raw = store.read_day(day, instrument_map=instrument_map, bar_end_is_available=bar_end_is_available)
        windows = _window_dataset(raw, samples, trading_grid, length=window_length, fields=fields, scope=scope,
                                  bar_end_is_available=bar_end_is_available)
        referenced = fit_row_union(windows, fit_start=fit_start, fit_end=fit_end)
        row_hash.update(pd.util.hash_pandas_object(referenced).to_numpy().tobytes())
        fit_count += len(referenced)
        values = raw.values.loc[referenced].to_numpy(dtype=np.float64)
        for column in range(len(FEATURES)):
            valid = values[np.isfinite(values[:, column]), column]
            added = len(valid)
            if not added:
                continue
            part_mean = float(np.mean(valid))
            part_m2 = float(np.sum((valid - part_mean) ** 2))
            total = int(count[column]) + added
            delta = part_mean - mean[column]
            m2[column] += part_m2 + delta * delta * int(count[column]) * added / total
            mean[column] += delta * added / total
            if count[column] == 0:
                first[column] = valid[0]
            count[column] = total
            minima[column], maxima[column] = min(minima[column], float(valid.min())), max(maxima[column], float(valid.max()))
    if fit_count == 0 or (count == 0).any():
        empty = [name for name, n in zip(FEATURES, count) if n == 0]
        raise ContractError(f"训练可见引用行没有完整 N1 拟合观测：{empty}")
    scale = np.sqrt(m2 / count)
    constant_columns = np.flatnonzero(minima == maxima).tolist()
    exact_constants = _constant_column_statistics(store, train_expected, trading_grid, columns=constant_columns,
        first=first, count=count, fit_count=fit_count, fold=fold, window_length=window_length, fields=fields,
        instrument_map=instrument_map, scope=scope, bar_end_is_available=bar_end_is_available)
    for column, (center, std) in exact_constants.items():
        mean[column], scale[column] = center, std
    ignored = scale == 0
    constants = {name: float(value) for name, value, ignore in zip(FEATURES, first, ignored) if ignore}
    mean[ignored], scale[ignored] = 0.0, 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ContractError("流式 N1 产生无效中心或尺度")
    payload = {"state_version": 1, "qlib_commit": QLIB_COMMIT, "fields": list(FEATURES), "mode": "N1", "base": "N1",
               "axis": "per_field", "clip_outlier": False, "fit_start": fit_start.isoformat(),
               "fit_end": (fit_end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)).isoformat(),
               "fit_row_count": fit_count, "fit_rows_fingerprint": row_hash.hexdigest(), "valid_counts": count.tolist(),
               "training_constants": constants, "scale_review": scale_review,
               "statistics": {"center": mean.tolist(), "scale": scale.tolist(), "ignore": ignored.tolist()}}
    processor = FrozenPreprocessor.from_state_dict({**payload, "fingerprint": _json_digest(payload)})
    diagnostics = {"implementation": "pooled_float64_population_moments.v1", "reference": f"Qlib ZScoreNorm {QLIB_COMMIT}",
                   "fit_row_count": fit_count, "fit_rows_fingerprint": row_hash.hexdigest(), "fit_window_length": window_length,
                   "fields": {name: {"valid_count": int(n), "raw_minimum": float(low), "raw_maximum": float(high),
                                      "center": float(center), "scale": float(std), "zero_variance_training": bool(ignore)}
                              for name, n, low, high, center, std, ignore in zip(FEATURES, count, minima, maxima, mean, scale, ignored)},
                   "scale_review": scale_review, "scope": scope, "automatic_float32_finite_check": False}
    diagnostics["constant_field_exact_reduction"] = [FEATURES[column] for column in constant_columns]
    diagnostics["constant_reduction_reference"] = "NumPy 2.2.6 fast-axis pairwise sums; bounded 16384-element buffer"
    diagnostics["constant_reduction_numpy_iterator_buffer_size"] = np.getbufsize()
    # Check A's exact frozen transform on every training partition, without gathering days.
    for day, samples in train_expected.groupby(train_expected.decision_time.dt.normalize(), sort=True):
        raw = store.read_day(day, instrument_map=instrument_map, bar_end_is_available=bar_end_is_available)
        processor.transform(raw.values, raw.observed)
    diagnostics["automatic_float32_finite_check"] = True
    return processor, diagnostics


class PartitionedWindowLoader:
    """Reiterable full-decision batches, holding one raw day at a time."""

    def __init__(self, prepared: PreparedLocalFold, *, split: str, epoch: int = 0):
        self.prepared, self.split, self.epoch = prepared, split, epoch
        self.profile = {"partition_wait_seconds": 0.0, "partitions": 0, "batches": 0, "samples": 0}

    def __len__(self) -> int:
        return self._expected.decision_id.nunique()

    @property
    def _expected(self) -> pd.DataFrame:
        return self.prepared.train_expected if self.split == "train" else self.prepared.validation_expected

    def __iter__(self) -> Iterator[WindowBatch]:
        prepared = self.prepared
        rng = np.random.default_rng(prepared.seed + self.epoch)
        fingerprint = prepared.processor.fingerprint
        for day, expected in self._expected.groupby(self._expected.decision_time.dt.normalize(), sort=True):
            started = time.perf_counter()
            raw = prepared.store.read_day(day, instrument_map=prepared.instrument_map,
                                          bar_end_is_available=prepared.bar_end_is_available)
            values = prepared.processor.transform(raw.values, raw.observed)
            processed = CanonicalData(values, raw.observed, raw.metadata, dict(raw.instrument_map))
            dataset = _window_dataset(processed, expected, prepared.trading_grid, length=prepared.window_length,
                                      fields=prepared.fields, scope=prepared.scope, bar_end_is_available=prepared.bar_end_is_available)
            groups = list(dataset.expected.groupby("decision_id", sort=False).indices.values())
            order = rng.permutation(len(groups)) if self.split == "train" else range(len(groups))
            self.profile["partition_wait_seconds"] += time.perf_counter() - started
            self.profile["partitions"] += 1
            for index in order:
                batch = dataset.get_batch(groups[int(index)])
                self.profile["batches"] += 1
                self.profile["samples"] += len(batch.metadata)
                yield batch
            if prepared.processor.fingerprint != fingerprint:
                raise ContractError("分区转换改变冻结 N1 状态")
            del dataset, processed, raw, values


@dataclass
class PreparedLocalFold:
    processor: FrozenPreprocessor
    store: LocalBarStore
    train_expected: pd.DataFrame
    validation_expected: pd.DataFrame
    train_targets: pd.DataFrame
    validation_targets: pd.DataFrame
    split_audit: pd.DataFrame
    trading_grid: pd.DatetimeIndex
    fold: TimeFold
    label_spec: LabelSpec
    scope: str
    window_length: int
    fit_window_length: int
    fields: tuple[str, ...]
    seed: int
    bar_end_is_available: bool
    instrument_map: dict[int, str] | None
    normalization_diagnostics: dict
    manifest: dict = field(default_factory=dict)

    def train_loader(self, epoch: int = 0) -> PartitionedWindowLoader:
        if type(epoch) is not int or epoch < 0:
            raise ContractError("epoch 必须为非负整数")
        return PartitionedWindowLoader(self, split="train", epoch=epoch)

    def validation_loader(self) -> PartitionedWindowLoader:
        return PartitionedWindowLoader(self, split="validation")

    def with_window(self, length: int, *, fields: Sequence[str] | None = None, seed: int | None = None) -> PreparedLocalFold:
        if type(length) is not int or not 0 < length <= self.fit_window_length:
            raise ContractError("比较窗口须为正整数，且不能超过共同 N1 的 fit_window_length")
        selected = self.fields if fields is None else tuple(fields)
        if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(FEATURES):
            raise ContractError("模型字段必须为 raw25 非空无重复子集")
        selected_seed = self.seed if seed is None else seed
        if type(selected_seed) is not int or not 0 <= selected_seed < 2**32:
            raise ContractError("seed 必须为 [0, 2**32) 范围内整数，与 TrainingConfig 一致")
        return replace(self, window_length=length, fields=tuple(selected), seed=selected_seed,
                       manifest={**self.manifest, "window_length": length, "fields": list(selected), "seed": selected_seed})


def _prepare(store: LocalBarStore, expected: pd.DataFrame, targets: pd.DataFrame, *, label_spec: LabelSpec,
             trading_grid: pd.DatetimeIndex, fold: TimeFold, instrument_map: Mapping[int, str] | None,
             scope: str, window_length: int, fit_window_length: int | None, fields: Sequence[str], seed: int,
             bar_end_is_available: bool, scale_review: str) -> PreparedLocalFold:
    if not isinstance(fold, TimeFold) or any(value != value.normalize() for value in
            (fold.train_start, fold.validation_start, fold.validation_end)):
        raise ContractError("本地训练按完整日分区，必须提供午夜边界 TimeFold")
    if type(bar_end_is_available) is not bool:
        raise ContractError("bar_end_is_available 必须为 bool")
    if not isinstance(expected, pd.DataFrame) or not expected.columns.is_unique or not set(_SAMPLE_COLUMNS).issubset(expected.columns):
        raise ContractError(f"必须显式提供采样键表：{_SAMPLE_COLUMNS}")
    expected = expected.copy()
    expected["decision_time"] = _timestamps(expected.decision_time, "decision_time")
    if expected[_SAMPLE_COLUMNS].isna().any().any() or not expected.sample_id.is_unique:
        raise ContractError("采样键必须完整，sample_id 必须唯一")
    if not pd.api.types.is_integer_dtype(expected.instrument_id):
        raise ContractError("采样 instrument_id 必须为整数")
    if expected.duplicated(["decision_time", "instrument_id"]).any():
        raise ContractError("采样时间和股票键存在重复")
    if (expected.groupby("decision_time").decision_id.nunique() != 1).any() or (
            expected.groupby("decision_id").decision_time.nunique() != 1).any():
        raise ContractError("decision_id 必须与决策时点一一对应")
    if not isinstance(trading_grid, pd.DatetimeIndex) or trading_grid.empty or not trading_grid.is_unique or not trading_grid.is_monotonic_increasing:
        raise ContractError("必须提供非空、唯一、排序的显式分钟网格")
    trading_grid = _timestamps(trading_grid, "trading_grid")
    if not expected.decision_time.isin(trading_grid).all():
        raise ContractError("全部决策端点必须在显式分钟网格内")
    if not fields or len(set(fields)) != len(fields) or not set(fields).issubset(FEATURES):
        raise ContractError("模型字段必须为 raw25 非空无重复子集")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ContractError("seed 必须为 [0, 2**32) 范围内整数，与 TrainingConfig 一致")
    train, validation, audit = split_local_targets(expected, targets, label_spec, fold)
    columns = _SAMPLE_COLUMNS + (["instrument"] if scope == "official" else [])
    train_expected, validation_expected = train[columns].copy(), validation[columns].copy()
    selected_days = pd.DatetimeIndex(pd.concat([train.decision_time, validation.decision_time]).dt.normalize().unique()).sort_values()
    selected_ids = _ids(pd.concat([train.instrument_id, validation.instrument_id]))
    store.materialize(selected_days, selected_ids)
    fit_length = window_length if fit_window_length is None else fit_window_length
    if type(window_length) is not int or type(fit_length) is not int or not 0 < window_length <= fit_length:
        raise ContractError("window_length 和 fit_window_length 必须为正整数，拟合窗口不能更短")
    processor, diagnostics = fit_partitioned_n1(store, train_expected, trading_grid, fold=fold, window_length=fit_length,
        fields=fields, instrument_map=instrument_map, scope=scope, bar_end_is_available=bar_end_is_available,
        scale_review=scale_review)
    keys = {"train": train_expected.sample_id.astype(str).tolist(), "validation": validation_expected.sample_id.astype(str).tolist()}
    manifest = {"scope": scope, "data_scope": "official_explicit_contracts" if scope == "official" else DIAGNOSTIC_DATA_SCOPE,
                "diagnostic_notice": "Synthetic keyed targets are not official labels or evidence of financial predictability."
                if scope != "official" else None, "source_cache": store.manifest, "normalization": diagnostics,
                "preprocessor_fingerprint": processor.fingerprint, "sampling_keys_fingerprint": _json_digest(keys),
                "label_spec": vars(label_spec), "fold": {"name": fold.name, "train_start": fold.train_start.isoformat(),
                "validation_start": fold.validation_start.isoformat(), "validation_end": fold.validation_end.isoformat()},
                "window_length": window_length, "fit_window_length": fit_length, "fields": list(fields), "seed": seed,
                "negative_order_policy": store.policy.negative_order_policy, "bar_end_is_available": bar_end_is_available}
    manifest["negative_order_cells_by_loaded_day"] = {key: dict(value) for key, value in store.anomaly_counts_by_day.items()}
    manifest["availability_origin_by_loaded_day"] = dict(store.availability_origin_by_day)
    prepared = PreparedLocalFold(processor, store, train_expected, validation_expected, train, validation, audit,
        trading_grid, fold, label_spec, scope, window_length, fit_length, tuple(fields), seed, bar_end_is_available,
        dict(instrument_map) if instrument_map is not None else None, diagnostics, manifest)
    return prepared.with_window(window_length)


def prepare_local_fold(store: LocalBarStore, expected: pd.DataFrame | None = None, targets: pd.DataFrame | None = None,
                       *, label_spec: LabelSpec | None = None, trading_grid: pd.DatetimeIndex | None = None,
                       fold: TimeFold | None = None, instrument_map: Mapping[int, str] | None = None,
                       labels: pd.DataFrame | None = None, support: pd.DataFrame | None = None,
                       window_length: int = 120, fit_window_length: int | None = None,
                       fields: Sequence[str] = FEATURES, seed: int = 7, bar_end_is_available: bool = False,
                       scale_review: str = "local-streaming-N1:automatic-finite-check; manual-scale-review-skipped") -> PreparedLocalFold:
    """Formal training fails before reading bars if official inputs are missing.

    Supply ``targets`` in A's contract or ``labels`` plus explicit ``support``.
    ``expected`` must preserve the official universe even for missing bars.
    """
    missing = []
    for name, value in (("official expected keys", expected), ("instrument_map", instrument_map),
                         ("LabelSpec", label_spec), ("trading_grid", trading_grid), ("TimeFold", fold)):
        if value is None:
            missing.append(name)
    if targets is None and (labels is None or support is None):
        missing.append("official targets or labels + explicit support")
    if missing:
        raise ContractError("正式训练缺少 " + ", ".join(missing) +
                            "；请从官方截面/标签合同接入，原始四频率行情不包含这些信息。工程诊断使用 prepare_diagnostic_fold。")
    mapping = _validate_mapping(instrument_map)
    if not isinstance(label_spec, LabelSpec):
        raise ContractError("正式训练需要 LabelSpec 实例")
    if not isinstance(expected, pd.DataFrame) or "instrument" not in expected:
        raise ContractError("正式 expected 必须包含官方股票代码 instrument；不能由行情猜测")
    if "instrument_id" not in expected:
        raise ContractError("正式 expected 必须包含 instrument_id")
    if not expected.instrument_id.map(mapping).eq(expected.instrument).all():
        raise ContractError("官方 expected 股票代码与 instrument_map 冲突或无法映射")
    if targets is not None and (labels is not None or support is not None):
        raise ContractError("targets 与 labels/support 只能提供一套，避免两份标签来源不一致")
    if targets is None:
        targets = official_targets(expected, labels, support, label_spec)
    return _prepare(store, expected, targets, label_spec=label_spec, trading_grid=trading_grid, fold=fold,
        instrument_map=mapping, scope="official", window_length=window_length, fit_window_length=fit_window_length,
        fields=fields, seed=seed, bar_end_is_available=bar_end_is_available, scale_review=scale_review)


def prepare_diagnostic_fold(store: LocalBarStore, *, days: Iterable, instrument_ids: Iterable[int],
                            validation_start, endpoint_minutes: Sequence[str] = ("10:00", "11:00", "14:00"),
                            window_length: int = 120, fit_window_length: int | None = None,
                            fields: Sequence[str] = FEATURES, seed: int = 7,
                            bar_end_is_available: bool = False) -> PreparedLocalFold:
    """Real-bar engineering exercise; endpoints and targets are not official.

    Pseudo-targets are reproducible hashes of sample IDs, with no return formula.
    The observed grid has no production trading-calendar/availability proof.
    """
    selected_days, instruments = _days(days), _ids(instrument_ids)
    store.materialize(selected_days, instruments)
    grid_parts, rows = [], []
    if not endpoint_minutes or len(set(endpoint_minutes)) != len(endpoint_minutes):
        raise ContractError("诊断 endpoint_minutes 必须非空且无重复")
    for minute in endpoint_minutes:
        if not isinstance(minute, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", minute):
            raise ContractError("诊断端点须显式为 HH:MM")
    for day in selected_days:
        raw = store.read_day(day, bar_end_is_available=bar_end_is_available)
        grid = pd.DatetimeIndex(raw.values.index.get_level_values("datetime").unique()).sort_values()
        if grid.empty:
            raise ContractError(f"请求的诊断股票在 {day.date()} 无观察分钟")
        grid_parts.append(grid)
        for minute in endpoint_minutes:
            endpoint = pd.Timestamp(f"{day.date()} {minute}")
            if endpoint not in grid:
                raise ContractError(f"诊断端点 {endpoint} 不在观察网格中；请显式选择已有观察分钟")
            decision_id = "diagnostic:" + endpoint.strftime("%Y%m%dT%H%M")
            rows.extend({"sample_id": f"{decision_id}:{instrument}", "decision_id": decision_id,
                         "instrument_id": instrument, "decision_time": endpoint} for instrument in instruments)
    expected = pd.DataFrame(rows)
    spec = LabelSpec("diagnostic-keyed-target.v1", "B engineering pseudo-targets, not an official table",
        "SHA256(sample_id) mapped to [-0.01,0.01]; no future market outcome or financial return",
        "not applicable; diagnostic only", "explicit one-minute intraday diagnostic support")
    targets = expected[["sample_id"]].copy()
    targets["target"] = [0.02 * int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) / (2**64 - 1) - 0.01
                         for value in expected.sample_id]
    targets["target_valid"] = True
    targets["target_start"] = expected.decision_time
    targets["target_end"] = expected.decision_time + pd.Timedelta(minutes=1)
    targets["label_spec_version"] = spec.version
    fold = TimeFold("diagnostic-real-bars", selected_days[0], pd.Timestamp(validation_start),
                    selected_days[-1] + pd.Timedelta(days=1))
    return _prepare(store, expected, targets, label_spec=spec, trading_grid=grid_parts[0].append(grid_parts[1:]),
        fold=fold, instrument_map=None, scope=DIAGNOSTIC_SCOPE, window_length=window_length, fit_window_length=fit_window_length,
        fields=fields, seed=seed, bar_end_is_available=bar_end_is_available,
        scale_review="diagnostic-only:streaming-N1-automatic-finite-check; manual-scale-review-skipped")


def _read_contract_table(path: str | Path | None, name: str) -> pd.DataFrame:
    if not path or not Path(path).is_file():
        raise ContractError(f"请提供存在的官方 {name} 文件：{path}")
    source = Path(path)
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    if source.suffix in {".feather", ".arrow"}:
        return pd.read_feather(source)
    if source.suffix == ".csv":
        return pd.read_csv(source)
    raise ContractError(f"官方 {name} 文件应为 parquet、feather 或 csv")


def _factory_fields(model_spec) -> tuple[str, ...]:
    # Ask the actual model for its encoder fields while preserving caller RNG.
    import torch
    from .temporal import create_score_model

    with torch.random.fork_rng(devices=[]):
        return tuple(create_score_model(model_spec).encoder.fields)


def local_training_factory(*, config: dict, model_spec, fold: TimeFold) -> tuple[PreparedLocalFold, dict[int, str]]:
    """``python -m ptop.train`` official data_factory for explicit local files.

    ``config`` is the CLI's ``data`` dict. Paths are never inferred from raw bars.
    Required keys: data_dir, cache_dir, expected_path, instrument_map_path,
    grid_path, label_spec and targets_path (or labels_path plus support_path).
    """
    if not isinstance(config, dict) or config.get("scope", "official") != "official":
        raise ContractError("local_training_factory 仅接收正式官方合同；诊断请显式用 diagnostic_training_factory")
    required = ("data_dir", "cache_dir", "expected_path", "instrument_map_path", "grid_path", "label_spec")
    missing = [name for name in required if not config.get(name)]
    if not config.get("targets_path") and not (config.get("labels_path") and config.get("support_path")):
        missing.append("targets_path or labels_path + support_path")
    if missing:
        raise ContractError("正式 data_factory 缺少 " + ", ".join(missing) + "；不能用四频率原始行情推测官方合同")
    entries = json.loads(Path(config["instrument_map_path"]).read_text(encoding="utf-8"))
    if not isinstance(entries, dict):
        raise ContractError("instrument_map_path 必须为 JSON ID 到股票代码字典")
    try:
        mapping = {int(key): value for key, value in entries.items()}
    except (ValueError, TypeError) as exc:
        raise ContractError("股票映射 JSON 的键必须为显式整数 ID") from exc
    if len(mapping) != len(entries):
        raise ContractError("股票映射包含等价重复整数 ID")
    mapping = _validate_mapping(mapping)
    expected = _read_contract_table(config["expected_path"], "expected keys")
    grid_frame = _read_contract_table(config["grid_path"], "trading_grid")
    column = config.get("grid_column", "datetime")
    if column not in grid_frame:
        raise ContractError(f"官方网格文件缺少显式时间列 {column}")
    grid = pd.DatetimeIndex(pd.to_datetime(grid_frame[column], errors="raise"))
    spec = LabelSpec(**config["label_spec"])
    store = LocalBarStore(config["data_dir"], cache_dir=config["cache_dir"],
        policy=ReadPolicy(negative_order_policy=config.get("negative_order_policy", "reject"), **config.get("read_budget", {})))
    kwargs = {"label_spec": spec, "trading_grid": grid, "fold": fold, "instrument_map": mapping,
              "window_length": model_spec.lookback, "fit_window_length": config.get("fit_window_length", model_spec.lookback),
              "fields": _factory_fields(model_spec), "seed": config.get("seed", 7),
              "bar_end_is_available": config.get("bar_end_is_available", False),
              "scale_review": config.get("scale_review", "local-factory:automatic-finite-check; manual-scale-review-skipped")}
    if config.get("targets_path"):
        if config.get("labels_path") or config.get("support_path"):
            raise ContractError("targets_path 与 labels/support_path 只能使用一套")
        kwargs["targets"] = _read_contract_table(config["targets_path"], "targets")
    else:
        kwargs.update(labels=_read_contract_table(config["labels_path"], "labels"),
                      support=_read_contract_table(config["support_path"], "target support"))
    return prepare_local_fold(store, expected, **kwargs), mapping


def diagnostic_training_factory(*, config: dict, model_spec, fold: TimeFold) -> tuple[PreparedLocalFold, None]:
    """Explicit diagnostic-only CLI factory; returns no guessed mapping."""
    if not isinstance(config, dict) or config.get("scope") != "diagnostic":
        raise ContractError("diagnostic_training_factory 必须在 data 中显式声明 scope=diagnostic")
    required = ("data_dir", "cache_dir", "days", "instrument_ids")
    if any(not config.get(name) for name in required):
        raise ContractError(f"诊断 data_factory 必须提供 {required}")
    days = _days(config["days"])
    if days[0] != fold.train_start or days[-1] + pd.Timedelta(days=1) != fold.validation_end:
        raise ContractError("诊断 days 必须与 CLI fold 的训练首日/验证末日边界一致")
    store = LocalBarStore(config["data_dir"], cache_dir=config["cache_dir"],
        policy=ReadPolicy(negative_order_policy=config.get("negative_order_policy", "reject"), **config.get("read_budget", {})))
    prepared = prepare_diagnostic_fold(store, days=days, instrument_ids=config["instrument_ids"],
        validation_start=fold.validation_start, endpoint_minutes=config.get("endpoint_minutes", ["10:00", "11:00", "14:00"]),
        window_length=model_spec.lookback, fit_window_length=config.get("fit_window_length", model_spec.lookback),
        fields=_factory_fields(model_spec), seed=config.get("seed", 7), bar_end_is_available=config.get("bar_end_is_available", False))
    prepared.fold = fold
    prepared.manifest["fold"]["name"] = fold.name
    return prepared, None
