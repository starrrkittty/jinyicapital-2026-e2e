"""Regression tests for bounded ingestion, frozen N1 and experiment fairness."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pytest
import torch

from ptop.data import concat_partitions
from ptop.experiments import (ExperimentBudget, ExperimentBudgetExceeded, ExperimentPlan,
                              benchmark_windows, requests_for_plan, run_experiments)
from ptop.local_training import (DIAGNOSTIC_SCOPE, LocalBarStore, ReadPolicy, _ColumnReader, _pinned_pairwise_sum,
                                diagnostic_training_factory, local_training_factory,
                                prepare_diagnostic_fold, prepare_local_fold, split_local_targets)
from ptop.preprocessing import FrozenPreprocessor
from ptop.schema import BOOK_ORDER_FIELDS, FEATURES, ContractError
from ptop.targets import LabelSpec, TimeFold
from ptop.windows import WindowDataset, fit_row_union


@pytest.fixture
def local_fixture(tmp_path):
    torch.set_num_threads(1)
    days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    grid = pd.DatetimeIndex([])
    for day in days:
        grid = grid.append(pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=4, freq="min"))
    bars = pd.MultiIndex.from_product([grid, [1, 2]], names=["date", "instrument_id"]).to_frame(index=False)
    for column, name in enumerate(FEATURES):
        bars[name] = np.arange(len(bars), dtype=np.int64) * (column + 1) + 1000 + column
    bars["deal_number"] = 0  # Qlib ignores constant fields instead of dividing by zero.
    bars.loc[0, "bid_price1"] = 0  # Known missing book price; one training cell is absent.
    bars[BOOK_ORDER_FIELDS[0]] += 50000  # Valid positive int32 counts must not wrap.
    bars["available_at"] = bars.date
    bars.loc[2, "available_at"] += pd.Timedelta(minutes=2)  # One bar delayed past its first decision.
    bars.loc[bars.date.dt.normalize().eq(days[-1]), "close"] += 10_000_000
    source = tmp_path / "source"
    source.mkdir()
    feather.write_feather(bars, source / "202401.0.feather", chunksize=4)
    mapping = {1: "SYNTHETIC_A", 2: "SYNTHETIC_B", 3: "SYNTHETIC_EMPTY"}
    samples = []
    for point in grid[grid.minute.isin([32, 34])]:
        decision = point.strftime("%Y%m%dT%H%M")
        for instrument in mapping:
            samples.append({"sample_id": f"{decision}:{instrument}", "decision_id": decision,
                            "decision_time": point, "instrument_id": instrument,
                            "instrument": mapping[instrument]})
    expected = pd.DataFrame(samples)
    spec = LabelSpec("synthetic-contract.v1", "explicit-test-table", "synthetic exact values", "none-test-only", "one-minute explicit support")
    targets = expected[["sample_id"]].assign(target=np.linspace(-0.01, 0.01, len(expected)),
        target_valid=True, target_start=expected.decision_time,
        target_end=expected.decision_time + pd.Timedelta(minutes=1), label_spec_version=spec.version)
    fold = TimeFold("synthetic-local", days[0], days[-1], days[-1] + pd.Timedelta(days=1))
    store = LocalBarStore(source, cache_dir=tmp_path / "cache")
    return source, bars, mapping, expected, targets, spec, grid, fold, store


def _prepared(fixture, **kwargs):
    _, _, mapping, expected, targets, spec, grid, fold, store = fixture
    return prepare_local_fold(store, expected, targets, instrument_map=mapping, label_spec=spec,
        trading_grid=grid, fold=fold, window_length=4, scale_review="synthetic numerical regression; no manual-review claim", **kwargs)


def test_partitioned_n1_equals_pinned_a_full_fit_and_validation_is_frozen(local_fixture):
    prepared = _prepared(local_fixture)
    _, _, mapping, _, _, _, grid, fold, store = local_fixture
    partitions = [store.read_day(day, instrument_map=mapping) for day in pd.to_datetime(["2024-01-02", "2024-01-03"])]
    whole_train = concat_partitions(partitions)
    windows = WindowDataset(whole_train, prepared.train_expected, grid, length=4)
    references = fit_row_union(windows, fit_start=fold.train_start, fit_end="2024-01-03")
    direct = FrozenPreprocessor("N1").fit(whole_train.values, whole_train.observed,
        referenced_rows=references, fit_start=fold.train_start, fit_end="2024-01-03")
    direct.freeze(scale_review="test direct pinned A fit")
    a, b = direct.state_dict(), prepared.processor.state_dict()
    for name in ("fields", "fit_row_count", "fit_rows_fingerprint", "valid_counts", "training_constants"):
        assert a[name] == b[name]
    assert a["statistics"]["ignore"] == b["statistics"]["ignore"]
    np.testing.assert_allclose(a["statistics"]["center"], b["statistics"]["center"], rtol=2e-15, atol=1e-10)
    np.testing.assert_allclose(a["statistics"]["scale"], b["statistics"]["scale"], rtol=2e-14, atol=1e-10)
    np.testing.assert_allclose(direct.transform(whole_train.values, whole_train.observed),
                               prepared.processor.transform(whole_train.values, whole_train.observed), rtol=1e-6, atol=1e-6)
    assert b["fit_row_count"] == 16
    assert b["valid_counts"][FEATURES.index("bid_price1")] == 15
    assert b["statistics"]["ignore"][FEATURES.index("deal_number")]
    before = prepared.processor.fingerprint
    list(prepared.validation_loader())
    assert prepared.processor.fingerprint == before
    assert prepared.normalization_diagnostics["automatic_float32_finite_check"]
    assert prepared.normalization_diagnostics["implementation"].startswith("pooled_")


def test_decimal_constant_fields_match_actual_a_ignore_with_asymmetric_partitions(local_fixture):
    source, bars, mapping, expected, targets, spec, grid, fold, store = local_fixture
    bars = bars.drop(index=1).reset_index(drop=True)  # First training day has seven rows.
    bars["open"] = 946  # 9.46 is not exactly representable in binary float64.
    bars["bid_price2"] = 2711
    bars.loc[0, "bid_price2"] = 0  # Constant observed price, plus a missing cell.
    extra = bars.loc[bars.date.eq(pd.Timestamp("2024-01-03 09:34"))].iloc[-1:].copy()
    extra["instrument_id"] = 4
    bars = pd.concat([bars, extra], ignore_index=True)  # Second training day has nine rows.
    feather.write_feather(bars, source / "202401.0.feather", chunksize=4)
    mapping = {**mapping, 4: "SYNTHETIC_EXTRA"}
    expected = pd.concat([expected, pd.DataFrame([{"sample_id": "20240103T0934:4", "decision_id": "20240103T0934",
        "decision_time": pd.Timestamp("2024-01-03 09:34"), "instrument_id": 4, "instrument": mapping[4]}])], ignore_index=True)
    targets = pd.concat([targets, pd.DataFrame([{"sample_id": "20240103T0934:4", "target": 0.001, "target_valid": True,
        "target_start": pd.Timestamp("2024-01-03 09:34"), "target_end": pd.Timestamp("2024-01-03 09:35"),
        "label_spec_version": spec.version}])], ignore_index=True)
    prepared = prepare_local_fold(store, expected, targets, label_spec=spec, trading_grid=grid, fold=fold,
        instrument_map=mapping, window_length=4, scale_review="decimal constant numerical regression")
    raw = concat_partitions([store.read_day(day, instrument_map=mapping) for day in pd.to_datetime(["2024-01-02", "2024-01-03"])])
    windows = WindowDataset(raw, prepared.train_expected, grid, length=4)
    references = fit_row_union(windows, fit_start=fold.train_start, fit_end="2024-01-03")
    direct = FrozenPreprocessor("N1").fit(raw.values, raw.observed, referenced_rows=references,
                                         fit_start=fold.train_start, fit_end="2024-01-03")
    direct.freeze(scale_review="direct pinned A decimal-constant fit")
    a, b = direct.state_dict(), prepared.processor.state_dict()
    assert a["statistics"]["ignore"] == b["statistics"]["ignore"]
    for field in ("open", "bid_price2"):
        position = FEATURES.index(field)
        assert a["statistics"]["center"][position] == b["statistics"]["center"][position]
        assert a["statistics"]["scale"][position] == b["statistics"]["scale"][position]
    assert a["fit_rows_fingerprint"] == b["fit_rows_fingerprint"]
    assert "bid_price2" in prepared.normalization_diagnostics["constant_field_exact_reduction"]


@pytest.mark.parametrize("size", [15, 129, 257, 8191, 8192, 8193, 12345, 100000])
def test_bounded_constant_missing_reduction_matches_pinned_numpy_bitwise(tmp_path, size):
    random = np.random.default_rng(7)
    values = np.full(size, 9.46)
    values[random.random(size) < 0.1] = 0.0
    path = tmp_path / "constant-column.f64"
    path.write_bytes(values.tobytes())
    reader = _ColumnReader(path, size)
    try:
        actual = _pinned_pairwise_sum(reader, 0, size)
        assert actual == np.sum(values)
        center = float(actual / np.count_nonzero(values))
        std = np.sqrt(_pinned_pairwise_sum(reader, 0, size, squared_center=center) / np.count_nonzero(values))
        assert std == np.nanstd(np.where(values != 0, values, np.nan))
        assert len(reader.buffer) <= 16384
    finally:
        reader.close()


def test_large_constant_missing_reference_matches_actual_pinned_qlib(tmp_path):
    from qlib.data.dataset.processor import ZScoreNorm

    values = np.full(12345, 9.46)
    values[np.random.default_rng(7).random(len(values)) < 0.1] = np.nan
    index = pd.MultiIndex.from_arrays([pd.DatetimeIndex(["2024-01-02 10:00"] * len(values)), np.arange(len(values))],
                                     names=["datetime", "instrument"])
    frame = pd.DataFrame({("feature", "constant"): values, ("feature", "other"): np.arange(len(values))}, index=index)
    reference = ZScoreNorm(fit_start_time="2024-01-02", fit_end_time="2024-01-02", fields_group="feature")
    reference.fit(frame)
    path = tmp_path / "qlib-reference-column.f64"
    path.write_bytes(np.nan_to_num(values, nan=0.0).tobytes())
    reader = _ColumnReader(path, len(values))
    try:
        count = int(np.isfinite(values).sum())
        center = float(_pinned_pairwise_sum(reader, 0, len(values)) / count)
        scale = float(np.sqrt(_pinned_pairwise_sum(reader, 0, len(values), squared_center=center) / count))
        assert reference.mean_train[0] == (0.0 if scale == 0 else center)
        assert reference.std_train[0] == (1.0 if scale == 0 else scale)
        assert bool(reference.ignore[0]) == (scale == 0)
    finally:
        reader.close()


def test_day_loader_visible_slots_empty_official_symbols_and_raw_only_cache(local_fixture):
    prepared = _prepared(local_fixture)
    batches = list(prepared.validation_loader())
    assert len(batches) == 2
    for batch in batches:
        assert len(batch.metadata) == 3 and batch.metadata.decision_id.nunique() == 1
        assert batch.metadata.loc[~batch.predictable, "instrument_id"].tolist() == [3]
        for row, metadata in batch.metadata.iterrows():
            times = batch.bar_times[row][batch.step_valid[row]]
            assert (times <= metadata.decision_time.to_datetime64()).all()
            assert (pd.DatetimeIndex(times).normalize() == metadata.decision_time.normalize()).all()
    delayed = next(batch for batch in prepared.train_loader() if batch.metadata.decision_time.iloc[0] == pd.Timestamp("2024-01-02 09:32"))
    row = int(np.flatnonzero(delayed.metadata.instrument_id.eq(1))[0])
    assert delayed.step_valid[row, -1]
    assert not delayed.observed[row, -1].any()
    assert delayed.row_positions[row, -1] == -1
    later = next(batch for batch in prepared.train_loader() if batch.metadata.decision_time.iloc[0] == pd.Timestamp("2024-01-02 09:34"))
    row = int(np.flatnonzero(later.metadata.instrument_id.eq(1))[0])
    assert later.observed[row, -3].any()
    assert prepared.store.profile["max_day_rows_read"] == 8
    assert prepared.store.manifest["cache_kind"] == "raw_only_no_learned_tokens"
    assert all(entry["rows"] == 8 for entry in prepared.store.manifest["partitions"].values())
    cache_files = list(prepared.store._scope_dir.iterdir())
    assert {path.suffix for path in cache_files} == {".arrow", ".json"}


def test_train_seed_is_epoch_explicit_and_shorter_windows_reuse_same_n1(local_fixture):
    prepared = _prepared(local_fixture)
    first = [batch.metadata.sample_id.tolist() for batch in prepared.train_loader(2)]
    repeated = [batch.metadata.sample_id.tolist() for batch in prepared.train_loader(2)]
    assert first == repeated
    shorter = prepared.with_window(2, seed=29)
    assert shorter.processor is prepared.processor
    assert shorter.manifest["sampling_keys_fingerprint"] == prepared.manifest["sampling_keys_fingerprint"]
    assert next(iter(shorter.validation_loader())).x.shape == (3, 2, 25)
    with pytest.raises(ContractError, match="fit_window_length"):
        prepared.with_window(5)


def test_negative_order_mask_is_opt_in_and_other_negatives_remain_errors(local_fixture, tmp_path):
    source, bars, *_ = local_fixture
    bars.loc[0, BOOK_ORDER_FIELDS[0]] = -3
    feather.write_feather(bars, source / "202401.0.feather", chunksize=4)
    rejecting = LocalBarStore(source, cache_dir=tmp_path / "rejecting")
    rejecting.materialize(["2024-01-02"], [1, 2])
    with pytest.raises(ContractError, match="未知负值"):
        rejecting.read_day("2024-01-02")
    masking = LocalBarStore(source, cache_dir=tmp_path / "masking", policy=ReadPolicy(negative_order_policy="mask_known"))
    masking.materialize(["2024-01-02"], [1, 2])
    raw = masking.read_day("2024-01-02")
    assert not raw.observed.iloc[0][BOOK_ORDER_FIELDS[0]]
    assert np.isnan(raw.values.iloc[0][BOOK_ORDER_FIELDS[0]])
    assert raw.values.iloc[1][BOOK_ORDER_FIELDS[0]] > 32767
    assert masking.profile["masked_negative_orders"][BOOK_ORDER_FIELDS[0]] == 1
    masking.read_day("2024-01-02")
    assert masking.anomaly_counts_by_day["2024-01-02"][BOOK_ORDER_FIELDS[0]] == 1
    assert masking.profile["masked_negative_orders"][BOOK_ORDER_FIELDS[0]] == 2
    bars.loc[0, "volume"] = -1
    feather.write_feather(bars, source / "202401.0.feather", chunksize=4)
    masking.materialize(["2024-01-02"], [1, 2])
    with pytest.raises(ContractError, match="volume"):
        masking.read_day("2024-01-02")


def test_source_and_daily_budgets_and_cache_integrity_are_enforced(local_fixture, tmp_path):
    source, *_ = local_fixture
    for policy, message in ((ReadPolicy(max_source_bytes=1), "max_source_bytes"),
                            (ReadPolicy(max_record_batch_rows=1), "max_record_batch_rows"),
                            (ReadPolicy(max_day_rows=1), "max_day_rows")):
        store = LocalBarStore(source, cache_dir=tmp_path / message, policy=policy)
        with pytest.raises(ContractError, match=message):
            store.materialize(["2024-01-02"], [1, 2])
    store = LocalBarStore(source, cache_dir=tmp_path / "integrity")
    store.materialize(["2024-01-02"], [1, 2])
    store.read_day("2024-01-02")
    file = store._scope_dir / "2024-01-02.arrow"
    with file.open("ab") as stream:
        stream.write(b"changed-after-first-read")
    with pytest.raises(ContractError, match="校验失败"):
        store.read_day("2024-01-02")


def test_missing_formal_contract_fails_before_source_read_and_no_mapping_is_guessed(local_fixture):
    *_, store = local_fixture
    with pytest.raises(ContractError, match="official expected keys.*instrument_map.*LabelSpec.*explicit support"):
        prepare_local_fold(store)
    assert store.manifest is None
    diagnostic = prepare_diagnostic_fold(store, days=["2024-01-02", "2024-01-03", "2024-01-04"],
        instrument_ids=[1, 2], validation_start="2024-01-04", endpoint_minutes=["09:32", "09:34"], window_length=4)
    assert diagnostic.scope == DIAGNOSTIC_SCOPE and diagnostic.instrument_map is None
    assert "not official" in diagnostic.manifest["diagnostic_notice"]
    batch = next(iter(diagnostic.validation_loader()))
    assert "instrument" not in batch.metadata
    assert not batch.metadata.production_visibility_verified.any()
    assert batch.metadata.audit_scope.eq("local_observed_grid").all()
    assert diagnostic.label_spec.version.startswith("diagnostic-")


def test_assumed_bar_end_is_not_misreported_as_source_provided_availability(local_fixture):
    source, bars, *_, store = local_fixture
    feather.write_feather(bars.drop(columns="available_at"), source / "202401.0.feather", chunksize=4)
    diagnostic = prepare_diagnostic_fold(store, days=["2024-01-02", "2024-01-03", "2024-01-04"],
        instrument_ids=[1, 2], validation_start="2024-01-04", endpoint_minutes=["09:32", "09:34"],
        window_length=4, bar_end_is_available=True)
    raw = store.read_day("2024-01-02", bar_end_is_available=True)
    assert raw.metadata.attrs["availability_origin"] == "assumed_bar_end"
    assert raw.metadata.available_at.eq(raw.values.index.get_level_values("datetime")).all()
    for batch in diagnostic.validation_loader():
        assert batch.metadata.visibility_basis.eq("bar_end_only").all()
        assert batch.metadata.availability_assumption.eq("bar_end_equals_available_at").all()
        assert not batch.metadata.production_visibility_verified.any()
    assert set(diagnostic.manifest["availability_origin_by_loaded_day"].values()) == {"assumed_bar_end"}
    # A provided timestamp remains distinct from an explicitly assumed bar end.
    feather.write_feather(bars, source / "202401.0.feather", chunksize=4)
    provided = prepare_diagnostic_fold(store, days=["2024-01-02", "2024-01-03", "2024-01-04"],
        instrument_ids=[1, 2], validation_start="2024-01-04", endpoint_minutes=["09:32", "09:34"], window_length=4)
    batch = next(iter(provided.validation_loader()))
    assert batch.metadata.visibility_basis.eq("provided_available_at").all()
    assert batch.metadata.availability_assumption.eq("none").all()


def test_target_support_at_split_boundary_is_purged_without_deleting_keys():
    expected = pd.DataFrame({"sample_id": ["a", "b", "c"], "decision_id": ["d1", "d2", "d3"],
        "instrument_id": [1, 1, 1], "decision_time": pd.to_datetime(["2024-01-02 10:00", "2024-01-02 10:02", "2024-01-02 10:03"])})
    spec = LabelSpec("test.v1", "test", "test", "test", "test")
    targets = expected[["sample_id"]].assign(target=1.0, target_valid=True, target_start=expected.decision_time,
        target_end=pd.to_datetime(["2024-01-02 10:02", "2024-01-02 10:03", "2024-01-02 10:05"]), label_spec_version=spec.version)
    fold = TimeFold("boundary", "2024-01-02 09:30", "2024-01-02 10:02", "2024-01-02 10:05")
    train, validation, audit = split_local_targets(expected, targets, spec, fold)
    assert train.sample_id.tolist() == ["a"] and not train.target_valid.any()
    assert validation.sample_id.tolist() == ["b", "c"]
    assert validation.target_valid.tolist() == [True, False]
    assert audit.supervision_exclusion.eq("target_reaches_split_boundary").sum() == 2
    assert pd.isna(train.target.iloc[0]) and pd.isna(validation.target.iloc[1])


def test_experiment_scheduler_locks_keys_labels_scaler_and_independent_budget(local_fixture, tmp_path, monkeypatch):
    prepared = _prepared(local_fixture)
    plan = ExperimentPlan(({"name": "model-a"}, {"name": "model-b"}), lookbacks=(2, 4), seeds=(7, 17),
                          budget=ExperimentBudget(epochs=2, max_wall_seconds=30.0))
    requests = requests_for_plan(plan)
    assert len(requests) == len({request.run_id for request in requests}) == 8
    calls = []

    def runner(request, current, output_dir):
        calls.append((request.run_id, current.window_length, current.seed, current.processor.fingerprint))
        flags = {key: bool(flag) for batch in current.validation_loader() for key, flag in zip(batch.metadata.sample_id, batch.predictable)}
        frame = current.validation_targets[["sample_id", "target_valid"]].assign(score=0.1)
        frame["predictable"] = frame.sample_id.map(flags)
        return {"validation_predictions": frame, "metrics": {"loss": 0.01}, "best_metric": 0.01,
                "selection_metric": "loss"}

    summary = run_experiments(plan, prepared, output_dir=tmp_path / "grid", runner=runner)
    assert summary.status.eq("completed").all()
    assert len({item[-1] for item in calls}) == 1
    assert summary.eligible_prediction_keys_fingerprint.nunique() == 1
    resumed = run_experiments(plan, prepared, output_dir=tmp_path / "grid", runner=runner, resume_completed=True)
    assert len(calls) == 8 and len(resumed) == 8
    with pytest.raises(ContractError, match="固定计划"):
        run_experiments(replace(plan, seeds=(29,)), prepared, output_dir=tmp_path / "grid", runner=runner)

    def altered_keys(request, current, output_dir):
        flags = {key: bool(flag) for batch in current.validation_loader() for key, flag in zip(batch.metadata.sample_id, batch.predictable)}
        frame = current.validation_targets[["sample_id", "target_valid"]].assign(score=0.1)
        frame["predictable"] = frame.sample_id.map(flags)
        if request.model_spec["name"] == "model-b":
            frame = frame.iloc[1:]
        return {"validation_predictions": frame, "metrics": {}, "best_metric": 0.1, "selection_metric": "loss"}

    mismatch = run_experiments(replace(plan, lookbacks=(4,), seeds=(7,)), prepared,
        output_dir=tmp_path / "mismatch", runner=altered_keys)
    assert mismatch.status.tolist() == ["completed", "failed"]
    assert "采样键" in mismatch.error.iloc[-1]

    def over_budget(request, current, output_dir):
        raise ExperimentBudgetExceeded("declared test budget")

    exceeded = run_experiments(replace(plan, lookbacks=(4,), seeds=(7,)), prepared,
        output_dir=tmp_path / "exceeded", runner=over_budget)
    assert exceeded.status.eq("budget_exceeded").all()
    monkeypatch.setattr("ptop.experiments._experiment_implementation_contract", lambda: {"algorithm": "changed"})
    with pytest.raises(ContractError, match="固定计划"):
        run_experiments(plan, prepared, output_dir=tmp_path / "grid", runner=runner, resume_completed=True)


def test_real_window_benchmark_reports_waits_and_memory_without_targets(local_fixture):
    prepared = _prepared(local_fixture)

    class TinyScore(torch.nn.Module):
        def forward(self, x, observed, step_valid):
            return x[:, -1, 0]

    metrics = benchmark_windows(prepared.validation_loader(), model=TinyScore(), device="cpu")
    assert metrics["batches"] == 2 and metrics["samples"] == 6 and metrics["predictable_samples"] == 4
    assert metrics["data_wait_seconds"] > 0 and metrics["forward_seconds"] > 0
    assert metrics["peak_rss_bytes_process"] > 0 and metrics["samples_per_second"] > 0
    assert metrics["peak_cuda_allocated_bytes"] is None


@pytest.mark.parametrize("model_name", ["R0", "R1", "B-GRU"])
def test_default_experiment_runner_trains_native_a_and_modern_b(local_fixture, tmp_path, model_name):
    from ptop.checkpoint import _read_checkpoint, load_inference_checkpoint
    from ptop.temporal import ModelSpec, create_score_model
    from ptop.training import seed_everything

    *_, store = local_fixture
    prepared = prepare_diagnostic_fold(store, days=["2024-01-02", "2024-01-03", "2024-01-04"],
        instrument_ids=[1, 2], validation_start="2024-01-04", endpoint_minutes=["09:32", "09:34"], window_length=4)
    contracts = {name: "synthetic engineering check" for name in
                 ("source", "sampling", "universe", "visibility", "training_scope")}
    contracts["label"] = prepared.label_spec.version
    plan = ExperimentPlan(({"name": model_name},), lookbacks=(4,), seeds=(7,),
        training={"selection_metric": "loss", "mode": "min", "precision": "fp32", "microbatch_size": 2},
        budget=ExperimentBudget(epochs=1, max_wall_seconds=30.0), contracts=contracts)
    comparison = run_experiments(plan, prepared, output_dir=tmp_path / "native-default-runner")
    assert comparison.status.tolist() == ["completed"], comparison.to_dict("records")
    run = tmp_path / "native-default-runner" / comparison.run_id.iloc[0]
    record = json.loads((run / "run.json").read_text())
    checkpoint = _read_checkpoint(run / "last.pt")
    assert checkpoint["metadata"]["epoch"] == 1 and checkpoint["metadata"]["global_step"] == 4
    counts = record["parameter_counts"]
    assert sum(counts.values()) == record["parameter_count"]
    if model_name == "R0":
        assert set(counts) == {"encoder", "rnn", "fc_out"} and counts["encoder"] == 0
        assert counts["rnn"] > 0 and counts["fc_out"] > 0
    elif model_name == "R1":
        assert set(counts) == {"encoder", "rnn", "att_net", "fc_out"}
        assert all(value > 0 for value in counts.values())
    else:
        assert all(counts[part] > 0 for part in ("encoder", "backbone", "head"))
    assert record["tcn_receptive_field_tokens"] is None
    seed_everything(7, deterministic=True)
    initial = create_score_model(ModelSpec.from_dict(record["model_spec"]))
    for part, count in counts.items():
        if count:
            assert any(not torch.equal(checkpoint["model"][name], value) for name, value in initial.state_dict().items()
                       if name.startswith(part + "."))
    restored, processor, _ = load_inference_checkpoint(run / "best.pt")
    assert processor.fingerprint == prepared.processor.fingerprint
    actual = {}
    for batch in prepared.validation_loader():
        with torch.no_grad():
            score = restored(*(torch.as_tensor(value) for value in (batch.x, batch.observed, batch.step_valid)))
        actual.update(zip(batch.metadata.sample_id, score.numpy()))
    stored = pd.read_parquet(run / "comparison_predictions.parquet").set_index("sample_id")
    assert len(stored) == 4 and stored.predictable.all() and stored.target_valid.all()
    np.testing.assert_array_equal(pd.Series(actual).loc[stored.index].to_numpy(), stored.score.to_numpy())


def test_cli_file_factories_require_official_contracts_and_preserve_explicit_fold(local_fixture, tmp_path, monkeypatch):
    source, _, mapping, expected, targets, spec, grid, fold, _ = local_fixture
    model_spec = SimpleNamespace(lookback=4)
    with pytest.raises(ContractError, match="正式 data_factory 缺少"):
        local_training_factory(config={"data_dir": str(source)}, model_spec=model_spec, fold=fold)
    monkeypatch.setattr("ptop.local_training._factory_fields", lambda spec: FEATURES)
    expected_path, target_path = tmp_path / "expected.parquet", tmp_path / "targets.parquet"
    expected.to_parquet(expected_path, index=False)
    targets.to_parquet(target_path, index=False)
    mapping_path = tmp_path / "instrument_map.json"
    mapping_path.write_text(json.dumps(mapping))
    grid_path = tmp_path / "grid.csv"
    pd.DataFrame({"datetime": grid}).to_csv(grid_path, index=False)
    config = {"data_dir": str(source), "cache_dir": str(tmp_path / "factory_cache"),
              "expected_path": str(expected_path), "targets_path": str(target_path),
              "instrument_map_path": str(mapping_path), "grid_path": str(grid_path), "label_spec": vars(spec)}
    prepared, returned_mapping = local_training_factory(config=config, model_spec=model_spec, fold=fold)
    assert returned_mapping == mapping and prepared.scope == "official"
    assert next(iter(prepared.validation_loader())).x.shape == (3, 4, 25)
    diagnostic_config = {"scope": "diagnostic", "data_dir": str(source), "cache_dir": str(tmp_path / "diagnostic_factory_cache"),
                         "days": ["2024-01-02", "2024-01-03", "2024-01-04"], "instrument_ids": [1, 2],
                         "endpoint_minutes": ["09:32", "09:34"]}
    diagnostic, absent_mapping = diagnostic_training_factory(config=diagnostic_config, model_spec=model_spec, fold=fold)
    assert absent_mapping is None and diagnostic.scope == DIAGNOSTIC_SCOPE
    assert diagnostic.fold.name == fold.name and diagnostic.manifest["fold"]["name"] == fold.name


def test_experiment_seed_range_matches_training_config():
    with pytest.raises(ContractError, match=r"2\*\*32"):
        ExperimentPlan(({"name": "test"},), seeds=(2**32,))
