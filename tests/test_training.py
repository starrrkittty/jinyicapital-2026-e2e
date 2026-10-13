"""Exercise the complete train/resume/export path against A's prepared fold."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from examples.cloud_bridge import prepare_fold
from ptop.bundle import REQUIRED_CONTRACTS, load_bundle
from ptop.data import normalize_bars
from ptop.official import expected_from_sections
from ptop.schema import FEATURES, ContractError
from ptop.targets import LabelSpec, TimeFold
from ptop.temporal import ModelSpec, create_score_model
from ptop.training import Trainer, _target_table, research_evaluator, seed_everything, validate_batch
from ptop.training_config import TrainingConfig
from ptop.windows import collate_windows


@pytest.fixture
def prepared_fold():
    torch.set_num_threads(1)
    grid = pd.date_range("2024-01-02 09:31", periods=4, freq="min").append(
        pd.date_range("2024-01-03 09:31", periods=4, freq="min"))
    mapping = {i: f"SYNTHETIC_{i}" for i in range(1, 6)}
    bars = pd.MultiIndex.from_product([grid, range(1, 5)], names=["date", "instrument_id"]).to_frame(index=False)
    bars["instrument"] = bars.instrument_id.map(mapping)
    bars["adjust_factor"] = 1.0
    for position, field in enumerate(FEATURES):
        bars[field] = 10.0 + position + np.arange(len(bars)) * (position + 1) * 0.1
    raw = normalize_bars(bars, source="cloud", instrument_map=mapping, bar_end_is_available=True)
    sections = pd.MultiIndex.from_product([grid[[1, 3, 5, 7]], mapping.values()],
                                          names=["date", "instrument"]).to_frame(index=False)
    expected = expected_from_sections(sections, mapping)
    spec = LabelSpec("fixture-v1", "synthetic", "explicit synthetic return", "none", "intraday support")
    targets = expected[["sample_id"]].assign(
        target=(expected.instrument_id.to_numpy() - 3) * 0.01,
        target_valid=expected.instrument_id.ne(4), target_start=expected.decision_time,
        target_end=expected.decision_time + pd.Timedelta(minutes=1), label_spec_version=spec.version)
    targets.loc[~targets.target_valid, "target"] = np.nan
    fold = TimeFold("fixture", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04"))
    prepared = prepare_fold(raw, expected, targets, label_spec=spec, trading_grid=grid, fold=fold,
                            scale_review="synthetic training integration only", window_length=4, seed=19)
    return prepared, mapping, fold


def make_trainer(directory, mapping, fold, *, config=None, model_spec=None, **kwargs):
    config = config or TrainingConfig(epochs=3, patience=5, selection_metric="loss", mode="min",
                                      seed=19, microbatch_size=2, log_every=1)
    model_spec = model_spec or ModelSpec(name="B-GRU", lookback=4, width=8, layers=1, head_width=8, dropout=0.1)
    seed_everything(config.seed, deterministic=config.deterministic)
    model = create_score_model(model_spec)
    contracts = {name: "fixture-contract-v1" for name in REQUIRED_CONTRACTS}
    contracts["label"] = "fixture-v1"
    return Trainer(model, config, output_dir=directory, model_spec=model_spec, contracts=contracts,
                   fold=fold, instrument_map=mapping, **kwargs)


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for name in first:
            assert_nested_equal(first[name], second[name])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_nested_equal(a, b)
    else:
        assert first == second


def test_joint_training_export_and_label_independent_validation(prepared_fold, tmp_path):
    prepared, mapping, fold = prepared_fold
    trainer = make_trainer(tmp_path / "joint", mapping, fold)
    before = {name: value.clone() for name, value in trainer.model.named_parameters()}
    fingerprint = prepared.processor.fingerprint
    result = trainer.fit(prepared)
    assert result.completed_epochs == 3 and result.last_checkpoint.is_file() and result.best_checkpoint.is_file()
    assert result.bundle_path.is_dir()
    assert any(not torch.equal(value, before[name]) for name, value in trainer.model.encoder.named_parameters(prefix="encoder"))
    assert any(not torch.equal(value, before[name]) for name, value in trainer.model.backbone.named_parameters(prefix="backbone"))
    assert prepared.processor.fingerprint == fingerprint
    predictions = pd.read_parquet(tmp_path / "joint/validation/epoch-0001.parquet")
    assert predictions.loc[predictions.instrument_id.eq(4), "score"].notna().all()
    assert predictions.loc[predictions.instrument_id.eq(5), "score"].isna().all()
    assert predictions.target_valid.sum() == 8  # six supervised + two input-empty expected records.
    assert result.history[0]["validation"]["count"] == 6
    assert result.history[0]["validation"]["valid_times"] == 2
    restored, processor, manifest = load_bundle(result.bundle_path)
    batch = next(iter(prepared.validation_loader))
    selected = batch.predictable
    inputs = [torch.as_tensor(value[selected]) for value in (batch.x, batch.observed, batch.step_valid)]
    from ptop.checkpoint import load_inference_checkpoint
    best, _, _ = load_inference_checkpoint(result.best_checkpoint)
    with torch.no_grad():
        torch.testing.assert_close(best(*inputs), restored(*inputs), rtol=0, atol=0)
    assert processor.fingerprint == fingerprint and manifest["model_spec"]["name"] == "B-GRU"
    assert manifest["training_record"]["training"] == trainer.config.to_dict()
    assert manifest["training_record"]["validation_precision"] == "fp32"
    assert manifest["training_record"]["inference_precision"] == "fp32"
    assert json.loads((tmp_path / "joint/result.json").read_text())["platform_evaluation_verified"] is False


def test_epoch_resume_matches_uninterrupted_weights_optimizer_and_scheduler(prepared_fold, tmp_path):
    prepared, mapping, fold = prepared_fold
    full = make_trainer(tmp_path / "full", mapping, fold)
    full_result = full.fit(prepared)
    partial = make_trainer(tmp_path / "restart", mapping, fold)
    limited = partial.fit(prepared, stop_after_epochs=1)
    assert limited.stopped_reason == "invocation_epoch_limit" and limited.completed_epochs == 1
    resumed = make_trainer(tmp_path / "restart", mapping, fold)
    resumed_result = resumed.fit(prepared, resume=limited.last_checkpoint)
    assert resumed_result.completed_epochs == full_result.completed_epochs
    assert_nested_equal(full.model.state_dict(), resumed.model.state_dict())
    assert_nested_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_nested_equal(full.scheduler.state_dict(), resumed.scheduler.state_dict())
    assert full.global_step == resumed.global_step
    assert [item["selection_value"] for item in full_result.history] == [item["selection_value"] for item in resumed_result.history]
    # A second resume at completion verifies an existing immutable best bundle.
    assert make_trainer(tmp_path / "restart", mapping, fold).fit(prepared, resume=resumed_result.last_checkpoint).completed_epochs == 3


def test_shuffled_label_table_joins_by_sample_key(prepared_fold, tmp_path):
    prepared, mapping, fold = prepared_fold
    original = make_trainer(tmp_path / "ordered", mapping, fold)
    original.fit(prepared)
    reordered = replace(prepared, train_targets=prepared.train_targets.iloc[::-1],
                        validation_targets=prepared.validation_targets.iloc[::-1])
    shuffled = make_trainer(tmp_path / "shuffled", mapping, fold)
    shuffled.fit(reordered)
    assert_nested_equal(original.model.state_dict(), shuffled.model.state_dict())


def test_resume_rejects_a_best_checkpoint_from_another_run(prepared_fold, tmp_path):
    import shutil
    prepared, mapping, fold = prepared_fold
    first = make_trainer(tmp_path / "first", mapping, fold)
    saved = first.fit(prepared, stop_after_epochs=1)
    second_config = replace(first.config, seed=23)
    second = make_trainer(tmp_path / "second", mapping, fold, config=second_config)
    other = second.fit(prepared, stop_after_epochs=1)
    shutil.copyfile(other.best_checkpoint, saved.best_checkpoint)
    with pytest.raises(ContractError, match="best.pt does not belong"):
        make_trainer(tmp_path / "first", mapping, fold).fit(prepared, resume=saved.last_checkpoint)


def constant_evaluator(predictions):
    assert len(predictions) == 10  # metrics see complete cross sections after all microbatches.
    return {"constant": 1.0}


def c_objective(scores, targets, decisions, valid):
    return torch.nn.functional.mse_loss(scores[valid].float(), targets[valid].float())


def test_c_hooks_early_stopping_and_earliest_tie(prepared_fold, tmp_path):
    prepared, mapping, fold = prepared_fold
    config = TrainingConfig(epochs=8, patience=2, selection_metric="constant", seed=19, microbatch_size=2)
    trainer = make_trainer(tmp_path / "custom", mapping, fold, config=config,
                           evaluator=constant_evaluator, objective=c_objective)
    result = trainer.fit(prepared)
    assert result.best_epoch == 1 and result.completed_epochs == 3 and result.stopped_reason == "early_stopping"


def test_raw_baseline_trains_without_parameterized_encoder(prepared_fold, tmp_path):
    prepared, mapping, fold = prepared_fold
    spec = ModelSpec(name="R0", lookback=4)
    config = TrainingConfig(epochs=1, patience=1, selection_metric="loss", mode="min", seed=19)
    trainer = make_trainer(tmp_path / "raw", mapping, fold, model_spec=spec, config=config)
    assert not list(trainer.model.encoder.parameters())
    assert trainer.fit(prepared).completed_epochs == 1


@pytest.mark.parametrize("name", ("R1", "R1-Book", "Net64", "Net192", "GroupedNet", "Patch1", "Patch5", "DeepLOB",
                                   "B-Transformer", "B-TCN", "B-SelectiveSSM"))
def test_complete_training_and_best_export_for_all_supported_families(name, prepared_fold, tmp_path):
    prepared, mapping, fold = prepared_fold
    spec = ModelSpec(name=name, lookback=4)
    config = TrainingConfig(epochs=1, patience=1, selection_metric="loss", mode="min", seed=19, microbatch_size=3)
    trainer = make_trainer(tmp_path / name, mapping, fold, model_spec=spec, config=config)
    result = trainer.fit(prepared)
    assert result.completed_epochs == 1 and result.bundle_path.is_dir()
    restored, _, manifest = load_bundle(result.bundle_path)
    assert manifest["model_spec"] == spec.to_dict()
    assert type(restored) is type(trainer.model)


def test_full_trainer_cpu_bfloat16_validation(prepared_fold, tmp_path):
    prepared, mapping, fold = prepared_fold
    config = TrainingConfig(epochs=1, patience=1, selection_metric="loss", mode="min", seed=19,
                            precision="bf16", microbatch_size=3)
    spec = ModelSpec(name="B-Transformer", lookback=4, width=8, layers=1, head_width=8)
    result = make_trainer(tmp_path / "bf16", mapping, fold, model_spec=spec, config=config).fit(prepared)
    assert np.isfinite(result.best_metric)
    assert_fp32_validation_replay(result, prepared, device="cpu")


def assert_fp32_validation_replay(result, prepared, *, device):
    """Saved selection scores must agree with a fresh default inference model."""
    from ptop.checkpoint import load_inference_checkpoint
    model, _, _ = load_inference_checkpoint(result.best_checkpoint, device=device)
    saved = pd.read_parquet(result.best_checkpoint.parent / "validation" /
                            f"epoch-{result.best_epoch:04d}.parquet").set_index("sample_id")
    for batch in prepared.validation_loader:
        selected = batch.predictable
        inputs = [torch.as_tensor(value[selected], device=device)
                  for value in (batch.x, batch.observed, batch.step_valid)]
        with torch.inference_mode():
            scores = model(*inputs).float().cpu().numpy()
        expected = saved.loc[batch.metadata.loc[selected, "sample_id"], "score"].to_numpy()
        np.testing.assert_allclose(scores, expected, rtol=1e-5, atol=1e-6)


def test_purge_is_by_explicit_support_not_prediction_time(prepared_fold):
    prepared, _, fold = prepared_fold
    table = prepared.train_targets.copy()
    sample = table.target_valid.idxmax()
    # An overnight support is rejected, rather than interpreting a guessed horizon.
    table.loc[sample, "target_end"] = fold.validation_start
    with pytest.raises(ContractError, match="overnight"):
        _target_table(table, "train", fold)
    intraday = TimeFold("intraday", fold.train_start, pd.Timestamp("2024-01-02 09:36"), fold.validation_end)
    table.loc[sample, "target_end"] = intraday.validation_start
    result = _target_table(table, "train", intraday)
    assert not result.loc[sample, "target_valid"]
    assert result.loc[sample, "supervision_exclusion"] == "target_reaches_split_boundary"


def test_future_input_invalid_labels_missing_batches_and_resume_config_fail(prepared_fold, tmp_path):
    prepared, mapping, fold = prepared_fold
    batch = next(iter(prepared.validation_loader))
    times = batch.bar_times.copy()
    times[0, -1] += np.timedelta64(1, "m")
    with pytest.raises(ContractError, match="future"):
        validate_batch(replace(batch, bar_times=times), FEATURES, length=4, scope="official")
    invalid = replace(prepared, validation_targets=prepared.validation_targets.assign(target_valid=False, target=np.nan))
    with pytest.raises(ContractError, match="no valid supervised"):
        make_trainer(tmp_path / "invalid", mapping, fold).fit(invalid)
    from torch.utils.data import DataLoader
    incomplete = replace(prepared, validation_loader=DataLoader(prepared.validation_loader.dataset,
                        batch_sampler=[list(range(5))], collate_fn=collate_windows))
    with pytest.raises(ContractError, match="complete declared"):
        make_trainer(tmp_path / "incomplete", mapping, fold).fit(incomplete)
    partial = make_trainer(tmp_path / "mismatch", mapping, fold).fit(prepared, stop_after_epochs=1)
    changed = TrainingConfig(epochs=4, patience=5, selection_metric="loss", mode="min", seed=19, microbatch_size=2, log_every=1)
    with pytest.raises(ContractError, match="config|配置"):
        make_trainer(tmp_path / "mismatch", mapping, fold, config=changed).fit(prepared, resume=partial.last_checkpoint)


@pytest.mark.parametrize("settings", [{"target_scale": "bad"}, {"epochs": True}, {"learning_rate": float("nan")},
                                       {"precision": "fp8"}, {"seed": -1}, {"microbatch_size": 0}, {"unknown": 7}])
def test_bad_training_settings_fail_explicitly(settings):
    with pytest.raises(ContractError):
        TrainingConfig.from_dict(settings)
