"""Synthetic interface checks; these do not claim official DAI/CUDA validation."""

import copy
import json
import os
import random
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from ptop.bundle import REQUIRED_CONTRACTS, create_model, load_bundle, save_bundle
from ptop.checkpoint import load_checkpoint, load_inference_checkpoint, save_checkpoint
from ptop.data import CanonicalData
from ptop.inference import (
    InferencePartition, predict_bundle, predict_bundle_partitions, predict_canonical,
    predict_cloud, predict_cloud_partitions, predict_windows,
)
from ptop.official import LABEL_TABLE, SECTION_TABLE, expected_from_sections
from ptop.preprocessing import FrozenPreprocessor
from ptop.schema import BOOK12_FIELDS, FEATURES, ContractError
from ptop.temporal import ModelSpec, create_score_model
from ptop.windows import DecisionBatchSampler, WindowDataset, collate_windows


CONTRACTS = {name: "synthetic-interface-v1" for name in REQUIRED_CONTRACTS}
CONFIG = {"seed": 17, "max_epochs": 4, "objective": {"name": "mse"}}
SPLIT = {"train_start": "2024-01-02", "validation_start": "2024-01-03", "validation_end": "2024-01-04"}


@pytest.fixture
def inputs():
    torch.set_num_threads(2)
    grid = pd.date_range("2024-01-02 09:31", periods=4, freq="min").append(
        pd.date_range("2024-01-03 09:31", periods=4, freq="min"))
    mapping = {11: "SYNTHETIC_A", 12: "SYNTHETIC_B", 13: "SYNTHETIC_EMPTY"}
    index = pd.MultiIndex.from_product([grid, [11, 12]], names=["datetime", "instrument"])
    values = pd.DataFrame(np.arange(len(index))[:, None] * 0.5 + np.arange(len(FEATURES))[None, :] + 10.0,
                          index=index, columns=FEATURES, dtype=np.float64)
    # One real minute has no observation; it must remain an actual time slot.
    values.loc[(pd.Timestamp("2024-01-03 09:32"), 11), :] = np.nan
    observed = values.notna()
    metadata = pd.DataFrame({"available_at": index.get_level_values("datetime")}, index=index)
    raw = CanonicalData(values, observed, metadata, mapping)
    training = index[index.get_level_values("datetime").normalize() == pd.Timestamp("2024-01-02")]
    processor = FrozenPreprocessor("N1").fit(values, observed, referenced_rows=training,
                                             fit_start="2024-01-02", fit_end="2024-01-02")
    processor.freeze(scale_review="synthetic interface fixture, no official validation")
    sections = pd.MultiIndex.from_product([grid[[1, 3, 5, 7]], mapping.values()],
                                          names=["date", "instrument"]).to_frame(index=False)
    expected = expected_from_sections(sections, mapping)
    transformed = CanonicalData(processor.transform(values, observed), observed, metadata, mapping)
    evaluation = expected.loc[expected.decision_time.dt.normalize().eq(pd.Timestamp("2024-01-03"))].copy()
    return raw, transformed, processor, expected, evaluation, grid, mapping, sections


def _spec(**kwargs):
    defaults = {"name": "B-GRU", "encoder": "ALSTM", "width": 8, "layers": 2,
                "lookback": 4, "heads": 2, "head_width": 8, "dropout": 0.25}
    defaults.update(kwargs)
    return ModelSpec(**defaults)


def _batch(inputs, *, fields=FEATURES, expected=None):
    _, transformed, _, _, evaluation, grid, _, _ = inputs
    table = evaluation if expected is None else expected
    return WindowDataset(transformed, table, grid, length=4, fields=fields).get_batch(list(range(len(table))))


def _save_bundle(path, model, processor, mapping, spec=None, **kwargs):
    arguments = {"model_name": "R1"} if spec is None else {"model_spec": spec.to_dict()}
    save_bundle(path, model, processor, window_length=4, contracts=CONTRACTS, seed=17,
                training_record={"scope": "synthetic interface tests"}, instrument_map=mapping,
                **arguments, **kwargs)


def _checkpoint(path, model, optimizer, processor, spec, **kwargs):
    save_checkpoint(path, model, optimizer, epoch=1, global_step=1, best_metric=0.25,
                    bad_epochs=0, config=CONFIG, preprocessor=processor, model_spec=spec.to_dict(),
                    contracts=CONTRACTS, split=SPLIT, **kwargs)


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_nested_equal(first, second)
    else:
        assert left == right


def _random_draws():
    return random.random(), float(np.random.random()), torch.rand(3)


@pytest.mark.parametrize("name", ["R0", "R1", "R1-Book", "Net64", "Net192", "GroupedNet", "Patch1", "Patch5", "DeepLOB"])
def test_reference_bundle_preserves_weights_and_rng(inputs, tmp_path, name):
    _, _, processor, _, _, _, mapping, _ = inputs
    model = create_model(name).eval()
    path = tmp_path / "legacy"
    save_bundle(path, model, processor, model_name=name, window_length=4, contracts=CONTRACTS, seed=17,
                training_record={"scope": "synthetic interface tests"}, instrument_map=mapping)
    rng = torch.random.get_rng_state().clone()
    restored, loaded_processor, manifest = load_bundle(path)
    assert manifest["bundle_version"] == 3 and "model_spec" not in manifest
    assert set(manifest["files"]) == {"model.json", "preprocessor.json"}
    assert loaded_processor.fingerprint == processor.fingerprint
    _assert_nested_equal(model.state_dict(), restored.state_dict())
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("name", ["B-GRU", "B-Transformer", "B-TCN", "B-SelectiveSSM"])
@pytest.mark.parametrize("weight_format", ["json", "torch"])
def test_modern_bundle_roundtrip_all_families(inputs, tmp_path, name, weight_format):
    _, _, processor, _, evaluation, _, mapping, _ = inputs
    spec = _spec(name=name)
    model = create_score_model(spec).eval()
    path = tmp_path / f"{name}-{weight_format}"
    _save_bundle(path, model, processor, mapping, spec, weight_format=weight_format)
    restored, loaded_processor, manifest = load_bundle(path)
    assert manifest["bundle_version"] == 4 and manifest["model_spec"] == spec.to_dict()
    assert manifest["weight_format"] == weight_format
    assert "source/ptop/temporal.py" in manifest["source_files"]
    assert "source/jyc_e2e/research_metrics.py" in manifest["source_files"]
    assert "source/jyc_e2e/losses.py" in manifest["source_files"]
    assert "licenses/qlib/LICENSE" in manifest["source_files"]
    assert "source/ptop/_vendor/qlib/models.py" in manifest["source_files"]
    assert "source/ptop/_vendor/qlib/processors.py" in manifest["source_files"]
    assert "source/ptop/_vendor/qlib/sampler.py" in manifest["source_files"]
    assert "source/ptop/_vendor/qlib/LICENSE" in manifest["source_files"]
    assert "pyqlib" not in manifest["dependencies"]
    assert "pyqlib" not in (path / "requirements.txt").read_text()
    assert (path / "THIRD_PARTY.md").is_file() and (path / "environment.json").is_file()
    assert (path / "requirements.txt").is_file()
    assert loaded_processor.fingerprint == processor.fingerprint
    _assert_nested_equal(model.state_dict(), restored.state_dict())
    first = predict_windows(model, [_batch(inputs)], evaluation, start_date="2024-01-03", end_date="2024-01-03")
    second = predict_windows(restored, [_batch(inputs)], evaluation, start_date="2024-01-03", end_date="2024-01-03")
    pd.testing.assert_frame_equal(first.records, second.records)


@pytest.mark.parametrize("file", ["model.pt", "preprocessor.json", "source/ptop/temporal.py", "requirements.txt", "licenses/qlib/LICENSE", "source/ptop/_vendor/qlib/processors.py"])
def test_bundle_modified_file_rejected(inputs, tmp_path, file):
    _, _, processor, _, _, _, mapping, _ = inputs
    spec = _spec()
    path = tmp_path / "bundle"
    _save_bundle(path, create_score_model(spec), processor, mapping, spec, weight_format="torch")
    with (path / file).open("ab") as stream:
        stream.write(b"\nmodified\n")
    with pytest.raises(ContractError, match="校验失败"):
        load_bundle(path)


@pytest.mark.parametrize("change", ["schema", "dependencies", "missing_spec_field", "window", "unknown_manifest_field"])
def test_bundle_manifest_contracts_rejected(inputs, tmp_path, change):
    _, _, processor, _, _, _, mapping, _ = inputs
    spec = _spec()
    path = tmp_path / "bundle"
    _save_bundle(path, create_score_model(spec), processor, mapping, spec, weight_format="torch", include_source=False)
    manifest = json.loads((path / "manifest.json").read_text())
    if change == "schema":
        manifest["fields"].reverse()
    elif change == "dependencies":
        manifest["dependencies"]["torch"] = "wrong-version"
    elif change == "missing_spec_field":
        del manifest["model_spec"]["heads"]
    elif change == "window":
        manifest["window_length"] = 5
    else:
        manifest["unrecognized"] = True
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError):
        load_bundle(path)


def test_bundle_checks_semantics_before_creating_destination_and_never_overwrites(inputs, tmp_path):
    _, _, processor, _, _, _, mapping, _ = inputs
    spec = _spec()
    model = create_score_model(spec)
    path = tmp_path / "bundle"
    with pytest.raises(ContractError, match="规格|模块配置"):
        _save_bundle(path, model, processor, mapping, replace(spec, dropout=0.5), weight_format="torch")
    assert not path.exists() and list(tmp_path.iterdir()) == []
    _save_bundle(path, model, processor, mapping, spec, weight_format="torch")
    before = (path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        _save_bundle(path, model, processor, mapping, spec, weight_format="torch")
    assert (path / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("change", ["pooling", "causal"])
def test_bundle_rejects_equal_shaped_weights_with_different_behavior(inputs, tmp_path, change):
    _, _, processor, _, _, _, mapping, _ = inputs
    spec = _spec(name="B-Transformer")
    model = create_score_model(spec)
    different = replace(spec, pooling="mean") if change == "pooling" else replace(spec, causal=False)
    path = tmp_path / "different"
    with pytest.raises(ContractError, match="规格|模块配置"):
        _save_bundle(path, model, processor, mapping, different, weight_format="torch")
    assert not path.exists()


def test_bundle_and_checkpoint_reject_changed_execution_implementation(inputs, tmp_path, monkeypatch):
    import ptop.bundle as bundle_module
    import ptop.checkpoint as checkpoint_module

    _, _, processor, _, _, _, mapping, _ = inputs
    spec = _spec()
    model, optimizer, _, _ = _training_objects(spec)
    bundle_path, checkpoint_path = tmp_path / "bundle", tmp_path / "last.pt"
    _save_bundle(bundle_path, model, processor, mapping, spec, weight_format="torch")
    _checkpoint(checkpoint_path, model, optimizer, processor, spec)
    changed = copy.deepcopy(bundle_module.execution_signature())
    changed["files"]["ptop/backbones.py"] = "0" * 64
    changed["fingerprint"] = "0" * 64
    monkeypatch.setattr(bundle_module, "execution_signature", lambda: changed)
    monkeypatch.setattr(checkpoint_module, "execution_signature", lambda: changed)
    with pytest.raises(ContractError, match="执行源码"):
        load_bundle(bundle_path)
    with pytest.raises(ContractError, match="执行源码"):
        load_checkpoint(checkpoint_path, model, optimizer)
    with pytest.raises(ContractError, match="执行源码"):
        load_inference_checkpoint(checkpoint_path)


def test_portable_bundle_uses_embedded_qlib_without_distribution_metadata(inputs, tmp_path, monkeypatch):
    import ptop.bundle as bundle_module

    original = bundle_module.distribution

    def installed_distribution(name):
        assert name != "pyqlib", "内置组件不得读取外部 Qlib 的分发元数据"
        return original(name)

    monkeypatch.setattr(bundle_module, "distribution", installed_distribution)
    _, _, processor, _, _, _, mapping, _ = inputs
    spec = _spec()
    path = tmp_path / "bundle"
    _save_bundle(path, create_score_model(spec), processor, mapping, spec, weight_format="torch")
    assert (path / "licenses/qlib/LICENSE").read_bytes() == (path / "source/ptop/_vendor/qlib/LICENSE").read_bytes()
    assert "Embedded Qlib components" in (path / "THIRD_PARTY.md").read_text()
    load_bundle(path)


def test_portable_source_reproduces_scores_in_fresh_interpreter(inputs, tmp_path):
    _, _, processor, _, _, _, mapping, _ = inputs
    spec = _spec(name="B-Transformer")
    model = create_score_model(spec).eval()
    path = tmp_path / "bundle"
    _save_bundle(path, model, processor, mapping, spec, weight_format="torch")
    batch = _batch(inputs)
    positions = batch.predictable
    arrays = tmp_path / "inputs.npz"
    np.savez(arrays, x=batch.x[positions], observed=batch.observed[positions], step_valid=batch.step_valid[positions])
    scores_file = tmp_path / "scores.npy"
    script = '''
import sys
from pathlib import Path
import numpy as np
import torch
import ptop.bundle
import jyc_e2e.research_metrics
import jyc_e2e.losses
from ptop.bundle import load_bundle
torch.set_num_threads(2)
bundle = Path(sys.argv[1])
assert Path(ptop.bundle.__file__).resolve() == (bundle / 'source/ptop/bundle.py').resolve()
assert Path(jyc_e2e.losses.__file__).resolve() == (bundle / 'source/jyc_e2e/losses.py').resolve()
model, processor, manifest = load_bundle(bundle)
arrays = np.load(sys.argv[2], allow_pickle=False)
with torch.inference_mode():
    scores = model(*(torch.as_tensor(arrays[key]) for key in ('x', 'observed', 'step_valid')))
np.save(sys.argv[3], scores.numpy())
'''
    environment = dict(os.environ, PYTHONPATH=str(path / "source"), CUDA_VISIBLE_DEVICES="",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    result = subprocess.run([sys.executable, "-c", script, str(path), str(arrays), str(scores_file)],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    with torch.inference_mode():
        expected = model(*(torch.as_tensor(value[positions]) for value in (batch.x, batch.observed, batch.step_valid)))
    np.testing.assert_allclose(np.load(scores_file), expected.numpy(), atol=2e-6, rtol=2e-6)


def _training_update(model, optimizer, scheduler, scaler, batch):
    model.train()
    selected = batch.predictable
    draws = _random_draws()
    target = torch.tensor([draws[0], draws[1], *draws[2].tolist()][:int(selected.sum())], dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    scores = model(*(torch.as_tensor(value[selected]) for value in (batch.x, batch.observed, batch.step_valid)))
    loss = torch.nn.functional.mse_loss(scores, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return scores.detach().clone(), loss.detach().clone()


def _training_objects(spec):
    model = create_score_model(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    scaler = torch.amp.GradScaler("cpu", init_scale=128, growth_interval=1)
    return model, optimizer, scheduler, scaler


def test_checkpoint_reproduces_next_update_rng_scheduler_amp_and_sampler(inputs, tmp_path):
    _, _, processor, _, _, _, _, _ = inputs
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    spec, batch = _spec(), _batch(inputs)
    model, optimizer, scheduler, scaler = _training_objects(spec)
    _training_update(model, optimizer, scheduler, scaler, batch)
    sampler_state = torch.Generator().manual_seed(21).get_state()
    extra = {"sampler_generator": sampler_state, "history": [{"epoch": 1, "loss": 0.25}],
             "objective": {"target_scale": 0.2}, "best_epoch": 1}
    path = tmp_path / "last.pt"
    _checkpoint(path, model, optimizer, processor, spec, scheduler=scheduler, scaler=scaler, extra_state=extra)
    expected_scores, expected_loss = _training_update(model, optimizer, scheduler, scaler, batch)
    expected_state = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler, expected_scaler = scheduler.state_dict(), scaler.state_dict()
    expected_draws = _random_draws()
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    resumed, resumed_optimizer, resumed_scheduler, resumed_scaler = _training_objects(spec)
    state = load_checkpoint(path, resumed, resumed_optimizer, scheduler=resumed_scheduler, scaler=resumed_scaler,
                            expected_config=CONFIG, expected_model_spec=spec, expected_contracts=CONTRACTS,
                            expected_split=SPLIT, expected_preprocessor=processor)
    assert state.epoch == state.global_step == 1 and state.bad_epochs == 0 and state.best_metric == 0.25
    assert state.preprocessor.fingerprint == processor.fingerprint
    _assert_nested_equal(extra, state.extra_state)
    scores, loss = _training_update(resumed, resumed_optimizer, resumed_scheduler, resumed_scaler, batch)
    torch.testing.assert_close(scores, expected_scores, atol=0, rtol=0)
    torch.testing.assert_close(loss, expected_loss, atol=0, rtol=0)
    _assert_nested_equal(expected_state, resumed.state_dict())
    _assert_nested_equal(expected_optimizer, resumed_optimizer.state_dict())
    _assert_nested_equal(expected_scheduler, resumed_scheduler.state_dict())
    _assert_nested_equal(expected_scaler, resumed_scaler.state_dict())
    actual_draws = _random_draws()
    assert actual_draws[:2] == expected_draws[:2]
    torch.testing.assert_close(actual_draws[2], expected_draws[2], atol=0, rtol=0)


@pytest.mark.parametrize("change", ["config", "split", "contracts", "spec", "optimizer_order", "scheduler", "scaler"])
def test_checkpoint_mismatch_fails_before_model_or_rng_mutation(inputs, tmp_path, change):
    _, _, processor, _, _, _, _, _ = inputs
    spec = _spec()
    model, optimizer, scheduler, scaler = _training_objects(spec)
    path = tmp_path / "last.pt"
    _checkpoint(path, model, optimizer, processor, spec, scheduler=scheduler, scaler=scaler)
    restored, restored_optimizer, restored_scheduler, restored_scaler = _training_objects(spec)
    kwargs = {"scheduler": restored_scheduler, "scaler": restored_scaler, "expected_config": CONFIG,
              "expected_model_spec": spec, "expected_contracts": CONTRACTS, "expected_split": SPLIT}
    if change == "config":
        kwargs["expected_config"] = {**CONFIG, "max_epochs": 5}
    elif change == "split":
        kwargs["expected_split"] = {**SPLIT, "train_start": "2024-01-01"}
    elif change == "contracts":
        kwargs["expected_contracts"] = {**CONTRACTS, "source": "different"}
    elif change == "spec":
        kwargs["expected_model_spec"] = replace(spec, pooling="mean")
    elif change == "optimizer_order":
        restored_optimizer = torch.optim.AdamW(list(restored.parameters())[::-1], lr=1e-3)
    else:
        kwargs[change] = None
    before = copy.deepcopy(restored.state_dict())
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ContractError):
        load_checkpoint(path, restored, restored_optimizer, **kwargs)
    _assert_nested_equal(before, restored.state_dict())
    assert torch.equal(rng, torch.random.get_rng_state())


def test_checkpoint_corruption_and_unsafe_python_object_rejected(inputs, tmp_path):
    _, _, processor, _, _, _, _, _ = inputs
    spec = _spec()
    model, optimizer, _, _ = _training_objects(spec)
    path = tmp_path / "last.pt"
    _checkpoint(path, model, optimizer, processor, spec)
    payload = torch.load(path, weights_only=True)
    first = next(iter(payload["model"].values()))
    first.reshape(-1)[0] += 1
    torch.save(payload, path)
    with pytest.raises(ContractError, match="完整性校验"):
        load_checkpoint(path, model, optimizer)
    torch.save({"unsupported": _UnsupportedCheckpointObject()}, path)
    rng = random.getstate()
    with pytest.raises(ContractError, match="安全读取"):
        load_checkpoint(path, model, optimizer)
    assert random.getstate() == rng


class _UnsupportedCheckpointObject:
    def __reduce__(self):
        # weights_only must reject this global instead of executing it.
        return random.seed, (999,)


def test_inference_checkpoint_does_not_restore_training_optimizer_or_rng(inputs, tmp_path):
    _, _, processor, _, _, _, _, _ = inputs
    spec = _spec()
    model, optimizer, scheduler, scaler = _training_objects(spec)
    _training_update(model, optimizer, scheduler, scaler, _batch(inputs))
    path = tmp_path / "last.pt"
    _checkpoint(path, model, optimizer, processor, spec, scheduler=scheduler, scaler=scaler)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    torch_rng, python_rng, numpy_rng = torch.random.get_rng_state().clone(), random.getstate(), np.random.get_state()
    loaded, loaded_processor, metadata = load_inference_checkpoint(path, expected_contracts=CONTRACTS)
    assert not loaded.training and metadata["epoch"] == 1
    assert loaded_processor.fingerprint == processor.fingerprint
    _assert_nested_equal(model.state_dict(), loaded.state_dict())
    _assert_nested_equal(optimizer_before, optimizer.state_dict())
    assert torch.equal(torch_rng, torch.random.get_rng_state()) and python_rng == random.getstate()
    np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
    assert numpy_rng[0] == np.random.get_state()[0] and numpy_rng[2:] == np.random.get_state()[2:]


@pytest.mark.parametrize("name", ["R1", "Net64", "DeepLOB", "B-Transformer", "B-SelectiveSSM"])
def test_inference_microbatch_reordering_and_expected_keys_preserved(inputs, name):
    _, _, _, _, evaluation, _, _, _ = inputs
    expected = evaluation.iloc[::-1].reset_index(drop=True)
    spec = _spec(name=name) if name.startswith("B-") else ModelSpec(name=name, lookback=4)
    model = create_score_model(spec)
    batch = _batch(inputs, fields=model.encoder.fields, expected=expected)
    before = copy.deepcopy(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    first = predict_windows(model, [batch], expected, start_date="2024-01-03", end_date="2024-01-03",
                            microbatch_size=128)
    dataset = WindowDataset(inputs[1], expected, inputs[5], length=4, fields=model.encoder.fields)
    reordered = [dataset.get_batch([3, 1, 5]), dataset.get_batch([0, 4, 2])]
    second = predict_windows(model, reordered, expected, start_date="2024-01-03", end_date="2024-01-03",
                             microbatch_size=1)
    assert first.records.sample_id.tolist() == expected.sample_id.tolist()
    assert first.records.instrument_id.tolist() == expected.instrument_id.tolist()
    pd.testing.assert_frame_equal(first.records.drop(columns="score"), second.records.drop(columns="score"))
    np.testing.assert_allclose(first.output.score, second.output.score, atol=2e-6, rtol=2e-6)
    assert list(first.output.columns) == ["date", "instrument", "score"]
    assert first.coverage.expected.eq(3).all() and first.coverage.predicted.eq(2).all()
    assert first.coverage.excluded.eq(1).all() and first.coverage.output_predicted.eq(2).all()
    assert model.training
    assert torch.equal(rng, torch.random.get_rng_state())
    _assert_nested_equal(before, model.state_dict())


def test_real_missing_minute_and_zero_fallback_are_explicit(inputs):
    model = create_score_model(_spec())
    batch, evaluation = _batch(inputs), inputs[4]
    position = batch.metadata.loc[batch.metadata.instrument_id.eq(11)].index[0]
    assert batch.step_valid[position, -1] and not batch.observed[position, -1].any()
    result = predict_windows(model, [batch], evaluation, start_date="2024-01-03", end_date="2024-01-03",
                             unpredictable_policy="score0")
    assert len(result.output) == len(evaluation) and result.excluded.empty
    assert len(result.fallback) == 2 and result.fallback.score.eq(0).all()
    assert result.coverage.predicted.eq(2).all() and result.coverage.output_predicted.eq(3).all()
    assert result.coverage.fallback_rate.eq(1 / 3).all()
    assert result.fallback.reason.eq("no_observation_in_model_fields").all()


@pytest.mark.parametrize("policy", ["drop", "score0"])
@pytest.mark.parametrize("whole_timepoint", [False, True])
def test_fallback_never_hides_insufficient_model_coverage(inputs, policy, whole_timepoint):
    batch, evaluation = _batch(inputs), inputs[4]
    missing = (batch.metadata.decision_time.eq(batch.metadata.decision_time.iloc[0]).to_numpy()
               if whole_timepoint else batch.metadata.instrument_id.eq(11).to_numpy())
    batch.x[missing] = 0
    batch.observed[missing] = False
    batch.predictable[missing] = False
    with pytest.raises(ContractError, match="逐时点覆盖不合格"):
        predict_windows(create_score_model(_spec()), [batch], evaluation,
                        start_date="2024-01-03", end_date="2024-01-03", unpredictable_policy=policy)


@pytest.mark.parametrize("change", ["predictable", "stock_key", "future_time", "cross_day", "row_source", "missing_sample", "duplicate"])
def test_inference_mask_keys_visibility_and_input_completeness_rejected(inputs, change):
    batch, evaluation = _batch(inputs), inputs[4]
    model = create_score_model(_spec(encoder="ALSTM-Book"))
    batches = [batch]
    if change == "predictable":
        indices = [FEATURES.index(field) for field in BOOK12_FIELDS]
        batch.observed[0, :, indices] = False
        batch.x[0, :, indices] = 0
    elif change == "stock_key":
        batch.metadata.loc[0, "instrument"] = "ANOTHER_CODE"
    elif change == "future_time":
        batch.bar_times[0, -1] += np.timedelta64(1, "m")
    elif change == "cross_day":
        batch.bar_times[0, -2] -= np.timedelta64(1, "D")
    elif change == "row_source":
        batch.row_positions[1, -1] = -1
    elif change == "missing_sample":
        batches = [WindowDataset(inputs[1], evaluation, inputs[5], length=4).get_batch([0, 1, 2])]
    else:
        batches = [batch, batch]
    with pytest.raises(ContractError):
        predict_windows(model, batches, evaluation, start_date="2024-01-03", end_date="2024-01-03")


def test_nonfinite_or_wrong_shape_model_scores_fail(inputs, monkeypatch):
    model = create_score_model(_spec())
    batch, evaluation = _batch(inputs), inputs[4]
    monkeypatch.setattr(model, "forward", lambda x, *args: torch.full((len(x),), float("nan")))
    with pytest.raises(ContractError, match="一维浮点分数"):
        predict_windows(model, [batch], evaluation, start_date="2024-01-03", end_date="2024-01-03")
    monkeypatch.setattr(model, "forward", lambda x, *args: torch.zeros((len(x), 1)))
    with pytest.raises(ContractError, match="一维浮点分数"):
        predict_windows(model, [batch], evaluation, start_date="2024-01-03", end_date="2024-01-03")


def test_canonical_inference_never_refits_or_consumes_caller_rng(inputs, monkeypatch):
    raw, _, processor, _, evaluation, grid, _, _ = inputs
    model = create_score_model(_spec())
    fingerprint = processor.fingerprint
    monkeypatch.setattr(FrozenPreprocessor, "fit", lambda *args, **kwargs: pytest.fail("inference must not fit"))
    rng = torch.random.get_rng_state().clone()
    result = predict_canonical(model, processor, raw, evaluation, grid,
                               start_date="2024-01-03", end_date="2024-01-03", window_length=4)
    assert len(result.output) == 4 and processor.fingerprint == fingerprint
    assert torch.equal(rng, torch.random.get_rng_state())


def test_external_a_loader_does_not_consume_caller_rng(inputs):
    model = create_score_model(_spec())
    evaluation = inputs[4]
    dataset = WindowDataset(inputs[1], evaluation, inputs[5], length=4)
    loader = DataLoader(dataset, batch_sampler=DecisionBatchSampler(dataset), collate_fn=collate_windows)
    rng = torch.random.get_rng_state().clone()
    result = predict_windows(model, loader, evaluation, start_date="2024-01-03", end_date="2024-01-03")
    assert len(result.output) == 4 and torch.equal(rng, torch.random.get_rng_state())


class _QueryRecorder:
    def __init__(self, inputs):
        raw, _, _, _, _, _, mapping, sections = inputs
        bars = raw.values.reset_index().rename(columns={"datetime": "date", "instrument": "instrument_id"})
        bars["instrument"] = bars.instrument_id.map(mapping)
        bars["available_at"] = raw.metadata.available_at.to_numpy()
        bars["adjust_factor"] = 1.0
        self.frames, self.calls = {"synthetic_bars": bars, SECTION_TABLE: sections}, []

    def __call__(self, sql, *, filters):
        self.calls.append((sql, filters))
        assert LABEL_TABLE not in sql
        table = next(name for name in self.frames if f"FROM {name}" in sql)
        frame = self.frames[table]
        frame = frame.loc[frame.date.between(*map(pd.Timestamp, filters["date"]))].copy()
        if "instrument" in filters:
            frame = frame.loc[frame.instrument.isin(filters["instrument"])]

        class _Result:
            def df(self):
                return frame

        return _Result()


def test_cloud_and_bundle_interfaces_query_only_inputs_and_keep_frozen_mapping(inputs, tmp_path):
    _, _, processor, _, _, grid, mapping, _ = inputs
    spec = _spec()
    model = create_score_model(spec)
    query = _QueryRecorder(inputs)
    direct = predict_cloud(query, {"bar1m": "synthetic_bars"}, model, processor,
                           start_date="2024-01-03", end_date="2024-01-03", history_start="2024-01-03",
                           instrument_map=mapping, trading_grid=grid, bar_end_is_available=True, window_length=4)
    assert len(query.calls) == 2
    path = tmp_path / "bundle"
    _save_bundle(path, model, processor, mapping, spec, weight_format="torch")
    restored = predict_bundle(path, query, {"bar1m": "synthetic_bars"}, start_date="2024-01-03",
                              end_date="2024-01-03", history_start="2024-01-03", trading_grid=grid,
                              bar_end_is_available=True, expected_contracts=CONTRACTS)
    pd.testing.assert_frame_equal(direct.records, restored.records)
    with pytest.raises(ContractError, match="映射不一致"):
        predict_bundle(path, query, {"bar1m": "synthetic_bars"}, start_date="2024-01-03",
                       end_date="2024-01-03", history_start="2024-01-03", trading_grid=grid,
                       bar_end_is_available=True, instrument_map={**mapping, 11: "DIFFERENT_CODE"})
    assert len(query.calls) == 4


def test_partition_stream_is_lazy_and_reuses_processor_and_weights(inputs, tmp_path):
    _, _, processor, _, _, grid, mapping, _ = inputs
    spec = _spec()
    model = create_score_model(spec)
    query = _QueryRecorder(inputs)
    partitions = [InferencePartition(day, day, day, grid) for day in ("2024-01-02", "2024-01-03")]
    iterator = predict_cloud_partitions(query, {"bar1m": "synthetic_bars"}, model, processor,
                                       partitions=partitions, instrument_map=mapping,
                                       bar_end_is_available=True, window_length=4)
    fingerprint, before = processor.fingerprint, copy.deepcopy(model.state_dict())
    assert len(query.calls) == 0
    first = next(iterator)
    assert len(query.calls) == 2 and first.output.date.dt.normalize().eq(pd.Timestamp("2024-01-02")).all()
    second = next(iterator)
    assert len(query.calls) == 4 and second.output.date.dt.normalize().eq(pd.Timestamp("2024-01-03")).all()
    with pytest.raises(StopIteration):
        next(iterator)
    assert processor.fingerprint == fingerprint
    _assert_nested_equal(before, model.state_dict())
    path = tmp_path / "bundle"
    _save_bundle(path, model, processor, mapping, spec, weight_format="torch")
    restored = list(predict_bundle_partitions(path, query, {"bar1m": "synthetic_bars"}, partitions=partitions,
                                             bar_end_is_available=True))
    pd.testing.assert_frame_equal(first.records, restored[0].records)
    pd.testing.assert_frame_equal(second.records, restored[1].records)
    invalid = predict_cloud_partitions(query, {"bar1m": "synthetic_bars"}, model, processor,
                                      partitions=[partitions[0], partitions[0]], instrument_map=mapping,
                                      bar_end_is_available=True, window_length=4)
    next(invalid)
    with pytest.raises(ContractError, match="不重叠"):
        next(invalid)
