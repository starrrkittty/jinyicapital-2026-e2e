"""Opt-in isolated-device AMP and complete GPU restart checks."""

import os

import numpy as np
import pytest
import torch

from test_training import assert_fp32_validation_replay, assert_nested_equal, make_trainer, prepared_fold
from ptop.checkpoint import load_inference_checkpoint
from ptop.temporal import ModelSpec
from ptop.training_config import TrainingConfig


pytestmark = pytest.mark.skipif(os.environ.get("PTOP_TEST_CUDA") != "1",
                               reason="GPU execution requires explicit isolated-device opt-in")


@pytest.mark.parametrize("name", ("B-GRU", "B-Transformer", "B-TCN", "B-SelectiveSSM"))
@pytest.mark.parametrize("precision", ("fp16", "bf16"))
def test_complete_cuda_amp_train_checkpoint_and_export(name, precision, prepared_fold, tmp_path):
    assert torch.cuda.is_available()
    prepared, mapping, fold = prepared_fold
    spec = ModelSpec(name=name, lookback=4, width=8, layers=1, head_width=8)
    config = TrainingConfig(epochs=1, patience=2, selection_metric="loss", mode="min", seed=19,
                            microbatch_size=3, precision=precision)
    trainer = make_trainer(tmp_path / (name + precision), mapping, fold, config=config,
                           model_spec=spec, device="cuda:0")
    result = trainer.fit(prepared)
    assert result.completed_epochs == 1 and np.isfinite(result.best_metric) and result.bundle_path.is_dir()
    assert result.history[0]["train"]["peak_cuda_allocated_bytes"] > 0
    restored, _, _ = load_inference_checkpoint(result.best_checkpoint, device="cuda:0")
    batch = next(iter(prepared.validation_loader))
    inputs = [torch.as_tensor(value[batch.predictable], device="cuda:0")
              for value in (batch.x, batch.observed, batch.step_valid)]
    with torch.inference_mode():
        assert torch.isfinite(restored(*inputs)).all()
    assert_fp32_validation_replay(result, prepared, device="cuda:0")


@pytest.mark.parametrize("layers", (1, 2))
def test_cuda_dropout_resume_restores_optimizer_scaler_and_rng(layers, prepared_fold, tmp_path):
    assert torch.cuda.is_available()
    prepared, mapping, fold = prepared_fold
    config = TrainingConfig(epochs=3, patience=5, selection_metric="loss", mode="min", seed=19,
                            microbatch_size=2, precision="fp16")
    spec = ModelSpec(name="B-GRU", lookback=4, width=8, layers=layers, head_width=8, dropout=0.1)
    full = make_trainer(tmp_path / "full", mapping, fold, config=config, model_spec=spec, device="cuda:0")
    full.fit(prepared)
    partial = make_trainer(tmp_path / "restart", mapping, fold, config=config, model_spec=spec, device="cuda:0")
    stopped = partial.fit(prepared, stop_after_epochs=1)
    restarted = make_trainer(tmp_path / "restart", mapping, fold, config=config, model_spec=spec, device="cuda:0")
    restarted.fit(prepared, resume=stopped.last_checkpoint)
    assert_nested_equal(full.model.state_dict(), restarted.model.state_dict())
    assert_nested_equal(full.optimizer.state_dict(), restarted.optimizer.state_dict())
    assert_nested_equal(full.scheduler.state_dict(), restarted.scheduler.state_dict())
    assert_nested_equal(full.scaler.state_dict(), restarted.scaler.state_dict())


@pytest.mark.parametrize("initial_scale", (1024.0, 1.0))
def test_fp16_overflow_skips_update_lowers_scale_and_remains_trainable(initial_scale, prepared_fold, tmp_path):
    prepared, mapping, fold = prepared_fold
    config = TrainingConfig(epochs=1, patience=2, selection_metric="loss", mode="min", seed=19,
                            microbatch_size=2, precision="fp16", amp_initial_scale=initial_scale)
    trainer = make_trainer(tmp_path / "overflow", mapping, fold, config=config, device="cuda:0")
    attempts = []
    def overflow_once(gradient):
        attempts.append(True)
        return torch.full_like(gradient, float("inf")) if len(attempts) == 1 else gradient
    handle = next(trainer.model.encoder.parameters()).register_hook(overflow_once)
    try:
        result = trainer.fit(prepared)
    finally:
        handle.remove()
    stats = result.history[0]["train"]
    assert stats["amp_skipped_steps"] == 1
    assert stats["optimizer_steps"] == stats["microbatches"] - 1 == trainer.global_step
    assert trainer.scaler.get_scale() == config.amp_initial_scale / 2
