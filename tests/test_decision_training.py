"""Full-decision objectives retain cross-stock gradients across forward chunks."""

from dataclasses import replace
import os

import numpy as np
import pytest
import torch

from test_training import prepared_fold, make_trainer, assert_nested_equal
from ptop.schema import ContractError
from ptop.training import _target_table
from ptop.training_config import TrainingConfig


class CenteredObjective(torch.nn.Module):
    requires_full_decision = True

    def __init__(self):
        super().__init__()
        self.counts = []

    def configuration(self):
        return {"name": "synthetic-centered-test"}

    def forward(self, scores, target, decisions, valid):
        scores, target = scores[valid].float(), target[valid].float()
        self.counts.append(len(scores))
        loss = ((scores - scores.mean()) - (target - target.mean())).square().mean()
        return loss + 0.1 * scores.square().mean(), {"count": len(scores)}


class Pointwise(torch.nn.Module):
    def __init__(self, fields):
        super().__init__()
        self.encoder = torch.nn.Identity()
        self.encoder.fields = fields
        self.weight = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, x, observed, step_valid):
        return self.weight * x[:, -1].mean(dim=-1)


def setup_phase(tmp_path, prepared_fold, *, microbatch_size=2, scope="decision"):
    prepared, mapping, fold = prepared_fold
    config = TrainingConfig(epochs=2, patience=3, selection_metric="loss", mode="min",
                            seed=19, microbatch_size=microbatch_size, objective_scope=scope)
    objective = CenteredObjective()
    trainer = make_trainer(tmp_path, mapping, fold, config=config, objective=objective)
    trainer.model = Pointwise(trainer.model.encoder.fields)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    tmp_path.mkdir(parents=True, exist_ok=True)
    table = _target_table(prepared.train_targets, "train", fold).set_index("sample_id", drop=False)
    return trainer, prepared, table, objective


def test_full_objective_crosses_forward_chunks_and_steps_once(tmp_path, prepared_fold):
    trainer, prepared, table, objective = setup_phase(tmp_path, prepared_fold)
    stats, _ = trainer._phase(prepared.train_loader, table, training=True, epoch=0)
    assert objective.counts == [3, 3]
    assert stats["optimizer_steps"] == 2
    assert stats["microbatches"] == 4
    assert stats["loss_reduction"] == "decision"


@pytest.mark.parametrize("mixed", [False, True])
def test_incomplete_or_mixed_decision_rejected_before_update(tmp_path, prepared_fold, mixed):
    trainer, prepared, table, _ = setup_phase(tmp_path, prepared_fold)
    from ptop.windows import collate_windows
    dataset = prepared.train_loader.dataset
    groups = list(dataset.expected.groupby("decision_id", sort=False).indices.values())
    positions = np.r_[groups[0][:2], groups[1][:1]] if mixed else groups[0][:2]
    batch = collate_windows([dataset[int(index)] for index in positions])
    with pytest.raises(ContractError, match="complete decision|one decision"):
        trainer._phase([batch], table, training=True, epoch=0)
    assert trainer.global_step == 0


@pytest.mark.parametrize("omitted_instrument", [4, 5])
def test_complete_contract_includes_unlabeled_and_unpredictable_rows(tmp_path, prepared_fold, omitted_instrument):
    trainer, prepared, table, objective = setup_phase(tmp_path, prepared_fold)
    from ptop.windows import collate_windows
    dataset = prepared.train_loader.dataset
    group = next(iter(dataset.expected.groupby("decision_id", sort=False).indices.values()))
    positions = [index for index in group if dataset.expected.iloc[index].instrument_id != omitted_instrument]
    batch = collate_windows([dataset[int(index)] for index in positions])
    with pytest.raises(ContractError, match="complete decision"):
        trainer._phase([batch], table, training=True, epoch=0)
    assert trainer.global_step == 0 and objective.counts == []


def test_same_count_cannot_replace_a_sample_from_another_decision(tmp_path, prepared_fold):
    trainer, prepared, table, objective = setup_phase(tmp_path, prepared_fold)
    from ptop.windows import collate_windows
    dataset = prepared.train_loader.dataset
    groups = list(dataset.expected.groupby("decision_id", sort=False).indices.values())
    positions = np.r_[groups[0][:-1], groups[1][:1]]
    batch = collate_windows([dataset[int(index)] for index in positions])
    # Forge a single decision ID while preserving the foreign sample's true
    # window timestamp; _align must check against the authoritative target row.
    batch.metadata["decision_id"] = dataset.expected.iloc[groups[0][0]].decision_id
    with pytest.raises(ContractError, match="label alignment conflicts"):
        trainer._phase([batch], table, training=True, epoch=0)
    assert trainer.global_step == 0 and objective.counts == []


def test_custom_full_decision_objective_rejects_microbatch_scope(tmp_path, prepared_fold):
    trainer, prepared, table, _ = setup_phase(tmp_path, prepared_fold, scope="microbatch")
    with pytest.raises(ContractError, match="decision"):
        trainer._phase(prepared.train_loader, table, training=True, epoch=0)
    assert trainer.global_step == 0


def test_validation_preserves_unlabeled_scores_and_full_loss(tmp_path, prepared_fold):
    trainer, prepared, _, objective = setup_phase(tmp_path, prepared_fold)
    table = _target_table(prepared.validation_targets, "validation", trainer.fold).set_index("sample_id", drop=False)
    stats, predictions = trainer._phase(prepared.validation_loader, table, training=False, epoch=0)
    assert predictions.loc[predictions.instrument_id.eq(4), "score"].notna().all()
    assert predictions.loc[predictions.instrument_id.eq(5), "score"].isna().all()
    assert objective.counts == [3, 3]
    assert stats["optimizer_steps"] == 0
    assert stats["microbatches"] == 4


def test_decision_loss_uses_equal_time_weight_with_unequal_supervision(tmp_path, prepared_fold):
    trainer, prepared, _, _ = setup_phase(tmp_path, prepared_fold)
    table = _target_table(prepared.validation_targets, "validation", trainer.fold).set_index("sample_id", drop=False)
    first = table.decision_id.iloc[0]
    invalid = table.decision_id.eq(first) & table.instrument_id.eq(1)
    table.loc[invalid, "target_valid"] = False
    table.loc[invalid, "target"] = np.nan
    table.loc[table.decision_id.eq(first) & table.target_valid, "target"] = 2.0
    table.loc[table.decision_id.ne(first) & table.target_valid, "target"] = 4.0

    def synthetic_loss(scores, target, decisions, valid):
        return scores.sum() * 0 + target[valid].mean(), {}

    trainer.objective = synthetic_loss
    stats, _ = trainer._phase(prepared.validation_loader, table, training=False, epoch=0)
    assert stats["loss"] == pytest.approx(3.0)
    assert stats["count"] == 5


def test_deterministic_pointwise_chunks_preserve_loss_gradients_and_weights(tmp_path, prepared_fold):
    first, prepared, table, _ = setup_phase(tmp_path / "split", prepared_fold, microbatch_size=2)
    second, _, _, _ = setup_phase(tmp_path / "whole", prepared_fold, microbatch_size=100)
    # Reuse the same materialized decisions and reset shuffle RNG for each run.
    loader = list(prepared.train_loader)
    torch.manual_seed(41)
    first_stats, _ = first._phase(loader, table, training=True, epoch=0)
    torch.manual_seed(41)
    second_stats, _ = second._phase(loader, table, training=True, epoch=0)
    assert first_stats["loss"] == pytest.approx(second_stats["loss"], rel=1e-6)
    torch.testing.assert_close(first.model.weight, second.model.weight)
    torch.testing.assert_close(first.model.weight.grad, second.model.weight.grad)


def test_decision_mode_epoch_resume_matches_continuous(tmp_path, prepared_fold):
    prepared, mapping, fold = prepared_fold
    config = TrainingConfig(epochs=2, patience=3, selection_metric="loss", mode="min",
                            seed=19, microbatch_size=2, objective_scope="decision")
    full = make_trainer(tmp_path / "full", mapping, fold, config=config, objective=CenteredObjective())
    full.fit(prepared)
    partial = make_trainer(tmp_path / "resume", mapping, fold, config=config, objective=CenteredObjective())
    result = partial.fit(prepared, stop_after_epochs=1)
    resumed = make_trainer(tmp_path / "resume", mapping, fold, config=config, objective=CenteredObjective())
    resumed.fit(prepared, resume=result.last_checkpoint)
    assert_nested_equal(full.model.state_dict(), resumed.model.state_dict())
    assert_nested_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_nested_equal(full.scheduler.state_dict(), resumed.scheduler.state_dict())


def test_configured_pearson_objective_uses_complete_decisions(tmp_path, prepared_fold):
    prepared, mapping, fold = prepared_fold
    config = TrainingConfig(epochs=1, patience=2, selection_metric="loss", mode="min",
                            seed=19, microbatch_size=2, objective_scope="decision", ic_weight=0.2)
    trainer = make_trainer(tmp_path, mapping, fold, config=config)
    result = trainer.fit(prepared)
    assert trainer.objective.requires_full_decision
    assert trainer.objective.configuration()["ic_weight"] == 0.2
    assert result.history[0]["train"]["optimizer_steps"] == 2
    assert result.history[0]["validation"]["loss_reduction"] == "decision"


@pytest.mark.parametrize("device,precision", [("cpu", "bf16"), ("cuda:0", "fp16"), ("cuda:0", "bf16")])
def test_mixed_objective_amp_keeps_joint_gradients(device, precision, tmp_path, prepared_fold):
    if device.startswith("cuda"):
        if os.environ.get("PTOP_TEST_CUDA") != "1" or not torch.cuda.is_available():
            pytest.skip("CUDA requires PTOP_TEST_CUDA=1 and an available isolated device")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            pytest.skip("CUDA device does not support bf16")
    prepared, mapping, fold = prepared_fold
    config = TrainingConfig(epochs=1, patience=2, selection_metric="loss", mode="min", seed=19,
                            precision=precision, microbatch_size=2, objective_scope="decision",
                            ic_weight=0.02, rank_weight=0.03, rank_weighting="target_gap", amp_initial_scale=1.0)
    trainer = make_trainer(tmp_path, mapping, fold, config=config, device=device)
    before = {name: value.detach().clone() for name, value in trainer.model.named_parameters()}
    result = trainer.fit(prepared)
    assert np.isfinite(result.best_metric)
    assert result.history[0]["train"]["optimizer_steps"] == 2
    for component in ("encoder", "backbone", "head"):
        parameters = [(name, value) for name, value in trainer.model.named_parameters()
                      if name.startswith(component + ".") and value.requires_grad]
        assert parameters
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for _, value in parameters)
        assert any(not torch.equal(before[name], value) for name, value in parameters)


@pytest.mark.parametrize("changed", [{"ic_weight": 0.04}, {"rank_temperature": 0.5}])
def test_changed_objective_contract_rejects_resume(changed, tmp_path, prepared_fold):
    prepared, mapping, fold = prepared_fold
    config = TrainingConfig(epochs=2, patience=3, selection_metric="loss", mode="min", seed=19,
                            microbatch_size=2, objective_scope="decision", ic_weight=0.02,
                            rank_weight=0.03, rank_weighting="target_gap")
    trainer = make_trainer(tmp_path, mapping, fold, config=config)
    stopped = trainer.fit(prepared, stop_after_epochs=1)
    resumed = make_trainer(tmp_path, mapping, fold, config=replace(config, **changed))
    with pytest.raises(ContractError, match="配置|config"):
        resumed.fit(prepared, resume=stopped.last_checkpoint)
