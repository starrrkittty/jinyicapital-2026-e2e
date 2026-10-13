"""Reproducible .py training entry point using explicit A/C data contracts."""

import argparse
import importlib
import json
from pathlib import Path

import pandas as pd
import torch

from .schema import ContractError
from .targets import TimeFold
from .temporal import ModelSpec, create_score_model
from .training import Trainer, seed_everything
from .training_config import TrainingConfig


def run(config: dict, *, output_dir, device="cpu", resume=None, stop_after_epochs=None):
    required = {"model", "training", "fold", "contracts", "scope", "data_factory", "data"}
    optional = {"objective_factory", "evaluator_factory"}
    if not isinstance(config, dict) or not required.issubset(config) or set(config) - required - optional:
        raise ContractError(f"run configuration needs {sorted(required)}; optional {sorted(optional)}")
    spec = ModelSpec.from_dict(config["model"])
    training = TrainingConfig.from_dict(config["training"])
    fold_config = config["fold"]
    if not isinstance(fold_config, dict) or set(fold_config) != {"name", "train_start", "validation_start", "validation_end"}:
        raise ContractError("fold requires name and the three explicit time boundaries")
    fold = TimeFold(fold_config["name"], *(pd.Timestamp(fold_config[key]) for key in
                    ("train_start", "validation_start", "validation_end")))
    if not isinstance(config["data"], dict):
        raise ContractError("data factory parameters must be an object")

    def resolve(reference):
        if not isinstance(reference, str) or reference.count(":") != 1:
            raise ContractError("factory must be an importable module:function")
        module_name, attribute = reference.split(":")
        factory = getattr(importlib.import_module(module_name), attribute, None)
        if not callable(factory):
            raise ContractError(f"factory is not callable: {reference}")
        return factory

    seed_everything(training.seed, deterministic=training.deterministic)
    prepared, instrument_map = resolve(config["data_factory"])(config=config["data"], model_spec=spec, fold=fold)
    # Data preparation must not change initialization for the same declared seed.
    seed_everything(training.seed, deterministic=training.deterministic)
    model = create_score_model(spec)
    hooks = {name: resolve(config[name + "_factory"])(config=config) for name in ("objective", "evaluator")
             if config.get(name + "_factory")}
    trainer = Trainer(model, training, output_dir=output_dir, model_spec=spec, contracts=config["contracts"],
                      fold=fold, instrument_map=instrument_map, device=device, scope=config["scope"], **hooks)
    return trainer.fit(prepared, resume=resume, stop_after_epochs=stop_after_epochs)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train B models against A's prepared fold and explicit C labels.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epochs", type=int)
    parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args(argv)
    if arguments.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(arguments.threads)
    try:
        config = json.loads(arguments.config.read_text(encoding="utf-8"))
        result = run(config, output_dir=arguments.output, device=arguments.device,
                     resume=arguments.resume, stop_after_epochs=arguments.stop_after_epochs)
    except (ContractError, ValueError, ImportError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"best_checkpoint": str(result.best_checkpoint), "best_epoch": result.best_epoch,
                      "best_metric": result.best_metric, "completed_epochs": result.completed_epochs,
                      "bundle_path": str(result.bundle_path) if result.bundle_path else None,
                      "stopped_reason": result.stopped_reason}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
