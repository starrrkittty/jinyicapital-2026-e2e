"""Run bounded local engineering diagnostics or explicitly contracted training.

Examples (outputs and raw cache stay on the private data disk)::

    python examples/b_real_data_smoke.py --config configs/b_diagnostic_cpu.json \
        --output-dir /home/data4/hts/jyc2026/b_runs/diagnostic-v1 --mode matrix

Artificial targets in diagnostic mode never become official labels or bundles.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import torch

from ptop.experiments import ExperimentPlan, benchmark_windows, run_experiments
from ptop.local_training import LocalBarStore, ReadPolicy, _write_json, prepare_diagnostic_fold, prepare_local_fold
from ptop.schema import ContractError
from ptop.targets import LabelSpec, TimeFold


def _table(path: str | None, name: str) -> pd.DataFrame:
    if not path:
        raise ContractError(f"正式训练未提供 {name}；请接入官方文件")
    source = Path(path)
    if not source.is_file():
        raise ContractError(f"{name} 文件不存在：{source}")
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    if source.suffix in {".feather", ".arrow"}:
        return pd.read_feather(source)
    if source.suffix == ".csv":
        return pd.read_csv(source)
    raise ContractError(f"{name} 必须为 parquet、feather 或 csv")


def _prepare(config: dict, *, output_dir: Path):
    data = config["data"]
    scope = config.get("scope")
    if scope not in {"diagnostic", "official"}:
        raise ContractError("配置必须显式选择 scope=diagnostic 或 official")
    if scope == "official":
        missing = [name for name in ("expected_path", "instrument_map_path", "label_spec", "grid_path", "fold") if not data.get(name)]
        if not data.get("targets_path") and not (data.get("labels_path") and data.get("support_path")):
            missing.append("targets_path or labels_path + support_path")
        if missing:
            raise ContractError("正式训练缺少 " + ", ".join(missing) + "；四频率原始行情不能替代官方股票池、标签与支持区间。")
    store = LocalBarStore(data["data_dir"], cache_dir=data.get("cache_dir") or output_dir / "raw_cache",
        policy=ReadPolicy(negative_order_policy=data.get("negative_order_policy", "reject"),
                          **data.get("read_budget", {})))
    common = {"window_length": data.get("window_length", 120), "fit_window_length": data.get("fit_window_length", 120),
              "seed": config.get("seed", 7), "bar_end_is_available": data.get("bar_end_is_available", False)}
    if scope == "diagnostic":
        return prepare_diagnostic_fold(store, days=data["days"], instrument_ids=data["instrument_ids"],
            validation_start=data["validation_start"], endpoint_minutes=data.get("endpoint_minutes", ["10:00", "11:00", "14:00"]),
            **common)
    entries = json.loads(Path(data["instrument_map_path"]).read_text(encoding="utf-8"))
    if not isinstance(entries, dict):
        raise ContractError("instrument_map_path 必须是显式 ID 到正式代码的 JSON 字典")
    mapping = {int(key): value for key, value in entries.items()}
    if len(mapping) != len(entries):
        raise ContractError("股票映射 JSON 包含等价重复整数键")
    grid_frame = _table(data["grid_path"], "official grid")
    grid_column = data.get("grid_column", "datetime")
    if grid_column not in grid_frame:
        raise ContractError(f"官方网格缺少显式 {grid_column} 列")
    grid = pd.DatetimeIndex(pd.to_datetime(grid_frame[grid_column], errors="raise"))
    expected = _table(data["expected_path"], "official expected")
    label_spec = LabelSpec(**data["label_spec"])
    fold = TimeFold(**data["fold"])
    kwargs = {"label_spec": label_spec, "trading_grid": grid, "fold": fold, "instrument_map": mapping,
              "scale_review": data.get("scale_review", "local-N1:automatic-finite-check; manual-review-skipped")}
    if data.get("targets_path"):
        kwargs["targets"] = _table(data["targets_path"], "official targets")
    else:
        kwargs.update(labels=_table(data["labels_path"], "official labels"),
                      support=_table(data["support_path"], "explicit target support"))
    return prepare_local_fold(store, expected, **kwargs, **common)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("benchmark", "matrix", "single"), default="matrix")
    parser.add_argument("--device", help="Explicit device override; defaults to the saved plan")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--models", help="Comma-separated subset of configured ModelSpec names")
    parser.add_argument("--seeds", help="Comma-separated seeds")
    parser.add_argument("--lookbacks", help="Comma-separated lookbacks")
    parser.add_argument("--epochs", type=int, help="Independent epochs budget for every requested run")
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args(argv)
    if args.threads <= 0:
        raise ContractError("threads 必须为正整数")
    torch.set_num_threads(args.threads)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    repository = Path(__file__).resolve().parents[1]
    if output_dir == repository or repository in output_dir.parents:
        raise ContractError("真实行情诊断产物和权重请保存到仓库外的私人目录")
    prepared = _prepare(config, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "prepared_manifest.json", prepared.manifest)
    _write_json(output_dir / "normalization.json", prepared.normalization_diagnostics)
    prepared.split_audit.to_parquet(output_dir / "split_audit.parquet", index=False)
    plan = ExperimentPlan.from_dict(config["experiment_plan"])
    if args.device:
        plan = replace(plan, device=args.device)
    if args.models:
        names = set(args.models.split(","))
        specs = tuple(spec for spec in plan.model_specs if spec["name"] in names)
        if {spec["name"] for spec in specs} != names:
            raise ContractError("--models 只能选择配置里已有的候选")
        plan = replace(plan, model_specs=specs)
    for name in ("seeds", "lookbacks"):
        if getattr(args, name):
            plan = replace(plan, **{name: tuple(int(value) for value in getattr(args, name).split(","))})
    if args.epochs is not None:
        plan = replace(plan, budget=replace(plan.budget, epochs=args.epochs))
    if args.mode == "single":
        plan = replace(plan, model_specs=plan.model_specs[:1], lookbacks=plan.lookbacks[:1], seeds=plan.seeds[:1])
    from ptop.temporal import ModelSpec, create_score_model
    from ptop.training import seed_everything
    # Set CUBLAS_WORKSPACE_CONFIG before the first CUDA benchmark initializes
    # cuBLAS; setting it only later in the runner cannot repair an existing handle.
    seed_everything(plan.seeds[0], deterministic=(plan.training or {}).get("deterministic", True))
    spec = ModelSpec.from_dict({**plan.model_specs[0], "lookback": prepared.window_length})
    benchmark = benchmark_windows(prepared.validation_loader(), model=create_score_model(spec), device=plan.device)
    benchmark.update(scope=prepared.scope, preprocessor_fingerprint=prepared.processor.fingerprint)
    _write_json(output_dir / "benchmark.json", benchmark)
    result = {"scope": prepared.scope, "benchmark": benchmark, "output_dir": str(output_dir)}
    if args.mode != "benchmark":
        comparison = run_experiments(plan, prepared, output_dir=output_dir / "matrix", resume_completed=args.resume_completed)
        result.update(planned_runs=len(comparison), completed_runs=int(comparison.status.eq("completed").sum()),
                      failed_runs=int(comparison.status.eq("failed").sum()),
                      budget_exceeded_runs=int(comparison.status.eq("budget_exceeded").sum()))
    _write_json(output_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result.get("failed_runs", 0) or result.get("budget_exceeded_runs", 0):
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
