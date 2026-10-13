#!/usr/bin/env python3
"""提交材料完整性校验（六项 + 运行约束）+ 冻结清单。

规则要求提交**完整模型包**：训练脚本、推理脚本、依赖声明、超参配置、随机种子、
已训练权重；且**有且仅有一个** Notebook。缺任一项，提交可视为无效。

除六项材料外，这里还静态复核三条赛制约束：

* Notebook 的 ``main`` 签名必须恰为 ``(datasources, start_date, end_date)``；
* 历史起点必须由 ``history_start_for`` 动态推算「评测区间前一年」，不得写死年份；
* 若提供 ``--runtime-seconds`` / ``--output-frame``，则复核 3 小时上限与输出区间。

这个脚本默认只用标准库，因此可以在平台环境或本地任意 Python 下直接运行；只有显式传入
``--output-frame`` 时才按需引入 pandas/numpy。

    python submission/verify_submission.py --bundle /path/to/bundles/epoch0007
    python submission/verify_submission.py --bundle ... --freeze   # 写入冻结清单
    python submission/verify_submission.py --bundle ...            # 冻结后比对，改动即失败

`--freeze` 把六项材料的 sha256 写入清单。冻结之后不带 --freeze 再跑，对不上就报错，
对应规则里「私榜候选冻结后不得修改代码、参数、权重、依赖和随机种子」。
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

SKIP_DIRS = {".git", ".ipynb_checkpoints", "__pycache__", "work", ".pytest_cache", "node_modules", ".venv"}
SKIP_NAMES = {".DS_Store"}
FILE_ITEMS = ("train_script", "inference_notebook", "requirements", "config", "weights")

RUNTIME_BUDGET_SECONDS = 3 * 60 * 60
MAIN_PARAMETERS = ("datasources", "start_date", "end_date")
HISTORY_HELPER = "history_start_for"
OUTPUT_COLUMNS = ["date", "instrument", "score"]


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    if path.is_dir():
        for item in sorted(path.rglob("*")):
            if item.is_file() and item.name not in SKIP_NAMES:
                checksum.update(item.relative_to(path).as_posix().encode())
                checksum.update(item.read_bytes())
        return checksum.hexdigest()
    checksum.update(path.read_bytes())
    return checksum.hexdigest()


def notebooks(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.ipynb")
        if not any(part in SKIP_DIRS for part in path.relative_to(root).parts)
    )


def notebook_code_source(path: Path) -> str:
    """把所有代码单元的源码拼起来；只解析文本，绝不执行 Notebook。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    )


def notebook_defines_main(path: Path) -> bool:
    return "def main(" in notebook_code_source(path)


def notebook_main_parameters(path: Path) -> list[str] | None:
    """Notebook 顶层 ``main`` 的参数名；无法解析（例如含 cell magic）时返回 None。"""
    try:
        tree = ast.parse(notebook_code_source(path))
    except (SyntaxError, ValueError):
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main":
            return [argument.arg for argument in node.args.args]
    return None


def check_output_frame(path: Path, start_date, end_date) -> list[str]:
    """可选：复核一次真实运行的输出帧（列顺序、分钟精度、唯一键、有限值、时间区间）。"""
    import numpy as np
    import pandas as pd

    try:
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    except Exception as error:  # noqa: BLE001 - 任何读取失败都应作为校验问题上报
        return [f"输出样本不可读：{path}（{error}）"]
    if list(frame.columns) != OUTPUT_COLUMNS:
        return [f"输出样本列必须为 {OUTPUT_COLUMNS}，实际 {list(frame.columns)}"]
    problems: list[str] = []
    if frame.empty:
        problems.append("输出样本为空")
    times = pd.to_datetime(frame["date"], errors="raise")
    if not times.eq(times.dt.floor("min")).all():
        problems.append("输出样本时间必须精确到分钟")
    if frame.duplicated(["date", "instrument"]).any():
        problems.append("输出样本 (date, instrument) 必须唯一")
    if not np.isfinite(pd.to_numeric(frame["score"], errors="coerce").to_numpy(dtype=float)).all():
        problems.append("输出样本 score 必须为有限数值")
    if start_date is not None and end_date is not None and not frame.empty:
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        if end == end.normalize():
            end += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        if not times.between(start, end).all():
            problems.append("输出样本时间超出 start_date..end_date")
    return problems


def check(root: Path, *, notebook: Path, train: Path, requirements: Path, config: Path,
          bundle: Path, output_frame: Path | None = None, start_date=None, end_date=None,
          runtime_seconds: float | None = None) -> tuple[list[str], list[str], dict, dict]:
    """Return (problems, notes, file_items, scalars). Any problem makes the submission invalid."""
    problems: list[str] = []
    notes: list[str] = []
    items: dict[str, str] = {}
    scalars: dict[str, object] = {}

    # 0) 规则：有且仅有一个 Notebook。
    found = notebooks(root)
    if len(found) != 1:
        problems.append(f"必须恰好有一个 Notebook，实际找到 {len(found)} 个：{[str(p) for p in found]}")
    elif found[0] != notebook:
        problems.append(f"唯一 Notebook 应为 {notebook}，实际为 {found[0]}")

    # 1) 训练脚本（.py；用于复现与合规检查，不是公榜重训入口）。
    if not train.is_file() or train.suffix != ".py":
        problems.append(f"缺少训练脚本 .py：{train}")
    elif "def main(" not in train.read_text(encoding="utf-8", errors="ignore"):
        problems.append(f"训练脚本缺少可调用入口 main()：{train}")
    else:
        items["train_script"] = str(train)

    # 2) 推理脚本：唯一 Notebook，且必须定义 main()，签名恰为三个参数。
    if notebook.is_file():
        if not notebook_defines_main(notebook):
            problems.append(f"Notebook 未定义 main()：{notebook}")
        else:
            items["inference_notebook"] = str(notebook)
            parameters = notebook_main_parameters(notebook)
            if parameters is None:
                notes.append("无法静态解析 Notebook 的 main 参数名（可能含 cell magic），请人工确认签名为 "
                             f"{list(MAIN_PARAMETERS)}")
            elif parameters != list(MAIN_PARAMETERS):
                problems.append(f"Notebook 的 main 签名必须恰为 {list(MAIN_PARAMETERS)}，实际 {parameters}")
        if HISTORY_HELPER not in notebook_code_source(notebook):
            problems.append("Notebook 未用 history_start_for 动态推算「评测区间前一年」历史，"
                            "不得写死年份")
    else:
        problems.append(f"缺少推理 Notebook：{notebook}")

    # 3) 依赖声明。
    if not requirements.is_file() or not requirements.read_text(encoding="utf-8").strip():
        problems.append(f"缺少或为空的依赖声明：{requirements}")
    else:
        items["requirements"] = str(requirements)
        if "未决项" in requirements.read_text(encoding="utf-8"):
            notes.append("依赖声明仍标注 Qlib 为推理期依赖，请先确认线上镜像是否自带（见文件内注释）")

    # 4) 超参配置 + 5) 随机种子。
    settings = None
    if not config.is_file():
        problems.append(f"缺少超参配置 JSON：{config}")
    else:
        try:
            settings = json.loads(config.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            problems.append(f"超参配置不是合法 JSON：{config}（{error}）")
        if isinstance(settings, dict):
            items["config"] = str(config)
            seed = settings.get("seed")
            if seed is None and isinstance(settings.get("training"), dict):
                seed = settings["training"].get("seed")
            if seed is None:
                problems.append(f"超参配置缺少随机种子 seed（顶层或 training.seed）：{config}")
            elif isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
                problems.append(f"seed 必须为 [0, 2**32) 的整数，实际为 {seed!r}")
            else:
                scalars["seed"] = seed
            if not {"model_specs", "model"} & set(settings):
                notes.append("配置里没有 model_specs/model 字段，请确认权重与配置一一对应")
            budget = settings.get("runtime_budget_seconds")
            if budget is not None:
                if isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget <= 0:
                    problems.append(f"runtime_budget_seconds 必须为正数，实际 {budget!r}")
                elif budget > RUNTIME_BUDGET_SECONDS:
                    problems.append(f"声明的 runtime_budget_seconds={budget} 超过 3 小时"
                                    f"（{RUNTIME_BUDGET_SECONDS}s）上限")

    # 6) 已训练权重（save_bundle 产出的目录）。
    if not bundle.is_dir():
        problems.append(f"缺少权重目录（save_bundle 产物）：{bundle}")
    else:
        missing = [name for name in ("manifest.json", "preprocessor.json") if not (bundle / name).is_file()]
        weight = next((bundle / name for name in ("model.json", "model.pt") if (bundle / name).is_file()), None)
        if missing:
            problems.append(f"权重目录缺少 {missing}：{bundle}")
        if weight is None:
            problems.append(f"权重目录缺少 model.json / model.pt：{bundle}")
        if not missing and weight is not None:
            items["weights"] = str(bundle)
            try:
                manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
                notes.append(
                    f"bundle: schema={manifest.get('schema_version')} seed={manifest.get('seed')} "
                    f"window={manifest.get('window_length')} model={manifest.get('model_name')}"
                )
                if "seed" in scalars and manifest.get("seed") != scalars["seed"]:
                    problems.append("bundle 内的 seed 与超参配置的 seed 不一致")
            except (OSError, json.JSONDecodeError):
                problems.append(f"bundle manifest 不可读：{bundle / 'manifest.json'}")

    # 7) 运行约束：3 小时上限与输出区间（仅在提供证据时复核）。
    if runtime_seconds is not None:
        if runtime_seconds > RUNTIME_BUDGET_SECONDS:
            problems.append(f"实测运行 {runtime_seconds:.0f}s 超过 3 小时（{RUNTIME_BUDGET_SECONDS}s）上限")
        else:
            notes.append(f"实测运行 {runtime_seconds:.0f}s ≤ 3 小时上限")
    if output_frame is not None:
        problems.extend(check_output_frame(output_frame, start_date, end_date))

    return problems, notes, items, scalars


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="校验六项提交材料与运行约束，并维护冻结清单")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--bundle", type=Path, required=True, help="save_bundle 产出的权重目录")
    parser.add_argument("--notebook", type=Path, default=None)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--requirements", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None,
                        help="超参配置 JSON；省略则优先取 submission/run_config.json")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-frame", type=Path, default=None,
                        help="可选：一次真实运行的输出帧（.parquet/.csv），复核列与时间区间")
    parser.add_argument("--start-date", default=None, help="配合 --output-frame 的评测区间起点")
    parser.add_argument("--end-date", default=None, help="配合 --output-frame 的评测区间终点")
    parser.add_argument("--runtime-seconds", type=float, default=None,
                        help="可选：一次真实运行的总耗时，复核是否超过 3 小时")
    parser.add_argument("--freeze", action="store_true", help="写入/覆盖冻结清单")
    arguments = parser.parse_args(argv)

    root = arguments.root.resolve()
    submission = root / "submission"
    notebook = (arguments.notebook or submission / "inference.ipynb").resolve()
    train = (arguments.train or root / "src" / "ptop" / "train.py").resolve()
    requirements = (arguments.requirements or submission / "requirements-inference.txt").resolve()
    bundle = arguments.bundle.resolve()
    default_config = submission / "run_config.json"
    if not default_config.is_file():
        default_config = bundle.parent / "run_config.json"
    config = (arguments.config or default_config).resolve()
    manifest_path = (arguments.manifest or submission / "MANIFEST.json").resolve()

    problems, notes, items, scalars = check(
        root, notebook=notebook, train=train, requirements=requirements, config=config, bundle=bundle,
        output_frame=arguments.output_frame, start_date=arguments.start_date, end_date=arguments.end_date,
        runtime_seconds=arguments.runtime_seconds)

    print("== 提交材料检查 ==")
    for name in ("train_script", "inference_notebook", "requirements", "config", "weights"):
        print(f"  {'OK  ' if name in items else 'MISS'} {name}: {items.get(name, '-')}")
    print(f"  {'OK  ' if 'seed' in scalars else 'MISS'} seed: {scalars.get('seed', '-')}")
    for note in notes:
        print(f"  note: {note}")
    for problem in problems:
        print(f"  FAIL: {problem}")

    if problems:
        print(f"\n结论：不合格（{len(problems)} 项问题）")
        return 1

    fingerprints = {name: {"path": items[name], "sha256": digest(Path(items[name]))} for name in FILE_ITEMS}

    if arguments.freeze:
        payload = {"frozen": True, "root": str(root), "seed": scalars.get("seed"), "items": fingerprints}
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n已冻结清单：{manifest_path}")
        return 0

    if not manifest_path.is_file():
        print("\n提示：尚无冻结清单；确认无误后加 --freeze 生成。")
        print("结论：六项齐全")
        return 0

    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = [name for name, record in frozen.get("items", {}).items()
               if fingerprints.get(name, {}).get("sha256") != record.get("sha256")]
    if frozen.get("seed") is not None and frozen.get("seed") != scalars.get("seed"):
        changed.append("seed")
    if changed:
        print(f"\nFAIL: 冻结后材料发生变化：{changed}")
        return 1
    print(f"\n结论：六项齐全，且与冻结清单一致（{len(fingerprints)} 项）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
