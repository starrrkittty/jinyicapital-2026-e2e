# 提交包说明（进益资本杯 2026 · 端到端赛道）

本目录是提交给平台的推理材料。规则要求**有且仅有一个** Notebook、其中定义可调用的
`main(datasources, start_date, end_date)`，且脚本、配置、模型参数齐备；评测环境**无外网**、
总运行时间不超过 **3 小时**。平台会切换到验证集**重新运行**本 Notebook，不接受预先算好的离线结果。

## 目录内容

| 材料 | 路径 | 说明 |
| --- | --- | --- |
| 推理 Notebook（唯一） | `inference.ipynb` | 只做胶水：取表名与区间 → `submission.run_submission` → 返回三列 |
| 训练脚本 | `../src/ptop/train.py` | `def main()` 可调用入口，用于复现与合规检查 |
| 推理依赖声明 | `requirements-inference.txt` | 离线镜像需预装，版本与权重包 manifest 一致 |
| 超参配置 | `run_config.json` | 含顶层 `seed`、`model`/`model_specs`，与权重一一对应 |
| 已训练权重 | `weights/epoch0007/` | `save_bundle` 产物：`manifest.json` + `preprocessor.json` + `model.json|model.pt` |
| 校验脚本 | `verify_submission.py` | 六项材料 + 运行约束的静态校验与冻结清单 |

共享代码在 `../src/`（`ptop` / `jyc_e2e`）。Notebook 启动时把仓库 `src/` 加入 `sys.path`，
因此无需在镜像里额外安装本仓库；若镜像已预装 `ptop` 则直接复用。

## 规则落点

- **一年历史**：`main` 用 `submission.history_start_for(start_date)` 动态推算「评测区间前一年」，
  **不写死年份**。写死更早的日期既超出授权，也会多读无谓行情。
- **结果范围**：输出经 `ptop.output.audit_output` 审核，`date` 必须落在 `start_date` 至 `end_date` 内。
- **3 小时 / 内存**：`run_submission` **逐日分区**推理（`predict_bundle_partitions`），
  每个分区只读当日行情并单独过审，避免一次性载入整年行情；`weight_format=json` 的权重便于逐位审计。
- **无外网**：推理不查询标签、不安装依赖、不重训；Qlib 等依赖必须由镜像预装。

## 放权重与自检

1. 训练产出形如 `runs/<fold>/bundles/epoch0007/`；把该目录复制为 `weights/epoch0007/`，
   并把训练目录下的 `run_config.json` 与本目录的 `run_config.json` 对齐（至少 `seed` 一致）。
2. 运行校验：

   ```bash
   python submission/verify_submission.py --bundle submission/weights/epoch0007
   ```

   六项材料齐全且默认与 `submission/run_config.json` 对齐后，可加 `--freeze` 生成冻结清单；
   之后再跑不带 `--freeze`，材料被改动即失败。

3. 如需用一次真实运行的输出与时长做证据，额外提供：

   ```bash
   python submission/verify_submission.py --bundle submission/weights/epoch0007 \
       --output-frame /path/to/output.parquet --start-date 2025-01-02 --end-date 2025-01-10 \
       --runtime-seconds 1234
   ```
