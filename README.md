# 数据处理、端到端训练与推理

> **找不到代码在哪？先看 [代码地图](docs/code_map.md)** —— 30 秒定位到要改的文件。

当前仓库同时包含 `src/ptop/` 的数据、预处理与 encoder 实验模块，以及 `src/jyc_e2e/` 的模型原型。损失设计与训练评估规划见 [第一阶段损失函数接入说明](docs/stage_one_losses.md) 和 [完整训练评估规划](docs/objective_training_plan.md)。

[src/ptop/](src/ptop/) 提供云端 1 分钟数据读取、N1 标准化、当日历史窗口和可训练编码器。默认窗口为 120 分钟，使用 25 个原始字段及观测掩码。

## 安装

推荐 Python 3.12。云端先准备匹配 GPU 环境的 PyTorch 2.8.0，然后安装：

```bash
python -m pip install -r requirements-a.txt
```

Qlib 源码安装需要 Git 和 C++ 编译环境；`dai` 使用平台提供的客户端。训练与推理保持依赖版本一致，正式运行时不下载依赖。

## 调用编码器

```python
from ptop.encoders import ALSTMEncoder

encoder = ALSTMEncoder().to(device)
tokens, token_valid = encoder(x, observed, step_valid)
```

- `x`：标准化后的 torch.float32 `[B,L,25]`，字段顺序取 `ptop.schema.FEATURES`。
- `observed`：torch.bool `[B,L,25]`，True 表示字段有效。
- `step_valid`：torch.bool `[B,L]`，True 表示真实时间槽；真实分钟缺行情时仍保留该槽。
- 返回 `tokens [B,L,64]` 和 `token_valid [B,L]`，可交给后续时序主干。

三个输入张量放在同一设备。接入自己的模型时，把编码器注册为 `self.encoder`，让优化器包含它的参数。

需要 A 的完整参照模型时，使用 `ptop.bundle.create_model("R1")`。B 的 `create_score_model(ModelSpec(...))` 支持 GRU、Transformer、TCN 和选择性状态空间研究模型，以及 A 的九个原有候选；调用参数相同，返回 `[B]` 分数。

## 准备数据

组合调用见 [examples/cloud_bridge.py](examples/cloud_bridge.py)：

| 函数 | 用途 |
| --- | --- |
| `fetch_cloud_training(dai.query, datasources, ...)` | 读取云端行情、官方截面与 `vwap_return` 标签 |
| `prepare_fold(...)` | 按 `TimeFold` 切分，拟合 N1，返回处理器与训练/验证 DataLoader |
| `supervised_step(...)` | 一批 MSE 训练；`optimizer=None` 时计算验证结果 |
| `predict_cloud(...)` | 使用已有模型和处理器，返回预测、排除记录及覆盖统计 |

调用者提供日期范围、`instrument_map`、`trading_grid` 和标签说明 `LabelSpec`。训练读取还需要 `support` 表，列为 `sample_id、target_valid、target_start、target_end、label_spec_version`，按真实标签定义填写。

行情表使用 `datasources["bar1m"]`，截面表为 `cpt_jyc_2026_instruments`，标签取 `cpt_jyc_2026_vwap.vwap_return`。云端价格和金额使用元单位，委托笔数保留 int32 或更宽类型。

`WindowBatch` 保存 NumPy 数组：先按 `batch.predictable` 排除全空样本，再用 `torch.as_tensor` 转到模型设备。两个 mask 均以 True 表示有效；缺失数值已填零。

N1 只在训练记录上拟合，验证和推理复用冻结状态。按能容纳于内存的区间读取数据，各分区共用本折 N1；窗口不跨日，决策时点需包含在分钟网格中。

## B 的训练与交接

`ptop.training.Trainer` 接收上述 prepared fold，完成联合训练、完整截面验证、早停、最佳/最新 checkpoint、epoch 恢复和模型包导出。`ptop.inference` 提供冻结模型的分批/日期分区推理，保留 sample_id 对齐与逐时点真实覆盖审核。A 的原有 v1 模型包继续兼容，B 的 v2 包保存参数化模型配置、代码、依赖和文件校验和。

```python
from ptop.temporal import ModelSpec, create_score_model
from ptop.training import Trainer, seed_everything
from ptop.training_config import TrainingConfig

config = TrainingConfig(seed=7)
spec = ModelSpec(name="B-GRU", lookback=120)
seed_everything(config.seed, deterministic=config.deterministic)
model = create_score_model(spec)
trainer = Trainer(model, config, output_dir=private_run_dir,
                  model_spec=spec, contracts=contracts, fold=fold,
                  instrument_map=instrument_map, device=device)
result = trainer.fit(prepared)
```

完整训练脚本入口为 `python -m ptop.train --config /private/run.json --output /private/run --device cpu`。合同、配置结构、恢复及 C 的目标/评价注入见 [B 训练与交接](docs/b_training_handoff.md)；模型论文与适配边界见 [时序研究](docs/b_temporal_research.md)，保存与推理见 [模型包与推理](docs/b_bundle_inference.md)。

本地月度 Feather 用 `ptop.local_training` 按批读取并生成日分区，避免一次加载五年数据。真实行情工程诊断与多模型、多窗口、多 seed 实验入口见 [本地实验](docs/b_local_experiments.md)。诊断目标不是官方收益；正式训练要求真实标签、采样集合、股票代码映射与可见性合同。官方评分、唯一提交 Notebook 和线上实际验证由 C 衔接。依赖来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。

固定依赖下的 CPU/CUDA 测试、真实窗口实验、恢复检查和资源测量见 [B 验证记录](docs/b_verification.md)。混合精度只用于训练，验证和默认推理均使用 FP32；训练环境与 C 的部署环境保持相同 Torch build。

四个下载包的行数、字段单位、实际同时间数据示例及 raw25 顺序见 [真实数据参考](docs/b_data_reference.md)。
