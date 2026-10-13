# B 工程验收与实际证据

本次实现基于 A 上游 `a7dd1f790d20b8d4089bbcc3599166fb9f405d7c`。验收对象是 B 的可运行代码与交接接口。真实行情已下载、读取并用于训练工程检查；人工诊断目标的 loss/IC 不代表比赛收益预测效果，也不替代官方平台验证。

## 对照四份 DOCX 的交付

| B 要求或比赛约束 | 具体实现 | 验收方式 |
| --- | --- | --- |
| 原始行情端到端学习，不加手工因子 | A raw25 encoder + B 完整 score model；整个模型参与优化 | encoder/backbone/head 或 A 实际注册模块的梯度与参数更新检查 |
| 基线、架构配置和模型升级 | A 九个原模型保持原类与权重；B-GRU/Transformer/TCN/SelectiveSSM | 原模型往返、全部家族训练和默认实验 runner 的 A/B 兼容检查 |
| 可提交的 .py 训练入口 | ptop.train、TrainingConfig、官方/诊断 data_factory | 实际 CLI 运行、早停、缺合同拒绝与独立进程恢复 |
| 严格时序验证 | TimeFold、显式标签支持区间 purge、完整 sample_id 对齐 | 未来输入、跨日标签、跨 split 支持、乱序标签与缺样本拒绝 |
| train-only 标准化 | A FrozenPreprocessor；本地流式 N1 冻结并校对实际 Qlib | 唯一 fit rows、常量/缺失列、NumPy 缓冲边界与实际 Qlib 对照 |
| 固定种子、超参数与模型比较 | ExperimentPlan、多 seed/窗口、独立预算和状态报告 | 同 N1、采样键、目标、预算；不把失败或超限换成简单候选 |
| 早停与保存最佳整模型 | Trainer、best.pt/last.pt、scheduler/scaler/RNG | 最佳权重所属运行检查、同分最早 epoch、完整恢复状态对照 |
| 冻结、离线可复现推理 | bundle v2、source/依赖/许可/校验和、纯推理 loader | 新进程使用携带源码复现；同形错误配置和修改文件被拒绝 |
| 单个 score 与合法输出集合 | ptop.inference + A output audit | 恰好三列、全键、有限分数、每时点至少 60% 真实模型覆盖 |
| 股票分批不改变推理语义 | stateless eval、microbatch 与日分区 | 股票重排、批大小、缺分钟、结构填充和重复调用对照 |
| 平台评分与唯一 Notebook | C 的目标/评价注入和 bundle/cloud 推理入口已提供 | C 接官方 BARRA/压力/换手模块及 Notebook；本次未执行平台提交 |

依据 `01_比赛规则与参赛流程.docx`，平台五项评分与日内合法采样、无外网和运行时限须再在比赛环境确认。本地 RankIC 是研究指标，不冒称其为官方最终 IC、夏普或总分。第一版只使用 bar1m 符合仓库 `02_量化名词与指标解释.docx` 的实现路径；四文件格式和真实示例见 [数据参考](b_data_reference.md)。

## 固定环境和测试命令

CPU 环境为 `/home/data4/hts/jyc2026_cpu_validation_env`，Torch 2.8.0+cpu；CUDA 环境为 `/home/data4/hts/jyc2026_training_env`，Torch 2.8.0/CUDA 12.8，实际设备 RTX 4090。共同依赖为 Python 3.12、NumPy 2.2.6、pandas 2.2.3、Arrow 20.0.0、pytest 8.4.2；Qlib 0.9.8 确认由固定提交 `54355232463878d2eebb91fe0ee5fa7fa1f5976c` 编译安装。两个环境的 pip check 均通过。

集成 core `4a4d18f` 执行 CPU **266 passed / 16 skipped**、CUDA **282 passed**。随后 `6cd71f1` 修复实验报告对 A 原生模块名的假设，新增三个完整默认 runner 回归检查。最终完整复测结果为 CPU **269 passed / 16 skipped（199.37 秒）**、CUDA **285 passed、无跳过（243.33 秒）**。CPU 跳过项都是明确 opt-in 的 CUDA 检查。两套测试使用独立目录，避免共享临时产物：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/data4/hts/jyc2026_cpu_validation_env/bin/python -m pytest -q tests \
  --basetemp=/home/data4/hts/jyc2026/b_work/pytest-final-cpu

# 本机运行前确认物理设备空闲；逻辑 cuda:0 对应所选可见设备。
CUDA_VISIBLE_DEVICES=9 PTOP_TEST_CUDA=1 PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/data4/hts/jyc2026_training_env/bin/python -m pytest -q tests \
  --basetemp=/home/data4/hts/jyc2026/b_work/pytest-final-gpu
```

测试包含 A 桥接、九个旧模型与四个新主干、mask/因果性/2049 槽容量、全部编码器组合、实际完整训练、标签键对齐、早停、精度、完整断点与模型包、调用者 RNG、覆盖门槛和惰性分区。CUDA 还检查四个主干的 FP16/BF16、默认 FP32 验证分数回放、单/双层 dropout GRU 的 epoch 恢复，以及 scale=1024 和 scale=1 的真实梯度溢出跳过机制。

完整日志为 `/home/data4/hts/jyc2026/b_work/final_cpu_isolated_tests.log` 和 `final_gpu_isolated_tests.log`。耗时受并行任务和共享机器负载影响，不将其作为固定性能保证。

## 真实行情工程检查

共同输入为 2020-01-02/03 的训练行情、2020-01-06 的验证行情，ID 11/16/19/30，端点 10:00/11:00/14:00。端点和 bar-end 可用性是显式工程假设，不代表官方合法采样或生产可见性。所有批次均核查 `visibility_basis=bar_end_only`、`availability_assumption=bar_end_equals_available_at` 和 `production_visibility_verified=False`。

四个 B 主干 × L60/120 × seed7/17/29 的 CPU 矩阵在 core `4a4d18f` 完成 **24/24**，共 48 epochs、288 次优化更新。共同冻结 N1 使用 1,520 个唯一训练可见行，处理器、有效预测键和目标尺度完全相同。每轮 12 个验证键和完整产物经逐文件检查，24 个最佳权重重新加载后的 FP32 分数均逐位一致；全部有参数的 encoder/backbone/head 实际更新。证据在 `/home/data4/hts/jyc2026/b_runs/diagnostic-cpu-final-v4/artifact_audit.json`。

同版本真实 `.py` CLI 的一次 2 epoch 完整运行与 stop1→新进程 resume2 对照，model、optimizer、optimizer layout、scheduler、scaler 和 RNG 六项在该样例中均逐位一致，模型最大差为 0。证据在 `/home/data4/hts/jyc2026/b_runs/diagnostic-cpu-resume-v4/resume_audit.json`。这一结论限定固定依赖和样例，不保证跨硬件精确一致，也不表示支持任意 batch 的崩溃恢复。

同样真实数据的 CUDA 检查为四个 B 主干 × FP32/BF16/FP16，L120、seed7，共 **12/12** 完成、24 epochs、144 次成功更新，0 次 AMP 跳过。全部最佳 checkpoint 的默认 FP32 推理与保存的 FP32 验证分数最大差为 **0**，三组参数均实际更新，诊断 scope 没有导出正式包。证据为 `/home/data4/hts/jyc2026/b_runs/diagnostic-gpu-final-v4-audit.json` 与各精度目录。

现代矩阵以上述 core 记录实现指纹。A 原生参照模型使用随后仅修复参数统计的 `6cd71f1`；模型、Trainer、数据、N1、标签、expected 和预算没有改变。A 的实际模块名与参数容量不同，不能把它们重命名为 B 的模块或假设同容量。详细参照结果见 [本地实验](b_local_experiments.md)。

R0/R1 × L60/120 × seed7/17/29 的 A 矩阵也完成 **12/12**，共 24 epochs、144 次更新，全部最佳分数回放逐位一致。A/B 共 36 格、72 epochs、432 次更新；从保存的 FP32 score 重算最佳 loss，最大差为 0。合并工程记录位于 `/home/data4/hts/jyc2026/b_runs/diagnostic-cpu-a-b-engineering-comparison-final/`。它提供同协议的工程对照，不用于在人工目标上选出金融收益最优模型。

## GPU 成本的实际限制

`examples/b_gpu_benchmark.py` 使用 L120、ALSTM encoder、width32、layers2、mean pooling、FP32 eval，重复四个真实窗口以测 batch 32/256/1024。每种配置预热三次、同步计时十次，以下是 batch1024 的中位前向耗时与峰值 allocated 显存：

| 主干 | 中位前向 | 峰值 allocated |
| --- | ---: | ---: |
| B-GRU | 7.15 ms | 198.49 MiB |
| B-Transformer | 9.93 ms | 916.93 MiB |
| B-TCN | 2.87 ms | 167.47 MiB |
| B-SelectiveSSM | 49.50 ms | 268.09 MiB |

这是容量探测，使用随机权重和重复窗口，排除了源查询、分区规范化、张量传输、输出审核和平台运行时间。不能据此宣称完整股票池或平台三小时已通过。SelectiveSSM 的 Python 顺序 scan 在此实现中明显更慢；线性序列复杂度不保证比 Transformer 快。原始报告在 `/home/data4/hts/jyc2026/b_runs/gpu-capacity-final/report.json`。

## 构建与最终衔接

仓库 wheel 已在固定环境用 `pip wheel --no-deps --no-build-isolation .` 构建成功，并核对所有 ptop Python 源码、Qlib 许可、contest Torch 2.8.0 pin 与 `ptop-train` 入口实际在 wheel 中。私人审计为 `/home/data4/hts/jyc2026/b_work/repository_wheels/final/wheel_audit.json`。真实行情、日缓存、checkpoint、运行产物和环境均留在数据盘。

正式训练的下一输入是 A/C 提供的官方采样表、正式股票映射、可见性说明、LabelSpec、target_valid 和支持区间。B 的正式 data_factory 与完整训练/矩阵入口已实现；这些文件到位后可直接进入连续日期时间折。C 随后加载最佳 bundle，在无外网环境执行唯一 Notebook、官方五项评价及完整评测区间，保存成功提交记录。TTT/扩散目前只有基于原论文的适配研究，未把未经验证的结构注册成可交付候选；见 [时序研究](b_temporal_research.md)。
