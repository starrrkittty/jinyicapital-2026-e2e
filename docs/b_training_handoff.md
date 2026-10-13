# B 成员：训练、实验与 A/C 交接

本实现以 A 的上游提交 `a7dd1f790d20b8d4089bbcc3599166fb9f405d7c` 和 `ptop.raw25.v1` 为准。B 提供完整训练入口、模型配置、按时间验证、最佳权重、恢复训练和推理模型包；A 负责官方数据与可见性合同，C 负责标签定义、官方评分与唯一提交 Notebook。职责依据仓库 `04_团队分工与开发日程.docx`；训练限制依据 `01_比赛规则与参赛流程.docx` 与 `03_模型架构与训练方案.docx`。

## 下载数据与正式训练的边界

四个本地包在 `/home/data4/hts/cpt_jyc_2026_e2e_bar{1m,5m,15m,30m}.tar.gz`，解压目录在 `/home/data4/hts/jyc2026/e2e_data/`。每个频率有 60 个月度 Feather 文件，覆盖 2020–2024 年。总计约 3.93 亿行，不能一次加载成五年 pandas 长表。月度 IPC 读取、日分区缓存和流式训练见 `b_local_experiments.md`。

这些包已经下载并完成完整性检查。`bmeta.bin` 是未获得格式说明的二进制元数据；未把它执行成 Python 对象，也未据其猜测股票代码。这里的缺项指正式训练合同，不表示原始行情尚未下载。

| 项目 | 下载包实际提供 | B 的处理与衔接 |
| --- | --- | --- |
| 原始结构 | 28 列：时间、整数 instrument_id、25 行情字段、adjust_factor | 输入严格采用 A 的 25 字段顺序；adjust_factor 不作为额外模型因子 |
| 价格/金额 | 本地整数分，云端表为元 | A 的 canonical 层统一单位，再做冻结 N1 |
| 缺失/异常 | OHLC=-1、报价=0；少量负委托笔数 | 保留 observed；负数按显式隔离策略处理，不用 abs 或溢出猜测修补 |
| 排序 | 存储顺序不保证同股时间递增 | 规范层以 datetime/instrument 唯一键排序和查窗 |
| 采样点/股票池 | 行情时间网格与历史整数 ID | 正式 expected 由官方截面提供，不把所有行情时刻或所有历史 ID 当应预测集合 |
| 标签 | 包内未发现独立收益标签和支持区间 | 正式训练接 C 的显式 target/target_valid/target_start/target_end/version |
| 多频率 | 5m/15m/30m 已包含聚合行情及期末盘口 | 第一版消费 A 的 1m 接口；不能按同名 date 猜测未完成粗频 K 线可见性 |

本地真实行情加人工诊断目标只验证读数、反向传播、资源和保存流程，所有结果明确标 `diagnostic`，不导出正式提交包，也不据此声称能预测市场收益。官方标签/代码映射等外部合同尚未具备时，正式训练会清晰报错。

## A → B：冻结数据接口

`PreparedFold` 或同结构流式对象交给 `Trainer.fit`，包含：

| 字段 | 约定 |
| --- | --- |
| processor | 已 fit/freeze 的 `FrozenPreprocessor`，训练窗口唯一原始行拟合，fit_end 早于验证开始 |
| train_loader / validation_loader | DataLoader，或可重入的 `train_loader(epoch)` / `validation_loader()` 工厂 |
| train_targets / validation_targets | 完整应预测集合，sample_id/decision_id/instrument_id/decision_time 及显式标签有效性与支持区间 |
| split_audit | 缺标签、全空输入、标签跨边界等样本及原因；不得用过滤后的集合替代覆盖分母 |

每批沿用 A 的 `WindowBatch`：float32 `x[B,L,25]`，bool `observed[B,L,25]` 和 `step_valid[B,L]`，以及 sample/decision/stock/time、bar_times、row_positions。真实分钟缺行仍是有效时间槽；左侧结构 padding 才是无效槽。窗口不跨日，默认 L=120。模型依赖字段必须与 loader 的 predictable 判断一致。

B 检查输入有限、缺失填零、只左 padding、每个实际 bar_time≤decision_time、每个样本唯一且能用 sample_id 对齐标签。标签表顺序可以打乱，不影响训练。每轮必须覆盖其声明的完整样本集合，不默默重复或遗漏。

## 模型路径与比较顺序

1. 先跑 A 的 R0/R1，确认相同折、字段与 N1 下训练和保存成立。R0 的 RawEncoder 没有参数，主干和头正常训练；其余编码器参数随整个模型共同优化。
2. 比较 B-GRU、B-TCN、B-Transformer；这些模型均接 A 编码器，不缓存训练后的 tokens，不加入手工指标。统一改变 `ModelSpec`，不修改训练循环。
3. 比较 B-SelectiveSSM。它是独立 PyTorch 选择性状态空间研究实现，含因果卷积和输入相关状态更新，不声称复现官方 Mamba 内核或吞吐。
4. TTT、时空扩散的论文、成本与比赛适配边界见 `b_temporal_research.md`。TTT 的内部快速权重更新需先确认比赛固定模型规则；扩散需要额外目标/采样合同与推理预算，不能用名字替代验证。

先比较模型和 L=60/120，再在固定时间折上搜索 width、lr、dropout、weight_decay、rank_weight。每种配置至少多个固定 seed，报告每 seed 和聚合均值/方差。标签、expected、标准化拟合集合与验证时段必须相同。最长窗口引用行拟合一次 N1 后复用给短窗口的方式需记录在实验 manifest；不得将不同预处理的成绩直接归因于模型。

按照 DOCX 的示意路径，可在官方覆盖允许后用 2020–2022 训练、2023 调参、2024 留出检验。每折创建独立处理器，标签 target_end 达到下一段边界的样本 purge；不得按分钟随机拆训练/验证。最终留出段不参与候选搜索和早停。

## 完整训练与恢复

`TrainingConfig` 统一 AdamW、余弦学习率、早停、梯度裁剪、精度、seed 和 microbatch。默认目标是 Huber：仅从有效训练标签估计正尺度，将 target/scale 与 score 比较；可叠加同一 decision 内的 pairwise 排序。scale 在验证与推理固定。

stock microbatch 控制显存，每个 microbatch 做一次优化更新；开启排序时仅比较当前 chunk 的同时间股票对，是显式近似，不等于整截面排序目标。A 的含 BatchNorm 编码器在训练中也受 chunk 分布影响。评估先合并同时间所有预测，再计算 RankIC，不用 microbatch IC 平均代替完整截面。

全空输入与无效标签不进训练损失，但留在原始分母和排除统计。验证会给所有输入可预测股票打分，不依标签是否有效过滤推理样本。验证不更新模型权重、标准化、优化器或学习率；学习率仅在完成训练与验证后推进。

FP16/BF16 只用于训练前向；验证、最佳 checkpoint 选择和默认冻结推理统一为 FP32。FP16 使用动态 loss scaling，遇到梯度溢出会跳过该次更新并降低 scale，记录实际更新数和跳过数；不会裁剪或应用非有限梯度。若整轮都没有成功更新则拒绝选权重。BF16 和 FP32 非有限梯度直接报错。

保存 `last.pt` 供恢复，`best.pt` 对应验证指标最优的较早 epoch，平分保留首次最佳。checkpoint 含整个模型、优化器、scheduler、AMP scaler、Python/NumPy/Torch RNG、epoch/step/patience、配置、数据/标签合同和冻结处理器。恢复严格核对这些合同；`--stop-after-epochs` 只限本次调用，不改变原定 scheduler horizon。当前保证 epoch 边界恢复，不声称任意 batch 崩溃续接。

CUDA 每轮按 `(seed + epoch_index) modulo 2**32` 重新设种子，以重置 cuDNN 的 RNN dropout 缓存；该缓存不属于普通 model.state_dict 或公开 CUDA RNG state。恢复时同时核对 best.pt 的配置、数据、最佳 epoch 和指标，防止复制其他运行的最佳权重。相同环境与样例中的精确恢复证据见 `b_verification.md`，不推广为跨硬件逐位一致保证。[PyTorch 2.8 cuDNN 实现](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cudnn/RNN.cpp)

训练和推理使用一致依赖。先准备与目标 GPU 匹配的 Torch 2.8.0，再安装 `requirements-b.txt`。正式运行前安装依赖，模型包内保存精确版本与离线代码。

依赖核对包含 Torch 的实际 build 标识：`2.8.0+cpu` 与 `2.8.0` CUDA build 不会被当成同一环境。若 C 使用 CUDA build，B 也应在该 build 下训练和导出；必要时该环境可选择 device=cpu，不能换 build 后绕过检查。源码、权重和依赖须在评测前部署，运行时不访问外网。

```bash
PYTHONPATH=src:. python -m ptop.train \
  --config /private/run.json --output /private/runs/gru-seed7 --device cuda:0

PYTHONPATH=src:. python -m ptop.train \
  --config /private/run.json --output /private/runs/gru-seed7 --device cuda:0 \
  --resume /private/runs/gru-seed7/last.pt
```

配置顶层为 model/training/fold/contracts/scope/data_factory/data。`data_factory` 指定 `module:function`，调用 `factory(config=data, model_spec=spec, fold=fold)` 返回 `(prepared, instrument_map)`。C 可通过 objective_factory/evaluator_factory 注入目标与评价；自定义 callable 的 `configuration()` 应完整声明不可变设置，避免恢复时换了目标却仍沿用旧权重。

## B → C：可检查交付

| 交付 | 内容与用途 |
| --- | --- |
| run_config.json | 所有架构与训练参数、随机种子、时间折、目标尺度、合同和处理器指纹 |
| training.jsonl / history.json | step/epoch 损失、完整截面研究 RankIC、排除数量、吞吐、数据等待、RSS/CUDA 显存 |
| validation/epoch-XXXX.parquet | sample/decision/stock/time、score、标签有效性与排除原因，供 C 独立复算 |
| best.pt / last.pt | 最佳推理权重和恢复训练状态，二者用途分开 |
| bundles/epochXXXX/ | 正式 scope 的最佳整模型、冻结预处理、映射、完整训练/架构配置、FP32 推理精度、代码、依赖与文件校验和 |
| result.json | 最佳 epoch/指标、完成轮数、停止原因和产物路径；平台验证状态独立记录 |

`ptop.inference` 按 A 的官方 expected 审核输出。股票分批、日期分区和整批预测必须一致；真正模型预测数与 fallback 数单独记录。全缺失时间不能靠给 0 掩盖，默认排除全空输入并检查每时点真实覆盖。

C 的 Notebook 只加载模型包并读取当前 datasources 输入，返回 date/instrument/score。不 fit 预处理、不读取标签、不更新模型；Notebook、官方风格处理/压力/换手评价和平台实际提交仍由 C 完成。B 的研究 RankIC 不等同于含 BARRA、压力、换手的官方五项评分。

## 验证与候选准入

- 运行已有 A 桥接测试，保证 raw25/N1/官方 expected 合同兼容。
- 验证每个时序主干和编码器的 mask、因果性、梯度、不同 padding/微批、CPU 及授权 CUDA 路径。
- 验证完整训练、标签乱序连接、无监督样本、未来输入拒绝、早停及连续训练与 epoch 恢复的权重/优化器/scheduler 一致。
- 验证 v3/v4 模型包、校验损坏、冻结处理器、调用者 RNG、逐时点覆盖与流式分区。
- 真数据诊断比较只给工程证据；正式候选须再具备官方标签验证、多 seed 稳定性、C 指标与平台长度推理通过记录。

实际执行结果记录在 `b_local_experiments.md`；未执行或未获官方合同的验证不会标为通过。

最终集成检查、真实 CUDA 精度对照和资源测量见 `b_verification.md`。人工目标的工程检查完成后，下一步由 A/C 提供官方 expected、映射、LabelSpec/support 与评价模块，B 在同一入口运行正式时间折和多 seed 候选比较，再交 C 做平台完整区间验证。
