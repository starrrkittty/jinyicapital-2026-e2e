# B 的本地分区训练与实验

`ptop.local_training` 把下载的月度 Feather 接到 A 的 raw25、冻结处理器和 `WindowBatch`。月文件逐个 Arrow batch 读取，请求的日期和股票被写入私人每日 raw cache；训练每次仅持有一天的原始、规范化和窗口索引。重叠窗口在取 batch 时展开，学习后的 encoder token 不进入缓存。

实际训练入口为 `prepare_local_fold`。它必须得到官方 expected 采样键、`instrument_map`、`LabelSpec`、显式分钟网格和 `TimeFold`，以及 A 合同格式的 targets，或官方 labels 加显式 support。没有这些信息会在读取行情前报出缺项。四份原始行情不能提供股票正式代码、官方有效采样集合或未经确认的目标定义。

```python
from ptop.local_training import LocalBarStore, prepare_local_fold

store = LocalBarStore(bar1m_dir, cache_dir=private_cache_dir)
prepared = prepare_local_fold(
    store, official_expected, official_targets,
    label_spec=label_spec, trading_grid=trading_grid, fold=fold,
    instrument_map=instrument_map, window_length=120,
    scale_review="实际尺度检查记录；如未人工检查需如实注明",
)
# 与 Trainer 的结构接口一致，窗口数组仍由 A 构造。
train_batches = prepared.train_loader(epoch=0)
validation_batches = prepared.validation_loader()
```

本地 fold 按午夜边界切分，保留完整 expected 集合。目标支持区间到达下一划分边界时，target 标为无效，并记录 `target_reaches_split_boundary`；不会因标签缺失删除官方采样键。行情缺失和实际模型字段全空由 A 的 predictable 判定分流。

N1 只使用训练端点可见的唯一原始行。每一天先通过 A 的 `fit_row_union` 取得引用行，再用 float64 的 pooled mean/M2 汇合各列 population mean/std。零方差列采用锁定 Qlib `ZScoreNorm` 的 mean=0、std=1、ignore=True 语义。状态记录 field order、valid_counts、排序唯一行指纹、实现版本、真实尺度检查说明，再用 A 的 `FrozenPreprocessor.from_state_dict` 恢复。这个路径是分区等价实现，不能称为调用了 A 的整表 `fit`。测试将它与实际 A/Qlib 小样本拟合比较；验证和推理仅复用冻结状态。

模型对比先用最长窗口拟合一份共同 N1，再用 `prepared.with_window(60)` 和 `prepared.with_window(120)`。所有候选保持相同采样键、目标有效性和目标值。调度器核验每次验证产物的完整 sample_id 集合、实际窗口 predictable 和有效预测键；不同覆盖不会被悄悄取交集。

对于浮点常量，简单按日汇合的舍入误差可能改变 Qlib 严格 `std==0` 的 ignore。实现额外保留精确常量归约：无缺失时使用虚拟 broadcast view，带缺失时将一列临时写入私人磁盘，采用最多 16,384 个元素的读取缓冲和原生 NumPy 叶归约，临时文件随拟合完成删除。排序与划分依据固定 [NumPy 2.2.6 pairwise reduction](https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/loops_utils.h.src)；实际算术由同版本 NumPy 执行。测试覆盖不对称分区中的货币常量及带缺失常量，不能仅用整数零字段证明等价。

第二轮固定依赖回归发现，仅复现递归 kernel 在超过 NumPy iterator 缓冲长度后仍会改变末位舍入。归约因此也保留实际 `numpy.getbufsize()` 外层分段，并记录该数值；测试检查 8191/8192/8193 等边界，以及实际锁定 Qlib 对 12,345 行缺失常量的 mean/std/ignore。

`ReadPolicy` 默认拒绝未知负值。下载文件中的盘口委托笔数是有符号 int16；显式 `negative_order_policy="mask_known"` 时，仅六个 `*_num_orders{1,2,3}` 的负值会改为缺失并计入读取审计。不会取绝对值、转换 uint16、加 65536 或推测原值。价格、金额和成交量等其他负值继续由 A 拒绝。正数笔数保持真实值，缺失 mask 继续进入 encoder。

可以在固定训练依赖环境执行真实行情工程诊断：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /home/data4/hts/jyc2026_training_env/bin/python \
  examples/b_real_data_smoke.py --config configs/b_diagnostic_cpu.json \
  --output-dir /home/data4/hts/jyc2026/b_runs/diagnostic-v1 --mode matrix --threads 2
```

配置使用 2020-01-02、01-03 训练，01-06 验证，原始 ID 为 11、16、19、30。`10:00/11:00/14:00` 是显式工程端点，只能证明它们在观察网格中，不能当作官方合法采样点。`bar_end_is_available=true` 是这次诊断的可见性假设，元数据明确保留 `production_visibility_verified=False`。

B 额外记录原始 available_at 的来源。A 规范化按显式 bar-end 假设生成的列，在送入 A 窗口构造器前恢复为这个显式声明，因而 batch 正确记录 `visibility_basis="bar_end_only"` 与 `availability_assumption="bar_end_equals_available_at"`；真实来源提供的 available_at 才记录为 `provided_available_at`。这项来源区分不会改变实际窗口数值或可见性筛选。

诊断目标是 sample_id 的 SHA256 映射到固定数值，**不是未来收益，不是官方标签，诊断 loss/IC 不代表金融预测效果**。默认矩阵为同一个 A ALSTM encoder 的 B-GRU、B-Transformer、B-TCN、B-SelectiveSSM，L=60/120、seeds=7/17/29，各 2 epochs、独立 600 秒预算。可用 `--models`、`--lookbacks`、`--seeds` 和 `--epochs` 选择有明确独立预算的子矩阵。

产物位于私人 output_dir：`prepared_manifest.json`、`normalization.json`、`split_audit.parquet`、真实读取/前向的 `benchmark.json`，以及 matrix 下的训练日志、checkpoint、完整验证结果、`comparison.csv`、`seed_summary.csv` 和 `report.md`。报告明确区分已完成、失败和预算超限；超限不替换成更简单模型，只有上一完整 epoch 的 checkpoint 保证可恢复。`--resume-completed` 会验证已完成实验的协议和验证文件哈希后跳过。

计划协议同时锁定 A/B 执行源码 SHA256；旧实现的结果不能在修改算法后无声当作当前已完成实验。模型记录实际总参数和真实注册模块的参数数；原生 A 为 encoder/rnn/att_net/fc_out，现代 B 为 encoder/backbone/pool/head，无参数模块保留 0。不把相同 width/layers 误称为相同容量。TCN 感受野与池化方式应一同阅读；默认 mean pooling 汇总整个有效窗口中的局部卷积结果。

benchmark 分别记录 loader 等待、张量传输、前向耗时、样本吞吐和内存。CUDA 测量显式同步并重置分配峰值；CPU RSS 是整个进程的累计峰值，不冒称为某一个模型独立占用。没有官方标签时不计算或宣称平台 BARRA、压力场景、换手百分位或五项总分。

正式入口模板为 `configs/b_official_template.json`。null 项需要填入真正的官方文件；它们是可操作的缺项而非可训练的假数据。正式标签准备好后，保持相同时间折和目标合同运行矩阵，选择开发验证候选，再由模型包与提交入口冻结交接。

统一 `.py` CLI 也可选择 `data_factory="ptop.local_training:local_training_factory"`，data 中填写上述官方文件路径及私人 cache_dir；正式 factory 的窗口字段取真实 model.encoder.fields。工程诊断可运行 `python -m ptop.train --config configs/b_train_cli_diagnostic.json --output /私人目录/cli-smoke --device cpu --threads 2`。此配置显式选择 `diagnostic_training_factory`，输出可恢复训练 checkpoint，但不导出包含假股票代码的正式模型包。

当前验证记录将由真实运行追加；只有确实执行并检查的结果才写入实验结论。

第一轮实际读取检查已在 pandas 2.2.3 / NumPy 2.2.6 / Arrow 20.0.0 环境执行。202001 月文件被流式拆出所选三日和四个原始 ID，每日 1,000 行、25 字段、24,808 个有效观测；没有股票代码映射，六个委托笔数字段在此范围没有负值。月拆分耗时约 0.58 秒，三日 A 规范化读取约 0.076 秒，进程累计 RSS 峰值约 291 MiB。这是实际 raw 分区与 A 规范化检查，未执行 N1/训练的阶段不能当作训练验证通过。私人记录：`/home/data4/hts/jyc2026/b_work/b_ingest_read_probe/report.json`。

前一版完整执行证据基于集成 core `8d6b25b`，固定 CPU PyTorch 2.8 与上述依赖、2 个计算线程，产物目录为 `/home/data4/hts/jyc2026/b_runs/diagnostic-cpu-final-v3`。24 格真实窗口矩阵全部完成，失败和预算超限均为 0；每格 2 个完整 epoch、12 次优化更新，共 48 个 epoch、288 次更新。每个 epoch 的 12 个验证键、best/last checkpoint 完整性和日志均已逐文件检查；24 个 best checkpoint 重新加载后的原始验证分数与各自保存结果逐位一致。每格 encoder、backbone、head 三个参数组相对正确 seed 初始化均实际发生更新。

共同 N1 使用 1,520 个唯一训练可见原始行；同一个冻结指纹、有效预测键指纹和训练目标尺度在 24 格间保持一致。所有实际 batch 的可见性来源断言为 bar-end 显式假设，production verification 为 False。上述记录可查 `artifact_audit.json`，不能仅凭 run.json 中的 completed 判定训练成功。

| 主干 | 总参数 | encoder | backbone | head |
| --- | ---: | ---: | ---: | ---: |
| B-GRU | 20,161 | 3,264 | 15,808 | 1,089 |
| B-Transformer | 31,905 | 3,264 | 27,552 | 1,089 |
| B-TCN | 19,041 | 3,264 | 14,688 | 1,089 |
| B-SelectiveSSM | 18,465 | 3,264 | 14,112 | 1,089 |

同次 GRU-L120 数据读取和前向 benchmark 共 12 样本，约 0.0847 秒，其中 loader 等待 0.0618 秒、传输 0.00023 秒、前向 0.0225 秒，约 142 样本/秒。整个 24 格进程的累计 RSS 峰值约 584 MiB；这个小范围检查不能证明全股票池或平台 3 小时时限。

真实 `.py` CLI 的 GRU-L120-seed7 恢复对照也已执行：一次完整 2 epoch，与在第 1 epoch 保存后重新启动进程恢复至第 2 epoch，比对 model、optimizer、optimizer layout、scheduler、scaler 和 RNG 六项均逐位一致，模型最大绝对差为 0。这项结论仅覆盖该固定环境和真实小样例；记录位于 `/home/data4/hts/jyc2026/b_runs/diagnostic-cpu-resume-v3/resume_audit.json`。早期 core 的相同对照曾出现约 6e-8 的末位差，保留旧产物而未将其冒称为当前验证。

这些检查证明分区、联合训练、权重保存、恢复与实验记录可运行。人工目标上的 loss/IC 仍不是金融预测表现；正式候选选择和比赛成绩需要真正的官方键、映射、标签合同与支持区间。

最终 CPU 证据更新到集成 core `4a4d18f`：验证使用与冻结推理入口相同的 FP32 forward，最新现代矩阵保存到 `/home/data4/hts/jyc2026/b_runs/diagnostic-cpu-final-v4`。24/24 格完成，失败和预算超限仍均为 0，共 48 个完整 epoch、288 次优化更新。`artifact_audit.json` 逐文件检查两轮完整验证、best/last 权重、梯度日志、全部三个训练参数组更新和各 best checkpoint 的 FP32 分数逐位回放。`selection_audit.json` 将完整 best-epoch 验证表与保存比较表核对，再按原 microbatch 与冻结 target_scale 从保存分数重算 loss；24 格最佳选择数值的最大绝对差为 0。

这次 CPU 矩阵的每格训练范围、目标、同键覆盖与 N1 都与前述诊断相同。GRU-L120 的 12 样本 benchmark 约 0.0908 秒，其中 loader 等待 0.0664 秒、张量传输 0.00030 秒、前向 0.0238 秒，约 132 样本/秒。24 格记录的逐实验耗时合计约 44.33 秒；进程累计 RSS 峰值约 578 MiB。这些测量仍只覆盖三日四个原始股票 ID 的小范围工程检查。

同一 core 的真实 CLI 恢复产物在 `/home/data4/hts/jyc2026/b_runs/diagnostic-cpu-resume-v4`，完整训练对照在 `/home/data4/hts/jyc2026/b_runs/diagnostic-cpu-uninterrupted-cli-v4`。第一轮保存后重新启动进程恢复至第二轮，与完整两轮相比，model、optimizer、optimizer layout、scheduler、scaler、RNG 六项均逐位一致，最大模型差为 0。`resume_audit.json` 保留实际比较和容差；逐位一致只描述这一个固定环境的 GRU-L120-seed7 检查。

原生 A 的 R0/R1 参照也已实际执行：集成 `6cd71f1` 仅修复实验入口对 backbone/head 模块名的假定，训练、数据和模型实现与 `4a4d18f` 的源码指纹逐项相同。默认 runner 在 `/home/data4/hts/jyc2026/b_runs/diagnostic-cpu-a-baselines-final` 完成 R0/R1 × L60/120 × seeds7/17/29 共 12 格，各两轮、独立 600 秒预算；失败和预算超限均为 0，共 24 个完整 epoch、144 次更新。复制的私人配置仅改变 model_specs，其余训练设置和数据配置保持一致。

| A 参照 | 总参数 | encoder | rnn | att_net | fc_out |
| --- | ---: | ---: | ---: | ---: | ---: |
| R0 | 47,297 | 0 | 47,232 | — | 65 |
| R1 | 55,425 | 3,264 | 49,920 | 2,112 | 129 |

A 的 raw encoder 无训练参数，不能声称 R0 encoder 权重发生更新。12 格所有实际可训练模块均更新，包括 R1 的 encoder、rnn、att_net、fc_out；全部最佳权重 FP32 分数回放逐位一致，保存分数重算的最佳 loss 最大绝对差也是 0。A/B 同时核验了采样键、训练/验证目标指纹、目标尺度、冻结 N1、字段、时间 fold、训练预算和数据/模型/Trainer 源码。调度器的参数统计源码指纹按各自运行版本保留，没有将旧矩阵伪装成新源码完成结果。

A 参照使用原生 hidden_size=64：R0 为 raw encoder 加 GRU 末端池化，R1 为 ALSTM encoder 加 GRU attention。现代 B 使用 width=32、ALSTM encoder 和 mean pooling；两组容量和池化不同，因此这是原生参照与现代候选的工程对比，不能视作只改变主干的等容量消融。全部 36 格共享 1,520 个唯一训练可见行的同一 N1、12 个验证键和 `0.005828910631702901` 的训练目标尺度；batch 可见性来源仍明确是未获生产验证的 bar-end 假设。

私人合并记录位于 `/home/data4/hts/jyc2026/b_runs/diagnostic-cpu-a-b-engineering-comparison-final`：`comparison.csv` 保存 36 格模块参数、实际训练/验证耗时、loader 等待和权重审计来源；`summary.csv` 按模型/窗口汇总三 seed 工程耗时；`manifest.json` 锁定两套来源和审计文件哈希；`report.md` 说明容量及范围。共 72 个完整 epoch、432 次优化更新。该工程报告不按人工 loss/IC 给金融能力排名，正式金融预测表现和比赛分数仍需官方输入合同。

入口兼容性回归在固定 CPU 环境实际通过 25 项本地测试，其中原生 R0、R1 与现代 B-GRU 都走默认 runner 完成完整 epoch、训练参数更新、完整采样键和保存分数回放检查。源码修复只涉及实验分组统计及可选 TCN 感受野，未改变训练或模型算法。
