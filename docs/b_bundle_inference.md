# B 的模型包、断点与推理

本模块直接复用 A 的 `ptop.raw25.v1`、固定 `FEATURES` 顺序、冻结处理器、当日窗口和官方 `expected`。旧 `save_bundle/load_bundle` 调用与九个固定模型名保持兼容。

## 保存和读取推理包

```python
from ptop.bundle import save_bundle, load_bundle
from ptop.temporal import ModelSpec, create_score_model

spec = ModelSpec(name="B-Transformer", encoder="ALSTM",
                 width=64, layers=2, heads=4, lookback=120)
model = create_score_model(spec)  # 先由 Trainer 完成训练
save_bundle(
    output_path, model, frozen_processor,
    model_spec=spec.to_dict(), window_length=120,
    contracts=contracts, seed=17, training_record=training_record,
    instrument_map=instrument_map, weight_format="torch",
)
loaded_model, loaded_processor, manifest = load_bundle(output_path)
```

`contracts` 必须恰好包含 `source/sampling/universe/visibility/label/training_scope` 六个非空版本。处理器必须通过 A 的冻结检查；模型类、编码器类、模块配置、字段、输出宽度和每个权重的名称、类型、形状均与配置核对。`window_length` 与 `ModelSpec.lookback` 必须一致。

仅传 `model_name` 时，默认保存 v3 的 `model.json/preprocessor.json/manifest.json`。传 `model_spec`、选择 `weight_format="torch"` 或要求携带源码时，保存 v4；v4 默认包含完整 `source/ptop/` Python 源码、`source/jyc_e2e/*.py`、内置 Qlib 组件及许可、完整安装环境快照 `environment.json`、固定版本 `requirements.txt`、`THIRD_PARTY.md` 及四个核心依赖的上游 LICENSE/NOTICE 文本，可用 `include_source=False` 关闭源码携带。所有模型配置全量保存。旧名称仍表示 A 的固定架构。

v4 的 `model.pt` 只含张量状态，加载显式使用 `weights_only=True`；JSON 导出同样支持现代候选。`manifest.files` 的 SHA-256 覆盖权重、处理器、携带源码、环境快照和许可文本。加载严格检查 schema、字段顺序、Qlib 固定提交、内置组件版本、依赖版本、合同、文件清单和校验和。缺少权重或不支持的配置直接失败。

执行指纹绑定 schema、数据、窗口、处理器、编码器、固定模型、现代模型、时序主干、官方输入、输出审核、推理及内置 Qlib 组件的实际文件。即使张量形状相同，本地算法源码改变也会拒绝加载；携带的核心源码须与此指纹一致。C 需要原实现时，在新 Python 进程中将可信 bundle 的 `source` 加入模块路径再加载。关闭源码携带仍会验证本地执行指纹。依赖外部 Qlib 的旧 v1/v2 包不兼容，已有权重须用当前模型工厂和冻结处理器重新导出；旧 checkpoint 的环境及执行指纹也会被拒绝。

内置 Qlib 组件的 MIT 许可和来源说明随 `source/ptop/_vendor/qlib/` 提供，模型包另复制许可到 `licenses/qlib/LICENSE`。仓库中的固定提交许可副本 `src/ptop/licenses/QLIB-MIT.txt` 也核对 SHA-256 后随包提供。其余核心依赖使用安装 distribution 中的 LICENSE/NOTICE。

保存先完成合同检查，再在同一父目录暂存、同步文件并改名；已有 bundle 路径不能覆盖。每个训练 epoch 可使用独立目录。加载调用可信的本地模型工厂，不执行包内 Python 源码，也不会拟合处理器或安装依赖。携带源码可供审核和在预先安装依赖的环境复现；部署端自行将可信 `source` 加入 Python 模块路径。训练记录应保存运行命令、配置、数据范围和对应验证证据。

`Trainer` 导出的 `training_record` 保存完整 TrainingConfig、目标配置/尺度、最佳 epoch/指标、时间折，以及 `validation_precision="fp32"`、`inference_precision="fp32"`。训练 AMP 精度与交付推理精度分开记录；不要求 C 猜测训练使用了哪种精度或超参数。精确依赖包括 Torch build 后缀，CPU 与 CUDA 环境须采用相同已安装 build。

## 训练断点

```python
from ptop.checkpoint import save_checkpoint, load_checkpoint

save_checkpoint(
    checkpoint_path, model, optimizer,
    scheduler=scheduler, scaler=scaler,
    epoch=completed_epochs, global_step=global_step,
    best_metric=best_metric, bad_epochs=bad_epochs,
    config=immutable_training_config, model_spec=spec.to_dict(),
    preprocessor=frozen_processor, contracts=contracts, split=split,
    extra_state={"history": history, "best_epoch": best_epoch,
                 "sampler_generator": generator.get_state()},
)

state = load_checkpoint(
    checkpoint_path, model, optimizer,
    scheduler=scheduler, scaler=scaler,
    expected_config=immutable_training_config, expected_model_spec=spec,
    expected_contracts=contracts, expected_split=split,
    expected_preprocessor=frozen_processor,
)
completed_epochs = state.epoch
frozen_processor = state.preprocessor
generator.set_state(state.extra_state["sampler_generator"])
```

`epoch` 表示已完成的 epoch 数，恢复后从下一轮继续；在 epoch 边界保存。断点包含完整模型与优化器状态、scheduler、AMP scaler、epoch/global_step/最佳指标/耐心计数、Python/NumPy/Torch CPU/CUDA 随机状态及确定性后端设置，另保存不可变配置、冻结处理器、模型配置、数据切分、schema、字段、Qlib 与依赖版本。

优化器的每个参数组同时记录参数名称和顺序，恢复时严格匹配，避免 PyTorch 按位置给错层恢复动量。配置或数据切分不匹配、scheduler/scaler 启用配置变化、冻结统计变化都会在修改模型前失败。完整性校验同时覆盖张量和元数据；断点也保存并校验与推理包相同的执行源码指纹。文件通过同目录临时文件、fsync 和原子替换保存；保存失败会清理临时文件。

`config` 和 `split` 为有限 JSON 字典。`config` 保留最大训练轮数和 scheduler 总周期等不可变信息；暂时运行到第几轮属于进程控制参数。`extra_state` 可含普通字典/list/tuple、有限标量和稠密张量，用于训练历史、目标缩放、最佳轮次和 sampler generator。NumPy 数组、任意类实例及可执行对象不允许进入断点。使用独立 generator 的 DataLoader、sampler 或目标模块须显式保存并恢复其状态；不能用全局 RNG 状态替代它们的独立状态。

`restore_rng=False` 可仅恢复训练参数状态。恢复 CUDA RNG 要求相同的可见设备数；训练任务应固定设备可见性。重新构造模型前先按运行 seed 初始化，恢复调用再还原断点 RNG。

上述独立 API 保存公开 RNG，不序列化 cuDNN 私有 RNN dropout 缓存。`Trainer` 另外在每个 CUDA epoch 开始重新设种子，使完整训练与 epoch 恢复使用相同 dropout 流；不要将独立 `load_checkpoint` 等同于自动恢复这个私有缓存。双层 GRU 的实际 CUDA 续训检查及其环境限制见 `b_verification.md`。

## 纯推理加载断点

```python
from ptop.checkpoint import load_inference_checkpoint

model, processor, metadata = load_inference_checkpoint(
    best_checkpoint_path, device="cpu", expected_contracts=contracts
)
```

该入口只使用已训练权重与冻结处理器，返回 eval 模型和运行元数据，保留调用者的全局随机状态。它适合从 best checkpoint 导出 bundle；正在训练的 optimizer、scheduler 和 scaler 无须参与。

## 从窗口、原始分区或模型包推理

`predict_windows(model, batches, expected, ...)` 接收已经用保存处理器标准化的 A `WindowBatch`。输入仍为 NumPy float32 `[B,L,25]`、bool `observed` 和 bool `step_valid`；元数据至少包含 `sample_id/decision_id/instrument_id/decision_time/instrument`。B 模型的窗口长度必须与保存的 lookback 一致。

每个官方 expected 样本必须且只能被处理一次，包括完全缺行情的股票。函数核对样本键、模型字段对应的可预测 mask、当日可见分钟槽、结构填充和源行标识。真实缺失分钟保留 `step_valid=True`。各输入批次可以改变遍历顺序，输出仍按显式 expected 顺序排列。微批只拆分当前股票计算；A 的固定模型与 B 的现代模型在 eval 模式逐股票独立评分。调用者原有 train/eval 标志被保留。

```python
from ptop.inference import predict_canonical

result = predict_canonical(
    model, processor, raw, expected, trading_grid,
    start_date=start_date, end_date=end_date,
    window_length=120, microbatch_size=256, device=device,
    unpredictable_policy="drop",
)
output = result.output        # 恰好 date、instrument、score
records = result.records     # 每个 sample_id 的分数、状态和原因
exclusions = result.excluded
coverage = result.coverage
```

`predict_canonical` 仅 transform 保存的冻结处理器，然后用 A `WindowDataset/DecisionBatchSampler` 构造窗口。`predict_cloud` 先调用 A `fetch_inference_inputs`，只查询行情和官方截面；`predict_bundle` 则先恢复模型包，再使用其中保存的股票映射、窗口长度和处理器。提供额外映射或合同预期时必须与 bundle 一致。模型、schema、权重或频率缺失均直接失败。

不可预测项有两种显式策略：

| 策略 | 输出行为 | 逐样本状态 |
| --- | --- | --- |
| `drop` | 不发出该股票分数，保留排除记录 | `excluded` |
| `score0` | 发出零分，保留回退原因 | `fallback_zero` |

模型计算的状态是 `model`。所有缺观测原因均为 `no_observation_in_model_fields`；`result.fallback` 返回回退记录。

覆盖分母始终使用官方 expected。先对实际模型分数运行 A 的原始 `audit_output`，每个官方时点必须有模型分数且模型覆盖率至少 60%；然后审核最终输出。零分回退不能补足此门槛。覆盖表中的 `predicted/missing_rate` 指模型实际评分，`output_predicted/fallback_zero/fallback_rate/excluded/excluded_rate` 单独记录最终输出、回退及排除情况。整个时点模型无法预测或超过 40% 样本不可预测时，两种策略都会失败。

## 按时间分区流式运行

```python
from ptop.inference import InferencePartition, predict_bundle_partitions

partitions = [
    InferencePartition(start_date=day, end_date=day,
                       history_start=day, trading_grid=official_trading_grid)
    for day in evaluation_days
]
for result in predict_bundle_partitions(
    bundle_path, query, datasources, partitions=partitions,
    bar_end_is_available=visibility_contract, microbatch_size=256,
    device=device, expected_contracts=contracts,
):
    write_partition(result.output, result.records, result.coverage)
```

分区须为已排序、不重叠的完整日；历史起点不晚于分区起始日午夜。A 的窗口不跨日，但显式 trading_grid 仍须包括当天足够的历史分钟。包只加载一次，每个分区使用同一冻结统计和权重，迭代到该分区时才发起查询。根据可用内存选择小的分区；单个分区会持有原始表、转换表、窗口和审计结果。无须缓存跨年的展开窗口或学习后的 token。

`predict_cloud_partitions` 提供已有模型和处理器的同类入口。官方 expected 和输出审核逐分区执行，调用者保存或提交各个审计结果，确保所列分区覆盖实际授权评测范围。

## 验证范围

`tests/test_checkpoint_inference.py` 使用合成、带空股票与真实缺失分钟的官方形状样本，验证固定候选 v3 与全部四个 B 模型 v4 JSON/torch 往返、文件修改拒绝、配置与依赖检查、完整模型/optimizer/scheduler/CPU AMP/随机状态恢复、下一次训练更新复现、独立 sampler 状态、微批与重排、掩码/键/时序验证、零分回退覆盖门槛、冻结处理器和分区查询惰性。历史 CUDA 和真实窗口集成证据见 [B 验证记录](b_verification.md)；真正的官方收益标签与 DAI 平台提交仍需由比赛训练及 C 的部署记录提供。

模型包子分支合入前，2026-10-12 在 Python 3.12、Torch 2.8.0+cpu、NumPy 2.2.6、Pandas 2.2.3、PyArrow 20.0.0 和固定提交的 Qlib 0.9.8 环境，使用两个 CPU 线程运行以下三套测试：**213 passed，4 skipped**（39.56 秒）。跳过的四项是时序模型显式 opt-in 的 CUDA 测试。九个旧模型均验证 v1 精确权重往返；携带源码还在隔离工作目录的新 Python 进程中加载，确认实际导入 `source/ptop` 和 `source/jyc_e2e` 并复现 Transformer 分数。此为子分支阶段记录，最终整套测试为 CPU269/CUDA285 passed。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q \
  tests/test_checkpoint_inference.py tests/test_cloud_bridge.py tests/test_temporal.py
```
