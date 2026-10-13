# B 时序模型：实现、适用性和验证依据

本分支以 A 的 raw25 编码合同为准。`TemporalScoreModel` 将 A 原编码器注册为 `self.encoder`，同一前向中接时序主干、池化、两层 MLP 和单个 score；优化器使用整个模型的参数，训练产生的 token 不缓存到数据集。官方 y、时间切分和评估属于训练流水线，模型没有自行构造收益标签或技术因子。

## 配置和调用

```python
from ptop.temporal import ModelSpec, create_score_model

spec = ModelSpec(
    name="B-Transformer", encoder="ALSTM", width=64, layers=2,
    heads=4, pooling="mean", lookback=120, dropout=0.1,
)
model = create_score_model(spec).to(device)
score = model(x, observed, step_valid)  # [B]
configuration = spec.to_dict()         # JSON 可读、字段完整
restored = create_score_model(ModelSpec.from_dict(configuration))
```

`x` 是 float32 `[B,L,25]`，字段顺序为 A 的 `FEATURES`。`observed` 是 bool `[B,L,25]`；缺失值为零。`step_valid` 是 bool `[B,L]`，结构填充只在左侧。真实分钟缺行情时 `step_valid=True`，即便全部字段的 `observed=False`。模型入口复用 A 的校验，按 `model.encoder.fields` 判断可预测性；字段子集全空必须先分流，不能靠网络 bias 生成该样本的预测。

所有模型支持 `ModelSpec.to_dict/from_dict`。配置包含 `name、encoder、width、layers、pooling、lookback、dropout、heads、ff_multiplier、kernel_size、state_dim、head_width、causal、fields`。`pooling` 为 last、mean 或 attention。自定义 `fields` 仅供 Raw/ALSTM 使用；序列张量仍保持 25 列。直接构造时字段用 tuple，JSON 中用 list。非法类型、未知字段、Transformer 宽度不能整除 heads、以及旧模型中会被忽略的架构参数会明确报错。

| 模型名称 | 主干 | 主要成本和配置 |
| --- | --- | --- |
| `B-GRU` | PyTorch 单向多层 GRU，打包有效分钟，LayerNorm | 序列上递归，宽度成本约平方；`width/layers/dropout` |
| `B-Transformer` | 独立初始化的 Pre-LN Transformer 层，运行时正弦位置编码 | 注意力约随 token 数平方增长；`heads/ff_multiplier/causal` |
| `B-TCN` | 两次因果卷积的残差块，dilation 逐层加倍 | 可并行卷积；`kernel_size/layers` 决定感受野 |
| `B-SelectiveSSM` | 输入相关的稳定对角状态转移、因果 depthwise 卷积、门控残差 | 纯 PyTorch 顺序 scan；`state_dim/kernel_size/layers` |

编码器可选 `Raw、ALSTM、ALSTM-Book、Net64、Net192、GroupedNet、Patch1、Patch5、DeepLOB`，全部来自 A。`ModelSpec()` 默认仍是 `R1`；A 原有 `R0、R1、R1-Book、Net64、Net192、GroupedNet、Patch1、Patch5、DeepLOB` 按原工厂返回原模型，保持原类型和 state_dict。旧名称固定架构；需要调整宽度、池化或主干时使用 B 名称。

## 窗口、状态和股票分批

主干仅将结构左填充移动到右侧，保留每个真实槽的顺序与长度；每个 forward 为每个样本新建 GRU/SSM 状态。预测不会把前一只股票或前一个窗口的状态带到后一只股票。推理必须使用 `model.eval()` 和 `torch.inference_mode()`；eval 下重排股票、逐股票 microbatch 与整体 batch 结果相同，浮点误差除外。A 的 Net/GroupedNet 使用 BatchNorm，训练时依赖批次；这是原编码器行为，eval 使用冻结运行统计。其他新增主干没有跨股票 BatchNorm。

新模型接受 `L <= spec.lookback`，超出时立即报错，避免训练配置与推理窗口静默错配。位置编码由当前 token 长度生成，没有隐藏的 512/1024 个 learned-position 表限制。测试配置验证 2049 槽容量；实际 A 的窗口仍只在当日显式 trading_grid 取数据，这个容量测试不意味着允许跨日构造窗口。

`causal=True` 默认让 Transformer 的 token 只看更早历史。设为 False 可在整段已知历史内做双向注意力；是否未来泄漏取决于窗口可见性，A 的 available_at 和决策时点约束仍必须成立。A DeepLOBEncoder 在已知历史内有非因果卷积，因此“主干因果”不会自动使整套 DeepLOB 组合的中间 token 因果。整套 score 仍只读取窗口中的已知原始行情。

Patch5 使用 A 原有、端点锚定的 5 槽拼块，把 values、observed、step_valid 交给可训练投影；120 槽约得到 24 个 token。它不是重新读取 bar5m，也不是计算手工 5m 因子。A 的 trading_grid 为当日分钟网格，午休不自动产生槽：patch 可能同时含上午和下午的已知分钟。当前三张量接口不携带 bar_times，本实现保留 A 的标准 tokenization，不能宣称连续自然分钟或 session 隔离。若实验要求块不跨午休，数据端需选择已确认的 grid/端点和窗口对齐，或由 A 扩展明确的 session 合同；不能把真实缺分钟设为结构填充来模拟 session 边界。

TCN 单个输出的感受野为 `1 + 2*(kernel_size-1)*(2**layers-1)`。默认两层、kernel=3 的感受野为 13 token；last 池化只会使用该局部范围。对 120 个分钟 token，5 层的感受野为 125；对 Patch5 的 24 token，3 层为 29。mean/attention 池化会汇总全窗口的局部表示。对比实验需记录这些配置，不能将感受野不足引起的差异解释成 TCN 的一般结论。

## 原论文和官方实现的研究结论

GRU 的更新门和重置门为共享的时序状态提供基础，适合作为有限窗口的低成本基线。本模型只复用其门控时序模块，不复现机器翻译的 encoder-decoder 任务。[Cho 等原论文](https://arxiv.org/abs/1406.1078)

Transformer 使用多头注意力、位置编码和前馈层。本实现是面向回归 score 的轻量编码器，只处理同一股票的已知历史，没有语言 decoder 或外部预训练权重。将其用于本比赛是任务适配，论文的翻译指标不能证明金融排序效果。[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

PatchTST 的主要设计包含 patching 和 channel independence。这里复用 A 的多字段 PatchEncoder，并接 Transformer；字段在 A 的投影中混合，且只有一个 score 头，所以命名为 `B-Transformer + Patch5`，不宣称完整复现 PatchTST。其缩短 token 序列的思路适合先验证速度/显存成本，再以同一日期、标签和随机种子比较收益。[PatchTST 原论文](https://arxiv.org/abs/2211.14730)、[官方代码](https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/layers/PatchTST_backbone.py)

TCN 原研究采用因果、膨胀卷积和残差连接。本实现采用这些结构思想，并独立加入 per-token LayerNorm、GELU 和显式槽位 mask；没有复制官方 weight-normalization 配置、初始化或训练实验。适用性的证据必须来自比赛时间验证集，而不是把其他任务的实验结论照搬。[TCN 原论文](https://arxiv.org/abs/1803.01271)、[官方实现](https://github.com/locuslab/TCN/blob/master/TCN/tcn.py)

Mamba 的关键是让 Δ、B、C 随输入变化，并用硬件优化的 scan 处理状态。本候选采用输入相关的 Δ/B/C，但用正值 softplus rate、tanh selectors 和精确零阶保持积分独立组织对角递归：`h_t=exp(-Δ_t*rate)*h_(t-1) + (1-exp(-Δ_t*rate))/rate * B_t * u_t`，输出结合 `C_t*h_t`、skip 和门控。它没有 Mamba 的低秩 Δ 投影、内部 expansion 或融合 kernel，所以明确叫 `SelectiveSSM`。负指数转移保证未训练初始递归的稳定性；参数梯度、长窗口和评分有限性仍需实测。[Mamba 原论文 §3](https://arxiv.org/html/2312.00752v2)、[官方 Mamba 模块](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py)

官方 Mamba 库另有 CUDA selective-scan 扩展及对应构建要求。候选当前只用项目已有 PyTorch，可离线导入、CPU 运行，不需现场编译扩展。它的 scan 在时间上顺序执行，训练仍保留递归的 autograd 图；“状态维度固定、序列 FLOPs 线性”不等于 GPU 高吞吐。禁止用官方 Mamba 的速度数字描述此实现。[官方安装说明](https://github.com/state-spaces/mamba)

TTT 将临时 hidden state 视为一个学习器，在每个序列内执行自监督梯度更新；外层训练参数与序列内部 fast weights 需要区分。将梯度运算函数化、每个股票窗口重新初始化 fast weights，在技术上可以保持外层提交参数固定，但比赛是否认可这种测试时内部训练需要官方口径，不能自行判定合规。其官方纯 PyTorch 教学实现也提示训练较慢。本分支保留研究结论，未注册未经比赛验证的 TTT 候选。继续研究的门槛是明确合规、固定外层权重、全空/缺分钟 mask、按股票分批等价和正式长度的成本证明。[TTT 原论文 §2](https://arxiv.org/html/2407.04620v2)、[官方 PyTorch 实现与限制](https://github.com/test-time-training/ttt-lm-pytorch)

CSDI 通过条件扩散学习缺失值分布，TimeGrad 用扩散生成概率时间序列预测；DiffSTG 则结合图结构和时序去噪来做概率时空预测。这些论文的 diffusion score 通常是分布的梯度，不能等同比赛的标量排序 score。当前 A 接口没有股票图或联合截面张量；要接 DiffSTG，需要事先确定合法图来源、历史股票池变化以及跨股票关系的时间可见性。构建全历史收益相关图会引入另一类泄漏风险。以上是对本比赛接口的适用性推断，原论文没有验证本赛。[CSDI 原论文](https://arxiv.org/abs/2107.03502)、[TimeGrad 原论文](https://proceedings.mlr.press/v139/rasul21a.html)、[DiffSTG 原论文](https://arxiv.org/html/2301.13629v4)

扩散路线还需解决多次去噪的推理成本与随机性。CSDI 的官方代码在每个 sample 内执行多轮随机采样；一个全局 RNG seed 不会自动保证股票重排或 microbatch 后的输出一致。若未来实验采用扩散，需按稳定样本键提供噪声、将生成分布聚合为有确定方向的 score、验证序列模型未跨样本状态、并做完整评测区间计时。当前四个候选的 eval 无随机采样，适合先把比赛训练/推理链路和有效模型对比做实。[CSDI 官方采样实现](https://github.com/ermongroup/CSDI/blob/main/main_model.py)

## 迭代与证据

第一轮实现落地四个主干及九种 A 编码器组合。代码审查确认 Transformer 层逐层独立初始化、用压紧后的每样本真实槽位置计算位置编码、TCN 每层输出有结构 mask；这些决策对应 batch 重排、左填充不变性和跨长度比较的验收目标。第二轮静态审查将容量测试从仅有 Patch5 扩展到 Raw/2049 槽，避免用约 410 个 patch token 的测试间接声称已经覆盖了超出 1024 token 的位置容量。第三轮接口审查发现 last/mean 池化及 causal True/False 并不改变 state_dict 形状，因此给完整模型、池化、Transformer 加入显示实际配置的 `extra_repr`，并通知 bundle 分支显式校验 `model.spec`，避免错误配置保存后改变预测。所有主干都显式保留真实缺分钟，SSM 状态为 forward 局部张量。

测试位于 `tests/test_temporal.py`，覆盖完整 A 九模型同种子逐 tensor 重建、配置往返和非法组合、encoder/backbone/head 梯度和参数更新、全部 A 编码器的可训练性、全部编码器与四主干 eval microbatch/股票重排、左填充、缺分钟、字段子集全空分流、last/mean/attention、2049 窗口、反复推理的参数/buffer/RNG 不变以及 Raw 组合的 token 因果性。另有 SSM 常量输入的闭式零阶保持解、CPU BF16 训练和 eval/inference_mode 验证、同形权重的配置可见性检查。

本次固定 CPU 环境为 Python 3.12.3、Torch 2.8.0+cpu、Qlib 包版本 0.9.8、NumPy 2.2.6、pandas 2.2.3、Arrow 20.0.0、pytest 8.4.2。运行结果为 **148 passed，4 skipped，7.45 秒**：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/data4/hts/jyc2026_cpu_validation_env/bin/python -m pytest tests/test_temporal.py -q
```

四个 skip 均为未指定 GPU 时的隔离 CUDA 检查。只有显式设置 `PTOP_TEST_CUDA=1` 且分配 `CUDA_VISIBLE_DEVICES` 才执行；GPU/AMP/续训由总集成分支统一验证，本分支不将 CPU 结果描述为 CUDA 结果。此前对 PyTorch CPU autocast 与融合 eval 路径的审查疑问没有在固定 2.8 环境复现：四个主干 BF16 训练和验证均通过，未因此修改 Transformer 的公开层路径或全局 MHA 开关。

## 合成拟合迭代：池化会改变端点信息保留

四个主干各运行 mean/last 两种池化和 seed 7/19/41，共 24 次实验。共享 512 个训练样本、128 个独立合成验证样本、16 槽序列、原始 25 字段和 mask；长度在 8 至 16 之间，倒数第六槽是真实但全缺行情的分钟。人工 teacher 为 `0.5*最后一槽 close + 0.25*倒数第二槽 volume + 0.15*倒数第四槽 ask_volume1`。这只是过去输入映射的容量诊断，不是比赛未来 VWAP 标签，也没有提供金融预测效果证据。

配置统一为 A ALSTMEncoder、width=32、layers=2、head_width=32、dropout=0、heads=4、kernel=3、state_dim=16，AdamW 学习率 0.003、weight_decay=1e-4、MSE、梯度裁剪 1.0、batch=32、80 步。每十步计算合成验证 MSE，报告其中最小值；表中为三个种子的均值 ± 样本标准差。

| 模型 | mean 池化最佳合成验证 MSE | last 池化最佳合成验证 MSE | 实测参数数 |
| --- | ---: | ---: | ---: |
| B-GRU | 0.31654 ± 0.00117 | 0.05444 ± 0.00926 | 20,161 |
| B-Transformer | 0.31297 ± 0.00255 | 0.10614 ± 0.00654 | 31,905 |
| B-TCN | 0.26921 ± 0.00744 | 0.02848 ± 0.00531 | 19,041 |
| B-SelectiveSSM | 0.29847 ± 0.00981 | 0.05458 ± 0.00459 | 18,465 |

last 池化的每次运行都相对初始合成验证误差降低至少 67%；mean 在这个端点 teacher 上弱得多。这个结果支持把池化作为真实验证中的对照变量，并保留可配置的 last/mean/attention；它不支持按这张表选择比赛最优主干。模型默认 last，真实源诊断若使用 mean，需另外复查端点信息是否被稀释。全部模型的 A 编码器有 3,264 个参数，预测头有 1,089 个参数；四个主干分别为 15,808、27,552、14,688、14,112，所以同 width/layers 并不等于同参数容量。

以下代码在项目根目录、同一固定环境与 `PYTHONPATH=src` 下复现 24 次合成拟合；benchmark 的原始记录另保存在本机 `work/temporal-study.json`。

```python
import torch
from ptop.schema import FEATURES
from ptop.temporal import MODERN_MODEL_NAMES, ModelSpec, create_score_model

torch.set_num_threads(2)

def data(count, seed):
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(8, 17, (count,), generator=generator)
    valid = torch.arange(16)[None, :] >= 16 - lengths[:, None]
    observed = (torch.rand((count, 16, 25), generator=generator) > 0.2) & valid[..., None]
    for offset, field in ((1, "close"), (2, "volume"), (4, "ask_volume1")):
        observed[:, -offset, FEATURES.index(field)] = True
    observed[:, -6, :] = False
    x = torch.randn((count, 16, 25), generator=generator).masked_fill(~observed, 0)
    y = (0.5*x[:, -1, FEATURES.index("close")] + 0.25*x[:, -2, FEATURES.index("volume")]
         + 0.15*x[:, -4, FEATURES.index("ask_volume1")])
    return (x, observed, valid), y

train, target = data(512, 201)
validation, truth = data(128, 202)
for name in MODERN_MODEL_NAMES:
    for pooling in ("mean", "last"):
        for seed in (7, 19, 41):
            torch.manual_seed(seed)
            model = create_score_model(ModelSpec(
                name=name, encoder="ALSTM", width=32, layers=2,
                head_width=32, pooling=pooling, lookback=240,
            ))  # 实际训练长度16；240为本容量实验的允许上限。
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
            model.eval()
            with torch.inference_mode():
                initial = float(torch.nn.functional.mse_loss(model(*validation), truth))
            best = initial
            generator = torch.Generator().manual_seed(500 + seed)
            order = torch.randperm(512, generator=generator)
            for step in range(80):
                if step % 16 == 0:
                    order = torch.randperm(512, generator=generator)
                chosen = order[(step % 16)*32:((step % 16)+1)*32]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(*(value[chosen] for value in train)), target[chosen])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if (step+1) % 10 == 0:
                    model.eval()
                    with torch.inference_mode():
                        best = min(best, float(torch.nn.functional.mse_loss(model(*validation), truth)))
            print(name, pooling, seed, initial, best)
```

## CPU token 成本

同一配置在 CPU 两线程、batch=32、全部为真实时间槽、120 槽输入、eval/inference_mode 下，三次预热后计时十二次：GRU 中位 13.72 ms、Transformer 31.03 ms、TCN 7.09 ms、SelectiveSSM 42.10 ms。SSM 虽然参数更少、scan 的序列复杂度线性，在这个长度和实现上依然最慢，不能用官方融合 Mamba 的速度结论替代本实现测量。

另一个独立 token 对照使用同样的 Transformer 宽度、层数和头，比较 A ALSTMEncoder 与 A Patch5；三次预热后十二次 forward 的中位如下：

| 原始槽数 | ALSTM token数 | ALSTM 时间 | Patch5 token数 | Patch5 时间 |
| ---: | ---: | ---: | ---: | ---: |
| 60 | 60 | 8.82 ms | 12 | 3.52 ms |
| 120 | 120 | 38.15 ms | 24 | 5.38 ms |
| 240 | 240 | 121.73 ms | 48 | 10.08 ms |

Patch5 组合有 45,025 个参数，较 ALSTM 组合多 13,120 个，不能把两个编码器的学习容量混为一谈。这些数值只计本机 tensor 前向，未含查询、预处理、窗口、股票映射或输出检验；并行任务和系统负载也会改变耗时。它们支持研究 token 压缩和主干成本，不构成指定评测环境或三小时总时限的通过证明。
