# 面向榜单指标的损失研究与实验协议

本文区分本次 PR 的工程能力、研究候选和官方待确认口径。目标是预测每只股票未来 30 分钟 VWAP 收益的相对强弱；工程测试或公开 synthetic 示例不能证明金融预测有效，更不能证明比赛提分。未取得官方 evaluator、最终评分权重和样本外实验结果前，不把任意训练代理称为“最优损失”或“官方总分”。

## 1. 本次实现的边界

本次保留默认 Huber，增加可配置 pairwise 和完整决策截面 Pearson IC。组合持仓、交易成本、换手约束、Sharpe/ICIR 比值、SoftRank、Group DRO 和 BARRA 中性化均不在此次实现范围。既有 `research_metrics.py` 的换手与 Sharpe 仍为通用研究诊断，不是官方回测器。

| 能力 | 本次接口与含义 | 明确限制 |
| --- | --- | --- |
| 归一化回归 | 模型预测 `q`，目标 `z=y/a`；`a=target_scale` 仅训练集拟合或显式提供并保存 | q 不是原始收益；混合排序目标后不保证校准 |
| 排序 | 温度、原始收益 gap 门槛、pair 上限及 uniform/target_gap 权重 | 局部代理，不等价于 RankIC、多空收益或官方总分 |
| Pearson IC | 同一时点有效标签截面的稳定相关损失 | 稳定化相关不是严格 Pearson；未复制官方预处理 |
| `objective_scope=microbatch` | 兼容默认：每个微批求损失并更新 | 微批内排序只覆盖微批股票，不能称完整截面 |
| `objective_scope=decision` | 一个完整决策 batch 分块 forward，保留图，拼接后一次 loss/backward/step | 不提供有界激活内存；不是连续多时点持仓训练 |
| 验证指标 | 原 `ic_mean`/`ic_ir` 继续表示 RankIC；新增显式 `rank_ic_mean`/`rank_ic_ir` 及 `pearson_ic_mean`/`pearson_ic_ir` | 保持旧选择语义；不能因为名称为 IC 就认定是官方口径 |

`decision` 要求输入 batch 真正覆盖一个决策时点的预期样本。该时点标签无效或完全不可预测的行可以不参与监督，但“监督有效截面”与“未来组合可交易股票池”必须区分。本次不生成持仓。

## 2. 本次目标的数学定义

记 `q` 为模型输出，`y` 为原始收益标签，`z=y/a`，基础损失为 Huber(q-z)。a 必须随训练状态保存，不得随验证区间重新拟合。`huber_delta`、排序温度和 gap 权重都应结合这个归一化尺度解释。

### 2.1 有界采样 pairwise

同一时点 t 的候选对先按原始收益筛选：

$$
P_t=\{(i,j): i\ne j,\ |y_{i,t}-y_{j,t}|>g_{\min}\},\qquad
d_{ij,t}=\operatorname{sign}(y_{i,t}-y_{j,t}),
$$

$$
\ell_{ij,t}=\operatorname{softplus}\left(-d_{ij,t}(q_{i,t}-q_{j,t})/\tau\right),
\qquad
L_{rank,t}=\frac{\sum_{(i,j)\in P_t}\omega_{ij,t}\ell_{ij,t}}
{\sum_{(i,j)\in P_t}\omega_{ij,t}}.
$$

`rank_weighting=uniform` 时权重为 1；`target_gap` 时取

$$
\omega_{ij,t}=\min\left(|y_{i,t}-y_{j,t}|/a,\ C\right).
$$

这里 `rank_min_target_gap` 使用原始 y 单位，`rank_gap_cap=C` 使用无量纲归一化 gap。以权重和归一化，使 cap 不只是任意放大整项损失；其主要作用是改变各 pair 相对权重。零 gap 不参与。无有效 pair 时报告跳过，不构造虚假的监督。

这是受 RankNet 概率排序思路启发的适配：原研究提出可微配对排序成本，使用搜索排序实验；本项目的收益 gap 筛选、加权及金融用途是额外设计，不能归为论文的金融实证结论。[Burges 等，Learning to Rank using Gradient Descent，作者机构原页](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)

更贴近金融截面的研究是 Poh 等的 cross-sectional learning-to-rank：它比较配对／列表排序学习用于截面动量组合，支持把“学习股票间次序”作为待验证方向。论文的市场、持有期、特征与策略评估不同于本题未来 30 分钟 VWAP 标签及未知官方处理流程；它不证明本项目的 gap weighting、Pearson 混合权重或参数取值有效。[Poh 等，Building Cross-Sectional Systematic Strategies By Learning to Rank，原论文](https://arxiv.org/pdf/2012.07149)

pair 上限限制配对构造内存，不限制模型 forward 激活内存。大截面直接有界抽样。通用 `PredictionObjective` 提供实际候选／有效 pair 计数；Trainer 当前使用的 `RegressionObjective` 仅返回 Huber、rank、IC、样本数及 IC 有效／跳过时点数，不返回实际 pair 数。Trainer 的配置契约保存 pair cap、温度、gap 等参数，运行保存训练种子，但不能据此声称每步实际 pair 已记录；需要该诊断时应另行接入。若有放回采样可能重复 pair，计数不等于独立信息量。gap weighting 的抽样加权分子／分母比值通常不是全 pair 加权平均的无偏估计，应视为有界采样近似并检查抽样敏感性。gap weighting 强调较大的已实现收益差，但该差也可能来自噪声／极端事件；不是“收益差越大越可靠”的保证。

### 2.2 稳定化 Pearson IC

在每个有效截面内中心化 q 和 z，按总体矩计算协方差和方差：

$$
\rho_{\epsilon,t}=\frac{\operatorname{mean}_i[(q_i-\bar q)(z_i-\bar z)]}
{\sqrt{\operatorname{Var}_i(q)+\epsilon^2}
\sqrt{\operatorname{Var}_i(z)+\epsilon^2}}.
$$

相关目标鼓励提高该值；具体损失为可评价时点的 `mean(1-rho_epsilon)`，在有效时点集合固定时与负平均相关具有相同梯度。有效股票少于 3、标签完全常数时跳过并计数。常数预测不能跳过：有效标签下保留分母稳定化和对应梯度，否则模型可能用常数输出规避惩罚。若所有组都不可评价，IC 项为与图连接的零，Huber 继续提供监督。

实现先用标签 `max==min` 判断常数，避免常数标签的浮点中心化误差被误认为真实方差；中心化后标签方差因下溢为零时也跳过。epsilon 的平方必须在工作 dtype 中可表示为有限正数，否则明确报错；中心化二阶矩、协方差或加上稳定项后的方差溢出也明确拒绝。有限输入不意味着所有中间运算都有限，数值拒绝不能解释为成功训练或额外金融鲁棒性。

总目标为

$$
L=L_{Huber}+\lambda_r L_{rank}+\lambda_{IC}L_{IC}.
$$

`ic_epsilon=1e-4` 是数值设置，不是实证最优值。分母稳定化削弱极小方差截面的相关值，也打破严格尺度不变性；归一化后的 q、z 尺度必须与 epsilon 一并记录。训练采用稳定化相关，验证计算普通 Pearson；常数预测在验证中仍应视为未定义并报告覆盖，不能静默填为零。三股票只是数学下限，不代表估计可靠。

Pearson 重视线性关系与极端值，RankIC 重视秩。回归、uniform pairwise、gap weighting 和 Pearson 对不同错误敏感，彼此没有普遍优劣关系。多项混合时应记录损失分量、有效组数和梯度规模，避免归一化回归损失与 O(1) 排序／相关项盲目等权相加。

## 3. 配置与隔离对照

下列是可合并到完整运行配置的 `training` 片段，不是独立可运行的配置文件。所有数值仅用于清晰展示实验，不能称推荐最优参数。保持完整配置中的字段、模型、窗口、数据切分、预处理及训练标签尺度相同。

**A：原微批 Huber 基线。**

```json
{"training": {"objective_scope": "microbatch", "rank_weight": 0.0, "ic_weight": 0.0, "selection_metric": "rank_ic_mean", "mode": "max", "seed": 7}}
```

**B：完整截面 Huber，对 scope 变化的零权重对照。**

```json
{"training": {"objective_scope": "decision", "rank_weight": 0.0, "ic_weight": 0.0, "selection_metric": "rank_ic_mean", "mode": "max", "seed": 7}}
```

**C：B 加 uniform pairwise。**

```json
{"training": {"objective_scope": "decision", "rank_weight": 0.01, "rank_temperature": 1.0, "rank_min_target_gap": 0.0, "rank_max_pairs_per_time": 8192, "rank_weighting": "uniform", "rank_gap_cap": 1.0, "ic_weight": 0.0, "selection_metric": "rank_ic_mean", "mode": "max", "seed": 7}}
```

**D：只将 C 改为 gap weighting。**

```json
{"training": {"objective_scope": "decision", "rank_weight": 0.01, "rank_temperature": 1.0, "rank_min_target_gap": 0.0, "rank_max_pairs_per_time": 8192, "rank_weighting": "target_gap", "rank_gap_cap": 1.0, "ic_weight": 0.0, "selection_metric": "rank_ic_mean", "mode": "max", "seed": 7}}
```

**E：B 仅加 Pearson，先不混 pairwise。**

```json
{"training": {"objective_scope": "decision", "rank_weight": 0.0, "ic_weight": 0.01, "ic_epsilon": 0.0001, "selection_metric": "rank_ic_mean", "mode": "max", "seed": 7}}
```

C−B 衡量新增配对目标，D−C 衡量 gap weighting，E−B 衡量 Pearson。A−B 衡量组织 batch 和优化步变化，不能归因于金融目标。以上统一按 RankIC 选检查点以隔离目标变化；后续如改 `selection_metric=pearson_ic_mean`，需作为独立的选择策略实验，不能悄悄同时改变。

默认配置保留 `objective_scope=microbatch`、`rank_weight=0`、`ic_weight=0`；排序默认温度 1、原始 gap 门槛 0、pair 上限 8192、uniform 权重、gap cap 1，IC epsilon 为 1e-4。参数变化要进入日志及 checkpoint 契约，恢复训练不得静默改变目标。

旧 JSON 省略新增字段时仍使用默认值，这属于配置输入兼容；不代表跨源码版本 checkpoint 兼容。当前训练／恢复有严格源码指纹校验，旧代码产生的 checkpoint 在源码改变后会被拒绝，本 PR 不提供迁移或绕过该校验。新运行在同一源码及相同配置契约下的 resume 应单独验证。

### 3.1 计算量与模型选择协议

完整截面模式每个决策时点只更新一次；微批模式每时点可更新多次。相同 epoch 不代表相同 optimizer step、学习率轨迹或训练预算。记录有效样本、时点、更新次数、墙钟时间、峰值显存及学习率；主对照可先固定数据遍历次数，另做预算可比的敏感性实验，不把两种公平性标准混在一张结果表中。

`decision` 分块 forward 仅降低单次算子输入规模：所有块的计算图在 backward 前保留，激活内存仍随完整截面增长。不能宣传为有界内存梯度累积。dropout 的随机数消费和 batch 相关层也可能使不同分块结果有差异；梯度等价测试应使用确定性模型，随机层差异需另行描述。

使用按时间的训练／验证／锁定 holdout，训练标签支持区间不得跨进验证；合法历史上下文可引用切分前记录。所有尺度仅训练集拟合。先小范围筛选，再至少三个种子比较关键候选，按连续日期 block 报告 IC、RankIC、各自 IR、覆盖和期间差异。分钟样本与重叠 30 分钟标签高度相关，不把股票×分钟数当独立样本计算显著性。固定 holdout 在候选基本确定后评估，不能不断调参直至 holdout 变好。

接受标准包括：无泄漏／非法输出、完整加载和恢复、可解释覆盖、资源可承受、样本外预测指标稳定；若官方口径可用，同时记录官方分。只有换手或单个短期指标变好不构成保留理由。未获得官方总分公式前，报告多指标取舍，不自造等权综合分。

## 4. 候选的接受／暂缓理由

| 目标 | 本次结论 | 理由及启动条件 |
| --- | --- | --- |
| Huber | 默认保留 | 收益幅度监督与既有训练兼容；delta 仍依赖标签尺度 |
| uniform pairwise | 可选实现 | 有界 pair 采样，易审计；先完整截面对照 |
| gap-weighted pairwise | 可选实现 | 可检验收益差重要性；可能过度追逐极端噪声 |
| Pearson IC | 可选实现 | 接近一种明确截面评价；保留 Huber 防尺度／常数退化 |
| soft rank／可微秩 | 暂缓 | 官方 IC 类型未确认；额外平滑参数、计算与代理差异 |
| 软权重换手 | 暂缓 | 缺连续时点对齐、真实股票池、持仓和边界契约 |
| 净成本效用／Sharpe | 暂缓 | 缺可实现收益、成本及资金核算；短窗比值噪声大 |
| ICIR 直接训练 | 暂缓 | 跨时间分母不适合当前独立决策 batch |
| Group DRO | 暂缓 | 缺可信场景定义、足够独立日期及最差组泛化证据 |
| BARRA 残差化目标 | 暂缓 | 官方暴露、可见性、回归权重与处理顺序均未明确 |

这里主要研究候选是 Blondel 等 2020 的可微排序／排名算子，通过 permutahedron 投影及 isotonic optimization 构造，论文给出 O(n log n) 时间、O(n) 空间复杂度。精确地说，论文 §6.3 用真实完整排名 r 与软排名之间的平方损失 `0.5*||r-r_soft(q)||²`，并称其为 differentiable Spearman；这与另行定义 `-corr(r_soft(q),rank(y))` 的中心化、归一化相关代理不能自动视为同一损失。硬完整排列的方差固定，平方距离与 Spearman 等价；软排名方差可变，该等价关系不能直接照搬。本项目若采用其中任一候选，需要分别定义目标、平滑强度、并列处理及有效截面规则，不把归一化相关适配器说成论文 §6.3 原式。[Blondel 等，Fast Differentiable Sorting and Ranking，原论文 §6.3](https://proceedings.mlr.press/v119/blondel20a/blondel20a.pdf)

作者参考库的 PyTorch wrapper 调用 `values.detach().numpy()` 和 NumPy 后端，再使用自定义 backward；它不是可直接接 CUDA tensor 的原生 GPU 实现。若后续采用该参考实现，需要明确 CPU/GPU 数据传输、性能、工作 dtype 及梯度验证，不能由论文算子复杂度推断当前训练 GPU 吞吐。[作者参考实现 pytorch_ops.py](https://github.com/google-research/fast-soft-sort/blob/master/fast_soft_sort/pytorch_ops.py)

Taylor 等 2008 的专名 **SoftRank** 使用随机平滑构造排名分布研究信息检索指标，是另一种方法，不能与 Blondel 算子或简单 pairwise sigmoid 混称。两类论文都不自动证明本题 RankIC 或金融收益改善。[Taylor 等原论文](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/SoftRankWsdm08Submitted.pdf)

## 5. 后续组合目标的必要前提

### 5.1 换手、股票池和持仓漂移

研究软权重可以取中心化分数除以截面 L1，但它与等权分位数组合不同，也没有复制 BARRA。非退化时净敞口为零、总绝对敞口为 1；近常数输出可能缩进 epsilon 区域，降低实际敞口以逃避换手，或产生很大梯度。高换手惩罚还能奖励固定股票排序。必须同时观察 score 方差、软总敞口、单股集中度、硬组合换手和预测能力。

换手应为目标持仓与交易前已有持仓之差：

$$
TO_t=\tfrac12\|w_t-\widetilde w_{t^-}\|_1.
$$

忽略漂移可以作为命名清楚的代理；实际漂移使用上次调仓至当前已实现收益、现金与净值计算，多空组合不能机械套用纯多仓权重归一化。未来标签有效性只决定可监督／可评价覆盖，不能决定当时可交易股票池，也不能删掉缺标签股票后悄悄重新归一化。

新增接口至少包括稳定 `instrument_id`、当时可知股票池、`transition_valid`、调仓时钟、午休／跨日规则、首期建仓、新增退出／停牌处理。随机 batch 邻接不代表时间邻接；缓存旧模型参数生成的上期权重再 detach 是另一种近似，不等价于同一参数下联合求导。先做连续采样但换手权重为 0 的对照，再研究非零惩罚。

### 5.2 净成本 Sharpe 与效用

Lim 等的式 35 将波动率缩放后相邻仓位的绝对变化成本扣入收益，再优化净成本 Sharpe；一般不等价于 `-Sharpe(gross)+lambda*turnover`。其日频期货与本题股票 30 分钟预测不同。[Lim、Zohren、Roberts 原论文，式 16、35](https://arxiv.org/pdf/1904.04912)

本题后续可研究：

$$
p_t=w_t^\top r_{t\to t^+}-\sum_i\kappa_{i,t}|w_{i,t}-\widetilde w_{i,t^-}|,
\quad
L_U=-\bar p+\gamma\widehat{\operatorname{Var}}(p),
\quad
L_S=-\bar p/\sqrt{\widehat{\operatorname{Var}}(p)+\epsilon_p^2}.
$$

若 kappa 按每单位买卖成交金额定义，成本乘完整 L1，常数成本换成 Half-L1 为 `2*kappa*TO`。效用式是本项目建议适配器，避免比值奇异性，但 gamma 依赖收益尺度，风险估计仍需足够跨度。收益、风险、交易成本和持有成本的权衡可参见 [Boyd 等原框架](https://stanford.edu/~boyd/papers/cvx_portfolio.html)；该框架不证明本项目训练损失能提升榜单。

未来 30 分钟 VWAP 标签不是逐分钟账户收益。逐分钟 `w_t*y_30m,t` 窗口重叠，只能先称标签组合收益代理；不能直接累计为自融资 PnL。真实模拟需要单步持有期收益，或资本分配明确的错峰 30 分钟子组合。Sharpe 的平方根年化依赖收益过程条件，自相关可改变其估计与排名。[Lo，The Statistics of Sharpe Ratios，作者专题页](https://alo.mit.edu/research-page/the-statistics-of-sharpe-ratios/)

短窗均值为负时，负 Sharpe 可能通过增加波动改善目标；epsilon 只缓和数值奇异性，不能消除该激励。K=2、4 可验证换手接口，不足以支撑可信 Sharpe／ICIR 分母。梯度累积小批比值也不等于大窗口比值。可研究 `-mean(IC)+eta*Var(IC)`，但这不是最大化 ICIR，且方差项可能偏好稳定的零 IC，必须保留预测能力约束。

### 5.3 压力场景与 Group DRO

Group DRO 的原目标是 `min_theta max_q sum_g q_g L_g`，q 位于概率单纯形；原算法交替更新模型与组权重，不是每个 batch 直接挑当前最差组。原论文也强调正则／早停对于最差组泛化的重要性，其分类实证不是金融保证。[Sagawa 等原论文，式 3、4及算法 1](https://arxiv.org/pdf/1911.08731)

本题先固定日期／过去可见状态分组，报告每组独立日期数、覆盖、IC 与组合代理，再试组均衡。高波动组可能天然 Huber 损失更大，DRO 会追逐不可预测噪声；极小组可能长期主导梯度。训练分组使用标签不自动等于泄漏，但未来状态不得进入当时输入／持仓，不得用 holdout 调分组。训练组鲁棒也不能保证全新压力事件鲁棒。官方压力定义未确认时，只能称研究代理。

### 5.4 风格剔除必须等待官方资料

中心化分数只能去截距，不会去规模／行业等暴露。若取得当时可知暴露 B、截面回归权重 W，可研究残差 `s_perp=[I-B(B'WB)^dagger B'W]s`；这是数学适配器，不代表官方 BARRA。残差信号再做非线性选股也不保证最终持仓风格敞口为零。MSCI 的原方法分别处理风格、行业、国家暴露与换手，说明这些约束不等同于资金净敞口。[MSCI Market Neutral Barra Factor Indices 原方法](https://www.msci.com/eqb/methodology/meth_docs/MSCI_Market_Neutral_Factor_Indices_Methodology.pdf)

向官方确认暴露版本及可见时间、缺失股票、回归权重、截距／行业编码、去极值→标准化→中性化的顺序、IC 对原标签还是残差标签、最终持仓生成及成本。信息缺失时不自行猜测，并列保存原始／官方处理后指标，防止原始 Pearson 上升却主要依赖随后被剔除的风格。

## 6. 证明范围与交付证据

公开 synthetic 测试适合证明：跨块 pair 和完整截面相关参与梯度；确定性模型分块／完整 forward 的损失与梯度一致；有效标签掩码、常数标签、常数预测、小截面、极端尺度和 AMP 数值行为；旧默认路径兼容；配置与恢复契约完整。股票重排应保持确定性完整数学目标不变；有界随机 pair 抽样仅保证相同种子／顺序复现，不声称每次采样都对排列严格不变。

工程测试不证明 IC 提升、换手改善或压力期稳健。后续仅在授权数据环境做固定切分、锁定 holdout 与三种子日期 block 实验；输出比较的实际运行日志、有效覆盖、成本／组合口径和资源。无官方总分时报告 Pareto 取舍，不用论文、合成测试或低训练损失替代榜单证据。

### 6.1 本次实际验证（2026-10-21）

环境：Python 3.11.14、Torch 2.11.0+cu128、NumPy 2.2.6、pandas 2.2.3、PyArrow 20.0.0、pytest 8.4.2；Qlib 安装并核验仓库固定提交 `54355232463878d2eebb91fe0ee5fa7fa1f5976c`。GPU 为 RTX 5070 Laptop（8 GiB）。本次没有修改依赖清单；测试 Torch build 不等于正式环境要求的 2.8.0，正式交付前仍须在 B/C 一致的固定 build 下复验。

实际分组执行如下，使用独立 `--basetemp` 和 pytest cache 目录；GPU 命令设置 `PTOP_TEST_CUDA=1`。

| 执行范围 | 实际结果 |
| --- | --- |
| `test_losses.py test_objectives.py test_research_metrics.py test_training.py` | 84 passed；随后补充最后一个溢出回归 |
| `test_decision_training.py` | 15 passed，2 个 CUDA 用例待显式启用 |
| `test_checkpoint_inference.py test_cloud_bridge.py test_local_training.py test_temporal.py` | 242 passed，4 个 CUDA 用例待显式启用 |
| `test_cuda_training.py test_decision_training.py -k 'cuda or mixed_objective'` | 15 passed，14 deselected |
| 最终 `test_losses.py test_objectives.py` | 55 passed，包含新增的稳定方差相加溢出检查 |
| 最终 `test_temporal.py test_decision_training.py -k 'isolated_cuda or mixed_objective'` | 7 passed，162 deselected |

`pytest --collect-only -q` 共收集 360 项；上述分组去重覆盖全部 360 项，其中 18 项为显式启用的 CUDA 检查。分组计数存在重复，不将各行通过数直接相加。新增加的四个测试文件共 75 项。五段文档 JSON 均通过 `TrainingConfig.from_dict`，`git diff --check` 通过。

行为回归先在独立进程恢复基线 `59b4f84` 的旧 `_phase` 方法：新跨块测试实际观察到 `[2,1,2,1]` 个样本分别进入目标，而要求的完整截面是 `[3,3]`，因此失败。当前实现通过同一测试。数值回归也先复现了常数标签舍入、二阶矩溢出、epsilon 平方异常与稳定项相加溢出，再通过修复验证。测试使用公开合成输入，未运行官方真实收益实验或提交榜单。
