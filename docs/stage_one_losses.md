# 第一阶段损失函数：接入说明

`src/jyc_e2e/losses.py` 中的 `PredictionLossConfig` 和 `PredictionObjective` 实现收益回归与同截面排序。当前不包含换手、夏普和压力损失。

## 输入与输出合同

- `scores`、`target`：浮点 `[M]`，均为原始收益单位。真实标签单位由 A 核实；模型最后一层不加 sigmoid。
- `times`：int64 `[M]`，同一决策时点跨股票必须使用相同 ID。
- `target_valid`：可选 bool `[M]`，True 表示标签有效；没有掩码时默认全部有效。有效标签必须有限，被屏蔽标签可以为 NaN。所有模型分数仍必须有限。
- 所有输入在同一设备，股票/时点主键唯一；数据层负责查重。
- 返回 `(loss, diagnostics)`，只对 loss 调用 backward。日志中的张量已 detach。
- 全部标签无效时抛错，由训练入口跳过并记录，避免静默训练无效 batch。

## 数学目标和尺度

固定尺度 a=`target_scale`，回归误差 e=(s-y)/a。MSE 使用 e²；Huber 对小误差采用平方惩罚，大误差采用线性惩罚。`huber_delta` 的单位是缩放后的收益单位。

同一时点内，标签差大于 `min_target_gap` 的股票对参与排序，方向 q=sign(y_i-y_j)：

$$L_{rank,t}=\operatorname{mean}_{(i,j)}\operatorname{softplus}\left(-q_{ij}\frac{s_i-s_j}{a\tau}\right),\qquad L=L_{reg}+\lambda L_{rank}$$

tau=`rank_temperature`。`min_target_gap` 使用原始收益单位。收益相同的股票对跳过；没有有效对的时点不参与排序均值。全部时点没有有效对时，排序项为可求导的零，回归仍计算。

默认回归先按每个时点平均，再对时点平均；排序对有有效对的时点等权平均。可配置 `reduction="sample"` 恢复按样本平均的回归。两项覆盖的时点数可能不同，应查看日志。

大截面超过配对上限时，不生成全部 O(N²) 股票对，而是均匀、有放回地抽取 `max_pairs_per_time` 个不同股票之间的候选对，再过滤标签差。因此有效对数可能低于上限，也可能抽到重复股票对。小截面精确枚举。采样依赖 PyTorch RNG，也可传入同设备 `torch.Generator`；B 保存恢复其状态。

`target_scale=1` 不改变单位。小收益标签下，`huber_delta=1` 可能几乎都落在平方区间。尺度应由训练标签或明确配置确定，随检查点保存，验证时不能重新拟合。损失值和梯度尺度会随 a 改变，lambda 和温度需要配套比较。

本实现模型仍输出原始收益 s，只在损失内部缩放，因此推理不再乘 a。若其他代码已经让模型输出 y/a，应先统一接口，不能重复缩放。

## B 的训练接入

```python
from dataclasses import asdict
from jyc_e2e.losses import PredictionLossConfig, PredictionObjective

# 数值仅为 API 示例，不是比赛最优配置。
cfg = PredictionLossConfig(
    regression="huber", reduction="time",
    target_scale=0.01, huber_delta=1.0,
    rank_weight=0.1, rank_temperature=1.0,
    min_target_gap=0.0, max_pairs_per_time=8192,
)
objective = PredictionObjective(cfg)
scores = model(batch["x"], batch["observed"])
loss, report = objective(
    scores, batch["target"], batch["decision_id"],
    target_valid=batch.get("target_valid"),
)
optimizer.zero_grad()
loss.backward()
optimizer.step()
log_values = {k: v.item() if hasattr(v, "item") else v
              for k, v in report.items()}
checkpoint_config = asdict(cfg)  # 连同模型、处理状态和 RNG 保存。
```

实验顺序：MSE 且 rank_weight=0 → Huber → Huber+排序。与官方原始 MSE 对照时设置 `target_scale=1, reduction="sample"`。默认配置是 Huber、time 平均、无排序，默认参数不代表经过真实数据调优。

旧 `score_loss` 保留原签名，仍为原始单位 Huber、sample 平均、可选排序。`pairwise_rank_loss` 保留原参数并新增 scale、temperature、mask 和 generator。大截面采样方式改变，不保证与旧版随机序列或损失逐位一致。

## 日志与协作

日志包括总损失、回归、排序、加权排序，有效/屏蔽样本数、有效时点数、排序时点数、候选/有效配对数、单股票时点数、无有效对时点数。排序关闭时配对计数为零，表示未计算，不表示数据不存在。

A 提供标签单位、时点标识、标签掩码和截面采样；B 挂载 objective，记录日志、梯度和配置。股票微批次不能各块单独排序后声称等价于完整截面。混合精度训练时，损失数学运算至少使用 float32。

标签仅用于 loss，绝不进入模型 forward；标签有效性不决定推理股票池。真实数据及含数据日志留在本人授权环境。

本次未运行 PyTorch 测试或训练，尚未验证梯度和实测效果。后续接入时核对跨时点隔离、标签掩码、梯度及保存加载，然后以固定时间切分比较官方成绩。
