# 数据处理与输入编码

[src/ptop/](src/ptop/) 提供云端 1 分钟数据读取、N1 标准化、当日历史窗口和可训练编码器。默认窗口为 120 分钟，使用 25 个原始字段及观测掩码。

## 安装

推荐 Python 3.12。云端先准备匹配 GPU 环境的 PyTorch 2.8.0，然后安装：

```bash
python -m pip install -r requirements-a.txt
```

所需 Qlib 组件已包含在 `src/ptop/_vendor/qlib/`，随 `ptop` 自动使用，无需安装 Qlib 或准备 C++ 编译器。`dai` 使用平台提供的客户端。训练与推理保持依赖版本一致，正式运行时不下载依赖。

直接上传源码时保留完整 `src/ptop/` 目录及其中的许可文件，并让 `src/` 位于 Python 导入路径中。内置组件的来源与修改见 [THIRD_PARTY.md](THIRD_PARTY.md)。

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

需要完整参照模型时，使用 `ptop.bundle.create_model("R1")`；调用参数相同，返回 `[B]` 分数。模型保存与恢复使用 `save_bundle/load_bundle`，仅支持代码中已有的模型配置。当前模型包格式为 v2，与 v1 包不兼容。

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

上述函数已有实现和本地接口测试。完整 epoch 循环、业务模型组装和平台提交入口由调用方组织；真实 DAI/Linux/CUDA 运行尚未验证。依赖来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。
