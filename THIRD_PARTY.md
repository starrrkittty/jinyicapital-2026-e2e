# 第三方来源

Qlib 固定提交为 [54355232463878d2eebb91fe0ee5fa7fa1f5976c](https://github.com/microsoft/qlib/tree/54355232463878d2eebb91fe0ee5fa7fa1f5976c)，使用 MIT 许可。所需模型、标准化、填空、采样和索引工具以源码保存在 [src/ptop/_vendor/qlib/](src/ptop/_vendor/qlib/)，内置组件版本为 1。[MIT 许可](src/ptop/_vendor/qlib/LICENSE)随源码及安装包提供；原始路径与修改见 [来源说明](src/ptop/_vendor/qlib/SOURCE.md)。

模型参数、初始化及采样计算沿用该提交。处理器使用项目自身的 JSON 状态管理，由 `FrozenPreprocessor` 直接组织拟合、标准化和填空。与原版的对照验证位于 [validation/test_qlib_equivalence.py](validation/test_qlib_equivalence.py)，仅此专项验证需要安装原版 Qlib。

DeepLOB 分支依据 [DeepLOB 论文](https://arxiv.org/html/1808.03668)和三档输入配置，独立组织 PyTorch 卷积、Inception、LSTM 与回归头。Patch 分支参考 [PatchTST 论文](https://arxiv.org/html/2211.14730)的线性分块形式，后接 GRU 与注意力。这些分支的本赛道效果需实际训练确定。

B 的 Transformer 与 GRU 使用 PyTorch 模块；TCN 和 SelectiveSSM 为独立组织的研究实现，参考 [TCN](https://arxiv.org/abs/1803.01271) 与 [Mamba](https://arxiv.org/abs/2312.00752) 的结构思想，没有携带 Mamba CUDA 扩展或外部预训练权重。论文、官方实现与本实现的具体区别见 [时序研究](docs/b_temporal_research.md)。模型包记录完整环境版本，携带四个核心依赖的 LICENSE/NOTICE，以及内置 Qlib 组件的 MIT 许可与来源说明。
