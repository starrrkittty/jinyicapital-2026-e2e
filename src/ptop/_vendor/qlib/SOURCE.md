# 内置 Qlib 组件

来源为 Microsoft Qlib 提交 `54355232463878d2eebb91fe0ee5fa7fa1f5976c`，组件版本为 1。版权归 Microsoft，使用本目录中的 MIT 许可；随源码和安装包保留许可。

| 项目文件 | 原始文件及组件 |
| --- | --- |
| `models.py` | `qlib/contrib/model/pytorch_alstm.py:ALSTMModel`、`pytorch_gru.py:GRUModel`、`pytorch_nn.py:Net` |
| `processors.py` | `qlib/data/dataset/processor.py:Fillna`、`ZScoreNorm`、`RobustZScoreNorm`；`qlib/constant.py:EPS` |
| `sampler.py` | `qlib/data/dataset/__init__.py:TSDataSampler` |
| `utils.py` | `qlib/data/dataset/utils.py:get_level_index`、`fetch_df_by_index`；`processor.py:get_group_columns`；`qlib/utils/__init__.py:np_ffill`、`time_to_slc_point`、`lazy_sort_index` |

适配内容：调整内部导入路径，保留模型、处理器和采样器方法的计算语句；省略上游说明文档及无关训练包装类。处理器不继承上游序列化基类，由 `FrozenPreprocessor` 使用已有 JSON 状态管理。`FrozenPreprocessor` 在独立且按索引排序的副本上拟合，变换依次执行标准化、Fillna、字段选择和原顺序恢复。`lazy_sort_index` 保留固定 pandas 2.2.3 使用的分支。

使用这些组件时无需安装 `pyqlib`。固定候选模型包 v3、参数化配置模型包 v4 记录内置组件版本及原始提交，预处理 JSON 的字段与统计语义保持一致。内置组件计算发生修改时须增加组件版本并重新执行专项对照测试。

原始文件 SHA-256（用于对照验证原版来源）：

| 原始路径 | SHA-256 |
| --- | --- |
| `qlib/contrib/model/pytorch_alstm.py` | `f7a19f9a2a6551f0659cc821cce30aaaa865a703874e7e83c4fbc8e0d1275a6b` |
| `qlib/contrib/model/pytorch_gru.py` | `75cc1c38dcbc4dfda90105d203f60399aec60c154360eac8eddb38f605efda7e` |
| `qlib/contrib/model/pytorch_nn.py` | `d4c7fec9a79ee3fecc8b467369a078d3e7d61b5a6c0bbe29e188192c2d8c7c21` |
| `qlib/data/dataset/processor.py` | `424af44e81c99467171c2d0b7978fb9f95d3b4c8b743a535ff4fe45d708cf517` |
| `qlib/data/dataset/__init__.py` | `f81de9c050e91e4f25ed0fb802273a7ca7dedc104d580ee30ebdd2a733fc68a8` |
| `qlib/data/dataset/utils.py` | `c556538a0de06597390001b2e9de361f87af11c96853c67ba9e8af4a88b40024` |
| `qlib/utils/__init__.py` | `f423a1b86444f8455a348ddce69c142dcbfa3ba68131897539c031bcf595de70` |
| `qlib/constant.py` | `92b0cc8d027742591558bbaee935909be9a58aa9c5541b714f41f0e1338fde94` |
