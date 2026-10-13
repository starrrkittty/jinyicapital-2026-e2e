"""Schema-independent components for minute-series research.

本包只保留仍在使用的研究组件：

- ``jyc_e2e.losses``           第一目标函数（预测目标与损失配置）；
- ``jyc_e2e.research_metrics`` 本地诊断指标。

早期的 A 层原型（``contracts`` / ``preprocessing`` / ``model``）已移除，不再在此导出；
需要这些符号的旧代码应改为直接依赖 ``ptop`` 的对应实现。此前 ``__init__`` 仍导入这些
已删模块，会让 ``import jyc_e2e`` 直接失败，并连带阻断 ``ptop.training`` / ``ptop.train``
/ ``ptop.objectives``。
"""

__all__: list[str] = []
