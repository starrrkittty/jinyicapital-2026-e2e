"""Optional-dependency access plus a preflight report.

Qlib is needed by A's legacy encoders (``ALSTMModel`` / ``Net`` / ``GRUModel``), by the
daily window sampler, and by the frozen preprocessor's transform path. It is **not**
guaranteed to exist in every runtime, and the submission environment must not reach the
network, so the dependency is made explicit instead of being triggered by a bare module
import: every Qlib symbol is fetched lazily at the point of use, and failures carry a
clear message rather than an ``ImportError`` from an unrelated import line.

What this does and does not buy you:

- ``import ptop.temporal`` (and the modern ``B-*`` score models) no longer requires Qlib
  as long as the chosen encoder is one that does not itself use a Qlib class
  (``Raw`` / ``Patch1`` / ``Patch5``).
- Training and inference **still** require Qlib, because ``FrozenPreprocessor`` applies
  the frozen state through Qlib's ``DataHandlerLP`` so the normalisation semantics stay
  bit-identical to the official implementation. See
  ``submission/requirements-inference.txt`` for the open decision on that.
"""

import functools
import importlib

from .schema import QLIB_COMMIT, ContractError

QLIB_HINT = (f"此功能需要 Qlib（本仓库锁定 QLIB_COMMIT={QLIB_COMMIT}）。"
             "离线环境请在镜像内预装；或改用不需要 Qlib 的编码器与预处理路径。"
             "详见 submission/requirements-inference.txt。")


@functools.lru_cache(maxsize=None)
def _module(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as error:  # pragma: no cover - depends on the runtime image
        raise ContractError(f"无法导入 {name}。{QLIB_HINT}") from error


def qlib_available() -> bool:
    try:
        importlib.import_module("qlib")
    except ImportError:
        return False
    return True


def alstm_model_class():
    """``qlib.contrib.model.pytorch_alstm.ALSTMModel`` (A's ALSTM encoder source)."""
    return _module("qlib.contrib.model.pytorch_alstm").ALSTMModel


def net_class():
    """``qlib.contrib.model.pytorch_nn.Net`` (A's unified grouped encoders)."""
    return _module("qlib.contrib.model.pytorch_nn").Net


def gru_model_class():
    """``qlib.contrib.model.pytorch_gru.GRUModel`` (A's R0 reference model)."""
    return _module("qlib.contrib.model.pytorch_gru").GRUModel


def ts_data_sampler_class():
    """``qlib.data.dataset.TSDataSampler`` (daily-window sampler)."""
    return _module("qlib.data.dataset").TSDataSampler


def qlib_epsilon() -> float:
    """``qlib.constant.EPS``; kept lazy because the frozen-state check compares against it."""
    return _module("qlib.constant").EPS


def qlib_data_handler():
    """``qlib.data.dataset.handler.DataHandlerLP``."""
    return _module("qlib.data.dataset.handler").DataHandlerLP


def qlib_static_loader():
    """``qlib.data.dataset.loader.StaticDataLoader``."""
    return _module("qlib.data.dataset.loader").StaticDataLoader


def qlib_processors():
    """The ``qlib.data.dataset.processor`` module (``Fillna`` / ``ZScoreNorm`` / ...)."""
    return _module("qlib.data.dataset.processor")


# 需要 Qlib 的功能点，供预检报告与排障使用。
QLIB_FEATURES = {
    "encoder:ALSTM": "A 的 ALSTM 编码器复用 qlib ALSTMModel.net",
    "encoder:ALSTM-Book": "同上（仅字段子集不同）",
    "encoder:Net64": "复用了 qlib Net 的逐分钟 MLP 与 BatchNorm",
    "encoder:Net192": "同上",
    "encoder:GroupedNet": "同上（分组 MLP）",
    "model:R0": "复用 qlib GRUModel 的 rnn/fc_out",
    "model:R1": "复用 qlib ALSTMModel 的 net/rnn/att_net/fc_out",
    "model:R1-Book": "同上",
    "preprocessing:*": "冻结预处理器通过 qlib DataHandlerLP 施加状态，保证与官方实现一致",
    "sampler:daily-window": "日分区窗口采样器使用 qlib TSDataSampler",
}


def dependency_report() -> dict:
    """Report which parts of the pipeline are usable in the current runtime.

    Purely descriptive: it never imports Qlib beyond the availability probe, so it is
    safe to call from a notebook preflight.
    """
    available = qlib_available()
    return {
        "qlib_available": available,
        "qlib_commit_expected": QLIB_COMMIT,
        "usable_without_qlib": [
            "ptop.schema", "ptop.data", "ptop.output", "ptop.targets",
            "ptop.backbones", "ptop.temporal（encoder 为 Raw/Patch1/Patch5 时）",
            "ptop.objectives", "jyc_e2e.losses", "jyc_e2e.research_metrics",
        ],
        "requires_qlib": [] if available else sorted(QLIB_FEATURES),
        "hint": QLIB_HINT,
    }


def require_qlib(feature: str) -> None:
    """Fail fast with a precise message before doing work that cannot succeed."""
    if not qlib_available():
        raise ContractError(f"{feature} 在当前运行时不可用。{QLIB_HINT}")
