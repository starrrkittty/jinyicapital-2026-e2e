# 专项对照测试使用真实原版 Qlib，常规测试和运行环境不需要该依赖。
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import pytest
import torch
from qlib.contrib.model.pytorch_alstm import ALSTMModel as OriginalALSTM
from qlib.contrib.model.pytorch_gru import GRUModel as OriginalGRU
from qlib.contrib.model.pytorch_nn import Net as OriginalNet
from qlib.data.dataset import TSDataSampler as OriginalSampler
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader
from qlib.data.dataset.processor import Fillna as OriginalFillna
from qlib.data.dataset.processor import RobustZScoreNorm as OriginalRobust
from qlib.data.dataset.processor import ZScoreNorm as OriginalZScore

from ptop._vendor.qlib.models import ALSTMModel, GRUModel, Net
from ptop._vendor.qlib.processors import Fillna, RobustZScoreNorm, ZScoreNorm
from ptop._vendor.qlib.sampler import TSDataSampler
from ptop.preprocessing import FrozenPreprocessor
from ptop.schema import FEATURES, PRICE_FIELDS


PAIRS = [(ALSTMModel, OriginalALSTM), (GRUModel, OriginalGRU), (Net, OriginalNet),
         (Fillna, OriginalFillna), (ZScoreNorm, OriginalZScore),
         (RobustZScoreNorm, OriginalRobust), (TSDataSampler, OriginalSampler)]
SOURCE_HASHES = {
    "contrib/model/pytorch_alstm.py": "f7a19f9a2a6551f0659cc821cce30aaaa865a703874e7e83c4fbc8e0d1275a6b",
    "contrib/model/pytorch_gru.py": "75cc1c38dcbc4dfda90105d203f60399aec60c154360eac8eddb38f605efda7e",
    "contrib/model/pytorch_nn.py": "d4c7fec9a79ee3fecc8b467369a078d3e7d61b5a6c0bbe29e188192c2d8c7c21",
    "data/dataset/processor.py": "424af44e81c99467171c2d0b7978fb9f95d3b4c8b743a535ff4fe45d708cf517",
    "data/dataset/__init__.py": "f81de9c050e91e4f25ed0fb802273a7ca7dedc104d580ee30ebdd2a733fc68a8",
    "data/dataset/utils.py": "c556538a0de06597390001b2e9de361f87af11c96853c67ba9e8af4a88b40024",
    "utils/__init__.py": "f423a1b86444f8455a348ddce69c142dcbfa3ba68131897539c031bcf595de70",
    "constant.py": "92b0cc8d027742591558bbaee935909be9a58aa9c5541b714f41f0e1338fde94",
}


@pytest.fixture(autouse=True)
def reference_environment():
    import qlib

    root = Path(qlib.__file__).parent
    for relative, expected in SOURCE_HASHES.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def computation_tree(method):
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


@pytest.mark.parametrize("embedded,original", PAIRS)
def test_method_computation_matches_fixed_source(embedded, original):
    methods = {name for name, value in vars(embedded).items()
               if inspect.isfunction(value) or isinstance(value, (staticmethod, property))}
    for name in methods:
        left, right = getattr(embedded, name), getattr(original, name)
        if isinstance(left, property):
            left, right = left.fget, right.fget
        assert computation_tree(left) == computation_tree(right), f"{embedded.__name__}.{name}"


@pytest.mark.parametrize("embedded,original,kwargs,width", [
    (ALSTMModel, OriginalALSTM, {"d_feat": 50}, 50 * 6),
    (ALSTMModel, OriginalALSTM, {"d_feat": 24, "rnn_type": "LSTM", "dropout": 0.2}, 24 * 6),
    (GRUModel, OriginalGRU, {"d_feat": 50}, 50 * 6),
    (Net, OriginalNet, {"input_dim": 50, "output_dim": 64}, 50),
    (Net, OriginalNet, {"input_dim": 14, "output_dim": 192, "layers": (32, 16), "act": "SiLU"}, 14),
])
@pytest.mark.parametrize("training", [False, True])
def test_model_initialization_forward_gradients_and_buffers(embedded, original, kwargs, width, training):
    torch.manual_seed(19)
    left = embedded(**kwargs).train(training)
    torch.manual_seed(19)
    right = original(**kwargs).train(training)
    assert left.state_dict().keys() == right.state_dict().keys()
    for name, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[name], atol=0, rtol=0)
    x = torch.randn(3, width, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    torch.manual_seed(23)
    actual = left(x)
    torch.manual_seed(23)
    expected = right(y)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=0, rtol=0)
    for name, value in left.named_parameters():
        torch.testing.assert_close(value.grad, dict(right.named_parameters())[name].grad, atol=0, rtol=0)
    for name, value in left.named_buffers():
        torch.testing.assert_close(value, dict(right.named_buffers())[name], atol=0, rtol=0)


def feature_frame(values):
    result = values.copy(deep=True)
    result.columns = pd.MultiIndex.from_product([["feature"], values.columns])
    return result


def original_fit(values, base, start, end):
    kwargs = {"clip_outlier": False} if base == "N0" else {}
    cls = OriginalRobust if base == "N0" else OriginalZScore
    processor = cls(start, end, fields_group="feature", **kwargs)
    handler = DataHandlerLP(data_loader=StaticDataLoader(feature_frame(values)),
                            infer_processors=[processor, OriginalFillna(fields_group="feature")],
                            shared_processors=[], learn_processors=[], init_data=False)
    handler.setup_data(init_type=DataHandlerLP.IT_FIT_IND)
    return processor


@pytest.mark.parametrize("mode,base", [("N0", "N0"), ("N1", "N1"), ("N2", "N0"), ("N2", "N1")])
@pytest.mark.parametrize("by_column", [False, True])
def test_frozen_preprocessing_matches_original_handler(mode, base, by_column):
    times = pd.date_range("2024-01-02 09:31", periods=18, freq="min")
    index = pd.MultiIndex.from_product([times, [1, 2]], names=["datetime", "instrument"])
    values = pd.DataFrame(np.random.default_rng(51).uniform(5, 100, (len(index), 25)), index=index, columns=FEATURES)
    values.iloc[3, 0] = np.nan
    values.iloc[7, 9] = np.nan
    if base == "N1":
        values["volume"] = 11.0
    later = values.iloc[:4].copy()
    later.index = pd.MultiIndex.from_arrays([later.index.get_level_values(0) + pd.Timedelta(days=1),
                                            later.index.get_level_values(1)], names=index.names)
    later[:] = 10000.0
    full = pd.concat([values, later])
    referenced = index.append(index[:9]).append(later.index)
    left = FrozenPreprocessor(mode, n2_base=base).fit(
        full, full.notna(), referenced_rows=referenced, fit_start="2024-01-02", fit_end="2024-01-02",
        by_column=by_column).freeze(scale_review="synthetic upstream comparison")
    start, end = left.fit_start, left.fit_end
    right = original_fit(values, base, start, end)
    if by_column:
        columns = [original_fit(values[[field]], base, start, end) for field in FEATURES]
        right.mean_train = np.concatenate([item.mean_train for item in columns])
        right.std_train = np.concatenate([item.std_train for item in columns])
        if base == "N1":
            right.ignore = np.concatenate([item.ignore for item in columns])
    if mode == "N2":
        pooled = values.loc[:, list(PRICE_FIELDS)].to_numpy().reshape(-1)
        pooled_index = pd.MultiIndex.from_arrays([np.repeat(index.get_level_values(0), len(PRICE_FIELDS)),
                                                  np.arange(len(pooled))], names=index.names)
        shared = original_fit(pd.DataFrame({"shared_price": pooled}, index=pooled_index), base, start, end)
        positions = [FEATURES.index(field) for field in PRICE_FIELDS]
        right.mean_train[positions] = shared.mean_train[0]
        right.std_train[positions] = shared.std_train[0]
        if base == "N1":
            right.ignore[positions] = shared.ignore[0]
    np.testing.assert_array_equal(left._processor.mean_train, right.mean_train)
    np.testing.assert_array_equal(left._processor.std_train, right.std_train)
    for incoming in (full, full.iloc[::-1], full.iloc[:5], full.iloc[5:]):
        snapshot = incoming.copy(deep=True)
        handler = DataHandlerLP(data_loader=StaticDataLoader(feature_frame(incoming)),
                                infer_processors=[right, OriginalFillna(fields_group="feature")],
                                shared_processors=[], learn_processors=[], init_data=False)
        handler.setup_data(init_type=DataHandlerLP.IT_LS)
        expected = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I).reindex(incoming.index).astype(np.float32)
        pd.testing.assert_frame_equal(left.transform(incoming, incoming.notna()), expected)
        pd.testing.assert_frame_equal(incoming, snapshot)


@pytest.mark.parametrize("fill", ["none", "ffill", "ffill+bfill"])
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("length", [1, 4, 20])
def test_sampler_matches_original_index_windows_filters_and_padding(fill, filtered, length):
    times = pd.DatetimeIndex(["2024-01-02 09:31", "2024-01-02 09:32", "2024-01-02 11:30",
                              "2024-01-02 13:01", "2024-01-02 13:02"])
    index = pd.MultiIndex.from_product([times, [2, 9]], names=["datetime", "instrument"])
    data = pd.DataFrame(np.arange(20, dtype=np.float64).reshape(10, 2), index=index, columns=["source", "slot"])
    data = data.drop(index[[2, 7]])
    flt = pd.Series([True, False] * 4, index=data.index) if filtered else None
    kwargs = dict(start=times[1], end=times[-1], step_len=length, fillna_type=fill, dtype=np.float64, flt_data=flt)
    original_input, embedded_input = data.copy(), data.copy()
    left, right = TSDataSampler(embedded_input, **kwargs), OriginalSampler(original_input, **kwargs)
    assert len(left) == len(right)
    assert not len(original_input.columns) and not len(embedded_input.columns)
    pd.testing.assert_index_equal(left.get_index(), right.get_index())
    np.testing.assert_array_equal(left.idx_map, right.idx_map)
    np.testing.assert_array_equal(left.data_arr, right.data_arr)
    for i in range(len(left)):
        np.testing.assert_array_equal(left[i], right[i])
    order = np.arange(len(left))[::-1]
    np.testing.assert_array_equal(left[order], right[order])
    for key in [(times[-1], 2), (times[-1], 9)]:
        np.testing.assert_array_equal(left[key], right[key])
    for sampler in (left, right):
        with pytest.raises(KeyError):
            sampler[len(sampler)]
