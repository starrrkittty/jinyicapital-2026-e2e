# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE in this directory.
# 摘自固定提交的 TSDataSampler；保留索引、填充、过滤和批量取窗计算。

import bisect
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from .utils import get_level_index, lazy_sort_index, np_ffill, time_to_slc_point


class TSDataSampler:
    data_arr: np.ndarray
    data_index: pd.MultiIndex
    idx_map: np.ndarray
    idx_df: pd.DataFrame

    def __init__(
        self,
        data: pd.DataFrame,
        start,
        end,
        step_len: int,
        fillna_type: str = "none",
        dtype=None,
        flt_data=None,
    ):
        self.start = start
        self.end = end
        self.step_len = step_len
        self.fillna_type = fillna_type
        assert get_level_index(data, "datetime") == 0
        self.data = data.swaplevel().sort_index().copy()
        # 上游会清空调用方的列，调用方必须传入专用副本。
        data.drop(data.columns, axis=1, inplace=True)
        kwargs = {"object": self.data}
        if dtype is not None:
            kwargs["dtype"] = dtype
        self.data_arr = np.array(**kwargs)
        self.data_arr = np.append(
            self.data_arr,
            np.full((1, self.data_arr.shape[1]), np.nan, dtype=self.data_arr.dtype),
            axis=0,
        )
        self.nan_idx = len(self.data_arr) - 1
        self.idx_df, self.idx_map = self.build_index(self.data)
        self.data_index = deepcopy(self.data.index)
        if flt_data is not None:
            if isinstance(flt_data, pd.DataFrame):
                assert len(flt_data.columns) == 1
                flt_data = flt_data.iloc[:, 0]
            flt_data = flt_data.swaplevel()
            flt_data = flt_data.reindex(self.data_index).fillna(False).astype(bool)
            self.flt_data = flt_data.values
            self.idx_map = self.flt_idx_map(self.flt_data, self.idx_map)
            self.data_index = self.data_index[np.where(self.flt_data)[0]]
        self.idx_map = self.idx_map2arr(self.idx_map)
        self.idx_map, self.data_index = self.slice_idx_map_and_data_index(
            self.idx_map, self.idx_df, self.data_index, start, end
        )
        self.idx_arr = np.array(self.idx_df.values, dtype=np.float64)
        del self.data

    @staticmethod
    def slice_idx_map_and_data_index(idx_map, idx_df, data_index, start, end):
        assert len(idx_map) == data_index.shape[0]
        start_row_idx, end_row_idx = idx_df.index.slice_locs(start=time_to_slc_point(start), end=time_to_slc_point(end))
        time_flter_idx = (idx_map[:, 0] < end_row_idx) & (idx_map[:, 0] >= start_row_idx)
        return idx_map[time_flter_idx], data_index[time_flter_idx]

    @staticmethod
    def idx_map2arr(idx_map):
        dtype = np.int32
        no_existing_idx = (np.iinfo(dtype).max, np.iinfo(dtype).max)
        max_idx = max(idx_map.keys())
        arr_map = []
        for i in range(max_idx + 1):
            arr_map.append(idx_map.get(i, no_existing_idx))
        arr_map = np.array(arr_map, dtype=dtype)
        return arr_map

    @staticmethod
    def flt_idx_map(flt_data, idx_map):
        idx = 0
        new_idx_map = {}
        for i, exist in enumerate(flt_data):
            if exist:
                new_idx_map[idx] = idx_map[i]
                idx += 1
        return new_idx_map

    def get_index(self):
        return self.data_index.swaplevel()

    def config(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def build_index(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        idx_df = pd.Series(range(data.shape[0]), index=data.index, dtype=object)
        idx_df = lazy_sort_index(idx_df.unstack())
        idx_df = lazy_sort_index(idx_df, axis=1).T
        idx_map = {}
        for i, (_, row) in enumerate(idx_df.iterrows()):
            for j, real_idx in enumerate(row):
                if not np.isnan(real_idx):
                    idx_map[real_idx] = (i, j)
        return idx_df, idx_map

    @property
    def empty(self):
        return len(self) == 0

    def _get_indices(self, row: int, col: int) -> np.array:
        indices = self.idx_arr[max(row - self.step_len + 1, 0) : row + 1, col]
        if len(indices) < self.step_len:
            indices = np.concatenate([np.full((self.step_len - len(indices),), np.nan), indices])
        if self.fillna_type == "ffill":
            indices = np_ffill(indices)
        elif self.fillna_type == "ffill+bfill":
            indices = np_ffill(np_ffill(indices)[::-1])[::-1]
        else:
            assert self.fillna_type == "none"
        return indices

    def _get_row_col(self, idx) -> Tuple[int]:
        if isinstance(idx, (int, np.integer)):
            real_idx = idx
            if 0 <= real_idx < len(self.idx_map):
                i, j = self.idx_map[real_idx]
            else:
                raise KeyError(f"{real_idx} is out of [0, {len(self.idx_map)})")
        elif isinstance(idx, tuple):
            date, inst = idx
            date = pd.Timestamp(date)
            i = bisect.bisect_right(self.idx_df.index, date) - 1
            j = bisect.bisect_left(self.idx_df.columns, inst)
        else:
            raise NotImplementedError(f"This type of input is not supported")
        return i, j

    def __getitem__(self, idx: Union[int, Tuple[object, str], List[int]]):
        mtit = (list, np.ndarray)
        if isinstance(idx, mtit):
            indices = [self._get_indices(*self._get_row_col(i)) for i in idx]
            indices = np.concatenate(indices)
        else:
            indices = self._get_indices(*self._get_row_col(idx))
        indices = np.nan_to_num(indices.astype(np.float64), nan=self.nan_idx).astype(int)
        if (np.diff(indices) == 1).all():
            data = self.data_arr[indices[0] : indices[-1] + 1]
        else:
            data = self.data_arr[indices]
        if isinstance(idx, mtit):
            data = data.reshape(-1, self.step_len, *data.shape[1:])
        return data

    def __len__(self):
        return len(self.idx_map)
