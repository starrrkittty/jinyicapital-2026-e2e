# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE in this directory.
# 仅保留采样与处理器需要的索引工具；固定 pandas 2.2.3 不使用旧版索引分支。

from typing import Text, Union

import numpy as np
import pandas as pd


def np_ffill(arr: np.array):
    mask = np.isnan(arr.astype(float))
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]


def time_to_slc_point(t: Union[None, str, pd.Timestamp]) -> Union[None, pd.Timestamp]:
    if t is None:
        return t
    else:
        return pd.Timestamp(t)


def lazy_sort_index(df: pd.DataFrame, axis=0) -> pd.DataFrame:
    idx = df.index if axis == 0 else df.columns
    if not idx.is_monotonic_increasing:
        return df.sort_index(axis=axis)
    else:
        return df


def get_level_index(df: pd.DataFrame, level: Union[str, int]) -> int:
    if isinstance(level, str):
        try:
            return df.index.names.index(level)
        except (AttributeError, ValueError):
            return ("datetime", "instrument").index(level)
    elif isinstance(level, int):
        return level
    else:
        raise NotImplementedError(f"This type of input is not supported")


def fetch_df_by_index(
    df: pd.DataFrame,
    selector: Union[pd.Timestamp, slice, str, list, pd.Index],
    level: Union[str, int],
    fetch_orig=True,
) -> pd.DataFrame:
    if level is None or isinstance(selector, pd.MultiIndex):
        return df.loc(axis=0)[selector]
    idx_slc = (selector, slice(None, None))
    if get_level_index(df, level) == 1:
        idx_slc = idx_slc[1], idx_slc[0]
    if fetch_orig:
        for slc in idx_slc:
            if slc != slice(None, None):
                return df.loc[pd.IndexSlice[idx_slc],]
        else:
            return df
    else:
        return df.loc[pd.IndexSlice[idx_slc],]


def get_group_columns(df: pd.DataFrame, group: Union[Text, None]):
    if group is None:
        return df.columns
    else:
        return df.columns[df.columns.get_loc(group)]
