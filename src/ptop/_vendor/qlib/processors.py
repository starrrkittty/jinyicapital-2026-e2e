# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE in this directory.
# 摘自固定提交；使用项目 JSON 状态管理，处理器只保留拟合与变换计算。

import numpy as np
import pandas as pd

from .utils import fetch_df_by_index, get_group_columns


EPS = 1e-12


class Fillna:
    def __init__(self, fields_group=None, fill_value=0):
        self.fields_group = fields_group
        self.fill_value = fill_value

    def __call__(self, df):
        if self.fields_group is None:
            df.fillna(self.fill_value, inplace=True)
        else:
            df[self.fields_group] = df[self.fields_group].fillna(self.fill_value)
        return df


class ZScoreNorm:
    def __init__(self, fit_start_time, fit_end_time, fields_group=None):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None):
        df = fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        cols = get_group_columns(df, self.fields_group)
        self.mean_train = np.nanmean(df[cols].values, axis=0)
        self.std_train = np.nanstd(df[cols].values, axis=0)
        self.ignore = self.std_train == 0
        for _i, _con in enumerate(self.ignore):
            if _con:
                self.std_train[_i] = 1
                self.mean_train[_i] = 0
        self.cols = cols

    def __call__(self, df):
        def normalize(x, mean_train=self.mean_train, std_train=self.std_train):
            return (x - mean_train) / std_train

        df.loc(axis=1)[self.cols] = normalize(df[self.cols].values)
        return df


class RobustZScoreNorm:
    def __init__(self, fit_start_time, fit_end_time, fields_group=None, clip_outlier=True):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier

    def fit(self, df: pd.DataFrame = None):
        df = fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        self.cols = get_group_columns(df, self.fields_group)
        X = df[self.cols].values
        self.mean_train = np.nanmedian(X, axis=0)
        self.std_train = np.nanmedian(np.abs(X - self.mean_train), axis=0)
        self.std_train += EPS
        self.std_train *= 1.4826

    def __call__(self, df):
        X = df[self.cols]
        X -= self.mean_train
        X /= self.std_train
        if self.clip_outlier:
            X = np.clip(X, -3, 3)
        df[self.cols] = X
        return df
