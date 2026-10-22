"""Research Pearson and Spearman retain distinct, explicit metric names."""

import numpy as np
import pandas as pd
import pytest

from jyc_e2e.research_metrics import pearson_ic_by_time, rank_ic_by_time


def test_pearson_matches_independent_oracle_and_differs_from_rank_ic():
    score = np.array([0., 1., 100., 0., 0., 2.])
    target = np.array([0., 1., 2., 0., 1., 2.])
    times = np.array([1, 1, 1, 2, 2, 2])
    result = pearson_ic_by_time(score, target, times)
    assert result[1] == pytest.approx(np.corrcoef(score[:3], target[:3])[0, 1])
    assert result[2] == pytest.approx(np.corrcoef(score[3:], target[3:])[0, 1])
    assert result[1] < rank_ic_by_time(score, target, times)[1]


def test_pearson_undefined_sections_and_nonfinite_filtering():
    score = np.array([0., 1., 2., np.nan, 1., 1., 1., 0., 1.])
    target = np.arange(9.)
    times = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3])
    result = pearson_ic_by_time(score, target, times)
    assert result[1] == pytest.approx(1.)
    assert np.isnan(result[2]) and np.isnan(result[3])
    constant_target = pearson_ic_by_time(np.arange(1000.), np.full(1000, 0.1), np.zeros(1000))
    assert np.isnan(constant_target[0])
    with pytest.raises(ValueError):
        pearson_ic_by_time(score[:, None], target, times)


def test_evaluator_keeps_legacy_rank_names_and_reports_pearson():
    from ptop.training import research_evaluator

    frame = pd.DataFrame({
        "decision_id": [1, 1, 1, 1, 1, 2, 2, 2],
        "score": [0., 1., 100., -100., -100., 3., 2., 1.],
        "target": [0., 1., 2., 3., np.nan, 0., 1., 2.],
        "predictable": [True, True, True, False, True, True, True, True],
        "target_valid": [True, True, True, True, False, True, True, True],
    })
    report = research_evaluator(frame)
    assert report["ic_mean"] == report["rank_ic_mean"] == pytest.approx(0.)
    assert report["ic_ir"] == report["rank_ic_ir"]
    expected = (np.corrcoef([0., 1., 100.], [0., 1., 2.])[0, 1] - 1) / 2
    assert report["pearson_ic_mean"] == pytest.approx(expected)
    assert report["pearson_valid_times"] == report["valid_times"] == 2
    assert report["pearson_invalid_times"] == 0
    assert report["official_score"] is None and report["research_only"] is True
