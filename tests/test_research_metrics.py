"""Research evaluation chain: alignment, weights, turnover, drift, scenarios."""

import numpy as np
import pytest

from jyc_e2e.research_metrics import (HIGHER_IS_BETTER, OFFICIAL_METRIC_WEIGHTS, align_cross_sections,
                                      drift_weights, five_metric_report, intraday_transitions,
                                      quantile_long_short_weights, scenario_report,
                                      turnover_from_weights, turnover_with_drift,
                                      visible_state_scenarios)


def test_align_cross_sections_keeps_missing_keys_as_nan():
    panel, times, stocks = align_cross_sections([1.0, 2.0, 3.0], [0, 0, 1], ["A", "B", "A"])
    assert times.tolist() == [0, 1]
    assert stocks.tolist() == ["A", "B"]
    np.testing.assert_allclose(panel, [[1.0, 2.0], [3.0, np.nan]], equal_nan=True)


def test_align_cross_sections_rejects_duplicate_keys_and_empty_input():
    with pytest.raises(ValueError, match="unique"):
        align_cross_sections([1.0, 2.0], [0, 0], ["A", "A"])
    with pytest.raises(ValueError, match="empty"):
        align_cross_sections([], [], [])


def test_quantile_weights_are_fixed_rule_and_net_zero():
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    weights, diagnostics = quantile_long_short_weights(
        scores, np.zeros(6), list("ABCDEF"), group_ratio=1 / 3, exposure=0.5)
    np.testing.assert_allclose(weights[0], [0.25, 0.25, 0.0, 0.0, -0.25, -0.25])
    assert diagnostics["n_long"][0] == 2 and diagnostics["n_short"][0] == 2
    assert abs(float(diagnostics["net_exposure"][0])) < 1e-12
    assert abs(float(diagnostics["gross_exposure"][0]) - 1.0) < 1e-12


def test_quantile_weights_stay_flat_when_one_side_cannot_form():
    weights, diagnostics = quantile_long_short_weights(
        np.ones(6), np.zeros(6), list("ABCDEF"), group_ratio=1 / 3)
    np.testing.assert_allclose(weights[0], np.zeros(6))
    assert diagnostics["empty_side_times"] == 1


def test_tie_policy_controls_the_boundary_block():
    scores = np.array([5.0, 5.0, 4.0, 3.0, 2.0, 2.0])
    included, _ = quantile_long_short_weights(
        scores, np.zeros(6), list("ABCDEF"), group_ratio=1 / 3, tie_policy="include")
    excluded, excluded_diagnostics = quantile_long_short_weights(
        scores, np.zeros(6), list("ABCDEF"), group_ratio=1 / 3, tie_policy="exclude")
    # include keeps both tied names on each side; exclude refuses to break the tie.
    np.testing.assert_allclose(included[0], [0.25, 0.25, 0.0, 0.0, -0.25, -0.25])
    np.testing.assert_allclose(excluded[0], np.zeros(6))
    assert excluded_diagnostics["empty_side_times"] == 1


def test_quantile_weights_reject_invalid_settings():
    for kwargs in ({"group_ratio": 0.0}, {"group_ratio": 0.6}, {"exposure": 0.0}, {"tie_policy": "maybe"}):
        with pytest.raises(ValueError):
            quantile_long_short_weights(np.arange(6.0), np.zeros(6), list("ABCDEF"), **kwargs)


def test_turnover_is_half_l1_and_masks_illegal_transitions():
    weights = np.array([[0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, 0.5, 0.0]])
    np.testing.assert_allclose(turnover_from_weights(weights), [0.0, 1.0])
    masked = turnover_from_weights(weights, transition_valid=np.array([True, False]))
    assert masked[0] == 0.0 and np.isnan(masked[1])


def test_intraday_transitions_exclude_overnight():
    times = np.array(["2024-01-02T10:00", "2024-01-02T10:30", "2024-01-03T10:00",
                      "2024-01-03T10:30"], dtype="datetime64[m]")
    assert intraday_transitions(times).tolist() == [True, False, True]


def test_drift_marks_the_held_book_to_market():
    weights = np.array([[0.5, 0.5]])
    drifted = drift_weights(weights, np.array([[0.10, 0.0]]))
    np.testing.assert_allclose(drifted[0], [0.55 / 1.05, 0.5 / 1.05], rtol=1e-12)
    # A book that has not moved does not drift.
    np.testing.assert_allclose(drift_weights(weights, np.zeros((1, 2))), weights)
    # A drift-free turnover is exactly the plain turnover.
    flat = np.array([[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_allclose(turnover_with_drift(flat, np.zeros((2, 2))), [0.0])


def test_turnover_with_drift_catches_rebalancing_that_naive_turnover_misses():
    """Ignoring drift understates turnover: identical targets can still need trading."""
    weights = np.array([[0.5, 0.5], [0.5, 0.5]])
    returns = np.array([[0.0, 0.0], [0.10, 0.0]])
    drifted = turnover_with_drift(weights, returns)
    naive = turnover_from_weights(weights)
    assert naive[0] == 0.0          # the two target books are identical ...
    assert drifted[0] > 0.0         # ... but the held book drifted, so a trade is needed


def test_scenario_report_aggregates_and_flags_unassigned():
    report = scenario_report({0: 0.1, 1: 0.2, 2: -0.1, 3: 0.05}, {"calm": [0, 1], "wild": [2]})
    assert report["calm"]["count"] == 2 and report["calm"]["mean"] == pytest.approx(0.15)
    assert report["wild"]["count"] == 1 and report["wild"]["mean"] == pytest.approx(-0.1)
    assert report["_unassigned"] == 1
    assert report["_scenario_spread"] == pytest.approx(0.25)


def test_five_metric_report_is_research_only_and_ranks_perfect_signals():
    times = np.repeat(np.arange(4), 20)
    stocks = np.tile(np.arange(20), 4)
    scores = np.tile(np.arange(20.0), 4)
    report = five_metric_report(scores, scores, times, stocks, group_ratio=0.2)
    assert report["ic_mean"] == pytest.approx(1.0)
    assert report["research_only"] is True and report["official_score"] is None
    assert report["valid_times"] == 4 and report["invalid_times"] == 0
    assert report["ticks"] == 80
    assert report["gross_exposure_mean"] == pytest.approx(1.0)
    for key in ("ic_ir", "long_short_sharpe", "turnover_mean"):
        assert key in report


# --- 官方规则固定下来的两条：权重与方向 ---


def test_official_weights_and_turnover_direction_are_recorded():
    """组委会给出：五项各占 20%，且「换手越低排名越高」。方向搞反会直接选错模型。"""
    assert sum(OFFICIAL_METRIC_WEIGHTS.values()) == pytest.approx(1.0)
    assert set(OFFICIAL_METRIC_WEIGHTS) == {"ic_mean", "ic_ir", "long_short_sharpe", "stress", "turnover_mean"}
    assert HIGHER_IS_BETTER["turnover_mean"] is False, "换手率必须标记为越低越好"
    assert all(HIGHER_IS_BETTER[name] for name in HIGHER_IS_BETTER if name != "turnover_mean")


def test_report_carries_the_direction_convention():
    times = np.repeat(np.arange(4), 20)
    rng = np.random.default_rng(0)
    scores = rng.normal(size=80)
    targets = 0.3 * np.repeat(scores.reshape(4, 20).mean(1), 20) + rng.normal(scale=0.7, size=80)
    report = five_metric_report(scores, targets, times, np.tile(np.arange(20), 4))
    assert report["higher_is_better"]["turnover_mean"] is False
    assert report["official_weights"] == OFFICIAL_METRIC_WEIGHTS


# --- 官方三轴压力场景（本地代理）---


def _grid(days=3, slots=8):
    clock = ["10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "15:00"][:slots]
    return np.array([f"2024-01-0{day}T{stamp}" for day in range(2, 2 + days) for stamp in clock],
                    dtype="datetime64[ns]")


def test_visible_state_scenarios_covers_all_three_official_axes():
    times = _grid()
    rng = np.random.default_rng(3)
    scenarios = visible_state_scenarios(times, past_volatility=rng.normal(size=times.size),
                                       past_volume=rng.normal(size=times.size))
    axes = {name.split("=")[0] for name in scenarios}
    assert axes == {"time_of_day", "volatility", "volume"}
    assert {name for name in scenarios if name.startswith("time_of_day")} == {
        "time_of_day=early", "time_of_day=mid", "time_of_day=late"}


def test_time_of_day_axis_splits_each_day_without_crossing_it():
    """8 槽三等分应得 2/3/3；且每天的同一段落在同一场景里，不跨日。"""
    times = _grid(days=3, slots=8)
    scenarios = visible_state_scenarios(times)
    sizes = {name: len(value) for name, value in scenarios.items()}
    assert sizes == {"time_of_day=early": 6, "time_of_day=mid": 9, "time_of_day=late": 9}
    for name, value in scenarios.items():
        days = {np.datetime64(item, "D") for item in value}
        assert len(days) == 3, f"{name} 应覆盖全部 3 天"
        assert len(set(map(str, value))) == len(value), f"{name} 有重复时点"
    early = scenarios["time_of_day=early"]
    assert all(str(np.datetime64(item, "m"))[-5:] in {"10:00", "10:30"} for item in early)


def test_scenarios_require_explicit_visible_series():
    times = _grid(days=1, slots=4)
    assert visible_state_scenarios(times) == {"time_of_day=early": [times[0]], "time_of_day=mid": [times[1]],
                                              "time_of_day=late": [times[2], times[3]]}
    with pytest.raises(ValueError, match="等长"):
        visible_state_scenarios(times, past_volatility=np.zeros(times.size + 1))
    with pytest.raises(ValueError, match="有限值"):
        visible_state_scenarios(times, past_volume=np.array([1.0, np.nan, 1.0, 1.0]))
    with pytest.raises(ValueError, match="非空"):
        visible_state_scenarios([])
    with pytest.raises(ValueError, match="buckets"):
        visible_state_scenarios(times, buckets=1)


def test_scenarios_feed_the_stress_component():
    times = _grid()
    rng = np.random.default_rng(5)
    scenarios = visible_state_scenarios(times, past_volatility=rng.normal(size=times.size))
    rows = times.size * 20
    scores = rng.normal(size=rows)
    targets = 0.3 * np.repeat(scores.reshape(times.size, 20).mean(1), 20) + rng.normal(scale=0.7, size=rows)
    report = five_metric_report(scores, targets, np.repeat(times, 20), np.tile(np.arange(20), times.size),
                                scenarios=scenarios)
    assert report["stress"] is not None
    assert len([key for key in report["stress"] if not key.startswith("_")]) == len(scenarios)
