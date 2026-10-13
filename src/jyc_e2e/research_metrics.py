"""Transparent local diagnostics, not a replica of the platform evaluator.

Two standing rules for everything here:

- These are **research** measurements. The platform ranks each of the five indicators
  as a percentile over *all valid submissions*, so no local code can produce the
  competition score. Report the raw indicators and let the team compare candidates.
- Nothing here may peek at future labels to *define* its own groups. Scenario labels
  are supplied by the caller from already-visible state; a scenario report that slices
  by realized outcome is a diagnostic, not an input.
"""

import numpy as np

TIE_POLICIES = ("exclude", "include")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1, len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[start:stop]] = (start + stop + 1) / 2.0
    return ranks


def rank_ic_by_time(
    scores: np.ndarray, target: np.ndarray, times: np.ndarray
) -> dict[object, float]:
    """Spearman correlation at each time, using average ranks for ties."""

    scores, target, times = map(np.asarray, (scores, target, times))
    if scores.ndim != 1 or scores.shape != target.shape or scores.shape != times.shape:
        raise ValueError("scores, target, and times must be equal-length vectors")
    result = {}
    for time in np.unique(times):
        within = (times == time) & np.isfinite(scores) & np.isfinite(target)
        if within.sum() < 3:
            result[time] = float("nan")
            continue
        score_rank = _average_ranks(scores[within])
        target_rank = _average_ranks(target[within])
        result[time] = (
            float(np.corrcoef(score_rank, target_rank)[0, 1])
            if np.std(score_rank) > 0 and np.std(target_rank) > 0
            else float("nan")
        )
    return result


def ic_summary(ic_by_time: dict[object, float]) -> dict[str, float]:
    values = np.asarray(list(ic_by_time.values()), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"ic_mean": float("nan"), "ic_ir": float("nan")}
    mean = float(values.mean())
    scale = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return {"ic_mean": mean, "ic_ir": mean / scale if scale > 0 else float("nan")}


def align_cross_sections(
    values: np.ndarray, times: np.ndarray, instruments: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot ragged ``(value, time, stock)`` records into a ``[times, stocks]`` panel.

    Absent ``(time, stock)`` keys stay NaN so a missing member is never mistaken for a
    zero signal or a zero weight. Returns the panel plus its sorted time and stock axes.
    """

    values, times, instruments = map(np.asarray, (values, times, instruments))
    if values.ndim != 1 or values.shape != times.shape or values.shape != instruments.shape:
        raise ValueError("values, times and instruments must be equal-length vectors")
    if not values.size:
        raise ValueError("cannot align an empty set of records")
    unique_times = np.unique(times)
    unique_stocks = np.unique(instruments)
    time_index = np.searchsorted(unique_times, times)
    stock_index = np.searchsorted(unique_stocks, instruments)
    keys = time_index.astype(np.int64) * unique_stocks.size + stock_index
    if np.unique(keys).size != values.size:
        raise ValueError("time/stock keys must be unique to build an aligned panel")
    panel = np.full((unique_times.size, unique_stocks.size), np.nan, dtype=np.float64)
    panel[time_index, stock_index] = np.asarray(values, dtype=np.float64)
    return panel, unique_times, unique_stocks


def quantile_long_short_weights(
    scores: np.ndarray,
    times: np.ndarray,
    instruments: np.ndarray,
    *,
    group_ratio: float = 0.2,
    exposure: float = 0.5,
    tie_policy: str = "exclude",
) -> tuple[np.ndarray, dict]:
    """Fixed-rule quantile long/short book, one row per decision time.

    The top ``group_ratio`` of the cross-section is equal-weighted long and the bottom
    equal-weighted short, so net exposure is zero and gross exposure is ``2 * exposure``
    (``exposure=0.5`` gives 1). ``tie_policy`` decides what happens when the group
    boundary is fully tied: ``"exclude"`` keeps the tied names out of both groups (the
    conservative default), ``"include"`` lets them in. A cross-section that cannot form
    both sides yields an all-zero row and is counted, never a one-sided book.

    Weights are 0 for stocks that are not selected and NaN for stocks absent at that
    time. The rule is fixed on purpose: it must not be re-tuned per validation result.
    """

    if tie_policy not in TIE_POLICIES:
        raise ValueError(f"tie_policy must be one of {TIE_POLICIES}")
    if not 0 < group_ratio <= 0.5:
        raise ValueError("group_ratio must be in (0, 0.5]")
    if not 0 < exposure <= 0.5:
        raise ValueError("exposure must be in (0, 0.5]")
    panel, unique_times, unique_stocks = align_cross_sections(scores, times, instruments)

    weights = np.zeros_like(panel)
    n_long = np.zeros(unique_times.size, dtype=np.int64)
    n_short = np.zeros(unique_times.size, dtype=np.int64)
    empty_side = 0
    too_few = 0
    for row in range(panel.shape[0]):
        values = panel[row]
        valid = np.isfinite(values)
        count = int(valid.sum())
        if count < 2:
            too_few += 1
            weights[row] = 0.0
            continue
        side = max(1, int(np.floor(group_ratio * count)))
        side = min(side, count // 2)
        valid_scores = values[valid]
        ascending = np.sort(valid_scores)
        order = np.argsort(valid_scores, kind="stable")

        boundary_low = ascending[side - 1]
        below = int(np.sum(valid_scores < boundary_low))
        equal_low = int(np.sum(valid_scores == boundary_low))
        low_count = below + equal_low if tie_policy == "include" else below + (1 if equal_low == 1 else 0)

        boundary_high = ascending[::-1][side - 1]
        above = int(np.sum(valid_scores > boundary_high))
        equal_high = int(np.sum(valid_scores == boundary_high))
        high_count = above + equal_high if tie_policy == "include" else above + (1 if equal_high == 1 else 0)

        if min(low_count, high_count) < 1 or low_count + high_count > count:
            empty_side += 1
            weights[row] = 0.0
            continue
        low_positions = order[:low_count]
        high_positions = order[::-1][:high_count]
        weights[row, valid.nonzero()[0][low_positions]] = -exposure / low_count
        weights[row, valid.nonzero()[0][high_positions]] = exposure / high_count
        n_long[row] = high_count
        n_short[row] = low_count

    net = np.nansum(weights, axis=1)
    gross = np.nansum(np.abs(weights), axis=1)
    diagnostics = {
        "group_ratio": float(group_ratio),
        "exposure": float(exposure),
        "tie_policy": tie_policy,
        "rebalance_times": unique_times,
        "stocks": unique_stocks,
        "n_long": n_long,
        "n_short": n_short,
        "empty_side_times": empty_side,
        "too_few_times": too_few,
        "net_exposure": net,
        "gross_exposure": gross,
    }
    return weights, diagnostics


def intraday_transitions(times: np.ndarray) -> np.ndarray:
    """True where two adjacent decision times fall on the same trading day.

    Turnover must not treat a lunch break or an overnight jump as a trade, so callers
    pass this as ``transition_valid`` instead of silently counting both.
    """

    times = np.asarray(times)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a 1-D array with at least two decision times")
    days = times.astype("datetime64[D]")
    return days[1:] == days[:-1]


def turnover_from_weights(weights: np.ndarray, *, transition_valid: np.ndarray | None = None) -> np.ndarray:
    """Half-L1 changes; columns must identify the same stocks at each time.

    ``transition_valid`` masks the ``times - 1`` adjacent pairs: a pair that is not a legal
    rebalance step (a lunch break or an overnight jump) is reported as NaN rather than 0,
    so averaging must use ``np.nanmean`` instead of being biased downwards.
    """

    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or not np.isfinite(weights).all():
        raise ValueError("weights must be a finite [times, aligned stocks] matrix")
    changes = 0.5 * np.abs(np.diff(weights, axis=0)).sum(axis=1)
    return _apply_transitions(changes, transition_valid)


def _apply_transitions(changes: np.ndarray, transition_valid: np.ndarray | None) -> np.ndarray:
    if transition_valid is None:
        return changes
    mask = np.asarray(transition_valid)
    if mask.dtype != np.bool_ or mask.shape != changes.shape:
        raise ValueError("transition_valid must be a boolean vector over adjacent time pairs")
    result = changes.copy()
    result[~mask] = np.nan
    return result


def drift_weights(weights: np.ndarray, interval_returns: np.ndarray) -> np.ndarray:
    """Mark a held book to market; the book drifts before the next rebalance.

    Uses only returns that have already happened between the two rebalances, never the
    label of the later decision point.
    """

    weights, interval_returns = np.asarray(weights, float), np.asarray(interval_returns, float)
    if weights.shape != interval_returns.shape or weights.ndim != 2:
        raise ValueError("weights and interval_returns must be aligned [times, stocks] matrices")
    gross = 1.0 + np.sum(weights * interval_returns, axis=1, keepdims=True)
    return weights * (1.0 + interval_returns) / gross


def turnover_with_drift(weights: np.ndarray, interval_returns: np.ndarray, *,
                        transition_valid: np.ndarray | None = None) -> np.ndarray:
    """Half-L1 turnover where the previous book is marked to market first.

    ``interval_returns[t]`` is the return realized over the interval ending at rebalance
    ``t``; the book held from ``t - 1`` is priced with it before the change is measured.
    """

    weights, interval_returns = np.asarray(weights, float), np.asarray(interval_returns, float)
    if weights.shape != interval_returns.shape or weights.ndim != 2 or weights.shape[0] < 2:
        raise ValueError("weights and interval_returns must be aligned matrices with >= 2 times")
    drifted = drift_weights(weights[:-1], interval_returns[1:])
    changes = 0.5 * np.abs(weights[1:] - drifted).sum(axis=1)
    return _apply_transitions(changes, transition_valid)


def scenario_report(per_time: dict[object, float], scenarios: dict[str, list]) -> dict:
    """Aggregate one per-time metric under caller-supplied scenario labels.

    ``scenarios`` maps a scenario name to the decision times that belong to it. Labels
    must come from already-visible state (calendar, or a state the model could have
    known); never split scenarios by the realized outcome of the label. Uncovered times
    are reported so a tiny scenario cannot silently dominate a gradient.
    """

    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("scenarios must be a nonempty mapping of names to times")
    covered = set()
    result = {}
    for name, times in scenarios.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("scenario names must be nonempty strings")
        selected = [float(per_time[time]) for time in times if time in per_time and np.isfinite(per_time[time])]
        covered.update(time for time in times if time in per_time)
        result[name] = {
            "count": len(selected),
            "mean": float(np.mean(selected)) if selected else float("nan"),
        }
    result["_unassigned"] = sum(1 for time in per_time if time not in covered)
    result["_scenario_spread"] = _spread([item["mean"] for name, item in result.items() if not name.startswith("_")])
    return result


def _spread(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(max(finite) - min(finite)) if len(finite) > 1 else float("nan")


def long_short_sharpe(
    weights: np.ndarray, returns: np.ndarray, *, periods_per_year: float | None = None
) -> float:
    """Generic research proxy; platform portfolio and annualization may differ."""

    weights, returns = np.asarray(weights), np.asarray(returns)
    if weights.shape != returns.shape or weights.ndim != 2:
        raise ValueError("weights and returns must be aligned [times, stocks] matrices")
    if periods_per_year is not None and periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    pnl = np.sum(weights * returns, axis=1)
    if pnl.size < 2 or not np.isfinite(pnl).all():
        return float("nan")
    scale = float(pnl.std(ddof=1))
    if scale == 0:
        return float("nan")
    annualizer = np.sqrt(periods_per_year) if periods_per_year is not None else 1.0
    return float(pnl.mean() / scale * annualizer)


# 官方规则（组委会给出）：五项原始指标各占 20%，先各自在**全体有效提交**中取 rank 百分位
# （[0,1]），再加权求和；**换手率方向相反（越低排名越高）**；团队分 = 队内最高分。
# 本地算不出 R（需要其他队的提交），所以这里只固定「权重」与「方向」两条，
# 避免下游比候选时把换手率的方向搞反——那会直接选错模型。
OFFICIAL_METRIC_WEIGHTS = {
    "ic_mean": 0.2, "ic_ir": 0.2, "long_short_sharpe": 0.2, "stress": 0.2, "turnover_mean": 0.2,
}
HIGHER_IS_BETTER = {
    "ic_mean": True, "ic_ir": True, "long_short_sharpe": True, "stress": True, "turnover_mean": False,
}
# 官方对压力稳定性的描述是「高低波动、放缩量和不同时段等场景下的稳健性」。
OFFICIAL_STRESS_AXES = ("volatility", "volume", "time_of_day")


def _bucket_names(count: int, low: str, high: str) -> list[str]:
    """Readable labels: 3 buckets get 'mid'; more get mid1, mid2, ... so names stay stable."""
    if count == 3:
        return [low, "mid", high]
    return [low] + [f"mid{index}" for index in range(1, count - 1)] + [high]


def visible_state_scenarios(times, *, past_volatility=None, past_volume=None, buckets: int = 3) -> dict[str, list]:
    """Build the three official stress axes from state observable *before* each decision.

    The platform names the axes (high/low volatility, expanding/contracting volume, different
    intraday sessions) but publishes no thresholds, so this is an explicitly **local proxy**,
    not the platform's definition. Two rules are kept strict:

    * the volatility and volume inputs are caller-supplied series that must be computed from
      data available at or before each decision time — never from the realized label, which
      would turn a diagnostic into a leak;
    * the time-of-day axis needs no external input at all, because the decision times are
      already visible. Each day is split into equal consecutive ``buckets`` by slot order,
      so the same code works for bar1m (80/80/80) and bar30m (3/3/2).
    """
    axis = np.asarray(times)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError("times 不能为空（需要非空的一维决策时点向量）")
    axis = axis.astype("datetime64[ns]")
    unique = np.unique(axis)
    if axis.size != unique.size:
        raise ValueError("scenario labels need one row per decision time")
    if type(buckets) is not int or buckets < 2:
        raise ValueError("buckets must be an integer >= 2")

    scenarios: dict[str, list] = {}
    days = unique.astype("datetime64[D]")
    session_names = _bucket_names(buckets, "early", "late")
    for day in np.unique(days):
        slots = unique[days == day]
        edges = np.linspace(0, slots.size, buckets + 1).astype(int)
        for index in range(buckets):
            chunk = slots[edges[index]:edges[index + 1]]
            if chunk.size:
                scenarios.setdefault(f"time_of_day={session_names[index]}", []).extend(list(chunk))

    state_names = _bucket_names(buckets, "low", "high")
    for label, series in (("volatility", past_volatility), ("volume", past_volume)):
        if series is None:
            continue
        values = np.asarray(series, dtype=np.float64)
        if values.shape != unique.shape or not np.isfinite(values).all():
            raise ValueError(f"past_{label} 必须与 times 等长且全为有限值")
        cuts = np.quantile(values, np.linspace(0, 1, buckets + 1)[1:-1])
        assignment = np.searchsorted(cuts, values, side="right")
        for index in range(buckets):
            member = unique[assignment == index]
            if member.size:
                scenarios.setdefault(f"{label}={state_names[index]}", []).extend(list(member))
    return scenarios


def five_metric_report(
    scores: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    instruments: np.ndarray,
    *,
    group_ratio: float = 0.2,
    exposure: float = 0.5,
    tie_policy: str = "exclude",
    scenarios: dict[str, list] | None = None,
    periods_per_year: float | None = None,
    transition_valid: np.ndarray | None = None,
) -> dict:
    """Assemble the five platform indicators on one aligned research panel.

    Returns IC mean, IC_IR, long/short Sharpe, a stress-scenario summary and turnover —
    all in research units. The platform's percentile rank and BARRA-residual score are
    deliberately absent: they depend on the other submissions and on style factors this
    repository is not entitled to use. ``research_only`` and ``official_score`` restate
    that so no one reads these numbers as a competition score.
    """

    panel_scores, unique_times, unique_stocks = align_cross_sections(scores, times, instruments)
    panel_targets, target_times, target_stocks = align_cross_sections(targets, times, instruments)
    if not np.array_equal(unique_times, target_times) or not np.array_equal(unique_stocks, target_stocks):
        raise ValueError("scores and targets must share the same time and stock axes")

    ravel_times = np.repeat(unique_times, panel_scores.shape[1])
    ic_by_time = rank_ic_by_time(panel_scores.reshape(-1), panel_targets.reshape(-1), ravel_times)
    summary = ic_summary(ic_by_time)

    weights, weight_diagnostics = quantile_long_short_weights(
        scores, times, instruments, group_ratio=group_ratio, exposure=exposure, tie_policy=tie_policy)
    # A stock absent from a cross-section cannot be held; treat its weight as zero for
    # turnover and P&L rather than letting NaN propagate.
    held = np.nan_to_num(weights, nan=0.0)
    realized = np.nan_to_num(panel_targets, nan=0.0)
    turnover = turnover_with_drift(held, realized, transition_valid=transition_valid)

    report = {
        "ic_mean": summary["ic_mean"],
        "ic_ir": summary["ic_ir"],
        "long_short_sharpe": long_short_sharpe(held, realized, periods_per_year=periods_per_year),
        "turnover_mean": float(np.nanmean(turnover)) if np.isfinite(turnover).any() else float("nan"),
        "stress": (scenario_report(ic_by_time, scenarios) if scenarios else None),
        "valid_times": int(sum(1 for value in ic_by_time.values() if np.isfinite(value))),
        "invalid_times": int(sum(1 for value in ic_by_time.values() if not np.isfinite(value))),
        "gross_exposure_mean": float(np.nanmean(weight_diagnostics["gross_exposure"])),
        "empty_side_times": weight_diagnostics["empty_side_times"],
        "ticks": int(panel_scores.size),
        "research_only": True,
        "official_score": None,
        # 方向与权重一起带出来：下游按原始值挑候选时必须知道换手率是"越低越好"。
        "higher_is_better": dict(HIGHER_IS_BETTER),
        "official_weights": dict(OFFICIAL_METRIC_WEIGHTS),
    }
    return report
