"""Transparent local diagnostics, not a replica of the platform evaluator."""

import numpy as np


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


def pearson_ic_by_time(
    scores: np.ndarray, target: np.ndarray, times: np.ndarray,
) -> dict[object, float]:
    """Raw Pearson IC, without the training loss's stabilizer or rank transform.

    Match RankIC's finite-pair filtering and minimum of three stocks. Constant
    predictions/labels remain undefined in evaluation, not a fabricated zero.
    """
    scores, target, times = map(np.asarray, (scores, target, times))
    if scores.ndim != 1 or scores.shape != target.shape or scores.shape != times.shape:
        raise ValueError("scores, target, and times must be equal-length vectors")
    result = {}
    for time in np.unique(times):
        within = (times == time) & np.isfinite(scores) & np.isfinite(target)
        if within.sum() < 3:
            result[time] = float("nan")
            continue
        x, y = scores[within].astype(np.float64), target[within].astype(np.float64)
        result[time] = (float(np.corrcoef(x, y)[0, 1])
                        if x.max() != x.min() and y.max() != y.min() else float("nan"))
    return result


def turnover_from_weights(weights: np.ndarray) -> np.ndarray:
    """Half-L1 changes; columns must identify the same stocks at each time."""

    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or not np.isfinite(weights).all():
        raise ValueError("weights must be a finite [times, aligned stocks] matrix")
    return 0.5 * np.abs(np.diff(weights, axis=0)).sum(axis=1)


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
