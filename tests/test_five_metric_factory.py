"""The CLI-facing evaluator factory: this is what makes the five-metric report reachable.

``run()`` mounts hooks as ``resolve(reference)(config=config)`` and the Trainer wants a
``(predictions) -> dict`` callable, so ``five_metric_evaluator`` cannot be named directly as
``evaluator_factory``. Without this factory a real training run only reports ``ic_mean`` and
``ic_ir``: long-short Sharpe, turnover and stress never get computed.

Only numpy/pandas/pyarrow are needed (``ptop.training`` imports ``ptop.windows`` → ``pyarrow``),
so these cases do not require the pinned Qlib.
"""

import numpy as np
import pandas as pd
import pytest

from ptop.schema import ContractError
from ptop.training import _callable_contract, five_metric_factory

METRICS = ("ic_mean", "ic_ir", "long_short_sharpe", "turnover_mean", "stress")


def _predictions(*, periods=12, stocks=20, seed=0):
    """A frame shaped like the Trainer's per-epoch prediction table."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2024-01-02 10:00", periods=periods, freq="30min")
    rows = [{"decision_id": stamp.strftime("%Y%m%dT%H%M"), "decision_time": stamp,
             "instrument_id": 100 + index, "score": rng.normal(), "target": 0.0,
             "predictable": True, "target_valid": True}
            for stamp in stamps for index in range(stocks)]
    frame = pd.DataFrame(rows)
    frame["target"] = 0.3 * frame["score"] + rng.normal(scale=0.7, size=len(frame))
    return frame, stamps


def test_factory_returns_every_scored_component():
    frame, _ = _predictions()
    report = five_metric_factory(config={"evaluation": {"group_ratio": 0.25}})(frame)
    for name in METRICS:
        assert name in report, f"{name} 未出现在报告里"
    assert report["research_only"] is True
    assert report["official_score"] is None, "本地不得声称算出平台总分"


def test_stress_stays_unreported_until_scenarios_are_supplied():
    """No official stress definition exists yet, so the component must stay absent, not invented."""
    frame, stamps = _predictions()
    without = five_metric_factory(config={})(frame)
    assert without["stress"] is None

    scenarios = {"morning": [stamp for stamp in stamps if stamp.hour < 12],
                 "afternoon": [stamp for stamp in stamps if stamp.hour >= 12]}
    with_scenarios = five_metric_factory(config={"evaluation": {"scenarios": scenarios}})(frame)
    assert with_scenarios["stress"] is not None, "显式给出场景后应当算出压力分量"
    assert set(with_scenarios["stress"]) >= {"morning", "afternoon"}


def test_factory_output_satisfies_the_trainer_callable_contract():
    """The Trainer records the evaluator contract in checkpoint metadata."""
    contract = _callable_contract(five_metric_factory(config={"evaluation": {"transition_policy": "all"}}))
    assert contract["name"] == "FiveMetricResearchReport"
    assert contract["transition_policy"] == "all"


def test_empty_or_absent_evaluation_block_uses_documented_defaults():
    frame, _ = _predictions()
    for config in ({}, {"evaluation": None}, {"evaluation": {}}):
        report = five_metric_factory(config=config)(frame)
        assert report["transition_policy"] == "intraday", "默认只计同日转换"


def test_unknown_evaluation_option_is_rejected():
    with pytest.raises(ContractError, match="未知选项"):
        five_metric_factory(config={"evaluation": {"group_ration": 0.2}})


def test_non_mapping_config_is_rejected():
    with pytest.raises(ContractError, match="非 mapping"):
        five_metric_factory(config=["not", "a", "mapping"])
