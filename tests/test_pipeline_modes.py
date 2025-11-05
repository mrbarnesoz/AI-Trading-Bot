from __future__ import annotations

import copy

import pandas as pd

from tests.live_stubs import install_live_stubs

install_live_stubs()

from ai_trading_bot.config import AppConfig
from ai_trading_bot.decision.mode_selector import ModeDecision
from ai_trading_bot.pipeline import _select_mode_config


def test_select_mode_config_applies_mode_overrides(monkeypatch):
    base_config = AppConfig()
    mode = copy.deepcopy(base_config.modes[0])
    mode.interval = "5m"

    decision = ModeDecision(
        mode=mode,
        data_config=copy.deepcopy(base_config.data),
        pipeline_config=copy.deepcopy(base_config.pipeline),
        backtest_config=copy.deepcopy(base_config.backtest),
        price_data=pd.DataFrame({"Close": [1.0, 1.1], "Volume": [100, 110]}),
        score=1.23,
        metrics={"volatility": 0.5, "trend_strength": 0.8},
    )

    decision.data_config.interval = "5m"
    decision.pipeline_config.long_threshold = 0.6
    decision.backtest_config.position_capital_fraction = 0.15

    monkeypatch.setattr(
        "ai_trading_bot.pipeline.select_mode",
        lambda cfg, force_download=False: decision,
    )

    active_config, returned_decision = _select_mode_config(base_config, force_download=False)

    assert returned_decision is decision
    assert active_config.data.interval == "5m"
    assert active_config.pipeline.long_threshold == 0.6
    assert active_config.backtest.position_capital_fraction == 0.15
