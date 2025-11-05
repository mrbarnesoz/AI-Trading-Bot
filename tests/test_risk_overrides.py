from __future__ import annotations

import copy

from tests.live_stubs import install_live_stubs

install_live_stubs()

from ai_trading_bot.config import AppConfig
from ai_trading_bot.pipeline import _apply_risk_overrides


def test_apply_risk_overrides_updates_config(monkeypatch):
    base_config = AppConfig()
    config = copy.deepcopy(base_config)

    candidate = {
        "atr_mult": 0.003,
        "hyst_k": 1.1,
        "min_hold": 36,
        "cap_fraction": 0.028,
        "cancel_spread_bps": 7.0,
        "trail_k": 2.2,
        "ttp_k": 3.4,
    }

    monkeypatch.setattr("ai_trading_bot.pipeline.load_top_candidate", lambda: candidate)

    overrides = _apply_risk_overrides(config)

    assert overrides["min_atr_frac"] == candidate["atr_mult"]
    assert config.filters.min_atr_frac == candidate["atr_mult"]
    assert config.filters.hysteresis_k_atr == candidate["hyst_k"]
    assert config.backtest.min_hold_bars == candidate["min_hold"]
    assert config.backtest.position_capital_fraction == candidate["cap_fraction"]
    assert config.backtest.max_total_capital_fraction == min(0.5, candidate["cap_fraction"] * 3.0)
    assert config.backtest.cancel_spread_bps == candidate["cancel_spread_bps"]
    assert config.risk.trailing.enabled is True
    assert config.risk.trailing.k_atr["stop"]["swing"] == candidate["trail_k"]
    assert config.risk.trailing.k_atr["take"]["swing"] == candidate["ttp_k"]
