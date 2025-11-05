from __future__ import annotations

import pandas as pd
import pytest

from ai_trading_bot.config import AppConfig, StrategyModeConfig
from ai_trading_bot.decision.mode_selector import ModeDecision, select_mode


def _make_price_frame(start: str, periods: int, freq: str, trend: float, noise: float) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    base = pd.Series(range(periods), index=index).astype(float)
    prices = 100 + trend * base + noise * (pd.Series(range(periods), index=index) % 3)
    volume = pd.Series(1_000_000 + 10_000 * pd.Series(range(periods), index=index), index=index)
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 0.5,
            "Low": prices - 0.5,
            "Close": prices,
            "Volume": volume,
        },
        index=index,
    )


def test_select_mode_prefers_trending_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AppConfig()
    config.data.symbol = "TEST"
    config.data.start_date = "2024-01-01T00:00:00Z"
    config.modes = [
        StrategyModeConfig(
            name="scalp",
            interval="1m",
            lookback_days=2,
            trend_weight=0.5,
            volatility_weight=2.0,
            long_threshold=0.6,
            short_threshold=0.4,
        ),
        StrategyModeConfig(
            name="swing",
            interval="1h",
            lookback_days=30,
            trend_weight=2.0,
            volatility_weight=0.5,
            long_threshold=0.55,
            short_threshold=0.45,
        ),
    ]

    scalp_data = _make_price_frame("2024-01-01", 120, "1min", trend=0.01, noise=0.5)
    swing_data = _make_price_frame("2024-01-01", 240, "1h", trend=0.5, noise=0.05)

    def fake_download(data_cfg, force_download=False):
        return scalp_data if data_cfg.interval == "1m" else swing_data

    monkeypatch.setattr("ai_trading_bot.decision.mode_selector.get_price_data", fake_download)

    decision = select_mode(config, force_download=False)

    assert isinstance(decision, ModeDecision)
    assert decision.mode.name == "swing"
    assert decision.pipeline_config.long_threshold == 0.55
    assert decision.backtest_config.position_capital_fraction == config.backtest.position_capital_fraction
