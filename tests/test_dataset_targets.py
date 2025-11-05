from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from ai_trading_bot.data.pipeline import prepare_dataset


@pytest.fixture
def base_config() -> dict:
    return {
        "data": {
            "symbol": "XBTUSD",
            "source": "bitmex",
            "interval": "1h",
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-01T05:00:00Z",
            "cache_dir": "data/raw",
        },
        "features": {
            "indicators": ["sma", "ema"],
            "sma_window": 2,
            "ema_window": 3,
            "rsi_window": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        },
        "model": {},
        "backtest": {},
        "risk": {"trailing": {"enabled": False}},
        "pipeline": {
            "lookahead": 1,
            "target_column": "target_return",
            "long_threshold": 0.55,
            "short_threshold": 0.45,
            "long_bands": [],
            "short_bands": [],
        },
        "modes": [],
    }


def test_prepare_dataset_targets(monkeypatch, base_config):
    index = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    close = pd.Series(np.linspace(100, 107, len(index)), index=index)
    price_df = pd.DataFrame(
        {
            "Open": close.values - 0.5,
            "High": close.values + 0.5,
            "Low": close.values - 1.0,
            "Close": close.values,
            "Volume": np.linspace(1000, 1700, len(index)),
        },
        index=index,
    )

    def fake_get_price_data(cfg, force_download=False):
        return price_df

    monkeypatch.setattr("ai_trading_bot.data.pipeline.get_price_data", fake_get_price_data)

    decision, engineered = prepare_dataset(deepcopy(base_config), force_download=False)

    assert {"target_return", "target"}.issubset(engineered.columns)
    assert all(col in decision.columns for col in ["Open", "High", "Low", "Close", "Volume", "target"])

    lookahead = base_config["pipeline"]["lookahead"]
    expected = close.shift(-lookahead) / close - 1.0
    engineered_returns = engineered["target_return"]
    expected_aligned = expected.loc[engineered.index].astype(engineered_returns.dtype)
    pd.testing.assert_series_equal(
        engineered_returns,
        expected_aligned,
        check_names=False,
        check_exact=False,
        atol=1e-8,
    )
    assert engineered["target"].isin({0, 1}).all()
