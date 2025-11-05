from __future__ import annotations

import numpy as np
import pandas as pd

from ai_trading_bot.backtesting.walk_forward import walk_forward_on_dataset
from ai_trading_bot.config import AppConfig


def _make_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=240, freq="H", tz="UTC")
    close = np.linspace(100.0, 110.0, len(index))
    open_px = close - 0.2
    high = close + 0.5
    low = close - 0.5
    volume = np.full(len(index), 1_000.0)

    decision = pd.DataFrame(
        {
            "Open": open_px,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )

    momentum = np.sin(np.linspace(0, 12, len(index)))
    target_return = momentum * 0.001
    target = (target_return > 0).astype(int)
    engineered = pd.DataFrame(
        {
            "Open": open_px,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "feature_one": close,
            "feature_two": momentum,
            "target_return": target_return,
            "target": target,
        },
        index=index,
    )
    return decision, engineered


def test_walk_forward_on_dataset_generates_segments():
    decision, engineered = _make_dataset()
    config = AppConfig()
    config.data.symbol = "TEST"
    config.backtest.initial_capital = 1_000.0
    config.backtest.transaction_cost = 0.0
    config.backtest.maker_fee_bps = 0.0
    config.backtest.taker_fee_bps = 0.0
    config.backtest.min_slippage_bps = 0.0
    config.pipeline.long_bands = []
    config.pipeline.short_bands = []
    config.pipeline.long_threshold = 0.55
    config.pipeline.short_threshold = 0.45

    report = walk_forward_on_dataset(
        config,
        decision,
        engineered,
        train_period=pd.Timedelta(days=5),
        test_period=pd.Timedelta(days=2),
        step_period=pd.Timedelta(days=2),
    )

    assert report.segments, "No walk-forward segments produced."
    assert "cumulative_return" in report.aggregate
