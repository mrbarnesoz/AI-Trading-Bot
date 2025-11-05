from __future__ import annotations

import pandas as pd

from ai_trading_bot.backtesting.simulator import run_backtest
from ai_trading_bot.config import BacktestConfig, TrailingConfig


def make_price_series(length: int = 50) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=length, freq="D")
    prices = pd.Series(100 + (index - index[0]).days * 0.2, index=index)
    return pd.DataFrame({"Close": prices, "Open": prices, "High": prices, "Low": prices}, index=index)


def make_descending_price_series(length: int = 50) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=length, freq="D")
    prices = pd.Series(100 - (index - index[0]).days * 0.2, index=index)
    return pd.DataFrame({"Close": prices, "Open": prices, "High": prices, "Low": prices}, index=index)


def test_run_backtest_produces_equity_curve():
    prices = make_price_series()
    signals = pd.Series(1, index=prices.index, name="signal")
    config = BacktestConfig(initial_capital=10000, transaction_cost=0.0)

    trailing = TrailingConfig(enabled=False)
    result = run_backtest(prices, signals, config, trailing_cfg=trailing, symbol="TEST", regime="intraday")

    assert "allocation" in result.performance.columns
    assert "execution_fee" in result.performance.columns
    assert "funding_cost" in result.performance.columns
    assert result.summary["final_equity"] >= config.initial_capital * 0.9
    assert len(result.equity_curve) == len(prices)
    assert "avg_slippage_bps" in result.summary
    assert "maker_trades" in result.summary
    assert "taker_trades" in result.summary
    assert "signal_entries" in result.summary
    assert "signal_flips" in result.summary
    assert "trades_per_day" in result.summary


def test_run_backtest_supports_short_positions():
    prices = make_descending_price_series()
    signals = pd.Series(-1, index=prices.index, name="signal")
    config = BacktestConfig(initial_capital=10000, transaction_cost=0.0)

    trailing = TrailingConfig(enabled=False)
    result = run_backtest(prices, signals, config, trailing_cfg=trailing, symbol="TEST", regime="intraday")

    assert result.summary["final_equity"] >= config.initial_capital
    assert (result.performance["position"] <= 0).all()


def test_backtest_applies_taker_fees():
    prices = make_price_series(length=5)
    signals = pd.Series([0, 1, 1, 0, 0], index=prices.index, name="signal")
    config = BacktestConfig(
        initial_capital=10000,
        transaction_cost=0.0,
        taker_fee_bps=10.0,
        signal_persistence_bars=1,
        flip_cooldown_bars=0,
        min_hold_bars=0,
        cooldown_bars_after_exit=0,
        entry_mode="auto",
        max_entry_wait_bars=1,
    )

    trailing = TrailingConfig(enabled=False)
    result = run_backtest(prices, signals, config, trailing_cfg=trailing, symbol="TEST", regime="intraday")

    assert result.summary["total_fees_paid"] > 0.0
    assert (result.performance["execution_fee"] > 0).any()
    assert result.summary["taker_trades"] >= 1


def test_backtest_applies_funding_costs():
    prices = make_price_series(length=6)
    prices["fundingRate"] = [0.0, 0.001, 0.0005, -0.001, 0.0, 0.0]
    signals = pd.Series(1, index=prices.index, name="signal")
    config = BacktestConfig(
        initial_capital=10000,
        transaction_cost=0.0,
        taker_fee_bps=0.0,
        funding_rate_column="fundingRate",
        signal_persistence_bars=1,
        flip_cooldown_bars=0,
    )

    trailing = TrailingConfig(enabled=False)
    result = run_backtest(prices, signals, config, trailing_cfg=trailing, symbol="TEST", regime="intraday")

    assert result.summary["total_funding_paid"] != 0.0
    assert (result.performance["funding_cost"] != 0.0).any()


def test_signal_persistence_requires_consecutive_confirmation():
    prices = make_price_series(length=12)
    zig_zag = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    signals = pd.Series(zig_zag, index=prices.index, name="signal")
    config = BacktestConfig(
        initial_capital=10000,
        transaction_cost=0.0,
        taker_fee_bps=0.0,
        maker_fee_bps=0.0,
        signal_persistence_bars=3,
        flip_cooldown_bars=2,
    )

    trailing = TrailingConfig(enabled=False)
    result = run_backtest(prices, signals, config, trailing_cfg=trailing, symbol="TEST", regime="intraday")

    assert result.summary["signal_entries"] == 0
    assert result.summary["signal_flips"] == 0
