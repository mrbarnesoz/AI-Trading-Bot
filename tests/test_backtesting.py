from __future__ import annotations

import pandas as pd

from ai_trading_bot.backtesting.simulator import run_backtest
from ai_trading_bot.config import BacktestConfig


def make_price_series(length: int = 50) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=length, freq="D")
    prices = pd.Series(100 + (index - index[0]).days * 0.2, index=index)
    return pd.DataFrame({"Adj Close": prices, "Close": prices, "Open": prices, "High": prices, "Low": prices}, index=index)


def make_descending_price_series(length: int = 50) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=length, freq="D")
    prices = pd.Series(100 - (index - index[0]).days * 0.2, index=index)
    return pd.DataFrame({"Adj Close": prices, "Close": prices, "Open": prices, "High": prices, "Low": prices}, index=index)


def test_run_backtest_produces_equity_curve():
    prices = make_price_series()
    signals = pd.Series(1, index=prices.index, name="signal")
    config = BacktestConfig(initial_capital=10000, transaction_cost=0.0)

    result = run_backtest(prices, signals, config)

    assert "allocation" in result.performance.columns
    assert result.summary["final_equity"] >= config.initial_capital * 0.9
    assert len(result.equity_curve) == len(prices)


def test_run_backtest_supports_short_positions():
    prices = make_descending_price_series()
    signals = pd.Series(-1, index=prices.index, name="signal")
    config = BacktestConfig(initial_capital=10000, transaction_cost=0.0)

    result = run_backtest(prices, signals, config)

    assert result.summary["final_equity"] >= config.initial_capital
    assert (result.performance["position"] <= 0).all()
