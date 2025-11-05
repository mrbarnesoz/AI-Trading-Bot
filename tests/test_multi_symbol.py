from __future__ import annotations

import copy

import pandas as pd

from tests.live_stubs import install_live_stubs

install_live_stubs()

from ai_trading_bot.config import AppConfig
from ai_trading_bot.orchestration.multi_symbol import run_multi_symbol_backtests


class DummyStrategyOutput:
    def __init__(self):
        self.signals = pd.Series(dtype=float)
        self.probabilities = pd.Series(dtype=float)


class DummyBacktestResult:
    def __init__(self, sharpe: float, total_return: float, trades: int):
        self.summary = {
            "calc_sharpe": sharpe,
            "total_return": total_return,
            "trades_count": trades,
        }
        self.performance = pd.DataFrame()
        self.equity_curve = pd.Series(dtype=float)


def test_multi_symbol_runner_aggregates_results(monkeypatch):
    base_config = AppConfig()
    config_dict = base_config.to_nested_dict()
    base_config = AppConfig.from_dict(config_dict)
    base_config.backtest.position_capital_fraction = 0.1

    call_order = []

    def fake_execute_backtest(config, **kwargs):
        call_order.append(config.data.symbol)
        if config.data.symbol == "XBTUSD":
            return DummyStrategyOutput(), DummyBacktestResult(0.5, 0.04, 200), {"symbol": "XBTUSD"}
        return DummyStrategyOutput(), DummyBacktestResult(0.8, 0.06, 150), {"symbol": config.data.symbol}

    monkeypatch.setattr("ai_trading_bot.orchestration.multi_symbol.execute_backtest", fake_execute_backtest)

    output = run_multi_symbol_backtests(base_config, ["XBTUSD", "ETHUSD"], max_portfolio_cap_fraction=0.5)

    assert call_order == ["XBTUSD", "ETHUSD"]
    agg = output["aggregate"]
    assert abs(agg["avg_calc_sharpe"] - 0.65) < 1e-9
    assert abs(agg["avg_total_return"] - 0.05) < 1e-9
    assert abs(agg["avg_trades"] - 175) < 1e-9


def test_multi_symbol_runner_enforces_cap(monkeypatch):
    base_config = AppConfig()
    base_config.backtest.position_capital_fraction = 0.4

    call_count = {"count": 0}

    def fake_execute_backtest(*args, **kwargs):
        call_count["count"] += 1
        return DummyStrategyOutput(), DummyBacktestResult(0.3, 0.02, 50), {"symbol": "stub"}

    monkeypatch.setattr("ai_trading_bot.orchestration.multi_symbol.execute_backtest", fake_execute_backtest)

    try:
        run_multi_symbol_backtests(base_config, ["XBTUSD", "ETHUSD"], max_portfolio_cap_fraction=0.7)
    except ValueError as exc:
        assert "exceeds cap" in str(exc)
        assert call_count["count"] == 1
    else:
        raise AssertionError("Expected ValueError for cap breach")
