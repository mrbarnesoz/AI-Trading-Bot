from __future__ import annotations

import copy
from typing import Dict

import pandas as pd

from tests.live_stubs import install_live_stubs

install_live_stubs()

from ai_trading_bot.config import AppConfig
from ai_trading_bot.training.rl_env import BacktestTradingEnv


class DummyStrategyOutput:
    def __init__(self, index):
        self.signals = pd.Series(0.0, index=index)
        self.probabilities = pd.Series(0.5, index=index)


class DummyBacktestResult:
    def __init__(self, summary: Dict[str, float]):
        self.summary = summary
        self.performance = pd.DataFrame()
        self.equity_curve = pd.Series(dtype=float)


def test_rl_env_step_returns_reward(monkeypatch):
    base_config = AppConfig()
    config_dict = base_config.to_nested_dict()

    timestamps = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    strategy_output = DummyStrategyOutput(timestamps)
    summary_payload = {
        "total_return": 0.05,
        "calc_sharpe": 0.75,
        "max_drawdown": -0.08,
        "trades_count": 120,
        "expectancy_after_costs": 0.0008,
    }
    backtest_result = DummyBacktestResult(summary_payload)

    def fake_execute_backtest(config, **kwargs):
        return strategy_output, backtest_result, {"symbol": config.data.symbol}

    monkeypatch.setattr("ai_trading_bot.training.rl_env.execute_backtest", fake_execute_backtest)

    env = BacktestTradingEnv(AppConfig.from_dict(config_dict), action_grid=[(0.0, 0.0), (0.02, -0.02)])
    initial_state = env.reset()
    assert initial_state["long_threshold"] == base_config.pipeline.long_threshold

    state, reward, done, info = env.step(1)

    assert done is True
    assert abs(state["long_threshold"] - (base_config.pipeline.long_threshold + 0.02)) < 1e-6
    assert reward == summary_payload["calc_sharpe"]
    assert "summary" in info
    assert env.history[-1].summary == summary_payload
