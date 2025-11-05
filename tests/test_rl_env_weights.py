from __future__ import annotations

from typing import Dict

import pandas as pd

from ai_trading_bot.config import AppConfig
from ai_trading_bot.backtesting.simulator import BacktestResult
from ai_trading_bot.strategies.ml_strategy import StrategyOutput
from ai_trading_bot.training.rl_env import BacktestTradingEnv


def test_rl_weight_mode(monkeypatch):
    index = pd.date_range("2024-01-01", periods=5, freq="H", tz="UTC")
    price_frame = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100, 101, 102, 103, 104],
            "Volume": [1_000] * 5,
        },
        index=index,
    )
    engineered = price_frame.assign(feature_one=1.0, target_return=0.001, target=1)

    def fake_prepare_dataset(cfg, force_download=False):
        return price_frame, engineered

    def fake_generate_signals(engineered_df, model_cfg, pipeline_cfg):
        signals = pd.Series([0, 1, 1, -1, -1], index=engineered_df.index, name="signal")
        probabilities = pd.Series([0.5, 0.6, 0.7, 0.4, 0.3], index=engineered_df.index, name="probability")
        components: Dict[str, pd.Series] = {
            "trend": signals.astype(float),
            "mean_reversion": pd.Series([0, -1, -1, 1, 1], index=engineered_df.index, dtype=float),
        }
        return StrategyOutput(signals=signals, probabilities=probabilities, data=engineered_df.copy(), components=components)

    def fake_simulate_backtest(price_data, signals, backtest_cfg, trailing_cfg=None, **kwargs):
        equity_curve = pd.Series(
            [backtest_cfg.initial_capital * (1 + 0.01 * i) for i in range(len(price_data))],
            index=price_data.index,
        )
        summary = {
            "calc_sharpe": 1.0,
            "total_return": float(signals.sum() * 0.01),
            "max_drawdown": -0.02,
            "trades_count": float((signals.diff().abs() > 0).sum()),
        }
        return BacktestResult(equity_curve=equity_curve, performance=price_data, summary=summary)

    monkeypatch.setattr("ai_trading_bot.training.rl_env.prepare_dataset", fake_prepare_dataset)
    monkeypatch.setattr("ai_trading_bot.training.rl_env.generate_signals", fake_generate_signals)
    monkeypatch.setattr("ai_trading_bot.training.rl_env.simulate_backtest", fake_simulate_backtest)

    config = AppConfig()
    config.data.symbol = "TEST"
    config.backtest.initial_capital = 1_000.0

    env = BacktestTradingEnv(
        config,
        reward_metric="sharpe",
        mode="weight",
        component_actions=[{"trend": 1.0}, {"trend": 0.0, "mean_reversion": 1.0}],
    )

    env.reset()
    state, reward, done, info = env.step(0)
    assert done is True
    assert reward == 1.0
    assert "summary" in info

    env.reset()
    state2, reward2, done2, info2 = env.step(1)
    assert done2 is True
    assert reward2 == 1.0
    assert info2["weights"] == {"trend": 0.0, "mean_reversion": 1.0}
