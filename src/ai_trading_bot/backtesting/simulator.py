"""Simple vectorised backtesting engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ai_trading_bot.config import BacktestConfig

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    performance: pd.DataFrame
    summary: Dict[str, float]


def run_backtest(
    price_data: pd.DataFrame,
    signals: pd.Series,
    backtest_cfg: BacktestConfig,
) -> BacktestResult:
    """Run a long-only backtest using generated signals."""
    data = price_data.loc[signals.index].copy()
    data["signal"] = signals
    data["position"] = data["signal"].shift(1).fillna(0)
    data["returns"] = data["Adj Close"].pct_change().fillna(0)
    trading_costs = (data["position"].diff().abs().fillna(0)) * backtest_cfg.transaction_cost
    data["strategy_return"] = data["position"] * data["returns"] - trading_costs
    data["equity_curve"] = (1 + data["strategy_return"]).cumprod() * backtest_cfg.initial_capital

    total_return = (data["equity_curve"].iloc[-1] / backtest_cfg.initial_capital) - 1
    ann_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(data)) - 1 if len(data) > 0 else 0.0
    ann_volatility = data["strategy_return"].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (
        (ann_return - backtest_cfg.risk_free_rate)
        / ann_volatility
        if ann_volatility and ann_volatility > 0
        else 0.0
    )
    win_rate = (data["strategy_return"] > 0).sum() / max(len(data), 1)

    summary = {
        "total_return": float(total_return),
        "annualised_return": float(ann_return),
        "annualised_volatility": float(ann_volatility),
        "sharpe_ratio": float(sharpe),
        "win_rate": float(win_rate),
        "max_drawdown": float(_max_drawdown(data["equity_curve"])),
        "final_equity": float(data["equity_curve"].iloc[-1]),
    }
    logger.info("Backtest summary: %s", summary)
    return BacktestResult(
        equity_curve=data["equity_curve"],
        performance=data[["signal", "position", "returns", "strategy_return", "equity_curve"]],
        summary=summary,
    )


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()
