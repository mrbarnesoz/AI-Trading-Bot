"""Simple vectorised backtesting engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ai_trading_bot.config import BacktestConfig, TrailingConfig
from live.policy.decision import MarketState, Probabilities
from live.risk.trailing import TrailingManager
from live.state.positions import PositionManager

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    performance: pd.DataFrame
    summary: Dict[str, float]


def _compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    prev_close = close.shift(1)
    tr_components = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1)
    true_range = tr_components.max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr.fillna(method="bfill")


def run_backtest(
    price_data: pd.DataFrame,
    signals: pd.Series,
    backtest_cfg: BacktestConfig,
    trailing_cfg: Optional[TrailingConfig] = None,
    symbol: str = "SYMBOL",
    regime: str = "intraday",
) -> BacktestResult:
    """Run a backtest using generated signals and risk controls."""
    data = price_data.loc[signals.index].copy()
    data["signal"] = signals.astype(float)

    use_trailing = trailing_cfg is not None and trailing_cfg.enabled

    if not use_trailing:
        unit_fraction = backtest_cfg.position_capital_fraction
        max_units = backtest_cfg.max_position_units
        max_total_fraction = backtest_cfg.max_total_capital_fraction

        allocation_units = np.clip(np.abs(data["signal"]), 0, max_units)
        requested_fraction = allocation_units * unit_fraction
        effective_fraction = np.minimum(requested_fraction, max_total_fraction)
        allocation_series = np.sign(data["signal"]) * effective_fraction
        allocation_series = pd.Series(allocation_series, index=data.index, name="allocation")

        data["allocation"] = allocation_series
        data["position"] = allocation_series.shift(1).fillna(0.0)
        data["returns"] = data["Adj Close"].pct_change().fillna(0)
        trade_change = allocation_series.diff().fillna(allocation_series)
        trading_costs = trade_change.abs() * backtest_cfg.transaction_cost
        data["strategy_return"] = data["position"] * data["returns"] - trading_costs
    else:
        positions = PositionManager()
        trailing_manager = TrailingManager(trailing_cfg, positions)
        atr_series = _compute_atr(data)
        tick_size = float(data["Close"].diff().abs().replace(0, np.nan).min() or 1.0)

        position_units: List[float] = []
        for idx, row in data.iterrows():
            current_position = positions.net_position(symbol)
            atr_value = float(atr_series.loc[idx] if not pd.isna(atr_series.loc[idx]) else 0.0)
            market_state = MarketState(
                symbol=symbol,
                regime=regime,
                probabilities=Probabilities(0.5, 0.5, 0.0),
                position=current_position,
                timestamp=idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx,
                atr=atr_value,
                price=float(row["Close"]),
                lot_size=1.0,
                high=float(row["High"]),
                low=float(row["Low"]),
                spread_bps=0.0,
                latency_ok=True,
                bar_closed=True,
                tick_size=tick_size,
            )

            events = trailing_manager.update(symbol, market_state)
            for event in events:
                trailing_manager.record_execution(symbol, event, market_state)

            desired_units = float(data.at[idx, "signal"])
            desired_units = max(
                min(desired_units, backtest_cfg.max_position_units),
                -backtest_cfg.max_position_units,
            )

            current_position = positions.net_position(symbol)
            delta = desired_units - current_position
            if abs(delta) > 1e-9:
                payload = {
                    "side": "buy" if delta > 0 else "sell",
                    "size": abs(delta),
                    "confidence": 0.9,
                    "cross": False,
                    "target_position": desired_units,
                }
                trailing_manager.record_execution(symbol, payload, market_state)

            position_units.append(positions.net_position(symbol))

        trailing_manager.flush()

        unit_fraction = backtest_cfg.position_capital_fraction
        max_units = backtest_cfg.max_position_units
        max_total_fraction = backtest_cfg.max_total_capital_fraction

        position_series = pd.Series(position_units, index=data.index)
        allocation_units = np.clip(position_series.abs(), 0, max_units)
        requested_fraction = allocation_units * unit_fraction
        effective_fraction = np.minimum(requested_fraction, max_total_fraction)
        allocation_series = np.sign(position_series) * effective_fraction
        allocation_series = pd.Series(allocation_series, index=data.index, name="allocation")

        data["signal"] = position_series
        data["allocation"] = allocation_series
        data["position"] = allocation_series.shift(1).fillna(0.0)
        data["returns"] = data["Adj Close"].pct_change().fillna(0)
        trade_change = allocation_series.diff().fillna(allocation_series)
        trading_costs = trade_change.abs() * backtest_cfg.transaction_cost
        data["strategy_return"] = data["position"] * data["returns"] - trading_costs

    data["equity_curve"] = (1 + data["strategy_return"]).cumprod() * backtest_cfg.initial_capital
    data["capital_in_position"] = data["position"].abs() * backtest_cfg.initial_capital

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
        "average_capital_fraction": float(data["position"].abs().mean()),
    }
    logger.info("Backtest summary: %s", summary)
    return BacktestResult(
        equity_curve=data["equity_curve"],
        performance=data[
            ["signal", "allocation", "position", "returns", "strategy_return", "equity_curve", "capital_in_position"]
        ],
        summary=summary,
    )


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()
