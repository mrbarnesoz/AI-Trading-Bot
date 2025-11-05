"""Utilities for coordinating multi-symbol backtests with portfolio-aware risk caps."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Dict, Iterable, List, Tuple

from ai_trading_bot.config import AppConfig
from ai_trading_bot.pipeline import execute_backtest
from ai_trading_bot.strategies.ml_strategy import StrategyOutput
from ai_trading_bot.backtesting.simulator import BacktestResult


def _clone_config(base_config: AppConfig, symbol: str) -> AppConfig:
    config = AppConfig.from_dict(base_config.to_nested_dict())
    config.data.symbol = symbol
    return config


def run_multi_symbol_backtests(
    base_config: AppConfig | Dict[str, object],
    symbols: Iterable[str],
    *,
    force_download: bool = False,
    max_portfolio_cap_fraction: float = 0.6,
) -> Dict[str, object]:
    if not symbols:
        raise ValueError("symbols must contain at least one market.")
    if not isinstance(base_config, AppConfig):
        base_config = AppConfig.from_dict(dict(base_config))

    summaries: Dict[str, Dict[str, float]] = {}
    metadata: Dict[str, dict] = {}
    cumulative_fraction = 0.0

    for symbol in symbols:
        config = _clone_config(base_config, symbol)
        cumulative_fraction += float(config.backtest.position_capital_fraction)
        if cumulative_fraction > max_portfolio_cap_fraction + 1e-6:
            raise ValueError(
                f"Portfolio capital fraction {cumulative_fraction:.3f} exceeds cap {max_portfolio_cap_fraction:.3f}"
            )

        strategy_output, result, meta = execute_backtest(
            config,
            force_download=force_download,
            enforce_gates=False,
        )
        summaries[symbol] = dict(result.summary)
        metadata[symbol] = meta

    aggregate = _aggregate_summaries(summaries)
    return {
        "summaries": summaries,
        "metadata": metadata,
        "aggregate": aggregate,
    }


def _aggregate_summaries(summaries: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    aggregate: Dict[str, float] = defaultdict(float)
    if not summaries:
        return {}

    for summary in summaries.values():
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                aggregate[key] += float(value)

    count = float(len(summaries))
    if count:
        aggregate["avg_calc_sharpe"] = aggregate.get("calc_sharpe", 0.0) / count
        aggregate["avg_total_return"] = aggregate.get("total_return", 0.0) / count
        aggregate["avg_trades"] = aggregate.get("trades_count", 0.0) / count
    return dict(aggregate)


__all__ = ["run_multi_symbol_backtests"]
