"""Mode selection logic for adaptive trading strategies."""

from __future__ import annotations

import logging
from dataclasses import replace, dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from ai_trading_bot.config import (
    AppConfig,
    BacktestConfig,
    DataConfig,
    PipelineConfig,
    StrategyModeConfig,
    _clean_probability_list,
)
from ai_trading_bot.data.fetch import download_price_data

logger = logging.getLogger(__name__)

EPSILON = 1e-9


@dataclass
class ModeDecision:
    mode: StrategyModeConfig
    data_config: DataConfig
    pipeline_config: PipelineConfig
    backtest_config: BacktestConfig
    price_data: pd.DataFrame
    score: float
    metrics: Dict[str, float]


def _resolve_start_date(base_start: str, lookback_days: int) -> str:
    if lookback_days <= 0:
        return base_start
    try:
        base_dt = datetime.strptime(base_start, "%Y-%m-%d").date()
    except ValueError:
        base_dt = datetime.utcnow().date() - timedelta(days=lookback_days)
    lookback_dt = (datetime.utcnow() - timedelta(days=lookback_days)).date()
    chosen = max(base_dt, lookback_dt)
    return chosen.strftime("%Y-%m-%d")


def _score_mode(price_data: pd.DataFrame, mode: StrategyModeConfig) -> Dict[str, float]:
    if price_data.empty:
        return {"score": float("-inf"), "volatility": 0.0, "trend_strength": 0.0, "liquidity": 0.0}

    returns = price_data["Adj Close"].pct_change().dropna()
    if returns.empty:
        return {"score": float("-inf"), "volatility": 0.0, "trend_strength": 0.0, "liquidity": 0.0}

    rolling_window = min(len(returns), max(10, int(0.1 * len(returns))))
    recent_returns = returns.tail(rolling_window)

    volatility = float(recent_returns.std())
    trend_strength = float(abs(recent_returns.mean()) / (volatility + EPSILON))
    liquidity = float(np.log1p(price_data["Volume"].tail(rolling_window).mean())) if "Volume" in price_data else 0.0

    score = mode.trend_weight * trend_strength + mode.volatility_weight * volatility + 0.1 * liquidity

    return {
        "score": float(score),
        "volatility": volatility,
        "trend_strength": trend_strength,
        "liquidity": liquidity,
    }


def _apply_mode_overrides(
    config: AppConfig, mode: StrategyModeConfig, data_cfg: DataConfig
) -> tuple[PipelineConfig, BacktestConfig]:
    pipeline_cfg = replace(config.pipeline)
    if mode.long_threshold is not None:
        pipeline_cfg.long_threshold = mode.long_threshold
    if mode.short_threshold is not None:
        pipeline_cfg.short_threshold = mode.short_threshold
    if mode.long_bands:
        pipeline_cfg.long_bands = _clean_probability_list(mode.long_bands, ascending=True)
    if mode.short_bands:
        pipeline_cfg.short_bands = _clean_probability_list(mode.short_bands, ascending=True)
    pipeline_cfg.__post_init__()  # ensure validation after overrides

    backtest_cfg = replace(config.backtest)
    if mode.position_fraction is not None:
        backtest_cfg.position_capital_fraction = mode.position_fraction
    if mode.max_total_fraction is not None:
        backtest_cfg.max_total_capital_fraction = mode.max_total_fraction
    if mode.max_position_units is not None:
        backtest_cfg.max_position_units = mode.max_position_units

    return pipeline_cfg, backtest_cfg


def select_mode(config: AppConfig, force_download: bool = False) -> ModeDecision:
    if not config.modes:
        raise ValueError("No strategy modes configured.")

    evaluations: List[ModeDecision] = []
    for mode in config.modes:
        data_cfg = replace(config.data)
        data_cfg.interval = mode.interval
        data_cfg.start_date = _resolve_start_date(config.data.start_date, mode.lookback_days)

        try:
            price_data = download_price_data(data_cfg, force_refresh=force_download)
        except Exception as exc:  # pragma: no cover - network errors handled gracefully
            logger.warning("Failed to download data for mode %s: %s", mode.name, exc)
            continue

        metrics = _score_mode(price_data, mode)
        if metrics["score"] == float("-inf"):
            logger.info("Mode %s skipped due to insufficient data.", mode.name)
            continue

        pipeline_cfg, backtest_cfg = _apply_mode_overrides(config, mode, data_cfg)
        decision = ModeDecision(
            mode=mode,
            data_config=data_cfg,
            pipeline_config=pipeline_cfg,
            backtest_config=backtest_cfg,
            price_data=price_data,
            score=metrics["score"],
            metrics=metrics,
        )
        evaluations.append(decision)

    if not evaluations:
        raise ValueError("Unable to evaluate any strategy mode with the available data.")

    best_decision = max(evaluations, key=lambda d: d.score)
    logger.info(
        "Selected mode '%s' (interval=%s, score=%.4f, metrics=%s)",
        best_decision.mode.name,
        best_decision.data_config.interval,
        best_decision.score,
        {k: round(v, 4) for k, v in best_decision.metrics.items()},
    )
    return best_decision
