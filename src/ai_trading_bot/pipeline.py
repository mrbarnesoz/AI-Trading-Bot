"""High-level orchestration for the AI trading bot pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ai_trading_bot.backtesting.simulator import BacktestResult, run_backtest
from ai_trading_bot.config import AppConfig, load_config
from ai_trading_bot.decision.mode_selector import ModeDecision, select_mode
from ai_trading_bot.features.indicators import engineer_features
from ai_trading_bot.models.predictor import train_model
from ai_trading_bot.strategies.ml_strategy import StrategyOutput, generate_signals
from ai_trading_bot.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def prepare_dataset(config: AppConfig, force_download: bool = False) -> tuple[ModeDecision, pd.DataFrame]:
    """Select the optimal mode, fetch data, and engineer features."""
    decision = select_mode(config, force_download=force_download)
    engineered = engineer_features(decision.price_data, config.features, decision.pipeline_config)
    return decision, engineered


def train(config_path: Path | str = "config.yaml", force_download: bool = False) -> dict:
    """Train the machine learning model."""
    configure_logging()
    config = load_config(config_path)
    decision, engineered = prepare_dataset(config, force_download=force_download)
    metrics = train_model(engineered, config.model, decision.pipeline_config)
    metrics["mode"] = decision.mode.name
    metrics["mode_score"] = decision.score
    logger.info("Training metrics: %s", metrics)
    return metrics


def backtest(
    config_path: Path | str = "config.yaml",
    force_download: bool = False,
    long_threshold: Optional[float] = None,
    short_threshold: Optional[float] = None,
) -> tuple[StrategyOutput, BacktestResult]:
    """Run the full pipeline through to backtesting."""
    configure_logging()
    config = load_config(config_path)
    decision, engineered = prepare_dataset(config, force_download=force_download)
    long_thr = long_threshold if long_threshold is not None else decision.pipeline_config.long_threshold
    short_thr = short_threshold if short_threshold is not None else decision.pipeline_config.short_threshold
    strategy_output = generate_signals(
        engineered,
        config.model,
        decision.pipeline_config,
        probability_long_threshold=long_thr,
        probability_short_threshold=short_thr,
    )
    result = run_backtest(decision.price_data, strategy_output.signals, decision.backtest_config)
    return decision, strategy_output, result
