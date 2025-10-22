"""High-level orchestration for the AI trading bot pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ai_trading_bot.backtesting.simulator import BacktestResult, run_backtest
from ai_trading_bot.config import AppConfig, load_config
from ai_trading_bot.data.fetch import download_price_data
from ai_trading_bot.features.indicators import engineer_features
from ai_trading_bot.models.predictor import train_model
from ai_trading_bot.strategies.ml_strategy import StrategyOutput, generate_signals
from ai_trading_bot.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def prepare_dataset(config: AppConfig, force_download: bool = False):
    """Fetch data and engineer features."""
    price_data = download_price_data(config.data, force_refresh=force_download)
    engineered = engineer_features(price_data, config.features, config.pipeline)
    return price_data, engineered


def train(config_path: Path | str = "config.yaml", force_download: bool = False) -> dict:
    """Train the machine learning model."""
    configure_logging()
    config = load_config(config_path)
    _, engineered = prepare_dataset(config, force_download=force_download)
    metrics = train_model(engineered, config.model, config.pipeline)
    logger.info("Training metrics: %s", metrics)
    return metrics


def backtest(
    config_path: Path | str = "config.yaml", force_download: bool = False, probability_threshold: float = 0.55
) -> tuple[StrategyOutput, BacktestResult]:
    """Run the full pipeline through to backtesting."""
    configure_logging()
    config = load_config(config_path)
    price_data, engineered = prepare_dataset(config, force_download=force_download)
    strategy_output = generate_signals(engineered, config.model, config.pipeline, probability_threshold)
    result = run_backtest(price_data, strategy_output.signals, config.backtest)
    return strategy_output, result
