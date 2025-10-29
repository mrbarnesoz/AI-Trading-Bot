"""High-level orchestration for the AI trading bot pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from ai_trading_bot.backtesting.simulator import BacktestResult, run_backtest
from ai_trading_bot.config import AppConfig, load_config
from ai_trading_bot.data_pipeline import prepare_dataset as build_dataset
from ai_trading_bot.meta.select import Context, MetaStrategySelector
from ai_trading_bot.models.predictor import train_model
from ai_trading_bot.strategies.ml_strategy import StrategyOutput, generate_signals
from ai_trading_bot.utils.logging import configure_logging

logger = logging.getLogger(__name__)
_META_SELECTOR = MetaStrategySelector()


def prepare_dataset(config: AppConfig, force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch raw data, engineer indicators, and build training/backtest dataset."""
    config_dict = config.to_nested_dict()
    return build_dataset(config_dict, force_download=force_download)


def train(config_path: Path | str = "config.yaml", force_download: bool = False) -> dict:
    """Train the machine learning model."""
    configure_logging()
    config = load_config(config_path)
    price_frame, engineered = prepare_dataset(config, force_download=force_download)
    metrics = train_model(engineered, config.model, config.pipeline)
    metrics.update(
        {
            "symbol": config.data.symbol,
            "interval": config.data.interval,
            "rows": len(engineered),
        }
    )
    logger.info("Training metrics: %s", metrics)
    return metrics


def backtest(
    config_path: Path | str = "config.yaml",
    force_download: bool = False,
    long_threshold: Optional[float] = None,
    short_threshold: Optional[float] = None,
) -> tuple[StrategyOutput, BacktestResult, dict]:
    """Run the full pipeline through to backtesting."""
    configure_logging()
    config = load_config(config_path)
    price_frame, engineered = prepare_dataset(config, force_download=force_download)
    long_thr = long_threshold if long_threshold is not None else config.pipeline.long_threshold
    short_thr = short_threshold if short_threshold is not None else config.pipeline.short_threshold

    strategy_output = generate_signals(
        engineered,
        config.model,
        config.pipeline,
        probability_long_threshold=long_thr,
        probability_short_threshold=short_thr,
    )

    strategy_output = _apply_meta_selector(strategy_output, engineered, config)

    result = run_backtest(price_frame, strategy_output.signals, config.backtest)
    metadata = {
        "symbol": config.data.symbol,
        "interval": config.data.interval,
        "rows": int(len(engineered)),
    }
    logger.info("Backtest summary: %s", result.summary)
    return strategy_output, result, metadata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_meta_selector(
    strategy_output: StrategyOutput,
    engineered: pd.DataFrame,
    config: AppConfig,
) -> StrategyOutput:
    selector = _META_SELECTOR
    regime = selector._normalise_regime(config.data.interval)  # type: ignore[attr-defined]
    base_fraction = config.backtest.position_capital_fraction
    max_units = config.backtest.max_position_units

    sized = []
    records = []
    index = strategy_output.signals.index

    for timestamp, prob in strategy_output.probabilities.reindex(index, fill_value=0.5).items():
        row = engineered.loc[timestamp] if timestamp in engineered.index else pd.Series()
        context = selector.create_context_from_row(
            timestamp=pd.Timestamp(timestamp).tz_localize("UTC") if not isinstance(timestamp, pd.Timestamp) else timestamp,
            symbol=config.data.symbol,
            regime=regime,
            row=row,
            probability=float(prob),
        )
        decision = selector.select_strategy(context)
        records.append(decision.to_record())
        if decision.direction == "long":
            units = min(decision.size_frac / base_fraction, max_units)
            sized.append(float(units))
        elif decision.direction == "short":
            units = min(decision.size_frac / base_fraction, max_units)
            sized.append(float(-units))
        else:
            sized.append(0.0)

    selector.flush()
    strategy_output.signals = pd.Series(sized, index=index, dtype=float)
    strategy_output.decisions = pd.DataFrame(records)
    return strategy_output
