"""Dataset builder for BitMEX price data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ai_trading_bot.config import AppConfig, DataConfig, FeatureConfig, PipelineConfig
from ai_trading_bot.data.fetch import get_price_data
from ai_trading_bot.features.indicators import engineer_features

logger = logging.getLogger(__name__)

ENGINEERED_DIR = Path("data/engineered")
DECISION_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "target_return", "target"]


def _ensure_app_config(config: Union[AppConfig, Dict]) -> AppConfig:
    if isinstance(config, AppConfig):
        return config
    if isinstance(config, dict):
        return AppConfig.from_dict(config)
    raise TypeError(f"Unsupported configuration type: {type(config)!r}")


def _normalise_timestamp(value: str | None) -> Tuple[Optional[pd.Timestamp], Optional[str]]:
    if value is None:
        return None, None
    ts = pd.to_datetime(value, utc=True)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts, ts.strftime("%Y%m%dT%H%M%SZ")


def _engineered_path(symbol: str, interval: str, start_label: str, end_label: str) -> Tuple[Path, Path]:
    ENGINEERED_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{symbol}_{interval}_{start_label}_{end_label}"
    return ENGINEERED_DIR / f"{stem}.parquet", ENGINEERED_DIR / f"{stem}.csv"


def _compute_warmup(feature_cfg: FeatureConfig, pipeline_cfg: PipelineConfig) -> int:
    windows = [int(pipeline_cfg.lookahead or 1), 5]  # rolling volatility window
    if "sma" in feature_cfg.indicators:
        windows.append(int(feature_cfg.sma_window))
    if "ema" in feature_cfg.indicators:
        windows.append(int(feature_cfg.ema_window))
    if "rsi" in feature_cfg.indicators:
        windows.append(int(feature_cfg.rsi_window))
    if "macd" in feature_cfg.indicators:
        windows.append(int(feature_cfg.macd_slow) + int(feature_cfg.macd_signal))
    windows.extend([14, 20])  # ATR/ADX/vwap windows
    return max(windows)


def prepare_dataset(config: Union[AppConfig, Dict], force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch BitMEX data, engineer features, and return decision & ML datasets."""
    app_config = _ensure_app_config(config)
    data_cfg: DataConfig = app_config.data
    feature_cfg: FeatureConfig = app_config.features
    pipeline_cfg: PipelineConfig = app_config.pipeline

    price_df = get_price_data(data_cfg, force_download=force_download)
    engineered = engineer_features(price_df, feature_cfg, pipeline_cfg)

    close = engineered["Close"]
    lookahead = int(pipeline_cfg.lookahead or 1)
    engineered["target_return"] = close.shift(-lookahead) / close - 1.0
    engineered["target"] = (engineered["target_return"] > 0).astype(np.int8)

    warmup = _compute_warmup(feature_cfg, pipeline_cfg)
    engineered = engineered.iloc[warmup:].dropna()
    engineered = engineered.astype({col: "float32" for col in engineered.select_dtypes(include="float").columns})
    engineered.index = pd.to_datetime(engineered.index, utc=True)
    engineered.index.name = "Date"

    missing = [col for col in DECISION_COLUMNS if col not in engineered.columns]
    if missing:
        raise ValueError(f"Engineered dataset missing required columns: {missing}")

    decision_cols = [col for col in DECISION_COLUMNS if col in engineered.columns]
    optional_cols = [col for col in ("fundingRate", "fundingRateDaily") if col in engineered.columns]
    decision = engineered[decision_cols + optional_cols].copy()

    _, start_label = _normalise_timestamp(data_cfg.start_date)
    _, end_label = _normalise_timestamp(data_cfg.end_date)
    if start_label is None:
        start_label = "start"
    if end_label is None:
        end_label = "now"

    parquet_path, csv_path = _engineered_path(data_cfg.symbol, data_cfg.interval, start_label, end_label)
    engineered.to_parquet(parquet_path)
    engineered.to_csv(csv_path, index=True)
    logger.info(
        "Engineered dataset for %s (%s) persisted to %s and %s",
        data_cfg.symbol,
        data_cfg.interval,
        parquet_path,
        csv_path,
    )

    return decision, engineered


__all__ = ["prepare_dataset"]
