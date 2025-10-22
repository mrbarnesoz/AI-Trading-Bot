"""Feature engineering utilities for technical indicators."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from ai_trading_bot.config import FeatureConfig, PipelineConfig

logger = logging.getLogger(__name__)


def _relative_strength_index(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    loss = down.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return pd.DataFrame({"macd": macd, "macd_signal": signal_line, "macd_hist": histogram})


def engineer_features(
    price_data: pd.DataFrame, feature_cfg: FeatureConfig, pipeline_cfg: PipelineConfig
) -> pd.DataFrame:
    """Generate indicator-based features and the prediction target."""
    if price_data.index.name != "Date":
        price_data = price_data.rename_axis("Date").sort_index()

    df = price_data.copy()
    required_columns = {"Close", "Open", "High", "Low", "Adj Close", "Volume"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input price data is missing required columns: {missing}")

    logger.info("Generating technical indicators with config: %s", feature_cfg)
    df["return_1d"] = df["Adj Close"].pct_change()
    df["volatility_5"] = df["return_1d"].rolling(window=5).std()
    df["rolling_volume"] = df["Volume"].rolling(window=5).mean()

    if "sma" in feature_cfg.indicators:
        df[f"sma_{feature_cfg.sma_window}"] = df["Adj Close"].rolling(window=feature_cfg.sma_window).mean()
    if "ema" in feature_cfg.indicators:
        df[f"ema_{feature_cfg.ema_window}"] = df["Adj Close"].ewm(span=feature_cfg.ema_window, adjust=False).mean()
    if "rsi" in feature_cfg.indicators:
        df[f"rsi_{feature_cfg.rsi_window}"] = _relative_strength_index(df["Adj Close"], feature_cfg.rsi_window)
    if "macd" in feature_cfg.indicators:
        macd_df = _macd(df["Adj Close"], feature_cfg.macd_fast, feature_cfg.macd_slow, feature_cfg.macd_signal)
        df = pd.concat([df, macd_df], axis=1)

    future_return = df["Adj Close"].pct_change(periods=pipeline_cfg.lookahead).shift(-pipeline_cfg.lookahead)
    df[pipeline_cfg.target_column] = future_return
    df["target"] = (future_return > 0).astype(int)
    engineered = df.dropna()
    logger.info("Engineered dataset contains %d samples and %d features", *engineered.shape)
    return engineered
