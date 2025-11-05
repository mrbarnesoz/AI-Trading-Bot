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


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    delta_high = high.diff()
    delta_low = -low.diff()

    plus_dm = np.where((delta_high > 0) & (delta_high > delta_low), delta_high, 0.0)
    minus_dm = np.where((delta_low > 0) & (delta_low > delta_high), delta_low, 0.0)

    tr = _true_range(high, low, close)
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / window, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / window, adjust=False).mean() / atr

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return 100 * dx.ewm(alpha=1 / window, adjust=False).mean()


def _rolling_slope(series: pd.Series, window: int = 20) -> pd.Series:
    idx = np.arange(window, dtype=float)
    denom = ((idx - idx.mean()) ** 2).sum()

    def _calc(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y = values
        y_mean = y.mean()
        slope = ((idx - idx.mean()) * (y - y_mean)).sum() / denom
        return slope

    return series.rolling(window=window, min_periods=window).apply(_calc, raw=True)


def engineer_features(
    price_data: pd.DataFrame, feature_cfg: FeatureConfig, pipeline_cfg: PipelineConfig
) -> pd.DataFrame:
    """Generate indicator-based features and the prediction target."""
    if price_data.index.name != "Date":
        price_data = price_data.rename_axis("Date").sort_index()

    df = price_data.copy()
    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input price data is missing required columns: {missing}")

    logger.info("Generating technical indicators with config: %s", feature_cfg)
    df["return_1d"] = df["Close"].pct_change()
    df["volatility_5"] = df["return_1d"].rolling(window=5).std()
    df["rolling_volume"] = df["Volume"].rolling(window=5).mean()

    atr = _compute_atr(df["High"], df["Low"], df["Close"], window=14)
    df["atr_14"] = atr
    df["adx_14"] = _compute_adx(df["High"], df["Low"], df["Close"], window=14)
    df["adx"] = df["adx_14"]

    slope_raw = _rolling_slope(df["Close"], window=20)
    df["trend_slope"] = slope_raw / (atr + 1e-9)

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_volume = df["Volume"].cumsum().replace(0, np.nan)
    vwap = (typical_price * df["Volume"]).cumsum() / cum_volume
    dev = typical_price - vwap
    dev_mean = dev.rolling(window=20, min_periods=20).mean()
    dev_std = dev.rolling(window=20, min_periods=20).std()
    df["vwap_dev_z"] = (dev - dev_mean) / dev_std.replace(0, np.nan)

    spread_proxy = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)
    spread_mean = spread_proxy.rolling(window=20, min_periods=20).mean()
    spread_std = spread_proxy.rolling(window=20, min_periods=20).std()
    df["spread_z"] = (spread_proxy - spread_mean) / spread_std.replace(0, np.nan)

    if "fundingRate" in df.columns:
        df["fundingRate"] = df["fundingRate"].ffill().fillna(0.0)
    if "fundingRateDaily" in df.columns:
        df["fundingRateDaily"] = df["fundingRateDaily"].ffill().fillna(0.0)

    if "sma" in feature_cfg.indicators:
        df[f"sma_{feature_cfg.sma_window}"] = df["Close"].rolling(window=feature_cfg.sma_window).mean()
    if "ema" in feature_cfg.indicators:
        df[f"ema_{feature_cfg.ema_window}"] = df["Close"].ewm(span=feature_cfg.ema_window, adjust=False).mean()
    if "rsi" in feature_cfg.indicators:
        df[f"rsi_{feature_cfg.rsi_window}"] = _relative_strength_index(df["Close"], feature_cfg.rsi_window)
    if "macd" in feature_cfg.indicators:
        macd_df = _macd(df["Close"], feature_cfg.macd_fast, feature_cfg.macd_slow, feature_cfg.macd_signal)
        df = pd.concat([df, macd_df], axis=1)

    future_return = df["Close"].pct_change(periods=pipeline_cfg.lookahead).shift(-pipeline_cfg.lookahead)
    df[pipeline_cfg.target_column] = future_return
    df["target"] = (future_return > 0).astype(int)
    engineered = df.dropna().copy()

    float_cols = engineered.select_dtypes(include=["float64", "float32"]).columns
    engineered = engineered.astype({col: "float32" for col in float_cols})
    logger.info("Engineered dataset contains %d samples and %d features", *engineered.shape)
    return engineered



