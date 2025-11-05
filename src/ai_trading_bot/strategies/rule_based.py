"""Rule-based strategy modules to complement the ML model."""

from __future__ import annotations

from math import atan
from typing import Tuple

import numpy as np
import pandas as pd

from ai_trading_bot.strategies.base import StrategySlice


def _get_price_series(engineered: pd.DataFrame) -> pd.Series:
    for candidate in ("Close", "close", "close_price", "last_price"):
        if candidate in engineered.columns:
            return engineered[candidate].astype(float)
    raise KeyError("Price column not found in engineered dataset.")


def _get_volume_series(engineered: pd.DataFrame) -> pd.Series:
    for candidate in ("Volume", "volume"):
        if candidate in engineered.columns:
            return engineered[candidate].astype(float)
    return pd.Series(1.0, index=engineered.index, dtype=float)


def mean_reversion_signals(engineered: pd.DataFrame) -> StrategySlice:
    """Generate mean reversion signals using VWAP deviation and RSI."""
    index = engineered.index
    vwap_dev = engineered.get("vwap_dev_z")
    if vwap_dev is None:
        vwap_dev = ((engineered.get("Close", engineered.get("close", pd.Series(0.0, index=index))) -
                     engineered.get("ema_20", engineered.get("ema_fast", 0.0))) / (
                        engineered.get("atr_14", 1.0) + 1e-9)).fillna(0.0)
    else:
        vwap_dev = vwap_dev.astype(float).fillna(0.0)

    rsi = None
    for candidate in ("rsi_14", "rsi", "rsi_fast"):
        if candidate in engineered.columns:
            rsi = engineered[candidate].astype(float).fillna(50.0)
            break
    if rsi is None:
        rsi = pd.Series(50.0, index=index)

    threshold = 1.0
    z_score = vwap_dev.clip(-6.0, 6.0)
    long_mask = (z_score <= -threshold) & (rsi <= 45.0)
    short_mask = (z_score >= threshold) & (rsi >= 55.0)

    signals = pd.Series(0.0, index=index, name="mean_reversion_signal")
    signals[long_mask] = 1.0
    signals[short_mask] = -1.0

    intensity = np.clip(np.abs(z_score) / (threshold * 3.0), 0.0, 1.0)
    prob = 0.5 + 0.5 * signals * intensity
    probabilities = pd.Series(prob, index=index, name="mean_reversion_probability").clip(0.0, 1.0)

    diagnostics = {
        "mean_dev": float(vwap_dev.mean()),
        "mean_rsi": float(rsi.mean()),
    }
    slice = StrategySlice(
        name="mean_reversion",
        signals=signals,
        probabilities=probabilities,
        diagnostics=diagnostics,
    )
    slice.ensure_index(index)
    return slice


def momentum_signals(engineered: pd.DataFrame) -> StrategySlice:
    """Momentum strategy using dual moving averages and slope."""
    close = _get_price_series(engineered)
    index = close.index
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=36, adjust=False).mean()
    diff = (fast - slow).fillna(0.0)

    slope = diff.diff().rolling(window=5, min_periods=1).mean().fillna(0.0)
    threshold = close.abs().mul(0.001).replace(0.0, 1e-6)

    signals = pd.Series(0.0, index=index, name="momentum_signal")
    signals[diff > threshold] = 1.0
    signals[diff < -threshold] = -1.0

    magnitude = np.tanh(np.abs(diff) / threshold.clip(lower=1e-6))
    probabilities = 0.5 + 0.5 * signals * magnitude
    probabilities = pd.Series(probabilities, index=index, name="momentum_probability").clip(0.0, 1.0)

    diagnostics = {
        "avg_diff": float(diff.mean()),
        "avg_slope": float(slope.mean()),
    }
    slice = StrategySlice(
        name="momentum",
        signals=signals,
        probabilities=probabilities,
        diagnostics=diagnostics,
    )
    slice.ensure_index(index)
    return slice


def breakout_signals(engineered: pd.DataFrame) -> StrategySlice:
    """Breakout strategy based on recent highs/lows and volatility."""
    high = engineered.get("High", engineered.get("high"))
    low = engineered.get("Low", engineered.get("low"))
    close = _get_price_series(engineered)
    if high is None or low is None:
        window_high = close.rolling(window=20, min_periods=5).max()
        window_low = close.rolling(window=20, min_periods=5).min()
    else:
        window_high = pd.Series(high, dtype=float).rolling(window=20, min_periods=5).max()
        window_low = pd.Series(low, dtype=float).rolling(window=20, min_periods=5).min()
    atr = engineered.get("atr_14")
    if atr is None:
        atr = (window_high - window_low).fillna(method="bfill").fillna(method="ffill").abs() / 2.0
    atr = atr.fillna(0.0)

    buffer = atr * 0.2
    signals = pd.Series(0.0, index=close.index, name="breakout_signal")
    signals[close >= (window_high - buffer)] = 1.0
    signals[close <= (window_low + buffer)] = -1.0

    mask_long = signals == 1.0
    mask_short = signals == -1.0
    distance_series = pd.Series(0.0, index=close.index, dtype=float)
    distance_series[mask_long] = (close - (window_high - buffer))[mask_long]
    distance_series[mask_short] = ((window_low + buffer) - close)[mask_short]
    scaled = np.clip(distance_series / (atr + 1e-6), 0.0, 2.0)
    probabilities = 0.5 + 0.5 * signals * np.clip(scaled / 2.0, 0.0, 1.0)
    probabilities = pd.Series(probabilities, index=close.index, name="breakout_probability").clip(0.0, 1.0)

    diagnostics = {
        "avg_distance": float(distance_series.mean()),
        "avg_atr": float(np.nanmean(atr)),
    }
    slice = StrategySlice(
        name="breakout",
        signals=signals,
        probabilities=probabilities,
        diagnostics=diagnostics,
    )
    slice.ensure_index(close.index)
    return slice
