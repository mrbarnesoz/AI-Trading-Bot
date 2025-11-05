"""Market regime detection utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _normalise_close(engineered: pd.DataFrame) -> pd.Series:
    for candidate in ("Close", "close", "close_price", "last_price"):
        if candidate in engineered.columns:
            return engineered[candidate].astype(float)
    return pd.Series(0.0, index=engineered.index, dtype=float)


def _get_adx(engineered: pd.DataFrame) -> pd.Series:
    for candidate in ("adx", "ADX", "trend_strength"):
        if candidate in engineered.columns:
            return engineered[candidate].astype(float).fillna(0.0)
    return pd.Series(0.0, index=engineered.index, dtype=float)


def _get_slope(engineered: pd.DataFrame) -> pd.Series:
    if "trend_slope" in engineered.columns:
        return engineered["trend_slope"].astype(float).fillna(0.0)
    close = _normalise_close(engineered)
    window = min(20, max(5, int(len(close) * 0.05)))
    if window < 3:
        window = 3
    slope = close.diff().rolling(window=window, min_periods=1).mean()
    denom = close.rolling(window=window, min_periods=1).mean().replace(0.0, np.nan)
    slope = slope / denom
    return slope.fillna(0.0)


def _get_vol_ratio(engineered: pd.DataFrame) -> pd.Series:
    close = _normalise_close(engineered).replace(0.0, np.nan)
    atr = None
    for candidate in ("atr_14", "ATR", "atr"):
        if candidate in engineered.columns:
            atr = engineered[candidate].astype(float)
            break
    if atr is None:
        high = engineered.get("High", engineered.get("high"))
        low = engineered.get("Low", engineered.get("low"))
        if high is not None and low is not None:
            atr = pd.Series(high, dtype=float).rolling(window=14, min_periods=3).max() - pd.Series(low, dtype=float).rolling(window=14, min_periods=3).min()
        else:
            atr = pd.Series(0.0, index=engineered.index)
    ratio = (atr / close).abs().replace([np.nan, np.inf, -np.inf], 0.0)
    return ratio.fillna(0.0)


def detect_regime(engineered: pd.DataFrame) -> pd.Series:
    """Classify each timestamp into trend/breakout/range regimes."""
    index = engineered.index
    adx = _get_adx(engineered)
    slope = _get_slope(engineered)
    vol_ratio = _get_vol_ratio(engineered)

    trend_threshold = 25.0
    slope_threshold = 0.0008
    breakout_threshold = 0.015

    regimes = pd.Series("range", index=index, dtype=object)

    strong_trend = (adx >= trend_threshold) | (slope.abs() >= slope_threshold)
    regimes.loc[strong_trend] = "trend"

    breakout_candidates = (vol_ratio >= breakout_threshold) & ~strong_trend
    regimes.loc[breakout_candidates] = "breakout"

    return regimes


def summarise_regime(regimes: pd.Series) -> Tuple[int, int, int]:
    counts = regimes.value_counts()
    return (
        int(counts.get("trend", 0)),
        int(counts.get("breakout", 0)),
        int(counts.get("range", 0)),
    )
