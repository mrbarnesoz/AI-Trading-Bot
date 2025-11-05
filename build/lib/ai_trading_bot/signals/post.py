"""Signal post-processing helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

_EPS = 1e-12


def _normalise_series(values: pd.Series | Iterable[float]) -> pd.Series:
    if isinstance(values, pd.Series):
        return values
    return pd.Series(list(values))


def apply_hysteresis(
    signals: pd.Series,
    prices: pd.Series,
    atr_values: pd.Series,
    *,
    k: float = 0.15,
) -> pd.Series:
    """Apply ATR-based hysteresis to reduce rapid signal flipping.

    Parameters
    ----------
    signals:
        Series of raw signals (positive for long, negative for short, zero for flat).
    prices:
        Matching series of asset prices (typically close prices).
    atr_values:
        Average True Range values aligned with the signal index.
    k:
        Multiple of ATR that incoming price must travel in the opposite direction
        before a flip is accepted.
    """

    if k <= 0:
        return signals

    sig = _normalise_series(signals).astype(float)
    px = _normalise_series(prices).astype(float).reindex(sig.index).ffill()
    atr = _normalise_series(atr_values).astype(float).reindex(sig.index).ffill()
    atr = atr.fillna(0.0)

    last_dir = 0.0
    last_price = np.nan
    output: list[float] = []

    for value, price, atr_val in zip(sig.to_numpy(), px.to_numpy(), atr.to_numpy()):
        direction = 0.0
        if value > _EPS:
            direction = 1.0
        elif value < -_EPS:
            direction = -1.0

        if direction == 0.0:
            output.append(0.0)
            continue

        if last_dir == 0.0:
            last_dir = direction
            last_price = price
            output.append(direction)
            continue

        if direction == last_dir:
            last_price = price
            output.append(direction)
            continue

        atr_threshold = max(atr_val, 0.0) * k
        accept_flip = False
        if last_dir > 0 and price <= last_price - atr_threshold:
            accept_flip = True
        elif last_dir < 0 and price >= last_price + atr_threshold:
            accept_flip = True

        if accept_flip:
            last_dir = direction
            last_price = price
        output.append(last_dir)

    return pd.Series(output, index=sig.index, dtype=float)

