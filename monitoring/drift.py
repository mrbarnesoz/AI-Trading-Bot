"""Feature drift monitoring utilities."""

from __future__ import annotations

import numpy as np
import polars as pl


def population_stability_index(base: pl.Series, current: pl.Series, bins: int = 20) -> float:
    base_hist, edges = np.histogram(base.to_numpy(), bins=bins, density=True)
    current_hist, _ = np.histogram(current.to_numpy(), bins=edges, density=True)
    base_hist = np.where(base_hist == 0, 1e-6, base_hist)
    current_hist = np.where(current_hist == 0, 1e-6, current_hist)
    psi = np.sum((current_hist - base_hist) * np.log(current_hist / base_hist))
    return float(psi)
